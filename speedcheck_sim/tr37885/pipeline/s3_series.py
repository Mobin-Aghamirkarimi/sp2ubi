"""Study 3: per-beacon series of every crossing of a stage-1 site record, produced as stage 2 produces them
(stage2_evaluate.evaluate_site: the same interpolation, SNR mapping, head offset, noise and impairment random numbers),
for every mapping x PRB variant and one arm: gen_on (TR 37.885 with the scatterer-motion term) or inhouse (our channel
model at the realised in-band K, paired with the same crossing). The statistics of the base estimator are stored beside
the series; they equal those of evaluate_site record by record (--check-against).

Output: <out>/<label>_site_<k>.npz, label = the family (gen_on) or the family + "in" (inhouse), with, per record r
(crossing x mapping x PRB): r<r>_t (air times), r<r>_fd (FDOA from the periodogram peaks, Hz), r<r>_fpp (pulse-pair
FDOA of the cross-head product, lag 1), r<r>_pa / r<r>_pb (received power per beacon and head, dB), r<r>_y (lateral
position) and r<r>_v (speed), both from the simulation and used only for the reported speed and for reporting;
and <out>/<label>_site_<k>.json with the record list (cell, v0, role, kin, j, jr, mapping, prb, the base estimator's
prof / match / mse, window_ok, stop_between).

Usage: python pipeline/s3_series.py --family U3 --run-dir runs/g1_U3_dev_s3 --sites 30 40 --out runs/s3_series
           [--arm inhouse] [--check-against runs/g1_U3_dev_s3/eval_dev_s3]
Families U (study 1), U2 (study 2) and H (study 1) are read the same way from their stage-1 records."""
import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
import interface as itf  # noqa: E402
from fdoa.channel import ChannelConfig  # noqa: E402
from fdoa.estimator import fdoa_from_samples  # noqa: E402
from fdoa.geometry import FC, Site, Trajectory  # noqa: E402
from fdoa.urban import pulse_pair  # noqa: E402
from pipeline import s3_family as FAM  # noqa: E402
from pipeline import stage1_channels as G  # noqa: E402
from pipeline import stage2_evaluate as EV  # noqa: E402

CHECKED = ("prof", "match", "mse", "window_ok", "stop_between")


def use_family(fam):
    """Register U3 (and its seeds) before one of its sites is read; the other families need nothing."""
    if fam == FAM.FAMILY:
        EV.seed_base = FAM.apply().seed_base


def series_site(fam, site, run_dir, out_dir, arm="gen_on"):
    use_family(fam)
    label = fam if arm == "gen_on" else fam + "in"
    out_npz = os.path.join(out_dir, f"{label}_site_{site:03d}.npz")
    if os.path.exists(out_npz):
        return out_npz
    S1 = json.load(open(os.path.join(run_dir, f"site_{site:03d}.json")))
    Z = np.load(os.path.join(run_dir, f"site_{site:03d}.npz"))
    F = G.FAMILIES[fam]
    prb_all = list(Z["prbs"])
    survey = Site(D=G.C["geometry"]["L_m"], pole_offset=float(Z["pole_offset"]), dh=EV.HEAD_Z - EV.SURVEY_Z)
    imp = copy.deepcopy(G.C["impairments"])
    IMP = G.seed_base(fam)["impairments"]
    off = np.random.default_rng([IMP, F["idx"], site]).uniform(-1, 1) * EV.HEAD_PPB * 1e-9 * FC
    arrays, recs = {}, []
    for m in S1["meta"]:
        k = m["k"]
        ks = m.get("pair_k", k)
        pre = f"t{k}_"
        t_b = Z[pre + "t_b"]
        tr = Z[pre + "traj"]
        traj = Trajectory(tr[0], tr[1], tr[2], tr[3], tr[4])
        x, y, vx, _ = traj.at(t_b)
        true_site = Site(D=survey.D, pole_offset=survey.pole_offset, dh=EV.HEAD_Z - m["z"])
        heads = true_site.heads()
        r = np.linalg.norm(heads[:, None, :] - np.stack([x, y, np.zeros_like(x)], -1)[None], axis=-1)
        u_x = (heads[:, 0][:, None] - x[None, :]) / r
        rho_true = np.hypot(survey.pole_offset - y, true_site.dh)
        gain = Z[pre + "gain"][[0, 1]]
        inw = (x >= EV.WIN_ARC[0] * survey.D) & (x <= EV.WIN_ARC[1] * survey.D)
        t_window = float(t_b[inw][-1] - t_b[inw][0]) if inw.sum() >= 2 else float("inf")
        for prb in prb_all:
            i = prb_all.index(prb)
            if arm == "gen_on":
                s, gain_fin = np.stack([itf.interp_q(Z[pre + "hq"][i, 1, h]) for h in range(2)]), gain
            else:
                Kb = Z[pre + "K_band_db"]
                cfg = ChannelConfig(K_db=float(Kb[0, i]), K_db_head=(float(Kb[0, i]), float(Kb[1, i])), scattering="iso", specular_db=None)
                t_in, s, _ = itf.inhouse_clean(G.track_rngs(fam, site, ks)[1], true_site, traj, cfg, doppler_scale=1.0)
                assert np.allclose(t_in, t_b), "beacon schedules of the paired arms differ"
                gain_fin = None
            for mp in ("e1", "lb"):
                snr = itf.snr_e1_db(r) if mp == "e1" else itf.snr_link_budget_db(gain)
                rf = np.random.default_rng([IMP, F["idx"], site, ks, prb, 1 if mp == "e1" else 2])
                yb = itf.finish(rf, s, snr, gain=gain_fin, head_offset_hz=(off / 2, -off / 2), t_b=t_b)
                ri = np.random.default_rng([IMP, F["idx"], site, ks, prb, 3])
                yb = itf.apply_impairments(ri, yb, t_b, u_x, imp)
                fd = fdoa_from_samples(yb, 1e-3)
                st = EV.E1.statistics(t_b, fd, vx, x, rho_true, survey)
                n = len(recs)
                arrays[f"r{n}_t"] = t_b
                arrays[f"r{n}_fd"] = fd.astype(np.float32)
                arrays[f"r{n}_fpp"] = pulse_pair(yb[0] * np.conj(yb[1]), 1)[0].astype(np.float32)
                arrays[f"r{n}_pa"] = (10 * np.log10(np.mean(np.abs(yb[0]) ** 2, axis=-1))).astype(np.float32)
                arrays[f"r{n}_pb"] = (10 * np.log10(np.mean(np.abs(yb[1]) ** 2, axis=-1))).astype(np.float32)
                arrays[f"r{n}_y"] = y.astype(np.float32)
                arrays[f"r{n}_v"] = vx.astype(np.float32)
                recs.append(dict(n=n, site=site, track=k, cell=m["cell"], v0=m["v0"], z=m["z"], kin=m["kin"], role=m["role"],
                                 j=m["j"], jr=m.get("jr"), pair_k=m.get("pair_k"), mapping=mp, prb=int(prb),
                                 prof=st[0], match=st[1], mse=st[4],
                                 t_window=t_window, window_ok=bool(t_window <= EV.BURST_S),
                                 stop_between=bool(np.any(vx[(x >= 0) & (x <= survey.D)] < 0.5))))
    np.savez_compressed(out_npz + ".tmp.npz", **arrays)
    os.replace(out_npz + ".tmp.npz", out_npz)
    json.dump(dict(family=label, site=site, pole_offset=float(Z["pole_offset"]), D=survey.D, dh=survey.dh,
                   stage1_npz_sha256=S1.get("npz_sha256"), recs=recs), open(out_npz[:-4] + ".json", "w"))
    return out_npz


def check_against(out_dir, label, site, eval_dir, arm):
    """Number of fields in which the base-estimator statistics stored here differ from the stage-2 records of
    evaluate_site (same arm, no Doppler scaling); 0 is required."""
    ours = json.load(open(os.path.join(out_dir, f"{label}_site_{site:03d}.json")))["recs"]
    ref = {(r["track"], r["mapping"], r["prb"]): r for r in json.load(open(os.path.join(eval_dir, f"site_{site:03d}.json")))["recs"]
           if r["arm"] == arm and r["beta"] == 0.0}
    bad = int(len(ours) != len(ref))
    for r in ours:
        q = ref.get((r["track"], r["mapping"], r["prb"]))
        for f in CHECKED:
            bad += q is None or not (q[f] == r[f] or (isinstance(q[f], float) and np.isnan(q[f]) and np.isnan(r[f])))
    return bad


def _job(a):
    return series_site(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sites", type=int, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arm", default="gen_on", choices=["gen_on", "inhouse"])
    ap.add_argument("--check-against", default=None, help="folder of the stage-2 records of the same sites (eval_<tag>)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sites = [s for s in range(*a.sites) if os.path.exists(os.path.join(a.run_dir, f"site_{s:03d}.npz"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(_job, [(a.family, s, a.run_dir, a.out, a.arm) for s in sites]):
            print(p, flush=True)
    if a.check_against:
        label = a.family if a.arm == "gen_on" else a.family + "in"
        bad = sum(check_against(a.out, label, s, a.check_against, a.arm) for s in sites)
        print(f"base-estimator statistics against {a.check_against}: {bad} differing fields in {len(sites)} sites")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
