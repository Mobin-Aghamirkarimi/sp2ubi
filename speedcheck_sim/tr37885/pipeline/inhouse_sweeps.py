"""In-house sweeps of study 1: the in-house channel only, no generator data, so
they may run before or in parallel with P2; they feed Fig. C (K4) and Fig. E and never change a rule.
  k4    per-beacon FDOA error against the raw-sample SNR (-5 ... 45 dB in 5-dB steps) at K in {0, 3.5, 6, 9, 15} dB and K = inf
        (noise only), 15 m/s, isotropic scattering, no facade, nominal impairments off (the per-beacon estimator alone):
        RMS, median |e| and q90 |e| per (K, SNR); the pass sigma (select statistic, SD over the passes, calibrated by the
        mean) per (K, SNR); the single-tone CRLB of one beacon (two heads: sqrt(2) x the single-tone bound at the
        post-decimation SNR, N = M samples over the packet) and the pass-level Fisher bound with the measured per-beacon
        RMS as the per-packet noise (nuisance-free; D = L);
  fige  per-crossing sigma (system as specified: qualification sd <= 6 %, residual gate 1.3, +-25 % clip) against the
        site K in {3, 4.5, 6, 8, 10, 13, 16, 20} dB, with and without the facade (specular -10 dB), 15 m/s:
        n sites x (30 reference + 60 test) passes per point; the qualified share and sigma with a site-bootstrap CI.
Resumable (one JSON per point under runs/inhouse_<tag>/), provenance on every file; results/inhouse_<kind>_<tag>.json.
Usage: python pipeline/inhouse_sweeps.py k4|fige --tag sens_inhouse [--passes N] [--sites N]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
import interface as itf  # noqa: E402
from fdoa.channel import ChannelConfig  # noqa: E402
from fdoa.estimator import fdoa_from_samples  # noqa: E402
from fdoa.geometry import LAMBDA, Site, make_trajectory  # noqa: E402
from pipeline.stage2_evaluate import one_stat  # noqa: E402
from pipeline.stage1_channels import C  # noqa: E402
from pipeline.provenance import check_tag, provenance, same_provenance  # noqa: E402
from pipeline import stage3_criteria as K  # noqa: E402

SEED = 2026102001                                         # own seed base
V0 = 15.0
K4_K = [0.0, 3.5, 6.0, 9.0, 15.0, None]                    # None = K infinite (no diffuse power)
K4_SNR = list(range(-5, 50, 5))
FE_K = [3.0, 4.5, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0]
FS_RAW, PACKET_S = 10e6, 1e-3


def _geo(rng):
    G = C["geometry"]
    return Site(D=G["L_m"], pole_offset=float(rng.uniform(*G["pole_offset_m_uniform"])),
                dh=G["head_height_m"] - G["vehicle_antenna_m"]["type2_car_roof"])


def one_pass(key, geo, K_db, snr_db=None, specular_db=None, impairments=False):
    """One in-house pass: per-beacon FDOA error (Hz), SNR used, and the stage-2 statistics. snr_db None: the in-house site-accuracy mapping."""
    rng = np.random.default_rng([SEED] + key + [0])
    traj = make_trajectory(rng, geo, V0, x_start=-0.35 * geo.D, x_end=1.35 * geo.D)
    kd = 300.0 if K_db is None else K_db
    cfg = ChannelConfig(K_db=kd, K_db_head=(kd, kd), scattering="iso", specular_db=specular_db)
    t_b, y, _ = itf.inhouse_clean(np.random.default_rng([SEED] + key + [1]), geo, traj, cfg)
    x, yy, vx, _ = traj.at(t_b)
    heads = geo.heads()
    pos = np.stack([x, yy, np.zeros_like(x)], -1)
    d_h = heads[:, None, :] - pos[None]
    r = np.linalg.norm(d_h, axis=-1)
    f_los = (d_h[..., 0] * vx[None, :]) / r / LAMBDA
    snr = itf.snr_e1_db(r) if snr_db is None else np.full_like(r, float(snr_db))
    yb = itf.finish(np.random.default_rng([SEED] + key + [2]), y, snr, gain=None, t_b=t_b)
    if impairments:                                     # K = inf with the nominal impairments
        import copy
        u_x = (heads[:, 0][:, None] - x[None, :]) / r
        yb = itf.apply_impairments(np.random.default_rng([SEED] + key + [3]), yb, t_b, u_x, copy.deepcopy(C["impairments"]))
    fd = fdoa_from_samples(yb, PACKET_S)
    st = one_stat(t_b, fd, vx, x, np.hypot(geo.pole_offset - yy, geo.dh), geo)
    return dict(err=(fd - (f_los[0] - f_los[1])).tolist(), match=float(st["match"]), prof=float(st["prof"]),
                mse=float(st["mse"]), finite=bool(np.isfinite(st["match"]) and np.isfinite(st["prof"])))


def crlb_beacon_hz(snr_raw_db, M=None):
    """sqrt(2) x single-tone CRLB of one beacon: N = M samples over the packet, post-decimation SNR = raw SNR x fs T / M."""
    M = M or C["pipeline_constants_from_code"]["channel_defaults"]["M"]
    Ts = PACKET_S / M
    snr = 10 ** (snr_raw_db / 10) * FS_RAW * PACKET_S / M
    return float(np.sqrt(2) * np.sqrt(6 / ((2 * np.pi) ** 2 * snr * M * (M ** 2 - 1) * Ts ** 2)))


def fisher_pass_pct(rms_hz, v=V0):
    """Pass-level Fisher bound (%), tied honest model without nuisance parameters (keep [x_m, v, acc]), per-packet FDOA noise = the measured per-beacon RMS. Geometry from the constants:
    dh = head height - car-roof antenna (3.4 m), lateral distance = the mean pole offset plus the test-lane offset
    (1.75 m), the window arc_L, the beacon period."""
    G = C["geometry"]
    D = G["L_m"]
    H = G["head_height_m"] - G["vehicle_antenna_m"]["type2_car_roof"]
    lat = float(np.mean(G["pole_offset_m_uniform"])) + 1.75
    a, b = C["window_rule"]["arc_L"]
    dt = C["pipeline_constants_from_code"]["BEACON_PERIOD_MS"] / 1000
    g = lambda x: (-x) / np.sqrt(x ** 2 + lat ** 2 + H ** 2) - (D - x) / np.sqrt((D - x) ** 2 + lat ** 2 + H ** 2)
    t = np.arange(0, (b - a) * D / v, dt); tc = t - t.mean()
    m = lambda q: (q[1] + q[2] * tc) / LAMBDA * g(q[0] + q[1] * tc + 0.5 * q[2] * tc ** 2)
    p = np.array([0.5 * D, v, 0.0]); e = 1e-5
    J = np.column_stack([(m(p + e * np.eye(3)[k]) - m(p - e * np.eye(3)[k])) / (2 * e) for k in range(3)])
    cov = np.linalg.inv(J.T @ J) * rms_hz ** 2
    return float(np.sqrt(cov[1, 1]) / v * 100)


def point_file(out_dir, name):
    return os.path.join(out_dir, name + ".json")


def run_point(path, prov, fn):
    if os.path.exists(path):
        d = json.load(open(path))
        if not same_provenance(d.get("provenance", {}), prov):
            raise RuntimeError(f"{path} exists with a different provenance: not reused")
        return d
    d = dict(fn(), provenance=prov)
    json.dump(d, open(path + ".tmp", "w"), default=float)
    os.replace(path + ".tmp", path)
    return d


def k4(a, out_dir):
    rows = []
    for ki, Kd in enumerate(K4_K):
        for si, snr in enumerate(K4_SNR):
            prov = provenance(a.tag, stage=3, kind="k4", K=Kd, snr=snr, passes=a.passes)

            def fn(Kd=Kd, snr=snr, ki=ki, si=si):
                ps = [one_pass([4, ki, si, k], _geo(np.random.default_rng([SEED, 4, k])), Kd, snr_db=snr)
                      for k in range(a.passes)]
                e = np.concatenate([np.asarray(p["err"]) for p in ps])
                e = e[np.isfinite(e)]
                m = np.array([p["match"] for p in ps if p["finite"]])
                rms = float(np.sqrt(np.mean(e ** 2)))
                return dict(K_db=Kd, snr_db=snr, n_beacons=int(e.size), rms_hz=rms, median_abs_hz=float(np.median(np.abs(e))),
                            q90_abs_hz=float(np.quantile(np.abs(e), 0.9)),
                            pass_sigma_pct=float(100 * np.std(m - m.mean(), ddof=1)) if m.size > 2 else np.nan,
                            n_passes_finite=int(m.size), crlb_hz=crlb_beacon_hz(snr), fisher_pass_pct=fisher_pass_pct(rms))
            rows.append(run_point(point_file(out_dir, f"k4_K{ki}_snr{si}"), prov, fn))
    for si, snr in enumerate(K4_SNR):                   # K = inf with nominal impairments: one extra curve
        prov = provenance(a.tag, stage=3, kind="k4imp", K=None, snr=snr, passes=a.passes)

        def fn(snr=snr, si=si):
            ps = [one_pass([6, si, k], _geo(np.random.default_rng([SEED, 6, k])), None, snr_db=snr, impairments=True)
                  for k in range(a.passes)]
            e = np.concatenate([np.asarray(p["err"]) for p in ps])
            e = e[np.isfinite(e)]
            return dict(K_db=None, snr_db=snr, impairments="nominal", n_beacons=int(e.size), rms_hz=float(np.sqrt(np.mean(e ** 2))),
                        median_abs_hz=float(np.median(np.abs(e))), q90_abs_hz=float(np.quantile(np.abs(e), 0.9)),
                        crlb_hz=crlb_beacon_hz(snr))
        rows.append(run_point(point_file(out_dir, f"k4imp_snr{si}"), prov, fn))
    return dict(kind="k4", v0=V0, K_db=K4_K, snr_db=K4_SNR, rows=[{k: v for k, v in r.items() if k != "provenance"} for r in rows])


def fige(a, out_dir):
    rows = []
    for ki, Kd in enumerate(FE_K):
        for fi, sp in enumerate((None, -10.0)):
            prov = provenance(a.tag, stage=3, kind="fige", K=Kd, specular_db=sp, sites=a.sites)

            def fn(Kd=Kd, sp=sp, ki=ki, fi=fi):
                recs = []
                for s in range(a.sites):
                    geo = _geo(np.random.default_rng([SEED, 5, ki, fi, s]))
                    for k in range(90):
                        p = one_pass([5, ki, fi, s, k], geo, Kd, specular_db=sp)
                        recs.append(dict(p, site=s, track=k, j=k, role="ref" if k < 30 else "test", cell="PFE", v0=V0,
                                         arm="inhouse", mapping="e1", prb=10, kin="sumo", beta=0.0, window_ok=True))
                for r in recs:
                    r.pop("err", None)
                T, _ = K.site_table(recs, "PFE", V0, "inhouse", "e1", 10)
                ss = sorted(T)
                e = K.pooled(T, ss, True)
                D_ = K.draws(("FIGE", ki, fi), len(ss))
                sb = [K.sigma(K.pooled(T, [ss[i] for i in idx], True)) for idx in D_]
                kq = int(sum(T[s]["sd_ref"] <= K.SIG_Q for s in ss))
                return dict(K_db=Kd, specular_db=sp, n_sites=len(ss), qualified=kq, qualified_share=kq / max(len(ss), 1),
                            sigma=K.sigma(e), sigma_ci=K.ci(sb), n_kept=int(e.size),
                            site_points=[dict(sd_ref=T[s]["sd_ref"], sigma_site=K.sigma(T[s]["e"][T[s]["keep_gate"]])) for s in ss])
            rows.append(run_point(point_file(out_dir, f"fige_K{ki}_f{fi}"), prov, fn))
    return dict(kind="fige", v0=V0, K_db=FE_K, rows=[{k: v for k, v in r.items() if k != "provenance"} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["k4", "fige"])
    ap.add_argument("--tag", required=True, help="sens_inhouse (pre-specified) or dev_<x>")
    ap.add_argument("--passes", type=int, default=20, help="k4: passes per (K, SNR)")
    ap.add_argument("--sites", type=int, default=10, help="fige: sites per point")
    a = ap.parse_args()
    check_tag(a.tag, stage=3)
    if not a.tag.startswith("dev_") and (a.passes != 20 or a.sites != 10):
        raise ValueError("pre-specified sweeps: --passes 20 --sites 10")
    out_dir = os.path.join(ER, "runs", f"inhouse_{a.tag}")
    os.makedirs(out_dir, exist_ok=True)
    K.B = 2000 if not a.tag.startswith("dev_") else 200
    res = (k4 if a.kind == "k4" else fige)(a, out_dir)
    res["provenance"] = provenance(a.tag, stage=3, kind=a.kind, passes=a.passes, sites=a.sites)
    os.makedirs(K.RESULTS, exist_ok=True)
    path = os.path.join(K.RESULTS, f"inhouse_{a.kind}_{a.tag}.json")
    json.dump(res, open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)
    print(path)


if __name__ == "__main__":
    main()
