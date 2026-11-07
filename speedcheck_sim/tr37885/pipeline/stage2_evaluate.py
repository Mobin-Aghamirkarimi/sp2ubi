"""Stage 2 (numpy only, no new channel draws): turns the stage-1 channel records of one site into per-pass statistics
of the in-house estimator for every arm, with common random numbers across arms:
  arms     gen_off, gen_on (term on: headline; C2-C7), inhouse (paired in-house:
           unchanged fdoa.channel.simulate_pass with the realised K per head, the same trajectory, beacon schedule,
           geometry, SNR array, offsets, noise and impairments)
  mapping  e1 (40 dB at 100 m, free-space slope) and lb (TR 37.885 link budget from the generator's gain; the paired
           in-house arm gets the same per-beacon SNR array)
  prb      10, 20, 50 (worst governs, constants `beacon.prb`)
Nominal impairments (constants `impairments`), no erasures (sensitivities: --erasure, --impairments).
Every record carries both the plain LS statistics (prof, match, mse) and the Huber statistics (d) (prof_d, match_d;
huber_fit.py), so the adoption decision and the ablation use the same records; mse stays the LS residual (gate,
monitor). β arms (k3 C4-light; the pilot on all its U/H cells): every path's
Doppler scaled by 1 + β on the same channel draws, arms gen_on and in-house, with the consistency statistic d (LS free
fit) and d_d (Huber free fit) on the β passes and on their honest (β = 0) counterparts.
P5 (families 5U, 5H): arms gen_on and inhouse; every record also carries the consistency statistic d (free fit);
the first 30 evaluated passes are also evaluated under beta = -5 % and -10 % (every path's Doppler scaled, the same
channel draws, offsets, noise and impairments: interface.interp_q(time_scale) / simulate_pass(doppler_scale)).
P3 (family N): head A switches per beacon between the LOS and NLOSv transmitters with the blockage loss
(nlosv_blockage.py.assemble_head_a); under the e1 mapping the loss is subtracted from the mapped SNR.
Output: runs/g1_<fam>/eval_<tag>/site_<k>.json with one record per (track, arm, mapping, prb), in the record format of
the in-house site-accuracy simulation plus the fields t_window / window_ok (window rule, RSU-observable: a pass
whose window [-0.05 L, 1.05 L] takes more than 40 s yields no entry), stop_between (reporting only), finite, pair_k / jr,
and for the first 10 test passes per cell the per-beacon FDOA / FOA errors and SNR (constants stage2_per_beacon).
C5 quadratic tracks reuse the seeds of their SUMO pair (erasures, schedule, noise, impairments). Provenance is written
into every output and checked on resume; the stage-1 SHA-256 is verified before use.
Usage: python pipeline/stage2_evaluate.py --family U [--sites 0 20] [--tag nominal] [--workers 4]"""
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
import nlosv_blockage as pb  # noqa: E402
import huber_fit as rbf  # noqa: E402
import site_accuracy as E1  # noqa: E402
from fdoa.channel import ChannelConfig  # noqa: E402
from fdoa.estimator import fdoa_from_samples, fit_pass_free, foa  # noqa: E402
from fdoa.geometry import LAMBDA  # noqa: E402
from pipeline.provenance import check_tag, diff_provenance, provenance, same_provenance, same_stage_sources, sha256_file  # noqa: E402
from fdoa.geometry import FC, Site, Trajectory  # noqa: E402
from pipeline.stage1_channels import C, FAMILIES, is_nlosv, seed_base, track_rngs  # noqa: E402

E1.OFFSET_FIT = True                                   # the site-accuracy simulation statistics (offset fit), as the paper's figures
HEAD_Z = C["geometry"]["head_height_m"]
SURVEY_Z = C["geometry"]["vehicle_antenna_m"]["type2_car_roof"]
HEAD_PPB = 5.0                                         # per-site head-to-head reference offset, uniform +-5 ppb
P5_BETAS = tuple(b for b in C["grid"]["P5"]["beta"] if b != 0.0)   # -0.05, -0.10
P5_N_BETA_FIRST = C["conventions"]["p5_n_beta_first"]  # beta arms on the first 30 evaluated passes
BURST_S = C["window_rule"]["burst_s"]                   # window rule: t_window > 40 s -> no entry
WIN_ARC = C["window_rule"]["arc_L"]
PER_BEACON_N = 10                                      # constants stage2_per_beacon: first 10 test passes per cell
EV4 = C["evaluation_v4"]


def beta_plan(fam, cell, v0, role, jr):
    """The β values of a track: P0 families on every test pass of their U/H cells
    (evaluation_v4.p0.beta_cells); primary on the first 30 test passes of the k3 cells (P1-15, P2-30)."""
    key = f"{cell}-{v0:g}"
    if role != "test":
        return []
    if FAMILIES[fam].get("seedset") == "p0":
        return list(EV4["p0"]["betas"]) if key in EV4["p0"]["beta_cells"] else []
    k3 = EV4["k3_c4_light"]
    return list(k3["betas"]) if (key in k3["cells"] and jr < k3["n_test_first"]) else []


def erase(rng, n, kind):
    """Erasure mask (True = beacon kept): None, ('iid', p) or ('ge', p, mean_burst) Gilbert-Elliott with the stationary
    loss probability p and mean burst length mean_burst beacons (good -> bad with p/(mean_burst (1-p)),
    bad -> good with 1/mean_burst)."""
    if kind is None:
        return np.ones(n, bool)
    if kind[0] == "iid":
        return rng.random(n) >= kind[1]
    p, mb = kind[1], kind[2]
    q_bg, q_gb = 1.0 / mb, p / (mb * (1 - p))
    keep = np.ones(n, bool)
    bad = rng.random() < p
    for i in range(n):
        keep[i] = not bad
        bad = (rng.random() >= q_bg) if bad else (rng.random() < q_gb)
    return keep


def one_stat(t_b, fd, vx, x, rho_true, site):
    full = E1.statistics(t_b, fd, vx, x, rho_true, site)
    p, tm = full[7], full[8]
    tc = tm + (site.D / 2 - p[0]) / max(p[1], 0.5)
    m = np.abs(t_b - tc) <= E1.HALF_WINDOW_S
    win = E1.statistics(t_b[m], fd[m], vx[m], x[m], rho_true[m], site) if m.sum() >= 12 else (np.nan,) * 6 + (False, None, None)
    try:                                                # Huber fit (d), always computed
        rd = rbf.statistics_huber(t_b, fd, vx, site)
    except Exception:                                   # noqa: BLE001  (a failed Huber fit is a non-finite (d) record)
        rd = (np.nan, np.nan, np.nan, 0, False, np.nan)
    return dict(prof=full[0], match=full[1], match_true=full[2], v_fit=full[3], mse=full[4], n=full[5], ok=full[6],
                w_prof=win[0], w_match=win[1], w_match_true=win[2], w_n=win[5],
                prof_d=rd[0], match_d=rd[1], v_fit_d=rd[2], irls_iter=rd[3], ok_d=rd[4], downweighted_share=rd[5])


def evaluate_site(fam, site, run_dir, tag, imp_level="nominal", erasure=None, arms=("gen_off", "gen_on", "inhouse"),
                  mappings=("e1", "lb"), prbs=None, lb_gain_shift_db=0.0, stage1_tag=None, decode=False):
    """decode (the SNR-offset tags only): a beacon enters the fit only if its
    raw-sample SNR at both heads is >= the decoding threshold of its PRB allocation (constants evaluation_v5.decoding)."""
    out_dir = os.path.join(run_dir, f"eval_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"site_{site:03d}.json")
    prov = provenance(tag, stage=2, impairments=imp_level, erasure=erasure, arms=list(arms), mappings=list(mappings), prbs=prbs,
                      lb_gain_shift_db=lb_gain_shift_db, decode=bool(decode))
    if os.path.exists(path):
        old = json.load(open(path)).get("provenance", {})
        if not same_provenance(old, prov):
            raise RuntimeError(f"{path} exists with a different provenance: not reused; {diff_provenance(old, prov)}")
        return path
    npz = os.path.join(run_dir, f"site_{site:03d}.npz")
    S1 = json.load(open(os.path.join(run_dir, f"site_{site:03d}.json")))
    if "npz_sha256" in S1 and sha256_file(npz) != S1["npz_sha256"]:
        raise RuntimeError(f"{npz}: SHA-256 differs from its stage-1 record")
    s1tag = stage1_tag or tag                          # P5 qualification evaluates under qual_<tag>_...
    if not s1tag.startswith("dev_"):                   # primary / fixed sensitivity: only fixed stage-1 records
        p1 = S1.get("provenance", {})                  # (dev_fullsize runs are never evaluated)
        if p1.get("tag") not in ("primary", s1tag) or not same_stage_sources(p1, prov, 1):
            raise RuntimeError(f"{npz}: stage-1 tag {p1.get('tag')!r} is not a primary record with this code's stage-1 "
                               "sources")
    Z = np.load(npz)
    meta = S1["meta"]
    F = FAMILIES[fam]
    prb_all = list(Z["prbs"])
    prbs = prb_all if prbs is None else prbs
    survey = Site(D=C["geometry"]["L_m"], pole_offset=float(Z["pole_offset"]), dh=HEAD_Z - SURVEY_Z)
    imp = copy.deepcopy(C["impairments"])
    if imp_level == "none":
        imp = None
    elif imp_level == "sensitivity":
        imp.update(pn_extra_db=imp["phase_noise_sensitivity_dB"], iq_image_dBc=imp["iq_image_sensitivity_dBc"],
                   adc_bits=imp["adc_bits_sensitivity"], pole_sway=dict(imp["pole_sway"]["sensitivity"]))
    IMP = seed_base(fam)["impairments"]
    rs = np.random.default_rng([IMP, F["idx"], site])
    off = rs.uniform(-1, 1) * HEAD_PPB * 1e-9 * FC                     # per-site head offset, both arms
    recs = []
    for m in meta:
        k = m["k"]
        ks = m.get("pair_k", k)                            # C5 quadratic tracks reuse their SUMO pair's seeds
        pre = f"t{k}_"
        t_b = Z[pre + "t_b"]
        tr = Z[pre + "traj"]
        traj = Trajectory(tr[0], tr[1], tr[2], tr[3], tr[4])
        x, y, vx, _ = traj.at(t_b)
        true_site = Site(D=survey.D, pole_offset=survey.pole_offset, dh=HEAD_Z - m["z"])
        heads = true_site.heads()
        r = np.linalg.norm(heads[:, None, :] - np.stack([x, y, np.zeros_like(x)], -1)[None], axis=-1)   # [2, n_b]
        u_x = (heads[:, 0][:, None] - x[None, :]) / r
        rho_true = np.hypot(survey.pole_offset - y, true_site.dh)
        gain_all = Z[pre + "gain"]
        Kr = Z[pre + "K_realised_db"]
        if is_nlosv(fam):
            state, loss = Z[pre + "nlosv_state"], Z[pre + "loss_db"]
            head_tx = (0, 1)
        else:
            state, loss = None, np.zeros(t_b.size)
            head_tx = (0, 1)
        keep = erase(np.random.default_rng([IMP, F["idx"], site, ks, 7]), t_b.size, erasure)
        inw = (x >= WIN_ARC[0] * survey.D) & (x <= WIN_ARC[1] * survey.D)
        t_window = float(t_b[inw][-1] - t_b[inw][0]) if inw.sum() >= 2 else float("inf")
        d_h = heads[:, None, :] - np.stack([x, y, np.zeros_like(x)], -1)[None]
        f_los = (d_h[..., 0] * vx[None, :]) / r / LAMBDA                    # true LoS Doppler per head (Hz)
        per_beacon = m["role"] == "test" and m.get("jr", m["j"]) < PER_BEACON_N and m["kin"] == "sumo"
        p5 = m["cell"] == "P5"
        bplan = beta_plan(fam, m["cell"], m["v0"], m["role"], m.get("jr", m["j"])) if not p5 else []
        betas = [0.0] + (list(P5_BETAS) if p5 and m["role"] == "eval" and m["jr"] < P5_N_BETA_FIRST else bplan)
        track_arms = ("gen_on", "inhouse") if p5 else arms
        with_d = p5 or bool(bplan)                         # consistency statistic on β tracks and their honest pass
        for beta in betas:
            for prb in prbs:
                i = prb_all.index(prb)
                sig_in = None
                if "inhouse" in track_arms:                # in-house: in-band K of this PRB
                    Kb = Z[pre + "K_band_db"] if (pre + "K_band_db") in Z else np.tile(Kr[:, None], (1, len(prb_all)))
                    K_head = (float(Kb[0, i]), float(Kb[1, i]))
                    cfg = ChannelConfig(K_db=K_head[0], K_db_head=K_head, scattering="iso", specular_db=None)
                    t_in, sig_in, _ = itf.inhouse_clean(track_rngs(fam, site, ks)[1], true_site, traj, cfg,
                                                        doppler_scale=1.0 + beta)
                    assert np.allclose(t_in, t_b), "beacon schedules of the paired arms differ"
                sig_g = {}
                for term, name in ((0, "gen_off"), (1, "gen_on")):
                    s = np.stack([itf.interp_q(Z[pre + "hq"][i, term, h], time_scale=1.0 + beta)
                                  for h in range(Z[pre + "hq"].shape[2])])
                    if is_nlosv(fam):
                        sa, ga = pb.assemble_head_a(s[0], s[2], gain_all[0], gain_all[2], state, loss)
                        sig_g[name] = (np.stack([sa, s[1]]), np.stack([ga, gain_all[1]]))
                    else:
                        sig_g[name] = (s[list(head_tx)], gain_all[list(head_tx)])
                gain_ab = sig_g["gen_off"][1]
                for mp in mappings:
                    if mp == "e1":
                        snr = itf.snr_e1_db(r)
                        snr[0] = snr[0] - loss
                    else:
                        # sens_ls: LB large-scale gain shifted, LB SNR only
                        snr = itf.snr_link_budget_db(gain_ab * 10 ** (lb_gain_shift_db / 10))
                    for arm in track_arms:
                        if beta != 0.0 and arm not in ("gen_on", "inhouse"):
                            continue                       # β arms: governing generator arm and in-house only
                        if arm == "inhouse":
                            sig, gain = sig_in, None
                        else:
                            sig, gain = sig_g[arm]
                        # common random numbers: the same offsets, noise and impairments for every arm and every beta
                        rf = np.random.default_rng([IMP, F["idx"], site, ks, prb, 1 if mp == "e1" else 2])
                        yb = itf.finish(rf, sig, snr, gain=gain, head_offset_hz=(off / 2, -off / 2), t_b=t_b)
                        ri = np.random.default_rng([IMP, F["idx"], site, ks, prb, 3])
                        yb = itf.apply_impairments(ri, yb, t_b, u_x, imp)
                        fd = fdoa_from_samples(yb, 1e-3)
                        kp = keep & (np.min(snr, axis=0) >= decode_threshold_db(prb)) if decode else keep
                        st = one_stat(t_b[kp], fd[kp], vx[kp], x[kp], rho_true[kp], survey)
                        if with_d and arm in ("gen_on", "inhouse"):   # the consistency simulation consistency statistic
                            pf, _, _, okf = fit_pass_free(t_b[kp], fd[kp], survey)
                            st["d"] = float((pf[-1] - pf[1]) / pf[1]) if okf else float("nan")
                            try:
                                st["d_d"] = rbf.d_huber(t_b[kp], fd[kp], survey)
                            except Exception:          # noqa: BLE001
                                st["d_d"] = float("nan")
                        if per_beacon and beta == 0.0:   # stage-2 per-beacon subset (constants stage2_per_beacon)
                            fa, fb = foa(yb[0], 1e-3), foa(yb[1], 1e-3)
                            st["per_beacon"] = dict(fdoa_err_hz=np.round(fd - (f_los[0] - f_los[1]), 3).tolist(),
                                                    foa_err_hz_A=np.round(fa - f_los[0], 3).tolist(),
                                                    foa_err_hz_B=np.round(fb - f_los[1], 3).tolist(),
                                                    snr_db=np.round(snr, 2).tolist())
                        recs.append(dict(site=site, track=k, j=m["j"], jr=m.get("jr"), pair_k=m.get("pair_k"), cell=m["cell"], v0=m["v0"], z=m["z"], kin=m["kin"],
                                         role=m["role"], arm=arm, mapping=mp, prb=int(prb), beta=beta, **st,
                                         v_mean=float(np.mean(vx)), y_mean=float(np.mean(y)), lane_change=bool(np.ptp(y) > 1.0),
                                         stopped=bool(np.min(vx) < 0.5),
                                         stop_between=bool(np.any(vx[(x >= 0) & (x <= survey.D)] < 0.5)),
                                         t_window=t_window, window_ok=bool(t_window <= BURST_S),
                                         finite=bool(np.isfinite(st["match"]) and np.isfinite(st["prof"])),
                                         K_realised_db=[float(v) for v in Kr[:2]],
                                         K_inhouse_db=list(K_head) if "inhouse" in track_arms else None,
                                         blocked_share=float(np.mean(state >= 0)) if state is not None else 0.0,
                                         snr_db_median=float(np.median(snr)), kept=int(kp.sum())))
    json.dump(dict(family=fam, site=site, tag=tag, impairments=imp_level, erasure=erasure, recs=recs, provenance=prov,
                   stage1_npz_sha256=S1.get("npz_sha256")),
              open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return path


def decode_threshold_db(prb):
    """Raw-sample SNR (10 MHz noise bandwidth, NF 9 dB) at which a beacon of `prb` PRBs carries the power of the V2X
    reference sensitivity per PRB: TS 36.101 V14.5.0 Table 7.3.1G-1, band 47, 10 MHz (50 RB): -90.4 dBm per port with
    the two-port receiver of clause 7.2, plus 10 log10(2) dB for the one-element head, i.e. 7.6 dB over -95.0 dBm noise for
    50 RB, scaled by 10 log10(prb / 50) (constants evaluation_v5.decoding). A
    minimum-performance requirement, used as the decodability line of the SNR rows."""
    d = C["evaluation_v5"]["decoding"]
    return d["snr_50rb_db"] + 10 * np.log10(prb / d["refsens_rb"])


def _job(a):
    return evaluate_site(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--sites", type=int, nargs=2, default=[0, 20])
    ap.add_argument("--tag", required=True, help="'primary' or dev_<x> / sens_<x> (sensitivities: sens_<x>)")
    ap.add_argument("--impairments", default="nominal", choices=["nominal", "none", "sensitivity"])
    ap.add_argument("--erasure", default=None, help="iid:0.1 | ge:0.3:3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--lb-gain-shift-db", type=float, default=0.0,
                    help="pre-specified sensitivity sens_ls only (constants sensitivities.sens_ls): LB mapping only")
    a = ap.parse_args()
    SL = C["sensitivities"]["sens_ls"]
    SV = C["evaluation_v5"]["sensitivity"]
    LB_TAGS = {SL["tag"]: (SL["lb_gain_shift_db"], SL["families"]),          # sens_ls and the SNR offsets
               **{t: (sh, SV["snr_families"]) for t, sh in SV["snr_tags"].items()}}
    if a.tag in LB_TAGS and (a.lb_gain_shift_db != LB_TAGS[a.tag][0] or a.family not in LB_TAGS[a.tag][1]):
        raise ValueError(f"{a.tag}: --lb-gain-shift-db {LB_TAGS[a.tag][0]} on families {LB_TAGS[a.tag][1]}")
    if a.lb_gain_shift_db and not (a.tag in LB_TAGS or a.tag.startswith("dev_")):
        raise ValueError(f"--lb-gain-shift-db only with the tags {sorted(LB_TAGS)}")
    check_tag(a.tag, stage=2)
    if a.tag == "primary" or a.tag in LB_TAGS:          # full site ranges
        if a.family not in EV4["p2_order"]:
            raise ValueError(f"{a.tag}: family {a.family} is not run")
        if list(a.sites) != EV4["primary_sites"]:      # no extension
            raise ValueError(f"{a.tag} {a.family}: --sites {EV4['primary_sites']}")
    if FAMILIES[a.family].get("seedset") == "p0" and a.tag != EV4["p0"]["tag"]:
        raise ValueError(f"{a.family}: pilot families are evaluated only under the tag {EV4['p0']['tag']!r}")
    if a.tag == "primary" and a.run_dir:
        raise ValueError("primary stage 2 reads runs/g1_<family> only (--run-dir is development only)")
    if a.tag == "primary" and (a.impairments != "nominal" or a.erasure):
        raise ValueError("primary stage 2: nominal impairments and no erasures (sensitivities use sens_<x> tags)")
    run_dir = a.run_dir or os.path.join(ER, "runs", f"g1_{a.family}")
    er = None
    if a.erasure:
        f = a.erasure.split(":")
        er = ("iid", float(f[1])) if f[0] == "iid" else ("ge", float(f[1]), float(f[2]))
    dec = a.tag in SV["snr_tags"] or (a.tag.startswith("dev_") and ("_snr" in a.tag))       # : SNR tags only
    extra = (("gen_on",), ("lb",), None, a.lb_gain_shift_db, None, dec) if a.lb_gain_shift_db else ()   # LB only
    jobs = [(a.family, s, run_dir, a.tag, a.impairments, er) + extra for s in range(*a.sites)
            if os.path.exists(os.path.join(run_dir, f"site_{s:03d}.npz"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(_job, jobs):
            print(p, flush=True)


if __name__ == "__main__":
    main()
