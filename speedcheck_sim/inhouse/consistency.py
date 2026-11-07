"""Consistency simulation: consistency test with RSU-side standardisation from traffic.

Per (site, speed class) the RSU standardises the consistency statistic d with the median and 1.4826 * MAD of d over
its recent traffic (N_TRAFFIC passes; d is physical and independent of the VSS; the median/MAD resist a manipulating
minority), instead of the 30 survey passes only. Head offset handling as the base consistency simulation (none in the free fit; constant offset
within +-5 ppb absorbed by the standardisation).
Per site (6 sites per cell): 30 survey passes, 150 honest traffic passes (running window), 100 further honest
passes (evaluated) and 30 manipulated passes per beta (-5 %, -10 %). For every evaluated pass both standardisations are stored:
  z_ref = (d - median_ref) / (1.15 * 1.4826 * MAD_ref)       (survey passes)
  z_trf = (d - median_trf) / (1.4826 * MAD_trf)              (traffic window)
Resumable: one checkpoint per (cell, site) job (checkpoint.py).
Usage: python inhouse/consistency.py [--smoke] [--workers 6]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

from checkpoint import job_seed, run_jobs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdoa.channel import ChannelConfig, draw_reflector, simulate_pass
from fdoa.estimator import fdoa_from_samples, fit_pass_free
from fdoa.geometry import FC, Site, make_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
BETAS = (-0.05, -0.10)                   # -3 % and the ramp are covered by the base consistency simulation
HEAD_PPB = 5.0
N_REF, N_TRAFFIC, N_EVAL, N_SPOOF = 30, 150, 100, 30
N_SITES = 6


def label(b):
    return f"ramp{b[1]:+.2f}" if isinstance(b, tuple) else f"{b:+.2f}"


def d_pass(rng, site, cell, cfg, refl, beta):
    traj = make_trajectory(rng, site, cell["v0"], p_lane_change=0.1)
    ds = beta if isinstance(beta, tuple) else 1.0 + beta
    t_b, y, _ = simulate_pass(rng, site, traj, cfg, reflector=refl, doppler_scale=ds)
    p, _, _, ok = fit_pass_free(t_b, fdoa_from_samples(y, cfg.packet_s), site)
    return float((p[-1] - p[1]) / p[1]) if ok else np.nan


def median_mad(d):
    d = d[np.isfinite(d)]
    med = np.median(d)
    return med, 1.4826 * np.median(np.abs(d - med))


def run_site(args):
    cell, n_ref, n_trf, n_eval, n_spoof, seed = args
    rng = np.random.default_rng(seed)
    off = rng.uniform(-1, 1) * HEAD_PPB * 1e-9 * FC
    cfg = ChannelConfig(K_db=cell["K_db"], scattering="iso", specular_db=cell["specular_db"],
                        head_offset_hz=(off / 2, -off / 2))
    site = Site(D=100.0, pole_offset=rng.uniform(6.0, 12.0), dh=rng.uniform(4.5, 6.5))
    refl = draw_reflector(rng, site)
    ref = np.array([d_pass(rng, site, cell, cfg, refl, 0.0) for _ in range(n_ref)])
    trf = np.array([d_pass(rng, site, cell, cfg, refl, 0.0) for _ in range(n_trf)])
    ev = np.array([d_pass(rng, site, cell, cfg, refl, 0.0) for _ in range(n_eval)])
    sp = {b: np.array([d_pass(rng, site, cell, cfg, refl, b) for _ in range(n_spoof)]) for b in BETAS}
    m_ref, s_ref = median_mad(ref)
    m_trf, s_trf = median_mad(trf)
    s_ref *= 1.15
    zr = lambda d: (d - m_ref) / s_ref
    zt = lambda d: (d - m_trf) / s_trf
    return dict(ev_ref=zr(ev), ev_trf=zt(ev), sp_ref={b: zr(v) for b, v in sp.items()},
                sp_trf={b: zt(v) for b, v in sp.items()}, sd_d_ref=float(s_ref / 1.15), sd_d_trf=float(s_trf),
                offset_hz=float(off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.smoke:
        cells = [dict(v0=20.0, K_db=10.0, specular_db=None)]; ns, nr, nt, ne, nsp = 2, 15, 30, 20, 6
    else:
        cells = [dict(v0=v, K_db=K, specular_db=sp) for v, K, sp in
                 itertools.product((10.0, 15.0, 20.0, 30.0, 38.0), (6.0, 10.0, 15.0, 20.0), (None, -10.0))]
        ns, nr, nt, ne, nsp = N_SITES, N_REF, N_TRAFFIC, N_EVAL, N_SPOOF
    t0 = time.time()
    tag = "smoke" if a.smoke else "full"
    jobs = [(f"c{i:02d}_s{sid:02d}", (c, nr, nt, ne, nsp, job_seed("e3v5", tag, i, sid)))
            for i, c in enumerate(cells) for sid in range(ns)]
    ckdir = os.path.join(OUT, f"ckpt_consistency_{tag}_s{ns}_r{nr}_t{nt}_e{ne}_p{nsp}_b{len(BETAS)}")   # config in the name
    done = run_jobs(run_site, jobs, ckdir, a.workers)
    arrays, rep = {}, []
    for i, c in enumerate(cells):
        parts = [done[f"c{i:02d}_s{sid:02d}"] for sid in range(ns)]
        row = dict(c, sites=[dict(sd_d_ref=p["sd_d_ref"], sd_d_trf=p["sd_d_trf"], offset_hz=p["offset_hz"]) for p in parts])
        for kind in ("ref", "trf"):
            arrays[f"c{i}_honest_{kind}"] = np.concatenate([p[f"ev_{kind}"] for p in parts])
            for b in BETAS:
                arrays[f"c{i}_beta{label(b)}_{kind}"] = np.concatenate([p[f"sp_{kind}"][b] for p in parts])
            z = arrays[f"c{i}_honest_{kind}"]; z = z[np.isfinite(z)]
            row[f"honest_{kind}"] = dict(mean=float(z.mean()), sd=float(z.std()), q001=float(np.quantile(z, 0.001)),
                                         share_lt_m3=float(np.mean(z < -3)))
        rep.append(row)
        print(f"v={c['v0']:.0f} K={c['K_db']:.0f} spec={c['specular_db']}: honest ref sd {row['honest_ref']['sd']:.2f} "
              f"(<-3 {row['honest_ref']['share_lt_m3']:.3f}) | traffic sd {row['honest_trf']['sd']:.2f} "
              f"(<-3 {row['honest_trf']['share_lt_m3']:.3f})", flush=True)
    json.dump(dict(cells=rep, head_ppb=HEAD_PPB, sites=ns, ref=nr, traffic=nt, eval=ne, spoof=nsp,
                   seconds=time.time() - t0), open(os.path.join(OUT, f"consistency_{tag}.json"), "w"), indent=1, default=str)
    np.savez_compressed(os.path.join(OUT, f"consistency_z_{tag}.npz"), **arrays)
    print(f"elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
