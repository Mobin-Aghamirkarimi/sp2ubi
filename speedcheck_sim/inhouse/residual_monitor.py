"""RSU self-monitor: change detection against the site's own reference.

Per site and speed class the RSU stores the reference level of the per-pass fit-residual RMS (median over the 30
survey passes). In operation it keeps the median over the last N_WIN passes (all traffic, no VSS data) and acts on
the ratio rho = running median / reference median:
  rho > RAISE  -> the disclosed sigma level is raised by one step (raise-only; at most one raise per epoch);
  rho > STOP   -> the (site, speed class) stops producing entries until re-survey by the survey vehicle.
Simulated per site: 30 reference passes, 100 passes before a change, 100 after. Changes (from pass 0 of the
"after" part): none (false-alarm control), a specular reflector appears (-6 dB), the K-factor drops by 6 dB, the
noise of the error grows via both (reflector + K - 6 dB). For every pass also the speed-check error of the "select"
statistic is recorded, so the actual sigma before/after each change is known.
Reported per cell: false raise/stop rate per window position without change; detection probability and latency
(passes after the change) of raise and stop; actual sigma ratio after/before.
Resumable: one checkpoint per (cell, site) job in inhouse/results/ckpt_monitor_<tag>/ (checkpoint.py).
Usage: python inhouse/residual_monitor.py [--sites 6 --workers 6] [--smoke]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

import site_accuracy as E1
from checkpoint import job_seed, run_jobs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdoa.channel import ChannelConfig, draw_reflector, simulate_pass
from fdoa.estimator import fdoa_from_samples
from fdoa.geometry import FC, Site, make_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
N_REF, N_PRE, N_POST = 30, 100, 100
N_WIN = 25
RAISE, STOP = 1.25, 1.6
HEAD_PPB = 5.0
CHANGES = ("none", "reflector", "kdrop", "both")


def run_site(args):
    """One (cell, site) job: reference, pre-change and post-change passes of one site."""
    cell, seed = args
    E1.OFFSET_FIT = True
    rng = np.random.default_rng(seed)
    off = rng.uniform(-1, 1) * HEAD_PPB * 1e-9 * FC
    site = Site(D=100.0, pole_offset=rng.uniform(6.0, 12.0), dh=rng.uniform(4.5, 6.5))
    base = dict(K_db=cell["K_db"], scattering="iso", head_offset_hz=(off / 2, -off / 2))
    cfg0 = ChannelConfig(specular_db=cell["specular_db"], **base)
    refl0 = draw_reflector(rng, site)
    new_refl = draw_reflector(rng, site)
    ch = cell["change"]
    if ch == "none":
        cfg1, refl1 = cfg0, refl0
    elif ch == "reflector":
        cfg1, refl1 = ChannelConfig(specular_db=-6.0, **base), new_refl
    elif ch == "kdrop":
        cfg1, refl1 = ChannelConfig(specular_db=cell["specular_db"], **dict(base, K_db=cell["K_db"] - 6.0)), refl0
    else:
        cfg1, refl1 = ChannelConfig(specular_db=-6.0, **dict(base, K_db=cell["K_db"] - 6.0)), new_refl
    rows = []
    for i in range(N_REF + N_PRE + N_POST):
        cfg, refl = (cfg0, refl0) if i < N_REF + N_PRE else (cfg1, refl1)
        traj = make_trajectory(rng, site, cell["v0"], p_lane_change=0.1)
        t_b, y, _ = simulate_pass(rng, site, traj, cfg, reflector=refl)
        fd = fdoa_from_samples(y, cfg.packet_s)
        x, yy, vx, _ = traj.at(t_b)
        st = E1.statistics(t_b, fd, vx, x, np.hypot(site.pole_offset - yy, site.dh), site)
        rows.append((st[0], st[1], np.sqrt(st[4])))
    return np.array(rows, float)


def analyse(cell, sites):
    fr_raise = fr_stop = n_pos = 0
    lat_raise, lat_stop, ratio = [], [], []
    for a in sites:
        prof, match, rms = a[:, 0], a[:, 1], a[:, 2]
        use_m = np.var(match[:N_REF]) <= np.var(prof[:N_REF])
        e = match if use_m else prof
        ref_level = np.median(rms[:N_REF])
        run = np.array([np.median(rms[max(N_REF, k - N_WIN + 1):k + 1]) for k in range(N_REF, len(rms))]) / ref_level
        pre = run[N_WIN - 1:N_PRE]                     # full windows before the change
        fr_raise += int(np.sum(pre > RAISE)); fr_stop += int(np.sum(pre > STOP)); n_pos += pre.size
        post = run[N_PRE:]
        r_hit = np.nonzero(post > RAISE)[0]; s_hit = np.nonzero(post > STOP)[0]
        lat_raise.append(int(r_hit[0]) + 1 if r_hit.size else None)
        lat_stop.append(int(s_hit[0]) + 1 if s_hit.size else None)
        ratio.append(float(np.std(e[N_REF + N_PRE:]) / np.std(e[N_REF:N_REF + N_PRE])))
    det_r = [x for x in lat_raise if x is not None]; det_s = [x for x in lat_stop if x is not None]
    return dict(cell, false_raise_rate=fr_raise / max(n_pos, 1), false_stop_rate=fr_stop / max(n_pos, 1),
                p_raise=len(det_r) / len(sites), p_stop=len(det_s) / len(sites),
                latency_raise_median=float(np.median(det_r)) if det_r else None,
                latency_stop_median=float(np.median(det_s)) if det_s else None,
                sigma_ratio_after_before_median=float(np.median(ratio)), sigma_ratio_all=ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cells = [dict(v0=v, K_db=K, specular_db=sp, change=ch) for v, K, sp, ch in
             itertools.product((10.0, 20.0, 30.0), (10.0, 20.0), (None,), CHANGES)]
    if a.smoke:
        cells = [dict(v0=20.0, K_db=15.0, specular_db=None, change="both")]
    ns = 1 if a.smoke else a.sites
    t0 = time.time()
    tag = "smoke" if a.smoke else "full"
    jobs = [(f"c{i:02d}_s{sid:02d}", (c, job_seed("monitor", tag, i, sid))) for i, c in enumerate(cells) for sid in range(ns)]
    done = run_jobs(run_site, jobs, os.path.join(OUT, f"ckpt_monitor_{tag}"), a.workers)
    res = [(c, [done[f"c{i:02d}_s{sid:02d}"] for sid in range(ns)]) for i, c in enumerate(cells)]
    rep = [analyse(c, s) for c, s in res]
    for r in rep:
        print(f"v={r['v0']:.0f} K={r['K_db']:.0f} change={r['change']:9s}: false raise {r['false_raise_rate']:.3f} stop "
              f"{r['false_stop_rate']:.3f} | P(raise) {r['p_raise']:.2f} lat {r['latency_raise_median']} | P(stop) "
              f"{r['p_stop']:.2f} lat {r['latency_stop_median']} | actual sigma after/before {r['sigma_ratio_after_before_median']:.2f}",
              flush=True)
    json.dump(dict(cells=rep, n_win=N_WIN, raise_=RAISE, stop=STOP, seconds=time.time() - t0),
              open(os.path.join(OUT, f"residual_monitor_{'smoke' if a.smoke else 'full'}.json"), "w"), indent=1)
    print(f"elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
