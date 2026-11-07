"""Site-accuracy simulation: per-pass speed-check error with site structure, for the matched and profile statistics.

Changes from e1_fdoa_accuracy.py (kept for the first grid):
  * sites: each cell has N_SITES sites with their own surveyed geometry (pole offset 6-12 m, head height 4.5-6.5 m
    above the vehicle antenna, known to the RSU) and, where present, a site-FIXED specular reflector; passes are
    nested in sites, so between-site bias is measured instead of averaged away;
  * lane changes with probability 0.1, optional mean deceleration (vehicles slowing towards a junction);
  * statistics per pass: profile (fitted smooth speed), matched (geometry from the fit, speed profile from the VSS),
    matched with TRUE geometry (diagnostic: separates estimation error of g from channel-induced bias);
  * two analysis windows: full pass, and +-4 s around the crossing of the mid-point between the heads;
  * per-pass records (incl. the fitted speed class, the RSU's own covariate) saved for the decision simulation.
Honest VSS here is perfect (v_sen = v_true); VSS errors are added in the decision simulation.

Usage: python inhouse/site_accuracy.py [--smoke] [--sites 20 --passes 100 --workers 8]
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdoa.channel import ChannelConfig, draw_reflector, simulate_pass
from fdoa.estimator import _g, fdoa_from_samples, fit_pass, fit_pass_offset, fitted_speed
from fdoa.geometry import FC, LAMBDA, Site, make_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
HALF_WINDOW_S = 4.0


OFFSET_FIT = False        # estimate the head-to-head offset delta as a nuisance parameter


def statistics(t_b, fd, vx, x_true, rho_true, site):
    """Fit the pass and return (profile, matched, matched_true_geometry, v_mid, rss/n, n) for these beacons."""
    if OFFSET_FIT:
        q, tm, resid, ok = fit_pass_offset(t_b, fd, site)
        p, delta = q[:4], q[4]
    else:
        p, tm, resid, ok = fit_pass(t_b, fd, site)
        delta = 0.0
    fdc = fd - delta
    tau = t_b - tm
    vf = fitted_speed(p, t_b, tm)
    g = _g(p[0] + p[1] * tau + 0.5 * p[2] * tau ** 2, site.D, p[3])
    g0 = _g(x_true, site.D, rho_true)
    return (float(np.sum(vf * vx) / np.sum(vf * vf) - 1), float(np.sum(g * g * vx) / (LAMBDA * np.sum(g * fdc)) - 1),
            float(np.sum(g0 * g0 * vx) / (LAMBDA * np.sum(g0 * fdc)) - 1), float(p[1]), float(np.mean(resid ** 2)),
            len(t_b), bool(ok), p, tm)


def run_cell(args):
    cell, n_sites, n_passes, seed = args
    global OFFSET_FIT
    OFFSET_FIT = cell.get("offset_fit", False)
    rng = np.random.default_rng(seed)
    recs = []
    for sid in range(n_sites):
        # per-site head-to-head reference offset (GNSS-disciplined RSU clocks), uniform +-head_ppb
        off = rng.uniform(-1, 1) * cell.get("head_ppb", 0.0) * 1e-9 * FC
        cfg = ChannelConfig(K_db=cell["K_db"], scattering=cell["scattering"], specular_db=cell["specular_db"],
                            packet_s=cell["packet_s"], head_offset_hz=(off / 2, -off / 2))
        site = Site(D=cell["D"], pole_offset=rng.uniform(6.0, 12.0), dh=rng.uniform(4.5, 6.5))
        refl = draw_reflector(rng, site)
        for _ in range(n_passes):
            traj = make_trajectory(rng, site, cell["v0"], p_lane_change=cell["p_lane_change"], mean_acc=cell["mean_acc"])
            t_b, y, fD = simulate_pass(rng, site, traj, cfg, reflector=refl)
            fd = fdoa_from_samples(y, cfg.packet_s)
            x, yy, vx, _ = traj.at(t_b)
            rho_true = np.hypot(site.pole_offset - yy, site.dh)
            full = statistics(t_b, fd, vx, x, rho_true, site)
            p, tm = full[7], full[8]
            # window: +-HALF_WINDOW_S around the fitted crossing of x = D/2
            tc = tm + (site.D / 2 - p[0]) / max(p[1], 0.5)
            m = np.abs(t_b - tc) <= HALF_WINDOW_S
            win = statistics(t_b[m], fd[m], vx[m], x[m], rho_true[m], site) if m.sum() >= 12 else (np.nan,) * 6 + (False, None, None)
            recs.append(dict(site=sid, prof=full[0], match=full[1], match_true=full[2], v_fit=full[3], mse=full[4],
                             n=full[5], ok=full[6], w_prof=win[0], w_match=win[1], w_match_true=win[2], w_n=win[5],
                             v_mean=float(np.mean(vx)), lane_change=bool(np.ptp(yy) > 1.0)))
    return cell, recs


def summarize(cell, recs):
    out = dict(cell)
    sites = np.array([r["site"] for r in recs])
    for key in ("prof", "match", "match_true", "w_prof", "w_match", "w_match_true"):
        e = np.array([r[key] for r in recs], float)
        ok = np.isfinite(e)
        site_means = np.array([np.nanmean(e[sites == s]) for s in np.unique(sites)])
        within = np.sqrt(np.nanmean([np.nanvar(e[sites == s]) for s in np.unique(sites)]))
        out.update({f"{key}_bias": 100 * float(np.nanmean(e)), f"{key}_std": 100 * float(np.nanstd(e)),
                    f"{key}_within": 100 * float(within), f"{key}_between": 100 * float(np.std(site_means)),
                    f"{key}_q01": 100 * float(np.nanquantile(e, 0.01)), f"{key}_q99": 100 * float(np.nanquantile(e, 0.99)),
                    f"{key}_valid": int(ok.sum())})
    out["fit_fail"] = int(sum(not r["ok"] for r in recs))
    out["lane_change_share"] = float(np.mean([r["lane_change"] for r in recs]))
    return out


def cells_v3():
    """the site-accuracy simulation grid: offset fit, per-site head offset up to 5 ppb, K down to 6 dB, v up to 38 m/s."""
    return [dict(D=100.0, v0=v, K_db=K, scattering=sc, specular_db=sp, packet_s=1e-3, p_lane_change=0.1, mean_acc=0.0,
                 offset_fit=True, head_ppb=5.0)
            for v, K, sc, sp in itertools.product((10.0, 15.0, 20.0, 30.0, 38.0), (6.0, 10.0, 15.0, 20.0),
                                                 ("iso", "side"), (None, -10.0))]


def cells(smoke, k_list=(15.0, 20.0), with_decel=True):
    if smoke:
        return [dict(D=100.0, v0=10.0, K_db=20.0, scattering="iso", specular_db=-10.0, packet_s=1e-3, p_lane_change=0.1, mean_acc=0.0)]
    base = [dict(D=100.0, v0=v, K_db=K, scattering=sc, specular_db=sp, packet_s=1e-3, p_lane_change=0.1, mean_acc=0.0)
            for v, K, sc, sp in itertools.product((10.0, 15.0, 20.0, 30.0), k_list, ("iso", "side"), (None, -10.0))]
    decel = [dict(D=100.0, v0=15.0, K_db=20.0, scattering="iso", specular_db=sp, packet_s=1e-3, p_lane_change=0.1, mean_acc=-0.5)
             for sp in (None, -10.0)]
    return base + (decel if with_decel else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sites", type=int, default=20)
    ap.add_argument("--passes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--k-list", type=float, nargs="+", default=[15.0, 20.0])
    ap.add_argument("--tag", default=None, help="output tag (default: smoke/full); extra K cells use e.g. --tag k10")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cs = cells_v3() if a.tag == "extra" else cells(a.smoke, tuple(a.k_list), with_decel=a.tag is None)
    ns, npass = (4, 15) if a.smoke else (a.sites, a.passes)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        base_seed = 5000 if a.tag is None else 5000 + 1000 * (1 + sum(map(ord, a.tag)) % 50)
        res = list(ex.map(run_cell, [(c, ns, npass, base_seed + i) for i, c in enumerate(cs)]))
    tag = a.tag or ("smoke" if a.smoke else "full")
    summ = [summarize(c, r) for c, r in res]
    json.dump(dict(cells=summ, sites=ns, passes_per_site=npass, half_window_s=HALF_WINDOW_S, seconds=time.time() - t0),
              open(os.path.join(OUT, f"site_accuracy_summary_{tag}.json"), "w"), indent=1)
    json.dump([dict(cell=c, recs=r) for c, r in res], open(os.path.join(OUT, f"site_accuracy_passes_{tag}.json"), "w"))
    for s in summ:
        print(f"v={s['v0']:.0f} K={s['K_db']:.0f} {s['scattering']:4s} spec={s['specular_db']} acc={s['mean_acc']} off={s.get('offset_fit', False)}: "
              f"prof {s['prof_std']:.2f}({s['prof_bias']:+.2f}) | match {s['match_std']:.2f}({s['match_bias']:+.2f}) "
              f"[within {s['match_within']:.2f}, between {s['match_between']:.2f}] | true-geom {s['match_true_std']:.2f}({s['match_true_bias']:+.2f}) "
              f"| win: prof {s['w_prof_std']:.2f} match {s['w_match_std']:.2f}({s['w_match_bias']:+.2f}) | fail {s['fit_fail']}", flush=True)
    print(f"elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
