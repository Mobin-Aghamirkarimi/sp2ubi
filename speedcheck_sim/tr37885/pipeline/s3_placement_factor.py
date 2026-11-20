"""Study 3: snapshot-placement factor of the urban estimator on the placement sites 40-44, with the factor of the base
estimator recomputed from the same crossings by the same code. Reported only; it is not the factor that enters the
conservative error.

Input (pipeline/s3_snapshot_placement.py): <dir>/P1_site<k>_series.npz (per-beacon series; records in the order calls
B, A, N x crossing x mapping e1, lb) and <dir>/P1_site<k>.json (statistics and inclusion flags of the base estimator
for the same crossings).
Per site, call and mapping, f_o and rho are the medians of the free fits (constant-acceleration model, pulse-pair
series) of the call's crossings, i.e. the constants come from the same crossings and not from a separate survey; the
urban estimator (fdoa/urban.py) then runs with these constants.
Ratio as in snapshot_placement.summarize: common included set over B, A, N, statistic selected on the B call, sigma_w
of the within-site-centered, clipped statistic, rho = sigma_w(A) / sigma_w(B), null rho_N = sigma_w(N) / sigma_w(B),
site bootstrap.
Output keys per mapping: "A" (urban estimator on the crossings that the base estimator includes and for which its own
statistics are finite), "A_own_inclusion" (urban estimator, its own finite statistics only), "registered_recomputed"
(base estimator); "s_A" is the larger rho of the urban estimator over the two mappings.
Usage: python pipeline/s3_placement_factor.py --dir runs/s_place_dev_s3_place --out <json>"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, "..")]
from fdoa import urban as U  # noqa: E402
from fdoa.geometry import Site  # noqa: E402
from pipeline import snapshot_placement as SPL  # noqa: E402
from pipeline.provenance import rng_for  # noqa: E402

CALLS, MAPS = ("B", "A", "N"), ("e1", "lb")


def site_stats(a):
    d, s, L, dh = a
    Z = np.load(os.path.join(d, f"P1_site{s:03d}_series.npz"))
    n_rec = int(Z["n_records"])
    n_pass = n_rec // 6
    if n_rec != 6 * n_pass:
        raise RuntimeError(f"site {s}: {n_rec} records, not 3 calls x crossings x 2 mappings")
    site = Site(D=L, pole_offset=float(Z["pole_offset"]), dh=dh)
    out = {c: {mp: [] for mp in MAPS} for c in CALLS}
    for ci, c in enumerate(CALLS):
        for mi, mp in enumerate(MAPS):
            idx = [ci * 2 * n_pass + 2 * j + mi for j in range(n_pass)]
            free = []
            for n in idx:
                try:
                    F = U.fit(Z[f"r{n}_t"], Z[f"r{n}_fpp"].astype(float), site)
                    free.append((F["fo"], F["rho"]))
                except Exception:                        # noqa: BLE001
                    free.append((np.nan, np.nan))
            fo, rho = (float(np.nanmedian([q[i] for q in free])) for i in (0, 1))
            for n in idx:
                fd, v = Z[f"r{n}_fpp"].astype(float), Z[f"r{n}_v"].astype(float)
                try:
                    F = U.fit(Z[f"r{n}_t"], fd, site, fo=fo, rho=rho, node_s=U.NODE_S)
                    prof, match, _ = U.statistics(F, fd, v, core=U.CORE, D=site.D)
                except Exception:                        # noqa: BLE001
                    prof = match = float("nan")
                out[c][mp].append([prof, match, bool(np.isfinite(prof) and np.isfinite(match))])
            out[c][mp + "_const"] = dict(fo=fo, rho=rho)
    return s, out


def sigma_w(per, sites, key_of, ok_of, win):
    e = []
    for s in sites:
        v = np.array([x[key_of[s]] for x in per[s]], float)[ok_of[s]]
        e.append(np.clip(v - v.mean(), -win, win))
    return float(np.std(np.concatenate(e))) if e else float("nan")


def ratio(recs, ok_given, mp, win, rng, boot):
    """recs[s][call][mp] = list of [prof, match, finite]; ok_given[s] = inclusion set from outside (bool array)."""
    key_of, ok_of = {}, {}
    for s in sorted(recs):
        ok = ok_given[s] & np.logical_and.reduce([np.array([x[2] for x in recs[s][c][mp]], bool) for c in CALLS])
        if ok.sum() < 2:
            continue
        b = recs[s]["B"][mp]
        pr, ma = np.array([x[0] for x in b], float)[ok], np.array([x[1] for x in b], float)[ok]
        key_of[s], ok_of[s] = (1 if np.var(ma) <= np.var(pr) else 0), ok
    sites = sorted(key_of)
    per = {c: {s: recs[s][c][mp] for s in sites} for c in CALLS}
    sb, sa, sn = (sigma_w(per[c], sites, key_of, ok_of, win) for c in CALLS)
    ba, bn = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(boot):
            ss = [sites[i] for i in rng.integers(0, len(sites), len(sites))]
            b0 = sigma_w(per["B"], ss, key_of, ok_of, win)
            ba.append(sigma_w(per["A"], ss, key_of, ok_of, win) / b0)
            bn.append(sigma_w(per["N"], ss, key_of, ok_of, win) / b0)
    ci = lambda x: [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]  # noqa: E731
    return dict(n_sites=len(sites), passes=int(sum(ok_of[s].sum() for s in sites)), sigma_w=dict(B=sb, A=sa, N=sn),
                rho=sa / sb, rho_ci=ci(ba), rho_null=sn / sb, rho_null_ci=ci(bn),
                select={str(s): ("match" if key_of[s] else "prof") for s in sites})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", type=int, nargs=2, default=[40, 45])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    C = SPL.C
    L, dh, win, boot = C["geometry"]["L_m"], float(SPL.HEAD_Z - SPL.VEH_Z), SPL.WIN, C["run"]["boot"]
    with ProcessPoolExecutor(a.workers) as ex:
        A = dict(ex.map(site_stats, [(a.dir, s, L, dh) for s in range(*a.sites)]))
    reg = {s: SPL.use_statistic(json.load(open(os.path.join(a.dir, f"P1_site{s:03d}.json")))["calls"]) for s in A}
    res = dict(label="placement factor under A, constants from the same crossings (not a separate survey)",
               sites=sorted(A), constants={str(s): {c: {mp: A[s][c][mp + "_const"] for mp in MAPS} for c in CALLS} for s in A})
    for mp in MAPS:
        ok_reg = {s: np.logical_and.reduce([np.array([x[2] for x in reg[s][c][mp]], bool) for c in CALLS]) for s in A}
        all_ok = {s: np.ones_like(ok_reg[s]) for s in A}
        res[mp] = dict(
            A=ratio(A, ok_reg, mp, win, rng_for("S_place_A", "P1", mp), boot),
            A_own_inclusion=ratio(A, all_ok, mp, win, rng_for("S_place_A_own", "P1", mp), boot),
            registered_recomputed=ratio(reg, all_ok, mp, win, rng_for("S_place", "P1", mp), boot))
    res["s_A"] = float(max(res[mp]["A"]["rho"] for mp in MAPS))
    json.dump(res, open(a.out + ".tmp", "w"), indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    print(a.out)


if __name__ == "__main__":
    main()
