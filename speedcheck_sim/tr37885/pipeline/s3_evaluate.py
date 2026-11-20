"""Study 3: evaluation of the base estimator and of the urban estimator (or a comparison variant) on the same crossings
of one site family. Per speed and estimator, each quantity is the least favorable value over the six mapping x PRB
variants (k: the smallest count); the mean errors and the entry yield are those of the variant with the largest sigma
of the qualified sites. Intervals: Clopper-Pearson for k/n; site bootstrap (BOOT draws, the same site draws for every
estimator) for sigma, the mean error after calibration, the out-of-window share and the ratio of sigma over all
crossings to that of the base estimator.

Output keys: <estimator>|<speed>, estimator = ls (base estimator) or a name of pipeline/s3_estimate.py.
Usage: python pipeline/s3_evaluate.py --res <dir> --series <dir> --family U3 --variants h_pp_now_c20 --out <json>
           [--survey-role reff] [--speeds 10 15]
--survey-role (U3 only): ref = survey in traffic (default), reff = free-road survey; the result folder must come from
pipeline/s3_estimate.py with the same role."""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pipeline import s3_rules as S  # noqa: E402

BOOT = 2000
SEED = 20261112
WORST_OF = ("sigma_qg", "sigma_all", "q99", "oow", "gate", "sd_ref_med", "site_mean_sd")


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def worst(per, key):
    v = [p[key] for p in per.values() if np.isfinite(p[key])]
    return max(v) if v else float("nan")


def worst_variant(per):
    """The mapping x PRB variant with the largest sigma of the qualified sites."""
    return max(per, key=lambda var: -1 if not np.isfinite(per[var]["sigma_qg"]) else per[var]["sigma_qg"])


def speed_tables(recs, v0):
    T = {var: S.tables(recs, v0, *var) for var in S.VARIANTS}
    return T, {var: S.cell_summary(T[var]) for var in S.VARIANTS}


def bootstrap_draw(T, sites, rng, keys):
    """One draw of the sites with replacement: the worst-of-variants value of each key and the mean error."""
    draw = rng.choice(sites, size=len(sites), replace=True)
    per = {var: S.cell_summary({i: T[var].get(s) for i, s in enumerate(draw)}) for var in T}
    out = {k: worst(per, k) for k in keys}
    out["bias_cal"] = per[worst_variant(per)]["bias_cal"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speeds", type=float, nargs="+", default=[10.0, 15.0])
    ap.add_argument("--survey-role", default="ref", choices=["ref", "reff"])
    a = ap.parse_args()
    names = [S.BASE] + a.variants
    res = {}
    pct = lambda x: f"{100 * x:6.2f}"  # noqa: E731
    for v0 in a.speeds:
        data = {n: speed_tables(S.load(a.res, a.series, a.family, n, a.survey_role), v0) for n in names}
        sites = sorted({s for var in S.VARIANTS for s in data[S.BASE][0][var]})
        for n in names:
            T, per = data[n]
            k_min, wv = min(p["k"] for p in per.values()), worst_variant(per)
            r = dict(k=k_min, k_max=max(p["k"] for p in per.values()), n=len(sites), cp=clopper_pearson(k_min, len(sites)),
                     **{k: worst(per, k) for k in WORST_OF},
                     bias_cal=per[wv]["bias_cal"], bias_uncal=per[wv]["bias_uncal"], entry_yield=per[wv]["entry_yield"],
                     worst_variant=f"{wv[0]}/{wv[1]}")
            rng = np.random.default_rng([SEED, int(v0)])
            rng_base = np.random.default_rng([SEED, int(v0)])
            bs, ratio = {k: [] for k in ("sigma_qg", "sigma_all", "bias_cal", "oow")}, []
            for _ in range(BOOT):
                o = bootstrap_draw(T, sites, rng, ("sigma_qg", "sigma_all", "oow"))
                for k in bs:
                    bs[k].append(o[k])
                if n != S.BASE:
                    ratio.append(o["sigma_all"] / bootstrap_draw(data[S.BASE][0], sites, rng_base, ("sigma_all",))["sigma_all"])
            for k, x in bs.items():
                x = np.array(x, float)
                x = x[np.isfinite(x)]
                r[k + "_ci"] = [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if x.size else [np.nan, np.nan]
                r[k + "_boot_nonfinite"] = float(1 - x.size / BOOT)
            if ratio:
                r["sigma_all_ratio"] = float(r["sigma_all"] / worst(data[S.BASE][1], "sigma_all"))
                r["sigma_all_ratio_ci"] = [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))]
            r["per_site"] = {int(s): dict(sd_ref=T[wv][s]["sd_ref"], qualified=T[wv][s]["qualified"],
                                          sd_test=S.sigma(T[wv][s]["e"][T[wv][s]["keep"]]),
                                          mean_e=float(np.mean(T[wv][s]["e"][T[wv][s]["keep"]])),
                                          stop_share=float(np.mean(T[wv][s]["stop"])), key=T[wv][s]["key"])
                             for s in sites if T[wv].get(s) is not None}
            res[f"{n}|{v0:g}"] = r
            print(f"v{v0:g} {n:24s} k {k_min}/{len(sites)} [{100 * r['cp'][0]:.1f}-{100 * r['cp'][1]:.1f}] sd_ref_med {pct(r['sd_ref_med'])} "
                  f"sig_qg {pct(r['sigma_qg'])} [{pct(r['sigma_qg_ci'][0])},{pct(r['sigma_qg_ci'][1])}] sig_all {pct(r['sigma_all'])} "
                  f"q99 {pct(r['q99'])} oow {pct(r['oow'])} gate {pct(r['gate'])} bias_cal {pct(r['bias_cal'])} "
                  f"[{pct(r['bias_cal_ci'][0])},{pct(r['bias_cal_ci'][1])}] site_sd {pct(r['site_mean_sd'])}"
                  + (f" ratio_all {r['sigma_all_ratio']:.2f} [{r['sigma_all_ratio_ci'][0]:.2f},{r['sigma_all_ratio_ci'][1]:.2f}]" if ratio else ""))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
