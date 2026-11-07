"""Base consistency analysis: the consistency rule of the final design, with exact tail probabilities.

The RSU discloses z_q = round(clip(z, -3, 3) / q) (q = 1.0: 7 values; q = 3.0: 3 values) when the consistency test is
enabled at the (site, speed class), z_q = 0 when the fit fails, and a distinct value NA ("not tested") when the test is
disabled; NA passes are excluded from the rule. (A first version disclosed 0 for disabled cells: the zeros narrowed the
insurer's population distribution and made the threshold too tight for tested cells -- worst-cell epoch false alarm
2.8e-2 at 26 % enabled -- so NA is required.) The insurer flags an epoch when the sum of the M TESTED disclosed values
is <= t_M, where t_M is the largest integer with P_pop(sum <= t_M) <= ALPHA under the honest POPULATION distribution
(pooled over tested passes with K-population x facade weights; drift arm 0). Because z_q is
discrete, P(sum of M) is computed exactly by M-fold convolution of the per-pass pmf (no Monte Carlo tail error).
Reported:
  * qualification: a (site, speed class) enables the test only if sd(d) on its reference passes <= D_MAX;
    weighted share of enabled passes per D_MAX;
  * honest false alarm per epoch: pooled (<= ALPHA by construction) and per single cell (a vehicle whose M passes all
    come from one cell; worst case), and per year through the 2-of-3 ladder (exact over 12 epochs);
  * drift arms (head offset moved by 0.1 / 0.3 / 0.5 ppb since calibration) and a site-common shift of z by
    -0.1 ... -0.4 sd: per-cell epoch and ladder rates -> the stability bound for H1;
  * power: P(flag per epoch) when all M passes are manipulated (beta -3 / -5 / -10 %, ramp -5 %), per cell and
    population-weighted per speed profile.
Usage: python inhouse/consistency_analysis_base.py [full]
"""
import json
import os
import sys

import numpy as np

from decisions_levels import PROFILES, k_weights

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 1e-3 / 12 / 5
CLIP = 3.0
MS = (5, 15, 40)
QS = (1.0, 3.0)
D_MAX = (0.02, 0.03, 0.04, None)
SHIFTS = (0.0, -0.1, -0.2, -0.3, -0.4)
BETAS = ("-0.03", "-0.05", "-0.10", "ramp-0.05")


def pmf(z, q):
    K = int(round(CLIP / q))
    zq = np.where(np.isfinite(z), np.round(np.clip(z, -CLIP, CLIP) / q), 0).astype(int)
    return np.bincount(zq + K, minlength=2 * K + 1) / max(len(zq), 1), K


def sum_cdf(p, K, M):
    """Exact CDF of the sum of M iid draws with pmf p on {-K..K}; returns (support values, cdf)."""
    d = np.array([1.0])
    for _ in range(M):
        d = np.convolve(d, p)
    vals = np.arange(-M * K, M * K + 1)
    return vals, np.cumsum(d)


def ladder_year(p, epochs=12):
    """P(at least one window of 3 consecutive epochs with >= 2 flags) for iid per-epoch flag probability p (exact DP
    over the last two flags)."""
    state = {(0, 0): 1.0}
    hit = 0.0
    for _ in range(epochs):
        new = {}
        for (a, b), pr in state.items():
            for f, pf in ((1, p), (0, 1 - p)):
                if a + b + f >= 2:
                    hit += pr * pf
                else:
                    new[(b, f)] = new.get((b, f), 0.0) + pr * pf
        state = new
    return hit


def main(tag="full"):
    meta = json.load(open(os.path.join(HERE, "results", f"consistency_base_{tag}.json")))
    arr = np.load(os.path.join(HERE, "results", f"consistency_base_z_{tag}.npz"))
    cells = meta["cells"]
    kw = k_weights(sorted({c["K_db"] for c in cells}), "floor")
    out = dict(alpha=ALPHA, k_weights=kw, results={})
    for q in QS:
        for dmax in D_MAX:
            # per cell: enabled honest/spoof z (sites whose reference sd(d) <= dmax); disabled sites disclose 0
            cell_pm, cell_w, enabled_share = [], [], []
            for i, c in enumerate(cells):
                sd_sites = np.array([s["sd_d_ref"] for s in c["sites"]])
                en_site = np.ones(sd_sites.size, bool) if dmax is None else sd_sites <= dmax
                site_of = arr[f"c{i}_honest_site"]
                en = en_site[site_of]
                enabled_share.append(float(en.mean()))
                pm = {}
                if en.any():
                    for shift in SHIFTS:
                        for dppb in meta["drift_ppb"]:
                            if shift and dppb:
                                continue
                            pm[("h", shift, dppb)] = pmf(arr[f"c{i}_honest_d{dppb}"][en] + shift, q)[0]
                    n_eval = arr[f"c{i}_beta-0.05"].size // max(sd_sites.size, 1)
                    en_eval = np.repeat(en_site, n_eval)
                    for b in BETAS:
                        z = arr[f"c{i}_beta{b}"]
                        pm[("s", b)] = pmf(z[en_eval[:z.size]], q)[0]
                cell_pm.append(pm)
                cell_w.append(kw[c["K_db"]] * 0.5 * float(en.mean()))     # weight of the cell's TESTED passes
            K = int(round(CLIP / q))
            res = dict(enabled_share_weighted=float(np.average(enabled_share, weights=cell_w)), M={})
            for M in MS:
                # population threshold from the pooled honest pmf (drift 0, no shift), K-population weights, all
                # speed classes equally (the profile weighting is applied to power below)
                use = [k for k, pm in enumerate(cell_pm) if pm]
                pool = np.average([cell_pm[k][("h", 0.0, 0.0)] for k in use], axis=0, weights=[cell_w[k] for k in use])
                vals, cdf = sum_cdf(pool, K, M)
                ok = np.nonzero(cdf <= ALPHA)[0]
                t = int(vals[ok[-1]]) if ok.size else None
                row = dict(threshold_sum=t, threshold_mean=None if t is None else t * q / M,
                           pooled_fa=float(cdf[ok[-1]]) if ok.size else 0.0, cells=[])
                for c, pm in zip(cells, cell_pm):
                    if not pm:
                        continue                                       # no tested site in this cell
                    def p_flag(p):
                        if t is None:
                            return 0.0
                        v, cd = sum_cdf(p, K, M)
                        return float(cd[np.searchsorted(v, t)])
                    cr = dict(v0=c["v0"], K_db=c["K_db"], spec=c["specular_db"] is not None)
                    for shift in SHIFTS:
                        fa = p_flag(pm[("h", shift, 0.0)])
                        cr[f"fa_shift{shift}"] = fa
                        cr[f"ladder_shift{shift}"] = ladder_year(fa)
                    for dppb in meta["drift_ppb"]:
                        fa = p_flag(pm[("h", 0.0, dppb)])
                        cr[f"fa_drift{dppb}"] = fa
                        cr[f"ladder_drift{dppb}"] = ladder_year(fa)
                    for b in BETAS:
                        cr[f"pd_{b}"] = p_flag(pm[("s", b)])
                    row["cells"].append(cr)
                # population-weighted power per speed profile
                # population power per speed profile: share of manipulated passes that are tested x P(flag | M tested
                # passes of the cell); untested passes give no detection by this rule
                tested = {(cr["v0"], cr["K_db"], cr["spec"]): cr for cr in row["cells"]}
                for prof, sp in PROFILES.items():
                    for b in BETAS:
                        num = den = 0.0
                        for c, cw, es in zip(cells, cell_w, enabled_share):
                            w_all = kw[c["K_db"]] * 0.5 * sp.get(c["v0"], 0.0)
                            den += w_all
                            cr = tested.get((c["v0"], c["K_db"], c["specular_db"] is not None))
                            if cr is not None:
                                num += w_all * es * cr[f"pd_{b}"]
                        if den > 0:
                            row[f"pd_{b}_{prof}"] = num / den
                for key in [f"fa_shift{s}" for s in SHIFTS] + [f"fa_drift{d}" for d in meta["drift_ppb"]]:
                    row[f"max_{key}"] = max(cr[key] for cr in row["cells"])
                    row[f"max_{key.replace('fa_', 'ladder_')}"] = max(cr[key.replace("fa_", "ladder_")] for cr in row["cells"])
                res["M"][M] = row
            out["results"][f"q{q}|dmax{dmax}"] = res
            r15 = res["M"][15]
            print(f"q={q} dmax={dmax}: enabled {res['enabled_share_weighted']:.2f} | M=15: t_mean {r15['threshold_mean']} "
                  f"pooled FA {r15['pooled_fa']:.1e}, worst cell FA {r15['max_fa_shift0.0']:.1e} (ladder/yr "
                  f"{r15['max_ladder_shift0.0']:.1e}) | drift 0.1/0.3/0.5 ppb worst ladder "
                  + "/".join(f"{r15[f'max_ladder_drift{d}']:.1e}" for d in meta["drift_ppb"][1:])
                  + " | shift -0.1/-0.2/-0.3/-0.4 worst ladder " + "/".join(f"{r15[f'max_ladder_shift{s}']:.1e}" for s in SHIFTS[1:])
                  + " | power -5% urban/sub/hwy " + "/".join(f"{r15.get(f'pd_-0.05_{p}', float('nan')):.2f}" for p in PROFILES)
                  + " -3% " + "/".join(f"{r15.get(f'pd_-0.03_{p}', float('nan')):.2f}" for p in PROFILES), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"consistency_base_analysis_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["full"]))
