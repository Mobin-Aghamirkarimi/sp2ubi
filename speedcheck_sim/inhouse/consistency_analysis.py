"""Consistency analysis: per-SITE consistency rule with survey-only vs traffic standardisation, exact rates.

Unit = one site (a vehicle whose M passes all come from that site: the worst case for a frequent user). z_q =
round(clip(z, -3, 3) / q); the insurer's threshold t_M is the largest integer with P_pop(sum of M <= t_M) <= ALPHA under
the pooled honest population of evaluated passes (K-population x facade weights). Per-site epoch false alarm and the
2-of-3 ladder per year are exact given the site's 100 evaluated honest passes; a site-common shift of z by -0.1 ... -0.4
sd (persistent calibration or offset error) is applied as a margin case. Power: all M passes manipulated, weighted
by the K population and the speed profiles. A small-sample check: the per-site rates are recomputed after removing
the single most negative evaluated value (sensitivity of the tail estimate to one pass).
Thresholds: "pooled" (one population threshold per M) and "level" (the insurer conditions on the disclosed sigma
level; qualification sigma_hat <= 6 % -- assigned by quantile within the cell; levels with < 12 sites are merged
with a neighbour; one threshold per level group and M).
Usage: python inhouse/consistency_analysis.py
"""
import json
import os

import numpy as np

from consistency_analysis_base import ALPHA, ladder_year, pmf, sum_cdf
from decisions_levels import PROFILES, k_weights

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = 3.0
BETAS = ("-0.05", "-0.10")
SHIFTS = (0.0, -0.1, -0.2, -0.3, -0.4)


def level_map():
    """(v, K, facade) -> sorted list of disclosed levels of the in-house site-accuracy sites of that type (iso scattering)."""
    import decisions_levels as E
    from levels_qualification import requalify
    A, qual = requalify(E.load_sites(["extra"]), 0.06, 8)
    out = {}
    for i, k in enumerate(A["keys"]):
        K, sc, sp = k[0]
        if sc != "iso":
            continue
        for j, v in enumerate(A["speeds"]):
            if qual[i, j]:
                out.setdefault((v, K, sp is not None), []).append(int(A["lvl"][i, j]))
    return {k: sorted(v) for k, v in out.items()}


def level_groups(levels, min_units=12):
    """Merge sparse levels with a neighbour: returns {level: group id}."""
    counts = {l: levels.count(l) for l in sorted(set(levels))}
    groups, cur, n = {}, 0, 0
    for l in sorted(counts):
        groups[l] = cur
        n += counts[l]
        if n >= min_units:
            cur, n = cur + 1, 0
    if n and cur > 0:                                        # last group too small: merge into the previous one
        for l in groups:
            if groups[l] == cur:
                groups[l] = cur - 1
    return groups


def main(tag="full"):
    meta = json.load(open(os.path.join(HERE, "results", f"consistency_{tag}.json")))
    arr = np.load(os.path.join(HERE, "results", f"consistency_z_{tag}.npz"))
    cells = meta["cells"]
    kw = k_weights(sorted({c["K_db"] for c in cells}), "floor")
    ne, nsp = meta["eval"], meta["spoof"]
    lmap = level_map()
    out = {}
    for kind, thr_mode in (("ref", "pooled"), ("trf", "pooled"), ("trf", "level")):
        for q in (1.0, 3.0):
            K = int(round(CLIP / q))
            units = []
            for i, c in enumerate(cells):
                ns = len(c["sites"])
                h = arr[f"c{i}_honest_{kind}"]
                for s in range(ns):
                    ev = h[s * ne:(s + 1) * ne]
                    fin = ev[np.isfinite(ev)]
                    trimmed = np.delete(fin, np.argmin(fin)) if fin.size > 1 else fin
                    lv_list = lmap[(c["v0"], c["K_db"], c["specular_db"] is not None)]
                    lvl = lv_list[min(len(lv_list) - 1, int((s + 0.5) / ns * len(lv_list)))]
                    units.append(dict(v0=c["v0"], K=c["K_db"], spec=c["specular_db"] is not None, site=s, lvl=lvl,
                                      w=kw[c["K_db"]] * 0.5 / ns,
                                      h={sh: pmf(ev + sh, q)[0] for sh in SHIFTS}, h_trim=pmf(trimmed, q)[0],
                                      s={b: pmf(arr[f"c{i}_beta{b}_{kind}"][s * nsp:(s + 1) * nsp], q)[0] for b in BETAS}))
            w = np.array([u["w"] for u in units])
            grp = level_groups([u["lvl"] for u in units]) if thr_mode == "level" else {u["lvl"]: 0 for u in units}
            for u in units:
                u["g"] = grp[u["lvl"]]
            res = {}
            for M in (5, 15, 40):
                thr = {}
                for g in sorted(set(u["g"] for u in units)):
                    sel = [k for k, u in enumerate(units) if u["g"] == g]
                    pool = np.average([units[k]["h"][0.0] for k in sel], axis=0, weights=w[sel])
                    vals, cdf = sum_cdf(pool, K, M)
                    ok = np.nonzero(cdf <= ALPHA)[0]
                    thr[g] = int(vals[ok[-1]]) if ok.size else None
                cur = {"t": None}

                def p_flag(p):
                    t = cur["t"]
                    if t is None:
                        return 0.0
                    v, cd = sum_cdf(p, K, M)
                    return float(cd[np.searchsorted(v, t)])

                def per_unit(f):
                    vals_ = []
                    for u in units:
                        cur["t"] = thr[u["g"]]
                        vals_.append(f(u))
                    return np.array(vals_)
                row = dict(thresholds_mean={str(g): (None if t is None else t * q / M) for g, t in thr.items()},
                           level_groups={str(l): g for l, g in grp.items()})
                for sh in SHIFTS:
                    lad = per_unit(lambda u: ladder_year(p_flag(u["h"][sh])))
                    row[f"shift{sh}_share_sites_ladder_gt_1e3"] = float(np.sum(w * (lad > 1e-3)) / w.sum())
                    row[f"shift{sh}_weighted_mean_ladder"] = float(np.sum(w * lad) / w.sum())
                    row[f"shift{sh}_worst_site_ladder"] = float(lad.max())
                lad_t = per_unit(lambda u: ladder_year(p_flag(u["h_trim"])))
                row["trim1_share_sites_ladder_gt_1e3"] = float(np.sum(w * (lad_t > 1e-3)) / w.sum())
                fa0 = per_unit(lambda u: p_flag(u["h"][0.0]))
                worst = int(np.argmax(fa0))
                row["worst_site"] = [units[worst][k] for k in ("v0", "K", "spec", "site")]
                row["worst_site_fa"] = float(fa0[worst])
                for b in BETAS:
                    pdv = per_unit(lambda u: p_flag(u["s"][b]))
                    for prof, sp in PROFILES.items():
                        wp = np.array([u["w"] * sp.get(u["v0"], 0.0) for u in units])
                        row[f"pd_{b}_{prof}"] = float(np.sum(wp * pdv) / wp.sum())
                    row[f"pd_{b}_K>=10"] = float(np.average(pdv[[u["K"] >= 10 for u in units]]))
                res[M] = row
            out[f"{kind}|{thr_mode}|q{q}"] = res
            r = res[15]
            print(f"{kind} {thr_mode} q={q}: M=15 thresholds {r['thresholds_mean']} | sites ladder>1e-3: "
                  + "/".join(f"{r[f'shift{sh}_share_sites_ladder_gt_1e3']:.3f}" for sh in SHIFTS)
                  + f" (shift 0/-0.1/-0.2/-0.3/-0.4); after dropping each site's most negative pass {r['trim1_share_sites_ladder_gt_1e3']:.3f}"
                  + f" | weighted mean ladder {r['shift0.0_weighted_mean_ladder']:.1e}, worst site ladder {r['shift0.0_worst_site_ladder']:.1e} at {r['worst_site']}"
                  + " | power -5 % " + "/".join(f"{r[f'pd_-0.05_{p}']:.2f}" for p in PROFILES)
                  + f" (K>=10 {r['pd_-0.05_K>=10']:.2f})"
                  + " -10 % " + "/".join(f"{r[f'pd_-0.10_{p}']:.2f}" for p in PROFILES), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"consistency_analysis_{tag}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
