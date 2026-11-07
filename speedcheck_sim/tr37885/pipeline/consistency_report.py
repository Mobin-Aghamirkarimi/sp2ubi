"""The speed-check simulation, C4 report (constants criteria.C4): from the P5 z files written by the
adapter (traffic standardisation 'trf'):
  * per site p- = P(z < -1.5) and p+ = P(z > 1.5) over its 100 evaluated honest passes, Clopper-Pearson 95 %;
  * the ladder criterion of the consistency simulation (z_q with q = 3; pooled threshold per M in {5; 15; 12 epochs; 2 of 3) per
    site with the plug-in pmf and with each site's upper bound of p- (p0 reduced accordingly): worst site <= 1e-3 per
    vehicle-year and share of sites above 1e-3 = 0 (equal site weights);
  * "no failing site among n" with the Clopper-Pearson upper bound for 0 of n (0.264 for 12, 0.459 for 6);
  * power: per-pass P(z_q = -1) and per-site epoch flag probability under beta = -5 % and -10 % (30 passes each);
  * the gate residual-RMS ratio under beta (median RMS of the beta passes / of the same honest passes), target 0.94-1.04;
  * consequence on failure: the threshold is recalibrated by the consistency-simulation procedure and power and claims are
    re-reported; the report states it, it does not tune anything.
Usage: python pipeline/consistency_report.py --tag primary"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
from consistency_analysis_base import ALPHA, ladder_year, pmf, sum_cdf  # noqa: E402
from pipeline.provenance import check_tag, provenance  # noqa: E402

C = json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study1_constants.json"))))
OUT = os.environ.get("SPEEDCHECK_ADAPTER_OUT", os.path.join(ER, "..", "results", "adapter"))
RESULTS = os.environ.get("SPEEDCHECK_RESULTS", os.path.join(ER, "..", "results"))
Q, MS, ZT = 3.0, (5, 15, 40), 1.5


def cp(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return [float(lo), float(hi)]


def threshold(pool, K, M):
    vals, cdf = sum_cdf(pool, K, M)
    ok = np.nonzero(cdf <= ALPHA)[0]
    return int(vals[ok[-1]]) if ok.size else None


def p_flag(p, K, M, t):
    if t is None:
        return 0.0
    v, cd = sum_cdf(p, K, M)
    return float(cd[np.searchsorted(v, t)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    check_tag(a.tag, stage=3)
    root = os.path.join(OUT, "e3", "results")
    meta = json.load(open(os.path.join(root, f"consistency_adapter_{a.tag}.json")))
    arr = np.load(os.path.join(root, f"consistency_z_adapter_{a.tag}.npz"))
    ne, nsp = meta["eval"], meta["spoof"]
    units = []
    for i, c in enumerate(meta["cells"]):
        h = arr[f"c{i}_honest_trf"]
        for s, site in enumerate(c["sites"]):
            ev = h[s * ne:(s + 1) * ne]
            assert ev.size == ne, "E3 block alignment"
            # the consistency simulation convention (pmf): a pass without a usable z (excluded, gated, failed fit) is kept and maps to
            # z_q = 0; the shares are over all ne passes; the finite-only shares are co-reported
            fin = ev[np.isfinite(ev)]
            km, kp = int(np.sum(fin < -ZT)), int(np.sum(fin > ZT))
            p, K = pmf(ev, Q)
            p_ub = p.copy()
            p_ub[0] = cp(km, ev.size)[1]
            p_ub[1] = max(0.0, 1.0 - p_ub[0] - p_ub[2])
            sp = {b: arr[f"c{i}_beta{b}_trf"][s * nsp:(s + 1) * nsp] for b in ("-0.05", "-0.10")}
            units.append(dict(v0=c["v0"], site=site.get("site", s), n=int(ev.size), n_finite=int(fin.size),
                              p_minus=km / ev.size, p_plus=kp / ev.size,
                              p_minus_ci=cp(km, ev.size), p_plus_ci=cp(kp, ev.size),
                              p_minus_finite_only=km / max(fin.size, 1), p_plus_finite_only=kp / max(fin.size, 1),
                              pmf=p, pmf_ub=p_ub, K=K,
                              beta_pmf={b: pmf(v, Q)[0] for b, v in sp.items()},
                              beta_pass_detect={b: float(np.mean(np.nan_to_num(v, nan=0.0) < -ZT)) for b, v in sp.items()},
                              gate_ratio_beta=site.get("gate_ratio_beta"), excluded=site.get("excluded")))
    out = dict(tag=a.tag, provenance=provenance(a.tag), units=[{k: v for k, v in u.items() if k not in ("pmf", "pmf_ub", "beta_pmf")}
                                                               for u in units], by_M={})
    K = units[0]["K"] if units else 1
    pool = np.mean([u["pmf"] for u in units], axis=0) if units else None
    for M in MS:
        t = threshold(pool, K, M)
        lad = np.array([ladder_year(p_flag(u["pmf"], K, M, t)) for u in units])
        lad_ub = np.array([ladder_year(p_flag(u["pmf_ub"], K, M, t)) for u in units])
        power = {b: float(np.mean([p_flag(u["beta_pmf"][b], K, M, t) for u in units])) for b in ("-0.05", "-0.10")}
        power_by_cell = {f"{v:g}": {b: float(np.mean([p_flag(u["beta_pmf"][b], K, M, t) for u in units if u["v0"] == v]))
                                    for b in ("-0.05", "-0.10")} for v in sorted({u["v0"] for u in units})}
        lim = C["criteria"]["C4"]["worst_site_per_year"]
        out["by_M"][str(M)] = dict(threshold=t, worst_site_ladder=float(lad.max()), share_sites_above=float(np.mean(lad > lim)),
                                  worst_site_ladder_ub=float(lad_ub.max()), share_sites_above_ub=float(np.mean(lad_ub > lim)),
                                  pass_plugin=bool(lad.max() <= lim), pass_upper_bound=bool(lad_ub.max() <= lim), power=power,
                                  power_by_cell=power_by_cell)
    n_sites = len(units)
    fails = int(sum(out["by_M"][str(M)]["share_sites_above"] > 0 for M in MS))
    out["no_failing_site_wording"] = (f"no failing site among {n_sites} (Clopper-Pearson upper bound {cp(0, n_sites)[1]:.3f}"
                                      f"; never '0 % of sites')" if fails == 0 and n_sites else None)
    gr = [v for u in units if u["gate_ratio_beta"] for v in u["gate_ratio_beta"].values()]
    out["gate_ratio_beta"] = dict(min=float(min(gr)), max=float(max(gr)), within_target=bool(min(gr) >= 0.94 and max(gr) <= 1.04)) if gr else None
    out["on_failure"] = "threshold recalibrated by the consistency-simulation procedure; power and claims re-reported (pre-specified)"
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, f"c4_{a.tag}.json"), "w"), indent=1, default=float)
    print(json.dumps(out["by_M"], indent=1, default=float))


if __name__ == "__main__":
    main()
