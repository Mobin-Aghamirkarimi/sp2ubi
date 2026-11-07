"""Stage 3: pre-specified criteria from the stage-2 records (constants `criteria`, `conventions`, `window_rule`,
`e1_reference`). Implemented here: C1 (paired accuracy ratio, three-way verdict, Holm m = 9, worst variant governs), the
headline rule incl. the comparison with the in-house site-accuracy simulation and the governing SNR mapping / PRB per
stratum, C2 (i)
TOST with Holm, C2 (ii) excess between-site variance (centred) with the τ̂² estimator, C2 (iii) transfer to type 3
(P6) and the per-lane description, C3 (qualified share and gate false-discard rate with CIs), C5 (SUMO vs quadratic,
select statistic, test passes, Holm, consequence list), C7 (P6 gate, monitor margin, false raise and false stop),
the window-rule yield loss, the stop subset (reporting only), the variance decomposition, per-beacon effective number
of beacons and FDOA error against SNR and K, realised K, P3 blocked shares, non-finite shares. C4 runs in
pipeline/consistency_report.py.

Definitions:
  * inclusion (identical for every consumer, RSU-observable): a pass enters only if its window [-0.05 L, 1.05 L] is
    traversed within one 40 s burst (window rule) and both statistics are finite (the pipeline drops a pass otherwise);
    stop_between is never an exclusion;
  * per site and speed class the statistic b̂ is "select": matched or profile, whichever has the smaller variance over
    the site's 30 reference passes (decided per arm); calibrated residual e = b̂ - β̂_s, β̂_s = mean of the references;
  * qualification: sd of the references <= 6 %; gate: discard if the fit-residual RMS > 1.3 x the median RMS of the
    references; "qualified_gated" = test passes of qualified sites kept by the gate;
  * σ (C1 and headline only) on e clipped to ±25 %; Δ, C2, C5, C7 unclipped (constants C1.clip_scope);
  * CIs: site-clustered percentile bootstrap, B = 10,000, one generator per key (provenance.rng_for), the same
    site draws for 'all' and 'qualified_gated'; 95 % CI = q2.5 / q97.5, upper bound = q97.5;
  * a "confirmed" / "pass" claim needs a Holm-adjusted one-sided bootstrap p <= 0.025 per component, p = (1 + #)/(B + 1)
    (constants conventions); each reported quantity is the worst over the 2 SNR mappings x 3 PRB values (and the two
    passes sets where both exist), with the variant that sets it recorded.
Under constants evaluation_v4 the two-sided C1 verdict is not used (C1 quantities stay as the descriptive model-diversity check, gen_off vs in-house); the run family is the 6 cells
of U, H, N on primary sites 0-9 (optional extension 10-19); new: the one-sided in-house optimism check (gen_on vs
in-house, unclipped 1.4826 MAD ratio, margin 1.5, widened site bootstrap, Holm m = 6, verdicts with reason codes) and the
scope rule (k/n qualified primary sites per cell from the reference passes, covered / partly / not covered, type rule);
with constants lever_d.adopted the Huber (d) statistics are the statistics and the LS ones the ablation (both reported).
C5 / C7 / C2 (iii) / P4 / P5 / P6 are not run.
Usage: python pipeline/stage3_criteria.py --tag primary [--families U H N]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline.stage1_channels import C  # noqa: E402
from pipeline.provenance import check_tag, provenance, rng_for, sha256_file  # noqa: E402

PC = C["pipeline_constants_from_code"]
N_REF, SIG_Q, GATE, WIN = PC["calibration_passes"], PC["site_qualification_sigma"], PC["gate_ratio"], PC["B_WINDOW"]
MON_WIN, MON_RAISE, MON_STOP = PC["monitor_window"], PC["monitor_raise"], PC["monitor_stop"]
R_MAX, D_MAX = C["criteria"]["C1"]["confirmed_upper_R"], C["criteria"]["C1"]["delta_band"]
ALPHA = C["conventions"]["alpha"]
MIN_REF = C["conventions"]["min_ref"]
HOLM_FAMILY = [("P1", 10.0), ("P1", 15.0), ("P2", 20.0), ("P2", 30.0), ("P2", 38.0), ("P3", 15.0)]   # m = 6
EV4 = C["evaluation_v4"]
OC = EV4["optimism_check"]
SCOPE = EV4["scope_rule"]
LEVER_D = bool(EV4["lever_d"]["adopted"])
C5_FAMILY = [("P1", 10.0), ("P1", 15.0), ("P2", 20.0), ("P2", 30.0), ("P2", 38.0)]
C7_FAMILY = [("P1", 15.0), ("P2", 30.0), ("P4", 20.0)]
STRATA = {"urban": ("P1", (10.0, 15.0)), "highway": ("P2", (20.0, 30.0, 38.0))}
MAPPINGS = ("e1", "lb")
PRBS = tuple(C["beacon"]["prb"]["swept"])
COMBOS = [(m, p) for m in MAPPINGS for p in PRBS]
B = int(os.environ.get("SPEEDCHECK_BOOT", C["run"]["boot"]))       # SPEEDCHECK_BOOT: synthetic smoke only
RUNS = os.environ.get("SPEEDCHECK_RUNS", os.path.join(ER, "runs"))
RESULTS = os.environ.get("SPEEDCHECK_RESULTS", os.path.join(ER, "..", "results"))
NOT_RUN_0 = "not run (no qualified site)"
ORDER = {"not run": -1, NOT_RUN_0: -1, "not confirmed": 0, "inconclusive": 1, "confirmed": 2}
PRIMARY_FAMILIES = list(EV4["p2_order"])                     # H, U, N
VERDICT_ORDER = {"not run": -1, "contradicted": 0, "undetermined": 1, "not contradicted": 2}
COMPLETED = {}                                              # family -> completed sites under --incomplete-run
# Site types and the P3 traffic condition: P3 is not a site type; its survey
# crossings are free-road LOS and NLOSv blockage of head A affects the traffic crossings only. The abstract, the footnote
# of the comparison table and the conclusion name site types from U and H only (site_type True).
TYPES = {"urban LOS": dict(cells=["P1-10", "P1-15"], site_type=True),
         "highway LOS": dict(cells=["P2-20", "P2-30", "P2-38"], site_type=True),
         "urban, blocked crossings (NLOSv)": dict(cells=["P3-15"], site_type=False)}


# ------------------------------------------------------------------------------------------------------------ loading
def assert_complete(families, tag, n_sites=None):
    """Primary: every stage-2 file of every family exists for sites 0-9 and, if any extension site exists, for all of
    10-19."""
    for fam in families:
        have = lambda k: os.path.exists(os.path.join(RUNS, f"g1_{fam}", f"eval_{tag}", f"site_{k:03d}.json"))
        ext = EV4.get("extension")                      # None: no extension
        top = n_sites or (ext["sites"][1] if ext and any(have(k) for k in range(*ext["sites"])) else EV4["primary_sites"][1])
        miss = [k for k in range(top) if not have(k)]
        if miss:
            raise RuntimeError(f"{fam} ({tag}): stage-2 files missing for sites {miss}")


def use_statistic(recs, stat):
    """'ls': the stored plain LS statistics; 'd': the Huber fit (d) (prof_d / match_d / d_d) in the fields every criterion reads;
    mse stays the LS residual (gate; monitor)."""
    if stat == "ls":
        return recs
    out = []
    for r in recs:
        q = dict(r)
        q["match"], q["prof"] = r.get("match_d", np.nan), r.get("prof_d", np.nan)
        q["finite"] = bool(np.isfinite(q["match"]) and np.isfinite(q["prof"]))
        if "d_d" in r:
            q["d"] = r["d_d"]
        out.append(q)
    return out


def stage2_consistent(provs, tag):
    """Stage-2 records of primary and sens_ tags: one set of stage-1/2 sources across all records of the tag."""
    seen = {tuple((p.get("stage_sha256") or {}).get(j) for j in ("1", "2")) for p in provs}
    if len(seen) > 1:
        raise RuntimeError(f"{tag}: the stage-2 records do not carry one set of stage-1/2 sources")


def load(families, tag, stat=None):
    recs, provs = [], []
    for fam in families:
        for f in sorted(glob.glob(os.path.join(RUNS, f"g1_{fam}", f"eval_{tag}", "site_*.json"))):
            d = json.load(open(f))
            provs.append(d.get("provenance", {}))
            for r in d["recs"]:
                r["family"] = fam
            recs += d["recs"]
    if tag == "primary" or tag.startswith("sens_"):
        stage2_consistent(provs, tag)
    return use_statistic(recs, stat or ("d" if LEVER_D else "ls")), []


def included(r):
    """The RSU-observable inclusion rule (window rule, finite statistics)."""
    return r.get("window_ok", True) and r.get("finite", np.isfinite(r["match"]) and np.isfinite(r["prof"]))


def ref_qualification(ref, min_ref=None):
    """The one site-qualification rule (criteria and the P5 selection): included reference
    passes (window rule, finite), at least MIN_REF of them, select statistic = the smaller variance of matched / profile,
    qualified if its sd <= 6 %. Returns dict(n_ref, key, sd_ref, qualified)."""
    min_ref = MIN_REF if min_ref is None else min_ref
    ref = [r for r in ref if included(r)]
    if len(ref) < min_ref:
        return dict(n_ref=len(ref), key=None, sd_ref=np.nan, qualified=False)
    em = np.array([r["match"] for r in ref], float); ep = np.array([r["prof"] for r in ref], float)
    key = "match" if np.var(em) <= np.var(ep) else "prof"
    sd = float(np.array([r[key] for r in ref], float).std(ddof=1))
    return dict(n_ref=len(ref), key=key, sd_ref=sd, qualified=bool(sd <= SIG_Q))


def select(recs, cell, v0, arm, mapping, prb, kin="sumo", beta=0.0):
    return [r for r in recs if r["cell"] == cell and r["v0"] == v0 and r["arm"] == arm and r["mapping"] == mapping
            and r["prb"] == prb and r["kin"] == kin and r.get("beta", 0.0) == beta]


def site_table(recs, cell, v0, arm, mapping, prb, kin="sumo", ref_cell=None, min_ref=None):
    """Per site: calibrated residuals of the included test passes (select statistic), reference sd, gate mask, RMS,
    stop flags, lane, pair index; plus the inclusion counts. ref_cell: take the references from another cell (P6)."""
    out, counts = {}, dict(test=0, window_excluded=0, nonfinite=0)
    test_all = select(recs, cell, v0, arm, mapping, prb, kin)
    ref_all = select(recs, ref_cell or cell, v0, arm, mapping, prb, "sumo")
    for s in sorted({r["site"] for r in test_all}):
        ref = sorted([r for r in ref_all if r["site"] == s and r["role"] == "ref" and included(r)], key=lambda r: r["track"])
        tst0 = sorted([r for r in test_all if r["site"] == s and r["role"] == "test"], key=lambda r: r["track"])
        counts["test"] += len(tst0)
        counts["window_excluded"] += sum(not r.get("window_ok", True) for r in tst0)
        counts["nonfinite"] += sum(r.get("window_ok", True) and not included(r) for r in tst0)
        tst = [r for r in tst0 if included(r)]
        q = ref_qualification(ref, min_ref)
        if q["key"] is None or not tst:                         # all included references (30 planned), at least 20
            continue
        key = q["key"]
        b_ref = np.array([r[key] for r in ref], float)
        rms_ref = np.sqrt(np.array([r["mse"] for r in ref], float))
        out[s] = dict(e=np.array([r[key] for r in tst], float) - b_ref.mean(), sd_ref=float(b_ref.std(ddof=1)),
                      b_ref=float(b_ref.mean()), n_test_all=len(tst0),
                      blocked=np.array([r.get("blocked_share", 0.0) > 0 for r in tst]),
                      blocked_major=np.array([r.get("blocked_share", 0.0) >= 0.5 for r in tst]),
                      n_ref=len(ref), key=key, rms=np.sqrt(np.array([r["mse"] for r in tst], float)),
                      rms_ref_med=float(np.median(rms_ref)),
                      keep_gate=np.sqrt(np.array([r["mse"] for r in tst], float)) <= GATE * np.median(rms_ref),
                      stop=np.array([bool(r.get("stop_between")) for r in tst]), j=np.array([r["j"] for r in tst]),
                      lane=np.array([int(round(-r.get("y_mean", 0.0) / lane_width(r))) for r in tst]))
    return out, counts


def lane_width(r):
    """Lane width of the record's scenario (constants sumo_implementation): highway 4.0 m, else 3.5 m."""
    return C["sumo_implementation"]["highway" if r.get("family") in ("H", "5H") else "urban"]["lane_width_m"]


def pooled(T, sites, qualified_gated=False, mask=None):
    es = []
    for s in sites:
        t = T[s]
        sel = np.ones(t["e"].size, bool) if mask is None else t[mask]
        if qualified_gated:
            if t["sd_ref"] > SIG_Q:
                continue
            sel = sel & t["keep_gate"]
        es.append(t["e"][sel])
    return np.concatenate(es) if es else np.array([])


def n_contributing(T, sites, qualified_gated=False):
    """Number of sites with >= 1 pass in pooled(T, sites, qualified_gated): the n of the sigma CI's c(n)."""
    return int(sum(pooled(T, [s], qualified_gated).size > 0 for s in sites))


def _div(a, b):
    """a / b in floating point: 0 / 0 = nan, x / 0 = +-inf, never an exception (degenerate cells)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.float64(a) / np.float64(b))


def sigma(e):
    return float(np.std(np.clip(e, -WIN, WIN))) if e.size else np.nan


def sigma_e2map(e):
    """σ with the decision simulation's out-of-window mapping (|e| > 25 % -> -25 %), co-reported."""
    return float(np.std(np.where(np.abs(e) > WIN, -WIN, e))) if e.size else np.nan


def scale_estimates(e):
    if not e.size:
        return (np.nan, np.nan)
    return float(1.4826 * np.median(np.abs(e - np.median(e)))), float((np.quantile(e, 0.84) - np.quantile(e, 0.16)) / 2)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if x.size else [np.nan, np.nan]


def nonfinite_share(x):
    """Share of non-finite bootstrap draws, reported next to every CI (constants conventions.nonfinite_bootstrap)."""
    x = np.asarray(x, float)
    return float(np.mean(~np.isfinite(x))) if x.size else 1.0


def pval(boot, cond):
    """One-sided bootstrap p = (1 + #exceed) / (B + 1); a non-finite draw counts as an exceedance (conservative)."""
    b = np.asarray(boot, float)
    with np.errstate(invalid="ignore"):
        exceed = np.asarray(cond(b), bool) | ~np.isfinite(b)
    return float((1 + np.sum(exceed)) / (b.size + 1))


def q_abs(e, q):
    return float(np.quantile(np.abs(e), q)) if e.size else np.nan


def draws(key, n_sites):
    rng = rng_for(*key)
    return [rng.integers(0, n_sites, n_sites) for _ in range(B)]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(p.size), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (p.size - k) * p[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------------------------------------------------------ C1
def c1_cell(recs, cell, v0, mapping, prb, rho=1.0):
    """rho: S-place factor on every generator sigma (constants sensitivities.S_place.rule); 1 = primary reading."""
    G, _ = site_table(recs, cell, v0, "gen_off", mapping, prb)
    P, _ = site_table(recs, cell, v0, "inhouse", mapping, prb)
    sites = sorted(set(G) & set(P))
    if len(sites) < 2:
        return None
    res = dict(cell=cell, v0=v0, mapping=mapping, prb=prb, n_sites=len(sites), s_place_factor=rho,
               nonfinite=dict(gen_off=site_table(recs, cell, v0, "gen_off", mapping, prb)[1],
                              inhouse=site_table(recs, cell, v0, "inhouse", mapping, prb)[1]))
    D = draws(("C1", cell, v0, mapping, prb), len(sites))
    mean = lambda e: float(np.mean(e)) if e.size else np.nan
    oow = lambda e: float(np.mean(np.abs(e) > WIN)) if e.size else np.nan
    for qg in (False, True):
        eg, ep = pooled(G, sites, qg), pooled(P, sites, qg)
        key = "qualified_gated" if qg else "all"
        if eg.size == 0 or ep.size == 0:                    # no qualified site in one arm
            res[key] = dict(empty=True, n_gen=int(eg.size), n_inhouse=int(ep.size), sigma_gen=sigma(eg),
                            sigma_inhouse=sigma(ep), R=np.nan, R_ci=[np.nan, np.nan], delta=np.nan,
                            delta_ci=[np.nan, np.nan], p_R=1.0, p_delta=1.0)
            continue
        R = rho * _div(sigma(eg), sigma(ep))
        Dl = mean(eg) - mean(ep)
        Rb, Db, Ob = [], [], []
        with np.errstate(invalid="ignore", divide="ignore"):
            for idx in D:
                ss = [sites[i] for i in idx]
                g, p = pooled(G, ss, qg), pooled(P, ss, qg)
                Rb.append(rho * _div(sigma(g), sigma(p))); Db.append(mean(g) - mean(p))
                Ob.append(oow(g) - oow(p))
        res[key] = dict(
            empty=False, sigma_gen=rho * sigma(eg), sigma_inhouse=sigma(ep), R=R, R_ci=ci(Rb), delta=Dl, delta_ci=ci(Db),
            R_ci_nonfinite_share=nonfinite_share(Rb), delta_ci_nonfinite_share=nonfinite_share(Db),
            p_R=pval(Rb, lambda b: b >= R_MAX), p_delta=max(pval(Db, lambda b: b >= D_MAX), pval(Db, lambda b: b <= -D_MAX)),
            oow_gen=oow(eg), oow_inhouse=oow(ep), oow_diff_ci=ci(Ob),
            scales_gen=scale_estimates(eg), scales_inhouse=scale_estimates(ep), sigma_e2map_gen=sigma_e2map(eg),
            q90_q99_gen=[q_abs(eg, 0.9), q_abs(eg, 0.99)], q90_q99_inhouse=[q_abs(ep, 0.9), q_abs(ep, 0.99)])
    return res


def c1_verdicts(c1):
    """Holm with m = 9 per (mapping, prb, set); missing cells p = 1 ('not run'); governing = worst of 12 per cell."""
    by = {(r["cell"], r["v0"], r["mapping"], r["prb"]): r for r in c1}
    for mp, prb in COMBOS:
        for key in ("all", "qualified_gated"):
            p = [max(by[(c, v, mp, prb)][key]["p_R"], by[(c, v, mp, prb)][key]["p_delta"]) if (c, v, mp, prb) in by else 1.0
                 for c, v in HOLM_FAMILY]
            adj = holm(p)
            for (c, v), a in zip(HOLM_FAMILY, adj):
                if (c, v, mp, prb) not in by:
                    continue
                x = by[(c, v, mp, prb)][key]
                if not (np.isfinite(x["R"]) and np.isfinite(x["delta"])):
                    verdict = NOT_RUN_0
                elif x["R"] > R_MAX or abs(x["delta"]) > D_MAX:
                    verdict = "not confirmed"
                elif a <= ALPHA:
                    verdict = "confirmed"
                else:
                    verdict = "inconclusive"
                x["p_holm"], x["verdict"] = float(a), verdict
    gov = {}
    for c, v in HOLM_FAMILY:
        best = None
        for mp, prb in COMBOS:
            r = by.get((c, v, mp, prb))
            for key in ("all", "qualified_gated"):
                vd = r[key]["verdict"] if r else "not run"
                if best is None or ORDER[vd] < ORDER[best["verdict"]]:
                    best = dict(verdict=vd, mapping=mp, prb=prb, set=key)
        gov[f"{c}-{v:g}"] = best
    return gov


# ------------------------------------------------------------------------------------------------------------ optimism check
def c_of_n(n):
    """K7 small-sample widening c(n) = t_0.975,n-1 / z_0.975 * sqrt(n / (n - 1))."""
    from scipy import stats as st
    return float(st.t.ppf(0.975, n - 1) / st.norm.ppf(0.975) * np.sqrt(n / (n - 1))) if n > 1 else np.inf


def mad(e):
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else np.nan


def optimism_draws(recs, cell, v0, mapping, prb, rho=1.0):
    """R, its site-cluster bootstrap draws, n (qualified generator sites with kept test passes) and the pooled residuals."""
    G, _ = site_table(recs, cell, v0, "gen_on", mapping, prb)
    P, _ = site_table(recs, cell, v0, "inhouse", mapping, prb)
    sites = sorted(set(G) & set(P))
    n = int(sum(G[s]["sd_ref"] <= SIG_Q and G[s]["keep_gate"].any() for s in sites))
    eg, ep = pooled(G, sites, True), pooled(P, sites, True)
    if eg.size == 0 or ep.size == 0:
        return np.nan, np.array([]), n, eg, ep
    R = rho * _div(mad(eg), mad(ep))
    Rb = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for idx in draws(("OPT", cell, v0, mapping, prb), len(sites)):
            ss = [sites[i] for i in idx]
            Rb.append(rho * _div(mad(pooled(G, ss, True)), mad(pooled(P, ss, True))))
    return R, np.asarray(Rb, float), n, eg, ep


def widen(R, Rb, c):
    """Widened draws on the log scale: log R_b' = log R + c (log R_b - log R)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.log(R) + c * (np.log(Rb) - np.log(R)))


def optimism_cell(recs, cell, v0, mapping, prb, c_k7=1.0, rho=1.0):
    """One-sided optimism check (constants evaluation_v4.optimism_check): R = MAD(e_gen_on) / MAD(e_inhouse) on the
    unclipped calibrated residuals of the system as specified (qualified + gated test passes), same sites; widened
    site-cluster bootstrap; p = P(R_b' >= margin). rho: S-place factor on the generator scale. n = qualified generator
    sites with test passes; n <= 3 -> not run."""
    R, Rb, n, eg, ep = optimism_draws(recs, cell, v0, mapping, prb, rho)
    G, _ = site_table(recs, cell, v0, "gen_on", mapping, prb)
    P, _ = site_table(recs, cell, v0, "inhouse", mapping, prb)
    base = dict(cell=cell, v0=v0, mapping=mapping, prb=prb, n_sites=int(n), s_place_factor=rho,
                n_qual_g1=int(sum(G[s]["sd_ref"] <= SIG_Q for s in G)),       # each arm's own qualification
                n_qual_inhouse=int(sum(P[s]["sd_ref"] <= SIG_Q for s in P)))
    fam = {"P1": "U", "P2": "H", "P3": "N"}.get(cell)
    if fam in EV4.get("p2_dqual", []):                  # D-qual family: no test data by design
        return dict(base, R=np.nan, p=1.0, c=None, not_run=True, reason="not run (qualification arm)")
    if n < OC["min_sites"] or eg.size == 0 or ep.size == 0:
        return dict(base, R=np.nan, p=1.0, c=None, not_run=True, reason=f"not run (n = {n} qualified sites with test passes)")
    c = max(c_of_n(n), c_k7)
    Rw = widen(R, Rb, c)
    sd_ratio = rho * _div(np.std(eg), np.std(ep))        # rho: S-place factor on the generator scale, as for R
    return dict(base, R=float(R), R_ci_widened=ci(Rw), c=float(c), c_n=c_of_n(n), c_k7=float(c_k7),
                p=pval(Rw, lambda b: b >= OC["margin"]), not_run=False,
                descriptive=dict(sd_ratio=sd_ratio, q99_ratio=rho * _div(q_abs(eg, 0.99), q_abs(ep, 0.99)),
                                 clip_share_gen=float(np.mean(np.abs(eg) > WIN)), clip_share_inhouse=float(np.mean(np.abs(ep) > WIN)),
                                 n_gen=int(eg.size), n_inhouse=int(ep.size)))


def optimism_verdicts(rows):
    """Holm over the 6 cells per variant (missing / not run: p = 1); verdict per variant; the cell's verdict is the worst
    variant (not run < contradicted < undetermined < not contradicted), with a reason code."""
    by = {(r["cell"], r["v0"], r["mapping"], r["prb"]): r for r in rows}
    for mp, prb in COMBOS:
        adj = holm([by[(c, v, mp, prb)]["p"] if (c, v, mp, prb) in by else 1.0 for c, v in HOLM_FAMILY])
        for (c, v), a in zip(HOLM_FAMILY, adj):
            r = by.get((c, v, mp, prb))
            if r is None:
                continue
            r["p_holm"] = float(a)
            if r["not_run"]:
                r["verdict"] = "not run"
            elif r["R"] > OC["margin"]:
                r["verdict"], r["reason"] = "contradicted", f"contradicted (R = {r['R']:.3f} > {OC['margin']})"
            elif a <= OC["alpha"] and not K7_STATE["calibrated"]:
                r["verdict"], r["reason"] = "undetermined", f"undetermined (K7 not calibrated; p_Holm = {a:.4f})"
            elif a <= OC["alpha"]:
                r["verdict"], r["reason"] = "not contradicted", f"not contradicted (p_Holm = {a:.4f})"
            else:
                r["verdict"], r["reason"] = "undetermined", f"undetermined (R = {r['R']:.3f}, p_Holm = {a:.4f})"
    gov = {}
    for c, v in HOLM_FAMILY:
        rs = [by[(c, v, mp, prb)] for mp, prb in COMBOS if (c, v, mp, prb) in by]
        if not rs:
            gov[f"{c}-{v:g}"] = dict(verdict="not run", reason="not run (no records)")
            continue
        w = min(rs, key=lambda r: VERDICT_ORDER[r["verdict"]])
        gov[f"{c}-{v:g}"] = dict(verdict=w["verdict"], reason=w["reason"], mapping=w["mapping"], prb=w["prb"], R=w["R"],
                                 p_holm=w["p_holm"], n_sites=w["n_sites"], c=w["c"])
    return gov


def bridge(gov, opt_rows=()):
    """Consequence per site type: in-house-only results are usable for a type only
    if every cell of the type is 'not contradicted' and, in each of these cells, the point ratio q99 |e| (worst variant; S-place factor applied) is <= the margin 1.5. case: 'usable', 'tails' (the central spread passes, a
    q99 ratio exceeds 1.5) or 'verdict' (a cell is not 'not contradicted')."""
    q99 = {}
    for r in opt_rows:
        if r is None or r.get("not_run"):
            continue
        key, v = f"{r['cell']}-{r['v0']:g}", r["descriptive"]["q99_ratio"]
        if np.isfinite(v):
            q99[key] = max(q99.get(key, -np.inf), float(v))
    out = {}
    for t, spec in TYPES.items():
        cs = spec["cells"]
        mad_ok = all(gov.get(c, {}).get("verdict") == "not contradicted" for c in cs)
        tails_ok = all(q99.get(c, np.inf) <= OC["margin"] for c in cs)
        usable = bool(mad_ok and tails_ok)
        out[t] = dict(inhouse_only_results_usable=usable, case="usable" if usable else ("tails" if mad_ok else "verdict"),
                      label=None if usable else "in-house channel only; not supported under TR 37.885",
                      q99_ratio_worst={c: q99.get(c) for c in cs}, site_type=spec["site_type"], equal_k_rule=OC["equal_k"])
    return out


K7_STATE = dict(calibrated=True, widen_sigma_ci=False)


def load_c_k7(tag):
    """The K7-calibrated widening factor; required for a primary stage 3. Sets K7_STATE:
    calibrated (False -> every optimism verdict capped at 'undetermined') and widen_sigma_ci."""
    ctag = C["criteria"]["C1"]["coverage_check"]["tag"] if tag == "primary" else f"{tag}_cov"
    path = os.path.join(RESULTS, f"coverage_check_{ctag}.json")
    if not os.path.exists(path):
        if tag == "primary":
            raise RuntimeError(f"{path} missing: the K7 check fixes c before stage 3")
        return 1.0
    d = json.load(open(path))
    K7_STATE.update(calibrated=bool(d.get("calibrated", True)), widen_sigma_ci=bool(d.get("widen_sigma_ci", False)))
    return float(d["c_calibrated"]) if d.get("c_calibrated") is not None else 1.0


def widen_sigma_draws(est, boot, n):
    """If K7 found the sigma-CI coverage < 0.93, the site-bootstrap draws of sigma are widened on
    the log scale by c(n): exp(log est + c(n) (log b - log est))."""
    if not K7_STATE["widen_sigma_ci"] or not (np.isfinite(est) and est > 0) or n < 2:
        return boot
    with np.errstate(invalid="ignore", divide="ignore"):
        return list(np.exp(np.log(est) + c_of_n(n) * (np.log(np.asarray(boot, float)) - np.log(est))))


# ------------------------------------------------------------------------------------------------------------ k3 C4-light
def separation(recs, cell, v0, arm, mapping, prb, T=None):
    """Frequency-only separation (constants evaluation_v4.k3_c4_light): per qualified site,
    z = (d - median_honest) / (1.4826 MAD_honest) over the site's honest test passes (beta 0, with d); Z = median over
    qualified sites of -median(z at beta = -10 %); z_q = -1 / 0 / +1 at the cut z_threshold (1.5): honest z_q shares and
    the per-crossing P(z_q = -1) at beta = -5 % and -10 %."""
    T = T if T is not None else site_table(recs, cell, v0, arm, mapping, prb)[0]
    cut = PC["z_threshold"]
    zs, zh, za, zps = [], [], {-0.05: [], -0.10: []}, []
    for s, t in T.items():
        if t["sd_ref"] > SIG_Q:
            continue
        rs = [r for r in recs if r["site"] == s and r["cell"] == cell and r["v0"] == v0 and r["arm"] == arm
              and r["mapping"] == mapping and r["prb"] == prb and r["role"] == "test" and "d" in r and included(r)]
        dh = np.array([r["d"] for r in rs if r.get("beta", 0.0) == 0.0], float)
        dh = dh[np.isfinite(dh)]
        if dh.size < 5:
            continue
        m, sc = np.median(dh), 1.4826 * np.median(np.abs(dh - np.median(dh)))
        if not sc > 0:
            continue
        zh.append((dh - m) / sc)
        zps.append(float(np.mean((dh - m) / sc < -cut)))
        for b in za:
            da = np.array([r["d"] for r in rs if np.isclose(r.get("beta", 0.0), b)], float)
            da = da[np.isfinite(da)]
            if da.size:
                za[b].append((da - m) / sc)
                if b == -0.10:
                    zs.append(-np.median((da - m) / sc))
    zq = lambda z: np.where(z < -cut, -1, np.where(z > cut, 1, 0))
    zhc = np.concatenate(zh) if zh else np.array([])
    return dict(Z=float(np.median(zs)) if zs else np.nan, n_sites=len(zs),
                zq_shares_honest=[float(np.mean(zq(zhc) == v)) for v in (-1, 0, 1)] if zhc.size else None,
                p_zq_minus1_honest_per_site=dict(values=zps, median=float(np.median(zps)) if zps else np.nan,
                                                 range=[float(min(zps)), float(max(zps))] if zps else None),
                p_zq_minus1={f"{b:+.2f}": (float(np.mean(zq(np.concatenate(za[b])) == -1)) if za[b] else np.nan) for b in za})


def c4_light(recs):
    """k3 per cell (P1-15, P2-30): separation per variant on gen_on and in-house; worst (smallest Z) governs; the rule."""
    K3 = EV4["k3_c4_light"]
    out = {}
    for key in K3["cells"]:
        cell, v0 = key.split("-")[0], float(key.split("-")[1])
        rows = []
        for arm in ("gen_on", "inhouse"):
            for mp, prb in COMBOS:
                rows.append(dict(arm=arm, mapping=mp, prb=prb, **separation(recs, cell, v0, arm, mp, prb)))
        gen = [r for r in rows if r["arm"] == "gen_on" and np.isfinite(r["Z"])]
        w = min(gen, key=lambda r: r["Z"]) if gen else None
        out[key] = dict(variants=rows, Z_worst=w["Z"] if w else np.nan, variant=[w["mapping"], w["prb"]] if w else None,
                        restrict_frequency_only_claim=bool(w is None or w["Z"] < 1.0))
    return out


# ------------------------------------------------------------------------------------------------------------ scope
def clopper_pearson(k, n):
    from scipy import stats as st
    if n == 0:
        return [0.0, 1.0]
    return [float(st.beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(st.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def rating(share):
    return "covered" if share >= SCOPE["covered"] else ("partly covered" if share >= SCOPE["partly_covered"] else "not covered")


def scope(recs, cells, n_sites):
    """Scope rule (constants evaluation_v4.scope_rule): per cell and variant, k = primary sites whose included references
    qualify (gen_on, criteria's one qualification rule; a site without enough references is unqualified), n = the primary
    sites run; the worst (smallest) k governs; rating and Clopper-Pearson; then the type rule."""
    out = {}
    for cell, v0 in cells:
        fam_sites = n_sites[cell]
        rows = []
        for mp, prb in COMBOS:
            ref = [r for r in select(recs, cell, v0, "gen_on", mp, prb) if r["role"] == "ref"]
            k = sum(ref_qualification([r for r in ref if r["site"] == s])["qualified"] for s in range(fam_sites))
            rows.append(dict(mapping=mp, prb=prb, k=int(k)))
        w = min(rows, key=lambda r: r["k"])
        out[f"{cell}-{v0:g}"] = dict(k=w["k"], n=fam_sites, share=w["k"] / fam_sites, cp95=clopper_pearson(w["k"], fam_sites),
                                     rating=rating(w["k"] / fam_sites), variant=[w["mapping"], w["prb"]], variants=rows)
    tout = {}
    for t, spec in TYPES.items():
        r = type_rating(out, [c for c in spec["cells"] if c in out])
        if r is not None:
            tout[t] = r
    return dict(cells=out, types=tout)


def type_rating(cells_out, cs):
    """The type rule of the scope rule on the cells cs (scope; and on the run cells under --incomplete-run)."""
    rs = [cells_out[c] for c in cs]
    if not rs:
        return None
    if all(r["rating"] == "covered" for r in rs):
        return "covered"
    if any(r["rating"] == "covered" for r in rs):
        v = min(float(c.split("-")[1]) for c in cs if cells_out[c]["rating"] == "covered")
        return (f"covered at >= {v:g} m/s" if all(cells_out[c]["rating"] == "covered" for c in cs
                                                  if float(c.split("-")[1]) >= v) else "partly covered")
    if any(r["rating"] == "partly covered" for r in rs):
        return "partly covered"
    return "not covered"


# ------------------------------------------------------------------------------------------------------------ headline, C3, governing
def cell_quantities(recs, cell, v0, mapping, prb, arm="gen_on"):
    T, cnt = site_table(recs, cell, v0, arm, mapping, prb)
    sites = sorted(T)
    if not sites:
        return None
    D = draws(("headline", cell, v0, mapping, prb), len(sites))
    out = dict(mapping=mapping, prb=prb, n_sites=len(sites), window_yield_loss=cnt["window_excluded"] / max(cnt["test"], 1),
               nonfinite_share=cnt["nonfinite"] / max(cnt["test"], 1))
    for qg in (False, True):
        e = pooled(T, sites, qg)
        n_c = n_contributing(T, sites, qg)               # c(n) at the sites that contribute passes
        sb = widen_sigma_draws(sigma(e), [sigma(pooled(T, [sites[i] for i in idx], qg)) for idx in D], n_c)
        out["qualified_gated" if qg else "all"] = dict(sigma=sigma(e), ci=ci(sb), ci_nonfinite_share=nonfinite_share(sb),
                                                        n_sites_contributing=n_c,
                                                        oow=float(np.mean(np.abs(e) > WIN)) if e.size else np.nan,
                                                        q99=q_abs(e, 0.99),
                                                        scales=scale_estimates(e), sigma_e2map=sigma_e2map(e))
    qs = lambda ss: float(np.mean([T[s]["sd_ref"] <= SIG_Q for s in ss]))
    gd = lambda ss: (float(np.mean(np.concatenate([~T[s]["keep_gate"] for s in ss if T[s]["sd_ref"] <= SIG_Q])))
                     if any(T[s]["sd_ref"] <= SIG_Q for s in ss) else np.nan)
    # bias rows: uncalibrated = mean raw statistic b_hat (vehicle speed sensor exact
    # in stage 2, so b_hat is the estimation error), calibrated = mean e = b_hat - survey mean; qualified + gated passes
    qg = lambda ss: [s for s in ss if T[s]["sd_ref"] <= SIG_Q]
    bu = lambda ss: (float(np.mean(np.concatenate([T[s]["e"][T[s]["keep_gate"]] + T[s]["b_ref"] for s in qg(ss)])))
                     if qg(ss) else np.nan)
    bc = lambda ss: float(np.mean(pooled(T, ss, True))) if qg(ss) and pooled(T, ss, True).size else np.nan
    with np.errstate(invalid="ignore"):
        out["bias_uncalibrated"] = dict(mean=bu(sites), ci=ci([bu([sites[i] for i in idx]) for idx in D]))
        out["bias_calibrated"] = dict(mean=bc(sites), ci=ci([bc([sites[i] for i in idx]) for idx in D]))
    # entry yield: crossings that give an entry (included and gated) over all test
    # crossings, at qualified sites
    ey = lambda ss: (float(sum(int(T[s]["keep_gate"].sum()) for s in qg(ss)) / max(sum(T[s]["n_test_all"] for s in qg(ss)), 1))
                     if qg(ss) else np.nan)
    out["entry_yield"], out["entry_yield_ci"] = ey(sites), ci([ey([sites[i] for i in idx]) for idx in D])
    out["qualified_share"], out["qualified_share_ci"] = qs(sites), ci([qs([sites[i] for i in idx]) for idx in D])
    out["gate_false_discard"], out["gate_false_discard_ci"] = gd(sites), ci([gd([sites[i] for i in idx]) for idx in D])
    # variance decomposition (gen_on, all included passes)
    means = [np.mean(T[s]["e"]) for s in sites]
    out["variance"] = dict(within_sd=float(np.sqrt(np.mean([np.var(T[s]["e"], ddof=1) for s in sites]))),
                           between_sd=float(np.std(means, ddof=1)) if len(means) > 1 else np.nan)
    # stop subset (reporting only)
    st = pooled(T, sites, mask="stop")
    nst = sum(int(T[s]["stop"].sum()) for s in sites)
    out["stop_subset"] = dict(share=nst / max(sum(T[s]["e"].size for s in sites), 1), sigma=sigma(st),
                              oow=float(np.mean(np.abs(st) > WIN)) if st.size else np.nan,
                              gate_discard=float(np.mean(np.concatenate([~T[s]["keep_gate"][T[s]["stop"]] for s in sites])))
                              if nst else np.nan)
    return out


def worst(rows, path, mode="max"):
    """Per-quantity worst over variants; returns (value, variant)."""
    best = None
    for r in rows:
        v = r
        for k in path:
            v = v[k]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if best is None or (v > best[0] if mode == "max" else v < best[0]):
            best = (v, dict(mapping=r["mapping"], prb=r["prb"]))
    return best


def place_factor(cell, sp):
    """S-place stratum factor (constants sensitivities.S_place.strata): urban -> P1, P1d09, P3; highway -> P2; P4 the larger."""
    u, h = sp.get("urban", 1.0), sp.get("highway", 1.0)
    return {"P1": u, "P1d09": u, "P3": u, "P2": h, "P4": max(u, h)}.get(cell, 1.0)


S_PLACE_STATUS = dict(status="not run")


def s_place_state(tag):
    """The state line of the S-place run (runs/STATE_<phase>_S), or ''."""
    phase = "p2" if tag == "primary" else None
    f = os.path.join(ER, "runs", f"STATE_{phase}_S") if phase else ""
    return open(f).read().strip() if phase and os.path.exists(f) else ""


def load_s_place(tag):
    """The fixed S-place result. A primary stage 3 needs it unless S-place
    failed or stopped (its run state) or the compute window has closed: its effect is then
    stated as unquantified. Sets S_PLACE_STATUS (status: finished | failed | window closed | not run)."""
    stag = C["sensitivities"]["S_place"]["tag"] if tag == "primary" else f"{tag}_place"   # dev tags: <tag>_place
    path = os.path.join(RESULTS, f"s_place_{stag}.json")
    if not os.path.exists(path):
        st = s_place_state(tag)
        if st.startswith(("FAILED", "STOPPED", "IDENTITY_MISMATCH", "PREFLIGHT_FAILED")):
            S_PLACE_STATUS.update(status="failed", state=st)
            return None
        if tag == "primary":
            from pipeline import stage3_extra_checks as XC
            if XC.window_closed():                       # S-place unfinished at
                S_PLACE_STATUS.update(status="window closed", state=st)   # the window end -> stated as unquantified
                return None
            raise RuntimeError(f"{path} missing: S-place must finish or fail before the primary stage 3 (or the window "
                               f"close); S state {st or 'absent'}")
        return None
    S_PLACE_STATUS.update(status="finished")
    sp = json.load(open(path))
    if tag == "primary":                                 # complete
        SP = C["sensitivities"]["S_place"]
        cells = sp.get("cells", {})
        if sp.get("tag") != SP["tag"]:
            raise RuntimeError(f"{path}: not the S-place sensitivity tag {SP['tag']!r}")
        if sorted(cells) != sorted(SP["cells"]) or any(c.get("n_sites_run") != SP["sites"] for c in cells.values()):
            raise RuntimeError(f"{path}: S-place incomplete (cells {sorted(cells)}, sites {[c.get('n_sites_run') for c in cells.values()]})")
    return sp


def sens_ls_rows(tag, cells):
    """sens_ls (constants sensitivities.sens_ls): per P1/P2 cell the LB variants under the gain shift; required for a
    primary stage 3."""
    SL = C["sensitivities"]["sens_ls"]
    ltag = SL["tag"] if tag == "primary" else f"{tag}_ls"                                 # dev tags: <tag>_ls
    recs, heads = load(SL["families"], ltag) if tag == "primary" or tag.startswith("dev_") else ([], [])
    if tag == "primary":
        if not recs:
            raise RuntimeError("sens_ls stage-2 records missing: run before the primary stage 3 (run_study1.sh)")
        for fam in SL["families"]:                        # the completed prefix under --incomplete-run
            assert_complete([fam], SL["tag"], n_sites=COMPLETED.get(fam))
    out = {}
    for cell, v0 in cells:
        if cell in SL["cells"]:
            rows = [x for x in (cell_quantities(recs, cell, v0, "lb", prb) for prb in PRBS) if x]
            if rows:
                out[(cell, v0)] = rows
    return out


def headline_and_c3(recs, cells, sp_factor=None, ls_rows=None):
    """sp_factor: S-place stratum factors (None = 1); ls_rows: sens_ls rows per cell (co-reported, worse governs)."""
    sp_factor, ls_rows = sp_factor or {}, ls_rows or {}
    out, rows_all = [], {}
    e1 = C["e1_reference"]
    for cell, v0 in cells:
        rows = [x for x in (cell_quantities(recs, cell, v0, mp, prb) for mp, prb in COMBOS) if x]
        if not rows:
            continue
        rows_all[(cell, v0)] = rows
        h = dict(cell=cell, v0=v0, variants=rows)
        for set_ in ("all", "qualified_gated"):
            h[f"worst_sigma_{set_}"] = worst(rows, (set_, "sigma"))
            h[f"worst_sigma_ci_upper_{set_}"] = worst(rows, (set_, "ci", 1))
            h[f"worst_oow_{set_}"] = worst(rows, (set_, "oow"))
        h["worst_qualified_share"] = worst(rows, ("qualified_share",), "min")
        h["worst_qualified_share_ci_lower"] = worst(rows, ("qualified_share_ci", 0), "min")
        h["worst_gate_false_discard"] = worst(rows, ("gate_false_discard",))
        h["worst_gate_false_discard_ci_upper"] = worst(rows, ("gate_false_discard_ci", 1))
        h["worst_window_yield_loss"] = worst(rows, ("window_yield_loss",))
        h["worst_q99_qualified_gated"] = worst(rows, ("qualified_gated", "q99"))
        # the paired in-house arm at the same sites (equal realised K): the worse-of of T1 / T2
        rin = [x for x in (cell_quantities(recs, cell, v0, mp, prb, arm="inhouse") for mp, prb in COMBOS) if x]
        h["inhouse_equalK"] = dict(variants=rin, worst_sigma=worst(rin, ("qualified_gated", "sigma")),
                                   worst_sigma_ci_upper=worst(rin, ("qualified_gated", "ci", 1)),
                                   worst_q99=worst(rin, ("qualified_gated", "q99")), worst_oow=worst(rin, ("qualified_gated", "oow")))
        e1s = e1["sigma_per_pass_pct"].get(f"{v0:g}")
        fac = place_factor(cell, sp_factor)
        h["s_place_factor"] = fac
        ls = ls_rows.get((cell, v0))
        if ls:                                               # sens_ls co-reported; the worse governs the headline
            h["sens_ls"] = dict(variants=ls)
            for set_ in ("all", "qualified_gated"):
                gl, gp = worst(ls, (set_, "sigma")), worst(rows, (set_, "sigma"))
                h["sens_ls"][f"worst_sigma_{set_}"] = gl
                prim_lb = {r["prb"]: r for r in rows if r["mapping"] == "lb"}
                stmt = any(prim_lb.get(r["prb"]) and np.isfinite(r[set_]["sigma"]) and
                           r[set_]["sigma"] - prim_lb[r["prb"]][set_]["sigma"] >
                           (prim_lb[r["prb"]][set_]["ci"][1] - prim_lb[r["prb"]][set_]["ci"][0]) / 2 for r in ls)
                h["sens_ls"][f"statement_lb_depends_on_placement_{set_}"] = bool(stmt)
                if gl is not None and (gp is None or gl[0] > gp[0]):
                    h[f"worst_sigma_{set_}"] = (gl[0], dict(gl[1], source="sens_ls"))
            h["sens_ls"]["worst_qualified_share"] = worst(ls, ("qualified_share",), "min")
            h["sens_ls"]["worst_gate_false_discard"] = worst(ls, ("gate_false_discard",))
        for set_ in ("all", "qualified_gated"):             # worse-of: TR 37.885 (S-place factor), paired in-house, in-house site accuracy
            g = h[f"worst_sigma_{set_}"]
            g = (g[0] * fac, dict(g[1], s_place_factor=fac)) if g is not None else None
            gi = worst(rin, (set_, "sigma"))
            if g is None:                                   # never print the in-house figure for a cell where nothing qualifies
                h[f"headline_final_{set_}"] = dict(value_pct=None, source="no qualified generator site",
                                                   e1_pct=e1s, generator_pct=None, inhouse_equalK_pct=None, variant=None)
                continue
            cand = [("generator", 100 * g[0])]
            if gi is not None:
                cand.append(("in-house (equal K)", 100 * gi[0]))
            if cell in ("P1", "P2") and e1s is not None:
                cand.append(("E1", e1s))
            src, val = max(cand, key=lambda x: x[1])
            h[f"headline_final_{set_}"] = dict(value_pct=val, source=src, e1_pct=e1s if cell in ("P1", "P2") else None,
                                               generator_pct=100 * g[0], inhouse_equalK_pct=100 * gi[0] if gi else None,
                                               variant=g[1])
        e1q = e1["qualified_share_pct"].get(f"{v0:g}")
        wq, wg = h["worst_qualified_share"], h["worst_gate_false_discard"]
        h["c3_final"] = dict(                               # never better than the site-accuracy simulation (constants conventions.never_better)
            qualified_share_pct=(min(100 * wq[0], e1q) if (wq and e1q is not None) else None),
            gate_false_discard_pct=(max(100 * wg[0], max(e1["gate_false_discard_pct"])) if wg else None),
            source="min(generator, E1) / max(generator, E1 upper value)")
        h["c3"] = dict(e1_qualified_share_pct=e1q, e1_gate_discard_pct=e1["gate_false_discard_pct"],
                       target_gate=C["criteria"]["C3"]["gate_target"],
                       gate_target_met=bool(h["worst_gate_false_discard_ci_upper"] and
                                            h["worst_gate_false_discard_ci_upper"][0] <= C["criteria"]["C3"]["gate_target"]))
        out.append(h)
    gov = {}
    for stratum, (cell, speeds) in STRATA.items():
        best = None
        for mp, prb in COMBOS:
            T = [site_table(recs, cell, v, "gen_on", mp, prb)[0] for v in speeds]
            e = np.concatenate([pooled(t, sorted(t)) for t in T if t]) if any(T) else np.array([])
            s = sigma(e)
            if np.isfinite(s) and (best is None or s > best["sigma"] + 1e-15):
                best = dict(mapping=mp, prb=prb, sigma=s)
        gov[stratum] = best if best else dict(mapping="e1", prb=10, sigma=None, note="no data: default e1/10")
    return out, gov


def assert_governing_complete(tag, governing):
    """A primary stage 3 never writes a default governing entry."""
    if tag == "primary" and any("note" in g for g in governing.values()):
        raise RuntimeError(f"primary: a stratum has no data for the governing file: {governing}")


# ------------------------------------------------------------------------------------------------------------ C2
def c2_cell(recs, cell, v0, mapping, prb):
    T, _ = site_table(recs, cell, v0, "gen_on", mapping, prb)
    sites = sorted(T)
    if len(sites) < 2:
        return None
    D = draws(("C2", cell, v0, mapping, prb), len(sites))
    mb = [np.mean(pooled(T, [sites[i] for i in idx])) for idx in D]
    e = pooled(T, sites)
    p_tost = max(pval(mb, lambda b: b >= D_MAX), pval(mb, lambda b: b <= -D_MAX))
    # (ii) excess between-site variance, centred at the pooled shift
    shift = float(np.mean([np.mean(T[s]["e"]) for s in sites]))
    se2 = lambda s: T[s]["sd_ref"] ** 2 / T[s]["n_ref"] + np.var(T[s]["e"], ddof=1) / T[s]["e"].size
    S = float(np.sum([(np.mean(T[s]["e"]) - shift) ** 2 / se2(s) for s in sites]))
    pool = np.concatenate([T[s]["e"] - np.mean(T[s]["e"]) for s in sites])
    rng = rng_for("C2ii", cell, v0, mapping, prb)
    null = np.empty(B)
    for b in range(B):
        acc, mus = [], []
        for s in sites:
            ref = rng.choice(pool, T[s]["n_ref"]); tst = rng.choice(pool, T[s]["e"].size)
            mus.append(tst.mean() - ref.mean())
            acc.append(ref.var(ddof=1) / ref.size + tst.var(ddof=1) / tst.size)
        mus = np.array(mus)
        null[b] = np.sum((mus - mus.mean()) ** 2 / np.array(acc))
    tau2 = max(0.0, float(np.var([np.mean(T[s]["e"]) for s in sites], ddof=1) - np.mean([se2(s) for s in sites])))
    return dict(cell=cell, v0=v0, mapping=mapping, prb=prb, bias=float(np.mean(e)), bias_ci=ci(mb), p_tost=p_tost,
                excess_S=S, excess_p=pval(null, lambda b: b >= S), excess_detected=bool(pval(null, lambda b: b >= S) <= 0.01),
                tau=float(np.sqrt(tau2)))


def c2_all(recs, cells):
    rows = [x for x in (c2_cell(recs, c, v, mp, prb) for c, v in cells for mp, prb in COMBOS) if x]
    fam = [(c, v) for c, v in HOLM_FAMILY]
    for mp, prb in COMBOS:
        by = {(r["cell"], r["v0"]): r for r in rows if r["mapping"] == mp and r["prb"] == prb}
        adj = holm([by[c]["p_tost"] if c in by else 1.0 for c in fam])
        for c, a in zip(fam, adj):
            if c in by:
                by[c]["p_tost_holm"] = float(a)
                by[c]["tost_pass"] = bool(a <= ALPHA)
    summary = {}
    for c, v in cells:
        rs = [r for r in rows if r["cell"] == c and r["v0"] == v]
        if rs:
            summary[f"{c}-{v:g}"] = dict(tost_pass_all_variants=all(r.get("tost_pass", False) for r in rs),
                                         excess_detected_any=any(r["excess_detected"] for r in rs),
                                         tau_max=max(r["tau"] for r in rs),
                                         tau_variant=max(rs, key=lambda r: r["tau"])[("mapping")] + f"/{max(rs, key=lambda r: r['tau'])['prb']}")
    return rows, summary


# ------------------------------------------------------------------------------------------------------------ C5
def c5_cell(recs, cell, v0, mapping, prb):
    T_s, _ = site_table(recs, cell, v0, "gen_on", mapping, prb, kin="sumo")
    sel_q = [r for r in select(recs, cell, v0, "gen_on", mapping, prb, kin="quadratic") if r["role"] == "test" and included(r)]
    if not sel_q or not T_s:
        return None
    sumo = {(r["site"], r["j"]): r for r in select(recs, cell, v0, "gen_on", mapping, prb) if r["role"] == "test" and included(r)}
    d, sid = [], []
    for r in sel_q:
        key = (r["site"], r["j"])
        if key in sumo and r["site"] in T_s:
            k = T_s[r["site"]]["key"]                          # select statistic chosen on the SUMO references
            d.append(sumo[key][k] - r[k]); sid.append(r["site"])
    d, sid = np.array(d), np.array(sid)
    sites = np.unique(sid)
    D = draws(("C5", cell, v0, mapping, prb), len(sites))
    mb = [np.mean(np.concatenate([d[sid == sites[i]] for i in idx])) for idx in D]
    p = max(pval(mb, lambda b: b >= D_MAX), pval(mb, lambda b: b <= -D_MAX))
    return dict(cell=cell, v0=v0, mapping=mapping, prb=prb, pairs=int(d.size), mean_diff=float(np.mean(d)), ci=ci(mb), p_tost=p)


def c5_all(recs):
    rows = [x for x in (c5_cell(recs, c, v, mp, prb) for c, v in C5_FAMILY for mp, prb in COMBOS) if x]
    for mp, prb in COMBOS:
        by = {(r["cell"], r["v0"]): r for r in rows if r["mapping"] == mp and r["prb"] == prb}
        adj = holm([by[c]["p_tost"] if c in by else 1.0 for c in C5_FAMILY])
        for c, a in zip(C5_FAMILY, adj):
            if c in by:
                by[c]["p_holm"], by[c]["pass_"] = float(a), bool(a <= ALPHA)
    failing = sorted({f"{r['cell']}-{r['v0']:g}" for r in rows if not r.get("pass_", False)})
    return rows, failing


# ------------------------------------------------------------------------------------------------------------ C7, C2 (iii)
def c7_cell(recs, base, v0, mapping, prb, qualified_only=True):
    T, _ = site_table(recs, "P6", v0, "gen_on", mapping, prb, ref_cell=base)
    sites = sorted(s for s in T if (T[s]["sd_ref"] <= SIG_Q or not qualified_only))
    if not sites:
        return None
    per = {}
    for s in sites:
        t = T[s]
        ratio = np.array([np.median(t["rms"][i:i + MON_WIN]) / t["rms_ref_med"] for i in range(0, t["rms"].size - MON_WIN + 1)])
        per[s] = dict(discard=~t["keep_gate"], ratio=ratio, e=t["e"])
    D = draws(("C7", base, v0, mapping, prb, qualified_only), len(sites))
    fd = lambda ss: float(np.mean(np.concatenate([per[x]["discard"] for x in ss])))
    q99 = lambda ss: float(np.quantile(np.concatenate([per[x]["ratio"] for x in ss]), 0.99))
    bias = lambda ss: float(np.mean(np.concatenate([per[x]["e"] for x in ss])))
    BS = [[sites[i] for i in idx] for idx in D]
    fdb, qb, bb = [fd(b) for b in BS], [q99(b) for b in BS], [bias(b) for b in BS]
    allr = np.concatenate([per[x]["ratio"] for x in sites])
    return dict(base=base, v0=v0, mapping=mapping, prb=prb, qualified_only=qualified_only, n_sites=len(sites),
                gate_false_discard=fd(sites), gate_ub=ci(fdb)[1], p_gate=pval(fdb, lambda b: b >= C["criteria"]["C7"]["gate_discard_upper"]),
                monitor_q99=q99(sites), monitor_q99_ub=ci(qb)[1], p_monitor=pval(qb, lambda b: b >= C["criteria"]["C7"]["monitor_ratio_q99_upper"]),
                windows=int(allr.size), windows_above_raise=int(np.sum(allr > MON_RAISE)),
                false_raise_rate=float(np.mean(allr > MON_RAISE)), false_stop_rate=float(np.mean(allr > MON_STOP)),
                transfer_bias=bias(sites), transfer_ci=ci(bb),
                p_transfer=max(pval(bb, lambda b: b >= D_MAX), pval(bb, lambda b: b <= -D_MAX)))


def c7_all(recs):
    rows = [x for x in (c7_cell(recs, b, v, mp, prb, q) for b, v in C7_FAMILY for mp, prb in COMBOS for q in (True, False)) if x]
    for mp, prb in COMBOS:
        prim = {(r["base"], r["v0"]): r for r in rows if r["mapping"] == mp and r["prb"] == prb and r["qualified_only"]}
        for key in ("gate", "monitor", "transfer"):
            adj = holm([prim[c][f"p_{key}"] if c in prim else 1.0 for c in C7_FAMILY])
            for c, a in zip(C7_FAMILY, adj):
                if c in prim:
                    prim[c][f"p_{key}_holm"], prim[c][f"{key}_pass"] = float(a), bool(a <= ALPHA)
    return rows


# ------------------------------------------------------------------------------------------------------------ per-beacon, K, P3, lanes
def per_beacon(recs, cells):
    out = {}
    for c, v in cells:
        rs = [r for r in recs if r["cell"] == c and r["v0"] == v and "per_beacon" in r and included(r)]
        if not rs:
            continue
        rows = {}
        for arm in ("gen_off", "gen_on", "inhouse"):
            ra = [r for r in rs if r["arm"] == arm]
            if not ra:
                continue
            acs, err, snr, kk = [], [], [], []
            for r in ra:
                e = np.array(r["per_beacon"]["fdoa_err_hz"], float)
                e = e - np.mean(e)
                if e.size > 10 and np.var(e) > 0:
                    acs.append([float(np.corrcoef(e[:-k], e[k:])[0, 1]) for k in range(1, 6)])
                err.append(np.array(r["per_beacon"]["fdoa_err_hz"], float)); snr.append(np.min(np.array(r["per_beacon"]["snr_db"]), axis=0))
                kk.append(np.full(len(r["per_beacon"]["fdoa_err_hz"]), min(r["K_realised_db"])))
            rho = np.nanmean(acs, axis=0) if acs else [np.nan] * 5
            err, snr, kk = np.concatenate(err), np.concatenate(snr), np.concatenate(kk)
            bins = np.arange(-5, 50, 5)
            rows[arm] = dict(lag_autocorr=[float(x) for x in rho], neff_factor=float((1 - rho[0]) / (1 + rho[0])),
                             fdoa_rms_by_snr={f"{b}": float(np.sqrt(np.mean(err[(snr >= b) & (snr < b + 5)] ** 2)))
                                              for b in bins if np.any((snr >= b) & (snr < b + 5))},
                             fdoa_rms_by_K={f"{b}": float(np.sqrt(np.mean(err[(kk >= b) & (kk < b + 3)] ** 2)))
                                            for b in range(-6, 21, 3) if np.any((kk >= b) & (kk < b + 3))})
        out[f"{c}-{v:g}"] = rows
    return out


def lane_transfer(recs, cells):
    """C2 (iii) descriptive: mean calibrated residual per lane (gen_on, every variant; references survey-like)."""
    out = {}
    for c, v in cells:
        for mp, prb in COMBOS:
            T, _ = site_table(recs, c, v, "gen_on", mp, prb)
            if not T:
                continue
            e = np.concatenate([T[s]["e"] for s in T]); ln = np.concatenate([T[s]["lane"] for s in T])
            out[f"{c}-{v:g}|{mp}|{prb}"] = {str(int(lv)): dict(n=int(np.sum(ln == lv)), bias=float(np.mean(e[ln == lv])))
                                            for lv in np.unique(ln)}
    return out


def realised_k(recs, cells):
    out = {}
    for c, v in cells:
        ks = np.array([r["K_realised_db"] for r in recs if r["cell"] == c and r["v0"] == v and r["arm"] == "gen_off"
                       and r["mapping"] == "e1" and r["prb"] == PRBS[0]], float)
        if ks.size:
            out[f"{c}-{v:g}"] = {h: dict(median=float(np.median(ks[:, i])), iqr=[float(np.quantile(ks[:, i], 0.25)),
                                                                             float(np.quantile(ks[:, i], 0.75))])
                                 for i, h in enumerate(("A", "B"))}
    return out


def k_report(tag, families):
    """Fixed K report from the stage-1 records. Per (cell, v0,
    scenario): realised broadband K (ls-based, the paired arm's basis) and in-band K per PRB — median, IQR and a
    site-bootstrap 95 % CI of the mean; condition 4 per cell: ΔK = K_on − K_off from the merged coefficients, broadband
    and in-band — signed mean with site-bootstrap CI and median |ΔK|, plus 'unbiased' = CI inside ±0.1 dB; per segment
    the realised K (term off) minus the builder's K — median, q10, q90; the per-beacon-normalised in-band K (diagnostic);
    the configured KF (mean, sd) from QuaDRiGa's config (constants quadriga.kf_reference_db). Descriptive only."""
    from pipeline.stage1_channels import FAMILIES
    ref = C["quadriga"]["kf_reference_db"]
    acc = {}
    for fam in families:
        cfg = FAMILIES[fam]["cfg"]
        for f in sorted(glob.glob(os.path.join(RUNS, f"g1_{fam}", "site_*.json"))):
            S1 = json.load(open(f))
            if S1.get("provenance", {}).get("tag") != tag:
                continue
            if os.path.exists(f[:-5] + ".npz"):
                Z = np.load(f[:-5] + ".npz")
            else:                                       # merged runner records carry the K fields only
                kx = f[:-5] + ".kx.json"
                if S1.get("kx_sha256") and sha256_file(kx) != S1["kx_sha256"]:
                    raise RuntimeError(f"{kx}: SHA-256 differs from its stage-1 record")
                Z = {k: np.asarray(v, float) for k, v in json.load(open(kx)).items()}
            site = int(os.path.basename(f)[5:8])
            for m in S1["meta"]:
                pre = f"t{m['k']}_"
                if m.get("kin") == "quadratic" or pre + "K_coef_on_db" not in Z:   # C5 control tracks excluded
                    continue
                cfgs = cfg if isinstance(cfg, list) else [cfg] * Z[pre + "K_realised_db"].size
                for h, c in enumerate(cfgs):
                    a = acc.setdefault((m["cell"], m["v0"], c), dict(site=[], Kr=[], Kb=[], dK=[], dKb=[], seg=[], pbn=[]))
                    a["site"].append(site)
                    a["Kr"].append(float(Z[pre + "K_realised_db"][h])); a["Kb"].append(Z[pre + "K_band_db"][h])
                    a["dK"].append(float(Z[pre + "K_coef_on_db"][h] - Z[pre + "K_coef_off_db"][h]))
                    a["dKb"].append(Z[pre + "K_band_on_db"][h] - Z[pre + "K_band_db"][h])
                    a["pbn"].append(Z[pre + "K_band_pbn_db"][h])
                    ks, kb = Z[pre + "K_seg_db"][h], Z[pre + "kf"][h]
                    n_ = min(ks.size, kb.size)
                    d = ks[:n_] - kb[:n_]
                    a["seg"] += list(d[np.isfinite(d)])

    def summ(x, sites, key):
        x, sites = np.asarray(x, float), np.asarray(sites)
        us = np.unique(sites)
        means = {u: np.mean(x[sites == u]) for u in us}
        D = draws(key, len(us))
        mb = [np.mean([means[us[i]] for i in idx]) for idx in D]
        return dict(n=int(x.size), n_sites=int(us.size), median=float(np.median(x)),
                    site_median_sd=float(np.std([np.median(x[sites == u]) for u in us], ddof=1)) if us.size > 1 else np.nan,
                    iqr=[float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))], mean=float(np.mean(x)), mean_ci=ci(mb),
                    median_abs=float(np.median(np.abs(x))))
    out = {}
    for (cell, v0, c), a in sorted(acc.items(), key=str):
        key = f"{cell}-{v0:g}|{c}"
        Kb, dKb, pbn = np.array(a["Kb"]), np.array(a["dKb"]), np.array(a["pbn"])
        dk = summ(a["dK"], a["site"], ("K", key, "dK"))
        row = dict(kf_configured_db=ref.get(c), K_broadband=summ(a["Kr"], a["site"], ("K", key, "Kr")),
                   K_in_band={str(p): summ(Kb[:, i], a["site"], ("K", key, "Kb", p)) for i, p in enumerate(PRBS)},
                   K_in_band_per_beacon_normalised={str(p): summ(pbn[:, i], a["site"], ("K", key, "pbn", p))
                                                    for i, p in enumerate(PRBS)},
                   condition4_dK_broadband=dict(dk, unbiased=bool(-0.1 <= dk["mean_ci"][0] and dk["mean_ci"][1] <= 0.1),
                                                pass_median_abs=bool(dk["median_abs"] <= 0.1)),
                   condition4_dK_in_band={str(p): summ(dKb[:, i], a["site"], ("K", key, "dKb", p)) for i, p in enumerate(PRBS)},
                   segment_realised_minus_builder=dict(n=len(a["seg"]), median=float(np.median(a["seg"])) if a["seg"] else None,
                                                       q10=float(np.quantile(a["seg"], 0.1)) if a["seg"] else None,
                                                       q90=float(np.quantile(a["seg"], 0.9)) if a["seg"] else None))
        out[key] = row
    return out


def realised_speed(recs, cells):
    """Realised mean speed per cell (test passes, sumo)."""
    out = {}
    for c, v in cells:
        vm = np.array([r["v_mean"] for r in recs if r["cell"] == c and r["v0"] == v and r["arm"] == "gen_on"
                       and r["mapping"] == "e1" and r["prb"] == PRBS[0] and r["kin"] == "sumo" and r["role"] == "test"
                       and "v_mean" in r])
        if vm.size:
            out[f"{c}-{v:g}"] = dict(median=float(np.median(vm)), q05=float(np.quantile(vm, 0.05)), q95=float(np.quantile(vm, 0.95)))
    return out


def p3_blocked(recs):
    b = np.array([r["blocked_share"] for r in recs if r["cell"] == "P3" and r["role"] == "test" and r["arm"] == "gen_on"
                  and r["mapping"] == "e1" and r["prb"] == PRBS[0]])
    return dict(n=int(b.size), mean=float(b.mean()), share_any=float(np.mean(b > 0)),
                quantiles=[float(np.quantile(b, q)) for q in (0.5, 0.9, 0.99)]) if b.size else None


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--families", nargs="+", default=PRIMARY_FAMILIES)
    ap.add_argument("--incomplete-run", action="store_true",
                    help="only after the pre-specified compute window has closed (primary)")
    a = ap.parse_args()
    check_tag(a.tag, stage=3)
    from pipeline import stage3_extra_checks as XC
    if a.incomplete_run and a.tag == "primary" and not XC.window_closed():
        raise RuntimeError("primary: --incomplete-run before the pre-specified compute window has closed")
    if a.tag == "primary" and not a.incomplete_run:
        assert_complete(PRIMARY_FAMILIES, a.tag)
    if a.tag == "primary" and sorted(a.families) != sorted(PRIMARY_FAMILIES):
        raise ValueError(f"primary: the full family list {PRIMARY_FAMILIES} is required (a subset changes the governing file)")
    recs, heads = load(a.families, a.tag)
    cells = sorted({(r["cell"], r["v0"]) for r in recs if r["cell"] in ("P1", "P2", "P3")})
    n_sites = {}
    for fam, cl in (("U", "P1"), ("H", "P2"), ("N", "P3")):
        ss = {r["site"] for r in recs if r.get("family") == fam}
        if ss:
            n_sites[cl] = max(ss) + 1                       # primary sites 0..n-1 (assert_complete checks completeness)
            if a.incomplete_run:                            # sites run in seed order; n = the completed sites
                n_sites[cl] = len(ss)
                COMPLETED[fam] = len(ss)
                if sorted(ss) != list(range(len(ss))):
                    raise RuntimeError(f"{fam}: completed sites {sorted(ss)} are not a seed-order prefix")
    # descriptive model-diversity check (C1 quantities; gen_off vs in-house; no verdict)
    c1 = [x for x in (c1_cell(recs, c, v, mp, prb) for c, v in cells for mp, prb in COMBOS) if x]
    sp = load_s_place(a.tag)
    spf = sp["stratum_factor"] if sp else {}
    c_k7 = load_c_k7(a.tag)
    opt = [x for x in (optimism_cell(recs, c, v, mp, prb, c_k7) for c, v in cells for mp, prb in COMBOS) if x]
    gov_opt = optimism_verdicts(opt)
    opt_place, gov_opt_place = None, None
    if any(place_factor(c, spf) != 1.0 for c, _ in cells):   # S-place correction: re-evaluate, the worse governs
        opt_place = [optimism_cell(recs, c, v, mp, prb, c_k7, place_factor(c, spf)) for c, v in cells for mp, prb in COMBOS]
        gov_opt_place = optimism_verdicts(opt_place)
        for k, g in gov_opt_place.items():
            if VERDICT_ORDER[g["verdict"]] < VERDICT_ORDER[gov_opt[k]["verdict"]]:
                gov_opt[k] = dict(g, source="S-place corrected (worse)")
    head, governing = headline_and_c3(recs, cells, spf, sens_ls_rows(a.tag, cells))
    assert_governing_complete(a.tag, governing)
    c2_rows, c2_sum = c2_all(recs, [x for x in cells if x in HOLM_FAMILY])
    sc = scope(recs, cells, n_sites)
    ablation = None
    if LEVER_D:                                             # Huber fit (d) adopted: the LS statistics as the ablation row
        rl, _ = load(a.families, a.tag, stat="ls")
        ablation = dict(statistic="ls", headline=headline_and_c3(rl, cells, spf)[0], scope=scope(rl, cells, n_sites))
    out = dict(tag=a.tag, provenance=provenance(a.tag, boot=B, families=sorted(a.families)),
               design="study-1 evaluation design (constants evaluation_v4)", statistic="d" if LEVER_D else "ls",
               optimism=opt, optimism_governing=gov_opt, optimism_s_place=opt_place, optimism_governing_s_place=gov_opt_place,
               bridge=bridge(gov_opt, opt + (opt_place or [])), scope=sc, model_diversity_c1=c1,
               headline=head, governing=governing, c2=c2_rows, c2_summary=c2_sum,
               per_beacon=per_beacon(recs, cells), realised_k=realised_k(recs, cells),
               k_report=k_report(a.tag, a.families), realised_speed=realised_speed(recs, cells),
               p3_blocked=p3_blocked(recs), s_place=sp, s_place_status=dict(S_PLACE_STATUS), c_k7=c_k7, ablation=ablation, c4_light=c4_light(recs),
               not_run=EV4["not_run"])
    cell_n = {f"{c}-{v:g}": n_sites.get(c, 0) for c, v in cells}
    XC.add_v5(out, recs, cells, cell_n, a.tag, a.families, incomplete=a.incomplete_run)
    d = RESULTS
    os.makedirs(d, exist_ok=True)
    for name, obj in ((f"criteria_{a.tag}.json", out),
                      (f"governing_{a.tag}.json", dict(governing, c5_failing=[], c2_tau={k: v["tau_max"] for k, v in c2_sum.items()},
                                                        scope=out["scope"]["cells"], provenance=provenance(a.tag)))):
        json.dump(obj, open(os.path.join(d, name + ".tmp"), "w"), indent=1, default=float)
        os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))   # atomic
    print(os.path.join(d, f"criteria_{a.tag}.json"))


if __name__ == "__main__":
    main()
