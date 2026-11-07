"""Study 2, development stage: the development diagnostic and the pre-specified adoption rules D0-R1 ... D0-R5,
written into step0_decision.json by rule. Development data only (Dev-1 P0U / P0N / P0H dev_s2 records, Dev-2 study-1
U / N records = primary stage-2 records for E-a / E-b / D0.1 / D0.3 / D0.5 plus the dev_s2 re-run for the new fields,
Dev-3 D3U). Refused before the development-stage design record is timestamped.

Baseline and readings: every D0 quantity compares a candidate
configuration with plain LS on the same input passes (same records, sites and pass indices of the same reading); each
configuration is evaluated on its own included and gated test passes (its inclusion rule incl. its exclusions, its RMS
gate), so an exclusion's effect enters S_all, q99_all, T_all and the yield; the paired values over the passes included
under both are printed, never gating. Worst variants of candidate and LS
are taken separately, per-variant ratios reported. The governing quantity of D0-R3 is the worse of gen_on and in-house
per (dataset, reading, cell), candidate and LS taken separately.
  excess bias  mean uncalibrated b̂ of the gated test passes - mean b̂ of the free-road reference passes (role "ref" as
               recorded) at the same sites = the pooled mean of the reading-F calibrated residual e of the gated test
               passes; per urban cell, arm, worst variant (largest |.|); never computed with the surrogate survey
  S_all, q99_all, T_all  as in study 1 (all included and gated test passes, not conditioned on qualification), own set
  yield        included and gated test passes / all test passes of the cell (each configuration on its own set)
  reading F    free-road reference as recorded (Dev-1 all 30 test passes; Dev-2 test passes jr >= 30)
  reading T    L-T surrogate survey (Dev-2 test passes jr < 30) for calibration, gate and qualification; test jr >= 30"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
from pipeline import stage3_criteria as K  # noqa: E402
from pipeline import s2_criteria as CS  # noqa: E402
from pipeline import s2_constants as S2  # noqa: E402
from pipeline.pilot_decision import all_gated_paired  # noqa: E402

ARMS = ("gen_on", "inhouse")
URBAN = (("P1", 10.0), ("P1", 15.0))
HIGHWAY_P0H = (("P2", 20.0), ("P2", 30.0))
P3 = (("P3", 15.0),)


def key(c):
    return f"{c[0]}-{c[1]:g}"


# ---------------------------------------------------------------------------------------------- per-variant quantities
def tables(recs, cell, v0, arm, mp, prb, kin="sumo"):
    return K.site_table(recs, cell, v0, arm, mp, prb, kin=kin)


def gated(T):
    return np.concatenate([T[s]["e"][T[s]["keep_gate"]] for s in sorted(T)]) if T else np.array([])


def excess_bias(T_F):
    e = gated(T_F)
    return float(np.mean(e)) if e.size else np.nan


def yield_share(T, counts):
    n = sum(int(T[s]["keep_gate"].sum()) for s in T)
    return K._div(n, counts["test"])


def _sat(e):
    """S_all, q99_all, T_all and n of one residual vector."""
    return (K.sigma(e), K.q_abs(e, 0.99), float(np.mean(np.abs(e) > K.WIN)) if e.size else np.nan, int(e.size))


def variant_rows(R_ls, R_c, cell, v0, arm, R_ls_F=None, R_c_F=None):
    """Per variant: S_all / q99_all / T_all of each configuration on its own included and gated test passes from the same
    input records, the paired values over the passes included under both (printed only), excess bias
    (reading F tables; R_*_F given when R_* is a reading-T set), calibrated bias, yields, non-finite shares."""
    rows = []
    for mp, prb in K.COMBOS:
        T_ls, c_ls = tables(R_ls, cell, v0, arm, mp, prb)
        T_c, c_c = tables(R_c, cell, v0, arm, mp, prb)
        e_ls, e_c = gated(T_ls), gated(T_c)                  # own sets (gating)
        p_ls, p_c = all_gated_paired(T_ls, T_c)             # common set (printed)
        TF_ls = tables(R_ls_F, cell, v0, arm, mp, prb)[0] if R_ls_F is not None else T_ls
        TF_c = tables(R_c_F, cell, v0, arm, mp, prb)[0] if R_c_F is not None else T_c
        (S_ls, q_ls, t_ls, n_ls), (S_c, q_c, t_c, n_c) = _sat(e_ls), _sat(e_c)
        (pS_ls, pq_ls, pt_ls, n_p), (pS_c, pq_c, pt_c, _) = _sat(p_ls), _sat(p_c)
        rows.append(dict(mapping=mp, prb=prb,
                         S_all_ls=S_ls, S_all_c=S_c, q99_all_ls=q_ls, q99_all_c=q_c, T_all_ls=t_ls, T_all_c=t_c,
                         n_ls=n_ls, n_c=n_c,
                         excess_bias_ls=excess_bias(TF_ls), excess_bias_c=excess_bias(TF_c),
                         cal_bias_ls=excess_bias(T_ls), cal_bias_c=excess_bias(T_c),
                         yield_ls=yield_share(T_ls, c_ls), yield_c=yield_share(T_c, c_c),
                         nonfinite_ls=K._div(c_ls["nonfinite"], c_ls["test"]), nonfinite_c=K._div(c_c["nonfinite"], c_c["test"]),
                         S_ratio=K._div(S_c, S_ls),
                         paired=dict(S_all_ls=pS_ls, S_all_c=pS_c, q99_all_ls=pq_ls, q99_all_c=pq_c, T_all_ls=pt_ls,
                                     T_all_c=pt_c, n=n_p, S_ratio=K._div(pS_c, pS_ls))))
    return rows


def worst(rows, field, mode="max", absval=False):
    v = [r for r in rows if np.isfinite(r[field])]
    if not v:
        return dict(value=np.nan, variant=None, n=0)
    f = (lambda r: abs(r[field])) if absval else (lambda r: r[field])
    r = (max if mode == "max" else min)(v, key=f)
    return dict(value=float(r[field]), variant=[r["mapping"], r["prb"]], n=int(r["n_c" if field.endswith("_c") else "n_ls"]))


def summarise(rows):
    """Worst variants taken separately for the candidate and LS; the per-variant rows (with the
    paired common-set values) are kept for the report."""
    out = {}
    for f in ("S_all", "q99_all", "T_all", "nonfinite"):
        out[f + "_ls"], out[f + "_c"] = worst(rows, f + "_ls"), worst(rows, f + "_c")
    for f in ("excess_bias", "cal_bias"):
        out[f + "_ls"], out[f + "_c"] = worst(rows, f + "_ls", absval=True), worst(rows, f + "_c", absval=True)
    out["yield_ls"], out["yield_c"] = worst(rows, "yield_ls", "min"), worst(rows, "yield_c", "min")
    out["per_variant"] = rows
    return out


GOV_MAX = ("S_all", "q99_all", "T_all", "nonfinite")      # worse = larger
GOV_ABS = ("excess_bias", "cal_bias")                      # worse = larger |.| (signed value kept)
GOV_MIN = ("yield",)                                       # worse = smaller


def governing(per_arm):
    """The study-1 governing quantity: per quantity the worse of the arms' worst-variant values, candidate and LS each taken separately. per_arm: {arm: summary}. Returns a summary of the same form with the
    governing arm of each value."""
    out = {}
    for side in ("_c", "_ls"):
        for f in GOV_MAX + GOV_ABS + GOV_MIN:
            vals = [(arm, sm[f + side]) for arm, sm in per_arm.items() if np.isfinite(sm[f + side]["value"])]
            if not vals:
                out[f + side] = dict(value=np.nan, variant=None, n=0, arm=None)
                continue
            key = ((lambda x: abs(x[1]["value"])) if f in GOV_ABS else (lambda x: x[1]["value"]))
            arm, w = (min if f in GOV_MIN else max)(vals, key=key)
            out[f + side] = dict(w, arm=arm)
    return out


def arm_ratios(per_arm):
    """Every arm's ratios (printed; the non-governing arm included)."""
    v = lambda sm, k: sm[k]["value"]
    return {arm: dict(S_ratio=K._div(v(sm, "S_all_c"), v(sm, "S_all_ls")), q99_ratio=K._div(v(sm, "q99_all_c"), v(sm, "q99_all_ls")),
                      T_ratio=K._div(v(sm, "T_all_c"), v(sm, "T_all_ls")), yield_ratio=K._div(v(sm, "yield_c"), v(sm, "yield_ls")),
                      excess_bias_c=v(sm, "excess_bias_c"), excess_bias_ls=v(sm, "excess_bias_ls"))
            for arm, sm in per_arm.items()}


# ---------------------------------------------------------------------------------------------- rule clauses
def bias_ok(eb_c, eb_ls, fall=S2.N8_BIAS_FALL, band=S2.N8_BIAS_BAND):
    """|excess bias| falls >= 50 % or is <= 0.5 %."""
    return bool(np.isfinite(eb_c) and (abs(eb_c) <= band or (np.isfinite(eb_ls) and abs(eb_c) <= (1 - fall) * abs(eb_ls))))


def urban_clauses(sm):
    """D0-R3 urban clauses on one (dataset, reading, cell) governing summary (N8, N9): |excess bias| falls >= 50 % or is
    <= 0.5 %; S_all falls >= 20 %; q99_all rises <= 5 % (relative); yield falls <= 30 % (relative); non-finite <= 5 %."""
    v = lambda k: sm[k]["value"]
    y_ok = bool(np.isfinite(v("yield_c")) and np.isfinite(v("yield_ls")) and
                v("yield_c") >= (1 - S2.N8_YIELD_LOSS_MAX) * v("yield_ls"))
    return dict(bias=bias_ok(v("excess_bias_c"), v("excess_bias_ls")),
                S_all=bool(np.isfinite(v("S_all_c")) and np.isfinite(v("S_all_ls")) and
                           v("S_all_c") <= (1 - S2.N8_S_ALL_FALL) * v("S_all_ls")),
                q99_all=bool(np.isfinite(v("q99_all_c")) and np.isfinite(v("q99_all_ls")) and
                             v("q99_all_c") <= (1 + S2.N8_Q99_MAY_RISE) * v("q99_all_ls")),
                yield_=y_ok,
                nonfinite=bool(np.isfinite(v("nonfinite_c")) and v("nonfinite_c") <= S2.N9_NONFINITE_MAX))


def dev1_guard(sm):
    """D0-R3 Dev-1 (reading F) non-regression guard (N8) on the governing summary: S_all rises by <= 5 %
    and |excess bias| rises by <= 0.5 % (absolute)."""
    v = lambda k: sm[k]["value"]
    return dict(S_all=bool(np.isfinite(v("S_all_c")) and np.isfinite(v("S_all_ls")) and
                           v("S_all_c") <= (1 + S2.N8_DEV1_S_RISE) * v("S_all_ls")),
                bias=bool(np.isfinite(v("excess_bias_c")) and np.isfinite(v("excess_bias_ls")) and
                          abs(v("excess_bias_c")) <= abs(v("excess_bias_ls")) + S2.N8_DEV1_BIAS_RISE_ABS))


def highway_clauses(sm):
    """D0-R3 P0H clause (N9): S_all rises <= 5 %; T_all rises <= max(5 %, one crossing)."""
    v = lambda k: sm[k]["value"]
    n = max(sm["T_all_ls"]["n"], 1)
    return dict(S_all=bool(np.isfinite(v("S_all_c")) and v("S_all_c") <= (1 + S2.N9_P0H_S_REL) * v("S_all_ls")),
                T_all=CS.t_all_ok(v("T_all_c"), v("T_all_ls"), n) if np.isfinite(v("T_all_ls")) else False)


# ---------------------------------------------------------------------------------------------- datasets
def configs(cand, cal, reading, variant=None, n2=None):
    return lambda recs: CS.map_records(recs, cand, cal, reading, variant or "no_entry", n2=n2)


def candidate_entries(dev1, dev2, cand, cal, variant=None):
    """Every (dataset, reading, cell, arm) summary that D0-R3 reads for one candidate. dev1 / dev2: dicts family ->
    records."""
    out = {}
    mF, mT = configs(cand, cal, "F", variant), configs(cand, cal, "T", variant)
    d1u = dev1.get("P0U", [])
    for c in URBAN:
        for arm in ARMS:
            out[("Dev-1", "F", key(c), arm)] = summarise(variant_rows(d1u, mF(d1u), c[0], c[1], arm))
    d1h = dev1.get("P0H", [])
    for c in HIGHWAY_P0H:
        for arm in ARMS:
            out[("Dev-1/P0H", "F", key(c), arm)] = summarise(variant_rows(d1h, mF(d1h), c[0], c[1], arm))
    u = dev2.get("U", [])
    uF, uT = CS.reading_F(u, dev2=True), CS.reading_T(u)
    for c in URBAN:
        for arm in ARMS:
            out[("Dev-2", "F", key(c), arm)] = summarise(variant_rows(uF, mF(uF), c[0], c[1], arm))
            out[("Dev-2", "T", key(c), arm)] = summarise(variant_rows(uT, mT(uT), c[0], c[1], arm,
                                                                      R_ls_F=uF, R_c_F=mF(uF)))
    return out


def governing_entries(entries):
    """Per (dataset, reading, cell): the governing (worse-of-arms) summary and every arm's ratios."""
    groups = {}
    for k, sm in entries.items():
        groups.setdefault(k[:3], {})[k[3]] = sm
    return {k: dict(gov=governing(per_arm), arms=arm_ratios(per_arm)) for k, per_arm in groups.items()}


def eligible(entries):
    """D0-R3 (N8, N9): on the governing quantity (worse of gen_on and in-house per cell, candidate and LS taken
    separately), Dev-2 readings F and T, both urban cells: the urban clauses; Dev-1 (reading F): the non-regression guard
    only; P0H: the study-1 R3 highway clause per arm (unchanged; each arm against its own LS)."""
    verdict, why = True, []
    gov = governing_entries({k: v for k, v in entries.items() if k[0] != "Dev-1/P0H"})
    for k, g in gov.items():
        cl = dev1_guard(g["gov"]) if k[0] == "Dev-1" else urban_clauses(g["gov"])
        if not all(cl.values()):
            verdict = False
            why.append(dict(entry=list(k) + ["governing"], failed=[c for c, ok in cl.items() if not ok]))
    for k, sm in entries.items():
        if k[0] != "Dev-1/P0H":
            continue
        cl = highway_clauses(sm)
        if not all(cl.values()):
            verdict = False
            why.append(dict(entry=list(k), failed=[c for c, ok in cl.items() if not ok]))
    return verdict, why


def s_reduction(entries):
    """Tie-break statistic: the mean relative reduction of the governing (worse-of-arms) S_all over the Dev-2 reading-T
    urban cells (consistent with the worse-of eligibility)."""
    gov = governing_entries(entries)
    r = [1 - gov[("Dev-2", "T", key(c))]["gov"]["S_all_c"]["value"] / gov[("Dev-2", "T", key(c))]["gov"]["S_all_ls"]["value"]
         for c in URBAN]
    return float(np.mean(r))


def choose(results):
    """Among eligible candidates the largest mean S_all reduction; within 2 % (relative) of the best -> the earlier
    letter (E-a < E-b < E-c < E-f < E-d < E-db < E-e). results: list of dict(name, eligible, reduction)
    in the pre-specified order. Returns the chosen name or None (no estimator change: L-T alone)."""
    el = [r for r in results if r["eligible"] and np.isfinite(r["reduction"])]
    if not el:
        return None
    best = max(r["reduction"] for r in el)
    tied = [r for r in el if abs(best - r["reduction"]) <= S2.N8_TIE_REL * abs(best)]
    return tied[0]["name"]


# ---------------------------------------------------------------------------------------------- D0-R1, D0-R2
def paired_kin_diff(recs, cell, v0, arm, mp, prb):
    """D0.2: per site, the paired differences SUMO - quadratic of b̂ (select statistic, same free-road calibration; the
    calibration cancels) of the test passes included and gated under both (the excess bias it is
    compared with is gated), paired by the SUMO pass index j (pair_k)."""
    T_s, _ = tables(recs, cell, v0, arm, mp, prb, "sumo")
    T_q, _ = tables(recs, cell, v0, arm, mp, prb, "quadratic")
    out = {}
    for s in sorted(set(T_s) & set(T_q)):
        jq = {int(j): i for i, j in enumerate(T_q[s]["j"]) if T_q[s]["keep_gate"][i]}
        d = [T_s[s]["e"][i] - T_q[s]["e"][jq[int(j)]] for i, j in enumerate(T_s[s]["j"])
             if int(j) in jq and T_s[s]["keep_gate"][i]]
        if d:
            out[s] = np.array(d)
    return out, T_s


def site_bootstrap_mean(by_site, key_):
    sites = sorted(by_site)
    if not sites:
        return np.nan, [np.nan, np.nan]
    rng = K.rng_for(*key_)
    m = [np.mean(np.concatenate([by_site[sites[i]] for i in rng.integers(0, len(sites), len(sites))])) for _ in range(K.B)]
    return float(np.mean(np.concatenate([by_site[s] for s in sites]))), K.ci(m)


def d0_r1(dev3, arm="gen_on"):
    """D0-R1 (N6): 'kinematic cause supported' if, in >= 1 urban cell, the mean paired difference (D0.2) is >= 1/2 of the
    excess bias (same sign) with a site-bootstrap 95 % CI excluding 0; evaluated at the cell's worst variant (largest
    |excess bias| of the Dev-3 SUMO test passes, reading F). Reported either way."""
    cells = {}
    for c in URBAN:
        rows = []
        for mp, prb in K.COMBOS:
            diffs, T_s = paired_kin_diff(dev3, c[0], c[1], arm, mp, prb)
            eb = excess_bias(T_s)
            md, ci = site_bootstrap_mean(diffs, ("S2-D0.2", c[0], c[1], arm, mp, prb))
            rows.append(dict(mapping=mp, prb=prb, excess_bias=eb, mean_diff=md, ci=ci,
                             n_pairs=int(sum(v.size for v in diffs.values()))))
        fin = [r for r in rows if np.isfinite(r["excess_bias"])]
        w = max(fin, key=lambda r: abs(r["excess_bias"])) if fin else None
        ok = bool(w and np.isfinite(w["mean_diff"]) and w["mean_diff"] * np.sign(w["excess_bias"]) >=
                  S2.N6_CAUSE_SHARE * abs(w["excess_bias"]) and (w["ci"][0] > 0 or w["ci"][1] < 0))
        cells[key(c)] = dict(variants=rows, worst=w, holds=ok)
    n = sum(v["holds"] for v in cells.values())
    return dict(cells=cells, verdict="kinematic cause supported" if n >= S2.N6_MIN_CELLS else "not isolated",
                holds=bool(n >= S2.N6_MIN_CELLS))


def nonstop_share(T_F):
    """Share of the excess bias carried by the non-stop passes: sum of e over the gated non-stop test passes / sum of e
    over all gated test passes (reading F; stop_between is the truth flag, reporting only)."""
    tot = sum(float(T_F[s]["e"][T_F[s]["keep_gate"]].sum()) for s in T_F)
    ns = sum(float(T_F[s]["e"][T_F[s]["keep_gate"] & ~T_F[s]["stop"]].sum()) for s in T_F)
    return K._div(ns, tot)


def d0_r2(r1, dev1, dev2, arm="gen_on"):
    """D0-R2 (N7): E-e enters development only if D0-R1 holds and the non-stop passes carry >= 1/2 of the excess bias
    (D0.1) in a cell where D0-R1 holds, in Dev-1 (reading F) and in Dev-2 (reading F), at the cell's worst variant."""
    if not r1["holds"]:
        return dict(holds=False, reason="D0-R1 does not hold")
    res = {}
    uF = CS.reading_F(dev2.get("U", []), dev2=True)
    for c in URBAN:
        if not r1["cells"][key(c)]["holds"]:
            continue
        sh = {}
        for name, recs in (("Dev-1", dev1.get("P0U", [])), ("Dev-2", uF)):
            rows = []
            for mp, prb in K.COMBOS:
                T_F, _ = tables(recs, c[0], c[1], arm, mp, prb)
                rows.append(dict(mapping=mp, prb=prb, excess_bias=excess_bias(T_F), share=nonstop_share(T_F)))
            fin = [r for r in rows if np.isfinite(r["excess_bias"])]
            sh[name] = max(fin, key=lambda r: abs(r["excess_bias"])) if fin else None
        ok = all(v is not None and np.isfinite(v["share"]) and v["share"] >= S2.N7_NONSTOP_SHARE for v in sh.values())
        res[key(c)] = dict(shares=sh, holds=bool(ok))
    return dict(cells=res, holds=bool(any(v["holds"] for v in res.values())))


# ---------------------------------------------------------------------------------------------- D0-R4 (L-B, L-B′)
def bias_reduction(pairs):
    """D0-R4 |excess bias| reduction: 1 - mean|eb_rule| / mean|eb_LS| over the (|eb_rule|, |eb_LS|) pairs where both
    are finite (a ratio of sums, so one entry with |eb_LS| near 0 cannot dominate); nan if no pair or the LS sum is 0."""
    fin = [(c, l) for c, l in pairs if np.isfinite(c) and np.isfinite(l)]
    if not fin:
        return np.nan
    x = 1 - K._div(sum(c for c, _ in fin), sum(l for _, l in fin))
    return float(x) if np.isfinite(x) else np.nan


def d0_r4_rule(dev1, dev2, cal, n2, arm_list=ARMS):
    """D0-R4 (N10) for one N2 rule (n2 = 'L-B' or 'L-B′'): adoptable if on Dev-1 P0N (reading F) and Dev-2 P3-15
    (readings F and T) |excess bias| falls >= 50 % or is <= 0.5 % at yield loss <= 30 %; in reading T additionally the
    calibrated bias falls >= 50 % or lies within 0.5 %. Per arm (the pre-specified text; each configuration on its own
    set). Also the rule's |excess bias| reduction = 1 - mean|eb_c| / mean|eb_ls| over these entries (a ratio of
    sums over the entries where both are finite; stable when one |eb_ls| is near 0)."""
    mF = lambda recs: CS.map_records(recs, None, cal, "F", n2=n2)
    mT = lambda recs: CS.map_records(recs, None, cal, "T", n2=n2)
    n = dev2.get("N", [])
    nF, nT = CS.reading_F(n, dev2=True), CS.reading_T(n)
    entries = {}
    p0n = dev1.get("P0N", [])
    for arm in arm_list:
        entries[("Dev-1", "F", "P3-15", arm)] = summarise(variant_rows(p0n, mF(p0n), "P3", 15.0, arm))
        entries[("Dev-2", "F", "P3-15", arm)] = summarise(variant_rows(nF, mF(nF), "P3", 15.0, arm))
        entries[("Dev-2", "T", "P3-15", arm)] = summarise(variant_rows(nT, mT(nT), "P3", 15.0, arm, R_ls_F=nF, R_c_F=mF(nF)))
    why, ok, pairs = [], True, []
    for k, sm in entries.items():
        v = lambda f: sm[f]["value"]
        cl = dict(bias=bias_ok(v("excess_bias_c"), v("excess_bias_ls"), S2.N10_LB_BIAS_FALL, S2.N10_LB_BIAS_BAND),
                  yield_=bool(np.isfinite(v("yield_c")) and v("yield_c") >= (1 - S2.N10_LB_YIELD_LOSS_MAX) * v("yield_ls")))
        if k[1] == "T":
            cl["cal_bias"] = bias_ok(v("cal_bias_c"), v("cal_bias_ls"), S2.N10_LB_BIAS_FALL, S2.N10_LB_BIAS_BAND)
        if not all(cl.values()):
            ok = False
            why.append(dict(entry=list(k), failed=[c for c, g in cl.items() if not g]))
        pairs.append((abs(v("excess_bias_c")), abs(v("excess_bias_ls"))))
    return dict(rule=n2, qualifies=bool(ok), failed=why, bias_reduction=bias_reduction(pairs),
                entries={"|".join(map(str, k)): v for k, v in entries.items()})


def d0_r4(dev1, dev2, cal, arm_list=ARMS):
    """D0-R4: L-B and L-B′ judged by the same conditions; if both qualify, the larger |excess bias| reduction is
    adopted, ties (within 2 % relative) to L-B; neither -> no N2 rule."""
    res = [d0_r4_rule(dev1, dev2, cal, n2, arm_list) for n2 in S2.LB_RULES]
    q = [r for r in res if r["qualifies"] and np.isfinite(r["bias_reduction"])]
    adopted = None
    if q:
        best = max(r["bias_reduction"] for r in q)
        adopted = [r for r in q if abs(best - r["bias_reduction"]) <= S2.N8_TIE_REL * abs(best)][0]["rule"]
    return dict(adopted=adopted, rules=res, tie_rule="larger |excess bias| reduction; within 2 % relative -> L-B")


# ---------------------------------------------------------------------------------------------- the decision
def decide(dev1, dev2, dev3, cal):
    """D0-R1 ... D0-R5 in the pre-specified order; every candidate with its values and verdict, rejected ones included.
    The design choices are the pre-specified values (s2_constants.OPTIONS; no alternative)."""
    S2.check_calibration(cal)
    r1 = d0_r1(dev3)
    r2 = d0_r2(r1, dev1, dev2)
    names = [(c, "no_entry" if c in ("E-d", "E-db") else None) for c in S2.CANDIDATES if c != "E-e" or r2["holds"]]
    results = []
    for cand, var in names:
        ent = candidate_entries(dev1, dev2, cand, cal, var)
        el, why = eligible(ent)
        results.append(dict(name=cand, candidate=cand, e_d_variant=var, eligible=el, failed=why, reduction=s_reduction(ent),
                            governing={"|".join(map(str, k)): v for k, v in governing_entries(ent).items()},
                            entries={"|".join(map(str, k)): v for k, v in ent.items()}))
    if not r2["holds"]:
        results.append(dict(name="E-e", candidate="E-e", eligible=False, reduction=np.nan,
                            failed=[dict(entry=["D0-R2"], failed=["not admitted"])], entries={}))
    results.sort(key=lambda r: S2.CANDIDATES.index(r["candidate"]))
    chosen = choose(results)
    ch = next((r for r in results if r["name"] == chosen), None)
    r4 = d0_r4(dev1, dev2, cal)
    selected = dict(estimator=ch["candidate"] if ch else None, e_d_variant=ch.get("e_d_variant") if ch else None,
                    n2_rule=r4["adopted"], survey="traffic (L-T, urban-type sites)")
    return dict(design=S2.SOURCE, D0_R1=r1, D0_R2=r2, candidates=results, D0_R3_chosen=chosen,
                D0_R4=r4, D0_R5=dict(h2_runs=bool(chosen is not None)), L_T="adopted unconditionally (urban-type sites)",
                selected=selected, calibration=cal, options=dict(S2.OPTIONS),
                label="development stage; Dev-2 numbers are 'development, study-1 records'")


def write_decision(out, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(out, open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)
    return path


def guard(consts=None):
    """The development stage runs only after its design record is timestamped and with the calibration."""
    ev6 = (consts or K.C).get("evaluation_v6")
    S2.check_against(ev6)
    if not ev6["step0"].get("timestamped"):
        raise RuntimeError("development stage before its design record is timestamped (evaluation_v6.step0.timestamped)")
    cal = ev6["step0"].get("calibration") or {}
    if any(cal.get(k) is None for k in S2.CALIBRATION_KEYS):
        raise RuntimeError("the development stage needs the calibration (a_max, k_stop, delta_blk_db, delta_rho) in "
                           "evaluation_v6.step0")
    S2.check_calibration(cal)
    if not ev6["step0"].get("study1_merge_done"):
        raise RuntimeError("Dev-2 is read only after the study-1 records are merged")
    return cal


# ---------------------------------------------------------------------------------------------- D0 diagnostics (reported)
def d0_1(recs, cells, arms=ARMS):
    """D0.1: bias and sigma of prof, match, w_prof, w_match, c_prof, c_match (each calibrated on the free-road survey of
    the same statistic), stratified by stop_between, minimum speed in the window, lane change and junction distance."""
    strata = dict(stop_between=lambda r: bool(r.get("stop_between")),
                  v_min_window=lambda r: _bin(r.get("v_min_window"), (0.5, 3.0, 6.0)),
                  lane_change=lambda r: bool(r.get("lane_change")),
                  junction_x=lambda r: _bin(r.get("junction_x"), (150.0, 200.0, 250.0)))
    out = {}
    for c in cells:
        for arm in arms:
            for mp, prb in K.COMBOS:
                sel = K.select(recs, c[0], c[1], arm, mp, prb)
                for stat in ("prof", "match", "w_prof", "w_match", "c_prof", "c_match"):
                    res = _residuals(sel, stat)
                    for sname, f in strata.items():
                        grp = {}
                        for r, e in res:
                            grp.setdefault(str(f(r)), []).append(e)
                        out[f"{key(c)}|{arm}|{mp}|{prb}|{stat}|{sname}"] = {
                            g: dict(bias=float(np.mean(v)), sigma=K.sigma(np.array(v)), n=len(v)) for g, v in grp.items()}
    return out


def _bin(v, edges):
    if v is None or not np.isfinite(v):
        return "n/a"
    i = int(np.searchsorted(edges, v, side="right"))
    lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
    hi = "inf" if i == len(edges) else f"{edges[i]:g}"
    return f"[{lo},{hi})"


def _residuals(sel, stat):
    out = []
    for s in sorted({r["site"] for r in sel}):
        ref = [r[stat] for r in sel if r["site"] == s and r["role"] == "ref" and K.included(r) and np.isfinite(r.get(stat, np.nan))]
        if len(ref) < K.MIN_REF:
            continue
        b = float(np.mean(ref))
        out += [(r, r[stat] - b) for r in sel if r["site"] == s and r["role"] == "test" and K.included(r)
                and np.isfinite(r.get(stat, np.nan))]
    return out


def d0_3(recs):
    """D0.3: share of the |g| change covered by the ±4 s window (E-a), and next to it the E-f core share of beacons
    (c_n / n_beacons) with the bias and sigma of c_prof / c_match against those of w_prof / w_match, per speed class
    (test passes; gen_on; free-road survey calibration)."""
    out = {}
    for v0 in sorted({r["v0"] for r in recs}):
        sel = [r for r in recs if r["v0"] == v0 and r["arm"] == "gen_on" and r["mapping"] == "e1" and r["prb"] == K.PRBS[0]
               and r.get("kin", "sumo") == "sumo" and r.get("beta", 0.0) == 0.0]
        tst = [r for r in sel if r["role"] == "test"]
        x = [r["g_share_window"] for r in tst if np.isfinite(r.get("g_share_window", np.nan))]
        cs = [r["c_n"] / r["n_beacons"] for r in tst if r.get("n_beacons")]
        stats = {}
        for st in ("w_prof", "w_match", "c_prof", "c_match"):
            e = np.array([e for _, e in _residuals(sel, st)], float)
            stats[st] = dict(bias=float(np.mean(e)) if e.size else np.nan, sigma=K.sigma(e) if e.size > 1 else np.nan,
                             n=int(e.size))
        out[f"{v0:g}"] = dict(mean=float(np.mean(x)) if x else np.nan, n=len(x),
                              e_f_core_share=dict(mean=float(np.mean(cs)) if cs else np.nan, n=len(cs)), statistics=stats)
    return out


def d0_4(recs, cal):
    """D0.4: the RSU-only low-speed-run (stop) detector of E-d / E-db against stop_between (sensitivity, specificity),
    test passes, gen_on, per variant, under the pre-specified span and site scale; with k_stop the implied speed level
    v = k_stop x rms_ref_med x λ / |g| at |g| = 1 and 2 (median over the sites) and the flagged share of the non-stop
    passes by minimum speed in the window (the 'steady slow crawl gets no entry' class)."""
    from fdoa.geometry import LAMBDA
    out = {}
    for mp, prb in K.COMBOS:
        sel = [r for r in recs if r["arm"] == "gen_on" and r["mapping"] == mp and r["prb"] == prb]
        ann = CS.annotate_rms_ref_med(sel)
        rs = [r for r in ann if r["role"] == "test" and r.get("kin", "sumo") == "sumo"
              and not np.isnan(r.get("stop_abs", np.nan))]                               # +inf: no admissible run
        det = np.array([CS.stop_detected(r, cal) for r in rs], bool)
        tru = np.array([bool(r.get("stop_between")) for r in rs], bool)
        scales = sorted({r["rms_ref_med"] for r in rs if np.isfinite(r["rms_ref_med"])})
        slow = {}
        for r, d in zip(rs, det):
            if not r.get("stop_between"):
                slow.setdefault(_bin(r.get("v_min_window"), (0.5, 3.0, 6.0)), []).append(bool(d))
        out[f"{mp}/{prb}"] = dict(sensitivity=K._div((det & tru).sum(), tru.sum()),
                                  specificity=K._div((~det & ~tru).sum(), (~tru).sum()), n=int(det.size),
                                  k_stop=cal["k_stop"],
                                  implied_speed_mps={f"g={g:g}": float(np.median(scales) * cal["k_stop"] * LAMBDA / g)
                                                     if scales else np.nan for g in (1.0, 2.0)},
                                  nonstop_flagged_by_v_min={b: dict(share=float(np.mean(v)), n=len(v)) for b, v in slow.items()})
    return out


def d0_5(u_recs, cells=URBAN):
    """D0.5: the surrogate traffic survey (Dev-2 only) against the free-road survey: R8 (ii), R8 (iii) Spearman and the
    per-site β̂_s, readings T and F."""
    from pipeline import stage3_extra_checks as V
    F, T = CS.reading_F(u_recs, dev2=True), CS.reading_T(u_recs)
    beta = {}
    for name, R in (("F", F), ("T", T)):
        for c in cells:
            for mp, prb in K.COMBOS:
                Tt, _ = tables(R, c[0], c[1], "gen_on", mp, prb)
                beta[f"{name}|{key(c)}|{mp}|{prb}"] = {str(s): Tt[s]["b_ref"] for s in Tt}
    return dict(R8_F=V.qual_validity(F, list(cells), {}), R8_T=V.qual_validity(T, list(cells), {}), beta_s=beta)


def d0_6(recs):
    """D0.6 (descriptive): the per-head power drop on P3 (L-B; also its 5-beacon running median) for the
    survey passes and the blocked / unblocked test passes, and ρ̂ (rho_fit) of the P3-15 passes against the site survey's
    ρ̂ per lane (rho_dev, the L-B′ quantity; and per TRUE lane the median rho_fit - survey ρ̂ of that lane, which D0.6 may
    use), gen_on, per variant."""
    out = {}
    for mp, prb in K.COMBOS:
        sel = [r for r in recs if r["cell"] == "P3" and r["arm"] == "gen_on" and r["mapping"] == mp and r["prb"] == prb]
        sel = CS.annotate_rho_dev(sel)
        grp = dict(ref=[r for r in sel if r["role"] == "ref"],
                   test_blocked=[r for r in sel if r["role"] == "test" and r.get("blocked_share", 0) > 0],
                   test_unblocked=[r for r in sel if r["role"] == "test" and r.get("blocked_share", 0) == 0])
        q = lambda v, f, p: float(np.quantile(v, p)) if v else np.nan
        rep = {}
        for g, v in grp.items():
            rep[g] = {}
            for f in ("blk_drop_max_db", "blk_drop_med5_db", "rho_dev"):
                x = [r[f] for r in v if np.isfinite(r.get(f, np.nan))]
                rep[g][f] = dict(q50=q(x, f, 0.5), q99=q(x, f, 0.99), n=len(x))
        surv = {}
        for r in sel:
            if CS.is_survey(r) and np.isfinite(r.get("rho_fit", np.nan)):
                surv.setdefault((r["site"], r["v0"], int(round(-r["y_mean"] / K.lane_width(r)))), []).append(r["rho_fit"])
        med = {k: float(np.median(v)) for k, v in surv.items()}
        lanes = {}
        for r in sel:
            if r["role"] != "test" or not np.isfinite(r.get("rho_fit", np.nan)):
                continue
            ln = int(round(-r.get("y_mean", 0.0) / K.lane_width(r)))           # true lane: descriptive only
            m = med.get((r["site"], r["v0"], ln))
            if m is not None:
                lanes.setdefault(str(ln), []).append(r["rho_fit"] - m)
        rep["rho_minus_survey_by_true_lane"] = {ln: dict(median=float(np.median(v)), n=len(v)) for ln, v in lanes.items()}
        out[f"{mp}/{prb}"] = rep
    return out
