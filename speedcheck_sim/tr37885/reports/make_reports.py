"""Report generators of the speed-check simulation: Table A, Table B, the figures fig_C, fig_D, fig_E, fig_S1 and the flat
number file numbers_<tag>.json that the macro generator (reports/number_macros.py) turns into LaTeX macros. Reads only
stage-3 / adapter / in-house outputs (no data is computed here except the error CDFs of fig_S1 from the stage-2 records).
Every value printed is a number from those files; nothing is typed by hand. A required input that is missing is an error
for the primary tag and a 'missing' entry for development tags.
Usage: python reports/make_reports.py --tag primary [--results DIR] [--out DIR]"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import stage3_criteria as K  # noqa: E402
from pipeline.provenance import provenance  # noqa: E402

C = K.C
CELLS = ["P1-10", "P1-15", "P2-20", "P2-30", "P2-38", "P3-15"]
NAMES = {"P1": "urban LOS", "P2": "highway LOS", "P3": "urban, blocked crossings (NLOSv)"}
MISSING = []


def need(path, strict):
    if os.path.exists(path):
        return json.load(open(path))
    if strict:
        raise RuntimeError(f"required input missing: {path}")
    MISSING.append(path)
    return None


def f(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def headline_row(cr, key):
    c, v = key.split("-")[0], float(key.split("-")[1])
    for h in cr.get("headline", []):
        if h["cell"] == c and h["v0"] == v:
            return h
    return None


def table_a(cr, nums):
    rows = []
    sc, opt = cr["scope"]["cells"], cr.get("optimism_governing", {})
    qv = cr.get("qualification_validity", {})
    inc = (cr.get("incomplete_run") or {}).get("labels") or {}
    for key in CELLS:
        h = headline_row(cr, key)
        if h is None:
            continue
        s, o = sc.get(key, {}), opt.get(key, {})
        gv = (h["worst_sigma_qualified_gated"] or [None, {}])
        var = [r for r in h["variants"] if gv[1] and r["mapping"] == gv[1].get("mapping") and r["prb"] == gv[1].get("prb")]
        r0 = var[0] if var else (h["variants"][0] if h["variants"] else {})
        hf = h.get("headline_final_qualified_gated", {})
        ie = h.get("inhouse_equalK") or {}
        wq = lambda x: (x or [None])[0]
        q99_both = max([v for v in (wq(h.get("worst_q99_qualified_gated")), wq(ie.get("worst_q99"))) if v is not None], default=None)
        oow_both = max([v for v in (wq(h.get("worst_oow_qualified_gated")), wq(ie.get("worst_oow"))) if v is not None], default=None)
        og = (cr.get("over_governing_r8iv") or {}).get(key) or {}
        kih = [r["n_qual_inhouse"] for r in (cr.get("optimism") or []) if r and f"{r['cell']}-{r['v0']:g}" == key
               and r.get("n_qual_inhouse") is not None]       # our channel model's qualified count, worst variant
        row = dict(cell=key, type=NAMES[key.split("-")[0]], k=s.get("k"), k_inhouse=min(kih) if kih else None,
                   n=s.get("n"), cp_lo=(s.get("cp95") or [None, None])[0],
                   cp_hi=(s.get("cp95") or [None, None])[1], rating=s.get("rating"),
                   sigma_g1=gv[0], sigma_g1_ci_hi=(h.get("worst_sigma_ci_upper_qualified_gated") or [None])[0],
                   sigma_e1=(hf.get("e1_pct") / 100 if hf.get("e1_pct") is not None else None),
                   sigma_inhouse_equalK=(hf.get("inhouse_equalK_pct") / 100 if hf.get("inhouse_equalK_pct") is not None else None),
                   governing_source=hf.get("source"),
                   sigma_governing=(hf.get("value_pct") / 100 if hf.get("value_pct") is not None else gv[0]),
                   q99=q99_both, oow=oow_both,           # worst variant, worse of both arms
                   gate_discard=(h.get("worst_gate_false_discard") or [None])[0],
                   entry_yield=r0.get("entry_yield"), bias_uncal=(r0.get("bias_uncalibrated") or {}).get("mean"),
                   bias_cal=(r0.get("bias_calibrated") or {}).get("mean"),
                   optimism=o.get("verdict"), R=o.get("R"), p_holm=o.get("p_holm"),
                   s_place=h.get("s_place_factor"), label=(inc.get(key) or {}).get("label"),
                   sigma_unqualified=((qv.get(key) or {}).get("at_governing") or {}).get("sigma_unqualified"),
                   qualified_over_margin=(((qv.get(key) or {}).get("at_governing") or {}).get("qualified_over_margin") or {}).get("share"),
                   qualified_over_margin_k=(((qv.get(key) or {}).get("at_governing") or {}).get("qualified_over_margin") or {}).get("k"),
                   qualified_over_margin_cp_lo=((((qv.get(key) or {}).get("at_governing") or {}).get("qualified_over_margin") or {}).get("cp95") or [None])[0],
                   qualified_over_margin_cp_hi=((((qv.get(key) or {}).get("at_governing") or {}).get("qualified_over_margin") or {}).get("cp95") or [None, None])[1],
                   over_governing_k=og.get("k"), over_governing_share=og.get("share"),
                   over_governing_cp_lo=(og.get("cp95") or [None])[0], over_governing_cp_hi=(og.get("cp95") or [None, None])[1],
                   spearman=((qv.get(key) or {}).get("at_governing") or {}).get("spearman_sdref_vs_test_sigma"))
        rows.append(row)
        for k_, v_ in row.items():
            if isinstance(v_, (int, float)) and v_ is not None:
                nums[f"A|{key}|{k_}"] = v_
    p3 = cr.get("p3_blocked")
    if p3:
        nums["A|P3-15|blocked_share_mean"], nums["A|P3-15|blocked_any"] = p3["mean"], p3["share_any"]
    v2v = cr.get("v2v_reading")
    if v2v:
        for k_ in ("w_v2v", "entry_yield_unblocked_P1_15", "entry_yield_blocked_P3_15", "entry_yield_v2v",
                   "entry_yield_blocked_majority", "factor_vs_unblocked", "n_blocked_majority"):
            nums[f"A|V2V|{k_}"] = v2v.get(k_)
    # per-type keys of the abstract skeleton: k min / max and n over the type's cells; the
    # governing sigma min / max over the cells with k >= 1
    for t, cells in (("P1", ["P1-10", "P1-15"]), ("P2", ["P2-20", "P2-30", "P2-38"]), ("P3", ["P3-15"])):
        rr = [r for r in rows if r["cell"] in cells]
        if rr:
            ks = [r["k"] for r in rr if r["k"] is not None]
            nums[f"A|{t}|k_min"], nums[f"A|{t}|k_max"] = (min(ks), max(ks)) if ks else (None, None)
            nums[f"A|{t}|n"] = max(r["n"] for r in rr if r["n"] is not None)
    sg = [r["sigma_governing"] for r in rows if r["k"] and r["sigma_governing"] is not None and r["cell"][:2] in ("P1", "P2")]
    if sg:
        nums["A|qual|sigma_min"], nums["A|qual|sigma_max"] = min(sg), max(sg)
    qk = cr.get("qualification_vs_k") or {}
    for name, v in (qk.get("k_realised") or {}).items():   # the type's realised K
        if v is not None:
            nums[f"K|{name}|median"] = v
    text_keys(cr, nums)
    return rows


def text_keys(cr, nums):
    """Number keys for brackets of the pre-specified texts whose values stage 3 already computes but which have no
    key of their own. Copies only; no statistic is computed here:
      A|<cell>|q99_ratio            T4: the cell's q99 |e| ratio TR 37.885 / in-house, worst variant, S-place factor applied,
                                    as stage3_criteria.bridge stores it (q99_ratio_worst) and select_texts reads it for T4|<cell>
      A|<cell>|R_ci_lo, |R_ci_hi    T4 'undetermined' [CI]: the widened 95 % interval of R (stage3_criteria.optimism_cell
                                    R_ci_widened) of the variant that governs the cell's verdict (optimism_governing)
      A|<cell>|optimism_reason      T4 'not run' [reason]: the reason stage3_criteria.optimism_verdicts stores, without its
                                    leading 'not run (' ... ')'
      A|<cell>|sigma_g1_ci_lo       T2 [95 % CI a-b] lower end: qualified_gated ci[0] of the variant Table A prints (the
                                    worst-sigma variant; the upper end stays A|<cell>|sigma_g1_ci_hi, the worst upper end
                                    over the variants, so the printed interval is never narrower than that variant's)
      A|<cell>|bias_uncal_ci_lo/hi, A|<cell>|bias_cal_ci_lo/hi   T2 [CI] of the mean error before / after calibration
                                    (stage3_criteria.cell_quantities bias_*.ci) at the same variant
      A|bridge|<type>|case, |label  T5 [bridge label]: stage3_criteria.bridge case and label per site type (P1, P2, P3)"""
    for t, d in (cr.get("bridge") or {}).items():
        d = d or {}
        code = {"urban LOS": "P1", "highway LOS": "P2"}.get(t, "P3")
        for c, q in (d.get("q99_ratio_worst") or {}).items():
            if isinstance(q, (int, float)):
                nums[f"A|{c}|q99_ratio"] = q
        for k_ in ("case", "label"):
            if d.get(k_) is not None:
                nums[f"A|bridge|{code}|{k_}"] = d[k_]
    opt = [r for r in (cr.get("optimism") or []) if r]
    for c, g in (cr.get("optimism_governing") or {}).items():
        r = [x for x in opt if f"{x['cell']}-{x['v0']:g}" == c and x.get("mapping") == g.get("mapping")
             and x.get("prb") == g.get("prb")]
        ci_r = (r[0].get("R_ci_widened") if r else None) or [None, None]
        if isinstance(ci_r[0], (int, float)) and isinstance(ci_r[1], (int, float)):
            nums[f"A|{c}|R_ci_lo"], nums[f"A|{c}|R_ci_hi"] = ci_r
        if g.get("verdict") == "not run" and g.get("reason"):
            why = g["reason"]
            nums[f"A|{c}|optimism_reason"] = (why[len("not run ("):-1] if why.startswith("not run (") and why.endswith(")")
                                              else why)
    for key in CELLS:
        h = headline_row(cr, key)
        if h is None:
            continue
        gv = (h.get("worst_sigma_qualified_gated") or [None, {}])[1] or {}
        var = [r for r in h["variants"] if gv and r["mapping"] == gv.get("mapping") and r["prb"] == gv.get("prb")]
        r0 = var[0] if var else (h["variants"][0] if h["variants"] else {})   # the variant table_a prints
        lo = ((r0.get("qualified_gated") or {}).get("ci") or [None])[0]
        if isinstance(lo, (int, float)):
            nums[f"A|{key}|sigma_g1_ci_lo"] = lo
        for src, dst in (("bias_uncalibrated", "bias_uncal"), ("bias_calibrated", "bias_cal")):
            b = (r0.get(src) or {}).get("ci") or [None, None]
            if isinstance(b[0], (int, float)) and isinstance(b[1], (int, float)):
                nums[f"A|{key}|{dst}_ci_lo"], nums[f"A|{key}|{dst}_ci_hi"] = b


INHOUSE_E2 = os.path.join(ER, "..", "inhouse", "results", "decisions_L8_full.json")   # in-house decision simulation, 8 levels
TB_METRICS = ("punitive_per_year", "punitive_ub95", "notice_per_year", "entries_per_epoch_mean",
              "crossings_per_epoch_mean", "low_m_share", "escalation_per_year", "inspections_per_year",
              "punitive_union_per_year", "punitive_union_ub95")
ESC_FALLBACK_UB95 = 1e-3      # the escalation is adopted only if the honest bound <= 1e-3
# R3 honest per-vehicle-year rate from its in-house ladder (consistency simulation, traffic standardisation, level
# groups, q3; nominal site shift; weighted mean over sites), per pass rate lambda; added to the decision-simulation
# union as a union bound
INHOUSE_E3 = os.path.join(ER, "..", "inhouse", "results", "consistency_analysis_full.json")
E3_CONFIG, E3_FIELD = "trf|level|q3.0", "shift0.0_weighted_mean_ladder"


class LadderError(KeyError):
    """A pass rate lambda for which the in-house R3 ladder has no entry (the ladder is evaluated at M = 5, 15, 40)."""


def r3_honest_rate(lam):
    """The R3 honest per-vehicle-year rate read from the in-house ladder at a fixed M = lambda: a stated fixed-M
    approximation (item 5 and T5;  (ii), ): the decision simulation's epochs carry Poisson(lambda)
    entries and an epoch with fewer than five entries draws no R3 flag, so at lambda = 5 the reading is conservative."""
    d = json.load(open(INHOUSE_E3))[E3_CONFIG]
    key = str(int(float(lam)))
    if key not in d:
        raise LadderError(f"R3 in-house ladder {E3_CONFIG} has no entry for lambda = {lam} (entries: {sorted(d)})")
    return float(d[key][E3_FIELD])


# Table A footnote, printed under Table A (wording only)
TABLE_A_NOTES = {
    "entry_yield": "Entry yield before the loss at burst boundaries; the decision-level results (Table B) apply it as "
                   "1 - 110 m / (v * 40 s), i.e. they require the whole evaluation window inside one burst, as the "
                   "per-pass window rule does."}
# Table B stratum notes, from the adapter's tables_<tag>.json (stratum_status, strata)
STRATUM_NOTE = {
    "covered_short": "Decision-level rows of the {s} stratum use {n_eval} gated test crossings per site and speed "
                     "class, fewer than the planned {min_eval} in {below} of {rows} rows; every row is truncated to "
                     "that number in track order, so each row's empirical error pool is correspondingly smaller.",
    "not_covered_floor": "The {s} stratum is not covered at the decision level (fewer than {floor} gated test crossings "
                         "per site and speed class); its Table B rows are not reported, and the suburban rows take the "
                         "other stratum alone.",
    "not_covered_unqualified": "The {s} stratum is not covered at the decision level (no site and speed class "
                               "qualified under the decision rules' own qualification); its Table B rows are not "
                               "reported, and the suburban rows take the other stratum alone.",
    "not_covered_c5": "The {s} stratum is not covered at the decision level (every speed class failed the kinematics "
                      "check); its Table B rows are not reported, and the suburban rows take the other stratum alone.",
    "no_records": "The {s} stratum has no decision-level records (its family completed no site); its Table B rows are not "
                  "reported, and the suburban rows take the other stratum alone."}


def stratum_rows(tables, e2, nums):
    """B|stratum|<s>|status for urban and highway (covered | covered_short | not_covered_floor | not_covered_unqualified
    | not_covered_c5 | no_records), and for a covered stratum B|stratum|<s>|n_eval, |rows_below_min_eval, |rows,
    |truncation; B|uncovered|<s>|<profile> = the C5 uncovered traffic share. Returns the printed notes."""
    out = []
    if not tables:
        return out
    st, strata = tables.get("stratum_status") or {}, tables.get("strata") or {}
    MIN_EVAL, FLOOR = C["conventions"]["min_eval_e2"], C["conventions"].get("min_eval_e2_floor")
    for s in ("urban", "highway"):
        status = st.get(s, "no_records")
        nums[f"B|stratum|{s}|status"] = status
        d = strata.get(s) if isinstance(strata.get(s), dict) else {}
        for k in ("n_eval", "rows_below_min_eval", "rows", "truncation"):
            if d.get(k) is not None:
                nums[f"B|stratum|{s}|{k}"] = d[k]
        for vt, x in ((d.get("uncovered_traffic_share") or {}) if d else {}).items():
            nums[f"B|uncovered|{s}|{vt}"] = x
        if status in STRATUM_NOTE and status != "covered":
            out.append(STRATUM_NOTE[status].format(s=s, n_eval=d.get("n_eval"), min_eval=MIN_EVAL,
                                                   below=d.get("rows_below_min_eval"), rows=d.get("rows"), floor=FLOOR))
    # section 6.1: if neither stratum is covered, the two notes are replaced by one sentence
    if all(st.get(s, "no_records") not in ("covered", "covered_short") for s in ("urban", "highway")):
        out = ["Neither the urban nor the highway stratum is covered at the decision level, so Table B reports no "
               "decision-level rows."]
    return out


# The four matching-simulation keys under the worse-of rule: min for the median number of candidates, max for
# P(unique); configuration 8 levels + z_q 3 values, lambda 5, k = 5 and k = 8, run.e2_vehicles policies; z from the
# in-house channel in every source
INHOUSE_E5 = os.path.join(ER, "..", "inhouse", "results", "matching_trf.json")
E5_CONF = "L8|z q3.0|lam5"
E5_WORST = {"median_candidates": min, "p_unique": max}


def e5_matching_keys(e5, nums, strict=False):
    rows = []
    if os.path.exists(INHOUSE_E5):
        d = json.load(open(INHOUSE_E5))
        if E5_CONF in d:
            rows.append(("in-house", d[E5_CONF]))
    elif strict:
        raise RuntimeError(f"required input missing: {INHOUSE_E5}")
    for s in ("urban", "highway"):
        rep = ((e5 or {}).get(s) or {}).get("e5") or {}
        if E5_CONF in rep:
            rows.append((f"generator {s}", rep[E5_CONF]))
    for k in ("k5", "k8"):
        for m, worst in E5_WORST.items():
            vals = [(e[k][m], src) for src, e in rows if k in e and isinstance(e[k].get(m), (int, float))]
            for v, src in vals:
                nums[f"E5|{src.replace(' ', '_')}|lam5|{k}|{m}"] = v
            if vals:
                v, src = worst(vals, key=lambda t: t[0])
                nums[f"E5|governing|lam5|{k}|{m}"] = v
                nums[f"E5|governing|lam5|{k}|{m}|source"] = src
    nums["E5|label"] = "z from the in-house channel"
    nums["E5|policies"] = C["run"]["e2_vehicles"]


# The R3 condition at M = 5 under the TR 37.885 honest share of z_q = -1 (descriptive, never gating): share = the
# largest per-site median over the C4-light cells (the worst for honest vehicles); the condition share^5 <= alpha
# (alpha = 1e-3 / 12 / 5 of the consistency analysis; alpha^(1/5) = alpha_pow_1_5 of consistency_min_crossings.json)
R3_MIN_M = os.path.join(ER, "..", "inhouse", "results", "consistency_min_crossings.json")


def r3_m5(nums):
    med = [v for k, v in nums.items() if k.startswith("B|zq_minus1_per_site|") and k.endswith("|median")
           and isinstance(v, (int, float)) and np.isfinite(v)]
    if not med or not os.path.exists(R3_MIN_M):
        return
    d = json.load(open(R3_MIN_M))
    p = max(med)
    nums["B|zq_minus1|share"] = p
    nums["B|zq_minus1|p5_le_alpha"] = int(p ** 5 <= d["alpha"])


def table_b(e2, nums, strict=False):
    """Table B (T1 / T5; d): per the decision simulation row the TR 37.885 value
    (B|<sec>|<key>|<metric>), the in-house value of the same row (B|<sec>|<key>|inhouse|<metric>: the in-house decision simulation (monitor variant) run
    on its own site-K population, TR 38.901 UMi N(9, 5^2) dB, not at equal K; labelled so in the table) and the worse of
    the two per metric direction (adapter direction; B|<sec>|<key>|governing|<metric>; descriptive metrics have no
    governing value). A row without an in-house value governs with its TR 37.885 value, flagged;
    a primary run refuses an honest row without an in-house value. Per honest row the bound = the decision-simulation
    value (escalation union, or R2
    alone) + the R3 honest rate of its in-house ladder at the row's lambda (a union bound over the three rules). The
    escalation is adopted ("adopted") if the worst bound with it is <= 1e-3, else "fallback"; "not evaluated" without
    an honest row (no U / H stratum covered). B|esc|status / B|esc|adopted; B|honest|governing|all_union_ub95 = the
    worst bound with the escalation (printed in both evaluated branches). The abstract keys
    B|honest|governing|punitive_per_year and |punitive_ub95 are the worst bounds of the adopted rule set.
    B|honest|governing|r2_* / union_* keep the decision-simulation-only values. sens_tyre rows are reported over both channels
    (B|sens_tyre|max|..., with its source), never tested."""
    rows = []
    if not e2:
        return rows
    from pipeline.decision_adapter import direction
    from pipeline.provenance import sha256_file
    ih = json.load(open(INHOUSE_E2)) if os.path.exists(INHOUSE_E2) else {}
    if ih:
        nums["B|inhouse_source_sha256"] = sha256_file(INHOUSE_E2)[:16]
    worst = {}
    for sec in ("honest", "power", "badsites", "margin"):
        for key, r in (e2.get("rows", {}).get(sec) or {}).items():
            if not isinstance(r, dict):
                continue
            ri = (ih.get(sec) or {}).get(key) or {}
            if not ri:
                nums["B|inhouse_missing_rows"] = nums.get("B|inhouse_missing_rows", 0) + 1
                if strict and sec == "honest":
                    raise RuntimeError(f"Table B: honest row {key} has no in-house value ({INHOUSE_E2})")
            row = dict(section=sec, key=key, source=r.get("source"), inhouse_source="in-house decision simulation, own K population (not equal K)",
                       governing_source="worse of both" if ri else "TR 37.885 only")
            for m in TB_METRICS:
                g, i, dr = r.get(m), ri.get(m), direction(sec, key, m)
                row[m], row[f"inhouse_{m}"] = g, i
                both = [v for v in (g, i) if isinstance(v, (int, float)) and np.isfinite(v)]
                row[f"governing_{m}"] = ((max(both) if dr == "high" else min(both)) if len(both) == 2 else
                                         (both[0] if both else None)) if dr else None
                for lab, v in ((None, g), ("inhouse", i), ("governing", row[f"governing_{m}"])):
                    if isinstance(v, (int, float)):
                        nums[f"B|{sec}|{key}|{lab + '|' if lab else ''}{m}"] = v
                if (sec == "honest" and m in ("punitive_per_year", "punitive_ub95", "punitive_union_per_year",
                                              "punitive_union_ub95") and row[f"governing_{m}"] is not None):
                    worst[m] = max(worst.get(m, -np.inf), row[f"governing_{m}"])
            rows.append(row)
    for src, rows_t in (("G1", e2.get("rows", {}).get("sens_tyre") or {}), ("in-house", ih.get("sens_tyre") or {})):
        for key, r in rows_t.items():                     # reported, never tested; worse of both channels
            for m in ("escalation_per_year", "inspections_per_year", "punitive_union_per_year"):
                if isinstance(r.get(m), (int, float)):
                    nums[f"B|sens_tyre|{key}|{'inhouse|' if src == 'in-house' else ''}{m}"] = r[m]
                    if r[m] > nums.get(f"B|sens_tyre|max|{m}", -np.inf):
                        nums[f"B|sens_tyre|max|{m}"], nums[f"B|sens_tyre|max|{m}|source"] = r[m], f"{src} {key}"
    for m in ("punitive_per_year", "punitive_ub95"):
        if m in worst:
            nums[f"B|honest|governing|r2_{m}"] = worst[m]
        um = m.replace("punitive_", "punitive_union_")
        if um in worst:
            nums[f"B|honest|governing|union_{m}"] = worst[um]
    # R3 as a union bound: per honest row, bound = the decision-simulation value of the rule set + the R3
    # in-house honest rate at the row's lambda; the escalation is adopted if the worst bound with it is <= 1e-3.
    # No honest row (no U / H stratum covered): "not evaluated". A honest row without the union: a primary run
    # stops, a development run treats it as not evaluated.
    hon = [r for r in rows if r["section"] == "honest"]
    missing = [r["key"] for r in hon if r.get("governing_punitive_union_ub95") is None]
    if missing and strict:
        raise RuntimeError(f"Table B: honest rows without the escalation union: {missing}")
    if not hon or missing:
        nums["B|esc|status"] = "not evaluated"
    else:
        r3 = {r["key"]: r3_honest_rate(r["key"].split("|")[-1]) for r in hon}
        nums["B|honest|r3_inhouse|max"] = max(r3.values())
        bound = lambda m: max(r[f"governing_{m}"] + r3[r["key"]] for r in hon)
        adopted = bound("punitive_union_ub95") <= ESC_FALLBACK_UB95
        nums["B|esc|status"] = "adopted" if adopted else "fallback"
        nums["B|honest|governing|all_union_ub95"] = bound("punitive_union_ub95")     # printed in both branches
        for m in ("punitive_per_year", "punitive_ub95"):
            nums[f"B|honest|governing|{m}"] = bound(m.replace("punitive_", "punitive_union_") if adopted else m)
    nums["B|esc|adopted"] = int(nums["B|esc|status"] == "adopted")
    for st, d in (e2.get("lambda_eff") or {}).items():
        for vt, r in d["profiles"].items():
            nums[f"B|lambda_eff|{st}|{vt}|q"] = r["q"]
            nums[f"B|lambda_eff|{st}|{vt}|q_tableA"] = r.get("q_tableA")
            for lam, v in r["lambda_eff"].items():
                nums[f"B|lambda_eff|{st}|{vt}|{lam}"] = v
    return rows


def zq_rows(cr, nums):
    for key, d in (cr.get("c4_light") or {}).items():
        g = [r for r in d["variants"] if r["arm"] == "gen_on" and r.get("p_zq_minus1_honest_per_site")]
        if g:
            w = min(g, key=lambda r: r["Z"] if np.isfinite(r["Z"]) else np.inf)
            ps = w["p_zq_minus1_honest_per_site"]
            nums[f"B|zq_minus1_per_site|{key}|median"] = ps["median"]
            if ps.get("range"):
                nums[f"B|zq_minus1_per_site|{key}|min"], nums[f"B|zq_minus1_per_site|{key}|max"] = ps["range"]
            nums[f"D|{key}|Z"] = d["Z_worst"]


def write_table(rows, path, notes=()):
    if notes:                                           # pre-specified footnotes, printed under the table
        with open(path + "_notes.txt", "w") as fh:
            fh.write("\n".join(notes) + "\n")
    if not rows:
        return
    keys = list(rows[0])
    with open(path + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    with open(path + ".md", "w") as fh:
        fh.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(f(v, 4) if isinstance(v, float) else str(v) for v in r.values()) + " |\n")
        for i, n in enumerate(notes):
            fh.write(("\n" if i == 0 else "") + f"Note {i + 1}. {n}\n")


def fig_c(cr, k4, path, nums):
    """Fig. C: in-house per-beacon RMS against SNR per K (and K = inf
    with nominal impairments if run), the single-tone CRLB, the TR 37.885 RMS with q10-q90 bands at the governing PRB,
    the histogram of the TR 37.885 operating SNR (twin axis) and the decoding line at each overlay's governing PRB (S22), with
    the share of the overlay's beacons below it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if k4:
        for Kd in k4["K_db"]:
            for imp, ls in (("off", "-"), ("nominal", ":")):
                rr = [r for r in k4["rows"] if r["K_db"] == Kd and r.get("impairments", "off") == imp]
                if rr:
                    ax.semilogy([r["snr_db"] for r in rr], [r["rms_hz"] for r in rr], ls, marker="o", ms=2.5, lw=1.1,
                                label=f"in-house K = {'inf' if Kd is None else f'{Kd:g} dB'}" + (" (impairments)" if imp != "off" else ""))
        rr = [r for r in k4["rows"] if r["K_db"] == k4["K_db"][0] and r.get("impairments", "off") == "off"]
        ax.semilogy([r["snr_db"] for r in rr], [r["crlb_hz"] for r in rr], "k--", lw=1.2, label="single-tone CRLB (2 heads)")
        for r in k4["rows"]:
            nums[f"C|k4|K{r['K_db']}|snr{r['snr_db']}|{r.get('impairments', 'off')}|rms_hz"] = r["rms_hz"]
        r35 = sorted([r for r in k4["rows"] if r["K_db"] == 3.5 and r.get("impairments", "off") == "off"], key=lambda r: r["snr_db"])
        if r35:                                         # caption rule: first SNR within 10 % of the 45 dB value
            top = r35[-1]["rms_hz"]
            nums["C|k4|snr_floor_db"] = next(r["snr_db"] for r in r35 if r["rms_hz"] <= 1.1 * top)
    pbg = (cr or {}).get("per_beacon_governing") or {}
    ax2 = ax.twinx()
    for key in ("P1-15", "P2-30"):
        pb = pbg.get(key)
        if not pb:
            continue
        xs = sorted(float(b) for b in pb["by_snr"])
        rows = [pb["by_snr"][f"{int(x)}"] for x in xs]
        xc = [x + 2.5 for x in xs]
        ax.semilogy(xc, [r["rms"] for r in rows], "s-", ms=4, lw=2, label=f"TR 37.885 {NAMES[key[:2]]} {key[3:]} m/s RMS")   # no code name
        ax.fill_between(xc, [max(r["q10"], 1e-3) for r in rows], [r["q90"] for r in rows], alpha=0.15)
        hc, he = pb["snr_hist"]["counts"], pb["snr_hist"]["edges"]
        ax2.stairs(np.asarray(hc) / max(sum(hc), 1), he, alpha=0.35)
        if pb.get("decoding_db") is not None:           # the decoding line at this overlay's governing PRB
            dl = f"decoding line, {pb['prb']} PRB (TS 36.101, one-element head)"   # one legend entry and one
            seen = dict(zip(ax.get_legend_handles_labels()[1], ax.get_legend_handles_labels()[0]))   # colour per label
            ax.axvline(pb["decoding_db"], color=seen[dl].get_color() if dl in seen else ax.lines[-1].get_color(),
                       ls=":", lw=1, label=None if dl in seen else dl)
            nums[f"C|{key}|decoding_db"] = pb["decoding_db"]
            nums[f"A|{key}|below_decoding_pct"] = 100 * pb["below_decoding_share"]
            nums[f"A|{key}|snr_q01_db"] = pb["snr_q01_db"]
    ax2.set_ylabel("share of TR 37.885 beacons (histogram)")   # no code name
    ax.set_xlabel("raw-sample SNR per beacon, weaker head (dB)")
    ax.set_ylabel("per-beacon FDOA error (Hz)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def fig_e(cr, fe, path, nums):
    """Fig. E: (a) sigma against the realised in-band K: in-house per-site survey sd and test sigma (configured K
    = realised by construction) with the median curve, TR 37.885 per-site survey sd and test sigma (weaker head, governing PRB),
    the 6 % line; (b) the qualification probability: binned shares with Clopper-Pearson and the logistic fit
    (descriptive)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    if fe:
        for sp, lab, mk in ((None, "in-house", "o"), (-10.0, "in-house + facade", "D")):
            rr = [r for r in fe["rows"] if r["specular_db"] == sp]
            ax.plot([r["K_db"] for r in rr], [100 * r["sigma"] for r in rr], lw=1.2, label=f"{lab}: pooled test sigma")
            for r in rr:
                ks = [r["K_db"]] * len(r["site_points"])
                ax.scatter(ks, [100 * p["sigma_site"] for p in r["site_points"]], s=6, marker=mk, alpha=0.25, color="grey")
    qk = (cr or {}).get("qualification_vs_k")
    if qk:
        for cell, mk in (("P1", "^"), ("P2", "s"), ("P3", "v")):
            pp = [p for p in qk["points"] if p["cell"].startswith(cell)]
            ax.scatter([p["k_db"] for p in pp], [100 * p["sigma_test"] for p in pp], marker=mk, s=16,
                       label=f"G1 {NAMES[cell]} (test sigma)")
            ax.scatter([p["k_db"] for p in pp], [100 * p["sd_ref"] for p in pp], marker=mk, s=16, facecolors="none",
                       edgecolors="k", lw=0.6, label=f"G1 {NAMES[cell]} (survey sd)")      # hollow
        b = [x for x in qk["binned"] if x["n"]]
        xs = [0.5 * (max(x["k_lo"], -2) + min(x["k_hi"], 14)) for x in b]
        bx.errorbar(xs, [x["share"] for x in b], yerr=[[x["share"] - x["cp95"][0] for x in b], [x["cp95"][1] - x["share"] for x in b]],
                    fmt="o", capsize=3, label="G1 sites, qualified share (CP 95 %)")
        if qk.get("logistic"):
            kk = np.linspace(-2, 16, 100)
            bx.plot(kk, 1 / (1 + np.exp(-(qk["logistic"][0] + qk["logistic"][1] * kk))), lw=1.2, label="logistic fit (descriptive)")
        if qk.get("k50_db") is not None:
            nums["E|k50_db"] = qk["k50_db"]
    ax.axhline(6.0, color="grey", ls=":")
    ax.set_xlabel("realised in-band Ricean K, weaker head (dB)")
    ax.set_ylabel("per-crossing sigma (%)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    bx.set_xlabel("realised in-band Ricean K (dB)")
    bx.set_ylabel("P(site qualifies)")
    bx.set_ylim(-0.02, 1.02)
    bx.legend(fontsize=6)
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def erasure_rows(tag, cr, nums):
    """K5 erasure rows of Table A: entry yield and kept sigma at iid / Gilbert-Elliott 10 % / 30 % for the U
    and H cells, at the stratum's governing variant."""
    gov = (cr or {}).get("governing") or {}
    for nm in ("iid10", "ge10", "iid30", "ge30"):
        t = f"sens_erasure_{nm}" if tag == "primary" else f"{tag}_erasure_{nm}"
        recs, _ = K.load(["U", "H"], t)                 # a refusal propagates; the merge requires the tags
        for key in ("P1-10", "P1-15", "P2-20", "P2-30", "P2-38"):
            c, v = key.split("-")[0], float(key.split("-")[1])
            g = gov.get("urban" if c == "P1" else "highway") or {}
            q = K.cell_quantities(recs, c, v, g.get("mapping", "e1"), int(g.get("prb", 10))) if recs else None
            if q:
                nums[f"A|erasure|{nm}|{key}|entry_yield"] = q.get("entry_yield")
                nums[f"A|erasure|{nm}|{key}|sigma"] = q["qualified_gated"]["sigma"]


def design_record_numbers(nums):
    """reg|date, reg|bundle_sha256, reg|commit from the study-1 design record (config/study1_design_record.json)."""
    f = os.path.join(ER, "config", "study1_design_record.json")
    if os.path.exists(f):
        d = json.load(open(f))
        for k in ("date", "bundle_sha256", "commit"):
            if k in d:
                nums[f"reg|{k}"] = d[k]


def inhouse_zq(cr, nums, tag="primary"):
    """Table B rows of the honest per-site P(z_q = -1) of the in-house channel:
      B|zq_inhouse|<cell>|...      the comparator: the paired in-house arm of the TR 37.885 records, same estimator and same
                                   (in-sample) standardisation as TR 37.885 (criteria.c4_light; arm inhouse);
      B|zq_inhouse_oos|<cell>|...  second row, 'out-of-sample traffic standardisation': the in-house consistency-simulation honest z at
                                   the equal-K cell (stage3_extra_checks.inhouse_k_cell of the one realised-K function),
                                   traffic passes; median and range over the in-house sites."""
    from pipeline import stage3_extra_checks as XC
    for key, d in ((cr or {}).get("c4_light") or {}).items():
        g = [r for r in d["variants"] if r["arm"] == "gen_on" and np.isfinite(r["Z"])]
        w = min(g, key=lambda r: r["Z"]) if g else None
        ih = [r for r in d["variants"] if w and r["arm"] == "inhouse" and r["mapping"] == w["mapping"] and r["prb"] == w["prb"]]
        ps = (ih[0].get("p_zq_minus1_honest_per_site") or {}) if ih else {}
        if ps.get("values"):
            nums[f"B|zq_inhouse|{key}|median"] = ps["median"]
            nums[f"B|zq_inhouse|{key}|min"], nums[f"B|zq_inhouse|{key}|max"] = ps["range"]
    e3 = os.path.join(ER, "..", "inhouse", "results")
    if not os.path.exists(os.path.join(e3, "consistency_full.json")):
        return
    cells = json.load(open(os.path.join(e3, "consistency_full.json")))["cells"]
    Z = np.load(os.path.join(e3, "consistency_z_full.npz"))
    cut = K.PC["z_threshold"]
    gov = (cr or {}).get("governing") or {}
    for key, fam, st in (("P1-15", "U", "urban"), ("P2-30", "H", "highway")):
        cell, v0 = key.split("-")[0], float(key.split("-")[1])
        recs, _ = K.load([fam], tag)
        Ksel = XC.inhouse_k_cell(XC.realised_k_median(recs, cell, int((gov.get(st) or {}).get("prb", 10))))
        idx = [i for i, c in enumerate(cells) if c["v0"] == v0 and c["K_db"] == Ksel and c["specular_db"] is None]
        if not idx:
            continue
        z = np.asarray(Z[f"c{idx[0]}_honest_trf"], float)
        n_sites = len(cells[idx[0]]["sites"])
        per = [float(np.mean(x < -cut)) for x in z.reshape(n_sites, -1)]
        nums[f"B|zq_inhouse_oos|{key}|K_db"] = Ksel
        nums[f"B|zq_inhouse_oos|{key}|median"], nums[f"B|zq_inhouse_oos|{key}|min"], nums[f"B|zq_inhouse_oos|{key}|max"] = (
            float(np.median(per)), float(min(per)), float(max(per)))


def k7_numbers(results, tag, nums):
    ctag = C["criteria"]["C1"]["coverage_check"]["tag"] if tag == "primary" else f"{tag}_cov"
    f = os.path.join(results, f"coverage_check_{ctag}.json")
    if os.path.exists(f):
        d = json.load(open(f))
        nums["S|k7|c"] = d.get("c_calibrated")
        nums["S|k7|calibrated"] = int(bool(d.get("calibrated", True)))
        for pop, r in (d.get("populations") or {}).items():
            nums[f"S|k7|pop_{pop}|runs"] = r.get("runs_at_margin")
            nums[f"S|k7|pop_{pop}|size"] = r.get("size_at_margin")


def fig_d(cr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    rows = (cr or {}).get("c4_light") or {}
    labs = []
    for i, (key, d) in enumerate(sorted(rows.items())):
        # (presentation only) one fixed colour per arm, so every cell uses the colours of the legend
        for arm, mk, col, lab in (("gen_on", "o", "C0", "TR 37.885"), ("inhouse", "s", "C1", "our channel model")):
            z = [r["Z"] for r in d["variants"] if r["arm"] == arm and np.isfinite(r["Z"])]
            ax.scatter([i + (0.1 if arm == "inhouse" else -0.1)] * len(z), z, marker=mk, s=16, color=col,
                       label=lab if i == 0 else None)
        labs.append({"P1-15": "urban LOS, 15 m/s", "P2-30": "highway LOS, 30 m/s"}.get(key, key))
    ax.axhline(1.0, color="grey", ls=":")
    ax.set_xticks(range(len(labs)), labs)
    ax.set_ylabel("separation at Doppler scale \u221210 %\n(honest standard deviations)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def fig_s1(tag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs, _ = K.load(K.PRIMARY_FAMILIES, tag)
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for key in ("P1-15", "P2-30", "P3-15"):
        c, v = key.split("-")[0], float(key.split("-")[1])
        for arm, ls in (("gen_on", "-"), ("inhouse", "--")):
            T, _ = K.site_table(recs, c, v, arm, "e1", K.PRBS[0])
            e = np.sort(np.abs(K.pooled(T, sorted(T), True)))
            if e.size:
                ax.plot(100 * e, np.arange(1, e.size + 1) / e.size, ls, label=f"{key} {arm}")
    ax.set_xscale("log")
    ax.set_xlabel("|e| (%)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", default=K.RESULTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    strict = a.tag == "primary"
    out = a.out or os.path.join(a.results, "reports", a.tag)
    os.makedirs(out, exist_ok=True)
    cr = need(os.path.join(a.results, f"criteria_{a.tag}.json"), strict)
    adir = os.environ.get("SPEEDCHECK_ADAPTER_OUT", os.path.join(a.results, "adapter"))
    e2 = need(os.path.join(adir, f"e2_extracted_{a.tag}.json"), strict)
    itag = "sens_inhouse" if strict else a.tag
    k4 = need(os.path.join(a.results, f"inhouse_k4_{itag}.json"), strict)
    fe = need(os.path.join(a.results, f"inhouse_fige_{itag}.json"), strict)
    nums = {}
    if cr:
        write_table(table_a(cr, nums), os.path.join(out, "table_A"), notes=list(TABLE_A_NOTES.values()))
        zq_rows(cr, nums)
        fig_d(cr, os.path.join(out, "fig_D"))
        fig_s1(a.tag, os.path.join(out, "fig_S1"))
        for key, row in (cr.get("sensitivity_thresholds") or {}).items():
            for grp in ("gate", "sigma_q", "window", "z_cut", "snr_offset_db"):
                for lev, q in (row.get(grp) or {}).items():
                    for k_, v_ in (q or {}).items():
                        if isinstance(v_, (int, float)):
                            nums[f"S|{key}|{grp}|{lev}|{k_}"] = v_
    tables = need(os.path.join(adir, f"tables_{a.tag}.json"), strict)
    e5 = need(os.path.join(adir, f"e5_extracted_{a.tag}.json"), strict)
    b_notes = stratum_rows(tables, e2, nums)
    write_table(table_b(e2, nums, strict), os.path.join(out, "table_B"), notes=b_notes)
    e5_matching_keys(e5, nums, strict)
    r3_m5(nums)
    erasure_rows(a.tag, cr, nums)
    design_record_numbers(nums)
    inhouse_zq(cr, nums, a.tag)
    k7_numbers(a.results, a.tag, nums)
    fig_c(cr, k4, os.path.join(out, "fig_C"), nums)
    fig_e(cr, fe, os.path.join(out, "fig_E"), nums)
    json.dump(dict(tag=a.tag, numbers={k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in nums.items()},
                   notes=dict(table_A=list(TABLE_A_NOTES.values()), table_B=b_notes),
                   missing=MISSING, provenance=provenance(a.tag)),
              open(os.path.join(out, f"numbers_{a.tag}.json"), "w"), indent=1, default=float)
    print(out, f"{len(nums)} numbers", f"missing: {MISSING}" if MISSING else "")


if __name__ == "__main__":
    main()
