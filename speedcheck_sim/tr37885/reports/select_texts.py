"""Branch selection of the pre-specified outcome texts: every branch of every
template is chosen here by its pre-specified rule from the stage-3 and report outputs, never by hand, and written to
text_branches_<tag>.json plus LaTeX macros \\spb{<key>} (speedpath-branches.tex). Keys:
  T3|<type>            covered | partly covered | not covered | not run (compute)   (scope rule, type level: the text
                        branch; "covered at >= v m/s" selects "partly covered" with T3suffix|<type> = "at >= v m/s")
  T3full|<type>        the full type rating string, incomplete-run suffix included
  T3suffix|<type>      "at >= v m/s" | the incomplete-run suffix | ""
  T3cell|<cell>        the same per cell
  T3u|<cell>           borne_out (sigma_unqualified > 6 %) | conservative
  T3share|<cell>       none | some (qualified sites with test sigma > 9 %)
  T4|<cell>            not contradicted | tails | contradicted | undetermined | not run ('tails' per cell: the
                        cell's own q99 ratio > margin and the cell not contradicted)
  T6|<type>            separates (Z >= 1) | restricted (Z < 1) | not run
  fnb|freq_types       the site types with Z >= 1 (footnote of the comparison table)
  fnb|freq_clause      ", and frequency-only manipulation at <types>" | ""  (whole clause)
  types|clause         the clause on the site types where frequency-only manipulation separates
  K|anchor             inside | below | above   (the realised urban K against the range of the S20 road-crossing means; read from the printed design constant lit_k_crossing)
  T2|bias|<cell>       reduced | not reduced (|calibrated| < |uncalibrated| or not; at the variant Table A prints:
                        the worst-sigma variant)
  T7                   adopted | not adopted;   dqual|<family>  yes | no;   ucont  yes | no;   incomplete|<cell> label
  splace|status        finished | failed | window closed | not run (S-place; its effect is unquantified unless finished)
  T3N|status, v2v|status, abs|qual, T8|<cell>   some / computed / applies, or the pre-specified "no qualified site"
                        alternative
  v2v|majority         computed (>= 20 included crossings blocked over most of their length at qualified P3 sites)
                        | too few
  esc|status           adopted | fallback | not evaluated (the R1 escalation; adopted only if the honest bound over
                        R2, the escalation and R3 is <= 1e-3; not evaluated without an honest U / H row)
  Bstratum|<stratum>   covered | covered_short | not_covered_floor | not_covered_unqualified | not_covered_c5 |
                        no_records (urban; highway; the adapter's stratum status; make_reports B|stratum|<s>|status)
  Bnote|<stratum>      printed (covered_short: the Table B stratum note with its n_eval keys) | none
  r3|m5                exists | absent | not run (the R3 critical region at five entries under the TR 37.885 honest
                        share of z_q = -1, B|zq_minus1|share against alpha_pow_1_5; descriptive)
  feas|status          claim | feasibility (claim if some U / H cell has k/n >= 0.3 and the honest punitive UB95 of
                        the adopted rule set is <= 1e-3; evaluated after esc|status)
Usage: python reports/select_texts.py --tag primary [--results DIR] [--out DIR]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import stage3_criteria as K  # noqa: E402

C = K.C


MIN_MAJORITY = 20                                         # crossings blocked over most of their length
FEAS_CELLS = ("P1-10", "P1-15", "P2-20", "P2-30", "P2-38")    # the U and H cells (P3 is not a site type)
FEAS_SHARE, FEAS_UB95 = 0.3, 1e-3


def k_anchor_range():
    """The range the K-anchor sentence prints (design constant lit_k_crossing, S20 road-crossing means), so the bracket
    rule and the printed text cannot differ."""
    dc = {c["key"]: c["value"] for c in json.load(open(os.path.join(ER, "config", "design_constants.json")))["constants"]}
    lo, hi = dc["lit_k_crossing"].replace("dB", "").replace("−", "-").split(" to ")
    return float(lo), float(hi)


def select(cr, nums):
    b = {}
    sc = cr["scope"]
    for t, r in sc["types"].items():
        b[f"T3full|{t}"] = r
        base, _, suf = r.partition(" (")
        if base.startswith("covered at"):
            b[f"T3|{t}"], b[f"T3suffix|{t}"] = "partly covered", base[len("covered "):]
        else:
            b[f"T3|{t}"], b[f"T3suffix|{t}"] = base, ("(" + suf) if suf else ""
        if r.startswith("not run (compute)"):
            b[f"T3|{t}"], b[f"T3suffix|{t}"] = "not run (compute)", ""
    for c, r in sc["cells"].items():
        b[f"T3cell|{c}"] = r["rating"]
    for c, q in (cr.get("qualification_validity") or {}).items():
        g = q.get("at_governing") or {}
        su = g.get("sigma_unqualified")
        b[f"T3u|{c}"] = "borne_out" if (su is not None and np.isfinite(su) and su > K.SIG_Q) else "conservative"
        k = (g.get("qualified_over_margin") or {}).get("k")
        b[f"T3share|{c}"] = "none" if not k else "some"
    bridge = cr.get("bridge") or {}
    for c, g in (cr.get("optimism_governing") or {}).items():
        t = [n for n, spec in K.TYPES.items() if c in spec["cells"]]
        q = ((bridge.get(t[0]) or {}).get("q99_ratio_worst") or {}).get(c) if t else None
        tails = g["verdict"] == "not contradicted" and q is not None and np.isfinite(q) and q > K.OC["margin"]
        b[f"T4|{c}"] = "tails" if tails else g["verdict"]
    freq = []
    for key, d in (cr.get("c4_light") or {}).items():
        t = "urban LOS" if key.startswith("P1") else "highway LOS"
        if not np.isfinite(d.get("Z_worst", np.nan)):
            b[f"T6|{t}"] = "not run"
        elif d["Z_worst"] >= 1.0:
            b[f"T6|{t}"] = "separates"
            freq.append(t)
        else:
            b[f"T6|{t}"] = "restricted"
    b["fnb|freq_types"] = ", ".join(freq) if freq else "none"
    b["fnb|freq_clause"] = f", and frequency-only manipulation at {', '.join(freq)} sites" if freq else ""
    run_t = [k for k, v in b.items() if k.startswith("T6|") and v != "not run"]
    rest = [k[3:] for k in run_t if b[k] == "restricted"]
    if freq and not rest:
        b["types|clause"] = " at all site types"
    elif freq:
        b["types|clause"] = (f" at {', '.join(freq)} sites; at {', '.join(rest)} sites the separation is below "
                           r"\dcn{z_sep} and the claim is restricted")
    else:
        b["types|clause"] = r"; at no site type does it reach \dcn{z_sep}, and the claim is restricted accordingly"
    km, (lo, hi) = nums.get("K|urban|median"), k_anchor_range()
    if km is not None:
        b["K|anchor"] = "inside" if lo <= km <= hi else ("below" if km < lo else "above")
    for h in cr.get("headline", []):
        key = f"{h['cell']}-{h['v0']:g}"
        gv = (h.get("worst_sigma_qualified_gated") or [None, {}])[1] or {}     # the variant Table A prints (r0)
        v = [r for r in h["variants"] if r.get("bias_uncalibrated") and r["mapping"] == gv.get("mapping")
             and r["prb"] == gv.get("prb")] or [r for r in h["variants"] if r.get("bias_uncalibrated")]
        if v:
            u, c_ = abs(v[0]["bias_uncalibrated"]["mean"]), abs(v[0]["bias_calibrated"]["mean"])
            b[f"T2|bias|{key}"] = "reduced" if (np.isfinite(u) and np.isfinite(c_) and c_ < u) else "not reduced"
    b["T7"] = "adopted" if K.LEVER_D else "not adopted"
    b["splace|status"] = (cr.get("s_place_status") or {}).get("status", "not run")
    # branches of the outcomes without qualified sites: the texts that print a number of
    # qualified sites have a pre-specified "no qualified site" alternative
    kc = {c: (r or {}).get("k") or 0 for c, r in sc["cells"].items()}
    b["T3N|status"] = "some" if kc.get("P3-15", 0) >= 1 else "none"
    v2v = cr.get("v2v_reading") or {}
    fin = lambda x: x is not None and np.isfinite(x)
    if kc.get("P3-15", 0) < 1 or kc.get("P1-15", 0) < 1 or not fin(v2v.get("entry_yield_unblocked_P1_15")):
        b["v2v|status"] = "no qualified site"
    elif not fin(v2v.get("entry_yield_blocked_P3_15")):
        b["v2v|status"] = "no blocked crossing"                  # qualified sites, none of their crossings blocked
    else:
        b["v2v|status"] = "computed" if fin(v2v.get("entry_yield_v2v")) else "no blocked crossing"
    b["v2v|majority"] = ("computed" if b["v2v|status"] == "computed" and fin(v2v.get("entry_yield_blocked_majority"))
                         and (v2v.get("n_blocked_majority") or 0) >= MIN_MAJORITY else "too few")
    b["abs|qual"] = "some" if any(k >= 1 for c, k in kc.items() if c[:2] in ("P1", "P2")) else "none"
    for c, k in kc.items():
        b[f"T8|{c}"] = "applies" if k >= 1 else "no qualified site"
    b["esc|status"] = nums.get("B|esc|status", "not evaluated")             # make_reports.table_b
    kn = {c: (r or {}).get("k", 0) / max((r or {}).get("n") or 1, 1) for c, r in sc["cells"].items()}
    ub = nums.get("B|honest|governing|punitive_ub95")             # of the adopted rule set (make_reports)
    reach = any(v >= FEAS_SHARE for c, v in kn.items() if c in FEAS_CELLS)
    b["feas|status"] = "claim" if (reach and ub is not None and ub <= FEAS_UB95) else "feasibility"
    for s in ("urban", "highway"):                       # Table B stratum notes
        st = nums.get(f"B|stratum|{s}|status", "no_records")
        b[f"Bstratum|{s}"] = st
        b[f"Bnote|{s}"] = "printed" if st == "covered_short" else "none"
    if nums.get("B|zq_minus1|p5_le_alpha") is None:      # R3 at five entries
        b["r3|m5"] = "not run"
    else:
        b["r3|m5"] = "exists" if nums["B|zq_minus1|p5_le_alpha"] else "absent"
    for fam in ("U", "N"):
        b[f"dqual|{fam}"] = "yes" if fam in K.EV4.get("p2_dqual", []) else "no"
    b["ucont"] = "yes" if C["evaluation_v5"]["memory"].get("u_contingency_applied") else "no"
    for c, lab in ((cr.get("incomplete_run") or {}).get("labels") or {}).items():
        b[f"incomplete|{c}"] = lab.get("label") or "complete"
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", default=K.RESULTS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "reports", a.tag)
    os.makedirs(out, exist_ok=True)
    cr = json.load(open(os.path.join(a.results, f"criteria_{a.tag}.json")))
    nf = os.path.join(out, f"numbers_{a.tag}.json")
    nums = json.load(open(nf))["numbers"] if os.path.exists(nf) else {}
    b = select(cr, nums)
    json.dump(dict(tag=a.tag, branches=b), open(os.path.join(out, f"text_branches_{a.tag}.json"), "w"), indent=1)
    lines = ["% written by reports/select_texts.py; do not edit", r"\makeatletter"]
    lines += [r"\expandafter\def\csname spb@" + k + r"\endcsname{" + str(v) + "}" for k, v in sorted(b.items())]
    lines += [r"\providecommand{\spb}[1]{\@ifundefined{spb@#1}{\textbf{??#1??}\PackageWarning{speedpath}{undefined branch #1}}"
              r"{\csname spb@#1\endcsname}}", r"\makeatother"]
    open(os.path.join(out, "speedpath-branches.tex"), "w").write("\n".join(lines) + "\n")
    print(out, len(b), "branches")


if __name__ == "__main__":
    main()
