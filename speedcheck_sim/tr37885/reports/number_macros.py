"""LaTeX number macros for the speed path: every speed-path number in the manuscript and the supplement is written
as \\spn{<key>} and filled from
results/reports/<tag>/numbers_<tag>.json. Formatting is fixed by the key's last field (below), so a number is
never rounded by hand. An undefined key prints a bold ??key?? and a LaTeX warning.
With --placeholder, every key of the numbers file gets the text [key] (for drafting before the data exist).
Usage: python reports/number_macros.py --tag primary [--out FILE] [--placeholder]"""
import argparse
import json
import os

ER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCT1 = ("sigma", "sigma_g1", "sigma_e1", "sigma_governing", "sigma_g1_ci_hi", "sigma_unqualified", "q99", "oow",
        "gate_discard", "entry_yield", "bias_uncal", "bias_cal", "cp_lo", "cp_hi", "qualified_over_margin",
        "blocked_share_mean", "blocked_any", "entry_yield_unblocked_P1_15", "entry_yield_blocked_P3_15", "entry_yield_v2v",
        "w_v2v", "q", "median", "min", "max", "gate_discard",
        # shares the pre-specified texts print with "%"
        "sigma_min", "sigma_max", "sigma_inhouse_equalK", "entry_yield_blocked_majority", "over_governing_share",
        "over_governing_cp_lo", "over_governing_cp_hi", "qualified_over_margin_cp_lo", "qualified_over_margin_cp_hi",
        "sigma_g1_ci_lo", "bias_uncal_ci_lo", "bias_uncal_ci_hi", "bias_cal_ci_lo", "bias_cal_ci_hi", "q_tableA")
INT = ("k", "n", "n_kept", "n_sites",
       # counts
       "k_min", "k_max", "k_inhouse", "over_governing_k", "qualified_over_margin_k", "n_blocked_majority", "n_eval",
       "rows", "rows_below_min_eval")
SCI = ("punitive_per_year", "punitive_ub95", "notice_per_year", "p_holm",
       # per-vehicle-year rates and their bounds (T5, T5-esc, abstract)
       "r2_punitive_per_year", "r2_punitive_ub95", "union_punitive_per_year", "union_punitive_ub95",
       "punitive_union_per_year", "punitive_union_ub95", "all_union_ub95", "escalation_per_year", "inspections_per_year")
# the type of a key whose last field alone does not fix it
PCT_PREFIX = ()
SCI_PREFIX = ("B|honest|r3_inhouse|",)       # R3 honest per-vehicle-year rate (a rate, not a share)
DB_PREFIX = ("K|",)                          # realised Ricean K in dB (K-anchor sentence), never x 100
DEC3 = ("p_unique", "share")                 # matching-simulation P(unique) (reference 0.014 / 0.163, matching numbers of Table 11) and
                                             # B|zq_minus1|share against alpha_pow_1_5 = 0.11 (T5-m5): three decimals
COUNT_LIKE = ("median_candidates",)          # matching-simulation median number of candidates (reference 192 / 6): no decimals


# ratios, means and dB values that print with two decimals by design; any other number key that matches no class is
# "unclassified" (still printed with two decimals and listed by main in the macro file)
DEC2 = ("R", "R_ci_lo", "R_ci_hi", "q99_ratio", "s_place", "spearman", "factor_vs_unblocked", "Z",
        "crossings_per_epoch_mean", "entries_per_epoch_mean", "low_m_share",
        "p_zq_minus1_beta10", "p_zq_minus1_honest", "below_decoding_pct")


def _sci(v):
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"\ensuremath{{{m}\times 10^{{{int(e)}}}}}"


def key_class(key, v):
    """The format class of a number key: int | db | sci | pct | hz | count | dec3 | int_value | dec2 |
    unclassified; str / none / bool for non-numbers. The order is the order of the rules in fmt."""
    last = key.split("|")[-1]
    if v is None:
        return "none"
    if isinstance(v, str):
        return "str"
    if isinstance(v, bool):
        return "bool"
    if last in INT:
        return "int"
    if key.startswith(DB_PREFIX):
        return "db"
    if last in SCI or key.startswith(SCI_PREFIX):
        return "sci"
    if last in PCT1 or (key.startswith(PCT_PREFIX) and last != "source_sha256"):
        return "pct"
    if last.endswith("_hz"):
        return "hz"
    if last in COUNT_LIKE:
        return "count"
    if last in DEC3:
        return "dec3"
    if isinstance(v, int):
        return "int_value"
    if last in DEC2 or last.endswith("_db") or key.startswith("B|lambda_eff|"):   # lambda_eff = lambda x q per epoch
        return "dec2"
    return "unclassified"


def fmt(key, v):
    c = key_class(key, v)
    if c == "none":
        return r"n/a"
    if c == "str":
        return v.replace("_", r"\_")
    if c == "bool":
        return str(int(v))
    if c == "int":
        return f"{int(round(v))}"
    if c == "db":
        return f"{v:.1f}"
    if c == "sci":
        return _sci(v)
    if c == "pct":
        return f"{100 * v:.1f}"
    if c == "hz":
        return f"{v:.0f}"
    if c == "count":
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    if c == "dec3":
        return f"{v:.3f}"
    if c == "int_value":                     # any other integer-valued count is printed without decimals
        return f"{v}"
    return f"{v:.2f}"                        # dec2 and unclassified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--numbers", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--placeholder", action="store_true")
    a = ap.parse_args()
    src = a.numbers or os.path.join(ER, "..", "results", "reports", a.tag, f"numbers_{a.tag}.json")
    nums = json.load(open(src))["numbers"]
    out = a.out or os.path.join(ER, "..", "results", "reports", a.tag, "speedpath-numbers.tex")
    lines = ["% written by reports/number_macros.py from " + os.path.relpath(src, ER) + "; do not edit"]
    unc = sorted(k for k in nums if key_class(k, nums[k]) == "unclassified")
    lines += ["% unclassified number keys (two decimals by default): " + (", ".join(unc) if unc else "none"),
              r"\makeatletter"]
    for k in sorted(nums):
        val = f"[{k}]" if a.placeholder else fmt(k, nums[k])
        lines.append(r"\expandafter\def\csname spn@" + k + r"\endcsname{" + val + "}")
    dc = json.load(open(os.path.join(ER, "config", "design_constants.json")))
    for c in dc["constants"]:                              # \dcn{key}: design constants
        lines.append(r"\expandafter\def\csname dcn@" + c["key"] + r"\endcsname{" + c["value"].replace("%", r"\%") + "}")
    lines += [r"\providecommand{\dcn}[1]{\@ifundefined{dcn@#1}{\textbf{??#1??}}{\csname dcn@#1\endcsname}}",
              r"\providecommand{\stdref}[2]{#1 #2}"]     # clauses through \stdref{doc}{clause} (checked by the clause table)
    lines += [r"\providecommand{\spn}[1]{\@ifundefined{spn@#1}{\textbf{??#1??}\PackageWarning{speedpath}{undefined number #1}}"
              r"{\csname spn@#1\endcsname}}", r"\makeatother"]
    open(out, "w").write("\n".join(lines) + "\n")
    print(out, len(nums), "macros", f"{len(unc)} unclassified")


if __name__ == "__main__":
    main()
