r"""Study 2: derived values from the stored held-out outputs.

Reads the stage-3 outputs under results/study2/heldout/, the study-2 snapshot-placement result and the
development-stage decision (step0_decision.json),
and writes results/study2/paper/speedpath-s2-derived.tex and study2_derived.json with keys S2d|... for values that the
study-2 text files print in another format or do not print. Every value is read, none is recomputed from records.

Format fixes:
  S2d|K|urban|median      the study-2 urban realised Ricean K in dB (which misses the dB class of number_macros.py)
  S2d|<cell>|p_holm       the Table A2 optimism p_Holm with three decimals
Values added:
  S2d|<cell>|k_min/k_max, S2d|<cell>|k|<mapping>|<prb>       qualified sites per evaluation variant (scope rule)
  S2d|<cell>|r8iii_min/r8iii_max, S2d|<cell>|r8iii|<m>|<prb>  rank correlation survey sd vs test sd per variant
  S2d|<cell>|p_tost_s2_holm_min, S2d|<cell>|tost_s2_pass_n    the L1 bias test over the six variants
  S2d|<cell>|excess_detected_n                                 variants in which C2 (ii) detects an excess
  S2d|P1-15|R_unscaled(_ci_lo/_ci_hi), R_n_sites, R_undetermined_n   optimism check without the S-place factor
  S2d|splace2|<mapping>|rho(_ci_lo/_ci_hi), passes, passes_planned, sites   the study-2 S-place light check
  S2d|d0r1|<cell>|eb_prb50_min/max                             development-stage excess bias at the 50-PRB variants (dev data)
  S2d|<cell>|rating                                            the study-1 scope rating (string)
Usage: python tr37885/reports/study2_derived_values.py [--check]
--check: regenerate in memory and fail if the stored derived file differs."""
import argparse
import importlib.util
import json
import math
import os
import sys

ER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HO = os.path.join(ER, "..", "results", "study2", "heldout")
OUT_DIR = os.path.join(ER, "..", "results", "study2", "paper")
OUT_TEX = os.path.join(OUT_DIR, "speedpath-s2-derived.tex")
OUT_JSON = os.path.join(OUT_DIR, "study2_derived.json")
CELLS = ("P1-10", "P1-15", "P3-15")
CONSTANTS = os.path.join(ER, "config", "study2_constants.json")

_spec = importlib.util.spec_from_file_location("speedpath_macros", os.path.join(ER, "reports", "number_macros.py"))
_mac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mac)


def load(path):
    return json.load(open(path))


def constants_s2():
    return json.load(open(CONSTANTS))


def minus(s):
    return r"\ensuremath{-}" + s[1:] if s.startswith("-") else s


def f2(v):
    return minus(f"{v:.2f}")


def f3(v):
    return minus(f"{v:.3f}")


def pct(v):
    return minus(f"{100 * v:.1f}")


def derive(crit, outcome, nums, splace, dev, consts):
    d = {}
    # --- format fixes
    d["S2d|K|urban|median"] = _mac.fmt("K|urban|median", nums["A2|K|urban|median"])
    for c in CELLS:
        d[f"S2d|{c}|p_holm"] = f3(nums[f"A2|{c}|p_holm"])
    # --- per-variant qualification and rank correlation
    for c in CELLS:
        sc = crit["scope"]["cells"][c]
        ks = [v["k"] for v in sc["variants"]]
        if min(ks) != sc["k"]:
            sys.exit(f"study2_derived_values: scope k of {c} is not the smallest over the variants")
        d[f"S2d|{c}|k_min"], d[f"S2d|{c}|k_max"] = str(min(ks)), str(max(ks))
        d[f"S2d|{c}|n_variants"] = str(len(ks))
        d[f"S2d|{c}|rating"] = sc["rating"]
        for v in sc["variants"]:
            d[f"S2d|{c}|k|{v['mapping']}|{v['prb']}"] = str(v["k"])
        qv = crit["qualification_validity"][c]["variants"]
        rs = [v["spearman_sdref_vs_test_sigma"] for v in qv]
        d[f"S2d|{c}|r8iii_min"], d[f"S2d|{c}|r8iii_max"] = f2(min(rs)), f2(max(rs))
        for v in qv:
            d[f"S2d|{c}|r8iii|{v['mapping']}|{v['prb']}"] = f2(v["spearman_sdref_vs_test_sigma"])
        if crit["qualification_validity"][c]["at_governing"]["spearman_sdref_vs_test_sigma"] != nums[f"S2|{c}|r8iii"]:
            sys.exit(f"study2_derived_values: S2|{c}|r8iii is not the at-governing rank correlation")
    # --- the L1 bias test over the six variants
    for c in CELLS:
        v0 = float(c.split("-")[1])
        cell = "P1" if c.startswith("P1") else "P3"
        rows = [r for r in crit["c2"] if r["cell"] == cell and float(r["v0"]) == v0]
        if len(rows) != 6:
            sys.exit(f"study2_derived_values: {len(rows)} C2 rows for {c}, expected 6")
        d[f"S2d|{c}|p_tost_s2_holm_min"] = f3(min(r["p_tost_s2_holm"] for r in rows))
        d[f"S2d|{c}|tost_s2_pass_n"] = str(sum(bool(r["tost_s2_pass"]) for r in rows))
        d[f"S2d|{c}|excess_detected_n"] = str(sum(bool(r["excess_detected"]) for r in rows))
    # --- optimism at P1-15 without the S-place factor
    og = crit["optimism_governing"]["P1-15"]
    unscaled = [r for r in crit["optimism"] if r["cell"] == "P1" and float(r["v0"]) == 15.0]
    row = next(r for r in unscaled if r["mapping"] == og["mapping"] and r["prb"] == og["prb"])
    fac = crit["s_place"]["governing_urban"]
    if not math.isclose(row["R"] * fac, og["R"], rel_tol=1e-9):
        sys.exit("study2_derived_values: governing R is not the unscaled R times the governing S-place factor")
    if not math.isclose(nums["A2|P1-15|R_ci_lo"], row["R_ci_widened"][0], rel_tol=1e-12):
        sys.exit("study2_derived_values: A2|P1-15|R_ci_lo is not the unscaled widened interval")
    d["S2d|P1-15|R_unscaled"] = f2(row["R"])
    d["S2d|P1-15|R_unscaled_ci_lo"], d["S2d|P1-15|R_unscaled_ci_hi"] = (f2(x) for x in row["R_ci_widened"])
    d["S2d|P1-15|R_n_sites"] = str(row["n_sites"])
    d["S2d|P1-15|R_undetermined_n"] = str(sum(r["verdict"] == "undetermined" for r in unscaled))
    d["S2d|P1-15|R_variants_n"] = str(len(unscaled))
    # --- study-2 S-place light check
    if crit["s_place"]["study2"]["urban"] != splace["stratum_factor"]["urban"]:
        sys.exit("study2_derived_values: the S-place summary is not the one stage 3 used")
    p1 = splace["cells"]["P1"]
    for mp in ("e1", "lb"):
        d[f"S2d|splace2|{mp}|rho"] = f2(p1[mp]["rho"])
        d[f"S2d|splace2|{mp}|rho_ci_lo"], d[f"S2d|splace2|{mp}|rho_ci_hi"] = (f2(x) for x in p1[mp]["rho_ci"])
        d[f"S2d|splace2|{mp}|passes"] = str(p1[mp]["passes"])
    sp = consts["evaluation_v6"]["s_place"]
    lo, hi = sp["sites"]
    d["S2d|splace2|sites"] = str(hi - lo + 1)
    d["S2d|splace2|passes_planned"] = str((hi - lo + 1) * sp["passes"])
    d["S2d|splace2|threshold"] = f2(consts["sensitivities"]["S_place"]["threshold"])
    # --- development stage: excess bias at the 50-PRB variants
    for c in ("P1-10", "P1-15"):
        eb = [v["excess_bias"] for v in dev["D0_R1"]["cells"][c]["variants"] if v["prb"] == 50]
        d[f"S2d|d0r1|{c}|eb_prb50_min"], d[f"S2d|d0r1|{c}|eb_prb50_max"] = pct(min(eb)), pct(max(eb))
        d[f"S2d|d0r1|{c}|prb50_ci_includes_0"] = str(int(all(v["ci"][0] <= 0 <= v["ci"][1] for v in
                                                          dev["D0_R1"]["cells"][c]["variants"] if v["prb"] == 50)))
    # --- rungs (outcome)
    for c, r in outcome["per_cell"].items():
        d[f"S2d|{c}|rung"] = r
    return d


def render(d):
    lines = ["% written by reports/study2_derived_values.py from the stored study-2 outputs; do not edit",
             r"\makeatletter"]
    for k in sorted(d):
        lines.append(r"\expandafter\def\csname spn@" + k + r"\endcsname{" + d[k] + "}")
    lines.append(r"\makeatother")
    return "\n".join(lines) + "\n"


def build():
    crit = load(os.path.join(HO, "criteria_s2_dev_s2_heldout.json"))
    outcome = load(os.path.join(HO, "outcome_s2_dev_s2_heldout.json"))
    nums = load(os.path.join(HO, "reports", "s2_numbers_dev_s2_heldout.json"))["numbers"]
    splace = load(os.path.join(ER, "..", "results", "s_place_dev_s2_heldout_place.json"))
    dev = load(os.path.join(ER, "..", "results", "study2", "step0_decision.json"))
    return derive(crit, outcome, nums, splace, dev, constants_s2())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = build()
    tex, js = render(d), json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if a.check:
        ok = os.path.exists(OUT_TEX) and open(OUT_TEX).read() == tex and open(OUT_JSON).read() == js
        print("study2_derived_values: stored derived file", "matches" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    open(OUT_TEX, "w").write(tex)
    open(OUT_JSON, "w").write(js)
    print(os.path.relpath(OUT_TEX, ER), len(d), "keys")


if __name__ == "__main__":
    main()
