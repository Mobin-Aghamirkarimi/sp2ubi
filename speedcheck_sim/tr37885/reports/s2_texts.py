"""Study 2 texts (keys S2|...; no hand-typed number). The study-2 keys, branches and templates live here; the study-1
report modules are not changed.
  numbers(crit, outcome, dec)   number keys S2|<cell>|..., S2|splace|..., S2|d0r1|..., S2|d0r2|<cell>|<Dev>|share / |eb, S2|design|..., the Table A2 keys A2|... (stage 3, make_reports.table_a)
  branches(outcome, crit, dec)  branch keys S2b|... (the outcome map: template; gate; rung per cell; k = 0; adopted
                                estimator / N2 rule; H2; the study-2 S-place status)
  texts(outcome, crit, dec)     the pre-specified sentences (cause, future work, disclosure, yield, K, the summary
                                paragraph, the L1 and T3-N texts) with the number keys inserted as \\spn{S2|...}
  table_b_keys(adapter_dir, tag) at L3: the fixed make_reports Table B / stratum / the matching simulation keys on the adapter outputs, S2|B|...
  write(...)                    s2_numbers_<tag>.json, s2_texts_<tag>.json, speedpath-s2.tex (\\spn / \\spb macros)
  d0_1_cell(entry)              the supplement cell of a D0.1 stratum (never sigma 0.0; every stratum prints n)
  abstract_word_counts(tex)     the abstract <= 250 words check over every branch combination of the study-1 abstract
                                with and without the study-2 L3 clause; with the
                                clause, the study-1 urban clause STUDY1_URBAN_CLAUSE is replaced; abstract_check(tex, branches) applies the pre-specified rule on the realised branch
Naming: the Dev-3 control tracks are "free-flow control crossings" in every study-2 text; the word
"quadratic" never appears in a text key."""
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)

U2_CELLS = ("P1-10", "P1-15")
DEVS = ("Dev-1", "Dev-2")
WIN = 0.25                                                # criteria.WIN (the +-25 % clip); checked in the tests
CONTROL_TRACKS = "free-flow control crossings"


# ---------------------------------------------------------------------------------------------- numbers
def _fin(x):
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def numbers(crit, outcome, dec):
    n = {}
    for key, c in crit["cells"].items():
        sc = c["scope"]
        p = f"S2|{key}|"
        n[p + "k"], n[p + "n"] = sc["k"], sc["n"]
        n[p + "v0"] = int(round(float(key.split("-")[1])))
        n[p + "cp_lo"], n[p + "cp_hi"] = sc["cp95"]
        n[p + "sigma_governing"] = c["sigma_governing"] if _fin(c["sigma_governing"]) else None
        n[p + "r8ii_over"] = c["r8"]["n_over_9pct"]
        n[p + "r8iii"] = c["r8"]["r8iii_at_governing"]["spearman"]
        rows = c.get("c2_rows") or []
        if rows:                                          # L1 statistic: C2 bias / bias_ci (gen_on), Holm s2 TOST p
            w = max(rows, key=lambda r: abs(r["bias"]))
            n[p + "bias"], (n[p + "bias_ci_lo"], n[p + "bias_ci_hi"]) = w["bias"], w["bias_ci"]
            n[p + "p_tost_s2_holm_max"] = max(r.get("p_tost_s2_holm", 1.0) for r in rows)
        hf = c.get("headline_final") or {}
        n[p + "sigma_source"] = hf.get("source")
    from pipeline import s2_constants as S2
    n["S2|l1_band"] = S2.N16_L1_BIAS_BAND
    sp = crit.get("s_place") or {}
    for k in ("study1", "study2"):
        if k in sp:
            n[f"S2|splace|{k}"] = sp[k]["urban"]
    if "governing_urban" in sp:
        n["S2|splace|governing"] = sp["governing_urban"]
    r1 = (dec.get("D0_R1") or {}).get("cells") or {}
    for key, c in r1.items():                             # the worst variant of D0-R1
        w = c.get("worst") or {}
        if w:
            n[f"S2|d0r1|{key}|diff"], n[f"S2|d0r1|{key}|eb"] = w.get("mean_diff"), w.get("excess_bias")
            n[f"S2|d0r1|{key}|ci_lo"], n[f"S2|d0r1|{key}|ci_hi"] = (w.get("ci") or [None, None])[:2]
    r2 = dec.get("D0_R2") or {}                           # the nested shares
    sh = []
    for key, v in (r2.get("cells") or {}).items():
        v = v or {}
        n[f"S2|d0r2|{key}|holds"] = bool(v.get("holds"))
        for src, x in (v.get("shares") or {}).items():
            if x:
                n[f"S2|d0r2|{key}|{src}|share"], n[f"S2|d0r2|{key}|{src}|eb"] = x.get("share"), x.get("excess_bias")
                if _fin(x.get("share")):
                    sh.append(x["share"])
    if sh:                                                # min / max over the four
        n["S2|d0r2|share_min"], n["S2|d0r2|share_max"] = min(sh), max(sh)
    n.update(design_numbers())
    for key, c in crit["cells"].items():                  # R8 (iii) site count, optimism check (Table A2, T4)
        n[f"S2|{key}|r8iii_n"] = (c["r8"].get("r8iii_at_governing") or {}).get("n_sites")
    for key, g in (crit.get("optimism_governing") or {}).items():
        n[f"S2|{key}|optimism"], n[f"S2|{key}|R"], n[f"S2|{key}|p_holm"] = g.get("verdict"), g.get("R"), g.get("p_holm")
    a2 = (crit.get("table_A2") or {}).get("numbers") or {}
    n.update(a2)
    p3 = crit.get("p3_blocked") or {}                     # T3-N
    if "P3-15" in crit["cells"]:
        n["S2|P3-15|blocked_share"] = p3.get("share_any")
        for dst, src in (("gate_discard", "gate_discard"), ("q99", "q99"), ("entry_yield", "entry_yield"),
                         ("sigma_g1", "sigma_g1")):           # the study-1 T3-N template prints
                                                          # A2|P3-15|sigma_g1 with A2|P3-15|sigma_g1_ci_hi; no sigma_ci_lo/hi
            n[f"S2|P3-15|{dst}"] = a2.get(f"A2|P3-15|{src}")
    k7 = crit.get("k7") or {}
    if isinstance(k7, dict) and "c_k7" in k7:
        n["S2|k7|c"] = k7["c_k7"]
    return n


def design_numbers(C=None):
    """Pre-specified design values printed in the texts, read from where they are pre-specified,
    never typed: the Dev-3 sites (evaluation_v6.dev3.sites, half-open), the SUMO degree of saturation (sumo_kinematics.py.DEMAND_VC)
    and the L2 share (evaluation_v4.scope_rule)."""
    C = C or json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study2_constants.json"))))
    lo, hi = C["evaluation_v6"]["dev3"]["sites"]
    import ast
    src = open(os.path.join(ER, "sumo_kinematics.py")).read()   # read without importing sumo_kinematics.py (it needs the SUMO tools)
    vc = next(ast.literal_eval(node.value) for node in ast.parse(src).body if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "DEMAND_VC" for t in node.targets))
    return {"S2|design|dev3_n_sites": hi - lo, "S2|design|dev3_sites": f"{lo}–{hi - 1}", "S2|design|saturation": vc,
            "S2|design|l2_share": C["evaluation_v4"]["scope_rule"]["partly_covered"]}


def fmt(key, v):
    """As reports/number_macros.py.fmt for the classes used here: shares and sigmas in per cent with one decimal, counts
    as integers, rank correlations and factors with two decimals, Holm p with three."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if key.startswith(("A2|", "S2|B|", "S2|E5|")):        # Table A2 / Table B keys: the study-1 formats of make_reports
        from reports.number_macros import fmt as f1
        return f1(("A|" + key[3:]) if key.startswith("A2|") else key[3:], v)
    if isinstance(v, str):
        return v.replace("_", r"\_")
    if key.startswith("S2|design|"):                      # pre-specified design values, printed as pre-specified
        return f"{v:g}" if isinstance(v, float) else f"{v}"
    last = key.split("|")[-1]
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if last in ("k", "n", "r8ii_over", "v0", "r8iii_n") or isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if last in ("r8iii", "R", "c") or key.startswith("S2|splace|"):
        return f"{v:.2f}"
    if last.startswith("p_"):
        return f"{v:.3f}"
    return f"{100 * v:.1f}"                               # sigma, bias, CI bounds, cp, d0r1 / d0r2 values: per cent


# ---------------------------------------------------------------------------------------------- branches
def branches(outcome, crit, dec):
    sel = dec.get("selected") or {}
    b = {"S2b|template": outcome["template"].split("|")[1], "S2b|gate": outcome["gate"],
         "S2b|h2": "run" if outcome.get("h2") == "run" else "not run",
         "S2b|estimator": sel.get("estimator") or "none", "S2b|n2rule": sel.get("n2_rule") or "none",
         "S2b|d0r1": "supported" if (dec.get("D0_R1") or {}).get("holds") else "not isolated",
         "S2b|summary": outcome["template"].split("|")[1]}        # the summary paragraph is selected by gate and rung
    for c in U2_CELLS:
        b[f"S2b|rung|{c}"] = outcome["per_cell"][c]
        b[f"S2b|k0|{c}"] = "yes" if crit["cells"][c]["scope"]["k"] == 0 else "no"
        b[f"S2b|named|{c}"] = "yes" if c in outcome["named_cells"] else "no"
    b["S2b|T3N"] = "some" if crit["cells"].get("P3-15", {}).get("scope", {}).get("k", 0) >= 1 else "none"
    b["S2b|splace2"] = ("unquantified" if ((crit.get("s_place") or {}).get("study2") or {}).get("status") == "unquantified"
                        else "quantified")
    return b


# ---------------------------------------------------------------------------------------------- texts
def _spn(k):
    return r"\spn{" + k + "}"


def _cells_phrase(cells):
    return " and ".join(_spn(f"S2|{c}|v0") for c in cells) + r"\,m/s"


def _at(c):
    return "At " + _spn(f"S2|{c}|v0") + r"\,m/s"


def _kn(cells):
    return " and ".join(_spn(f"S2|{c}|k") for c in cells) + " of " + _spn(f"S2|{cells[0]}|n")


def _sig(cells):
    return " and ".join(_spn(f"S2|{c}|sigma_governing") for c in cells) + "\\,\\%"


def l3_abstract_clause(cells):
    """L3 abstract clause. It replaces the study-1 urban
    clause STUDY1_URBAN_CLAUSE (with its leading '; ') after the study-1 sentence; the study-1 'not covered at the
    decision level' stays."""
    pl = len(cells) > 1
    return ("a second, separately pre-specified design qualified held-out urban street sites in " + _kn(cells) + " at "
            + _cells_phrase(cells) + " (error standard deviation" + ("s " if pl else " ") + _sig(cells)
            + ", Section VI-D)")


# Study-2 texts; keys and design values, no typed result
CAUSE_M1 = (
    "The development stage of study 2 supports a kinematic cause of the urban bias, in the sense of its pre-specified "
    "criterion. On "
    + _spn("S2|design|dev3_n_sites") + " development sites (the study-1 urban sites " + _spn("S2|design|dev3_sites")
    + "), each simulated traffic crossing was paired with a " + CONTROL_TRACKS[:-1] + " in the same lane, which cruises "
    "near the nominal speed without other traffic or lane change and reuses the crossing's beacon, erasure, noise and "
    "impairment seeds under the same site and channel model. At the variant with the largest bias, the mean paired "
    "difference of the uncalibrated speed error (traffic minus control) was " + _spn("S2|d0r1|P1-10|diff") + "\\,\\% at "
    + _spn("S2|P1-10|v0") + "\\,m/s and " + _spn("S2|d0r1|P1-15|diff") + "\\,\\% at " + _spn("S2|P1-15|v0")
    + "\\,m/s (site-bootstrap 95\\,\\% intervals " + _spn("S2|d0r1|P1-10|ci_lo") + " to " + _spn("S2|d0r1|P1-10|ci_hi")
    + "\\,\\% and " + _spn("S2|d0r1|P1-15|ci_lo") + " to " + _spn("S2|d0r1|P1-15|ci_hi") + "\\,\\%), against excess "
    "biases of " + _spn("S2|d0r1|P1-10|eb") + "\\,\\% and " + _spn("S2|d0r1|P1-15|eb") + "\\,\\%. On the study-1 urban "
    "records (development data for study 2), crossings without a stop between the heads carried most of the uncalibrated "
    "excess bias at the variant D0-R2 evaluates (" + _spn("S2|d0r2|P1-10|Dev-2|share") + "\\,\\% at "
    + _spn("S2|P1-10|v0") + "\\,m/s and " + _spn("S2|d0r2|P1-15|Dev-2|share") + "\\,\\% at " + _spn("S2|P1-15|v0")
    + "\\,m/s), so stops alone do not explain it. This is a development diagnosis under one traffic model (SUMO, "
    "Krauss car following, degree of saturation " + _spn("S2|design|saturation") + "): it concerns the mean error, not "
    "its spread, it was not tested on held-out sites, and it changed no part of the configuration.")
FUTURE_M2 = (
    "The fit of Eq. (\\ref{eq:fdoa}) already allows a constant acceleration, and none of the motion-related candidates "
    "pre-specified for study 2 (a two-segment speed profile, an acceleration gate, a stop detector, a spatial-core statistic "
    "and a combination of the last two) met its adoption rule on development data (Section VI-D). A motion model for "
    "urban street crossings in traffic, evaluated on its own held-out sites, is left to future work.")
DISCLOSURE_M4 = (
    "The development outputs were read three times as they arrived, before the full development run, whose decision "
    "equals that of the last reading (supplement); no held-out value was read before all held-out families had "
    "completed their final stage. The held-out evaluation, which the design prescribes after every development "
    "outcome, was run with the development outcome known.")
K_SENTENCE = (
    "The dependence of the within-site error on the Ricean K factor is descriptive (E1, in-house model) and enters no "
    "rule of study 2.")
YIELD_SENTENCE = (                                      # kept as a key
    "Neither an estimator candidate nor a rule for blocked crossings was adopted, so no candidate yield loss arises; the "
    "entry yield under the traffic survey is given in Table A2.")
SPLACE2_UNQUANTIFIED = ("The study-2 S-place check did not complete; the urban S-place correction therefore uses the "
                        "study-1 factor alone, and the effect of the study-2 sites on the urban standard deviation is not "
                        "quantified.")
PRECISION_RUNG = ("the precision rung (a qualified share of the held-out sites of at least " + _spn("S2|design|l2_share")
                  + " and a per-crossing error standard deviation of at most \\dcn{qual_limit})")
L1_VIIC = ("a second pre-specified design (Section VI-D) made qualification valid in traffic and held the calibrated bias within "
           "$\\pm$" + _spn("S2|l1_band") + "\\,\\%, but did not reach " + PRECISION_RUNG)
SUMMARY_OPEN = (
    "A second, separately pre-specified study of urban street sites (Section VI-D; design record timestamped before "
    "its development stage and again before its held-out run, with seeds derived from the Bitcoin block that holds "
    "the timestamp) replaced "
    "the free-road survey by a survey in traffic and was judged once on ten held-out sites under the unchanged study-1 "
    "criteria. No estimator change was adopted on development data, so highway processing and every highway number are "
    "unchanged and the highway check was not run.")
SUMMARY_END = (" The development-stage diagnosis and every rejected candidate are reported in Section VI-D and the "
               "supplement.")


def t3n_text(k, rating=None):
    """T3-N on the study-2 records: the pre-specified study-1 T3-N template exactly (access.tex T3-N),
    'free-road survey' -> 'survey in traffic' only: sigma_g1 with the upper end of its
    95 % CI (A2|P3-15|sigma_g1, A2|P3-15|sigma_g1_ci_hi); the limitation sentence only when the scope rating is
    'not covered', as in study-1."""
    c = "S2|P3-15|"
    if k == 0:
        return ("At the held-out urban sites of study 2, no site qualified on a survey in traffic in the far lane ("
                + _spn(c + "k") + "/" + _spn(c + "n") + ", 95\\,\\% CI " + _spn(c + "cp_lo") + "--" + _spn(c + "cp_hi")
                + "\\,\\%, at " + _spn(c + "v0") + "\\,m/s); we state this as a limitation of the site-qualified design.")
    t = ("At the held-out urban sites of study 2, qualified on a survey in traffic in the far lane (" + _spn(c + "k") + "/"
         + _spn(c + "n") + ", 95\\,\\% CI " + _spn(c + "cp_lo") + "--" + _spn(c + "cp_hi") + "\\,\\%, at " + _spn(c + "v0")
         + "\\,m/s), " + _spn(c + "blocked_share") + "\\,\\% of the traffic crossings had one head blocked by a vehicle "
         "in the nearer lane for part of the crossing (TR 37.885 NLOSv). The quality gate discarded "
         + _spn(c + "gate_discard") + "\\,\\% of the crossings (no entry), and the kept crossings have a per-crossing "
         "error of " + _spn("A2|P3-15|sigma_g1") + "\\,\\% (upper end of the 95\\,\\% CI, site bootstrap, "
         + _spn("A2|P3-15|sigma_g1_ci_hi") + "\\,\\%; q99 $|e|$ $=$ " + _spn(c + "q99") + "\\,\\%); the entry yield is "
         + _spn(c + "entry_yield") + "\\,\\% of the traffic crossings.")
    if rating == "not covered":
        t += (" These held-out urban sites qualified in only " + _spn(c + "k") + "/" + _spn(c + "n") + " draws; we state "
              "this as a limitation of the site-qualified design.")
    return t


def summary(tpl, named):
    """The summary paragraph of study 2, one per rung."""
    if tpl == "L3":
        body = (" Urban street sites qualified in " + _kn(named) + " held-out draws at " + _cells_phrase(named)
                + ", with per-crossing error standard deviation" + ("s of " if len(named) > 1 else " of ") + _sig(named)
                + ", and the urban stratum is covered at the decision level under the unchanged rules (Tables A2 and B); "
                "the abstract adds one clause.")
    elif tpl == "L2":
        body = (" Urban street sites qualified in " + _kn(named) + " held-out draws at " + _cells_phrase(named)
                + " with per-crossing error standard deviation" + ("s of " if len(named) > 1 else " of ") + _sig(named)
                + "; the urban stratum is not covered at the decision level, so no urban decision rate is claimed and the "
                "abstract is unchanged.")
    elif tpl == "L1":
        body = (" Qualification became valid in traffic (calibrated bias within $\\pm$" + _spn("S2|l1_band")
                + "\\,\\%), but did not reach " + PRECISION_RUNG + "; urban street sites remain not covered and the "
                "abstract is unchanged.")
    elif tpl == "L0":
        body = (" The design did not cover urban street sites either; every urban number is reported in Section VI-D and "
                "the abstract is unchanged.")
    elif tpl == "L0_k0":
        body = (" The traffic survey qualified no held-out urban site (the survey–test rank correlation is printed in "
                "Section VI-D); the abstract is unchanged.")
    else:
        return None                                       # G_fails: not reachable under L-T alone (no wording)
    return SUMMARY_OPEN + body + SUMMARY_END


def texts(outcome, crit, dec):
    """The pre-specified sentences selected by the outcome (keys S2|text|...)."""
    t = {}
    tpl = outcome["template"].split("|")[1]
    named = outcome["named_cells"]
    sel = dec.get("selected") or {}
    est, n2 = sel.get("estimator"), sel.get("n2_rule")
    t["S2|text|VIID|opening"] = (
        "Study 2 was designed after the study-1 results had been seen. Its candidate changes were chosen on development "
        "data by rules fixed before any candidate statistic was computed, and it was evaluated once, on held-out sites "
        "with seeds derived from the timestamped design record and the Bitcoin block that holds its timestamp.")
    t["S2|text|VIID|disclosure"] = DISCLOSURE_M4          # after the opening, every outcome
    t["S2|text|VIID|cause"] = CAUSE_M1                    # Section VI-D and supplement, every outcome
    t["S2|text|future_work"] = FUTURE_M2                  # future work, every outcome
    t["S2|text|VIID|K"] = K_SENTENCE                      # every outcome
    cands = [c.get("name") for c in dec.get("candidates", [])]
    rej = [c for c in cands if c != dec.get("D0_R3_chosen")]
    t["S2|text|VIID|candidates"] = (
        ("No estimator candidate met the pre-specified adoption rule (" + ", ".join(rej) + ")"
         if est is None else f"Candidate {est} was adopted; the others did not meet the pre-specified adoption rule ("
         + ", ".join(rej) + ")") + "; " +
        ("neither rule for blocked crossings (L-B, L-B′) was adopted." if n2 is None
         else f"the rule for blocked crossings {n2} was adopted."))
    if outcome["gate"] == "not run":
        t["S2|text|VIID|h2"] = ("No estimator change was adopted, so highway processing is unchanged and the highway "
                                "check was not run; gate G does not apply.")
    if est is None and n2 is None and tpl in ("L0", "L0_k0", "L1", "L2"):
        t["S2|text|VIID|yield"] = YIELD_SENTENCE
    if b_unquantified(crit):
        t["S2|text|VIID|splace2"] = SPLACE2_UNQUANTIFIED
    t["S2|text|VIID|highway_survey"] = "Study-1 highway qualification used the free-road survey."
    t["S2|text|VIID|r8iii"] = ("The rank correlation between the survey standard deviation and the test standard "
                               "deviation (qualification-validity row (iii)) is " +
                               " and ".join(f"{_spn(f'S2|{c}|r8iii')} over {_spn(f'S2|{c}|r8iii_n')} sites at "
                                            f"{_spn(f'S2|{c}|v0')}" + r"\,m/s" for c in U2_CELLS) + ".")
    t["S2|text|VIID|l1_statistic"] = (
        "The bias statistic of the validity rung is the calibrated bias of the between-site check and its "
        "site-bootstrap interval over all sites with a valid "
        "survey (generator arm), with its Holm-adjusted equivalence test at a band of " + _spn("S2|l1_band") + "\\,\\% on "
        "the same draws; it differs from the calibrated bias of Table A, which uses qualified sites and gated passes, and "
        "both are printed.")
    for c in U2_CELLS:
        r = outcome["per_cell"][c]
        if r in ("L1", "L2"):
            t[f"S2|text|VIID|l1|{c}"] = _at(c) + " the calibrated bias is within $\\pm$" + _spn("S2|l1_band") + "\\,\\%."
        if crit["cells"][c]["scope"]["k"] == 0:
            t[f"S2|text|VIID|k0|{c}"] = (_at(c) + " the traffic survey qualified no site; the rank correlation over "
                                         + _spn(f"S2|{c}|r8iii_n") + f" sites (row (iii)) is {_spn(f'S2|{c}|r8iii')}.")
    t["S2|text|VIID|N2"] = ("The blocked urban crossings (NLOSv) are reported with the study-1 NLOSv template on the "
                            "study-2 records; they are not on the ladder.")
    if "P3-15" in crit["cells"]:
        sc3 = crit["cells"]["P3-15"]["scope"]
        t["S2|text|T3N"] = t3n_text(sc3["k"], sc3.get("rating"))
    if est in ("E-c", "E-d", "E-db"):
        t["S2|text|avoidance"] = (
            "A vehicle that stops or brakes deliberately inside the heads yields no entry at any site type where the "
            "configuration runs; urban: the yield loss is reported as uncovered traffic; highway "
            "(H2): the H2 yield loss and λ_eff are printed and no highway claim is made.")
    if est in ("E-d", "E-db"):
        t["S2|text|stop_detector"] = (
            "The stop detector is a low-speed-run detector: a crossing whose Doppler level inside the heads stays below "
            f"the threshold (speed level {_spn('S2|k_stop_speed')} at |g| = 1), including a steady slow crawl without a "
            "stop, yields no entry.")
    if n2 in ("L-B", "L-B′"):
        t["S2|text|N2_avoidance"] = (
            "A crossing whose head-A beacons are flagged as blocked (L-B) or whose ρ̂ disagrees with the survey (L-B′) "
            "yields no N2 entry; the yield loss is reported as uncovered traffic."
            + (" A distortion that moves ρ̂ onto another surveyed lane's value is not flagged." if n2 == "L-B′" else ""))
    if tpl == "G_fails":
        t["S2|text|all_rungs"] = "configuration not adopted (highway regression, Section VI-D)"
    elif tpl == "L3":
        t["S2|text|abstract_clause"] = l3_abstract_clause(named)
        t["S2|text|urban_pointer"] = "(urban: study 2, Section VI-D)"
        t["S2|text|VB_survey"] = "a survey in traffic qualifies urban sites (Section VI-D)"
    elif tpl == "L2":
        t["S2|text|VIIC_conclusion"] = (
            "qualified at " + " and ".join(f"{_spn(f'S2|{c}|k')} of {_spn(f'S2|{c}|n')}" for c in named) +
            " held-out urban sites with σ " + " and ".join(_spn(f"S2|{c}|sigma_governing") + "\\,\\%" for c in named) +
            f" at {_cells_phrase(named)}; not covered at the decision level")
    elif tpl == "L1":
        t["S2|text|VIIC"] = L1_VIIC
    elif tpl == "L0":
        t["S2|text|VIIC"] = "a second, separately pre-specified design (Section VI-D) also did not cover them"
    elif tpl == "L0_k0":
        t["S2|text|VIIC"] = "a second, separately pre-specified design (Section VI-D) also qualified no held-out urban site"
    lt = summary(tpl, named)
    if lt:
        t["S2|text|summary"] = lt
    return t


def b_unquantified(crit):
    return ((crit.get("s_place") or {}).get("study2") or {}).get("status") == "unquantified"


def table_b_keys(adapter_dir, tag):
    """At L3: the reports/make_reports functions for the Table B stratum rows,
    Table B (honest bound, escalation status) and the matching-simulation keys, applied to the decision adapter's outputs on the
    held-out records (tables_, e2_extracted_, e5_extracted_<tag>.json); every key prefixed S2|... Returns (keys, notes)."""
    from reports import make_reports as MR

    def load(name):
        p = os.path.join(adapter_dir, f"{name}_{tag}.json")
        return json.load(open(p)) if os.path.exists(p) else None
    tables, e2, e5 = load("tables"), load("e2_extracted"), load("e5_extracted")
    if tables is None or e2 is None or e5 is None:
        raise RuntimeError(f"L3: adapter outputs missing in {adapter_dir} (tables / e2_extracted / e5_extracted)")
    nums = {}
    notes = MR.stratum_rows(tables, e2, nums)
    MR.table_b(e2, nums, strict=False)
    MR.e5_matching_keys(e5, nums, strict=False)
    return {"S2|" + k: v for k, v in nums.items()}, notes


def write(crit, outcome, dec, out_dir, tag, adapter_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    nums, br, tx = numbers(crit, outcome, dec), branches(outcome, crit, dec), texts(outcome, crit, dec)
    notes = []
    if outcome["template"] == "S2|L3":                    # Table B on the study-2 records (honest bound printed)
        if not adapter_dir:
            raise RuntimeError("L3: the adapter output folder is required for the Table B keys")
        bk, notes = table_b_keys(adapter_dir, tag)
        nums.update(bk)
    for name, obj in ((f"s2_numbers_{tag}.json", dict(numbers=nums)),
                      (f"s2_texts_{tag}.json", dict(branches=br, texts=tx, table_b_notes=notes))):
        json.dump(obj, open(os.path.join(out_dir, name), "w"), indent=1, default=float)
    lines = ["% written by reports/s2_texts.py; do not edit", r"\makeatletter"]
    lines += [r"\expandafter\def\csname spn@" + k + r"\endcsname{" + fmt(k, v) + "}" for k, v in sorted(nums.items())]
    lines += [r"\expandafter\def\csname spb@" + k + r"\endcsname{" + v + "}" for k, v in sorted(br.items())]
    lines += [r"\makeatother"]
    p = os.path.join(out_dir, "speedpath-s2.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------------------------- D0.1 printing
def d0_1_cell(entry):
    """One D0.1 stratum (s2_selection.d0_1: bias, sigma (residuals clipped at +-25 %, stage3_criteria.sigma), n) as printed in the
    supplement. sigma = 0 with n >= 2 and |bias| > 25 % means every residual was clipped on the same side: printed as
    "all |e| > 25 %"; n < 2: sigma is not defined. n is always printed."""
    b, s, n = entry.get("bias"), entry.get("sigma"), int(entry.get("n", 0))
    bs = f"{100 * b:.1f} %" if _fin(b) else "n/a"
    if n < 2:
        ss = "n/a (n < 2)"
    elif _fin(s) and s == 0.0 and _fin(b) and abs(b) > WIN:
        ss = "all |e| > 25 %"
    else:
        ss = f"{100 * s:.1f} %" if _fin(s) else "n/a"
    return dict(bias=bs, sigma=ss, n=str(n))


def d0_1_rows(d01):
    """Every D0.1 key (cell|arm|mapping|prb|stat|stratum) and group as supplement rows."""
    rows = []
    for k in sorted(d01):
        for g in sorted(d01[k]):
            rows.append(dict(key=k, group=g, **d0_1_cell(d01[k][g])))
    return rows


# ---------------------------------------------------------------------------------------------- abstract <= 250 words
def _arg(s, i):
    """The balanced {...} argument starting at s[i] == '{'; returns (content, index after)."""
    if i >= len(s) or s[i] != "{":
        raise ValueError(f"expected '{{' at {i}: {s[i:i + 30]!r}")
    d, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def _skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def expand(s, choice, defs, dcn):
    """Expand the abstract's macros under one branch choice. choice: {'sphwonly': 0|1, 'spfeas': 0|1,
    'spb|<key>': value}. Numbers become the token 0.0; design constants their value; unknown macros keep their text."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "%" and (i == 0 or s[i - 1] != "\\"):
            j = s.find("\n", i)
            i = len(s) if j < 0 else j + 1
            continue
        if c != "\\":
            out.append(" " if c == "~" else ("" if c in "{}" else c))
            i += 1
            continue
        m = re.match(r"\\([A-Za-z@]+)\*?", s[i:])
        if not m:
            out.append(" ")
            i += 2
            continue
        name, i = m.group(1), i + m.end()
        if name == "revdel":
            _, i = _arg(s, _skip_ws(s, i))
        elif name in ("rev", "textit", "textbf", "emph"):
            a, i = _arg(s, _skip_ws(s, i))
            out.append(expand(a, choice, defs, dcn))
        elif name in ("spn",):
            _, i = _arg(s, i)
            out.append("0.0")
        elif name == "dcn":
            a, i = _arg(s, i)
            out.append(dcn.get(a, "0.0"))
        elif name in ("ref", "label", "cite", "eqref"):
            _, i = _arg(s, i)
            out.append("1")
        elif name == "sprange":
            a, i = _arg(s, i)
            _, i = _arg(s, i)
            out.append(expand(a, choice, defs, dcn) + "–0.0")
        elif name in ("sphwonly", "spfeas"):
            a, i = _arg(s, i)
            b, i = _arg(s, i)
            out.append(expand((a, b)[choice[name]], choice, defs, dcn))
        elif name == "spif":
            cond, i = _arg(s, i)
            val, i = _arg(s, i)
            body, i = _arg(s, i)
            key = re.match(r"\\spb\{([^{}]*)\}", cond.strip()).group(1)
            if choice.get("spb|" + key) == val:
                out.append(expand(body, choice, defs, dcn))
        elif name in defs:
            out.append(expand(defs[name], choice, defs, dcn))
        elif name.startswith("sp@"):
            out.append("0.0")
        else:
            out.append(" ")
    return "".join(out)


def word_count(text):
    return sum(1 for w in text.split() if re.search(r"[A-Za-z0-9]", w))


def abstract_parts(tex):
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not a:
        raise ValueError("no abstract")
    defs = {}
    for m in re.finditer(r"\\newcommand\{\\(sp[A-Za-z@]+)\}(?!\[)", tex):
        try:
            defs[m.group(1)], _ = _arg(tex, m.end())
        except ValueError:
            pass
    spb = sorted(set((k, v) for k, v in re.findall(r"\\spif\{\\spb\{([^{}]*)\}\}\{([^{}]*)\}", a.group(1))))
    return a.group(1), defs, spb


# At L3 this study-1 clause of the abstract (the
# Bstratum|urban = not_covered_unqualified branch; its content stays in Section VI-D) is replaced by the L3 clause
STUDY1_URBAN_CLAUSE = "; the survey underestimates their error in traffic, which exceeds the qualification limit"
ABSTRACT_LIMIT = 250


def _with_clause(base, clause_txt):
    """The abstract at L3: the study-1 urban clause, if printed in this branch, is replaced by the L3 clause; otherwise
    the clause follows the study-1 sentence."""
    if STUDY1_URBAN_CLAUSE in base:
        return base.replace(STUDY1_URBAN_CLAUSE, " ; " + clause_txt, 1)
    return base + " ; " + clause_txt


def abstract_word_counts(tex, design_constants=None):
    """Word counts of the abstract for every branch combination (sphwonly x spfeas x each spb key's tested values plus
    'other'), each without and with the study-2 L3 clause naming one or two cells (with the clause, the replacement rule
    of _with_clause). Returns [(choice, clause, words)]."""
    body, defs, spb = abstract_parts(tex)
    dcn = _dcn(design_constants)
    keys = sorted({k for k, _ in spb})
    vals = {k: sorted({v for kk, v in spb if kk == k}) + ["other"] for k in keys}
    clauses = [None, ("P1-15",), ("P1-10", "P1-15")]
    out = []
    for hw, fe, *bv in itertools.product((0, 1), (0, 1), *[vals[k] for k in keys]):
        choice = dict(sphwonly=hw, spfeas=fe, **{"spb|" + k: v for k, v in zip(keys, bv)})
        base = expand(body, choice, defs, dcn)
        for cl in clauses:
            txt = base if cl is None else _with_clause(base, expand(l3_abstract_clause(cl), choice, defs, dcn))
            out.append((choice, cl, word_count(txt)))
    return out


def _dcn(design_constants=None):
    return {c["key"]: c["value"] for c in (design_constants or json.load(open(os.path.join(
        ER, "config", "design_constants.json"))))["constants"]}


NOT_COVERED = ("not_covered_floor", "not_covered_unqualified", "not_covered_c5", "no_records")


def realised_choice(tex, branches_tex):
    """The realised study-1 branch of the abstract from the generated branch file (speedpath/speedpath-branches.tex):
    \\sphwonly prints its first argument iff Bstratum|urban is not covered and Bstratum|highway is covered or
    covered_short (access.tex \\sphwonly / \\spstratum); \\spfeas its first argument unless feas|status = feasibility;
    each tested \\spb key its realised value ('other' if untested)."""
    br = dict(re.findall(r"\\csname spb@([^\\]*)\\endcsname\{([^{}]*)\}", branches_tex))
    _, _, spb = abstract_parts(tex)
    hw_scoped = br.get("Bstratum|urban") in NOT_COVERED and br.get("Bstratum|highway") in ("covered", "covered_short")
    choice = dict(sphwonly=0 if hw_scoped else 1, spfeas=1 if br.get("feas|status") == "feasibility" else 0)
    for k in sorted({k for k, _ in spb}):
        tested = {v for kk, v in spb if kk == k}
        choice["spb|" + k] = br.get(k) if br.get(k) in tested else "other"
    return choice


def abstract_check(tex, branches_tex=None, design_constants=None):
    """The pre-specified abstract rule (abstract <= 250 words): every branch without the study-2 clause is <= 250
    words, and the realised study-1 branch with the two-cell L3 clause (L3 needs both U2 cells at L2) is <= 250 words. The one-cell count is reported only."""
    res = abstract_word_counts(tex, design_constants)
    out = dict(max_without=max(w for _, cl, w in res if cl is None), combinations=len(res))
    if branches_tex is not None:
        body, defs, _ = abstract_parts(tex)
        dcn = _dcn(design_constants)
        ch = realised_choice(tex, branches_tex)
        base = expand(body, ch, defs, dcn)
        out["realised_without"] = word_count(base)
        for name, cl in (("realised_one_cell", ("P1-15",)), ("realised_two_cells", ("P1-10", "P1-15"))):
            out[name] = word_count(_with_clause(base, expand(l3_abstract_clause(cl), ch, defs, dcn)))
        out["ok"] = bool(out["max_without"] <= ABSTRACT_LIMIT and out["realised_two_cells"] <= ABSTRACT_LIMIT)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstract", help="paper .tex: the abstract <= 250 words rule (with --branches: on the realised branch)")
    ap.add_argument("--branches", help="the generated study-1 branch file speedpath/speedpath-branches.tex")
    ap.add_argument("--criteria"), ap.add_argument("--outcome"), ap.add_argument("--decision")
    ap.add_argument("--adapter-dir", help="the adapter outputs of stage 3 (Table B keys at L3)")
    ap.add_argument("--out"), ap.add_argument("--tag")
    a = ap.parse_args()
    if a.abstract:
        r = abstract_check(open(a.abstract).read(), open(a.branches).read() if a.branches else None)
        print(f"{r['combinations']} combinations; max {r['max_without']} words without the study-2 clause"
              + (f"; realised branch {r['realised_without']} words, with the L3 clause {r['realised_two_cells']} (two "
                 f"cells) / {r['realised_one_cell']} (one cell); <= 250: {r['ok']}" if "ok" in r else
                 f"; <= 250 without the clause: {r['max_without'] <= ABSTRACT_LIMIT} (give --branches for the L3 rule)"))
        return
    print(write(json.load(open(a.criteria)), json.load(open(a.outcome)), json.load(open(a.decision)), a.out, a.tag,
                a.adapter_dir))


if __name__ == "__main__":
    main()
