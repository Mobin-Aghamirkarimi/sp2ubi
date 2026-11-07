"""Study 2 stage 3: the two pre-specified additions to the study-1 stage-3 path and the study-2 outcome rules.
pipeline/stage3_criteria.py and pipeline/stage3_extra_checks.py are not changed: the study-2 mode calls their functions
on records mapped here.
  1. mode s2: use_statistic_s2 puts the study-2 configuration's statistic and inclusion into the fields
     every criterion reads (match, prof, finite, window_ok), as mode d does for the Huber fit (d); mse stays the plain-LS
     residual (gate, monitor). An exclusion of the configuration (E-c, E-d / E-db no entry, L-B, L-B′) is a "no entry":
     window_ok = False with s2_excluded = True. An exclusion applies to survey
     and test passes alike (the RSU applies one entry rule to every crossing; a site whose included survey falls below
     MIN_REF then yields no entries, counted in the yield). Addition 1 also holds the N16 C2 band: c2_tost_s2
     recomputes the C2 TOST on the same site-bootstrap draws with the band 3 % (stage3_criteria.D_MAX untouched).
  2. the study-2 Holm family (N4): m = 3 over U2 P1-10, U2 P1-15, N2 P3-15 for every study-1 Holm family over
     cells (C2 TOST, optimism check): holm_family_s2 runs the stage3_criteria functions with that family.
Also: readings F / T of the development records, gate G (N3), the ladder L0-L3 (N1, N2), the S-place rule (N5) and the
outcome -> text map. The new thresholds are the NEW constants of pipeline/s2_constants.py."""
import os
import sys
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import stage3_criteria as K  # noqa: E402
from pipeline import s2_constants as S2  # noqa: E402

LADDER = ("L0", "L1", "L2")
RUNG_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


# ---------------------------------------------------------------------------------------------- readings
def reading_F(recs, dev2):
    """Reading F: free-road reference as recorded. On Dev-2 the test passes are those with jr >= 30 (the surrogate half
    is never a test pass); Dev-1: all test passes."""
    if not dev2:
        return list(recs)
    return [r for r in recs if r["role"] != "test" or (r["jr"] is not None and r["jr"] >= S2.SURROGATE_SPLIT_JR)]


def reading_T(recs):
    """Reading T (Dev-2 only, L-T surrogate): the test passes jr < 30 act as the survey (role 'ref', surrogate = True)
    for calibration, gate and qualification; the test passes jr >= 30 are the test; the free-road references are not
    used. C5 quadratic passes are dropped (none on Dev-2)."""
    out = []
    for r in recs:
        if r["role"] != "test" or r.get("kin", "sumo") != "sumo":
            continue
        if r["jr"] is None:
            raise ValueError("reading T needs jr on every test record")
        q = dict(r)
        if r["jr"] < S2.SURROGATE_SPLIT_JR:
            q["role"], q["surrogate"] = "ref", True
        out.append(q)
    return out


# ---------------------------------------------------------------------------------------------- mode s2
def _set_stat(q, match, prof):
    q["match"], q["prof"] = match, prof
    q["finite"] = bool(np.isfinite(q["match"]) and np.isfinite(q["prof"]))


def _exclude(q):
    q["window_ok"], q["s2_excluded"] = False, True


def stop_detected(r, cal):
    """The E-d / E-db RSU-only low-speed-run (stop) detector (N12) under the pre-specified span and scale: stop_abs (Hz, the
    smallest run maximum of |FDOA - δ̂| inside the inter-head span) < k_stop x the site's survey rms_ref_med (annotated
    by annotate_rms_ref_med on the same records; reading T: the surrogate survey). A missing site scale -> no detection
    (the site has no qualified survey and yields no entry anyway)."""
    rm = r.get("rms_ref_med", np.nan)
    return bool(np.isfinite(rm) and r.get("stop_abs", np.nan) < cal["k_stop"] * rm)


def is_survey(r):
    """A survey pass of the reading (role 'ref'; reading T: the surrogate), SUMO, beta = 0, included (stage3_criteria.included)."""
    return r["role"] == "ref" and r.get("kin", "sumo") == "sumo" and r.get("beta", 0.0) == 0.0 and K.included(r)


def annotate_rms_ref_med(recs):
    """Add rms_ref_med (median sqrt(mse) of the site's included survey passes, same cell / variant / arm: the study-1 gate's
    scale, stage3_criteria.site_table) to every record; the survey is role 'ref' of these records (reading T: the surrogate)."""
    by = {}
    for r in recs:
        if is_survey(r):
            by.setdefault((r["site"], r["cell"], r["v0"], r["arm"], r["mapping"], r["prb"]), []).append(np.sqrt(r["mse"]))
    med = {k: float(np.median(v)) for k, v in by.items()}
    out = []
    for r in recs:
        q = dict(r)
        q["rms_ref_med"] = med.get((r["site"], r["cell"], r["v0"], r["arm"], r["mapping"], r["prb"]), np.nan)
        out.append(q)
    return out


def annotate_rho_dev(recs):
    """Add rho_dev (N18, L-B′): min over the site's surveyed lanes of |ρ̂ - median survey ρ̂ of the lane|, the survey
    = the reading's included survey passes (their lane known by design), leave-one-out for a survey pass itself; the test pass's SUMO lane is never used."""
    from pipeline.s2_fields import rho_lane_offset
    dev = rho_lane_offset(recs, is_survey, K.lane_width)
    return [dict(r, rho_dev=d) for r, d in zip(recs, dev)]


def check_n2(n2):
    if n2 not in (None,) + S2.LB_RULES:
        raise ValueError(f"not a pre-specified N2 rule: {n2!r} (pre-specified: {S2.LB_RULES})")
    return n2


def map_records(recs, cand, cal, reading="F", e_d_variant="no_entry", n2=None):
    """map_record over a record set, with the site annotations the configuration needs (rms_ref_med for E-d / E-db,
    rho_dev for L-B′), computed on the unmapped records (one pass, no iteration)."""
    if cand in ("E-d", "E-db"):
        recs = annotate_rms_ref_med(recs)
    if check_n2(n2) == "L-B′":
        recs = annotate_rho_dev(recs)
    return [map_record(r, cand, cal, reading, e_d_variant, n2) for r in recs]


def map_record(r, cand, cal, reading="F", e_d_variant="no_entry", n2=None):
    """One record under a configuration: cand in (None = plain LS, 'E-a', ..., 'E-e', 'E-db'); cal = the calibration dict
    (a_max, k_stop, delta_blk_db, delta_rho); reading 'T' selects the reading-T E-e fields; n2: the N2 inclusion rule
    (None, 'L-B' or 'L-B′'; N2 only). E-d has one variant, 'no entry'; any other is refused."""
    if e_d_variant not in S2.E_D_VARIANTS:
        raise ValueError(f"E-d variant {e_d_variant!r} is not pre-specified (pre-specified: {S2.E_D_VARIANTS})")
    check_n2(n2)
    q = dict(r)
    if cand in (None, "LS"):
        pass
    elif cand == "E-a":
        _set_stat(q, r.get("w_match", np.nan), r.get("w_prof", np.nan))
    elif cand == "E-b":                                   # matched forced: no per-site select (prof := match)
        _set_stat(q, r["match"], r["match"])
    elif cand == "E-c":
        a = r.get("acc_fit", np.nan)
        if not (np.isfinite(a) and abs(a) <= cal["a_max"]):
            _exclude(q)
    elif cand in ("E-d", "E-db"):
        if stop_detected(r, cal):                         # "no entry"
            _exclude(q)
        if cand == "E-db" and not q.get("s2_excluded"):      # E-d detector, statistic forced to E-f
            _set_stat(q, r.get("c_match", np.nan), r.get("c_prof", np.nan))
    elif cand == "E-e":
        suf = "_T" if reading == "T" else ""
        _set_stat(q, r.get("match_2s" + suf, np.nan), r.get("prof_2s" + suf, np.nan))
    elif cand == "E-f":                                   # N17: spatial core [-0.05 D, 1.05 D] of the full fit
        _set_stat(q, r.get("c_match", np.nan), r.get("c_prof", np.nan))
    else:
        raise ValueError(f"not a candidate of the closed list: {cand!r}")
    if n2 == "L-B":
        d = r.get("blk_drop_max_db", np.nan)
        if np.isfinite(d) and d > cal["delta_blk_db"]:
            _exclude(q)
    elif n2 == "L-B′":
        if "rho_dev" not in r:
            raise ValueError("L-B′ needs rho_dev (map_records / use_statistic_s2 annotate it on the record set)")
        d = r["rho_dev"]
        if np.isfinite(d) and d > cal["delta_rho"]:
            _exclude(q)
    if cand not in (None, "LS") or n2:
        q["s2_config"] = dict(cand=cand, e_d_variant=e_d_variant if cand in ("E-d", "E-db") else None, n2=n2,
                              reading=reading)
    return q


def use_statistic_s2(recs, selected, cal, reading="F"):
    """Mode s2: the selected configuration (constants evaluation_v6.selected: estimator = candidate or None, e_d_variant,
    n2_rule = the adopted N2 rule 'L-B' / 'L-B′' or None) applied to every record; the N2 rule only on the N-type
    records (cell P3)."""
    cand = selected.get("estimator")
    n2 = check_n2(selected.get("n2_rule"))
    if cand in ("E-d", "E-db"):
        recs = annotate_rms_ref_med(recs)
    if n2 == "L-B′":
        recs = annotate_rho_dev(recs)
    return [map_record(r, cand, cal, reading, selected.get("e_d_variant") or "no_entry",
                       n2=n2 if r["cell"] == "P3" else None) for r in recs]


# ---------------------------------------------------------------------------------------------- Holm m = 3
@contextmanager
def holm_family_s2():
    """Run the study-1 Holm families over cells (stage3_criteria.c2_all, stage3_criteria.optimism_verdicts) with the
    study-2 family (N4, m = 3); the study-1 family is restored afterwards, also on an exception."""
    old = K.HOLM_FAMILY
    K.HOLM_FAMILY = [tuple(c) for c in S2.N4_HOLM_FAMILY]
    try:
        yield K.HOLM_FAMILY
    finally:
        K.HOLM_FAMILY = old


def c2_s2(recs, cells):
    """C2 with the study-2 Holm family, on every site
    with a valid survey (l1_c2_sites = "all"). Reported; the L1 gate reads c2_tost_s2."""
    with holm_family_s2():
        return K.c2_all(recs, [c for c in cells if c in K.HOLM_FAMILY])


def c2_boot_means(recs, cell, v0, mapping, prb):
    """The C2 site-bootstrap means of c2_cell (stage3_criteria.py), the same draws (K.draws key ("C2", cell, v0, mapping,
    prb)) and the same pooling; None with fewer than 2 sites (as c2_cell)."""
    T, _ = K.site_table(recs, cell, v0, "gen_on", mapping, prb)
    sites = sorted(T)
    if len(sites) < 2:
        return None
    D = K.draws(("C2", cell, v0, mapping, prb), len(sites))
    return [np.mean(K.pooled(T, [sites[i] for i in idx])) for idx in D]


def p_tost_band(mb, band):
    """The two one-sided bootstrap p of c2_cell (stage3_criteria.py) at a given band: max(P(b >= band), P(b <= -band))."""
    return max(K.pval(mb, lambda b: b >= band), K.pval(mb, lambda b: b <= -band))


def c2_tost_s2(recs, cells, band=S2.N16_L1_BIAS_BAND):
    """N16, the L1 bias test: per (cell, mapping, PRB) the C2 TOST on the same site-bootstrap draws with the separate
    band of mode s2 (3 %), Holm over the m = 3 family (N4) per (mapping, PRB) (a missing cell enters with p = 1, as
    c2_all), alpha = 0.05. stage3_criteria.D_MAX (0.5 %) is neither read nor changed. Rows carry the study-1 C2 fields of
    c2_s2 (bias, bias_ci, the 0.5 % p_tost and its Holm value, excess test) beside p_tost_s2 / p_tost_s2_holm /
    tost_s2_pass."""
    rows, summ = c2_s2(recs, cells)
    fam = [tuple(c) for c in S2.N4_HOLM_FAMILY]
    for r in rows:
        r["band_s2"] = float(band)
        r["p_tost_s2"] = float(p_tost_band(c2_boot_means(recs, r["cell"], r["v0"], r["mapping"], r["prb"]), band))
    for mp, prb in K.COMBOS:
        by = {(r["cell"], r["v0"]): r for r in rows if r["mapping"] == mp and r["prb"] == prb}
        adj = K.holm([by[c]["p_tost_s2"] if c in by else 1.0 for c in fam])
        for c, a in zip(fam, adj):
            if c in by:
                by[c]["p_tost_s2_holm"] = float(a)
                by[c]["tost_s2_pass"] = bool(a <= K.ALPHA)
    return rows, summ


def optimism_s2(rows):
    with holm_family_s2():
        return K.optimism_verdicts(rows)


# ---------------------------------------------------------------------------------------------- S-place (N5)
def s_place_factor(f_study1, f_study2):
    """N5: the governing urban S-place factor is the larger of the study-1 and the study-2 factors (a missing factor
    never lowers it)."""
    vals = [f for f in (f_study1, f_study2) if f is not None and np.isfinite(f)]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------------------------- gate G (N3)
def t_all_ok(t_cfg, t_ls, n):
    """T_all(config) <= T_all(LS) + max(0.05 T_all(LS), 1 / n): the larger of 5 % and one crossing."""
    return bool(np.isfinite(t_cfg) and np.isfinite(t_ls) and t_cfg <= t_ls + max(S2.N3_G_T_REL * t_ls, 1.0 / max(n, 1)))


def gate_G(h2_ran, coverage_rating, sigma_governing, s_all_cfg, s_all_ls, t_all_cfg, t_all_ls, n_crossings,
           sigma_ratio_study1=None, sigma_ratio_ci=None, s_ratio_ci=None, t_ratio_ci=None, paired=None, yield_ratio=None):
    """Gate G for the H2 cell P2-20 (worst variant, worse-of). Returns dict(status = 'not run' | 'holds' | 'fails', the
    clauses, the reported-only values). (a) the coverage rule (rating covered and governing sigma <= 6 %);
    (b) on the same H2 records (same input passes), each configuration on its own included and gated passes (as
    D0-R3): point ratios S_all(config) / S_all(LS) <= 1.05 and T_all within max(5 %, one crossing) (the
    caller passes own-set values; the paired common-set ratios and the H2 yield ratio are printed, never gating);
    (c) the ratio of the governing sigma to the study-1 P2-20 value: reported, never gating."""
    if not h2_ran:
        return dict(status="not run", reason="no estimator candidate adopted (D0-R5): H2 not run, G not applicable")
    a = bool(coverage_rating == "covered" and np.isfinite(sigma_governing) and sigma_governing <= K.SIG_Q)
    s_ratio = K._div(s_all_cfg, s_all_ls)
    b_s = bool(np.isfinite(s_ratio) and s_ratio <= S2.N3_G_S_RATIO_MAX)
    b_t = t_all_ok(t_all_cfg, t_all_ls, n_crossings)
    return dict(status="holds" if (a and b_s and b_t) else "fails", a_coverage=a, b_S_ratio=s_ratio, b_S_ok=b_s,
                b_T_ok=b_t, ci_reported=dict(S_ratio=s_ratio_ci, T_ratio=t_ratio_ci),
                printed=dict(paired_common_set=paired, h2_yield_ratio=yield_ratio),
                c_reported=dict(sigma_ratio_study1=sigma_ratio_study1, ci=sigma_ratio_ci))


# ---------------------------------------------------------------------------------------------- ladder (N1, N2)
def l1_bias_ok(c2_rows):
    """N16, the L1 bias criterion of one U2 cell: in every variant (every (mapping, PRB) row of C2,
    gen_on) the Holm-adjusted s2 TOST passes (p_tost_s2_holm <= 0.05, band 3 %, m = 3) and C2 (ii) detects no excess.
    c2_rows: the cell's rows of c2_tost_s2. None if C2 has no row (not computable). bias_ci is printed, not gating."""
    if not c2_rows:
        return None
    return bool(all(r.get("tost_s2_pass") is True and not r["excess_detected"] for r in c2_rows))


def l1_holds(k, tost_pass, n_over_9pct):
    """L1 validity: k >= 1 (N1); the bias criterion (tost_pass = l1_bias_ok of the cell, N16); R8 (ii) as a gate:
    <= 1 of the k qualified sites with test sigma > 9 % (N2)."""
    return bool(k >= S2.N1_K_MIN_QUALIFIED and tost_pass and n_over_9pct <= S2.N2_R8II_MAX_OVER)


def l2_holds(k, n, sigma_governing):
    """L2 precision: the study-1 POINT rating k / n >= 0.3 (partly covered or covered; CP reported) and governing sigma <= 6 %
    (the S-place factor of N5 already applied)."""
    return bool(n > 0 and k / n >= K.SCOPE["partly_covered"] and np.isfinite(sigma_governing) and sigma_governing <= K.SIG_Q)


def cell_rung(k, n, tost_pass, n_over_9pct, sigma_governing):
    """L0 / L1 / L2 of one U2 cell (each rung needs every lower one). k = 0 -> L0 'no site qualified' (C2 and R8 (ii) are
    not evaluated then); tost_pass None (C2 not computable, < 2 sites) -> L1 fails with that reason."""
    if k < S2.N1_K_MIN_QUALIFIED:
        return dict(rung="L0", no_site_qualified=True, reason="k = 0", cp95=K.clopper_pearson(k, n))
    if tost_pass is None:
        return dict(rung="L0", no_site_qualified=False, reason="C2 not computable", cp95=K.clopper_pearson(k, n))
    if not l1_holds(k, tost_pass, n_over_9pct):
        return dict(rung="L0", no_site_qualified=False,
                    reason="C2 bias criterion fails" if not tost_pass else "R8 (ii) gate fails", cp95=K.clopper_pearson(k, n))
    if not l2_holds(k, n, sigma_governing):
        return dict(rung="L1", cp95=K.clopper_pearson(k, n))
    return dict(rung="L2", cp95=K.clopper_pearson(k, n))


def l3_holds(bstratum_urban_covered, adapter_covered_cells, rungs):
    """L3 (per stratum): Bstratum|urban covered by the adapter rule and every cell the adapter counts as covered reaches L2."""
    return bool(bstratum_urban_covered and adapter_covered_cells
                and all(rungs.get(c) == "L2" for c in adapter_covered_cells))


# ---------------------------------------------------------------------------------------------- outcome -> text
def select_outcome(gate, rungs, l3, k_zero=None):
    """The complete outcome map. gate: 'holds' | 'fails' | 'not run'; rungs: {'P1-10': 'L0'|'L1'|'L2', 'P1-15': ...};
    l3: bool (stratum). Returns dict(template, named_cells, per_cell). G fails -> the G text whatever the rungs.
    Otherwise the template of the highest rung reached by any U2 cell (L3 for the stratum); the abstract / conclusion
    clause names only the cells that reach that rung; every other cell is printed in Section VI-D / Table A2 with its own rung."""
    if gate not in ("holds", "fails", "not run"):
        raise ValueError(gate)
    for c in ("P1-10", "P1-15"):
        if rungs.get(c) not in LADDER:
            raise ValueError(f"{c}: rung {rungs.get(c)!r}")
    per_cell = dict(rungs)
    if gate == "fails":
        return dict(template="S2|G_fails", named_cells=[], per_cell=per_cell,
                    text_key="configuration not adopted (highway regression, Section VI-D)", gate=gate)
    if l3:
        top = "L3"
        named = [c for c in ("P1-10", "P1-15") if rungs[c] == "L2"]
        if not named:
            raise ValueError("L3 without a cell at L2 cannot occur (l3_holds: every adapter-covered cell reaches L2)")
    else:
        top = max((rungs[c] for c in ("P1-10", "P1-15")), key=lambda r: RUNG_ORDER[r])
        named = [c for c in ("P1-10", "P1-15") if rungs[c] == top]
    k0 = bool(top == "L0" and k_zero and all(k_zero.get(c) for c in named))
    return dict(template=f"S2|{top}" + ("_k0" if k0 else ""), named_cells=named, per_cell=per_cell, gate=gate,
                h2="not run" if gate == "not run" else "run")


def all_outcomes():
    """Every combination of the outcome map: {G holds / fails / not run} x {P1-10, P1-15 rung} x {L3 yes / no}, plus the k = 0
    variant of L0. L3 = yes with no cell at L2 cannot occur (l3_holds) and is listed as impossible, except under G fails,
    whose text does not depend on the rungs."""
    out = []
    for g in ("holds", "fails", "not run"):
        for r10 in LADDER:
            for r15 in LADDER:
                for l3 in (False, True):
                    for kz in ((False, True) if "L0" in (r10, r15) else (False,)):
                        key = (g, r10, r15, l3, kz)
                        kzd = {c: kz for c, r in (("P1-10", r10), ("P1-15", r15)) if r == "L0"}
                        try:
                            out.append((key, select_outcome(g, {"P1-10": r10, "P1-15": r15}, l3, kzd)))
                        except ValueError:
                            out.append((key, dict(template="impossible")))
    return out
