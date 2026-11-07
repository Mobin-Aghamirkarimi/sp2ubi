"""Study 2: the pre-specified values that are new in study 2 (thresholds N1-N18) and the fixed design items, as literals
in one place. No import, no I/O: make_constants.py --study2 copies them into the constants block evaluation_v6, and
every study-2 module reads them from here and checks that the constants agree (`check_against`; if they disagree the
constants govern and the run stops).

Every threshold that study 1 does not contain is marked NEW. Values that study 1 already fixes are taken from the
study-1 constants by the modules (SIG_Q, gate, window, MIN_REF, C2 band, ratings) and are not repeated here.
The four calibration values a_max (N11), k_stop (N12), Delta_blk (N13) and Delta_rho (N18) are not literals: they are
the outputs of pipeline/s2_calibrate.py on the Dev-1 survey passes (role == "ref" only), written into
evaluation_v6.step0.calibration by make_constants.py --study2 before the development stage; only their rules
(quantile / false-detection share) are here. OPTIONS holds the pre-specified design choices only, and check_against
refuses an evaluation_v6 block whose options, candidate list, E-d variants or calibration keys differ from this module."""

SOURCE = "study-2 design (supplement, study 2)"

# ---------------------------------------------------------------------------------------------- new thresholds
N1_K_MIN_QUALIFIED = 1            # NEW: ladder rung L1, k >= 1 qualified site per cell
N2_R8II_SIGMA = 0.09              # NEW: rung L1, test sigma > 9 % (= study-1 R8 (ii) 1.5 x 6 %) ...
N2_R8II_MAX_OVER = 1              # NEW: ... at most 1 of the k qualified sites (point count; CP reported)
N3_G_S_RATIO_MAX = 1.05           # NEW: gate G (b), S_all(config) / S_all(LS) <= 1.05 (point), same H2 records
N3_G_T_REL = 0.05                 # NEW: gate G (b), T_all(config) <= T_all(LS) + max(0.05 T_all(LS), 1 / n)
N4_HOLM_FAMILY = (("P1", 10.0), ("P1", 15.0), ("P3", 15.0))   # NEW: Holm m = 3
N5_S_PLACE_RULE = "max"           # NEW: governing urban S-place factor = max(study 1, study 2)
N6_CAUSE_SHARE = 0.5              # NEW: D0-R1, mean paired difference >= 1/2 of the excess bias, CI excl. 0
N6_MIN_CELLS = 1                  # NEW: ... in >= 1 urban cell
N7_NONSTOP_SHARE = 0.5            # NEW: D0-R2, non-stop passes carry >= 1/2 of the excess bias
N8_BIAS_FALL = 0.5                # NEW: D0-R3, |excess bias| falls >= 50 % ...
N8_BIAS_BAND = 0.005              # NEW use: ... or is <= 0.5 %
N8_S_ALL_FALL = 0.20              # NEW: S_all falls >= 20 %
N8_Q99_MAY_RISE = 0.05            # NEW use: q99_all rises by <= 5 % (relative)
N8_DEV1_S_RISE = 0.05             # NEW use: Dev-1 non-regression guard, S_all rises <= 5 %
N8_DEV1_BIAS_RISE_ABS = 0.005     # NEW use: Dev-1 guard, |excess bias| rises <= 0.5 % absolute
N8_YIELD_LOSS_MAX = 0.30          # NEW: entry yield falls <= 30 % (relative)
N8_TIE_REL = 0.02                 # NEW: tie within 2 % (relative) -> earlier letter
N9_NONFINITE_MAX = 0.05           # NEW use: non-finite share <= 5 %
N9_P0H_S_REL = 0.05               # NEW use: P0H, S_all rises <= 5 %
N9_P0H_T_REL = 0.05               # NEW use: P0H, T_all rises <= max(5 %, one crossing)
N10_LB_BIAS_FALL = 0.5            # NEW: D0-R4 (as N8)
N10_LB_BIAS_BAND = 0.005          # NEW use
N10_LB_YIELD_LOSS_MAX = 0.30      # NEW
N11_A_MAX_QUANTILE = 0.99         # NEW: E-c, a_max = 99th percentile of |p[2]| on Dev-1 survey passes
N12_STOP_RUN_S = 1.0              # NEW: E-d, stop run spans >= 1 s
N12_FALSE_DETECT_MAX = 0.01       # NEW: k_stop = the largest value with <= 1 % false detections (Dev-1 survey;
                                  # false detections rise with k_stop)
N13_FALSE_FLAG_MAX = 0.01         # NEW: L-B, Delta_blk = smallest value with <= 1 % false flags (Dev-1 P0N survey)
N14_DEV3_SITE_TIMEOUT_H = 39      # NEW: Dev-3, ceil(23 h x 300 / 180) (operational only)
N18_FALSE_FLAG_MAX = 0.01         # NEW: L-B′, Delta_rho = smallest value with <= 1 % false flags (Dev-1 P0N survey,
                                  # leave-one-out lane medians for the survey pass itself)


# ---------------------------------------------------------------------------------------------- fixed design
S2_FAMILIES = {                   # held-out families: site ranges on the study-2 seed base
    "U2": dict(base="U", cells=["P1-10", "P1-15"], sites=[10, 20], ladder=True),
    "N2": dict(base="N", cells=["P3-15"], sites=[10, 20], ladder=False),
    "H2": dict(base="H", cells=["P2-20"], sites=[10, 15], ladder=False, only_if="D0-R5"),
}
S_PLACE_SITES = [20, 24]          # S-place light, sites 20-24 (inclusive) on the study-2 seed base, P1-15, 20 passes
S_PLACE_PASSES = 20
SURROGATE_SPLIT_JR = 30           # L-T: Dev-2 test passes jr < 30 = surrogate survey, jr >= 30 = test
DEV2_FAMILIES = ("U", "N")        # study-1 main-run U and N records (sites 0-9)
DEV1_FAMILIES = ("P0U", "P0N", "P0H")
DEV1_CAL_FAMILIES = {"a_max": ("P0U", "P0N"), "k_stop": ("P0U", "P0N"), "delta_blk": ("P0N",), "delta_rho": ("P0N",)}
DEV3 = dict(family="D3U", base="U", sites=[0, 5], tag="dev_s2", tracks_per_site=300,
            site_timeout_h=N14_DEV3_SITE_TIMEOUT_H, workers_stage1=1,
            parts=[["a", 0, 3], ["b", 3, 5]])           # part a = sites 0-2, part b = sites 3-4 (contiguous)
DEV_TAG = "dev_s2"                # development tag of the stage-2 re-runs and of Dev-3
CAL_TAG = "dev_s2_cal"            # calibration reads (Dev-1 reference passes only)
CANDIDATES = ("E-a", "E-b", "E-c", "E-f", "E-d", "E-db", "E-e")   # closed list, in tie order
E_D_VARIANTS = ("no_entry",)      # E-d: "no entry" only
LB_RULES = ("L-B", "L-B′")        # inclusion candidates of N2, D0-R4; ties go to L-B
SEED_NAMES = ("G1_site_base", "G1_pass_base", "sumo", "impairments", "bootstrap")
SEED_DOMAIN = "study2"
CALIBRATION_KEYS = ("a_max", "k_stop", "delta_blk_db", "delta_rho")

# Pre-specified design choices, checked by check_against; no alternative is implemented in the decision path.
OPTIONS = dict(
    junction="site",             # for seedset s2 one junction offset per site, U(50, 150) m from
                                 # default_rng(sd["site"] + [77]), shared by both roles and both speeds
    e_d_scale="site_ref_med",    # the site's survey median residual RMS (rms_ref_med)
    e_d_bracket="fdoa_or_fit",   # inter-head span = FDOA bracket (0.25 max |FDOA - δ̂|) ∪ x̂ in [0, D]
    l1_rule="tost_s2",           # C2 TOST, band 3 % in mode s2, Holm m = 3; D_MAX untouched
    l1_c2_sites="all",           # C2 on all sites with a valid survey
)
N12_BRACKET_FRAC = 0.25          # NEW: the FDOA head-bracketing fraction of max |FDOA - δ̂|
N15_JUNCTION_STREAM = 77          # NEW: the junction stream default_rng(sd["site"] + [77])
N15_JUNCTION_RANGE_M = (50.0, 150.0)   # study-1 (sumo_kinematics.py:155)
N16_L1_BIAS_BAND = 0.03           # NEW: the C2 TOST band of mode s2 (Holm m = 3 at alpha 0.05);
                                  # stage3_criteria.D_MAX = 0.5 % is unchanged everywhere (C1; C5; transfer)
N17_CORE_MIN_BEACONS = 12         # E-f: >= 12 core beacons (as stage2_evaluate.py); bounds = window_rule.arc_L

NEW_THRESHOLDS = {k: v for k, v in dict(globals()).items() if k[:1] == "N" and k[1:2].isdigit()}


def block():
    """The evaluation_v6 constants block without calibration and without seeds (make_constants.py --study2)."""
    return dict(source=SOURCE, new_thresholds=dict(NEW_THRESHOLDS),
                families=S2_FAMILIES, s_place=dict(sites=S_PLACE_SITES, passes=S_PLACE_PASSES, cell="P1-15"),
                surrogate=dict(split_jr=SURROGATE_SPLIT_JR, families=list(DEV2_FAMILIES)),
                dev1=dict(families=list(DEV1_FAMILIES), calibration_families=DEV1_CAL_FAMILIES),
                dev3=dict(DEV3, c5_families=[DEV3["family"]]),
                step0=dict(tag=DEV_TAG, cal_tag=CAL_TAG, candidates=list(CANDIDATES), e_d_variants=list(E_D_VARIANTS),
                           n2_rules=list(LB_RULES), calibration=None, timestamped=False, study1_merge_done=False),
                seed_rule=dict(names=list(SEED_NAMES), domain=SEED_DOMAIN,
                               formula="int(sha256(bundle_sha_hex + block_hash_hex + 'study2' + name)[:8], 16)"),
                options=dict(OPTIONS), selected=None)


def _tup(v):
    if isinstance(v, dict):
        return {k: _tup(x) for k, x in v.items()}
    return tuple(_tup(x) for x in v) if isinstance(v, (list, tuple)) else v


def check_calibration(cal):
    """Every calibration value is present and finite (an infinite k_stop would flag every pass with an
    admissible run); returns cal."""
    import math
    bad = [k for k in CALIBRATION_KEYS if not isinstance((cal or {}).get(k), (int, float))
           or isinstance(cal.get(k), bool) or not math.isfinite(cal[k])]
    if bad:
        raise RuntimeError(f"evaluation_v6.step0.calibration: missing or non-finite {bad} (pipeline/s2_calibrate.py)")
    return cal


DEV_FLAG_KEYS = ("step0.timestamped", "step0.study1_merge_done", "written_by")  # may change after the development record
EVAL_KEYS = ("selected", "seeds", "seed_record")                               # added when the evaluation is fixed


def _without(d, paths):
    import json
    d = json.loads(json.dumps(d, default=str))          # the JSON form (as read from the constants file)
    for p in paths:
        *head, last = p.split(".")
        x = d
        for k in head:
            x = x.get(k) if isinstance(x, dict) else None
        if isinstance(x, dict):
            x.pop(last, None)
    return d


def check_same_as_reference(ev6, reference, allowed=DEV_FLAG_KEYS):
    """The evaluation_v6 in force equals the reference block (the timestamped development-stage constants) except the
    keys in `allowed`. Raises otherwise; returns True."""
    if reference is None or ev6 is None:
        raise RuntimeError("evaluation_v6: the block in force or the reference block is missing")
    a, b = _without(ev6, allowed), _without(reference, allowed)
    if a != b:
        bad = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise RuntimeError(f"evaluation_v6 differs from the reference constants in {bad} (only {list(allowed)} may "
                           "differ)")
    return True


def check_against(ev6):
    """The constants govern: raise if evaluation_v6 differs from this module in the new thresholds, the pre-specified design
    choices (options), the candidate list, the E-d variants, the N2 rules or the calibration keys. A calibration block, if
    present, must hold exactly the pre-specified keys, each finite (plus its file and SHA-256)."""
    if ev6 is None:
        raise RuntimeError("the constants file has no evaluation_v6 block (make_constants.py --study2)")
    got = {k: _tup(v) for k, v in ev6.get("new_thresholds", {}).items()}
    if got != NEW_THRESHOLDS:
        bad = sorted(set(got) ^ set(NEW_THRESHOLDS) | {k for k in got if k in NEW_THRESHOLDS and got[k] != NEW_THRESHOLDS[k]})
        raise RuntimeError(f"evaluation_v6.new_thresholds disagrees with pipeline/s2_constants.py: {bad}")
    if _tup(ev6.get("options")) != _tup(OPTIONS):
        raise RuntimeError(f"evaluation_v6.options {ev6.get('options')} are not the pre-specified values {OPTIONS}")
    st0 = ev6.get("step0") or {}
    for k, want in (("candidates", CANDIDATES), ("e_d_variants", E_D_VARIANTS), ("n2_rules", LB_RULES)):
        if _tup(st0.get(k)) != want:
            raise RuntimeError(f"evaluation_v6.step0.{k} = {st0.get(k)} is not the pre-specified {list(want)}")
    if _tup((ev6.get("dev3") or {}).get("parts")) != _tup(DEV3["parts"]):
        raise RuntimeError("evaluation_v6.dev3.parts is not the pre-specified Dev-3 split")
    cal = st0.get("calibration")
    if cal is not None:
        extra = sorted(set(cal) - set(CALIBRATION_KEYS) - {"file", "sha256"})
        if extra:
            raise RuntimeError(f"evaluation_v6.step0.calibration: keys {extra} are not pre-specified")
        check_calibration(cal)
    return True
