"""Study 2, held-out run: the pre-specified roles, the held-out guard, the stage-1 entry point of the held-out families
and the stage-2 tag list. Called by run_study2.sh. No rule, threshold or seed is defined here: every value is read from
the constants file (evaluation_v6, run.*, sensitivities.*) or from pipeline/s2_constants.py, and the development-stage
outcome is read from evaluation_v6.selected and step0_decision.json.

Stage 1 of a held-out site runs through stage1_channels.site_record / write_manifest (stage1_channels.main knows the
study-1 families and D3U only). provenance.check_tag admits 'primary', 'dev_*' and 'sens_*'; every study-2 record
carries the 'dev_' tag TAG below. The guard of this module (no SPEEDCHECK_* knob, the patched QuaDRiGa tree, the
study-2 constants, the seeds re-derived from the seed record, the development-stage decision) runs before every
stage-1, stage-2 and stage-3 call.

Fault rules: a held-out family takes the study-1 values of the study-1 family it is built on (U2 <- U, N2 <- N,
H2 <- H; constants run.site_timeout_h and run.workers_stage1). The study-1 compute window is not applied in study 2.

Usage: python pipeline/s2_heldout.py --guard [--no-tools]
       python pipeline/s2_heldout.py --role-info <role> <key>     (family, sites, site_timeout_h, workers_stage2, ...)
       python pipeline/s2_heldout.py --stage1 <family> <site> --out <dir>
       python pipeline/s2_heldout.py --stage2-plan <family>      (one line per stage-2 run: tag + args)"""
import argparse
import contextlib
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)

TAG = "dev_s2_heldout"                    # stage-1 / stage-2 tag of U2, N2 (and H2 if it ran)
PLACE_TAG = TAG + "_place"                # S-place light under study 2
DECISION = os.path.join("..", "results", "study2", "step0_decision.json")   # written by pipeline/s2_selection_run.py
BASE_FAMILY = {"U2": "U", "N2": "N", "H2": "H"}                  # the study-1 family each held-out family copies
# Roles: U2-a (sites 10-14), U2-b (15-19), N2, S2 (S-place light sites 20-24; then H2 if D0-R5).
# Site ranges are half-open [lo, hi).
ROLES = {"U2-a": dict(family="U2", sites=(10, 15)), "U2-b": dict(family="U2", sites=(15, 20)),
         "N2": dict(family="N2", sites=(10, 20)), "S2": dict(family="S", sites=(20, 25))}
H2_ROLE = dict(family="H2", sites=(10, 15))
K5_NAMES = {"iid:0.1": "iid10", "ge:0.1:3": "ge10", "iid:0.3": "iid30", "ge:0.3:3": "ge30"}   # as run_study1.sh


def constants(path=None):
    return json.load(open(path or os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study2_constants.json"))))


# ---------------------------------------------------------------------------------------------- roles and settings
def h2_runs(C=None, decision=None):
    """D0-R5: H2 runs iff an estimator candidate is adopted. Read from the constants
    (evaluation_v6.selected.estimator) and, if given, cross-checked with the decision file (D0_R5.h2_runs)."""
    sel = ((C or constants()).get("evaluation_v6") or {}).get("selected") or {}
    runs = sel.get("estimator") is not None
    if decision is not None and bool((decision.get("D0_R5") or {}).get("h2_runs")) != runs:
        raise RuntimeError("step0_decision.json D0_R5.h2_runs disagrees with evaluation_v6.selected.estimator")
    return runs


def roles(C=None):
    """The pre-specified roles; H2 only if D0-R5 (it is then refused by check_supported until its code exists)."""
    out = dict(ROLES)
    if h2_runs(C):
        out["H2"] = dict(H2_ROLE)
    return out


def role(name, C=None):
    rs = roles(C)
    if name not in rs:
        if name == "H2":
            raise ValueError("role H2 refused: no estimator candidate was adopted (D0-R5), so H2 is not run")
        raise ValueError(f"not a held-out role: {name!r} (roles: {sorted(rs)})")
    return rs[name]


def check_supported(C=None):
    """The held-out code implements the configurations whose stage 2 is the study-1 stage 2: no estimator candidate and no
    N2 rule (the decision 'L-T alone'). Any other selected configuration needs the study-2 stage-2 fields and, for an
    estimator, H2 and gate G inputs: refused here, so a held-out run never starts on code that does not implement the
    selected configuration."""
    ev6 = (C or constants()).get("evaluation_v6") or {}
    sel = ev6.get("selected")
    if not sel:
        raise RuntimeError("evaluation_v6.selected is not set (make_constants.py --study2-selected)")
    if sel.get("estimator") is not None or sel.get("n2_rule") is not None:
        raise RuntimeError(f"selected configuration {sel} needs a held-out stage 2 with the study-2 fields (and H2 for an "
                           "estimator); not implemented in this held-out code")
    return sel


def setting(family, key, C=None):
    """study-1 fault-rule values of the study-1 family a held-out family is built on."""
    C = C or constants()
    base = BASE_FAMILY[family]
    if key == "site_timeout_h":
        return C["run"]["site_timeout_h"][base]
    if key == "workers_stage1":
        return C["run"]["workers_stage1"][base]
    if key == "workers_stage2":
        return C["run"].get("workers_stage2") or 2
    raise KeyError(key)


def stage2_plan(family, C=None):
    """The stage-2 runs of a held-out family on its own sites: the main tag, and for U2 the sensitivities that study 2
    keeps from study 1 (sens_ls: the LB gain shift of constants sensitivities.sens_ls; K5 erasures: constants
    evaluation_v5.k5_erasures_v5). The SNR offsets are not run in study 2. Returns [(tag, [extra stage2_evaluate.py
    args])]."""
    C = C or constants()
    plan = [(TAG, [])]
    if family == "U2":
        plan.append((TAG + "_ls", ["--lb-gain-shift-db", str(C["sensitivities"]["sens_ls"]["lb_gain_shift_db"])]))
        for spec in C["evaluation_v5"]["k5_erasures_v5"]:
            plan.append((f"{TAG}_erasure_{K5_NAMES[spec]}", ["--erasure", spec]))
    return plan


# ---------------------------------------------------------------------------------------------- the held-out guard
def decision(path=None):
    p = path or os.path.join(ER, DECISION)
    if not os.path.exists(p):
        raise RuntimeError(f"{p} missing: the development-stage decision is needed")
    return json.load(open(p))


def check_decision(C, dec):
    """evaluation_v6.selected equals the decision's 'selected' (make_constants.py --study2-selected copies it) and D0-R5
    agrees; returns the selected configuration."""
    sel = (C.get("evaluation_v6") or {}).get("selected")
    if sel != dec.get("selected"):
        raise RuntimeError(f"evaluation_v6.selected {sel} differs from step0_decision.json selected {dec.get('selected')}")
    h2_runs(C, dec)
    return sel


def check_seeds(C):
    """evaluation_v6.seeds re-derived from the seed record (pipeline/s2_seeds.py); the block's own time is past."""
    from pipeline import s2_seeds as SD
    ev6 = C.get("evaluation_v6") or {}
    rec0 = ev6.get("seed_record") or {}
    if not ev6.get("seeds") or not rec0.get("file"):
        raise RuntimeError("evaluation_v6.seeds / seed_record missing: make_constants.py --study2-seeds precedes "
                           "every held-out run")
    path = os.path.join(ER, rec0["file"])
    seeds, rec = SD.seeds_from_record(path, C)
    if seeds != ev6["seeds"]:
        raise RuntimeError("evaluation_v6.seeds differ from the seeds re-derived from the seed record")
    for k in ("bundle_sha256", "block_height", "block_hash"):
        if rec[k] != rec0.get(k):
            raise RuntimeError(f"evaluation_v6.seed_record.{k} differs from the re-derived value")
    hdr = json.load(open(path))["block_header_hex"]
    if SD.header_time(hdr) >= dt.datetime.now(dt.timezone.utc):
        raise RuntimeError("the block's time is not yet past")
    return seeds


def check_tools(tools=True):
    """No SPEEDCHECK_* knob is set, and the patched QuaDRiGa tree is the one in the constants (tools=False skips the
    tree check, e.g. before the tools are installed)."""
    from pipeline.provenance import ENV_KNOBS, quadriga_tree_sha256
    C = constants()
    for v in ENV_KNOBS:
        if v in os.environ:
            raise RuntimeError(f"held-out run: {v} must not be set")
    if tools and quadriga_tree_sha256() != C["quadriga"]["patched_tree_sha256"]:
        raise RuntimeError("held-out run: the patched QuaDRiGa tree differs from constants quadriga.patched_tree_sha256")


def guard(C=None, dec=None, reference=None, env=True, tools=True):
    """Everything a held-out stage-1 / stage-2 / stage-3 call needs; raises with the reason, returns the selected
    configuration. With `reference` (a constants file of the development stage) the block must equal it except the
    flags, selected and seeds. env=False skips the environment and tool checks (synthetic constants)."""
    from pipeline import s2_constants as S2
    C = C or constants()
    ev6 = C.get("evaluation_v6")
    S2.check_against(ev6)
    if reference is not None:
        S2.check_same_as_reference(ev6, reference.get("evaluation_v6"), allowed=S2.DEV_FLAG_KEYS + S2.EVAL_KEYS)
    if not (ev6["step0"].get("timestamped") and ev6["step0"].get("study1_merge_done")):
        raise RuntimeError("evaluation_v6.step0 flags (timestamped, study1_merge_done) are not both set")
    sel = check_decision(C, dec if dec is not None else decision())
    check_supported(C)
    check_seeds(C)
    if env:
        check_tools(tools=tools)
    return sel


@contextlib.contextmanager
def s2_bootstrap(C=None):
    """'bootstrap' is one of the five study-2 seeds (evaluation_v6.seeds.bootstrap). The study-1 stage-3 helpers draw
    their site bootstrap through provenance.rng_for, which reads the study-1 value constants seeds.bootstrap; inside
    this context rng_for reads the study-2 value (restored afterwards, also on an exception). Used by the study-2
    stage 3 and the study-2 S-place summary."""
    from pipeline import provenance as P
    s2 = ((C or constants()).get("evaluation_v6") or {}).get("seeds") or {}
    if "bootstrap" not in s2:
        raise RuntimeError("evaluation_v6.seeds.bootstrap missing (make_constants.py --study2-seeds)")
    old = P.C["seeds"]["bootstrap"]
    P.C["seeds"]["bootstrap"] = s2["bootstrap"]
    try:
        yield s2["bootstrap"]
    finally:
        P.C["seeds"]["bootstrap"] = old


# ---------------------------------------------------------------------------------------------- stage 1
def stage1(family, site, out):
    """One held-out site through stage1_channels.site_record (seeds: evaluation_v6.seeds via
    stage1_channels.seed_base), then write_manifest."""
    from pipeline import stage1_channels as G
    from pipeline.provenance import check_tag
    if family not in BASE_FAMILY:
        raise ValueError(f"not a held-out family: {family}")
    if family == "H2" and not h2_runs():
        raise ValueError("H2 is not run (D0-R5)")
    check_tag(TAG, stage=1, Q=G.C["run"]["Q"], n_scale=G.C["run"]["n_scale"])
    os.makedirs(out, exist_ok=True)
    path, t = G.site_record(family, site, G.C["run"]["Q"], out, TAG, G.C["run"]["n_scale"])
    G.write_manifest(out)
    print(path, "skipped (exists, same provenance)" if t is None else "done", flush=True)   # no value printed
    return path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--guard", action="store_true")
    ap.add_argument("--no-tools", action="store_true", help="with --guard: skip the QuaDRiGa tree check")
    g.add_argument("--role-info", nargs=2, metavar=("ROLE", "KEY"))
    g.add_argument("--stage1", nargs=2, metavar=("FAMILY", "SITE"))
    g.add_argument("--stage2-plan", metavar="FAMILY")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.guard:
        guard(tools=not a.no_tools)
        print("held-out guard passed")
        return
    if a.role_info:
        r = role(a.role_info[0])
        k = a.role_info[1]
        if k == "family":
            print(r["family"])
        elif k == "sites":
            print(" ".join(str(s) for s in range(*r["sites"])))
        else:
            print(setting(r["family"], k))
        return
    if a.stage1:
        guard()
        stage1(a.stage1[0], int(a.stage1[1]), a.out or os.path.join(ER, "runs", f"g1_{a.stage1[0]}_{TAG}"))
        return
    for tag, extra in stage2_plan(a.stage2_plan):
        print(tag, *extra)


if __name__ == "__main__":
    main()
