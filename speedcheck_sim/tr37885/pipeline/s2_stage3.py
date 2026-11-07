"""Study 2 stage 3 on the held-out records: the held-out counterpart of pipeline/stage3_criteria.py's main for U2
(P1-10, P1-15) and N2 (P3-15). This driver loads the held-out stage-2 records, applies mode s2
(s2_criteria.use_statistic_s2 with evaluation_v6.selected) and calls the functions of stage3_criteria.py,
stage3_extra_checks.py and s2_criteria.py:
  * scope per cell: k = sites whose included gen_on survey passes qualify
    (stage3_criteria.ref_qualification), worst variant, n = 10, point rating, Clopper-Pearson;
  * the headline / governing sigma (stage3_criteria.headline_and_c3: worse of the generator x the S-place factor;
    sens_ls co-reported; the worse governs), with the S-place factor of N5 (s2_criteria.s_place_factor: the larger of
    the study-1 and the study-2 urban factors);
  * C2 (0.5 % band; reported) and the L1 bias test (s2_criteria.c2_tost_s2: band 3 %, Holm m = 3; N16);
  * R8 (i)-(iii) (stage3_extra_checks.qual_validity; R8 (ii) gates L1 (N2): <= 1 of the k qualified sites with test
    sigma > 9 %, read per variant, the worst (largest) count governs; R8 (iii) printed in every outcome);
  * the rung per U2 cell (s2_criteria.cell_rung), gate G (H2 not run -> 'not run'), L3 (below), the outcome
    (s2_criteria.select_outcome);
  * N2 (P3-15) with the same quantities, outside the ladder (T3-N on the study-2 records);
  * K5 erasures (U2): worst gated sigma per cell under each erasure tag, reported only.
Every site bootstrap uses the study-2 'bootstrap' seed (s2_heldout.s2_bootstrap).
L3: L3 holds iff the decision adapter's urban stratum status (tables_<tag>.json stratum_status.urban) is 'covered' or
'covered_short' and every U2 cell whose speed class the adapter keeps in the urban stratum (not in
strata.urban.uncovered_speeds; C5 is not run on held-out data, so both cells) reaches L2. The Table B honest bound is
computed and printed as pre-specified, not gated. If no U2 cell reaches L2, L3 cannot hold and the adapter is not run.
An adapter status outside the adapter's own list (adapter STATUS) means L3 fails: it is recorded (status_unknown), the
template is the highest rung reached, and stage 3 does not stop for it (a rule chosen after held-out values are
visible would be post hoc; this one can only under-claim).
Also, as stage3_criteria.main / stage3_extra_checks.add_v5 compute them for study 1: the study-1 K7 calibration (the
study-1 result coverage_check_sens_coverage.json, pinned by SHA-256: c_K7 and the sigma-CI widening; K7 is not re-run,
the statistic is unchanged), the optimism check with the study-2 Holm family (m = 3, N4; s2_criteria.optimism_s2), its
S-place-corrected re-evaluation (the worse verdict governs), the bridge, the model-diversity C1 quantities
(descriptive), per-beacon N, realised K and the K report, realised speed, the P3 blocked shares, C4-light, and the
study-1 keys of stage3_extra_checks.add_v5 (R8 (i)-(iii), threshold sensitivities without the SNR offset rows, the V2V
reading, qualification vs K, R8 (iv), per-beacon at the governing variant); Table A2 = reports/make_reports.table_a on
these quantities, keys re-prefixed A2|...
The study-1 S-place result is pinned by SHA-256; a study-2 S-place result that is missing (role S2 failed or stopped)
is stated as unquantified and the N5 factor is the study-1 factor.
No held-out number is printed: the outputs are files; the console gets paths and counts only.
Usage: python pipeline/s2_stage3.py --study1-s-place <s_place_sens_place.json>
           --study1-k7 <coverage_check_sens_coverage.json> [--study2-s-place-missing] [--runs DIR] [--out DIR]"""
import argparse
import contextlib
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import stage3_criteria as K  # noqa: E402
from pipeline import s2_criteria as CS  # noqa: E402
from pipeline import stage3_extra_checks as XC  # noqa: E402
from pipeline import s2_constants as S2  # noqa: E402
from pipeline import s2_heldout as HO  # noqa: E402

PTR = {"U2": "U", "N2": "N"}                             # held-out family -> the family name the study-1 functions read
CELLS = {"U2": [("P1", 10.0), ("P1", 15.0)], "N2": [("P3", 15.0)]}
U2_CELLS = ("P1-10", "P1-15")


def ck(cell, v0):
    return f"{cell}-{v0:g}"


# ---------------------------------------------------------------------------------------------- loading
def family_dir(runs, fam):
    return os.path.join(runs, f"g1_{fam}_{HO.TAG}")


def load_family(runs, fam, tag, sites, check_prov=True):
    """The stage-2 records of one held-out family and tag on its pre-specified sites; every site file must exist and carry
    this tree's stage-1/2 sources."""
    from pipeline.provenance import stage_hashes
    cur = stage_hashes() if check_prov else None
    recs, files = [], {}
    if check_prov:                                   # stage-1 provenance: stage2_evaluate.py skips its stage-1 check
        for s in range(*sites):                      # for dev_ tags, so it is done here
            f1 = os.path.join(family_dir(runs, fam), f"site_{s:03d}.json")
            if not os.path.exists(f1):
                raise RuntimeError(f"{f1}: stage-1 record missing")
            p1 = json.load(open(f1)).get("provenance") or {}
            if p1.get("tag") != HO.TAG:
                raise RuntimeError(f"{f1}: not a {HO.TAG} stage-1 record")
    d = os.path.join(family_dir(runs, fam), f"eval_{tag}")
    miss = [s for s in range(*sites) if not os.path.exists(os.path.join(d, f"site_{s:03d}.json"))]
    if miss:
        raise RuntimeError(f"{fam} ({tag}): stage-2 files missing for sites {miss} (stage 3 needs every held-out site; "
                           "study 2 has no incomplete-run rule)")
    extra = sorted(set(glob.glob(os.path.join(d, "site_*.json"))) -
                   {os.path.join(d, f"site_{s:03d}.json") for s in range(*sites)})
    if extra:
        raise RuntimeError(f"{fam} ({tag}): stage-2 files outside the pre-specified sites: {extra}")
    for s in range(*sites):
        f = os.path.join(d, f"site_{s:03d}.json")
        b = open(f, "rb").read()
        files[os.path.relpath(f, runs)] = hashlib.sha256(b).hexdigest()
        x = json.loads(b)
        p = x.get("provenance") or {}
        if check_prov:
            if p.get("tag") != tag or any((p.get("stage_sha256") or {}).get(j) != cur[j] for j in ("1", "2")):
                raise RuntimeError(f"{f}: not a {tag} record of this tree's stage-1/2 sources")
        for r in x["recs"]:
            r["family"] = PTR[fam]
            if r["site"] != s:
                raise RuntimeError(f"{f}: a record of site {r['site']}")
        recs += x["recs"]
    return recs, files


def statistic(recs, selected, cal):
    """The study-1 reader rule (stage3_criteria.use_statistic: plain LS, the Huber fit (d) not adopted) then mode s2."""
    recs = K.use_statistic(recs, "d" if K.LEVER_D else "ls")
    return CS.use_statistic_s2(recs, selected, cal)


# ---------------------------------------------------------------------------------------------- quantities
def scope_s2(recs, cell, v0, sites):
    """stage3_criteria.scope's rule on the pre-specified held-out sites: per variant k = qualified sites (gen_on, included survey
    passes, stage3_criteria.ref_qualification), the worst (smallest) k governs, n = the pre-specified sites."""
    n = sites[1] - sites[0]
    rows = []
    for mp, prb in K.COMBOS:
        ref = [r for r in K.select(recs, cell, v0, "gen_on", mp, prb) if r["role"] == "ref"]
        k = sum(K.ref_qualification([r for r in ref if r["site"] == s])["qualified"] for s in range(*sites))
        rows.append(dict(mapping=mp, prb=prb, k=int(k)))
    w = min(rows, key=lambda r: r["k"])
    return dict(k=w["k"], n=n, share=w["k"] / n, cp95=K.clopper_pearson(w["k"], n), rating=K.rating(w["k"] / n),
                variant=[w["mapping"], w["prb"]], variants=rows)


def ls_rows(recs_ls, cells):
    """sens_ls rows as stage3_criteria.sens_ls_rows: the LB variants under the gain shift, cells of constants sens_ls.cells."""
    SL = K.C["sensitivities"]["sens_ls"]
    out = {}
    for cell, v0 in cells:
        if cell in SL["cells"]:
            rows = [x for x in (K.cell_quantities(recs_ls, cell, v0, "lb", prb) for prb in K.PRBS) if x]
            if rows:
                out[(cell, v0)] = rows
    return out


def r8(qv, key):
    """R8 (ii) count per variant (qualified sites with gated test sigma > 9 %) and the worst count; R8 (iii) per variant
    and at the governing variant (reported)."""
    v = (qv.get(key) or {}).get("variants") or []
    over = [dict(mapping=r["mapping"], prb=r["prb"], k=r["qualified_over_margin"]["k"], n=r["qualified_over_margin"]["n"],
                 cp95=r["qualified_over_margin"]["cp95"]) for r in v]
    worst = max(over, key=lambda r: r["k"]) if over else None
    g = (qv.get(key) or {}).get("at_governing") or {}
    return dict(r8ii_variants=over, r8ii_worst=worst, n_over_9pct=worst["k"] if worst else 0,
                r8iii_variants=[dict(mapping=r["mapping"], prb=r["prb"], spearman=r["spearman_sdref_vs_test_sigma"],
                                     n_sites=r["n_sites"]) for r in v],
                r8iii_at_governing=dict(spearman=g.get("spearman_sdref_vs_test_sigma"), n_sites=g.get("n_sites"),
                                        mapping=g.get("mapping"), prb=g.get("prb")))


def governing_sigma(head, key):
    h = next((h for h in head if ck(h["cell"], h["v0"]) == key), None)
    hf = (h or {}).get("headline_final_qualified_gated") or {}
    return (hf["value_pct"] / 100.0 if hf.get("value_pct") is not None else np.nan), hf


def k5_rows(recs_by_tag, cells):
    out = {}
    for tag, recs in recs_by_tag.items():
        for cell, v0 in cells:
            rows = [x for x in (K.cell_quantities(recs, cell, v0, mp, prb) for mp, prb in K.COMBOS) if x]
            w = K.worst(rows, ("qualified_gated", "sigma")) if rows else None
            out[f"{tag}|{ck(cell, v0)}"] = dict(worst_sigma_qualified_gated=w[0] if w else None,
                                                variant=w[1] if w else None)
    return out


# study-1 results that stage 3 reads (stored outputs of study 1)
STUDY1_S_PLACE_SHA256 = "ceff73896cb7fde5de3ad42cc3c1701ce0740b91e5a5c7b43172d18b4a2d67e1"   # s_place_sens_place.json
STUDY1_K7_SHA256 = "77ecbcf2cbf8197e270c3b87c78339828cab604500c73a83e4725e61a9042412"        # coverage_check_sens_coverage.json
S_PLACE_ROLE = "S2"                                     # the S-place role (s2_heldout.ROLES)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _pinned(path, sha, what):
    if not path or not os.path.exists(path):
        raise RuntimeError(f"{what} missing ({path})")
    h = _sha(path)
    if sha is not None and h != sha:
        raise RuntimeError(f"{what} {path}: SHA-256 {h[:12]}... is not the pinned {sha[:12]}... (stored study-1 result)")
    return h


def s_place_state(runs):
    """The final state of the S-place role (runs/STATE_FINAL_s2_S2, written by run_study2.sh): 'done' | 'failed'
    (stated as unquantified); anything else is refused."""
    p = os.path.join(runs, f"STATE_FINAL_s2_{S_PLACE_ROLE}")
    st = open(p).read().strip() if os.path.exists(p) else ""
    if st.startswith("DONE"):
        return "done"
    if st.startswith(("FAILED", "STOPPED")):
        return "failed"
    raise RuntimeError(f"role S2 final state {st or 'none'!r} is neither DONE nor FAILED / STOPPED")


def s_place_factors(study1_path, study2_path, study2_state="done", study1_sha=STUDY1_S_PLACE_SHA256):
    """N5: the governing urban factor = max(study 1, study 2). The study-1 file is the stored study-1 S-place result
    (SHA-256 pinned). The study-2 file is required when role S2 is done; when it failed or stopped the study-2 factor is
    unquantified and the governing factor is the study-1 factor (s2_criteria.s_place_factor ignores a missing
    factor)."""
    out = {}
    h1 = _pinned(study1_path, study1_sha, "S-place study1 result")
    d = json.load(open(study1_path))
    out["study1"] = dict(path=os.path.relpath(study1_path, ER), sha256=h1, urban=float(d["stratum_factor"]["urban"]),
                         tag=d.get("tag"))
    if study2_state == "failed":
        out["study2"] = dict(status="unquantified", urban=None,
                             note="role S2 failed or stopped: the study-2 S-place check did not complete")
    elif study2_state == "done":
        if not study2_path or not os.path.exists(study2_path):
            raise RuntimeError(f"S-place study2 result missing ({study2_path}) although role S2 is done (N5)")
        d = json.load(open(study2_path))
        out["study2"] = dict(path=os.path.relpath(study2_path, ER), sha256=_sha(study2_path), status="done",
                             urban=float(d["stratum_factor"]["urban"]), tag=d.get("tag"))
    else:
        raise ValueError(study2_state)
    out["governing_urban"] = CS.s_place_factor(out["study1"]["urban"], out["study2"]["urban"])
    return out


def load_k7(path, sha=STUDY1_K7_SHA256):
    """The stored study-1 K7 result (K7 is not re-run and its calibration applies): c_K7 and the K7 state (calibrated,
    widen_sigma_ci), read as stage3_criteria.load_c_k7 reads them."""
    h = _pinned(path, sha, "study-1 K7 result")
    d = json.load(open(path))
    state = dict(calibrated=bool(d.get("calibrated", True)), widen_sigma_ci=bool(d.get("widen_sigma_ci", False)))
    c = float(d["c_calibrated"]) if d.get("c_calibrated") is not None else 1.0
    return dict(path=os.path.relpath(path, ER), sha256=h, c_k7=c, state=state,
                source="study 1 (coverage_check_sens_coverage.json), statistic unchanged")


@contextlib.contextmanager
def k7_state(state):
    """stage3_criteria.K7_STATE set to the study-1 K7 state for the duration of stage 3 (sigma-CI widening, optimism cap);
    restored afterwards, also on an exception."""
    old = dict(K.K7_STATE)
    K.K7_STATE.update(state)
    try:
        yield K.K7_STATE
    finally:
        K.K7_STATE.clear()
        K.K7_STATE.update(old)


@contextlib.contextmanager
def runs_pointer(runs):
    """criteria.RUNS pointed at a folder whose g1_U / g1_N are the held-out family folders (as the adapter's pointer
    folder): the fixed K-report and in-band-K readers (stage3_criteria.k_report, stage3_extra_checks.site_inband_k) read the stage-1
    records of U2 / N2 under the family names they know; restored afterwards."""
    import tempfile
    ptr = tempfile.mkdtemp(prefix="s2ptr_")
    for fam, p in PTR.items():
        if os.path.isdir(family_dir(runs, fam)):
            os.symlink(os.path.abspath(family_dir(runs, fam)), os.path.join(ptr, f"g1_{p}"))
    old = K.RUNS
    K.RUNS = ptr
    try:
        yield ptr
    finally:
        K.RUNS = old


@contextlib.contextmanager
def no_snr_offset_rows():
    """lists sens_ls and K5 only: stage3_extra_checks.add_v5 runs without the SNR offset rows."""
    old = XC.snr_offset_rows
    XC.snr_offset_rows = lambda tag, families, governing: None
    try:
        yield
    finally:
        XC.snr_offset_rows = old


def evaluated_types(cells):
    """The site types of criteria.TYPES whose cells are all evaluated in study 2 (urban LOS, the NLOSv crossings)."""
    keys = {ck(c, v) for c, v in cells}
    return {t for t, spec in K.TYPES.items() if set(spec["cells"]) <= keys}


def table_a2(out):
    """Table A2 ("Table A2 (U2, N2, H2 rows as Table A)"): the fixed reports/make_reports.table_a
    and its text keys on the stage-3 s2 output; every key re-prefixed A2|... (A|<cell>|... -> A2|<cell>|...)."""
    from reports import make_reports as MR
    nums = {}
    rows = MR.table_a(out, nums)
    a2 = {("A2|" + k[2:] if k.startswith("A|") else "A2|" + k): v for k, v in nums.items()}
    return dict(rows=rows, numbers=a2)


# ---------------------------------------------------------------------------------------------- the s2 stage 3
def compute(recs, cells_by_fam, sites_by_fam, spf_urban, recs_ls=None, recs_k5=None, c_k7=1.0, tag=HO.TAG,
            families=("U", "N")):
    """Everything of the study-2 stage 3 that does not need the adapter. recs: mode-s2 records of U2 and N2 (families
    'U', 'N'); c_k7: the study-1 K7 factor (the K7 state is set by the caller); returns the output dict with
    per-cell scope, sigma, C2 / L1, R8, the rung of each U2 cell, and the study-1 stage-3 quantities (module docstring)."""
    cells = [c for f in ("U2", "N2") for c in cells_by_fam.get(f, [])]
    spf = {"urban": spf_urban, "highway": 1.0}
    with CS.holm_family_s2():
        head, gov = K.headline_and_c3(recs, cells, spf, ls_rows(recs_ls or [], cells))
        c2_rows, c2_sum = CS.c2_tost_s2(recs, cells)
        qv = XC.qual_validity(recs, cells, gov)
        # the optimism check with the study-2 Holm family, and its S-place-corrected re-evaluation, as
        # as stage3_criteria.main: the worse verdict governs
        opt = [x for x in (K.optimism_cell(recs, c, v, mp, prb, c_k7) for c, v in cells for mp, prb in K.COMBOS) if x]
        gov_opt = CS.optimism_s2(opt)
        opt_place, gov_opt_place = None, None
        if any(K.place_factor(c, spf) != 1.0 for c, _ in cells):
            opt_place = [K.optimism_cell(recs, c, v, mp, prb, c_k7, K.place_factor(c, spf)) for c, v in cells
                         for mp, prb in K.COMBOS]
            gov_opt_place = CS.optimism_s2(opt_place)
            for k, g in gov_opt_place.items():
                if K.VERDICT_ORDER[g["verdict"]] < K.VERDICT_ORDER[gov_opt[k]["verdict"]]:
                    gov_opt[k] = dict(g, source="S-place corrected (worse)")
        types = evaluated_types(cells)
        br = {t: v for t, v in K.bridge(gov_opt, opt + (opt_place or [])).items() if t in types}
        c1 = [x for x in (K.c1_cell(recs, c, v, mp, prb) for c, v in cells for mp, prb in K.COMBOS) if x]
    out = dict(cells={}, headline=head, governing=gov, c2=c2_rows, c2_summary=c2_sum, qualification_validity=qv,
               optimism=opt, optimism_governing=gov_opt, optimism_s_place=opt_place,
               optimism_governing_s_place=gov_opt_place, bridge=br, model_diversity_c1=c1, c_k7=c_k7,
               per_beacon=K.per_beacon(recs, cells), realised_k=K.realised_k(recs, cells),
               k_report=K.k_report(tag, list(families)), realised_speed=K.realised_speed(recs, cells),
               p3_blocked=K.p3_blocked(recs), c4_light=K.c4_light(recs))
    for fam in ("U2", "N2"):
        for cell, v0 in cells_by_fam.get(fam, []):
            key = ck(cell, v0)
            sc = scope_s2(recs, cell, v0, sites_by_fam[fam])
            sg, hf = governing_sigma(head, key)
            rows = [r for r in c2_rows if (r["cell"], r["v0"]) == (cell, v0)]
            tost = CS.l1_bias_ok(rows)
            rr = r8(qv, key)
            row = dict(family=fam, scope=sc, sigma_governing=sg, headline_final=hf, l1_bias_ok=tost, r8=rr,
                       c2_rows=[dict(mapping=r["mapping"], prb=r["prb"], bias=r["bias"], bias_ci=r["bias_ci"],
                                     p_tost=r["p_tost"], p_tost_holm=r.get("p_tost_holm"), band_s2=r["band_s2"],
                                     p_tost_s2=r["p_tost_s2"], p_tost_s2_holm=r.get("p_tost_s2_holm"),
                                     tost_s2_pass=r.get("tost_s2_pass"), excess_detected=r["excess_detected"],
                                     excess_p=r["excess_p"], tau=r["tau"]) for r in rows])
            if fam == "U2":
                row["rung"] = CS.cell_rung(sc["k"], sc["n"], tost, rr["n_over_9pct"], sg)
            out["cells"][key] = row
    out["scope"] = dict(cells={k: v["scope"] for k, v in out["cells"].items()}, rule="study-1 scope rule on the pre-specified "
                        "held-out sites (scope_s2)")
    with CS.holm_family_s2(), no_snr_offset_rows():
        XC.add_v5(out, recs, cells, {ck(c, v): 10 for c, v in cells}, tag, list(families), incomplete=False)
    out["sensitivity_note"] = "threshold sensitivities without the SNR offset rows (sens_ls and K5 only)"
    if recs_k5:
        out["k5_erasures"] = k5_rows(recs_k5, cells_by_fam.get("U2", []))
    out["table_A2"] = table_a2(out)
    return out


L3_COVERED = ("covered", "covered_short")


def l3(out, run_adapter, statuses=None):
    """L3 (module docstring). run_adapter -> (urban stratum status, uncovered speed classes) of
    the adapter e2 on the held-out records. statuses: the adapter's status list (adapter STATUS)."""
    rungs = {k: out["cells"][k]["rung"]["rung"] for k in U2_CELLS}
    if not any(r == "L2" for r in rungs.values()):
        return dict(holds=False, reason="no U2 cell reaches L2 (a covered stratum has a qualified class, which would have "
                                        "to reach L2)", adapter_run=False)
    status, drop = run_adapter()
    if status not in (statuses if statuses is not None else adapter_statuses()):      # L3 fails, recorded, no stop
        return dict(holds=False, adapter_run=True, stratum_status=status, status_unknown=True,
                    uncovered_speeds=list(drop or []), reason=(f"Bstratum|urban = {status!r} is not one of the adapter's "
                                                               "statuses: L3 fails; the template is the highest rung reached"))
    covered = status in L3_COVERED
    kept = [c for c in U2_CELLS if float(c.split("-")[1]) not in {float(x) for x in (drop or [])}]
    holds = CS.l3_holds(covered, kept, rungs)
    return dict(holds=holds, adapter_run=True, stratum_status=status, uncovered_speeds=list(drop or []),
                adapter_covered_cells=kept, reason=(f"Bstratum|urban = {status}; L3 = adapter status covered / "
                                                    "covered_short and every kept U2 cell at L2"))


def adapter_statuses():
    from pipeline import decision_adapter as AD
    return tuple(AD.STATUS)


def adapter_run(runs, results, tag, what="e2", env_extra=None):
    """Run the decision adapter on the held-out records through a pointer folder; its outputs go to results/adapter."""
    import tempfile
    ptr = tempfile.mkdtemp(prefix="s2ptr_")
    for fam, p in PTR.items():
        os.symlink(os.path.abspath(family_dir(runs, fam)), os.path.join(ptr, f"g1_{p}"))
    aout = os.path.join(results, "adapter")
    env = dict(os.environ, SPEEDCHECK_RUNS=ptr, SPEEDCHECK_RESULTS=results, SPEEDCHECK_ADAPTER_OUT=aout, **(env_extra or {}))
    r = subprocess.run([sys.executable, os.path.join(ER, "pipeline", "decision_adapter.py"), what, "--tag", tag],
                       cwd=os.path.join(ER, "..", "inhouse"), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"adapter {what} failed (exit {r.returncode}): {r.stderr[-2000:]}")
    return aout


def adapter_status(runs, results, tag, env_extra=None):
    """The urban stratum status and the uncovered speed classes of the decision adapter e2 (tables_<tag>.json)."""
    aout = adapter_run(runs, results, tag, "e2", env_extra)
    t = json.load(open(os.path.join(aout, f"tables_{tag}.json")))
    u = (t.get("strata") or {}).get("urban")
    return ((t.get("stratum_status") or {}).get("urban", "no_records"),
            list(u.get("uncovered_speeds") or []) if isinstance(u, dict) else [])


def write_governing(out, results, tag):
    """The governing file the decision adapter reads (stage3_criteria.main's format): governing variants, no C5 class failed
    (C5 is not run on held-out data), the C2 (ii) tau per cell, the s2 scope."""
    g = dict(out["governing"], c5_failing=[], c2_tau={k: v["tau_max"] for k, v in out["c2_summary"].items()},
             scope={k: v["scope"] for k, v in out["cells"].items()})
    p = os.path.join(results, f"governing_{tag}.json")
    json.dump(g, open(p + ".tmp", "w"), indent=1, default=float)
    os.replace(p + ".tmp", p)
    return p


def decide(out, dec, l3res, gate):
    rungs = {k: out["cells"][k]["rung"]["rung"] for k in U2_CELLS}
    kz = {k: out["cells"][k]["scope"]["k"] == 0 for k in U2_CELLS}
    o = CS.select_outcome(gate["status"], rungs, l3res["holds"], kz)
    o["l3"] = l3res
    o["gate_G"] = gate
    o["step0"] = dict(D0_R1=(dec.get("D0_R1") or {}).get("verdict", (dec.get("D0_R1") or {}).get("holds")),
                      chosen=dec.get("D0_R3_chosen"), n2_rule=(dec.get("D0_R4") or {}).get("adopted"),
                      candidates=[dict(name=c.get("name"), eligible=c.get("eligible")) for c in dec.get("candidates", [])])
    return o


def run(runs, results, study1_s_place, study1_k7, tag=HO.TAG, C=None, dec=None, check_prov=True,
        adapter=None, study2_s_place=None, study2_state=None, adapter_e5=None, statuses=None,
        study1_sha=(STUDY1_S_PLACE_SHA256, STUDY1_K7_SHA256)):
    """The whole s2 stage 3; writes criteria_s2_<tag>.json and outcome_s2_<tag>.json into results; returns both paths.
    study1_s_place / study1_k7: the stored study-1 S-place and K7 results (SHA-256 pinned); study2_s_place: the
    study-2 S-place summary; study2_state: 'done' | 'failed' (default: the final state of role S2 in runs). At L3 the
    decision adapter e5 also runs (Table B honest bound, printed)."""
    C = C or HO.constants()
    dec = dec if dec is not None else HO.decision()
    sel = HO.check_decision(C, dec)
    HO.check_supported(C)
    cal = S2.check_calibration(C["evaluation_v6"]["step0"]["calibration"])
    k7 = load_k7(study1_k7, study1_sha[1])
    sites = {f: tuple(S2.S2_FAMILIES[f]["sites"]) for f in ("U2", "N2")}
    files, recs, recs_ls, recs_k5 = {}, [], [], {}
    for fam in ("U2", "N2"):
        for t, _ in HO.stage2_plan(fam, C):
            rs, fh = load_family(runs, fam, t, sites[fam], check_prov)
            files.update(fh)
            rs = statistic(rs, sel, cal)
            if t == tag:
                recs += rs
            elif t == tag + "_ls":
                recs_ls += rs
            else:
                recs_k5[t] = rs
    spf = s_place_factors(study1_s_place, study2_s_place or
                          os.path.join(ER, "..", "results", f"s_place_{HO.PLACE_TAG}.json"),
                          study2_state or s_place_state(runs), study1_sha[0])
    with HO.s2_bootstrap(C), k7_state(k7["state"]), runs_pointer(runs):
        out = compute(recs, CELLS, sites, spf["governing_urban"], recs_ls, recs_k5, k7["c_k7"], tag)
    out.update(tag=tag, s_place=spf, selected=sel, input_sha256=files, design=S2.SOURCE,
               statistic="mode s2 on " + ("d" if K.LEVER_D else "ls"), k7=k7)
    os.makedirs(results, exist_ok=True)
    write_governing(out, results, tag)
    gate = CS.gate_G(h2_ran=HO.h2_runs(C, dec), coverage_rating=None, sigma_governing=np.nan, s_all_cfg=np.nan,
                     s_all_ls=np.nan, t_all_cfg=np.nan, t_all_ls=np.nan, n_crossings=0)
    l3res = l3(out, adapter or (lambda: adapter_status(runs, results, tag)), statuses)
    o = decide(out, dec, l3res, gate)
    if o["template"] == "S2|L3":                      # L3: decision simulation / Table B rows on the study-2 records,
        aout = (adapter_e5 or (lambda: adapter_run(runs, results, tag, "e5")))()      # honest bound printed
        o["l3"]["adapter_outputs"] = os.path.relpath(aout, results) if os.path.isabs(aout) else aout
    from pipeline.provenance import provenance
    out["provenance"] = provenance(tag, stage=3)
    p1, p2 = os.path.join(results, f"criteria_s2_{tag}.json"), os.path.join(results, f"outcome_s2_{tag}.json")
    _dump(out, p1)
    _dump(o, p2)
    return p1, p2


def _dump(obj, path):
    def key(k):
        return "|".join(map(str, k)) if isinstance(k, tuple) else k

    def clean(x):
        if isinstance(x, dict):
            return {key(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x
    json.dump(clean(obj), open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study1-s-place", required=True, help="the stored study-1 S-place result (SHA-256 pinned)")
    ap.add_argument("--study1-k7", required=True, help="the stored study-1 K7 result (SHA-256 pinned)")
    ap.add_argument("--runs", default=os.path.join(ER, "runs"))
    ap.add_argument("--out", default=os.path.join(ER, "..", "results", "study2", "heldout"))
    a = ap.parse_args()
    HO.guard()
    p1, p2 = run(a.runs, a.out, a.study1_s_place, a.study1_k7)
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
