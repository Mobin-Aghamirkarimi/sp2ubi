"""Decision adapter (constants `adapter_e2`, `window_rule`, `c2_consequence`, `c5_consequence`): runs the in-house
decision simulation (monitor variant), the consistency analysis and the matching simulation unchanged on the
TR 37.885 generator records by wrapping them. Only their inputs are replaced:
  * site tables in the structure of `inhouse/decisions_levels.py.load_sites` from the stage-2 records of the governing
    SNR mapping and PRB of the stratum (a primary run refuses any other value). Inclusion as in stage3_criteria.py:
    window rule and finite statistics; gated inputs (constants window_rule.e2_e3_e5_inputs): test passes discarded by
    the residual gate give no entry, qualification is the decision simulation's own requalify; stop_between is never
    an exclusion. Select statistic per site and speed (smaller variance over the references), every site the
    placeholder K = 9 dB (equal weights);
  * consequences decided before the data: C2 (ii) between-site excess τ̂ per cell (governing file) -> each site gets a
    seeded offset N(0, τ̂²) on its evaluated residuals and sqrt(sd_ref² + τ̂²) as reference sd; C5 failing classes are
    removed from the stratum's speed list and from the speed-profile table given to the decision and matching
    simulations (renormalised; the removed mass per vehicle profile is reported as uncovered traffic share);
  * strata: urban = P1 (U), highway = P2 (H); a stratum without any qualified (site, class) is reported "not covered";
  * one 8-level grid over the qualified cells of P1 + P2 (requalify replaced by the same function on that grid); P4
    (38.901) runs with its own grid as the secondary decision simulation, never pooled;
  * fixed row extraction: urban vehicle-profile rows from the urban table, highway rows from the highway table,
    suburban rows from both with the worse reported per metric (EXTRACTION); other rows are not reported;
  * the consistency simulation: P5 per-site z files in the format of inhouse/consistency.py (survey 30, traffic 150,
    evaluated 100; β arms = the first 30 evaluated passes, jr < 30, asserted), only the selected sites of
    selection.json (required), analysed by inhouse/consistency_analysis.py with the level map from the common grid and
    equal weights;
  * the matching simulation: the same site tables; z from the stratum's own P5 cell for every speed of the stratum (never across strata);
  * outputs only under results/adapter/, each with the SHA-256 of the stage-2 files used (input hash).
The level grid is the fixed in-house design constant (evaluation_v4.level_grid; the data-derived grid runs only as the
sensitivity `--grid data`); the top-level clipped share is reported. P5 is not run, so `e3` is not run on generator data
and the matching simulation takes the in-house consistency z of inhouse/results/consistency_z_full.npz at the in-house K
cell given by the equal-K rule (the largest in-house K value <= the stratum's realised in-band K median at the governing
PRB, at least 6 dB; no specular), labelled; with lever_d.adopted the Huber (d) statistics are read.
Usage: python pipeline/decision_adapter.py e2|e5 --tag primary [--smoke] [--grid data]   (run from inhouse/)"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
E4 = os.path.join(ER, "..", "inhouse")
sys.path.insert(0, ER)
sys.path.insert(0, E4)
import decisions_levels as E  # noqa: E402
from decisions_base import SIGMA_INFL  # noqa: E402
from pipeline.provenance import check_tag, provenance  # noqa: E402
from pipeline.stage3_criteria import LEVER_D, use_statistic  # noqa: E402

C = json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study1_constants.json"))))
OUT = os.environ.get("SPEEDCHECK_ADAPTER_OUT", os.path.join(ER, "..", "results", "adapter"))
RUNS = os.environ.get("SPEEDCHECK_RUNS", os.path.join(ER, "runs"))
RESULTS = os.environ.get("SPEEDCHECK_RESULTS", os.path.join(ER, "..", "results"))
K_PLACEHOLDER = 9.0
BASE_PROFILES = {k: dict(v) for k, v in E.PROFILES.items()}
PROV = {}                                               # provenance of this adapter run, written into every output
PC = C["pipeline_constants_from_code"]
GATE, SMAX = PC["gate_ratio"], PC["site_qualification_sigma"]
MIN_REF, MIN_EVAL = C["conventions"]["min_ref"], C["conventions"]["min_eval_e2"]
# With conventions.min_eval_e2_floor (= min_ref = 20) set, a stratum whose smallest gated test count per (site, speed)
# n_eval is below min_eval_e2 is not an abort: the decision and matching simulations run on n_eval passes per row (every
# row truncated to n_eval in track order) and the stratum carries the pre-specified note (status "covered_short",
# printed as the Table B stratum note by reports/make_reports.py); only below the floor is the stratum "not covered".
# Without the floor: abort below min_eval_e2.
MIN_EVAL_FLOOR = C["conventions"].get("min_eval_e2_floor")


class ShortStratum(RuntimeError):
    """n_eval below the floor conventions.min_eval_e2_floor: the stratum is reported "not covered"."""
class NoClass(RuntimeError):
    """Every speed class of the stratum removed by C5: the stratum is reported "not covered (every class failed C5)"."""


STRATA = {"urban": ("U", "P1"), "highway": ("H", "P2")}         # P4 (38.901) not run
# Stratum status identifiers (tables_<tag>.json "stratum_status"; reports/make_reports.py B|stratum|<s>|status and the
# pre-specified Table B stratum notes)
STRATUM_STATUS = {}                                     # of the last build_tables call
STATUS = ("covered", "covered_short", "not_covered_floor", "not_covered_unqualified", "not_covered_c5", "no_records")
EV4 = C["evaluation_v4"]
FIXED_LEVELS = np.array(EV4["level_grid"]["levels"])
INHOUSE_E3 = os.path.join(E4, "results", "consistency_z_full.npz")
INHOUSE_K_GRID = (6.0, 10.0, 15.0, 20.0)
EXTRACTION = {"urban": ["urban"], "highway": ["highway"], "suburban": ["urban", "highway"]}
# Direction of every reported decision-simulation metric per section: honest / margin / badsites rows describe honest
# vehicles (false accusation); power rows describe manipulating vehicles (detection: lower is worse), except the
# framing rows, where the punished vehicle is the framed honest one (higher is worse). Metrics without a direction are
# descriptive: both strata's values are reported (`<metric>_by_source`).
HONEST_HIGH = ("punitive_per_year", "punitive_ub95", "notice_per_year", "oow_share", "escalation_per_year",
               "escalation_ub95", "inspections_per_year", "punitive_union_per_year", "punitive_union_ub95")
POWER_LOW = ("punitive_per_year", "punitive_ub95", "notice_per_year", "escalation_per_year",
             "punitive_union_per_year", "punitive_union_ub95")
POWER_HIGH = ()
E5_HIGH, E5_LOW = ("p_unique",), ("median_candidates", "anon_median")


def direction(section, key, metric):
    if section == "power":
        if key.startswith("framing|"):
            return "high" if metric in HONEST_HIGH else None
        return "low" if metric in POWER_LOW else ("high" if metric in POWER_HIGH else None)
    if section in ("honest", "margin", "badsites", "sens_tyre"):
        return "high" if metric in HONEST_HIGH else None
    return "high" if metric in E5_HIGH else ("low" if metric in E5_LOW else None)


def load_recs(fam, tag):
    files = sorted(glob.glob(os.path.join(RUNS, f"g1_{fam}", f"eval_{tag}", "site_*.json")))
    recs, h = [], hashlib.sha256()
    for f in files:
        b = open(f, "rb").read()
        h.update(b)
        recs += json.loads(b)["recs"]
    return use_statistic(recs, "d" if LEVER_D else "ls"), h.hexdigest()


def sha256_path(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def included(r):
    return r.get("window_ok", True) and r.get("finite", np.isfinite(r["match"]) and np.isfinite(r["prof"]))


def site_table(recs, cell, mapping, prb, arm="gen_on", tau=0.0, drop_speeds=(), rng=None):
    """Same structure as inhouse/decisions_levels.py.load_sites (ev, dh, dt, sraw, swithin, nu, levels, lvl, sg, K, keys, speeds,
    nS, n_eval) plus `truncation` (passes beyond n_eval per row) and `uncovered` (C5 dropped classes)."""
    sel = [r for r in recs if r["cell"] == cell and r["arm"] == arm and r["mapping"] == mapping and r["prb"] == prb
           and r["kin"] == "sumo" and r.get("beta", 0.0) == 0.0 and included(r)]
    speeds_full = tuple(sorted({r["v0"] for r in sel}))
    speeds = tuple(v for v in speeds_full if v not in set(drop_speeds))
    rows = []
    for s in sorted({r["site"] for r in sel}):
        off = rng.normal(0.0, tau) if tau > 0 else 0.0          # C2 (ii) consequence: site offset N(0, tau^2)
        for v in speeds:
            rs = sorted([r for r in sel if r["site"] == s and r["v0"] == v], key=lambda r: r["track"])
            ref = [r for r in rs if r["role"] == "ref"]
            if len(ref) < MIN_REF:
                raise RuntimeError(f"{cell} site {s} v {v}: only {len(ref)} included reference passes")
            em = np.array([r["match"] for r in ref], float); ep = np.array([r["prof"] for r in ref], float)
            key = "match" if np.var(em) <= np.var(ep) else "prof"
            rms_ref = np.median(np.sqrt([r["mse"] for r in ref]))
            tst = [r for r in rs if r["role"] == "test" and np.sqrt(r["mse"]) <= GATE * rms_ref]   # gated inputs
            ref_v = np.array([r[key] for r in ref], float)
            ev = np.array([r[key] for r in tst], float) + off
            rows.append(dict(cc=(K_PLACEHOLDER, cell, None), site=s, v=v, ref=ref_v, ev=ev, all=np.concatenate([ref_v, ev]),
                             sraw=float(np.sqrt(ref_v.var(ddof=1) + tau ** 2))))
    keys = sorted({(r["cc"], r["site"]) for r in rows}, key=str)
    kidx = {k: i for i, k in enumerate(keys)}
    nS, nV = len(keys), len(speeds)
    if not rows:                                        # every speed class removed by C5 (C5 is not run)
        raise NoClass(f"{cell}: no speed class left after the C5 removal {list(drop_speeds)}")
    n_eval = min(len(r["ev"]) for r in rows)
    if n_eval < MIN_EVAL and MIN_EVAL_FLOOR is None:
        raise RuntimeError(f"{cell}: n_eval {n_eval} < {MIN_EVAL}")
    if MIN_EVAL_FLOOR is not None and n_eval < MIN_EVAL_FLOOR:
        raise ShortStratum(f"{cell}: n_eval {n_eval} < floor {MIN_EVAL_FLOOR}")
    A = dict(ev=np.full((nS, nV, n_eval), np.nan), dh=np.full((nS, nV), np.nan), dt=np.full((nS, nV), np.nan),
             sraw=np.full((nS, nV), np.nan), swithin=np.full((nS, nV), np.nan), nu=np.full((nS, nV), np.nan),
             sdref=np.full((nS, nV), np.nan),
             nS=nS, n_eval=n_eval, keys=keys, speeds=speeds, speeds_full=speeds_full, uncovered=list(drop_speeds),
             truncation=int(sum(len(r["ev"]) - n_eval for r in rows)),
             rows_below_min_eval=int(sum(len(r["ev"]) < MIN_EVAL for r in rows)), rows=len(rows))
    cell_resid = {}
    for r in rows:
        i, j = kidx[(r["cc"], r["site"])], speeds.index(r["v"])
        A["ev"][i, j] = r["ev"][:n_eval]; A["dh"][i, j] = r["ref"].mean(); A["dt"][i, j] = r["all"].mean()
        A["sraw"][i, j] = r["sraw"]; A["swithin"][i, j] = r["all"].std(ddof=1)
        A["sdref"][i, j] = r["ref"].std(ddof=1)               # survey sd without the C2 (ii) excess (Table A reading)
        cell_resid.setdefault(r["v"], []).append(r["ref"] - r["ref"].mean())
    assert np.isfinite(A["dh"]).all(), "every (site, speed) needs passes"
    nu_cell = {k: max(stats.t.fit(np.concatenate(v), floc=0.0)[0], 2.5) for k, v in cell_resid.items()}
    for r in rows:
        A["nu"][kidx[(r["cc"], r["site"])], speeds.index(r["v"])] = nu_cell[r["v"]]
    A["K"] = np.full(nS, K_PLACEHOLDER)
    return A


def profiles_without(speeds_full, drop):
    """C5 consequence: the decision simulation's `profile` would give the mass of a removed class to the
    nearest remaining one, i.e. still verify that traffic with a neighbouring class's errors. The adapter therefore
    hands the decision simulation/the matching simulation a speed-profile TABLE (an input, not code) from which every profile speed whose nearest simulated class
    (on the full class list) failed C5 is removed and the rest renormalised. Returns (profiles, uncovered share per
    vehicle profile); a profile whose whole mass is removed is left out and reported as uncovered 1.0."""
    full = np.array(speeds_full, float)
    prof, unc = {}, {}
    for vt, table in BASE_PROFILES.items():
        keep = {v: q for v, q in table.items() if float(full[np.argmin(np.abs(full - v))]) not in set(drop)}
        tot = sum(table.values())
        unc[vt] = float(1.0 - sum(keep.values()) / tot)
        if keep:
            k = sum(keep.values())
            prof[vt] = {v: q / k for v, q in keep.items()}
    return prof, unc


def lambda_eff(tables, lams=(5, 15, 40)):
    """Per stratum and vehicle profile, q = the share of the profile's
    crossings that fall on qualified (site, speed class) cells (each profile speed takes its nearest simulated class; the
    sites of a class count equally), and lambda_eff = lambda x q. The window, gate and erasure losses act inside the decision simulation
    through the gated inputs and are not applied again."""
    out = {}
    for name, A in tables.items():
        speeds = np.array(A["speeds"], float)
        qshare = {v: float(np.mean(A["sraw"][:, j] <= SMAX)) for j, v in enumerate(A["speeds"])}
        qtab = {v: float(np.mean(A["sdref"][:, j] <= SMAX)) for j, v in enumerate(A["speeds"])}
        nearest = lambda v: float(speeds[np.argmin(np.abs(speeds - v))])
        rows = {}
        for vt, prof in A["profiles"].items():
            q = float(sum(w * qshare[nearest(v)] for v, w in prof.items()))
            qt = float(sum(w * qtab[nearest(v)] for v, w in prof.items()))
            rows[vt] = dict(q=q, q_tableA=qt, lambda_eff={str(l): l * q for l in lams})
        out[name] = dict(class_qualified_share={f"{v:g}": x for v, x in qshare.items()},
                         class_qualified_share_tableA={f"{v:g}": x for v, x in qtab.items()}, profiles=rows,
                         note="q: E2's qualification (survey sd inflated by the C2 (ii) excess tau, which drives E2's power); "
                              "q_tableA: survey sd <= 6 % as in Table A")
    return out


def common_levels(tables, sigma_max, n_levels):
    """One log-spaced grid over the qualified (site, speed) cells of all tables (P1 + P2), as requalify does per table."""
    s = np.concatenate([(SIGMA_INFL * T["sraw"])[T["sraw"] <= sigma_max] for T in tables] + [np.zeros(0)])
    if not s.size:                                      # no qualified cell (no stratum covered, or a forced dev stratum):
        return FIXED_LEVELS.copy()                      # the data grid is undefined; the fixed grid stands in (sensitivity)
    return np.exp(np.linspace(np.log(s.min()), np.log(s.max()), n_levels))


def requalify_on(levels):
    def requalify(A, sigma_max, n_levels):
        B = dict(A)
        qual = A["sraw"] <= sigma_max if sigma_max else np.ones_like(A["sraw"], bool)
        if A.get("force_qualify"):                      # development only (--dev-force-cover; never primary)
            qual = np.ones_like(A["sraw"], bool)
        lv = levels if n_levels == len(levels) else np.exp(np.linspace(np.log(levels[0]), np.log(levels[-1]), n_levels))
        s_infl = SIGMA_INFL * A["sraw"]
        lvl = np.minimum(np.searchsorted(lv, s_infl - 1e-15), n_levels - 1)
        B["levels"], B["lvl"], B["sg"] = lv, lvl, lv[lvl]
        return B, qual
    return requalify


def governing(a):
    path = os.path.join(RESULTS, f"governing_{a.tag}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"{path} missing: run pipeline/stage3_criteria.py first")
    g = json.load(open(path))
    if a.tag == "primary" and (a.mapping or a.prb):
        raise ValueError("primary: the governing mapping / PRB are read from the governing file, not given")
    return g


def dev_switches(a):
    """Development-only switches of the plumbing test: --dev-tau0
    sets the C2 (ii) excess tau to 0; --dev-force-cover also
    forces a stratum that would still be "not covered" (no qualified (site, class), or every class failed C5) to be
    covered: every (site, class) qualified and no C5 drop. Both are refused unless the tag starts with dev_, so a primary
    or sens_ run can never take them; their outputs are plumbing evidence only (exit codes, file names, branch ids)."""
    tau0, force = bool(getattr(a, "dev_tau0", False)), bool(getattr(a, "dev_force_cover", False))
    if (tau0 or force) and not a.tag.startswith("dev_"):
        raise ValueError(f"--dev-tau0 / --dev-force-cover are development switches: refused for tag {a.tag!r}")
    return tau0 or force, force


def build_tables(a, g):
    tables, hashes, notes, status = {}, {}, {}, {}
    tau0, force = dev_switches(a)
    for name, (fam, cell) in STRATA.items():
        recs, h = load_recs(fam, a.tag)
        if not recs:
            status[name] = "no_records"
            continue
        gs = g.get(name if name != "secondary_38901" else "urban")
        mp, prb = (a.mapping, a.prb) if (a.mapping and a.tag != "primary") else (gs["mapping"], int(gs["prb"]))
        drop = tuple(float(c.split("-")[1]) for c in g.get("c5_failing", []) if c.startswith(cell + "-"))
        tau = max([t for k, t in g.get("c2_tau", {}).items() if k.startswith(cell + "-")] or [0.0])
        if tau0:
            tau = 0.0
        rng = np.random.default_rng([C["seeds"]["bootstrap"], 2, STRATA[name][0].encode()[0]])
        forced = []
        try:
            try:
                A = site_table(recs, cell, mp, prb, tau=tau, drop_speeds=drop, rng=rng)
            except NoClass:                             # every class failed C5: reported as not covered
                if not force:
                    notes[name] = "not covered (every class failed C5)"
                    status[name] = "not_covered_c5"
                    continue
                A = site_table(recs, cell, mp, prb, tau=tau, drop_speeds=(), rng=rng)
                A["force_qualify"] = True
                drop = ()
                forced.append("no_c5_drop")
        except ShortStratum as e:                       # floor (min_eval_e2_floor): pre-specified note
            notes[name] = f"not covered (gated test passes per (site, speed) below the floor: {e})"
            status[name] = "not_covered_floor"
            continue
        if not np.any(A["sraw"] <= SMAX):
            if not force:
                notes[name] = "not covered (no qualified (site, class))"
                status[name] = "not_covered_unqualified"
                continue
            A["force_qualify"] = True
            forced.append("all_qualified")
        A["profiles"], A["uncovered_share"] = profiles_without(A["speeds_full"], drop)
        if not A["profiles"] or not A["speeds"]:
            if not force:
                notes[name] = "not covered (every class failed C5)"
                status[name] = "not_covered_c5"
                continue
            A = site_table(recs, cell, mp, prb, tau=tau, drop_speeds=(), rng=np.random.default_rng(
                [C["seeds"]["bootstrap"], 2, STRATA[name][0].encode()[0]]))
            A["force_qualify"] = True
            A["profiles"], A["uncovered_share"] = profiles_without(A["speeds_full"], ())
            drop = ()
            forced.append("no_c5_drop")
        tables[name], hashes[name] = A, h
        status[name] = "covered" if A["n_eval"] >= MIN_EVAL else "covered_short"
        notes[name] = dict(mapping=mp, prb=prb, tau=tau, uncovered_speeds=list(drop), truncation=A["truncation"],
                           uncovered_traffic_share=A["uncovered_share"], n_eval=A["n_eval"],
                           rows_below_min_eval=A["rows_below_min_eval"], rows=A["rows"], status=status[name])
        if tau0:
            notes[name]["dev_forced"] = ["tau0"] + forced           # development only (dev_switches)
        if A["n_eval"] < MIN_EVAL:                      # floor active (else site_table aborted)
            notes[name]["n_eval_note"] = (f"E2 / E5 on {A['n_eval']} gated test passes per (site, speed), fewer than "
                                          f"min_eval_e2 = {MIN_EVAL} ({A['rows_below_min_eval']} of {A['rows']} rows below; "
                                          f"{A['truncation']} passes truncated)")
    STRATUM_STATUS.clear()
    STRATUM_STATUS.update(status)
    return tables, hashes, notes


def extract(rep_by_stratum):
    """Fixed row extraction: profile rows from the matching stratum; suburban: worse per metric of both."""
    out = {}
    for prof, sources in EXTRACTION.items():
        for name in sources:
            rep = rep_by_stratum.get(name)
            if rep is None:
                continue
            for section, rows in rep.items():
                if not isinstance(rows, dict):
                    continue
                for key, r in rows.items():
                    if f"|{prof}|" not in f"|{key}|" or not isinstance(r, dict):
                        continue
                    tgt = out.setdefault(section, {}).setdefault(key, dict(r, source=name))
                    if tgt.get("source") == name:
                        continue
                    first = tgt["source"]
                    for m, v in r.items():
                        if m == "source" or m not in tgt or not isinstance(v, (int, float)):
                            continue
                        dr = direction(section, key, m)
                        if dr == "high":
                            tgt[m] = max(tgt[m], v)
                        elif dr == "low":
                            tgt[m] = min(tgt[m], v)
                        else:                                       # descriptive: both values reported
                            tgt[f"{m}_by_source"] = {first: tgt[m], name: v}
                    tgt["source"] = "worse of urban and highway (per metric direction)"
    return out


def run_e2(a, tables, levels, hashes):
    import decisions as E2                              # the decision simulation (monitor variant) + the R1 escalation
    reps = {}
    for name, A in tables.items():
        E.load_sites = lambda tags, A=A: A
        E.PROFILES = A["profiles"]                               # C5 consequence: failing classes removed
        E2.requalify = requalify_on(common_levels([A], SMAX, 8) if name == "secondary_38901" else levels)
        E2.OUT = os.path.join(OUT, f"decisions_{a.tag}_{name}_{hashes[name][:12]}")
        sys.argv = ["decisions", "--levels", "8"] + (["--smoke"] if a.smoke else ["--vehicles", str(C["run"]["e2_vehicles"])])
        E2.main()
        f = sorted(glob.glob(os.path.join(E2.OUT, "decisions_L8_*.json")))
        reps[name] = json.load(open([x for x in f if "partial" not in x][0])) if f else None
    E.PROFILES = BASE_PROFILES
    json.dump(dict(rows=extract({k: v for k, v in reps.items() if k != "secondary_38901"}),
                   secondary_38901=reps.get("secondary_38901"), input_sha256=hashes, provenance=PROV,
                   uncovered_traffic_share={k: A["uncovered_share"] for k, A in tables.items()},
                   n_eval={k: dict(n_eval=A["n_eval"], min_eval_e2=MIN_EVAL, rows_below_min_eval=A["rows_below_min_eval"],
                                   rows=A["rows"]) for k, A in tables.items()},
                   lambda_eff=lambda_eff({k: A for k, A in tables.items() if k != "secondary_38901"})),
              open(os.path.join(OUT, f"e2_extracted_{a.tag}.json"), "w"), indent=1, default=float)


def e3_arrays(recs, v0, mapping, prb, arm="gen_on"):
    """Per P5 site: z of the evaluated and beta passes under the survey and the traffic standardisation. every pass keeps its slot in track order: a pass that
    is not included (window rule, non-finite fit) or is discarded by the residual gate (constants window_rule.
    e2_e3_e5_inputs: gated inputs) carries NaN, which the consistency simulation maps to z_q = 0 as for a failed fit; the
    fixed per-site blocks of consistency_analysis / consistency_report (100 evaluated, 30 per beta) therefore stay
    aligned. References
    are never gated (they define the standardisation); traffic passes are gated like evaluated passes. Asserts 100 / 30;
    reports the excluded and gated counts per site."""
    ne, nsp = 100, 30
    sel = [r for r in recs if r["cell"] == "P5" and r["v0"] == v0 and r["arm"] == arm and r["mapping"] == mapping
           and r["prb"] == prb]
    sites = sorted({r["site"] for r in sel})

    def median_mad(d):
        d = d[np.isfinite(d)]
        m = np.median(d)
        return m, 1.4826 * np.median(np.abs(d - m))
    out = []
    for s in sites:
        rs = sorted([r for r in sel if r["site"] == s], key=lambda r: r["track"])
        ref_ok = [r for r in rs if r["role"] == "ref" and r.get("beta", 0.0) == 0.0 and included(r)]
        if len(ref_ok) < MIN_REF:                       # as stage3_criteria: site dropped, reported
            out.append(dict(site=s, dropped=f"only {len(ref_ok)} included reference passes (< {MIN_REF})"))
            continue
        rms_ref = np.median(np.sqrt([r["mse"] for r in ref_ok]))
        keep = lambda r: included(r) and (r["role"] == "ref" or np.sqrt(r["mse"]) <= GATE * rms_ref)
        d = lambda role, beta=0.0: np.array([r["d"] if keep(r) else np.nan for r in rs
                                             if r["role"] == role and r.get("beta", 0.0) == beta], float)
        m_ref, s_ref = median_mad(d("ref")); m_trf, s_trf = median_mad(d("traffic"))
        s_ref *= 1.15
        ev = d("eval")
        sp = {b: d("eval", b) for b in (-0.05, -0.10)}
        if ev.size != ne or any(v.size != nsp for v in sp.values()):
            raise RuntimeError(f"P5 site {s}: {ev.size} evaluated / {[v.size for v in sp.values()]} beta passes "
                               f"({ne} / {nsp} expected)")
        cnt = lambda role, beta=0.0: dict(
            not_included=int(sum(not included(r) for r in rs if r["role"] == role and r.get("beta", 0.0) == beta)),
            gated=int(sum(included(r) and not keep(r) for r in rs if r["role"] == role and r.get("beta", 0.0) == beta)))
        rms = {b: np.median([np.sqrt(r["mse"]) for r in rs if r["role"] == "eval" and r.get("beta", 0.0) == b
                             and r["jr"] < nsp and included(r)]) for b in (0.0, -0.05, -0.10)}
        out.append(dict(site=s, ev_ref=(ev - m_ref) / s_ref, ev_trf=(ev - m_trf) / s_trf,
                        sp_ref={b: (v - m_ref) / s_ref for b, v in sp.items()},
                        sp_trf={b: (v - m_trf) / s_trf for b, v in sp.items()}, sd_d_ref=float(s_ref / 1.15),
                        sd_d_trf=float(s_trf), gate_ratio_beta={str(b): float(rms[b] / rms[0.0]) for b in (-0.05, -0.10)},
                        excluded=dict(eval=cnt("eval"), traffic=cnt("traffic"),
                                      **{f"beta{b:+.2f}": cnt("eval", b) for b in (-0.05, -0.10)})))
    return out


def write_e3(a, root, g):
    from consistency import label
    cells, arrays, extra = [], {}, {}
    ne, nsp = 100, 30
    for i, (fam, v0, stratum) in enumerate((("5U", 15.0, "urban"), ("5H", 30.0, "highway"))):
        sel_path = os.path.join(RUNS, f"g1_{fam}", "selection.json")
        if not os.path.exists(sel_path):
            raise RuntimeError(f"{sel_path} missing: P5 site selection must exist")
        sel = json.load(open(sel_path))
        if sel.get("not_applicable"):
            extra[stratum] = "not applicable"
            continue
        recs, _ = load_recs(fam, a.tag)
        gs = g[stratum]
        allp = [p for p in e3_arrays(recs, v0, gs["mapping"], int(gs["prb"])) if p["site"] in sel["selected"]]
        assert len(allp) == len(sel["selected"]), "selected P5 sites without stage-2 records"
        parts = [p for p in allp if "dropped" not in p]
        extra[f"{stratum}_dropped_sites"] = {str(p["site"]): p["dropped"] for p in allp if "dropped" in p}
        if not parts:
            extra[stratum] = "not covered (every selected P5 site dropped)"
            continue
        cells.append(dict(v0=v0, K_db=K_PLACEHOLDER, specular_db=None,
                          sites=[dict(site=p["site"], sd_d_ref=p["sd_d_ref"], sd_d_trf=p["sd_d_trf"],
                                      gate_ratio_beta=p["gate_ratio_beta"], excluded=p["excluded"]) for p in parts]))
        j = len(cells) - 1
        for kind in ("ref", "trf"):
            arrays[f"c{j}_honest_{kind}"] = np.concatenate([p[f"ev_{kind}"] for p in parts])
            assert arrays[f"c{j}_honest_{kind}"].size == ne * len(parts), "E3 block alignment"
            for b in (-0.05, -0.10):
                arrays[f"c{j}_beta{label(b)}_{kind}"] = np.concatenate([p[f"sp_{kind}"][b] for p in parts])
                assert arrays[f"c{j}_beta{label(b)}_{kind}"].size == nsp * len(parts), "E3 block alignment"
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    tag = f"adapter_{a.tag}"
    json.dump(dict(cells=cells, eval=ne, spoof=nsp, source="decision adapter", not_applicable=extra, provenance=PROV,
                   input_sha256={fam: load_recs(fam, a.tag)[1] for fam in ("5U", "5H")}),
              open(os.path.join(root, "results", f"consistency_{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(root, "results", f"consistency_z_{tag}.npz"), **arrays)
    return tag


def run_e3(a, tables, levels, g):
    import consistency_analysis as A3
    root = os.path.join(OUT, "e3")
    tag = write_e3(a, root, g)
    lmap = {}
    for name, v0 in (("urban", 15.0), ("highway", 30.0)):
        if name not in tables:
            continue
        B, qual = requalify_on(levels)(tables[name], SMAX, 8)
        j = B["speeds"].index(v0) if v0 in B["speeds"] else int(np.argmin(np.abs(np.array(B["speeds"]) - v0)))
        lmap[(v0, K_PLACEHOLDER, False)] = sorted(int(x) for x in B["lvl"][qual[:, j], j])
    A3.level_map = lambda: lmap
    A3.k_weights = lambda ks, mapping="floor": {k: 1.0 / len(ks) for k in ks}
    A3.HERE = root
    A3.main(tag)
    # sensitivity: every P5 site at its own sigma level on the common grid instead of the
    # site-index quantile of the stratum's level list; same z files under the tag <tag>_ownlevel
    import shutil
    from pipeline.stage3_criteria import ref_qualification
    meta = json.load(open(os.path.join(root, "results", f"consistency_{tag}.json")))
    own = {}
    for c in meta["cells"]:
        fam = "5U" if c["v0"] == 15.0 else "5H"
        gs = g["urban" if fam == "5U" else "highway"]
        recs, _ = load_recs(fam, a.tag)
        lv = []
        for site in c["sites"]:
            ref = [r for r in recs if r["cell"] == "P5" and r["site"] == site["site"] and r["role"] == "ref"
                   and r["arm"] == "gen_on" and r["mapping"] == gs["mapping"] and r["prb"] == int(gs["prb"])
                   and r.get("beta", 0.0) == 0.0]
            sd = ref_qualification(ref)["sd_ref"]
            lv.append(int(min(np.searchsorted(levels, SIGMA_INFL * sd - 1e-15), len(levels) - 1)))
        own[(c["v0"], K_PLACEHOLDER, False)] = lv               # site order: index (s + 0.5) / n * n = s
    for ext in ("json", "npz"):
        src = os.path.join(root, "results", f"consistency_{'z_' if ext == 'npz' else ''}{tag}.{ext}")
        shutil.copyfile(src, src.replace(tag, f"{tag}_ownlevel"))
    A3.level_map = lambda: own
    A3.main(f"{tag}_ownlevel")


def realised_k_median(fam, cell, tag, prb):
    """stage3_extra_checks.realised_k_median on the family's records (one realised-K function)."""
    from pipeline.stage3_extra_checks import realised_k_median as rkm
    return rkm(load_recs(fam, tag)[0], cell, prb)


def inhouse_k_cell(k_med):
    from pipeline.stage3_extra_checks import inhouse_k_cell as ikc
    return ikc(k_med)


def run_e5(a, tables, levels, g, hashes=None):
    import matching as E5
    orig = E5.load_consistency_z
    E5.requalify = requalify_on(levels)
    reps, zsrc = {}, {}
    for name in ("urban", "highway"):
        if name not in tables:
            continue
        A = tables[name]
        fam, cell = STRATA[name]
        k_med = realised_k_median(fam, cell, a.tag, int(g[name]["prb"]))
        if a.tag == "primary" and not np.isfinite(k_med):
            raise RuntimeError(f"{name}: no realised in-band K in the records (equal-K rule needs it)")
        K_sel = inhouse_k_cell(k_med)
        zsrc[name] = dict(file=os.path.relpath(INHOUSE_E3, ER), sha256=sha256_path(INHOUSE_E3), realised_K_median_db=k_med,
                          inhouse_K_cell_db=K_sel, label="in-house consistency z (P5 not run)")

        def stratum_z(path, kind="trf", A=A, K_sel=K_sel):    # in-house z at the equal-K cell for every speed
            zin = orig(path, kind)
            return {(v, K_PLACEHOLDER, False): zin[(v, K_sel, False)] for v in A["speeds"]}
        E5.load_consistency_z = stratum_z
        E.load_sites = lambda tags, A=A: A
        E.PROFILES = A["profiles"]                               # C5 consequence: failing classes removed
        E5.OUT = os.path.join(OUT, f"matching_{a.tag}_{name}")
        os.makedirs(E5.OUT, exist_ok=True)
        sys.argv = ["matching", "--vehicles", str(400 if a.smoke else C["run"]["e2_vehicles"]), "--e3", INHOUSE_E3,
                    "--kind", "trf"]
        E5.main()
        f = sorted(glob.glob(os.path.join(E5.OUT, "*.json")))
        reps[name] = {"e5": json.load(open(f[0]))} if f else None
    E.PROFILES = BASE_PROFILES
    json.dump(dict(urban=reps.get("urban"), highway=reps.get("highway"), provenance=PROV, input_sha256=hashes,
                   e3_z_source=zsrc,
                   uncovered_traffic_share={k: A["uncovered_share"] for k, A in tables.items()},
                   lambda_eff=lambda_eff({k: A for k, A in tables.items() if k != "secondary_38901"})),
              open(os.path.join(OUT, f"e5_extracted_{a.tag}.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["e2", "e3", "e5"])
    ap.add_argument("--grid", default="fixed", choices=["fixed", "data"], help="data: the data-derived grid (sensitivity row)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mapping", default=None, help="development only")
    ap.add_argument("--prb", type=int, default=None, help="development only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev-tau0", action="store_true", help="development only (dev_ tags): C2 (ii) tau = 0 (plumbing test)")
    ap.add_argument("--dev-force-cover", action="store_true",
                    help="development only (dev_ tags): tau = 0 and a stratum that would be 'not covered' is forced "
                         "covered (every (site, class) qualified, no C5 drop); plumbing test, numbers never read")
    a = ap.parse_args()
    dev_switches(a)                                     # refused unless the tag starts with dev_
    if a.tag == "primary" and a.smoke:
        raise ValueError("primary run cannot be a smoke")
    check_tag(a.tag, stage=3)                           # tag format and fixed knobs
    if a.what == "e3":
        raise ValueError("the consistency simulation is not run on generator data (P5 not run); it stays in-house")
    global OUT
    if a.grid == "data":
        OUT = OUT + "_datagrid"                        # sensitivity row, never governs
    PROV.update(provenance(a.tag, stage=3, what=a.what, smoke=bool(a.smoke),
                           **({"dev_tau0": True} if a.dev_tau0 else {}),
                           **({"dev_force_cover": True} if a.dev_force_cover else {})))
    g = governing(a)
    tables, hashes, notes = build_tables(a, g)
    data_levels = common_levels([tables[k] for k in ("urban", "highway") if k in tables], SMAX, 8)
    levels = FIXED_LEVELS if a.grid == "fixed" else data_levels
    qual = np.concatenate([(SIGMA_INFL * tables[k]["sraw"])[tables[k]["sraw"] <= SMAX] for k in ("urban", "highway")
                           if k in tables] + [np.zeros(0)])
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(levels=levels.tolist(), grid=a.grid, data_derived_levels=data_levels.tolist(),
                   top_level_clipped_share=float(np.mean(qual > FIXED_LEVELS[-1])) if qual.size else None,
                   tag=a.tag, strata=notes, stratum_status=dict(STRATUM_STATUS), input_sha256=hashes, provenance=PROV),
              open(os.path.join(OUT, f"tables_{a.tag}.json"), "w"), indent=1, default=float)
    if a.what == "e2":
        run_e2(a, tables, levels, hashes)
    else:
        run_e5(a, {k: v for k, v in tables.items() if k != "secondary_38901"}, levels, g, hashes)


if __name__ == "__main__":
    main()
