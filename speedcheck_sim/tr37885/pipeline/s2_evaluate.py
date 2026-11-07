"""Study 2: stage-2 re-runs from stage-1 records with the study-2 fields of pipeline/s2_fields.py beside the study-1
fields. pipeline/stage2_evaluate.py is not changed: its evaluate_site runs on a work copy of the site whose stage-1 meta
lists the
wanted passes only (the npz is linked, its SHA-256 checked as usual); two of its module functions are wrapped for the
duration of the call, each returning exactly what the original returns: fdoa_from_samples (to see the received samples
of the record) and one_stat (to hand the record's beacons to S2Fields.core). The study-2 fields are then added to the
finished records (S2Fields.attach), so every study-1 field is the study-1 code's own value. Two tags:
  dev_s2_cal  Dev-1 P0U / P0N, reference passes (role == "ref", free road) only. The only development computation
              before the development stage.
  dev_s2      every pass (the stage-2 fields do not depend on the calibration: the E-d decision is made at stage 3
              only). Refused before the development-stage design record is timestamped and without a complete, finite
              calibration in the constants; on the Dev-2 families U / N also refused before the study-1 records are
              merged (evaluation_v6.step0.study1_merge_done) and run with the reading-T E-e fields (surrogate survey,
              test jr < 30).
  Dev-3 (D3U) is not run here: its stage 2 is the study-1 stage 2 (no study-2 fields), run per part by run_study2.sh.
After each site the study-1 field check compares the study-1 fields of the new records with those of the existing
records of the same passes (--check-against <eval dir of the archive>); a mismatch stops the run. Only counts and field
names are printed, never a value. Output: <run dir>/eval_<tag>/site_<k>.json.
Usage: python pipeline/s2_evaluate.py --family P0U --tag dev_s2_cal --run-dir <stage-1 record dir>
           --check-against <run dir>/eval_dev_p0 [--sites 0 6] [--workers 2]"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import s2_constants as S2  # noqa: E402
from pipeline.stage1_channels import C, FAMILIES  # noqa: E402
from pipeline.s2_fields import VERSION, S2Fields, study1_field_check  # noqa: E402


def lane_width(fam):
    return C["sumo_implementation"]["highway" if FAMILIES[fam]["scenario"] == "highway" else "urban"]["lane_width_m"]


def win_arc():
    return tuple(C["window_rule"]["arc_L"])


@contextmanager
def wrapped(E, s2):
    """Wrap evaluate.fdoa_from_samples and evaluate.one_stat (module globals that evaluate_site looks up at call time)
    for one call of evaluate_site; both return exactly what the originals return; restored afterwards."""
    cap = {}
    f0, o0 = E.fdoa_from_samples, E.one_stat

    def fdoa_cap(yb, packet_s):
        cap["yb"] = yb
        return f0(yb, packet_s)

    def one_stat_s2(t_b, fd, vx, x, rho_true, site):
        st = o0(t_b, fd, vx, x, rho_true, site)
        yb = cap.pop("yb")
        if yb.shape[1] != t_b.size:                    # dev_s2 runs have no erasures and no decoding mask
            raise RuntimeError("study-2 fields: the record's beacons are a subset of the received beacons")
        s2.core(t_b, fd, vx, x, rho_true, site, yb)
        return st

    E.fdoa_from_samples, E.one_stat = fdoa_cap, one_stat_s2
    try:
        yield
    finally:
        E.fdoa_from_samples, E.one_stat = f0, o0


def evaluate_site_s2(fam, site, run_dir, tag, s2, **kw):
    """One site: the stage 2 on the wanted passes plus the study-2 fields. Resume: an existing output with
    the same study-2 provenance is kept; another provenance is refused."""
    from pipeline import stage2_evaluate as E
    from pipeline.provenance import diff_provenance, provenance, same_provenance
    out_dir = os.path.join(run_dir, f"eval_{tag}")
    final = os.path.join(out_dir, f"site_{site:03d}.json")
    prov = provenance(tag, stage=2, s2=s2.knobs(), **json.loads(json.dumps(kw)))   # kw (mappings, prbs): tests only
    if os.path.exists(final):
        old = json.load(open(final)).get("s2_provenance", {})
        if not same_provenance(old, prov):
            raise RuntimeError(f"{final} exists with a different provenance: {diff_provenance(old, prov)}")
        return final
    work = os.path.join(run_dir, f".s2work_{tag}_{site:03d}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    os.symlink(os.path.abspath(os.path.join(run_dir, f"site_{site:03d}.npz")), os.path.join(work, f"site_{site:03d}.npz"))
    S1 = json.load(open(os.path.join(run_dir, f"site_{site:03d}.json")))
    meta = S1["meta"]
    S1["meta"] = [m for m in meta if s2.wants(m)]
    json.dump(S1, open(os.path.join(work, f"site_{site:03d}.json"), "w"))
    with wrapped(E, s2):
        p = E.evaluate_site(fam, site, work, tag, **kw)
    d = json.load(open(p))
    s2.attach(d["recs"], meta)
    d.update(s2_provenance=prov, s2_fields=VERSION)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(d, open(final + ".tmp", "w"))
    os.replace(final + ".tmp", final)
    shutil.rmtree(work, ignore_errors=True)
    return final


def plan(fam, tag, consts=None):
    """The S2Fields configuration of (family, tag), or an exception if the design does not allow the run now."""
    ev6 = (consts or C).get("evaluation_v6")
    S2.check_against(ev6)
    st0 = ev6["step0"]
    if tag == S2.CAL_TAG:
        cal_fams = sorted({f for v in S2.DEV1_CAL_FAMILIES.values() for f in v})
        if fam not in cal_fams:
            raise ValueError(f"{tag}: only the Dev-1 calibration families {cal_fams}")
        return dict(only_roles=("ref",), surrogate=False, lane_width=lane_width(fam), win_arc=win_arc())
    if tag != S2.DEV_TAG:
        raise ValueError(f"tag must be {S2.CAL_TAG!r} or {S2.DEV_TAG!r}")
    if not st0.get("timestamped"):
        raise RuntimeError("dev_s2 before the development-stage design record is timestamped")
    if st0.get("calibration") is None:
        raise RuntimeError("dev_s2 needs evaluation_v6.step0.calibration (pipeline/s2_calibrate.py)")
    S2.check_calibration(st0["calibration"])
    if fam in S2.DEV2_FAMILIES:
        if not st0.get("study1_merge_done"):
            raise RuntimeError("Dev-2 (study-1 main-run records) only after the study-1 records are merged")
        return dict(only_roles=None, surrogate=True, lane_width=lane_width(fam), win_arc=win_arc())
    if fam in S2.DEV1_FAMILIES:
        return dict(only_roles=None, surrogate=False, lane_width=lane_width(fam), win_arc=win_arc())
    if fam == S2.DEV3["family"]:
        raise ValueError(f"{fam}: Dev-3 stage 2 is the study-1 stage 2 (pipeline/stage2_evaluate.py, tag {S2.DEV_TAG}) "
                         "run by run_study2.sh, not this wrapper")
    raise ValueError(f"{tag}: family {fam} is not a development family of study 2")


def _job(a):
    fam, site, run_dir, tag, cfg, check = a
    path = evaluate_site_s2(fam, site, run_dir, tag, S2Fields(**cfg))
    n_bad = None
    if check:
        new = json.load(open(path))["recs"]
        old = json.load(open(os.path.join(check, f"site_{site:03d}.json")))["recs"]
        bad = study1_field_check(new, old)
        n_bad = len(bad)
        if bad:
            fields = sorted({str(f) for _, f in bad})
            raise RuntimeError(f"study-1 field check failed: {fam} site {site}: {n_bad} mismatches in fields {fields}")
    return path, n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tag", required=True, choices=[S2.CAL_TAG, S2.DEV_TAG])
    ap.add_argument("--run-dir", required=True, help="folder with the stage-1 records (site_<k>.npz / .json)")
    ap.add_argument("--check-against", required=True, help="the existing eval_<tag> folder of the same passes")
    ap.add_argument("--sites", type=int, nargs=2, default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = plan(a.family, a.tag)
    if a.sites is None:
        n = C["evaluation_v4"]["p0"]["families"][a.family]["sites"] if a.family in S2.DEV1_FAMILIES else 10
        a.sites = [0, n]
    jobs = [(a.family, s, a.run_dir, a.tag, cfg, a.check_against) for s in range(*a.sites)
            if os.path.exists(os.path.join(a.run_dir, f"site_{s:03d}.npz"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for p, nb in ex.map(_job, jobs):
            print(p, f"study-1 field check: {nb} mismatches", flush=True)


if __name__ == "__main__":
    main()
