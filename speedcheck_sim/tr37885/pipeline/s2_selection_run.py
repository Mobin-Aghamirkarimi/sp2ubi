"""Study 2, development stage: loads exactly the pre-specified development records, refuses any record that
this code did not produce, runs the pre-specified rules and the diagnostics D0.1-D0.6, and writes step0_decision.json
and step0_report.json. It holds no rule of its own and prints no number.

Inputs:
  --dev1 P0U=<dir> --dev1 P0N=<dir> --dev1 P0H=<dir>   Dev-1: the dev_s2 stage-2 re-run on the pilot stage-1 records
                                                       (tag dev_s2; all passes)
  --dev2 U=<dir> --dev2 N=<dir>                        Dev-2: the dev_s2 re-run on the study-1 U / N stage-1 records
  --dev3 <dir> [--dev3 <dir>]                          Dev-3: D3U parts a and b, the study-1 stage 2 under tag dev_s2;
                                                       merged by site, both parts required
  --reference <constants file>                         optional: evaluation_v6 must equal its block except the flags
Refusals (each stops the loader before any rule runs):
  * the guard of s2_selection: the development-stage record timestamped, the study-1 records merged and a complete,
    finite calibration recorded;
  * every record file: tag dev_s2, the expected family and stage-1 and stage-2 source hashes (its provenance; the
    study-2 files' s2_provenance) equal to this tree's stage_hashes; Dev-1 / Dev-2 files also carry the study-2 fields
    of this code version; Dev-3 must hold both parts' sites (0-4).
Usage: python pipeline/s2_selection_run.py --dev1 P0U=<dir> ... --out-dir ../results/study2"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import s2_constants as S2  # noqa: E402


def current_ev6():
    from pipeline import stage3_criteria as K
    return K.C.get("evaluation_v6")


def _stage(prov, k):
    return ((prov or {}).get("stage_sha256") or {}).get(str(k))


def load_dir(d, fam, kind, now):
    """The records of one stage-2 folder; kind 's2' or 'study-1' (Dev-3)."""
    from pipeline.s2_fields import VERSION
    paths = sorted(glob.glob(os.path.join(d, "site_*.json")))
    if not paths:
        raise RuntimeError(f"{fam}: no stage-2 files in {d}")
    recs = []
    for f in paths:
        x = json.load(open(f))
        if x.get("tag") != S2.DEV_TAG or x.get("family") != fam:
            raise RuntimeError(f"{f}: tag {x.get('tag')!r} / family {x.get('family')!r} is not {S2.DEV_TAG!r} / {fam!r}")
        prov = x.get("s2_provenance") if kind == "s2" else x.get("provenance")
        if kind == "s2" and x.get("s2_fields") != VERSION:
            raise RuntimeError(f"{f}: study-2 fields {x.get('s2_fields')!r} are not this code's {VERSION!r}")
        for k in (1, 2):
            ok = now[str(k)] if isinstance(now[str(k)], (set, frozenset)) else {now[str(k)]}
            if _stage(prov, k) not in ok:
                raise RuntimeError(f"{f}: stage-{k} sources differ from this code")
        for r in x["recs"]:
            r["family"] = fam
        recs += x["recs"]
    return recs


def load_all(a):
    from pipeline.provenance import stage_hashes
    now = {kk: {v} for kk, v in stage_hashes().items()}
    pairs = lambda specs: dict(s.split("=", 1) for s in specs)
    d1, d2 = pairs(a.dev1), pairs(a.dev2)
    if sorted(d1) != sorted(S2.DEV1_FAMILIES) or sorted(d2) != sorted(S2.DEV2_FAMILIES):
        raise RuntimeError(f"the inputs are Dev-1 {S2.DEV1_FAMILIES} and Dev-2 {S2.DEV2_FAMILIES}")
    dev1 = {f: load_dir(p, f, "s2", now) for f, p in d1.items()}
    dev2 = {f: load_dir(p, f, "s2", now) for f, p in d2.items()}
    dev3 = []
    for p in a.dev3:
        dev3 += load_dir(p, S2.DEV3["family"], "study1", now)
    want = set(range(*S2.DEV3["sites"]))
    got = {r["site"] for r in dev3}
    if got != want:
        raise RuntimeError(f"Dev-3 must hold the sites of both parts {sorted(want)} (got {sorted(got)}): D0.2 runs only "
                           "after both parts are complete")
    return dev1, dev2, dev3


def run(a):
    from pipeline import s2_selection as S0
    cal = S0.guard()
    if a.reference:
        S2.check_same_as_reference(current_ev6(), json.load(open(a.reference)).get("evaluation_v6"), S2.DEV_FLAG_KEYS)
    dev1, dev2, dev3 = load_all(a)
    dec = S0.decide(dev1, dev2, dev3, {k: cal[k] for k in S2.CALIBRATION_KEYS})
    u = dev2["U"]
    rep = dict(D0_1=dict(Dev1=S0.d0_1(dev1["P0U"], S0.URBAN), Dev2=S0.d0_1(u, S0.URBAN)),
               D0_2=dec["D0_R1"], D0_3=dict(Dev1=S0.d0_3(dev1["P0U"]), Dev2=S0.d0_3(u)),
               D0_4=dict(Dev1=S0.d0_4(dev1["P0U"], cal), Dev2=S0.d0_4(S0.CS.reading_F(u, dev2=True), cal)), D0_5=S0.d0_5(u),
               D0_6=dict(Dev1=S0.d0_6(dev1["P0N"]), Dev2=S0.d0_6(dev2["N"])),
               label="development stage; Dev-2 numbers are 'development, study-1 records'")
    os.makedirs(a.out_dir, exist_ok=True)
    S0.write_decision(dec, os.path.join(a.out_dir, "step0_decision.json"))
    S0.write_decision(rep, os.path.join(a.out_dir, "step0_report.json"))
    print("written", os.path.join(a.out_dir, "step0_decision.json"), os.path.join(a.out_dir, "step0_report.json"))
    return dec, rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None, help="optional: a constants file of the development stage; "
                    "evaluation_v6 must equal it except the flags")
    ap.add_argument("--dev1", action="append", required=True, help="FAMILY=<eval_dev_s2 folder>")
    ap.add_argument("--dev2", action="append", required=True, help="FAMILY=<eval_dev_s2 folder>")
    ap.add_argument("--dev3", action="append", required=True, help="<eval_dev_s2 folder of a D3U part>")
    ap.add_argument("--out-dir", required=True)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
