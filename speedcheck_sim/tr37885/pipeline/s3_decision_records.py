"""Study 3, detection rules with the urban estimator: the stage-2 record set of the urban sites for one decision-level
run. The carrier records are the stage-2 records of the U3 sites 30-39 (stage2_evaluate.py, arms gen_off, gen_on and
inhouse); the statistics of the urban estimator come from pipeline/s3_estimate.py (variant h_pp_now_c20).

Rules:
* survey: with "free", the records of role "ref" (survey in traffic) are dropped and role "reff" is relabelled "ref"
  (stage 3 and the decision adapter read "ref" as the survey); with "traffic", role "reff" is dropped.
* estimator "urban": prof, match and mse of a record are replaced by those of the urban estimator (matched by site,
  track, mapping and PRB; arm gen_on from the series label U3, arm inhouse from U3in), and finite is set from the new
  values. The consistency statistics d and d_d of the base estimator are removed, the records with a Doppler scale
  (beta != 0) and the arm gen_off are dropped: the urban estimator has none of them, and no quantity of the detection
  rules reads them.
* channel "tr": the arms gen_on and inhouse are written; channel "inhouse": only the arm inhouse, relabelled gen_on
  (stage 3 and the adapter take their governing quantities from gen_on).
* estimator "base" (control run, channel "tr" only): the carrier records unchanged except for the survey rule.
Every call re-reads what it wrote and checks it against its inputs: each substituted statistic equals the value of the
urban estimator, each of its records is used exactly once, the matched fields agree, and with estimator "base" each
written record equals its carrier record. Prints counts only.
Usage: python pipeline/s3_decision_records.py --base runs/g1_U3_dev_s3/eval_dev_s3 --res <s3_estimate folder of the
           survey> --survey free|traffic --channel tr|inhouse --estimator urban|base --out <eval_<tag> folder>"""
import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pipeline import s3_family as FAM  # noqa: E402

VARIANT = "h_pp_now_c20"
MATCH_FIELDS = ("cell", "v0", "role", "kin", "j", "jr", "window_ok", "stop_between")
STATS = ("prof", "match", "mse")
LABEL = {"gen_on": FAM.FAMILY, "inhouse": FAM.FAMILY + "in"}


def same(a, b):
    return (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def key(r):
    return r["site"], r["track"], r["mapping"], r["prb"]


def relabel(recs, survey):
    """Keep the survey role in force as "ref" and drop the other one."""
    drop = "ref" if survey == "free" else "reff"
    return [dict(r, role="ref") if r["role"] == "reff" else r for r in recs if r["role"] != drop]


def load_urban(res, label, s):
    idx = {}
    for r in json.load(open(os.path.join(res, f"{label}_site_{s:03d}_{VARIANT}.json"))):
        if key(r) in idx:
            raise RuntimeError(f"{label} site {s}: duplicate record {key(r)}")
        idx[key(r)] = r
    return idx


def arms(a):
    return [arm for arm in ("gen_on", "inhouse") if not (a.channel == "inhouse" and arm == "gen_on")]


def build_site(a, s):
    B = json.load(open(os.path.join(a.base, f"site_{s:03d}.json")))
    recs = B["recs"]
    if any(r["site"] != s for r in recs):
        raise RuntimeError(f"site {s}: a carrier record of another site")
    if a.estimator == "base":
        out = relabel(recs, a.survey)
        return B, out, dict(carrier=len(recs), written=len(out))
    roles = (FAM.SURVEY_ROLE[a.survey], "test")
    used, out = set(), []
    for arm in arms(a):
        idx = load_urban(a.res, LABEL[arm], s)
        for r in recs:
            if r["arm"] != arm or r.get("beta", 0.0) != 0.0 or r["role"] not in roles:
                continue
            q = idx.get(key(r))
            if q is None:
                raise RuntimeError(f"{LABEL[arm]} site {s}: no record of the urban estimator for {key(r)}")
            for f in MATCH_FIELDS:
                if not same(r.get(f), q.get(f)):
                    raise RuntimeError(f"{LABEL[arm]} site {s} {key(r)}: field {f} differs ({r.get(f)!r} vs {q.get(f)!r})")
            if (arm, key(r)) in used:
                raise RuntimeError(f"{LABEL[arm]} site {s}: record {key(r)} used twice")
            used.add((arm, key(r)))
            n = {x: v for x, v in r.items() if x not in ("d", "d_d")}
            for x in STATS:
                n[x] = float(q[x])
            n["finite"] = bool(math.isfinite(n["match"]) and math.isfinite(n["prof"]))
            n["s3_estimator"] = VARIANT
            if a.channel == "inhouse":
                n["arm"], n["s3_source_arm"] = "gen_on", "inhouse"
            out.append(n)
        missing = [k for k, q in idx.items() if q["role"] in roles and (arm, k) not in used]
        if missing:
            raise RuntimeError(f"{LABEL[arm]} site {s}: {len(missing)} records of the urban estimator not matched to a carrier record")
    return B, relabel(out, a.survey), dict(carrier=len(recs), substituted=len(used), written=len(out))


def check_written(a, s, B, path):
    """Re-read the written file and compare it with the inputs."""
    W = json.load(open(path))["recs"]
    if a.estimator == "base":
        C = relabel(B["recs"], a.survey)
        if len(W) != len(C) or any(w != c for w, c in zip(W, C)):
            raise RuntimeError(f"site {s}: written records differ from the carrier records")
        return dict(base_equal=len(W))
    n = 0
    survey_role = FAM.SURVEY_ROLE[a.survey]
    for arm in arms(a):
        idx = load_urban(a.res, LABEL[arm], s)
        ws = [w for w in W if w.get("s3_source_arm", w["arm"]) == arm]
        if len(ws) != sum(q["role"] in (survey_role, "test") for q in idx.values()):
            raise RuntimeError(f"site {s} {LABEL[arm]}: {len(ws)} written records, another count from the urban estimator")
        for w in ws:
            q = idx[key(w)]
            if w["role"] != ("ref" if q["role"] == survey_role else q["role"]) or any(not same(w[x], float(q[x])) for x in STATS):
                raise RuntimeError(f"site {s} {LABEL[arm]}: a written value differs from the urban estimator's")
            n += 1
    return dict(urban_equal=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--res", default=None)
    ap.add_argument("--survey", required=True, choices=sorted(FAM.SURVEY_ROLE))
    ap.add_argument("--channel", default="tr", choices=["tr", "inhouse"])
    ap.add_argument("--estimator", required=True, choices=["urban", "base"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.estimator == "base" and a.channel != "tr":
        raise SystemExit("the control run with the base estimator uses the TR 37.885 channel")
    if a.estimator == "urban" and not a.res:
        raise SystemExit("--res is required with --estimator urban")
    os.makedirs(a.out, exist_ok=True)
    total = {}
    for s in range(*FAM.SITES):
        B, recs, counts = build_site(a, s)
        p = os.path.join(a.out, f"site_{s:03d}.json")
        meta = dict(survey=a.survey, channel=a.channel, estimator=a.estimator, variant=VARIANT if a.estimator == "urban" else None,
                    carrier_sha256=hashlib.sha256(open(os.path.join(a.base, f"site_{s:03d}.json"), "rb").read()).hexdigest())
        json.dump(dict({k: v for k, v in B.items() if k != "recs"}, recs=recs, s3=meta), open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
        counts.update(check_written(a, s, B, p))
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    print(json.dumps(dict(out=a.out, survey=a.survey, channel=a.channel, estimator=a.estimator, counts=total, check="pass")))


if __name__ == "__main__":
    main()
