"""Study 3, detection rules with the urban estimator: read-out of the five decision-level runs of run_study3.sh.

  run      estimator  channel model      survey
  trfree   urban      TR 37.885          free road
  trtraf   urban      TR 37.885          traffic
  ctl      base       TR 37.885          free road      (control)
  infree   urban      our channel model  free road
  intraf   urban      our channel model  traffic

Input per run <runs>/<run>/res/: governing_<tag>.json (stage 3), adapter/tables_<tag>.json and
adapter/e2_extracted_<tag>.json (decision adapter), tag = dev_s3_<run>; for trfree and trtraf also reports/table_B.csv
(reports/make_reports.py). --pub: the stored study-1 results (../results), for the published highway-profile and
suburban-profile rows.
Output:
  urban_rows.csv        one line per row and metric of the urban and suburban profiles: the values under TR 37.885 and
                        under our channel model and the worse of the two (direction of the decision adapter), for the
                        free-road survey and for the survey in traffic; the 'governing' column of each TR 37.885 run's
                        table_B.csv; for suburban rows the published study-1 value.
  detection_runs.json   run-level fields copied from the outputs of the five runs.
Usage: python pipeline/s3_decision_readout.py --runs runs/s3_decision --pub ../results --out <dir>"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import s3_family as FAM  # noqa: E402

RUNS = (("trfree", "urban", "TR 37.885", "free road"), ("trtraf", "urban", "TR 37.885", "traffic"),
        ("ctl", "base", "TR 37.885", "free road"), ("infree", "urban", "our channel model", "free road"),
        ("intraf", "urban", "our channel model", "traffic"))
SURVEYS = (("free", "trfree", "infree"), ("traffic", "trtraf", "intraf"))
SECTIONS = ("honest", "margin", "badsites", "power", "sens_tyre")
URBAN_CELLS = ("P1-10", "P1-15")
COLUMNS = ["section", "key", "metric", "direction", "free_tr", "free_inhouse", "free_worse", "free_tableB_governing",
           "traffic_tr", "traffic_inhouse", "traffic_worse", "traffic_tableB_governing", "published_suburban"]
DESCRIPTION = (
    "Detection rules with the urban estimator on the ten newly simulated urban LOS sites of study 3: "
    "run-level fields of the five decision-level runs, copied unchanged from their stored outputs. "
    "qualified_combinations counts the combinations of site and speed class (ten sites, 10 and 15 m/s) "
    "that meet the stricter site condition of the detection rules; n_eval_urban is the number of gated "
    "test crossings per site and speed class; q_*_profile is the share of a profile's crossings at "
    "qualified combinations; highway_rows_equal_published states that every highway-profile row of the "
    "run equals that of the published study-1 run in every metric the published file holds.")


def tag(run):
    return f"{FAM.TAG}_{run}"


def worse(direction, a, b):
    if a is None or b is None or direction is None:
        return None
    return max(a, b) if direction == "high" else min(a, b)


def urban_rows(E2, TB, published, direction):
    rows = []
    for sec, section_rows in E2["trfree"]["rows"].items():
        for key, row in section_rows.items():
            profile = key.split("|")
            if "urban" not in profile and "suburban" not in profile:
                continue
            for m, v in row.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                d = direction(sec, key, m)
                rec = dict(section=sec, key=key, metric=m, direction=d)
                for survey, tr, ih in SURVEYS:
                    x = E2[tr]["rows"].get(sec, {}).get(key, {}).get(m)
                    y = E2[ih]["rows"].get(sec, {}).get(key, {}).get(m)
                    rec[f"{survey}_tr"], rec[f"{survey}_inhouse"], rec[f"{survey}_worse"] = x, y, worse(d, x, y)
                    tb = TB[tr].get((sec, key), {}).get(f"governing_{m}")
                    rec[f"{survey}_tableB_governing"] = float(tb) if tb not in (None, "") else None
                if "suburban" in profile:
                    rec["published_suburban"] = published.get(sec, {}).get(key, {}).get(m)
                rows.append(rec)
    return rows


def run_summary(estimator, channel, survey, e2, tables, governing, published):
    le = e2["lambda_eff"].get("urban")
    share = le["class_qualified_share"] if le else None
    n_sites = FAM.SITES[1] - FAM.SITES[0]
    highway_equal = all(e2["rows"][s][k][m] == v for s in SECTIONS for k in published[s] if k.split("|")[-2] == "highway"
                        for m, v in published[s][k].items())
    return dict(
        estimator=estimator, channel_model=channel, survey=survey,
        status_urban=tables["stratum_status"]["urban"], status_highway=tables["stratum_status"]["highway"],
        qualified_combinations=round(sum(share.values()) * n_sites) if share else 0,
        combinations=n_sites * (len(share) if share else len(URBAN_CELLS)), qualified_share_per_class=share,
        n_eval_urban=e2["n_eval"].get("urban"),
        excess_between_site_sd={c: governing["c2_tau"][c] for c in URBAN_CELLS},
        q_urban_profile=None if le is None else le["profiles"]["urban"]["q"],
        q_suburban_profile=None if le is None else le["profiles"]["suburban"]["q"],
        highway_rows_equal_published=highway_equal, input_sha256_urban=e2["input_sha256"].get("urban"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pipeline.decision_adapter import direction      # imported here: the adapter reads its inputs on import
    os.makedirs(a.out, exist_ok=True)
    published = json.load(open(os.path.join(a.pub, "adapter", "e2_extracted_primary.json")))["rows"]
    E2, summary = {}, {}
    for run, estimator, channel, survey in RUNS:
        d = os.path.join(a.runs, run, "res")
        E2[run] = json.load(open(os.path.join(d, "adapter", f"e2_extracted_{tag(run)}.json")))
        summary[run] = run_summary(estimator, channel, survey, E2[run],
                                   json.load(open(os.path.join(d, "adapter", f"tables_{tag(run)}.json"))),
                                   json.load(open(os.path.join(d, f"governing_{tag(run)}.json"))), published)
    TB = {}
    for run in ("trfree", "trtraf"):
        with open(os.path.join(a.runs, run, "res", "reports", "table_B.csv")) as f:
            TB[run] = {(r["section"], r["key"]): r for r in csv.DictReader(f)}
    rows = urban_rows(E2, TB, published, direction)
    with open(os.path.join(a.out, "urban_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLUMNS})
    with open(os.path.join(a.out, "detection_runs.json"), "w") as f:
        f.write(json.dumps(dict(description=DESCRIPTION, runs=summary), indent=1, sort_keys=True) + "\n")
    print(f"{len(rows)} rows; runs: " + ", ".join(f"{r} {s['qualified_combinations']}/{s['combinations']}" for r, s in summary.items()))


if __name__ == "__main__":
    main()
