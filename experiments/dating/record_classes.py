"""What the insurer alone learns from the unordered multiset of exact daily risk scores of a policy-epoch, with the
integer model (models/so2021_additive_poisson.json) and the protocol's daily-record generator on So et al. (2021)
policies (31-day epochs starting on a Monday).

Classes are decided from the score alone: (a) score = w_0 (no weekday mileage, no hard events: a non-driving day or an
event-free weekend drive, since weekend miles have weight 0); (b) scores uniquely explainable by hard events only;
(c) other. Weekday driving: number of feature vectors (off-peak, rush in 0.1 mi, hard brakes, hard accelerations)
consistent with the score, counted with mileage <= the OBU cap (16,383) and hard events <= 30 per day, and the share
of unique ones. Per policy-epoch: share of records whose (off-peak, rush, hard brakes, hard accelerations) is
determined; share of policies whose number of driving days is determined.

Run (from the repository root):  python experiments/dating/record_classes.py [--policies 2000]
Output: experiments/dating/results/record_classes.json
"""
import argparse, json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from sp2ubi.params import DEFAULT as P  # noqa: E402
from sp2ubi.risk_model import load_model  # noqa: E402
from sp2ubi.workload import daily_records, iter_policies  # noqa: E402
DATA = os.path.join(ROOT, "data", "telematics_syn-032021.csv")


def candidates(V, w, cap_m, cap_e, limit=50):
    """Number (capped) of (off, rush, hb, ha) with w1 off + w2 rush + w4 hb + w5 ha = V within the caps."""
    n = 0
    for hb in range(0, min(cap_e, V // w[4]) + 1):
        for ha in range(0, min(cap_e, (V - w[4] * hb) // w[5]) + 1):
            R = V - w[4] * hb - w[5] * ha
            a = np.arange(0, min(cap_m, R // w[1]) + 1)
            rest = R - w[1] * a
            n += int(((rest % w[2] == 0) & (rest // w[2] <= cap_m)).sum())
            if n >= limit:
                return limit
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policies", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003); a = ap.parse_args()
    w = load_model().w
    rows = random.Random(a.seed).sample(list(iter_policies(DATA)), a.policies)
    total = 0
    # classes decided from the score alone: score = w_0 | event-only explainable | other
    cls = {"score_eq_w0": 0, "event_only_explainable": 0, "weekday_mileage_needed": 0}
    w0_split = {"non_driving_day": 0, "event_free_weekend_drive": 0}
    weekday_unique, weekday_total, weekday_cands = 0, 0, []
    vec_known, driving_days_known = [], 0
    for k, row in enumerate(rows):
        recs = daily_records(row, 31, seed=a.seed + k)
        known = 0
        ambiguous_w0 = False
        for x in recs:
            total += 1
            S = sum(wi * xi for wi, xi in zip(w, x))
            V = S - w[0]
            if V == 0:
                cls["score_eq_w0"] += 1
                if x[3] > 0:
                    w0_split["event_free_weekend_drive"] += 1; ambiguous_w0 = True
                else:
                    w0_split["non_driving_day"] += 1
                known += 1                     # (off, rush, hb, ha) = 0 is determined; weekend miles never visible
                continue
            c = candidates(V, w, P.daily_mile_cap, 30)
            ev_only = any((V - w[4] * hb) % w[5] == 0 for hb in range(0, V // w[4] + 1))
            if ev_only and c == 1:
                cls["event_only_explainable"] += 1
            else:
                cls["weekday_mileage_needed"] += 1
            known += (c == 1)
            if x[1] + x[2] > 0:
                weekday_total += 1; weekday_cands.append(c); weekday_unique += c == 1
        vec_known.append(known / len(recs))
        driving_days_known += not ambiguous_w0 and all(x[1] + x[2] + x[4] + x[5] > 0 or x[3] == 0 for x in recs)
    q = lambda v, p: float(np.percentile(v, p))
    out = {"policies": a.policies, "records": total, "weights": w,
           "classes_from_score_alone_share": {k: v / total for k, v in cls.items()},
           "score_eq_w0_split": w0_split,
           "weekday_driving_records": weekday_total,
           "weekday_vector_unique_share": weekday_unique / weekday_total,
           "weekday_candidates_median": q(weekday_cands, 50), "weekday_candidates_q90": q(weekday_cands, 90),
           "per_policy_share_of_records_with_unique_weekday_feature_vector": {"median": q(vec_known, 50),
                                                                                "q10": q(vec_known, 10)},
           "policies_whose_driving_day_count_is_determined": driving_days_known / a.policies,
           "bounds": "mileage <= 16,383 (OBU cap), hard events <= 30 per day for the candidate count",
           "note": "score = w_0 means no weekday mileage and no hard events: a non-driving day or an event-free weekend "
                   "drive (weekend miles have weight 0)"}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "record_classes.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
