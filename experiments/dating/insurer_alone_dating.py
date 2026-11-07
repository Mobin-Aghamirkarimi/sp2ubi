"""Best-guess dating of a single daily risk record by the insurer alone when its posterior is the uniform distribution
over the assignments of the policy-epoch's multiset to the days that are consistent with the classes the scores reveal
(the prior conditioned on the contents).

A class-blind value would be 1/D for a record equal to w_0. The multiset also reveals the per-policy class counts: with
n_wd weekday-mileage records and n_ev event-only records in a 31-day epoch from a Monday (23 weekdays, 8 weekend
days), the n_0 = 31 - n_wd - n_ev records equal to w_0 fill the free days, so one w_0 record lies on a given free
weekday with probability (23 - n_wd)/23 / n_0 and on a given free weekend day with probability (8 - n_ev)/8 / n_0; the
best single-day guess is the larger of the two. A weekday-mileage record is on each weekday with probability 1/23 and
an event-only record on each weekend day with probability 1/8 (exact by symmetry).

The class "weekday mileage needed" of record_classes.py also holds scores that admit an event-only explanation but are
not unique ("ambiguous"); their share is reported. In the main result they count as weekday records; the sensitivity
row counts them as weekend (event-only) records.

Sampling as record_classes.py (So et al. policies, random.Random(seed).sample, daily_records(row, 31, seed + k)), so
the class shares match. Classes from the score alone; uniform within the consistent assignments; no driving-habit
prior, so the values are lower bounds on the best-guess probability.

Run (from the repository root):  python experiments/dating/insurer_alone_dating.py [--policies 2000] [--seed 20261003]
Output: experiments/dating/results/insurer_alone_dating.json (deterministic)
"""
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from sp2ubi.params import DEFAULT as P  # noqa: E402
from sp2ubi.risk_model import load_model  # noqa: E402
from sp2ubi.workload import daily_records, iter_policies  # noqa: E402
from record_classes import DATA, candidates  # noqa: E402

OUT = os.path.join(HERE, "results", "insurer_alone_dating.json")
WEEKDAYS, WEEKEND, DAYS = 23, 8, 31


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "q10": float(np.percentile(a, 10)),
            "q90": float(np.percentile(a, 90)), "max": float(a.max()), "median_as_1_over": float(1 / np.median(a)),
            "share_ge_1_over_8": float((a >= 1 / 8 - 1e-12).mean()), "share_eq_1": float((a >= 0.999).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003)
    a = ap.parse_args()
    w = load_model().w
    rows = random.Random(a.seed).sample(list(iter_policies(DATA)), a.policies)
    shares = {"score_eq_w0": 0, "event_only_unique": 0, "ambiguous_event_only_or_weekday": 0, "weekday_mileage_only": 0}
    p_main, p_sens, w0_on_weekend_guess = [], [], 0
    for k, row in enumerate(rows):
        recs = daily_records(row, DAYS, seed=a.seed + k)
        n0 = nev = namb = nwd = 0
        for x in recs:
            V = sum(wi * xi for wi, xi in zip(w, x)) - w[0]
            if V == 0:
                n0 += 1
                continue
            c = candidates(V, w, P.daily_mile_cap, 30)
            ev = any((V - w[4] * hb) % w[5] == 0 for hb in range(0, V // w[4] + 1))
            if ev and c == 1:
                nev += 1
            elif ev:
                namb += 1
            else:
                nwd += 1
        shares["score_eq_w0"] += n0
        shares["event_only_unique"] += nev
        shares["ambiguous_event_only_or_weekday"] += namb
        shares["weekday_mileage_only"] += nwd
        if n0 == 0:
            continue
        wd, we = (WEEKDAYS - nwd - namb) / WEEKDAYS, (WEEKEND - nev) / WEEKEND      # ambiguous = weekday
        p_main += [max(wd, we) / n0] * n0
        w0_on_weekend_guess += n0 * (we > wd)
        wd2, we2 = (WEEKDAYS - nwd) / WEEKDAYS, max(0, WEEKEND - nev - namb) / WEEKEND   # sensitivity: ambiguous = weekend
        p_sens += [max(wd2, we2) / n0] * n0
    total = sum(shares.values())
    out = {"policies": a.policies, "seed": a.seed, "records": total, "epoch": "31 days from a Monday (23 weekdays, 8 weekend days)",
           "class_shares": {k: v / total for k, v in shares.items()},
           "P_best_guess_score_eq_w0": stats(p_main),
           "share_of_w0_records_best_guessed_on_a_weekend_day": w0_on_weekend_guess / len(p_main),
           "P_best_guess_score_eq_w0_sensitivity_ambiguous_as_weekend": stats(p_sens),
           "P_exact_weekday_mileage_record": 1 / WEEKDAYS, "P_exact_event_only_record": 1 / WEEKEND,
           "class_blind_value_replaced": 1 / DAYS,
           "note": "uniform over the class-consistent assignments of the policy-epoch (per-policy class counts used); "
                   "no driving-habit prior, so lower bounds on the best-guess probability"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
