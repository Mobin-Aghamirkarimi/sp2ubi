"""Daily records for protocol simulations, derived from So, Boucher & Valdez (2021) policies.

The dataset gives per-policy totals over the coverage period, not daily logs, so daily records are simulated under
explicit assumptions:
  * an epoch of `days` consecutive days, starting on a Monday; days 5 and 6 of each week are the weekend;
  * the policy drives on a day with probability Avgdays.week / 7 (at least one driving day per category with miles);
  * miles per calendar weekday = Pct.drive.wkday * M / (Duration * 5/7), per calendar weekend day =
    Pct.drive.wkend * M / (Duration * 2/7); the epoch total of each category (that rate times the number of days of
    that type in the epoch) is split over its driving days with Dirichlet(1) shares; on weekday driving days the
    rush/off-peak split equals the policy's ratio;
  * hard brakes / accelerations on a day are Poisson with mean (count per mile) * miles of that day.
Distances are rounded to 0.1 mile and every value is capped at the OBU's daily caps (Params.daily_mile_cap = 2^14 - 1
for mileage, Params.daily_event_cap = 4095 for event counts).
"""
import csv
import math
import random
from typing import Dict, Iterator, List

from .params import DEFAULT

CAP = DEFAULT.daily_mile_cap
EVENT_CAP = DEFAULT.daily_event_cap


def iter_policies(path: str) -> Iterator[Dict[str, str]]:
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:                                            # normal approximation for large means
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    L, k, prod = math.exp(-lam), 0, rng.random()
    while prod > L:
        k += 1
        prod *= rng.random()
    return k


def _split(rng: random.Random, total: float, n: int) -> List[float]:
    g = [rng.expovariate(1.0) for _ in range(n)]          # Dirichlet(1, ..., 1) shares
    s = sum(g)
    return [total * x / s for x in g]


def daily_records(row: Dict[str, str], days: int = 30, seed: int = 0) -> List[List[int]]:
    rng = random.Random(seed)
    M, D = float(row["Total.miles.driven"]), float(row["Duration"])
    wkday, wkend = float(row["Pct.drive.wkday"]), float(row["Pct.drive.wkend"])
    rush_share = min(float(row["Pct.drive.rush am"]) + float(row["Pct.drive.rush pm"]), wkday)
    is_weekend = [(d % 7) >= 5 for d in range(days)]
    n_we = sum(is_weekend)
    weekday_total = wkday * M / (D * 5 / 7) * (days - n_we)
    weekend_total = wkend * M / (D * 2 / 7) * n_we
    rush_ratio = rush_share / wkday if wkday > 0 else 0.0
    brake_rate, accel_rate = float(row["Brake.08miles"]) / 1000.0, float(row["Accel.08miles"]) / 1000.0
    p_drive = min(1.0, float(row["Avgdays.week"]) / 7.0)

    drives = [rng.random() < p_drive for _ in range(days)]
    miles = [0.0] * days
    for weekend, total in ((False, weekday_total), (True, weekend_total)):
        cand = [d for d in range(days) if is_weekend[d] == weekend]
        active = [d for d in cand if drives[d]] or ([rng.choice(cand)] if total > 0 else [])
        for d, m in zip(active, _split(rng, total, len(active)) if active else []):
            miles[d] = m

    out = []
    for d in range(days):
        m = miles[d]
        rush = 0.0 if is_weekend[d] else m * rush_ratio
        offpeak = 0.0 if is_weekend[d] else m - rush
        weekend = m if is_weekend[d] else 0.0
        rec = [1, round(offpeak * 10), round(rush * 10), round(weekend * 10),
               _poisson(rng, brake_rate * m), _poisson(rng, accel_rate * m)]
        caps = (1, CAP, CAP, CAP, EVENT_CAP, EVENT_CAP)
        out.append([min(v, c) for v, c in zip(rec, caps)])
    return out
