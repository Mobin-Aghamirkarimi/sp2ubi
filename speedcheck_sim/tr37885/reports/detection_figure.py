"""Cell values of the detection figure of the paper (detection rate of a manipulated speed sensor per vehicle profile
and pass rate), from two stored outputs: results/reports/primary/table_B.csv (study 1) and
results/study3/urban_rows.csv (detection rules with the urban estimator, free-road survey).

Metric: punitive_union_per_year, the share of vehicle-years in which the residual rule or the notice rule flags.
  highway profile   the governing value of table_B.csv (highway-type sites, base estimator)
  urban profile     the worse of the two channel models in urban_rows.csv (column free_worse)
  suburban profile  the worse of that value and of the governing value of table_B.csv (highway sites alone)
A lower detection rate is the worse one. The honest row is the largest false-flag rate over the honest rows of both
files and the largest upper bound.
Output: detection_figure.csv with one line per row, profile and pass rate (rate as a fraction and as the percent text
of the figure: 0 for an exact zero, <0.1 below 0.05 %, one decimal below 10 % and from 99.5 %, an integer otherwise).
Usage: python tr37885/reports/detection_figure.py --check          (fails if the stored file differs)
       python tr37885/reports/detection_figure.py --out <file>"""
import argparse
import csv
import io
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

ER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ER, "..", "results")
TABLE_B = os.path.join(RESULTS, "reports", "primary", "table_B.csv")
URBAN_ROWS = os.path.join(RESULTS, "study3", "urban_rows.csv")
STORED = os.path.join(RESULTS, "study3", "detection_figure.csv")
METRIC, UPPER = "punitive_union_per_year", "punitive_union_ub95"
ROWS = [("scale error -5 %", "persistent|-0.05"), ("scale error -10 %", "persistent|-0.1"),
        ("scale error -30 %", "persistent|-0.3"), ("-15 % on 10 % of the days", "days|-0.15|0.1"),
        ("-15 % on 30 % of the days", "days|-0.15|0.3")]
PROFILES = ("highway", "urban", "suburban")
PASS_RATES = ("5", "15", "40")
HONEST = ("cv0.0", "cv1.0")


def percent_text(x):
    d = Decimal(repr(float(x))) * 100
    if d == 0:
        return "0"
    if d < Decimal("0.05"):
        return "<0.1"
    one = d.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if one < 10 or (Decimal("99.5") <= d < 100 and one < 100):
        return f"{one}"
    return f"{d.quantize(Decimal('1'), rounding=ROUND_HALF_UP)}"


def build():
    with open(TABLE_B) as f:
        B = {(r["section"], r["key"]): r for r in csv.DictReader(f)}
    with open(URBAN_ROWS, newline="") as f:
        U = {(r["section"], r["key"], r["metric"]): r for r in csv.DictReader(f)}
    out = []
    for label, cond in ROWS:
        for prof in PROFILES:
            for nu in PASS_RATES:
                key = f"{cond}|{prof}|{nu}"
                pub = B.get(("power", key), {}).get("governing_" + METRIC)
                pub = float(pub) if pub not in (None, "") else None
                if prof == "highway":
                    tr = ih = None
                    v = pub
                else:
                    u = U[("power", key, METRIC)]
                    tr, ih = float(u["free_tr"]), float(u["free_inhouse"])
                    v = min(tr, ih) if prof == "urban" else min(tr, ih, pub)
                out.append(dict(row=label, key=key, profile=prof, pass_rate=nu, rate=v, percent=percent_text(v),
                                tr37885=tr, our_channel_model=ih, table_B_governing=pub))
    rates, bounds = [], []
    for cv in HONEST:
        for prof in PROFILES:
            for nu in PASS_RATES:
                key = f"{cv}|{prof}|{nu}"
                b = B.get(("honest", key))
                if b:
                    rates.append(float(b["governing_" + METRIC]))
                    bounds.append(float(b["governing_" + UPPER]))
                if prof != "highway":
                    for col in ("free_tr", "free_inhouse"):
                        rates.append(float(U[("honest", key, METRIC)][col]))
                        bounds.append(float(U[("honest", key, UPPER)][col]))
    out.append(dict(row="honest vehicle", key="cv*|*|*", profile="all", pass_rate="all", rate=max(rates),
                    percent=percent_text(max(rates)), tr37885=None, our_channel_model=None, table_B_governing=None))
    out.append(dict(row="honest vehicle, 95 % upper bound", key="cv*|*|*", profile="all", pass_rate="all", rate=max(bounds),
                    percent=f"{100 * max(bounds):.3f}", tr37885=None, our_channel_model=None, table_B_governing=None))
    return out


def render(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--out", default=None)
    a = ap.parse_args()
    text = render(build())
    if a.check:
        ok = os.path.exists(STORED) and open(STORED).read() == text
        print("detection_figure: stored file", "matches" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)
    open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
