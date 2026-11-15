"""Study 3: mean error after calibration of one estimator per traffic lane (lane 0 is the lane nearer the heads; the
lane is read from the simulated lateral position and serves reporting only). Qualified sites, gated traffic crossings,
errors clipped to the accepted window, pooled over the six mapping x PRB variants; site bootstrap for the interval.
Usage: python pipeline/s3_per_lane.py --res <dir> --series <dir> --family U3 --variant h_pp_now_c20 [--survey-role reff]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pipeline import s3_rules as S  # noqa: E402

LANE_WIDTH_M = 3.5
BOOT = 2000
SEED = (20261112, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--survey-role", default="ref", choices=["ref", "reff"])
    a = ap.parse_args()
    lane, survey_lanes = {}, {}
    for f in sorted(glob.glob(os.path.join(a.series, f"{a.family}_site_???.json"))):
        J = json.load(open(f))
        Z = np.load(f[:-5] + ".npz")
        for r in J["recs"]:
            ln = int(round(-float(np.mean(Z[f"r{r['n']}_y"])) / LANE_WIDTH_M))
            lane[(r["site"], r["track"])] = ln
            if r["role"] == "ref" and r["mapping"] == "lb" and r["prb"] == 10:
                survey_lanes[ln] = survey_lanes.get(ln, 0) + 1
    recs = S.load(a.res, a.series, a.family, a.variant, a.survey_role)
    print(f"{a.family} {a.variant}: survey crossings per lane (all sites, both speeds) {survey_lanes}")
    for v0 in (10.0, 15.0):
        per_site = {}
        for mp, prb in S.VARIANTS:
            by = {}
            for r in recs:
                if r["v0"] == v0 and r["mapping"] == mp and r["prb"] == prb:
                    by.setdefault(r["site"], []).append(r)
            for s, rs in by.items():
                t = S.site_cell(rs)
                if t is None or not t["qualified"]:
                    continue
                tst = [r for r in rs if r["role"] == "test" and S.included(r)]
                for r, e, k in zip(tst, t["e"], t["keep"]):
                    if k:
                        per_site.setdefault(s, {}).setdefault(lane[(s, r["track"])], []).append(float(np.clip(e, -S.WIN, S.WIN)))
        sites = sorted(per_site)
        rng = np.random.default_rng([SEED[0], int(v0), SEED[1]])
        for ln in (0, 1):
            pool = lambda ss: np.concatenate([per_site[s].get(ln, []) for s in ss]) if ss else np.array([])  # noqa: E731
            x = pool(sites)
            b = [np.mean(pool(list(rng.choice(sites, len(sites))))) for _ in range(BOOT)]
            print(f"  v{v0:g} lane {ln}: n = {x.size:5d} (crossing x variant), mean {100 * x.mean():+.2f} % "
                  f"[{100 * np.nanquantile(b, .025):+.2f}, {100 * np.nanquantile(b, .975):+.2f}], sd {100 * x.std():.2f} %")


if __name__ == "__main__":
    main()
