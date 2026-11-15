"""Study 3, stage 1 of one U3 site through stage1_channels.site_record (QuaDRiGa and SUMO, as every other family).
Needs Octave and the patched QuaDRiGa tree (QUADRIGA_PATCHED_DIR) and SPEEDCHECK_CONSTANTS = config/study2_constants.json
(run_study3.sh sets it).
Usage: python pipeline/s3_stage1.py --site 30 [--out runs/g1_U3_dev_s3]
       python pipeline/s3_stage1.py --site 90 --n-scale 0.034 --out <dir>     (smoke test on a site outside 30-44)"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import s3_family as FAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(ER, "runs", f"g1_{FAM.FAMILY}_{FAM.TAG}"))
    ap.add_argument("--n-scale", type=float, default=None, help="smoke test only (sites outside 30-44)")
    a = ap.parse_args()
    G = FAM.apply()
    smoke = a.n_scale is not None
    if not smoke and not (FAM.SITES[0] <= a.site < FAM.SITES[1]):
        raise SystemExit(f"U3 sites are {FAM.SITES[0]}-{FAM.SITES[1] - 1}")
    if smoke and FAM.SITES[0] <= a.site < FAM.PLACE_SITES[1]:
        raise SystemExit("a smoke test must not draw a site of the evaluation")
    os.makedirs(a.out, exist_ok=True)
    path, seconds = G.site_record(FAM.FAMILY, a.site, G.C["run"]["Q"], a.out, FAM.TAG + "_smoke" if smoke else FAM.TAG,
                                  a.n_scale if smoke else G.C["run"]["n_scale"])
    G.write_manifest(a.out)
    print(path, "skipped (exists)" if seconds is None else "done", flush=True)


if __name__ == "__main__":
    main()
