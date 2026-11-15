"""Study 3, stage 2 of the U3 sites with the base estimator: stage2_evaluate.evaluate_site with its defaults (arms
gen_off, gen_on and inhouse, both mappings, every PRB, nominal impairments). These records carry the base estimator in
the evaluation and are the carrier records of the decision-level runs (pipeline/s3_decision_records.py).
Usage: python pipeline/s3_stage2.py [--sites 30 40] [--run-dir runs/g1_U3_dev_s3] [--workers 2]"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import s3_family as FAM  # noqa: E402
from pipeline import stage2_evaluate as EV  # noqa: E402


def _job(a):
    EV.seed_base = FAM.apply().seed_base
    return EV.evaluate_site(FAM.FAMILY, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, nargs=2, default=list(FAM.SITES))
    ap.add_argument("--run-dir", default=os.path.join(ER, "runs", f"g1_{FAM.FAMILY}_{FAM.TAG}"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(_job, [(s, a.run_dir, FAM.TAG) for s in range(*a.sites)]):
            print(p, flush=True)


if __name__ == "__main__":
    main()
