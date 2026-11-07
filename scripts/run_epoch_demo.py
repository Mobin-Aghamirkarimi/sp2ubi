"""Run one honest epoch and every deviation of a malicious TSP; print what the insurer detects.

Usage (from the repository root):  python scripts/run_epoch_demo.py [--policies 3 --days 7]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sp2ubi.fhe import RiskContext  # noqa: E402
from sp2ubi.params import DEFAULT  # noqa: E402
from sp2ubi.protocol import ATTACKS, run_epoch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policies", type=int, default=3)
ap.add_argument("--days", type=int, default=7)
args = ap.parse_args()

print("FHE parameters:", RiskContext(DEFAULT.p, DEFAULT.m).info(), "| MAC prime p =", DEFAULT.p)
for attack in (None,) + ATTACKS:
    r = run_epoch(args.policies, args.days, attack)
    issues = sorted({rep.status for rep in r.batch.reports.values() if rep.status != "ok"})
    if attack == "replay":
        dup = sum(rep.identical_duplicates for rep in r.batch.reports.values())
        print(f"\n[replay] no effect: {dup} identical copy counted once, scores unchanged = "
              f"{r.expected_claims == r.plain_expected_claims}")
    else:
        print(f"\n[{attack or 'honest'}] detected={r.detected}")
    for i in issues[:4] + r.batch.diagnostics[:2]:
        print("   -", i)
    if attack is None:
        for ident, rep in r.batch.reports.items():
            print(f"   policy {ident}: {rep.accepted_records}/{args.days} accepted records, epoch score {sum(rep.scores)}"
                  f" -> expected claims {r.expected_claims[ident]:.5f} (plaintext model: {r.plain_expected_claims[ident]:.5f})")
        print("   daily features from the So et al. dataset:", r.used_dataset)
        print("   dispute opening works:", r.disputes_ok, "| receipts ok:", r.receipts_ok, "| outsider rejected:", r.outsider_rejected)
        print("   sizes (bytes):", r.sizes_B)
