import os

import pytest

from sp2ubi.protocol import ATTACKS, DATASET, run_epoch


def test_honest_epoch_is_accepted():
    res = run_epoch(n_policies=3, days=5)
    assert not res.detected and res.batch.rejected == 0
    assert all(r.status == "ok" and r.accepted_records == 5 for r in res.batch.reports.values())
    assert res.confirmation_ok and res.receipts_ok and res.disputes_ok
    assert res.traced_ok and res.outsider_rejected
    assert res.sizes_B["group_signature"] == 336 and 2 * 10 ** 5 < res.sizes_B["ct_payload"] < 3 * 10 ** 5   # N = 8192


@pytest.mark.parametrize("attack", [a for a in ATTACKS if a != "replay"])
def test_deviating_tsp_is_detected(attack):
    res = run_epoch(n_policies=3, days=4, attack=attack)
    assert res.detected, attack
    assert set(res.batch.verdicts) <= {"accepted", "rejected"}       # one outward outcome for every rejection


def test_replayed_output_has_no_effect():
    honest = run_epoch(n_policies=3, days=4, seed=11)
    res = run_epoch(n_policies=3, days=4, attack="replay", seed=11)
    assert not res.detected
    assert sum(r.identical_duplicates for r in res.batch.reports.values()) == 1
    assert res.expected_claims == res.plain_expected_claims
    assert sorted(res.expected_claims.values()) == sorted(honest.expected_claims.values())


@pytest.mark.skipif(not os.path.exists(DATASET), reason="So et al. dataset not available")
def test_encrypted_epoch_score_equals_plaintext_model_exactly():
    """The insurer's decrypted daily scores reproduce the plaintext model on the same daily records (daily default)."""
    res = run_epoch(n_policies=3, days=7)
    assert res.used_dataset
    assert not res.detected
    for ident, claims in res.expected_claims.items():
        assert claims == res.plain_expected_claims[ident] and claims > 0


def test_partial_epoch_contract_start_or_end():
    res = run_epoch(n_policies=3, days=6, insured_days=[6, 3, 1])
    assert not res.detected
    assert res.expected_claims == res.plain_expected_claims


def test_monthly_configuration():
    """Aggregation interval of 31 days (one record per policy-epoch): the monthly configuration."""
    from sp2ubi.params import Params
    res = run_epoch(n_policies=3, days=31, params=Params(interval_days=31))
    assert not res.detected and all(r.accepted_records == 1 for r in res.batch.reports.values())
    assert res.expected_claims == res.plain_expected_claims


def test_uu_outage_until_the_deadline_leaves_records_missing():
    """With the cellular link down from day 2 until after the settlement deadline, the
    records whose drawn upload time falls inside the outage never reach the TSP; their policies are 'missing',
    nothing is rejected, and records sent before the outage are accepted."""
    from sp2ubi.params import DEFAULT
    deadline = DEFAULT.settlement_deadline_s(3)
    res = run_epoch(n_policies=2, days=3, seed=5, outages=[(2 * 86400, deadline + 1)])
    reps = list(res.batch.reports.values())
    assert res.batch.rejected == 0 and not res.receipts_ok
    assert all(r.status == "missing" and r.accepted_records < 3 for r in reps)
    assert res.disputes_ok and res.traced_ok and res.outsider_rejected


def test_epoch_demo_script_runs():
    """The demo script runs to the end (honest epoch and every attack), so it cannot drift from the epoch result type."""
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "scripts/run_epoch_demo.py", "--policies", "1", "--days", "2"],
                         cwd=root, check=True, capture_output=True, text=True).stdout
    assert "[honest] detected=False" in out and "dispute opening works:" in out
