import os

import pytest

from sp2ubi.params import DEFAULT
from sp2ubi.protocol import DATASET, sample_policies
from sp2ubi.risk_model import FEATURES, load_model
from sp2ubi.workload import daily_records

needs_dataset = pytest.mark.skipif(not os.path.exists(DATASET), reason="So et al. dataset not available")


def test_model_file_matches_features_and_bounds():
    m = load_model()
    assert len(m.w) == DEFAULT.m == len(FEATURES)
    assert all(0 <= wi < (1 << DEFAULT.w_bits) for wi in m.w)
    assert m.scale == 2 ** 32


def test_largest_epoch_score_is_below_half_the_plaintext_modulus():
    """Largest 31-day score with the fitted weights at the daily caps: about 3.3e11, below p/2."""
    w = load_model().w
    s_max = DEFAULT.score_bounds(w, DEFAULT.max_epoch_days)[1]
    assert s_max == 31 * (w[0] + DEFAULT.daily_mile_cap * sum(w[1:4]) + DEFAULT.daily_event_cap * sum(w[4:6]))
    assert 3.2e11 < s_max < 3.3e11 and s_max < DEFAULT.p // 2
    DEFAULT.check_bounds(w)


@needs_dataset
@pytest.mark.parametrize("days", [4, 7, 30])
def test_daily_records_are_consistent_with_policy_totals(days):
    for k, row in enumerate(sample_policies(200, seed=3)):
        recs = daily_records(row, days, seed=k)
        assert len(recs) == days and all(len(r) == DEFAULT.m and r[0] == 1 for r in recs)
        assert all(0 <= v < (1 << DEFAULT.x_bits) for r in recs for v in r)
        M, D = float(row["Total.miles.driven"]), float(row["Duration"])
        n_we = sum(1 for d in range(days) if d % 7 >= 5)
        miles = sum(r[1] + r[2] + r[3] for r in recs) / 10
        expected = (float(row["Pct.drive.wkday"]) * M / (D * 5 / 7) * (days - n_we)
                    + float(row["Pct.drive.wkend"]) * M / (D * 2 / 7) * n_we)
        assert abs(miles - expected) <= 0.1 * days + 1e-6 * expected   # off-peak and rush each rounded to 0.1 mile
        assert all(r[3] == 0 for d, r in enumerate(recs) if d % 7 < 5)       # weekend miles only on weekends
        assert all(r[1] == r[2] == 0 for d, r in enumerate(recs) if d % 7 >= 5)
