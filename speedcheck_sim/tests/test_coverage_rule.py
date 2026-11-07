"""K7 calibration rule of per-replicate c_eff = max(c(n), c);
conditional size with >= 20 runnable replicates, unconditional otherwise; 'not calibrated' instead of an error."""
import os
import sys

import numpy as np

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885")
sys.path[:0] = [ER, os.path.join(ER, "..")]
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
from pipeline import coverage_check as CV  # noqa: E402
from pipeline import stage3_criteria as K  # noqa: E402


def draw(R, spread, n, rng):
    return (R, R * np.exp(rng.normal(0, spread, 400)), n)


def test_conditional_vs_unconditional_size():
    rng = np.random.default_rng(0)
    passing = [draw(1.0, 0.02, 10, rng) for _ in range(25)]          # tight draws below the margin: 'not contradicted'
    sz, runs, cond = CV.size_at(1.0, passing)
    assert runs == 25 and cond and sz == 1.0
    few = [draw(1.0, 0.02, 10, rng) for _ in range(3)] + [(np.nan, np.array([]), 2)] * 97
    sz, runs, cond = CV.size_at(1.0, few)
    assert runs == 3 and not cond and np.isclose(sz, 3 / 100)       # unconditional: over all 100 replicates


def test_per_replicate_widening_uses_c_of_n():
    rng = np.random.default_rng(1)
    d = [draw(1.2, 0.12, 4, rng) for _ in range(30)]                 # n = 4: c(4) = 1.875 dominates c = 1
    s_small_c, _, _ = CV.size_at(1.0, d)
    s_c_n, _, _ = CV.size_at(K.c_of_n(4), d)
    assert np.isclose(s_small_c, s_c_n)                              # max(c(n), c) applied per replicate


def test_not_calibrated_caps_verdict(monkeypatch):
    monkeypatch.setitem(K.K7_STATE, "calibrated", False)
    rows = [dict(cell=c, v0=v, mapping=mp, prb=prb, p=0.0001, R=1.0, not_run=False, n_sites=10, c=1.2)
            for c, v in K.HOLM_FAMILY for mp, prb in K.COMBOS]
    g = K.optimism_verdicts(rows)
    assert all(x["verdict"] == "undetermined" and "K7 not calibrated" in x["reason"] for x in g.values())


def test_sigma_widening(monkeypatch):
    monkeypatch.setitem(K.K7_STATE, "widen_sigma_ci", True)
    b = [0.04, 0.05, 0.06]
    w = K.widen_sigma_draws(0.05, b, 10)
    assert np.isclose(np.log(w[0]) - np.log(0.05), K.c_of_n(10) * (np.log(0.04) - np.log(0.05)))
    monkeypatch.setitem(K.K7_STATE, "widen_sigma_ci", False)
    assert K.widen_sigma_draws(0.05, b, 10) == b
