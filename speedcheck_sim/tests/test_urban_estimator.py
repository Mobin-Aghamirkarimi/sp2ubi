"""Urban estimator (fdoa/urban.py) and the study-3 site rules on synthetic crossings without a channel: the pulse-pair
frequency, the fit with speed nodes on a crossing that slows down, the statistics and their core rule, the reduction to
the base fit, the seeds of the family U3 and the survey rule of the decision records."""
import os
import sys

import numpy as np

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885")
sys.path[:0] = [ER, os.path.join(ER, "..")]
from fdoa import urban as U  # noqa: E402
from fdoa.estimator import _g, fit_pass_offset  # noqa: E402
from fdoa.geometry import LAMBDA, Site  # noqa: E402
from pipeline import s3_decision_records as REC  # noqa: E402
from pipeline import s3_family as FAM  # noqa: E402
from pipeline import s3_rules as S  # noqa: E402

SITE = Site(D=100.0, pole_offset=8.0, dh=4.4)
RHO, FO = float(np.hypot(SITE.pole_offset, SITE.dh)), 37.0


def crossing(speed, t_end, noise_hz=0.0, seed=0):
    """Beacons at 10 Hz from x = -20 m; speed(t) is the true speed profile. Returns (t, FDOA, v, x)."""
    t = np.arange(0.0, t_end, 0.1)
    v = speed(t)
    x = -20.0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    fd = v / LAMBDA * _g(x, SITE.D, RHO) + FO + noise_hz * np.random.default_rng(seed).standard_normal(t.size)
    return t, fd, v, x


def test_pulse_pair_frequency_of_a_tone():
    M, packet_s = 64, 1e-3
    n = (np.arange(M) + 0.5) / M * packet_s
    z = np.exp(2j * np.pi * np.array([[-310.0], [45.0], [270.0]]) * n)
    f, coherence = U.pulse_pair(z, 1, packet_s)
    assert np.allclose(f, [-310.0, 45.0, 270.0], atol=1e-6) and np.allclose(coherence, 1.0)


def test_without_options_the_fit_is_the_base_fit():
    t, fd, _, _ = crossing(lambda t: 12.0 + 0.3 * t, 9.0, noise_hz=2.0)
    F = U.fit(t, fd, SITE)
    q, _, resid, ok = fit_pass_offset(t, fd, SITE)
    assert ok and F["ok"]
    assert np.allclose(F["p"], q[:3], rtol=1e-6) and abs(F["rho"] - q[3]) < 1e-5 and abs(F["fo"] - q[4]) < 1e-4
    assert np.allclose(F["resid"], resid, atol=1e-4)


def test_speed_nodes_follow_a_crossing_that_slows_and_recovers():
    speed = lambda t: 12.0 - 8.0 * np.exp(-0.5 * ((t - 6.0) / 1.5) ** 2)  # noqa: E731
    t, fd, v, x = crossing(speed, 16.0)
    nodes = U.fit(t, fd, SITE, fo=FO, rho=RHO, node_s=U.NODE_S)
    quad = U.fit(t, fd, SITE, fo=FO, rho=RHO)
    core = (x >= U.CORE[0] * SITE.D) & (x <= U.CORE[1] * SITE.D)
    assert np.sqrt(np.mean(nodes["resid"] ** 2)) < 0.2 * np.sqrt(np.mean(quad["resid"] ** 2))
    assert np.max(np.abs(nodes["v"][core] - v[core])) < 1.0
    for b in (0.0, -0.10):                               # a reported speed scaled by 1 + b gives the statistic b
        prof, match, _ = U.statistics(nodes, fd, (1 + b) * v, core=U.CORE, D=SITE.D)
        assert abs(prof - b) < 0.01 and abs(match - b) < 0.01


def test_core_rule_and_weights():
    t, fd, v, _ = crossing(lambda t: 30.0 + 0.0 * t, 5.0)
    F = U.fit(t, fd, SITE, fo=FO, rho=RHO, node_s=U.NODE_S)
    assert int(((F["x"] >= 45.0) & (F["x"] <= 55.0)).sum()) < U.MIN_CORE_BEACONS
    prof, match, mse = U.statistics(F, fd, v, core=(0.45, 0.55), D=SITE.D)
    assert np.isnan(prof) and np.isnan(match) and np.isfinite(mse)
    p = -60.0 + np.zeros(60)
    w = U.envelope_weights(p, p - 3.0)                   # a constant level difference between the heads: equal weights
    assert np.allclose(w, 1.0)
    faded = p.copy()
    faded[30] -= 20.0                                    # a 20 dB fade at one head: 1 / (1 + 10 + 0.1) against 1 / 3
    w = U.envelope_weights(faded, p)
    assert abs(w[30] / w[0] - 3.0 / 11.1) < 1e-9 and np.allclose(np.delete(w, 30), w[0]) and abs(np.mean(w) - 1.0) < 1e-12


def test_family_seeds_and_roles():
    assert FAM.seeds() == dict(G1_site_base=1598222420, G1_pass_base=3249020294, sumo=531424638,
                               impairments=2237204963, bootstrap=201875557)
    assert [(r, n) for r, n, _ in FAM.ROLES] == [("ref", 30), ("reff", 30), ("test", 60)]


def test_site_rules_and_survey_rule():
    rng = np.random.default_rng(1)

    def rec(role, b, mse=1.0):
        return dict(role=role, kin="sumo", window_ok=True, prof=b, match=b, mse=mse, stop_between=False,
                    site=0, v0=10.0, mapping="e1", prb=10)

    survey = [rec("ref", 0.02 + 0.03 * e) for e in rng.standard_normal(30)]
    test = [rec("test", 0.02 + 0.03 * e) for e in rng.standard_normal(60)] + [rec("test", 0.5, mse=4.0)]
    t = S.site_cell(survey + test)
    assert t["qualified"] and abs(t["b_ref"] - 0.02) < 0.02 and t["keep"].sum() == 60 and not t["keep"][-1]
    assert S.site_cell(survey[:S.MIN_REF - 1] + test) is None
    wide = [rec("ref", 0.1 * e) for e in rng.standard_normal(30)]
    assert not S.site_cell(wide + test)["qualified"]
    c = S.cell_summary({0: t, 1: S.site_cell(wide + test)})
    assert c["n_sites"] == 2 and c["k"] == 1 and c["sigma_all"] > c["sigma_qg"]
    recs = [dict(role=r) for r in ("ref", "reff", "test")]
    assert [r["role"] for r in REC.relabel(recs, "free")] == ["ref", "test"]
    assert REC.relabel(recs, "free")[0] is not recs[0] and REC.relabel(recs, "traffic") == [recs[0], recs[2]]
