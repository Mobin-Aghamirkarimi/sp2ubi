import os
import sys

import numpy as np
import pytest

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885"), os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885"), "..")]
import scatterer_mobility as sm  # noqa: E402

LAM = 299_792_458.0 / 5.9e9


def test_bounds_and_moments_uniform_alpha():
    rng = np.random.default_rng(1)
    nu = sm.draw_extra_doppler(rng, 12, 20, 16.7, LAM)
    assert nu.shape == (12, 20)
    big = sm.draw_extra_doppler(np.random.default_rng(2), 2000, 20, 16.7, LAM).ravel()
    vmax = 2 * 16.7 / LAM
    assert np.all(np.abs(big) <= vmax + 1e-9)
    assert abs(big.mean()) < 0.01 * vmax
    # 2*alpha*D/lam, alpha~U(0,1), D~U(-v,v): E[x^2] = 4 E[a^2] E[D^2]/lam^2 = 4 (1/3)(v^2/3)/lam^2
    assert np.var(big) == pytest.approx(4 / 9 * 16.7 ** 2 / LAM ** 2, rel=0.02)


def test_cluster_mode_is_coherent():
    nu = sm.draw_extra_doppler(np.random.default_rng(3), 19, 20, 38.9, LAM, granularity="cluster")
    assert np.all(nu == nu[:, :1])


def test_bernoulli_alpha_share():
    nu = sm.draw_extra_doppler(np.random.default_rng(4), 5000, 1, 16.7, LAM, alpha="bernoulli", p_bernoulli=0.2)
    assert np.mean(nu != 0) == pytest.approx(0.2, abs=0.02)


def test_apply_shifts_frequency_and_keeps_los():
    t = np.arange(4096) / 4096.0                     # 1 s at 4096 Hz
    coeff = np.ones((1, 3, t.size), complex)          # paths: LoS, two delayed
    out = sm.apply(coeff, t, np.array([0.0, 100.0, -250.0]), los_index=0)
    assert np.allclose(out[0, 0], 1.0)
    for k, f in ((1, 100.0), (2, -250.0)):
        spec = np.fft.fft(out[0, k])
        assert np.fft.fftfreq(t.size, 1 / 4096.0)[np.argmax(np.abs(spec))] == pytest.approx(f)


def test_bad_arguments():
    with pytest.raises(ValueError):
        sm.draw_extra_doppler(np.random.default_rng(0), 1, 1, 1.0, LAM, granularity="beam")
    with pytest.raises(ValueError):
        sm.apply(np.ones((1, 2, 3)), np.arange(3), np.array([1.0, 2.0, 3.0]))


def _cdf_abs(x, vmax):
    """Exact CDF of |2 a D / lam| for a ~ U(0,1), D ~ U(-v, v): with u = x / vmax in (0, 1], F = u (1 - ln u)."""
    u = np.clip(np.asarray(x, float) / vmax, 1e-300, 1.0)
    return u * (1.0 - np.log(u))


def test_exact_distribution_ks_and_symmetry():
    """KS against the exact distribution at n = 1e5 (p >= 0.01 <=> D_n < 1.628/sqrt(n))."""
    n, v = 100_000, 38.9
    vmax = 2 * v / LAM
    nu = sm.draw_extra_doppler(np.random.default_rng(7), n, 1, v, LAM).ravel()
    a = np.sort(np.abs(nu))
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    f = _cdf_abs(a, vmax)
    d_n = max(np.max(emp_hi - f), np.max(f - emp_lo))
    assert d_n < 1.628 / np.sqrt(n)
    assert abs(np.mean(nu > 0) - 0.5) < 0.005           # symmetric
    assert np.mean(np.abs(nu)) == pytest.approx(vmax / 4, rel=0.01)   # E|aD| = E[a] E|D| = 1/2 * v/2
