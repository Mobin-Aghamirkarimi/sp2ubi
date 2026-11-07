import os
import sys

import numpy as np
import pytest

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885"), os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885"), "..")]
import interface as itf  # noqa: E402


def test_phase_noise_variance_matches_profile():
    rng = np.random.default_rng(1)
    prof = {1e2: -80.0, 1e5: -80.0}                         # flat L = -80 dBc/Hz
    x = itf.phase_noise(rng, 4000, profile=prof, block=16)
    fs, N = 128e3, 128 * 16
    expected = 2 * 10 ** (-8) * (fs / 2 - fs / N)           # integral of S_phi over the represented band
    assert np.var(x) == pytest.approx(expected, rel=0.05)


def test_iq_image_level():
    t = np.arange(4096) / 4096
    y = np.exp(2j * np.pi * 100 * t)
    z = itf.iq_imbalance(y, -40.0, 0.3)
    S = np.abs(np.fft.fft(z)) ** 2
    assert 10 * np.log10(S[-100] / S[100]) == pytest.approx(-40.0, abs=0.01)


def test_adc_sqnr_and_scale_invariance():
    rng = np.random.default_rng(2)
    y = (rng.standard_normal((50, 128)) + 1j * rng.standard_normal((50, 128))) * 1e-3
    q = itf.adc(y, bits=12)
    sqnr = 10 * np.log10(np.mean(np.abs(y) ** 2) / np.mean(np.abs(q - y) ** 2))
    assert sqnr > 55                                         # 12 bits, Gaussian crest factor and 3 dB headroom
    q8 = itf.adc(y, bits=8)
    assert 10 * np.log10(np.mean(np.abs(y) ** 2) / np.mean(np.abs(q8 - y) ** 2)) < sqnr - 15


def test_sway_doppler_bound():
    f = itf.sway_doppler(np.linspace(0, 2, 2001), 1.0, 0.005, 1.0)
    assert np.max(np.abs(f)) == pytest.approx(2 * np.pi * 0.005 / (299_792_458.0 / 5.9e9), rel=1e-3)   # 0.62 Hz


def test_time_scale_equals_doppler_scaling():
    """interp_q(time_scale = 1 + beta) of a two-path channel equals the channel with both Dopplers scaled."""
    Q, M, beta = 16, 128, -0.10
    tq = (np.arange(Q) + 0.5) / Q * 1e-3 - 0.5e-3
    tm = itf.tau_grid(M)
    c, f = np.array([1.0, 0.3j]), np.array([350.0, -120.0])
    h = np.sum(c[:, None] * np.exp(2j * np.pi * f[:, None] * tq[None, :]), 0)[None, :]
    want = np.sum(c[:, None] * np.exp(2j * np.pi * (1 + beta) * f[:, None] * tm[None, :]), 0)
    got = itf.interp_q(h, M, time_scale=1 + beta)[0]
    assert np.max(np.abs(got - want)) < 1e-4
