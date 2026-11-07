"""interface.band_weights closed form (Dirichlet kernel) equals the explicit subcarrier mean within 1e-12."""
import os
import sys

import numpy as np

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885")
sys.path[:0] = [ER, os.path.join(ER, "..")]
import interface as itf  # noqa: E402


def explicit(tau, prb):
    n = 12 * prb
    fk = (np.arange(n) - (n - 1) / 2) * itf.SCS
    return np.mean(np.exp(-2j * np.pi * np.asarray(tau)[..., None] * fk), axis=-1)


def test_closed_form_matches_mean():
    rng = np.random.default_rng(0)
    tau = np.concatenate([rng.uniform(-5e-6, 5e-6, 4000), [0.0, 1e-15, -1e-13], rng.uniform(-2e-7, 2e-7, 1000),
                          np.array([1, -1, 2, 3]) / itf.SCS])                    # the singular points k / SCS
    for prb in (10, 20, 50):
        assert np.max(np.abs(itf.band_weights(tau, prb) - explicit(tau, prb))) <= 1e-12
    assert itf.band_weights(np.zeros((3, 4)), 10).shape == (3, 4)
