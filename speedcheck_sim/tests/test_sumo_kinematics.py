import os
import sys

import numpy as np
import pytest

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885")
sys.path[:0] = [ER, os.path.join(ER, "..")]
import sumo_kinematics as sk  # noqa: E402
from fdoa.geometry import Site  # noqa: E402


@pytest.mark.skipif(not os.path.exists(os.path.join(sk.BIN, "sumo")), reason="SUMO not installed")
@pytest.mark.parametrize("sc,v0", [(sk.URBAN, 15.0), (sk.HIGHWAY, 30.0)])
def test_site_passes_window_and_consistency(sc, v0):
    rng = np.random.default_rng(11)
    site = Site(D=100.0, pole_offset=8.0, dh=3.4)
    tr, info = sk.site_passes(rng, site, v0, 6, sc)
    assert info["n"] == 6 and info["acc_violations"] == 0
    for T in tr:
        assert T.x[0] >= -35.0 - 1e-9 and T.x[0] < -34.0 and T.x[-1] <= 135.0 + 1e-9 and T.x[-1] > 134.0
        xi = T.x[0] + np.concatenate([[0], np.cumsum(0.5 * (T.vx[1:] + T.vx[:-1]) * np.diff(T.t))])
        assert np.abs(T.x - xi).max() < 1e-6
        assert np.abs(np.diff(T.vx) / np.diff(T.t)).max() <= sk.A_MAX + 1e-6
        assert T.vx.max() <= v0 * 1.0 + 1e-6
        lanes = np.round(T.y / sc.lane_width)                  # y on lane centres except during a lane change
        assert set(np.unique(lanes)) <= set(-np.arange(sc.lanes))
