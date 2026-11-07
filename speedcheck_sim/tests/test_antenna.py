"""TR 37.885 head element: Python reference (antenna.py) against the QuaDRiGa arrayant built by quadriga/tr37885_element.m."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885")
sys.path[:0] = [ER, os.path.join(ER, "..")]
import antenna  # noqa: E402

QUADRIGA = os.environ.get("QUADRIGA_PATCHED_DIR", "QuaDRiGa_patched")


def test_head_gain_reference_values():
    g = antenna.head_gain_db(np.array([96.0, 90.0, 0.0, 180.0]))
    assert np.allclose(g, [3.0, 3.0 - 12 * (6 / 120) ** 2, 3.0 - 12 * (96 / 120) ** 2, 3.0 - 12 * (84 / 120) ** 2])
    assert antenna.head_gain_db(np.array([300.0]))[0] == pytest.approx(3.0 - 20.0)    # SLA / A_m floor


@pytest.mark.skipif(shutil.which("octave") is None or not os.path.isdir(QUADRIGA),
                    reason="Octave / patched QuaDRiGa not installed")
@pytest.mark.parametrize("el", [antenna.HEAD, antenna.VEH_TYPE2])
def test_quadriga_element_matches_reference(el):
    elev = np.array([-60.0, -20.0, -6.0, 0.0, 10.0, 45.0, 90.0, -90.0])
    v = " ".join(f"{x:.6g}" for x in antenna.octave_vector(el))
    e = " ".join(f"{x:.6g}" for x in elev)
    code = (f"addpath(genpath('{QUADRIGA}/quadriga_src')); addpath('{ER}/quadriga');"
            f"a = tr37885_element([{v}]); el = [{e}]*pi/180; [V,H] = a.interpolate(zeros(size(el)), el);"
            "printf('%.10f ', 10*log10(abs(V).^2)); printf('\\n%.3e\\n', max(abs(H(:))));")
    r = subprocess.run(["octave", "--no-gui", "-q", "--eval", code], capture_output=True, text=True, timeout=300)
    lines = r.stdout.strip().splitlines()
    got = np.array(lines[-2].split(), float)
    assert np.allclose(got, antenna.element_gain_db(90.0 - elev, el), atol=1e-6)
    assert float(lines[-1]) == 0.0
