"""TR 37.885 V15.3.0 antenna elements used by the speed-check simulation.
Head (UE-type RSU, Table 6.1.4-4, 6 GHz): A_E,V(theta'') = -min{12((theta'' - 90)/theta_3dB)^2, SLA_V}, theta_3dB = 120 deg,
SLA_V = 20 dB; no horizontal pattern (omnidirectional in azimuth); A'' = -min{-A_E,V, A_m}, A_m = 20 dB; 3 dBi;
tilt 96 deg (Table 6.1.4-5), applied as theta'' = theta - (tilt - 90) for every azimuth (bearing has no effect; the
mechanical-rotation reading differs by <= 0.37 dB on the LoS of every placement; separate check, not part of this repository).
Vehicle (primary): sealed roof unit, iso, 0 dBi. Sensitivity: Option 1 vehicle type 2 rooftop element
(Table 6.1.4-8, 6 GHz): theta_3dB = 90 deg, SLA_V = 20 dB, A_E,H = 0, A_m = 20 dB, 3 dBi, tilt 90 deg.
Angles: zenith theta in degrees (0 = up, 90 = horizon)."""
import json
import os

import numpy as np

_C = json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "study1_constants.json"))))
HEAD = _C["geometry"]["head_element"]
VEH_TYPE2 = dict(theta_3dB_deg=90.0, SLA_V_dB=20.0, A_m_dB=20.0, max_gain_dBi=3.0, tilt_deg=90.0)


def element_gain_db(theta_deg, el):
    """Gain (dBi) of a vertical-pattern-only TR 37.885 element at zenith angle theta (deg, GCS)."""
    thpp = np.asarray(theta_deg, float) - (el["tilt_deg"] - 90.0)
    av = -np.minimum(12.0 * ((thpp - 90.0) / el["theta_3dB_deg"]) ** 2, el["SLA_V_dB"])
    return el["max_gain_dBi"] + np.maximum(av, -el["A_m_dB"])


def head_gain_db(theta_deg):
    return element_gain_db(theta_deg, HEAD)


def octave_vector(el):
    """[theta_3dB, SLA_V, A_m, max gain, tilt] for quadriga/site.m."""
    return np.array([el["theta_3dB_deg"], el["SLA_V_dB"], el["A_m_dB"], el["max_gain_dBi"], el["tilt_deg"]], float)
