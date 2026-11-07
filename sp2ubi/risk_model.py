"""The insurer's risk model: additive non-negative Poisson model for the claim count.

    expected claims over the epoch = (sum_d sum_i w_i x_{i,d}) / S,   S = 2^32

with x_{0,d} = 1 (one insured day) and the integer daily features below. Fitted on So, Boucher & Valdez (2021)
by experiments/risk_model/fit_risk_model.py (out-of-fold Poisson D^2 = 0.136, Gini = 0.495). The insurer may
multiply the result by a per-policy factor exp(gamma . z) from traditional rating variables it already knows; that
step happens in plaintext after decryption and is outside the encrypted computation.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                          "so2021_additive_poisson.json")

# Feature order of the risk model (x_0 is the intercept slot); the daily values sent by the OBU are integers.
FEATURES = ("days", "offpeak_miles", "rush_miles", "weekend_miles", "hard_brakes", "hard_accels")


@dataclass(frozen=True)
class IntegerModel:
    weights: Dict[str, int]
    scale: float
    units: Dict[str, float]

    @property
    def w(self) -> List[int]:
        return [self.weights[f] for f in FEATURES]

    def daily_score(self, x: Sequence[int]) -> int:
        return sum(wi * xi for wi, xi in zip(self.w, x))

    def expected_claims(self, score_sum: int) -> float:
        return score_sum / self.scale


def load_model(path: str = MODEL_FILE) -> IntegerModel:
    m = json.load(open(path))
    if tuple(m["feature_order"]) != FEATURES:
        raise ValueError("feature order of the model file does not match FEATURES")
    return IntegerModel(m["weights"], float(m["scale_S"]), m["units"])

