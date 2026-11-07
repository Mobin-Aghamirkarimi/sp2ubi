"""Sensitivity of the level variant: the R2 dispersion rule when the actual noise exceeds the disclosed sigma.
Configuration: 8 levels (3 bits), site qualification sigma_hat <= 6 %. R2 variants:
  * fixed allowance c: R2 standardises residuals with c * sigma_eff (c = 1.0 / 1.2 / 1.3); R1 unchanged;
  * population calibration: per epoch and disclosed level, the insurer estimates the actual scale of standardised
    residuals from all policies (1.4826 * MAD, corrected for the policy-mean fit) and uses max(1, scale_level) *
    sigma_eff in R2. The calibration population is honest (manipulators are a minority).
Honest false punitive at actual noise = 1.0 / 1.2 / 1.5 x disclosed sigma_eff (per-cell multiplier), and power at
nominal noise for day-level manipulation (-15 %, f 0.3).
Usage: python inhouse/residual_scale.py [--vehicles 10000]
"""
import argparse
import json
import os

import numpy as np

import decisions_base as DB
import decisions_levels as E
from levels_qualification import requalify, scale_to_disclosed
from decisions_base import EPOCHS, seed

HERE = os.path.dirname(os.path.abspath(__file__))


def population_scale(sim, b, n_levels):
    """Per-level scale (1.4826 MAD) of residuals (b - policy-epoch mean) / sigma_eff over the whole population."""
    key, se = sim["key"], sim["se_i"]
    nE = sim["n_veh"] * EPOCHS
    w = 1 / se ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.bincount(key, w * b, nE) / np.bincount(key, w, nE)
    M = np.bincount(key, minlength=nE)[key]
    r = (b - m[key]) / se / np.sqrt(np.maximum(1 - 1 / np.maximum(M, 2), 0.5))
    f = np.ones(n_levels)
    for l in range(n_levels):
        x = r[(sim["lvl"] == l) & (M >= E.M_MIN)]
        if x.size > 200:
            f[l] = max(1.0, 1.4826 * np.median(np.abs(x - np.median(x))))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=10_000)
    a = ap.parse_args()
    qtab = DB.r2_quantiles(10_000_000, 1)
    A0 = E.load_sites(["extra"])
    sw = E.site_weights(A0, "floor")
    A, qual = requalify(A0, 0.06, 8)
    nv = a.vehicles
    out = {}
    strata = (("urban", 15), ("highway", 40), ("suburban", 5))
    for variant in ("c1.0", "c1.2", "c1.3", "popcal"):
        row = dict(honest={}, power={})
        for vt, lam in strata:
            for k in (1.0, 1.2, 1.5):
                mult = scale_to_disclosed(A, k)
                B = dict(A); B["ev"] = A["dt"][:, :, None] + (A["ev"] - A["dt"][:, :, None]) * mult[:, :, None]
                B["swithin"] = A["swithin"] * mult
                sim = E.simulate(seed("rr-h", vt, lam, k), B, sw, nv // 2, lam, vt, "empirical", qual=qual)
                if variant == "popcal":
                    f = population_scale(sim, sim["b_hon"], len(A["levels"])); r2s = f[sim["lvl"]]
                else:
                    f = None; r2s = np.full(sim["se_i"].shape, float(variant[1:]))
                h = E.honest_summary(E.decide(sim, sim["b_hon"], qtab, r2_scale=r2s))
                row["honest"][f"{vt}|{lam}|x{k}"] = dict(punitive=h["punitive_per_year"], ub95=h["punitive_ub95"],
                                                         scale=None if f is None else f.round(3).tolist())
            for att in (("days", -0.15, 0.3),):
                sim = E.simulate(seed("rr-p", att, vt, lam), A, sw, max(nv // 5, 200), lam, vt, "empirical",
                                 attack=att, rng_att=seed("rr-a", att, vt, lam), qual=qual)
                if variant == "popcal":
                    f = population_scale(sim, sim["b_hon"], len(A["levels"])); r2s = f[sim["lvl"]]
                else:
                    r2s = np.full(sim["se_i"].shape, float(variant[1:]))
                da = E.decide(sim, sim["b_att"], qtab, r2_scale=r2s)
                row["power"][f"{att[0]}|{vt}|{lam}"] = float(da["punitive"].mean())
        out[variant] = row
        print(f"{variant:7s} honest punitive " + " ".join(f"{k}:{v['punitive']:.1e}" for k, v in row["honest"].items())
              + " | power " + " ".join(f"{k}:{v:.2f}" for k, v in row["power"].items()), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "residual_scale.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
