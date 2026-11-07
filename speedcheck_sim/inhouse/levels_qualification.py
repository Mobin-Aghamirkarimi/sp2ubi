"""Sensitivity of the level variant: disclosed sigma levels and site qualification under the realistic K population.

With K = 6 dB sites in the population the 4 log-spaced levels span 0.26-17.7 % (factor 4.1 per level), so the
disclosed sigma is far above the measured one at most cells and power is lost. Variants:
  * N_LEVELS 4 (2 bits) or 8 (3 bits);
  * site qualification: a (site, speed class) cell produces entries only if its reference-pass sigma_hat <= SIGMA_MAX
    (no qualification / 6 % / 4 % / 3 %); levels are spanned over the qualified cells only.
Reported per variant: share of passes kept, levels, disclosed / measured sigma_eff (median, 5-95 %), honest
false-punitive (empirical, 3 strata), margin at actual / disclosed sigma_eff = 1.0 / 1.2 / 1.5 (i.e. relative to what
the insurer is told), and power for two threat classes (day-level, persistent).
Usage: python inhouse/levels_qualification.py [--vehicles 10000]
"""
import argparse
import json
import os

import numpy as np

import decisions_base as DB
import decisions_levels as E
from decisions_base import SIGMA_INFL, seed

HERE = os.path.dirname(os.path.abspath(__file__))


def requalify(A, sigma_max, n_levels):
    B = dict(A)
    qual = A["sraw"] <= sigma_max if sigma_max else np.ones_like(A["sraw"], bool)
    s_infl = SIGMA_INFL * A["sraw"]
    levels = np.exp(np.linspace(np.log(s_infl[qual].min()), np.log(s_infl[qual].max()), n_levels))
    lvl = np.minimum(np.searchsorted(levels, s_infl - 1e-15), n_levels - 1)
    B["levels"], B["lvl"], B["sg"] = levels, lvl, levels[lvl]
    return B, qual


def scale_to_disclosed(A, k):
    """Honest noise such that the actual within-site sd equals k x the DISCLOSED sigma_eff in every cell: SIG_UNDER is
    applied per (site, speed) as k * sigma_eff_disclosed / sigma_measured (vector), via a per-cell multiplier."""
    se = np.sqrt(A["sg"] ** 2 + E.HONEST["SIGMA0"] ** 2)
    return k * se / np.maximum(A["swithin"], 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=10_000)
    ap.add_argument("--tag", default="extra")
    a = ap.parse_args()
    qtab = DB.r2_quantiles(10_000_000, 1)
    A0 = E.load_sites([a.tag])
    sw = E.site_weights(A0, "floor")
    nv = a.vehicles
    out = {}
    orig_sim = E.simulate
    for n_levels in (4, 8):
        for smax in (None, 0.06, 0.04, 0.03):
            A, qual = requalify(A0, smax, n_levels)
            tag = f"L{n_levels}|qual{smax}"
            se = np.sqrt(A["sg"] ** 2 + E.HONEST["SIGMA0"] ** 2)
            ratio = (se / A["swithin"])[qual]
            # share of passes kept (site weights x speed profile, averaged over the three profiles)
            kept = np.mean([float((sw[:, None] * E.profile(A, vt)[None, :] * qual).sum()) for vt in E.PROFILES])
            row = dict(levels=[round(100 * x, 2) for x in A["levels"]], kept_share=kept,
                       disclosed_over_measured=dict(q05=float(np.quantile(ratio, 0.05)), median=float(np.median(ratio)),
                                                    q95=float(np.quantile(ratio, 0.95))), honest={}, margin={}, power={})
            for vt, lam in (("urban", 15), ("highway", 40), ("suburban", 5)):
                sim = E.simulate(seed("lq-h", tag, vt, lam), A, sw, nv, lam, vt, "empirical", qual=qual)
                h = E.honest_summary(E.decide(sim, sim["b_hon"], qtab))
                row["honest"][f"{vt}|{lam}"] = dict(punitive=h["punitive_per_year"], ub95=h["punitive_ub95"],
                                                    notice=h["notice_per_year"], low_m=h["low_m_epoch_share"])
                # margin relative to the disclosed sigma: per-cell multiplier on the measured error
                for k in (1.0, 1.2, 1.5):
                    mult = scale_to_disclosed(A, k)
                    B = dict(A); B["ev"] = A["dt"][:, :, None] + (A["ev"] - A["dt"][:, :, None]) * mult[:, :, None]
                    B["swithin"] = A["swithin"] * mult
                    sim = E.simulate(seed("lq-m", tag, vt, lam, k), B, sw, nv // 2, lam, vt, "empirical", qual=qual)
                    hm = E.honest_summary(E.decide(sim, sim["b_hon"], qtab))
                    row["margin"][f"{vt}|{lam}|x{k}"] = dict(punitive=hm["punitive_per_year"], ub95=hm["punitive_ub95"])
            for att in (("days", -0.15, 0.3), ("persistent", -0.10, 1.0)):
                for vt, lam in (("urban", 15), ("suburban", 5), ("highway", 40)):
                    sim = E.simulate(seed("lq-p", tag, att, vt, lam), A, sw, max(nv // 5, 200), lam, vt, "empirical",
                                     attack=att, rng_att=seed("lq-a", tag, att, vt, lam), qual=qual)
                    da = E.decide(sim, sim["b_att"], qtab)
                    row["power"][f"{att[0]}|{att[1]}|{att[2]}|{vt}|{lam}"] = dict(
                        punitive=float(da["punitive"].mean()), notice=float(da["notice"].mean()))
            out[tag] = row
            hon = row["honest"]; mar = row["margin"]; pw = row["power"]
            print(f"{tag:16s} kept {kept:.2f} levels {row['levels']} disc/meas med {row['disclosed_over_measured']['median']:.2f} "
                  f"({row['disclosed_over_measured']['q05']:.2f}-{row['disclosed_over_measured']['q95']:.2f}) | honest pun "
                  + " ".join(f"{k}:{v['punitive']:.1e}" for k, v in hon.items())
                  + " | margin x1.2 " + " ".join(f"{v['punitive']:.1e}" for k, v in mar.items() if k.endswith("x1.2"))
                  + " x1.5 " + " ".join(f"{v['punitive']:.1e}" for k, v in mar.items() if k.endswith("x1.5"))
                  + " | P(pun) days.3 " + " ".join(f"{pw[f'days|-0.15|0.3|{vt}|{lam}']['punitive']:.2f}" for vt, lam in (("urban", 15), ("suburban", 5), ("highway", 40))),
                  flush=True)
    assert E.simulate is orig_sim
    json.dump(out, open(os.path.join(HERE, "results", "levels_qualification.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
