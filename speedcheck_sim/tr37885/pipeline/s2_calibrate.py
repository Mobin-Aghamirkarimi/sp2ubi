"""Study 2 calibration reads (N11-N13; N18): the threshold-fixing script. Reads
only the dev_s2_cal stage-2 records of the Dev-1 survey passes (role == "ref", free road) and refuses any other role or
tag (no test or traffic pass is ever read). Fixes, before the development stage:
  a_max     (N11)  99th percentile of |acc_fit| over the P0U + P0N survey passes                        (E-c gate)
  k_stop    (N12)  the constant of the RSU-only low-speed-run (stop) detector of E-d / E-db: detection iff the pass's
                   stop_abs (Hz) < k_stop x its site's survey rms_ref_med; on the P0U + P0N
                   survey passes (no stops) the false-detection share rises with k_stop, so k_stop = the largest value with
                   <= 1 % false detections = the floor(0.01 n)-th smallest critical value (counted from 0) over the n
                   (pass, variant) records of the pool; a pass with no admissible run (+inf) never fires. Because the threshold is set by the residual level of
                   free-road passes at their own speed, it is a level of |FDOA - δ̂|: k_stop is reported with the implied
                   speed level v = k_stop x rms_ref_med x λ / |g| at |g| = 1 and 2 (a steady crawl below it is flagged
                   too; stated in the supplement).
  Δ_blk     (N13)  the L-B flag constant: flag iff blk_drop_max_db > Δ_blk; the smallest Δ with <= 1 % flagged P0N survey
                   passes (= the sorted drops at index n - 1 - floor(0.01 n)).
  Δ_rho     (N18)  the L-B′ flag constant: flag iff min over the site's surveyed lanes of |ρ̂ - ρ̂_survey,lane| > Δ_rho
                   (leave-one-out lane medians for the survey pass itself); the smallest Δ with <= 1 %
                   flagged P0N survey passes (same index rule as Δ_blk).
Pool (one value per constant, as pre-specified): the gen_on arm, beta = 0, SUMO kinematics, all six (SNR
mapping, PRB) variants pooled, included passes only (window rule); shares per (pass, variant)
record. The per-variant values and the achieved shares of the in-house arm at the calibrated values are reported beside
the pooled value, never used. A non-finite value is refused. Output: JSON with the values, the counts, the SHA-256 of
every input file and of this script.
Usage: python pipeline/s2_calibrate.py --eval P0U=<dir>/eval_dev_s2_cal --eval P0N=<dir>/eval_dev_s2_cal
           --out config/study2_calibration.json"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
from fdoa.geometry import LAMBDA  # noqa: E402
from pipeline import s2_constants as S2  # noqa: E402
from pipeline.s2_fields import included_s, rho_lane_offset  # noqa: E402

ARM = "gen_on"
ARMS_REPORTED = ("gen_on", "inhouse")
URBAN_LANE_W = None                                   # set from the constants in main (sumo_implementation.urban)


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_cal(fam, eval_dir):
    """The dev_s2_cal records of one family; refuses any record that is not a survey pass or not of tag dev_s2_cal."""
    recs, files = [], {}
    paths = sorted(glob.glob(os.path.join(eval_dir, "site_*.json")))
    if not paths:
        raise RuntimeError(f"{fam}: no stage-2 files in {eval_dir}")
    for f in paths:
        d = json.load(open(f))
        if d.get("tag") != S2.CAL_TAG:
            raise RuntimeError(f"{f}: tag {d.get('tag')!r} is not {S2.CAL_TAG!r}: refused")
        bad = sorted({r.get("role") for r in d["recs"]} - {"ref"})
        if bad:
            raise RuntimeError(f"{f}: records with role(s) {bad}: the calibration reads survey passes only (refused)")
        files[os.path.relpath(f, ER) if f.startswith(ER) else f] = sha256_file(f)
        for r in d["recs"]:
            r["family"] = fam
        recs += d["recs"]
    return recs, files


def pool(recs, fams, arm=ARM):
    return [r for r in recs if r["family"] in fams and r["arm"] == arm and r.get("beta", 0.0) == 0.0
            and r["kin"] == "sumo" and included_s(r)]


def a_max(vals):
    v = np.abs(np.asarray(vals, float))
    v = v[np.isfinite(v)]
    return float(np.quantile(v, S2.N11_A_MAX_QUANTILE)) if v.size else float("nan")


def k_stop(crit):
    """N12: the largest k with share(crit < k) <= 1 % (see the module note); +inf entries (no admissible run) never
    fire; NaN entries are dropped."""
    c = np.sort(np.asarray(crit, float))
    c = c[~np.isnan(c)]
    if not c.size:
        return float("nan")
    m = int(math.floor(S2.N12_FALSE_DETECT_MAX * c.size))
    return float(c[m])


def false_detect_share(crit, k):
    c = np.asarray(crit, float)
    c = c[~np.isnan(c)]
    return float(np.mean(c < k)) if c.size else float("nan")


def delta_blk(drops, max_share=S2.N13_FALSE_FLAG_MAX):
    """Smallest Δ with share(drop > Δ) <= 1 % (N13; with max_share = N18 also Δ_rho)."""
    d = np.sort(np.asarray(drops, float))
    d = d[np.isfinite(d)]
    if not d.size:
        return float("nan")
    m = int(math.floor(max_share * d.size))
    return float(d[d.size - 1 - m])


def delta_rho(devs):
    """N18: smallest Δ_rho with share(rho_dev > Δ_rho) <= 1 %."""
    return delta_blk(devs, S2.N18_FALSE_FLAG_MAX)


def false_flag_share(drops, dlt):
    d = np.asarray(drops, float)
    d = d[np.isfinite(d)]
    return float(np.mean(d > dlt)) if d.size else float("nan")


def _site_key(r):
    return (r["family"], r["site"], r["cell"], r["v0"], r["arm"], r["mapping"], r["prb"])


def site_scales(pk):
    """The site scale of E-d (N12): the median sqrt(mse) of the site's included survey passes (same family,
    cell, speed, arm and variant: stage3_criteria.site_table's rms_ref_med)."""
    med = {}
    for r in pk:
        med.setdefault(_site_key(r), []).append(np.sqrt(r["mse"]))
    return {k: float(np.median(v)) for k, v in med.items()}


def crit_values(pk):
    """The per-pass critical values of the E-d detector under the pre-specified span and scale: stop_abs / the site's
    rms_ref_med (k_stop is compared with these)."""
    med = site_scales(pk)
    return [K_div(r["stop_abs"], med[_site_key(r)]) for r in pk]


def K_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.float64(a) / np.float64(b))


def rho_devs(pb):
    """N18: rho_dev of the pooled P0N survey passes (all are survey passes; leave-one-out lane medians)."""
    return rho_lane_offset(pb, lambda r: True, lambda r: URBAN_LANE_W or 3.5)


def implied_speed(pk, k):
    """N12: the speed level the threshold corresponds to: v = k x rms_ref_med x λ / |g| (|f| = v |g| / λ)
    at |g| = 1 and 2, summarised over the pool's site scales per (family, speed class): median, min, max (m/s), and as a
    share of the speed class v0. Reported only."""
    out = {}
    for key, sc in site_scales(pk).items():
        out.setdefault(f"{key[0]}|{key[3]:g}", []).append(sc)
    rep = {}
    for k_, v in sorted(out.items()):
        v0 = float(k_.split("|")[1])
        for g in (1.0, 2.0):
            sp = np.asarray(v) * k * LAMBDA / g
            rep[f"{k_}|g={g:g}"] = dict(median=float(np.median(sp)), min=float(np.min(sp)), max=float(np.max(sp)),
                                       median_over_v0=float(np.median(sp) / v0), n_site_variants=len(v))
    return rep


def shares_at(recs, fams, vals):
    """Achieved shares at the calibrated values on one arm's pool (per (pass, variant) record; reported only)."""
    pa, pk, pb = (pool(recs, fams["a_max"], vals["arm"]), pool(recs, fams["k_stop"], vals["arm"]),
                  pool(recs, fams["delta_blk"], vals["arm"]))
    acc = np.abs(np.asarray([r["acc_fit"] for r in pa], float))
    acc = acc[np.isfinite(acc)]
    return dict(a_max_exceed=float(np.mean(acc > vals["a_max"])) if acc.size else float("nan"),
                k_stop_false_detect=false_detect_share(crit_values(pk), vals["k_stop"]),
                delta_blk_false_flag=false_flag_share([r["blk_drop_max_db"] for r in pb], vals["delta_blk_db"]),
                delta_rho_false_flag=false_flag_share(rho_devs(pb), vals["delta_rho"]),
                n=dict(a_max=len(pa), k_stop=len(pk), delta_blk=len(pb)))


def calibrate(recs):
    """The four pre-specified constants from the pool (see the module note); refuses a non-finite value."""
    fams = S2.DEV1_CAL_FAMILIES
    pa, pk, pb = pool(recs, fams["a_max"]), pool(recs, fams["k_stop"]), pool(recs, fams["delta_blk"])
    if fams["delta_rho"] != fams["delta_blk"]:
        raise RuntimeError("Δ_rho and Δ_blk are fixed on the same P0N survey pool")
    vals = dict(a_max=a_max([r["acc_fit"] for r in pa]), k_stop=k_stop(crit_values(pk)),
                delta_blk_db=delta_blk([r["blk_drop_max_db"] for r in pb]), delta_rho=delta_rho(rho_devs(pb)))
    S2.check_calibration(vals)                          # no NaN / inf constant is accepted
    per_variant = {}
    for mp, prb in sorted({(r["mapping"], r["prb"]) for r in pa + pb}):
        sel = lambda P: [r for r in P if r["mapping"] == mp and r["prb"] == prb]
        per_variant[f"{mp}/{prb}"] = dict(a_max=a_max([r["acc_fit"] for r in sel(pa)]),
                                          k_stop=k_stop(crit_values(sel(pk))),      # pre-specified span and scale
                                          delta_blk_db=delta_blk([r["blk_drop_max_db"] for r in sel(pb)]),
                                          delta_rho=delta_rho(rho_devs(sel(pb))))
    return dict(vals,
                n=dict(a_max=len(pa), k_stop=len(pk), delta_blk=len(pb), delta_rho=len(pb)),
                achieved={arm: shares_at(recs, fams, dict(vals, arm=arm)) for arm in ARMS_REPORTED},
                k_stop_implied_speed_mps=implied_speed(pk, vals["k_stop"]),
                per_variant_reported_only=per_variant,
                rules=dict(e_d_span=S2.OPTIONS["e_d_bracket"], e_d_scale=S2.OPTIONS["e_d_scale"],
                                k_stop_rule="largest value with <= 1 % false detections",
                                delta_rho_rule="smallest value with <= 1 % false flags, leave-one-out lane medians"),
                pool=dict(arm=ARM, beta=0.0, kin="sumo", variants="all mappings and PRBs pooled", included_only=True,
                          shares="per (pass, variant) record", families=fams,
                          reported_only="per-variant values; in-house achieved shares; implied speed level"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True, help="FAMILY=<eval_dev_s2_cal folder>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs, files = [], {}
    for spec in a.eval:
        fam, d = spec.split("=", 1)
        if fam not in S2.DEV1_FAMILIES:
            raise ValueError(f"{fam}: not a Dev-1 family")
        r, f = load_cal(fam, d)
        recs += r
        files.update(f)
    from pipeline.stage1_channels import C
    global URBAN_LANE_W
    URBAN_LANE_W = float(C["sumo_implementation"]["urban"]["lane_width_m"])
    out = calibrate(recs)
    out.update(tag=S2.CAL_TAG, inputs_sha256=files, script_sha256=sha256_file(os.path.abspath(__file__)),
               design=S2.SOURCE + " (N11-N13, N18)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("written", a.out)                             # the values are not printed
    return out


if __name__ == "__main__":
    main()
