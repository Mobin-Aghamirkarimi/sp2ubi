"""Study 3: the per-site rules of stage 3 applied to the per-crossing statistics of any estimator, so that the urban
estimator is scored exactly like the base estimator (pipeline/stage3_criteria.py: inclusion, selection of the statistic
on the survey, site qualification at the calibration limit, calibration by the survey mean, quality gate at GATE times
the survey's median RMS residual, sigma on errors clipped to +-WIN, worst of the six mapping x PRB variants)."""
import glob
import json
import os

import numpy as np

SIG_Q, GATE, WIN, MIN_REF = 0.06, 1.3, 0.25, 20
VARIANTS = [(m, p) for m in ("e1", "lb") for p in (10, 20, 50)]
BASE = "ls"                 # name of the base estimator in the result files


def load(res, series, fam, name, survey_role="ref"):
    """Records of estimator `name` for family `fam`: BASE from the series folder (pipeline/s3_series.py), any other
    name from the result folder of pipeline/s3_estimate.py. Only SUMO crossings. U3 has two survey roles: the one in
    force is returned as "ref" and the other one is dropped."""
    recs = []
    if name == BASE:
        for f in sorted(glob.glob(os.path.join(series, f"{fam}_site_???.json"))):
            recs += json.load(open(f))["recs"]
    else:
        for f in sorted(glob.glob(os.path.join(res, f"{fam}_site_???_{name}.json"))):
            recs += json.load(open(f))
    out = []
    for r in recs:
        if r["kin"] != "sumo" or (r["role"] in ("ref", "reff") and r["role"] != survey_role):
            continue
        out.append(dict(r, role="ref") if r["role"] == survey_role else r)
    return out


def included(r):
    return r["window_ok"] and np.isfinite(r["prof"]) and np.isfinite(r["match"])


def site_cell(recs):
    """recs: the records of one site, speed and variant. Returns the per-site table of the rules, or None with fewer
    than MIN_REF included survey crossings or no included test crossing."""
    ref = [r for r in recs if r["role"] == "ref" and included(r)]
    tst0 = [r for r in recs if r["role"] == "test"]
    tst = [r for r in tst0 if included(r)]
    if len(ref) < MIN_REF or not tst:
        return None
    em, ep = np.array([r["match"] for r in ref]), np.array([r["prof"] for r in ref])
    key = "match" if np.var(em) <= np.var(ep) else "prof"
    b_ref = np.array([r[key] for r in ref])
    rms_ref_med = float(np.median(np.sqrt([r["mse"] for r in ref])))
    b = np.array([r[key] for r in tst])
    keep = np.sqrt(np.array([r["mse"] for r in tst])) <= GATE * rms_ref_med
    return dict(key=key, sd_ref=float(b_ref.std(ddof=1)), b_ref=float(b_ref.mean()), qualified=bool(b_ref.std(ddof=1) <= SIG_Q),
                e=b - b_ref.mean(), b=b, keep=keep, n_test_all=len(tst0), stop=np.array([r["stop_between"] for r in tst]))


def tables(recs, v0, mapping, prb):
    by = {}
    for r in recs:
        if r["v0"] == v0 and r["mapping"] == mapping and r["prb"] == prb:
            by.setdefault(r["site"], []).append(r)
    return {s: site_cell(v) for s, v in by.items()}


def sigma(e):
    return float(np.std(np.clip(e, -WIN, WIN))) if e.size else float("nan")


def cell_summary(tables_):
    """tables_: {site: site_cell(...)} for one speed and variant. Pooled quantities: `qg` = qualified sites, gated
    crossings; `all` = every included crossing of every site."""
    T = {s: t for s, t in tables_.items() if t is not None}
    q = [s for s in T if T[s]["qualified"]]
    e_all = np.concatenate([T[s]["e"] for s in T]) if T else np.array([])
    e_qg = np.concatenate([T[s]["e"][T[s]["keep"]] for s in q]) if q else np.array([])
    b_qg = np.concatenate([T[s]["b"][T[s]["keep"]] for s in q]) if q else np.array([])
    n_q_all = sum(T[s]["n_test_all"] for s in q)
    return dict(n_sites=len(T), k=len(q), sd_ref_med=float(np.median([T[s]["sd_ref"] for s in T])) if T else np.nan,
                sigma_all=sigma(e_all), sigma_qg=sigma(e_qg),
                bias_cal=float(np.mean(e_qg)) if e_qg.size else np.nan, bias_uncal=float(np.mean(b_qg)) if b_qg.size else np.nan,
                q99=float(np.quantile(np.abs(e_qg), 0.99)) if e_qg.size else np.nan,
                oow=float(np.mean(np.abs(e_qg) > WIN)) if e_qg.size else np.nan,
                gate=float(1 - np.mean(np.concatenate([T[s]["keep"] for s in q]))) if q else np.nan,
                entry_yield=float(sum(T[s]["keep"].sum() for s in q) / n_q_all) if n_q_all else np.nan,
                site_mean_sd=float(np.std([np.mean(T[s]["e"][T[s]["keep"]]) for s in q], ddof=1)) if len(q) > 1 else np.nan)
