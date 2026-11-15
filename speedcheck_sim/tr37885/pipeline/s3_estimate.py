"""Study 3: statistics of the urban estimator (fdoa/urban.py) and of its comparison variants for every crossing of the
per-beacon series written by pipeline/s3_series.py.

VARIANTS (the names are those of the stored result files):
  h_pp_now            the urban estimator: pulse-pair FDOA, no weights, f_o and rho fixed to the medians of the site's
                      survey fits (same speed, mapping and PRB), speed nodes
  h_pp_now_quad       the same with the constant-acceleration model (no nodes)
  h_peak_now          the same as h_pp_now with the FDOA of the periodogram peaks
  g_s30_survey        the same as h_pp_now with the envelope weights
  h_peak_now_fofree   the same as h_peak_now with f_o fitted per crossing
Every fit is written twice: with the statistics over the whole crossing (no suffix) and over the fitted positions in
fdoa.urban.CORE (suffix _c20). The urban estimator of the paper is h_pp_now_c20.

Output: <out>/<label>_site_<k>_<variant>[_c20].json, a list of records (site, track, cell, v0, role, kin, j, jr,
pair_k, mapping, prb, window_ok, stop_between, prof, match, mse, fo, rho).
Usage: python pipeline/s3_estimate.py --series <dir> --family U3 --sites 30 40 --out <dir> [--variants h_pp_now]
           [--survey-role reff]
--survey-role (U3 only): the role whose crossings give f_o and rho, ref (survey in traffic, default) or reff (free-road
survey)."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ER, "..")]
from fdoa import urban as U  # noqa: E402
from fdoa.geometry import Site  # noqa: E402

VARIANTS = {
    "h_pp_now":          dict(freq="fpp", weights=False, fo_survey=True, nodes=True),
    "h_pp_now_quad":     dict(freq="fpp", weights=False, fo_survey=True, nodes=False),
    "h_peak_now":        dict(freq="fd", weights=False, fo_survey=True, nodes=True),
    "g_s30_survey":      dict(freq="fpp", weights=True, fo_survey=True, nodes=True),
    "h_peak_now_fofree": dict(freq="fd", weights=False, fo_survey=False, nodes=True),
}
CORES = {"": None, "_c20": U.CORE}
COPIED = ("site", "track", "cell", "v0", "role", "kin", "j", "jr", "pair_k", "mapping", "prb", "window_ok", "stop_between")


def one(Z, r, site, cfg, fo=None, rho=None, nodes=None):
    """Fit one crossing; returns the statistics per suffix of CORES. A failed fit gives a non-finite record."""
    n = r["n"]
    t, fd, v = Z[f"r{n}_t"], Z[f"r{n}_{cfg['freq']}"].astype(float), Z[f"r{n}_v"].astype(float)
    w = U.envelope_weights(Z[f"r{n}_pa"], Z[f"r{n}_pb"]) if cfg["weights"] else None
    nodes = cfg["nodes"] if nodes is None else nodes
    out = {}
    try:
        F = U.fit(t, fd, site, w=w, fo=fo, rho=rho, node_s=U.NODE_S if nodes else None)
        for suf, core in CORES.items():
            prof, match, mse = U.statistics(F, fd, v, q=w, core=core, D=site.D)
            out[suf] = dict(prof=prof, match=match, mse=mse, fo=F["fo"], rho=F["rho"])
    except Exception:                                    # noqa: BLE001
        nan = float("nan")
        out = {suf: dict(prof=nan, match=nan, mse=nan, fo=nan, rho=nan) for suf in CORES}
    return out


def run_site(a):
    series, label, s, out, name, survey_role = a
    paths = {suf: os.path.join(out, f"{label}_site_{s:03d}_{name}{suf}.json") for suf in CORES}
    if all(os.path.exists(p) for p in paths.values()):
        return paths[""]
    cfg = VARIANTS[name]
    J = json.load(open(os.path.join(series, f"{label}_site_{s:03d}.json")))
    Z = np.load(os.path.join(series, f"{label}_site_{s:03d}.npz"))
    site = Site(D=J["D"], pole_offset=J["pole_offset"], dh=J["dh"])
    recs = J["recs"]
    # survey constants: per-crossing fits with f_o and rho free and the constant-acceleration model
    free = {r["n"]: one(Z, r, site, cfg, nodes=False)[""] for r in recs}
    groups = {}
    for r in recs:
        groups.setdefault((r["cell"], r["v0"], r["mapping"], r["prb"]), []).append(r)
    out_recs = {suf: [] for suf in CORES}
    for g in groups.values():
        fo = rho = None
        sv = [free[r["n"]] for r in g if r["role"] == survey_role and r["kin"] == "sumo" and r["window_ok"]]
        if sv:
            rho = float(np.nanmedian([q["rho"] for q in sv]))
            if cfg["fo_survey"]:
                fo = float(np.nanmedian([q["fo"] for q in sv]))
        for r in g:
            st = one(Z, r, site, cfg, fo=fo, rho=rho)
            for suf in CORES:
                out_recs[suf].append(dict({k: r[k] for k in COPIED}, **st[suf]))
    for suf, p in paths.items():
        json.dump(out_recs[suf], open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
    return paths[""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--family", required=True, help="label of the series files (U3, U3in, U, U2, U2in, H)")
    ap.add_argument("--sites", type=int, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["h_pp_now"], choices=sorted(VARIANTS))
    ap.add_argument("--survey-role", default="ref", choices=["ref", "reff"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.series, a.family, s, a.out, v, a.survey_role) for v in a.variants for s in range(*a.sites)
            if os.path.exists(os.path.join(a.series, f"{a.family}_site_{s:03d}.json"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run_site, jobs):
            print(os.path.basename(p), flush=True)


if __name__ == "__main__":
    main()
