"""Base matching simulation: leakage on the speed-verification path of an earlier design (2-of-2 split of the speed-verification ElGamal key
and of the BBS04 opening key between insurer and TSP; RSU = verifier and relay, no storage; TSP stores and shuffles).

Which view can link what is fixed by the message flow. This script quantifies the residual
channels that the key split does not remove:
  (1) insurer alone (with the TSP's partials on shuffled entries): per pass (policy, b_hat, level, z). Mutual
      information of the disclosed per-pass values with the speed class and with the site type.
  (2) insurer + colluding RSUs: the RSU computes level and z in plaintext and knows site, time and roadside speed of
      the pass; the insurer later decrypts the same level and z next to the policy. Matching on these values links
      policy to (site, time, speed). Measured for several disclosure resolutions of z: anonymity set of one RSU
      record in the epoch batch, and the number of candidate policies when colluding RSUs can group k records of one
      vehicle (e.g. consecutive sites of one trip).
  (3) cost of coarse z for the consistency test: epoch power and honest false-alarm rate of the mean-z rule with
      quantised z, in a Gaussian model per the consistency simulation cell.
City model as in e5_leakage.py: N_VEH policies, 2-10 sites each, Poisson(lambda) passes per epoch, urban / suburban /
highway speed profiles, site table from the base decision simulation. Honest z ~ N(mean, sd) of the matching the consistency simulation
cell (nearest K).
Usage: python inhouse/matching_base.py [--vehicles 20000] [--extra k10]
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy import sparse, stats

import decisions_base as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CLIP = 3.0
# disclosure options for the per-pass consistency value z (level is always disclosed: 2 bits)
Z_OPTIONS = {"z raw (0.01)": 0.01, "z q=0.25": 0.25, "z q=0.5": 0.5, "z q=1.0": 1.0, "level only (z epoch sum)": None}
K_LINK = (1, 2, 3, 5, 8)


def mutual_information(x, y):
    n = len(x); cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    return sum(c / n * np.log2((c / n) / (cx[a] / n * cy[b] / n)) for (a, b), c in cxy.items())


def quantise(z, q):
    return np.round(np.clip(z, -CLIP, CLIP) / q).astype(int)


def e3_cells():
    d = json.load(open(os.path.join(HERE, "results", "consistency_first_full.json")))
    return {(c["v0"], c["K_db"], c["scattering"], c["specular_db"]): c for c in d["cells"]}


def e3_cell_for(e3, v, cc):
    K, sc, sp = cc
    Ks = sorted({k[1] for k in e3})
    Kn = min(Ks, key=lambda k: abs(k - K))
    return e3[(v, Kn, sc, sp)]


def city(A, rng, n_veh, lam, e3):
    nS = A["nS"]; types = list(E.VEHICLE_TYPES)
    pol, site, spd = [], [], []
    for p in range(n_veh):
        probs = np.array(E.VEHICLE_TYPES[types[p % len(types)]])
        sites = rng.choice(nS, rng.integers(2, 11), replace=False)
        n = rng.poisson(lam)
        pol += [p] * n; site += list(rng.choice(sites, n)); spd += list(rng.choice(4, n, p=probs))
    pol, site, spd = np.array(pol), np.array(site), np.array(spd)
    lvl = A["lvl"][site, spd]
    keys = A["keys"]
    mu = np.array([e3_cell_for(e3, E.SPEEDS[v], keys[s][0])["honest_mean"] for s, v in zip(site, spd)])
    sd = np.array([e3_cell_for(e3, E.SPEEDS[v], keys[s][0])["honest_sd"] for s, v in zip(site, spd)])
    z = mu + sd * rng.standard_normal(len(pol))
    return pol, site, spd, lvl, z


def tuple_ids(lvl, z, q):
    if q is None:
        return lvl.copy()
    zq = quantise(z, q)
    zq -= zq.min()
    return lvl * (zq.max() + 1) + zq


def linkage(pol, tid, n_veh, rng, n_targets=3000):
    """Per-pass anonymity set in the epoch batch, and candidate policies for k grouped records of one vehicle."""
    cnt = np.bincount(tid)
    anon = cnt[tid]
    _, t = np.unique(tid, return_inverse=True)
    Mx = sparse.csc_matrix((np.ones(len(pol)), (pol, t)), shape=(n_veh, t.max() + 1))
    Mx.sum_duplicates()
    out = dict(anon_median=float(np.median(anon)), anon_p01=float(np.quantile(anon, 0.01)), anon_min=int(anon.min()),
               distinct_values=int(t.max() + 1))
    order = np.argsort(pol, kind="stable")
    starts = np.searchsorted(pol[order], np.arange(n_veh + 1))
    for k in K_LINK:
        cands = []
        for p in rng.choice(n_veh, min(n_targets, n_veh), replace=False):
            idx = order[starts[p]:starts[p + 1]]
            if len(idx) < k:
                continue
            need = Counter(t[rng.choice(idx, k, replace=False)].tolist())
            ok = np.ones(n_veh, bool)
            for col, c in need.items():
                colv = np.zeros(n_veh); s, e = Mx.indptr[col], Mx.indptr[col + 1]
                colv[Mx.indices[s:e]] = Mx.data[s:e]
                ok &= colv >= c
            cands.append(int(ok.sum()))
        cands = np.array(cands)
        out[f"k{k}"] = dict(n=int(len(cands)), median_candidates=float(np.median(cands)),
                            p10_candidates=float(np.quantile(cands, 0.10)), p_unique=float(np.mean(cands == 1)))
    return out


def power_under_quantisation(e3, reps=20_000, alpha=1e-3 / 12 / 5, seed=7):
    """Mean-z epoch rule of the consistency simulation with z disclosed at resolution q (Gaussian model per cell)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(2_000_000)
    res = {}
    for name, q in Z_OPTIONS.items():
        if q is None:
            continue
        null = np.clip(base, -CLIP, CLIP) if q < 0.05 else quantise(base, q) * q
        sd0 = float(null.std())
        rows = []
        for c in e3.values():
            for beta, key in ((-0.03, "beta-0.03_mean_z"), (-0.05, "beta-0.05_mean_z")):
                for M in (5, 15, 40):
                    thr = stats.norm.ppf(alpha) * sd0 / np.sqrt(M)
                    def ep(mu):
                        z = mu + c["honest_sd"] * rng.standard_normal((reps, M))
                        zz = np.clip(z, -CLIP, CLIP) if q < 0.05 else quantise(z, q) * q
                        return float(np.mean(zz.mean(1) < thr))
                    rows.append(dict(v=c["v0"], K=c["K_db"], sc=c["scattering"], sp=c["specular_db"], beta=beta, M=M,
                                     pd=ep(c[key]), fa=ep(c["honest_mean"]) if beta == -0.03 else None))
        res[name] = dict(null_sd=sd0, rows=rows)
    summ = {}
    for name, r in res.items():
        s = {}
        for beta in (-0.03, -0.05):
            for M in (5, 15, 40):
                pds = [x["pd"] for x in r["rows"] if x["beta"] == beta and x["M"] == M]
                s[f"pd_beta{beta}_M{M}_min"] = float(min(pds)); s[f"pd_beta{beta}_M{M}_mean"] = float(np.mean(pds))
        for M in (5, 15, 40):
            fas = [x["fa"] for x in r["rows"] if x["fa"] is not None and x["M"] == M]
            s[f"honest_fa_M{M}_max"] = float(max(fas))
        summ[name] = s
    return summ, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=20_000)
    ap.add_argument("--lam", type=float, default=15)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    A = E.load_sites(tuple(a.extra))
    e3 = e3_cells()
    rng = np.random.default_rng(a.seed)
    pol, site, spd, lvl, z = city(A, rng, a.vehicles, a.lam, e3)
    keys = A["keys"]
    spec = np.array([keys[s][0][2] is not None for s in site])
    rep = dict(passes=int(len(pol)), vehicles=a.vehicles, lam=a.lam, n_sites=int(A["nS"]), levels=A["levels"].tolist(),
               insurer_alone={}, insurer_rsu={}, z_quantisation={})
    rep["insurer_alone"]["mi_level_speed_bits"] = float(mutual_information(lvl.tolist(), spd.tolist()))
    rep["insurer_alone"]["mi_level_specular_bits"] = float(mutual_information(lvl.tolist(), spec.tolist()))
    rep["insurer_alone"]["H_speed_bits"] = float(stats.entropy(np.bincount(spd), base=2))
    for name, q in Z_OPTIONS.items():
        if q is not None:
            zq = quantise(z, q).tolist()
            rep["insurer_alone"][f"mi_{name}_speed_bits"] = float(mutual_information(zq, spd.tolist()))
            rep["insurer_alone"][f"mi_{name}_specular_bits"] = float(mutual_information(zq, spec.tolist()))
            # plug-in MI is biased upward for many values: same estimate with the speed class permuted
            rep["insurer_alone"][f"mi_{name}_speed_bits_permuted"] = float(
                mutual_information(zq, rng.permutation(spd).tolist()))
        tid = tuple_ids(lvl, z, q)
        rep["insurer_rsu"][name] = linkage(pol, tid, a.vehicles, np.random.default_rng(a.seed + 1))
        r = rep["insurer_rsu"][name]
        print(f"{name:26s}: values {r['distinct_values']:5d} anon median {r['anon_median']:.0f} min {r['anon_min']} | " +
              " | ".join(f"k={k}: cand median {r[f'k{k}']['median_candidates']:.0f} P(unique) {r[f'k{k}']['p_unique']:.2f}"
                         for k in K_LINK), flush=True)
    summ, rows = power_under_quantisation(e3)
    rep["z_quantisation"] = summ
    for name, s in summ.items():
        print(f"{name:26s}: " + " ".join(f"{k}={v:.3g}" for k, v in s.items()), flush=True)
    tag = "_".join(a.extra + ([f"lam{a.lam:g}"] if a.lam != 15 else []))
    json.dump(rep, open(os.path.join(OUT, f"matching_base{('_' + tag) if tag else ''}.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(OUT, f"matching_base_zq_rows{('_' + tag) if tag else ''}.json"), "w"), indent=1)
    for k, v in rep["insurer_alone"].items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
