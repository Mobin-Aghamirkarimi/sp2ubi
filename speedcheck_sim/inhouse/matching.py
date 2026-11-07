"""Matching simulation: insurer + RSU matching under the candidate configurations, used to fix the number of disclosed sigma levels and on the
z resolution.

City: N_VEH policies, 2-10 sites each drawn with the K-population site weights of the decision simulation (level variant) (the site-accuracy simulation sites, qualification
sigma_hat <= SIGMA_MAX per (site, speed class); unqualified cells produce no entry), Poisson(lambda) passes per epoch,
urban / suburban / highway speed profiles. Per pass the RSU discloses the sigma level (4 or 8 levels spanned
over the qualified cells) and z_q; z is drawn from the honest per-pass z of the matching the base consistency simulation cell (same speed, K,
facade; drift arm 0), i.e. the empirical distribution including its tails.
Colluding RSUs group k records of one vehicle (with non-overlapping bursts: the crossings of one burst, k set by site
density along a corridor). Reported: median candidate policies and P(unique) for k in {1, 2, 3, 5, 8}; target for
the choice: median >= 100 and P(unique) <= 0.05 at k = 5, lambda = 5.
Resumable: results saved after every configuration to inhouse/results/matching<tag>.partial.json.
Usage: python inhouse/matching.py [--vehicles 20000]
"""
import argparse
import json
import os

import numpy as np

import decisions_levels as E
from levels_qualification import requalify
from matching_base import linkage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SIGMA_MAX = 0.06
Z_OPTIONS = {"z q1.0": 1.0, "z q3.0": 3.0, "level only": None}
CLIP = 3.0


def load_consistency_z(path, kind="d0.0"):
    """Honest per-pass z per cell: the base consistency simulation (kind 'd0.0') or the consistency simulation (kind 'trf': traffic standardisation)."""
    meta = json.load(open(path.replace("_z_", "_").replace(".npz", ".json")))
    arr = np.load(path)
    zmap = {}
    for i, c in enumerate(meta["cells"]):
        z = arr[f"c{i}_honest_{kind}"]
        zmap[(c["v0"], c["K_db"], c["specular_db"] is not None)] = z[np.isfinite(z)]
    return zmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=20_000)
    ap.add_argument("--e3", default=os.path.join(HERE, "results", "consistency_base_z_full.npz"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--kind", default="d0.0", help="d0.0 (base consistency z) or trf (traffic standardisation)")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    zmap = load_consistency_z(a.e3, a.kind)
    A0 = E.load_sites(["extra"])
    sw = E.site_weights(A0, "floor")
    part = os.path.join(OUT, f"matching{a.out_tag}.partial.json")
    rep = json.load(open(part)) if os.path.exists(part) else {}
    for n_levels in (4, 8):
        A, qual = requalify(A0, SIGMA_MAX, n_levels)
        for lam in (5, 15):
            if all(f"L{n_levels}|{zn}|lam{lam}" in rep for zn in Z_OPTIONS):
                continue                                     # resumed: this (levels, lambda) is complete
            rng = np.random.default_rng(a.seed + lam)
            nS, types = A["nS"], list(E.PROFILES)
            pol, site, spd = [], [], []
            gumbel_keys = np.log(sw)
            for p in range(a.vehicles):
                probs = E.profile(A, types[p % len(types)])
                ns = rng.integers(2, 11)
                sites = np.argsort(-(gumbel_keys - np.log(-np.log(rng.random(nS)))))[:ns]
                for _ in range(rng.poisson(lam)):
                    s = int(rng.choice(sites)); v = int(rng.choice(len(probs), p=probs))
                    if qual[s, v]:
                        pol.append(p); site.append(s); spd.append(v)
            pol, site, spd = np.array(pol), np.array(site), np.array(spd)
            lvl = A["lvl"][site, spd]
            keys = A["keys"]
            z = np.empty(len(pol))
            for idx in range(len(pol)):
                K, _, sp = keys[site[idx]][0]
                pool = zmap[(A["speeds"][spd[idx]], K, sp is not None)]
                z[idx] = pool[rng.integers(0, pool.size)]
            for zname, q in Z_OPTIONS.items():
                if f"L{n_levels}|{zname}|lam{lam}" in rep:
                    continue
                if q is None:
                    tid = lvl.copy()
                else:
                    zq = np.round(np.clip(z, -CLIP, CLIP) / q).astype(int); zq -= zq.min()
                    tid = lvl * (zq.max() + 1) + zq
                r = linkage(pol, tid, a.vehicles, np.random.default_rng(a.seed + 1), n_targets=2000)
                r.update(passes=int(len(pol)), n_levels=n_levels, lam=lam, z=zname)
                tag = f"L{n_levels}|{zname}|lam{lam}"
                rep[tag] = r
                tmp = part + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(rep, f, indent=1); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, part)
                print(f"{tag:22s} values {r['distinct_values']:4d} anon median {r['anon_median']:.0f} | " + " | ".join(
                    f"k={k}: {r[f'k{k}']['median_candidates']:.0f} ({r[f'k{k}']['p_unique']:.2f})" for k in (1, 2, 3, 5, 8)
                    if f"k{k}" in r), flush=True)
    json.dump(rep, open(os.path.join(OUT, f"matching{a.out_tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
