"""Study 2 Dev-3 run path (N14): the two pre-specified parts of the Dev-3 family D3U, its run settings and the
per-part identity test. The Dev-3 split is pre-specified here (evaluation_v6.dev3.parts, checked by
s2_constants.check_against); run_study2.sh runs each part through this module.

  parts          part a = sites 0-2, part b = sites 3-4 (contiguous ranges; as the pilot split); the
                 split changes no draw: seeds are per (family, site, k) and there is one Octave call per site
  settings       evaluation_v6.dev3: site_timeout_h = 39 h (N14), workers_stage1 = 1 (run.* unchanged, so the study-1
                 constants are unchanged)
  guard          Dev-3 stage 1 starts only after the development-stage design record is timestamped and the
                 study-1 records are merged
  identity       per part, on that part's own sites only: for every SUMO pass of D3U the track index k, meta, SUMO
                 trajectory, lateral offset, track_rngs draws (nu) and beacon schedule equal study-1 family U's; the
                 quadratic passes come after all SUMO tracks (k >= the SUMO track count) and reuse their pair's seeds.
                 A failure stops the part (exit 3). QuaDRiGa coefficients are not promised (one Octave call per site).
Usage: python pipeline/s2_dev3.py --part-sites a | --check-part a [s e] | --setting site_timeout_h |
           --guard | --identity a --out runs/identity_dev3_a.json [--n-scale 1.0]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import s2_constants as S2  # noqa: E402

FAM = S2.DEV3["family"]


def _ev6(consts=None):
    if consts is None:
        from pipeline.stage1_channels import C as consts
    ev6 = consts.get("evaluation_v6")
    S2.check_against(ev6)
    return ev6


def parts(consts=None):
    """((part, (site, ...)), ...) of the pre-specified Dev-3 split, in part order."""
    return tuple((x, tuple(range(s, e))) for x, s, e in _ev6(consts)["dev3"]["parts"])


def part_sites(part, consts=None):
    P = dict(parts(consts))
    if part not in P:
        raise ValueError(f"{FAM}: part {part!r} is not pre-specified (parts {sorted(P)})")
    return P[part]


def check_part(part, s0=None, s1=None, consts=None):
    """A part's sites; --sites s e, if given, must equal the part's pre-specified range."""
    ks = part_sites(part, consts)
    if s0 is not None and (s0, s1) != (ks[0], ks[-1] + 1):
        raise ValueError(f"{FAM} part {part}: --sites {s0} {s1} differ from the pre-specified range [{ks[0]}, {ks[-1] + 1})")
    return ks


def setting(name, consts=None):
    if name not in ("site_timeout_h", "workers_stage1"):
        raise ValueError(name)
    return _ev6(consts)["dev3"][name]


def guard(consts=None):
    st0 = _ev6(consts)["step0"]
    if not st0.get("timestamped"):
        raise RuntimeError("Dev-3 before the development-stage design record is timestamped")
    if not st0.get("study1_merge_done"):
        raise RuntimeError("Dev-3 stage 1 starts only after the study-1 records are merged")
    if st0.get("calibration") is None:          # a null calibration serves dev_s2_cal only
        raise RuntimeError("Dev-3 (tag dev_s2) needs evaluation_v6.step0.calibration (only dev_s2_cal runs without it)")
    S2.check_calibration(st0["calibration"])
    return True


def same_pass(a, b):
    return bool(np.array_equal(a.t_beacon, b.t_beacon) and np.array_equal(a.nu, b.nu)
                and np.array_equal(a.traj.x, b.traj.x) and np.array_equal(a.traj.y, b.traj.y)
                and np.array_equal(a.traj.vx, b.traj.vx) and np.array_equal(a.t_snap, b.t_snap))


def identity_site(site, Q, n_scale=1.0):
    """The Dev-3 identity of one site: D3U's SUMO passes equal study-1 U's, the quadratic passes follow all SUMO
    tracks and reuse their pair's seeds. Returns dict(ok, checks)."""
    from pipeline import stage1_channels as gf
    _, pu, mu, sdu = gf.build_site("U", site, Q, n_scale)
    _, pd, md, sdd = gf.build_site(FAM, site, Q, n_scale)
    sumo = [(m, p) for m, p in zip(md, pd) if m["kin"] == "sumo"]
    quad = [m for m in md if m["kin"] == "quadratic"]
    n_sumo = len(pu)
    by_k = {m["k"]: m for m in md}
    chk = dict(seeds=bool(sdu == sdd), n_sumo=bool(len(sumo) == n_sumo),
               sumo_passes=bool(len(sumo) == n_sumo and all(m == m0 and same_pass(p, p0)
                                                            for (m, p), m0, p0 in zip(sumo, mu, pu))),
               quadratic_after_loop=bool(quad and all(m["k"] >= n_sumo for m in quad)),
               quadratic_pairs=bool(len(quad) == sum(m["role"] == "test" for m in mu)
                                    and all(by_k[m["pair_k"]]["kin"] == "sumo" and by_k[m["pair_k"]]["j"] == m["j"]
                                            for m in quad)))
    return dict(ok=bool(all(chk.values())), checks=chk, n_sumo=n_sumo, n_quadratic=len(quad))


def identity_part(part, n_scale=1.0, Q=None, consts=None):
    """The identity test of one part, on that part's own sites only."""
    from pipeline.stage1_channels import C
    guard(consts)
    Q = C["run"]["Q"] if Q is None else Q
    res = {str(s): identity_site(s, Q, n_scale) for s in part_sites(part, consts)}
    return dict(family=FAM, part=part, sites=list(part_sites(part, consts)), n_scale=n_scale, Q=Q,
                ok=bool(all(r["ok"] for r in res.values())), per_site=res,
                design=S2.SOURCE + " (Dev-3 identity, parts)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--part-sites")
    g.add_argument("--check-part", nargs="+")
    g.add_argument("--setting")
    g.add_argument("--guard", action="store_true")
    g.add_argument("--identity")
    ap.add_argument("--out")
    ap.add_argument("--n-scale", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.part_sites:
        print(" ".join(map(str, part_sites(a.part_sites))))
    elif a.check_part:
        p, *se = a.check_part
        check_part(p, *(int(x) for x in se)) if se else check_part(p)
    elif a.setting:
        print(setting(a.setting))
    elif a.guard:
        guard()
    else:
        res = identity_part(a.identity, a.n_scale)
        if a.out:
            json.dump(res, open(a.out + ".tmp", "w"), indent=1)
            os.replace(a.out + ".tmp", a.out)
        print(f"Dev-3 identity part {a.identity}: {'ok' if res['ok'] else 'FAILED'}")
        if not res["ok"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
