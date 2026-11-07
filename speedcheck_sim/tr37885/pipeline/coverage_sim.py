"""K7 site simulation: the in-house per-site passes of the coverage check, cached
per site by pipeline/coverage_check.py. This is the stage-2 part of K7 (in-house chain + estimator; both statistics are
stored), listed in pipeline/provenance.STAGE_SOURCES[2], so a change here invalidates the cache through its stage-2 hash;
the stage-3 part (scaling, bootstrap, size, calibration) stays in coverage_check.py.
  * one_pass: one pass of site s, copy c, track k (in-house site-accuracy kinematics at 15 m/s, pole offset U(6, 12) m, the site K, the in-house site-accuracy simulation
    SNR mapping, nominal impairments, the site-accuracy simulation statistics, window rule);
  * pool: the sites of one replicate, reused from the cache only with the same (stage-2) provenance;
  * POPS: the two site-K populations (a: the site-accuracy simulation, N(9, 5^2) floored at 5 dB; b: urban LOS realised in-band K, N(4.5, 1.5^2)
    floored at 1 dB; disjoint seeds)."""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
import interface as itf  # noqa: E402
from fdoa.channel import ChannelConfig  # noqa: E402
from fdoa.estimator import fdoa_from_samples  # noqa: E402
from fdoa.geometry import FC, Site, make_trajectory  # noqa: E402
from pipeline.stage2_evaluate import HEAD_PPB, one_stat  # noqa: E402
from pipeline.stage1_channels import C  # noqa: E402
from pipeline.provenance import same_provenance  # noqa: E402

CC = C["criteria"]["C1"]["coverage_check"]
V0, BURST_S, WIN_ARC = 15.0, C["window_rule"]["burst_s"], C["window_rule"]["arc_L"]
V5K7 = C.get("evaluation_v5", {}).get("k7", {"min_runnable_replicates": 20, "c_max": 10.0})
POPS = {"a": dict(mean=9.0, sd=5.0, floor=5.0, seed=[]),          # the site-accuracy simulation site population
        "b": dict(mean=4.5, sd=1.5, floor=1.0, seed=[1])}         # urban-LOS realised in-band K, disjoint seeds


def one_pass(pop_seed, s, copy_, k, geo, K_db):
    base = [CC["seed"]] + pop_seed + [s, copy_, k]
    rng = np.random.default_rng(base + [0])
    traj = make_trajectory(rng, geo, V0, x_start=-0.35 * geo.D, x_end=1.35 * geo.D)
    cfg = ChannelConfig(K_db=K_db, K_db_head=(K_db, K_db), scattering="iso", specular_db=None)
    t_b, y, _ = itf.inhouse_clean(np.random.default_rng(base + [1]), geo, traj, cfg)
    x, yy, vx, _ = traj.at(t_b)
    heads = geo.heads()
    r = np.linalg.norm(heads[:, None, :] - np.stack([x, yy, np.zeros_like(x)], -1)[None], axis=-1)
    u_x = (heads[:, 0][:, None] - x[None, :]) / r
    off = np.random.default_rng(base + [2]).uniform(-1, 1) * HEAD_PPB * 1e-9 * FC
    yb = itf.finish(np.random.default_rng(base + [3]), y, itf.snr_e1_db(r), gain=None, head_offset_hz=(off / 2, -off / 2), t_b=t_b)
    yb = itf.apply_impairments(np.random.default_rng(base + [4]), yb, t_b, u_x, copy.deepcopy(C["impairments"]))
    st = one_stat(t_b, fdoa_from_samples(yb, 1e-3), vx, x, np.hypot(geo.pole_offset - yy, geo.dh), geo)
    inw = (x >= WIN_ARC[0] * geo.D) & (x <= WIN_ARC[1] * geo.D)
    t_window = float(t_b[inw][-1] - t_b[inw][0]) if inw.sum() >= 2 else float("inf")
    return dict(match=float(st["match"]), prof=float(st["prof"]), mse=float(st["mse"]), window_ok=bool(t_window <= BURST_S),
                finite=bool(np.isfinite(st["match"]) and np.isfinite(st["prof"])),
                match_d=float(st["match_d"]), prof_d=float(st["prof_d"]))


def pool(site_ids, n_ref, n_test, out_dir, prov, pop="a"):
    P = POPS[pop]
    sites = []
    for s in site_ids:
        path = os.path.join(out_dir, f"site_{s:03d}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            if not same_provenance(d.get("provenance", {}), prov):
                raise RuntimeError(f"{path} exists with a different provenance: not reused")
            sites.append(d)
            continue
        rs = np.random.default_rng([CC["seed"]] + P["seed"] + [s])
        geo = Site(D=C["geometry"]["L_m"], pole_offset=float(rs.uniform(6, 12)), dh=3.4)
        K_db = float(max(rs.normal(P["mean"], P["sd"]), P["floor"]))
        d = dict(site=s, pop=pop, K_db=K_db, provenance=prov,
                 copies=[[one_pass(P["seed"], s, c, k, geo, K_db) for k in range(n_ref + n_test)] for c in (0, 1)])
        json.dump(d, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        sites.append(d)
    return sites


