"""C8 interface null test (must pass before any other cell runs): a LoS-only channel
through the generator interface (analytic LoS paths at Q snapshots per beacon -> band average -> cubic interpolation
to M samples) must reproduce the in-house LoS-only result (simulate_pass with K = 400 dB) with common random numbers:
|mean difference of b_hat| <= 0.05 % and sd ratio in [0.97, 1.03] over 500 passes, for the profile and the matched
statistic. Geometry: head 5 m, vehicle antenna 1.6 m (dh = 3.4 m), pole offset
U(6, 12) m, L = 100 m, v0 = 20 m/s, lane-change probability 0.1.
Generator-driver arms, same criterion and common random numbers: --arm g1 (QuaDRiGa
'LOSonly' through quadriga/site.m / run_site / the stage-1 record path; term off and on; 100
passes in 10 sites of 10) and --arm g2 (Sionna RT max_depth 0 through run_pass / pass_record / narrowband_from_record).
Run: python pipeline/null_test.py --tag primary|dev_<x> [--passes 500 --Q 16 --prb 10]
Output: results/c8_<arm>_prb<p>_<tag>.json with provenance; run_study1.sh requires every C8 file to pass before
any other stage."""
import argparse
import copy
import json
import os
import sys

import numpy as np

ER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
import interface as itf  # noqa: E402
import site_accuracy as E1  # noqa: E402
from fdoa.channel import ChannelConfig  # noqa: E402
from fdoa.estimator import fdoa_from_samples  # noqa: E402
from fdoa.geometry import Site, make_trajectory  # noqa: E402
from quadriga.driver import snapshot_times  # noqa: E402
from pipeline.provenance import check_tag, provenance, same_provenance  # noqa: E402

C = json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study1_constants.json"))))
C8 = C["criteria"]["C8"]
OUT = os.environ.get("SPEEDCHECK_RESULTS", os.path.join(ER, "..", "results"))


def write(a, out):
    """Formal output with provenance: results/c8_<arm>_prb<p>_<tag>.json."""
    out["provenance"] = provenance(a.tag, stage=1, arm=a.arm, passes=a.passes, Q=a.Q, prb=a.prb, seed=a.seed)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"c8_{a.arm}_prb{a.prb}_{a.tag}.json")
    json.dump(out, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)                     # atomic
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1))
    print(path)

E1.OFFSET_FIT = True


def one_pass(seed, Q, prb, v0=20.0):
    rng = np.random.default_rng(seed)
    site = Site(D=100.0, pole_offset=rng.uniform(6, 12), dh=3.4)
    traj = make_trajectory(rng, site, v0, p_lane_change=0.1)
    cfg = ChannelConfig(K_db=400.0)
    rs = np.random.default_rng(seed + 10_000_000)
    rs2 = copy.deepcopy(rs)
    t_b, y_in, _ = itf.inhouse_clean(rs, site, traj, cfg)             # in-house LoS-only, clean
    t_b2 = t_b                                                          # same schedule (same first draw)
    assert np.allclose(t_b2, itf.inhouse_clean(rs2, site, traj, cfg)[0])
    t_s = snapshot_times(t_b, Q)
    sig_g = np.stack([itf.narrowband_from_paths(c, d, Q, prb) for c, d in itf.los_paths(site.heads(), traj, t_s)])
    x, yy, vx, _ = traj.at(t_b)
    r = np.linalg.norm(site.heads()[:, None, :] - np.stack([x, yy, np.zeros_like(x)], -1)[None], axis=-1)
    off = rng.uniform(-1, 1) * 5e-9 * 5.9e9
    res = {}
    snr_db = itf.snr_e1_db(r)
    gain_g = np.ones_like(r)                                            # LoS-only, unit amplitude: sum of ray powers = 1
    for name, sig, gain in (("inhouse", y_in, None), ("generator", sig_g, gain_g)):
        rf = np.random.default_rng(seed + 20_000_000)                   # common random numbers for finish
        y = itf.finish(rf, sig, snr_db, gain=gain, head_offset_hz=(off / 2, -off / 2), t_b=t_b)
        fd = fdoa_from_samples(y, 1e-3)
        rho = np.hypot(site.pole_offset - yy, site.dh)
        st = E1.statistics(t_b, fd, vx, x, rho, site)
        res[name] = (st[0], st[1])
    return res


def compare(t_b, traj, site, sig_g, gain_g, sig_in, seed, mapping):
    """Both arms through the identical finish with common random numbers; returns {arm: (profile, matched)}."""
    x, yy, vx, _ = traj.at(t_b)
    r = np.linalg.norm(site.heads()[:, None, :] - np.stack([x, yy, np.zeros_like(x)], -1)[None], axis=-1)
    off = np.random.default_rng(seed + 30_000_000).uniform(-1, 1) * 5e-9 * 5.9e9
    snr_db = itf.snr_e1_db(r) if mapping == "e1" else itf.snr_link_budget_db(gain_g)
    res = {}
    for name, sig, gain in (("inhouse", sig_in, None), ("generator", sig_g, gain_g)):
        rf = np.random.default_rng(seed + 20_000_000)
        y = itf.finish(rf, sig, snr_db, gain=gain, head_offset_hz=(off / 2, -off / 2), t_b=t_b)
        st = E1.statistics(t_b, fdoa_from_samples(y, 1e-3), vx, x, np.hypot(site.pole_offset - yy, site.dh), site)
        res[name] = (st[0], st[1])
    return res


def g1_driver_site(site_seed, n, Q, prb, v0=20.0):
    """Generator-driver arm: QuaDRiGa 'LOSonly' through quadriga/site.m / run_site and the stage-1
    record path (band_avg_q of term off and term on with their own delays, ls normalisation, both SNR mappings)."""
    from quadriga.driver import build_pass, run_site
    rng = np.random.default_rng(site_seed)
    site = Site(D=100.0, pole_offset=rng.uniform(6, 12), dh=3.4)
    trajs = [make_trajectory(rng, site, v0, p_lane_change=0.1) for _ in range(n)]
    passes = [build_pass(np.random.default_rng([site_seed, i, 0]), site, v0, Q, 0.0, traj=T,
                         rng_schedule=np.random.default_rng([site_seed, i, 1])) for i, T in enumerate(trajs)]
    res = run_site("LOSonly", site, passes, seed=int(site_seed % 100000))
    cfg = ChannelConfig(K_db=400.0)
    out = []
    for i, (T, p) in enumerate(zip(trajs, passes)):
        t_b, y_in, _ = itf.inhouse_clean(np.random.default_rng([site_seed, i, 1]), site, T, cfg)
        assert np.allclose(t_b, p.t_beacon)
        for term, dkey in (("off", "dl"), ("on", "dlon")):
            sig = np.stack([itf.interp_q(itf.band_avg_q(res[(i, h)][term], res[(i, h)][dkey], Q, prb)) for h in range(2)])
            gain = np.stack([itf.beacon_gain(res[(i, h)]["ls"], Q) for h in range(2)])
            for mp in ("e1", "lb"):
                out.append((f"g1_{term}_{mp}", compare(t_b, T, site, sig, gain, y_in, site_seed * 1000 + i, mp)))
    return out


def g2_driver_passes(n, prb, seed, v0=20.0):
    """Generator-driver arm for G2: Sionna RT LoS only (max_depth 0) through run_pass / pass_record /
    narrowband_from_record (carrier-phase convention, path selection, gain normalisation), both SNR mappings."""
    import tempfile
    from quadriga.driver import beacon_schedule
    from g2.g2_driver import HEAD_Z, LANE_Y, POLE_Y, VEH_Z, G2Canyon, narrowband_from_record, pass_record, run_pass
    g2 = G2Canyon(tempfile.mkdtemp(prefix="c8g2_"), scatter=0.0)
    g2.place_heads(-50.0)
    site = Site(D=100.0, pole_offset=LANE_Y - POLE_Y, dh=HEAD_Z - VEH_Z)
    out = []
    cfg = ChannelConfig(K_db=400.0)
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        T = make_trajectory(rng, site, v0, x_start=-35, x_end=135)
        T.y[:] = 0.0; T.vy[:] = 0.0
        t_b = beacon_schedule(np.random.default_rng([seed, i, 1]), T)
        t_in, y_in, _ = itf.inhouse_clean(np.random.default_rng([seed, i, 1]), site, T, cfg)
        assert np.allclose(t_in, t_b)
        rec = pass_record(run_pass(g2, T, t_b, 0.0, 0.0, 42, max_depth=0), (prb,))
        sig = np.stack([narrowband_from_record(rec[h], prb)[0] for h in range(2)])
        gain = np.stack([narrowband_from_record(rec[h], prb)[1] for h in range(2)])
        for mp in ("e1", "lb"):
            out.append((f"g2_{mp}", compare(t_b, T, site, sig, gain, y_in, seed * 1000 + i, mp)))
    return out


def summarise(R):
    out, ok = {}, True
    for j, stat in enumerate(("profile", "matched")):
        ih = np.array([r["inhouse"][j] for r in R]); gn = np.array([r["generator"][j] for r in R])
        d = float(np.mean(gn - ih)); ratio = float(np.std(gn) / np.std(ih))
        passed = abs(d) <= 0.0005 and 0.97 <= ratio <= 1.03
        ok &= passed
        out[stat] = dict(mean_diff=d, sd_inhouse=float(np.std(ih)), sd_generator=float(np.std(gn)), sd_ratio=ratio,
                         max_abs_pass_diff=float(np.max(np.abs(gn - ih))), pass_criterion=bool(passed))
    out["C8_pass"] = bool(ok)
    return out


def main_drivers(a):
    groups = {}
    if a.arm == "g1":
        for sidx in range(a.passes // 10):
            for name, r in g1_driver_site(a.seed + 7919 * sidx, 10, a.Q, a.prb):
                groups.setdefault(name, []).append(r)
    else:
        for name, r in g2_driver_passes(a.passes, a.prb, a.seed):
            groups.setdefault(name, []).append(r)
    out = dict(arm=a.arm, passes=a.passes, Q=a.Q, prb=a.prb, seed=a.seed, groups={k: summarise(v) for k, v in groups.items()})
    out["C8_pass"] = all(v["C8_pass"] for v in out["groups"].values())
    write(a, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=500)
    ap.add_argument("--Q", type=int, default=16)
    ap.add_argument("--prb", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026101001)
    ap.add_argument("--arm", default="interface", choices=["interface", "g1", "g2"],
                    help="interface: analytic LoS (formal 500 passes); g1 / g2: generator drivers")
    ap.add_argument("--tag", required=True, help="'primary' (formal check, fixed counts) or dev_<x>")
    a = ap.parse_args()
    check_tag(a.tag, stage=1)
    if a.tag == "primary":                              # fixed counts and PRBs (constants criteria.C8)
        n_ok = C8["passes"] if a.arm == "interface" else C8["driver_passes"][a.arm]
        prbs = C8["interface_prbs"] if a.arm == "interface" else C8["driver_prbs"][a.arm]
        if a.passes != n_ok or a.prb not in prbs or a.Q != C["run"]["Q"] or a.seed != C8["seed"]:
            raise ValueError(f"primary C8 {a.arm}: passes {n_ok}, PRB in {prbs}, Q {C['run']['Q']}, seed {C8['seed']}")
    path = os.path.join(OUT, f"c8_{a.arm}_prb{a.prb}_{a.tag}.json")       # skip-if-exists:
    if os.path.exists(path) and same_provenance(json.load(open(path)).get("provenance", {}),   # a restart never repeats
                                                provenance(a.tag, stage=1, arm=a.arm, passes=a.passes, Q=a.Q, prb=a.prb, seed=a.seed)):
        print(path, "skipped (exists, same provenance)")                    # a finished gate
        return
    if a.arm != "interface":
        return main_drivers(a)
    R = [one_pass(a.seed + i, a.Q, a.prb) for i in range(a.passes)]
    out = dict(passes=a.passes, Q=a.Q, prb=a.prb, seed=a.seed)
    ok = True
    for j, stat in enumerate(("profile", "matched")):
        ih = np.array([r["inhouse"][j] for r in R]); gn = np.array([r["generator"][j] for r in R])
        d = float(np.mean(gn - ih)); ratio = float(np.std(gn) / np.std(ih))
        passed = abs(d) <= 0.0005 and 0.97 <= ratio <= 1.03
        ok &= passed
        out[stat] = dict(mean_diff=d, sd_inhouse=float(np.std(ih)), sd_generator=float(np.std(gn)), sd_ratio=ratio,
                         max_abs_pass_diff=float(np.max(np.abs(gn - ih))), pass_criterion=bool(passed))
    out["C8_pass"] = bool(ok)
    out["arm"] = "interface"
    write(a, out)


if __name__ == "__main__":
    main()
