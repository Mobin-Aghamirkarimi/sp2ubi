"""The speed-check simulation fixed sensitivity S-place (constants
sensitivities.S_place): does TR 37.885's sampling of QuaDRiGa at the beacons only (production) change the
C1 / headline accuracy statistic against QuaDRiGa's own uniform sampling (beacon + uniform snapshots at lambda / 2.4)?

Design (fixed before any S-place data):
  * cells P1 (urban LOS, 15 m/s) and P2 (highway LOS, 38 m/s: fewest beacons per segment), 5 sites x 20 test passes
    (S-place light)
    each (SUMO with background traffic, lateral offset as production), own seeds (never the production seeds);
  * three QuaDRiGa calls per site with the same seed: B beacon-only (production), A beacon + uniform lambda / 2.4, N
    beacon-only + one dummy pass (environment-change null); term off, PRB 10, both SNR mappings, nominal impairments,
    common random numbers (identical offsets, noise and impairments in the three calls);
  * statistic: sigma_w = sd of the within-site-centred, +-25 % clipped select statistic (select = matched or profile,
    whichever has the smaller variance over the site's B passes, the same choice for A and N), over included passes
    (window rule, finite); rho = sigma_w(A) / sigma_w(B), rho_N = sigma_w(N) / sigma_w(B), site-bootstrap 95 % CIs;
  * rule (constants sensitivities.S_place.rule_v4): rho_hat > 1.10 in either mapping -> correction by the larger
    such rho_hat, whatever the CI (point estimate; the CI is reported); the original rule also required CI lower limit > 1.
    A correction means: every generator sigma
    of the stratum in headline_final and in C1 (sigma_gen, R and its CI) is multiplied by rho_hat, the C1 verdicts are
    re-evaluated and the worse governs, and the paper states that TR 37.885 depends on QuaDRiGa's snapshot placement. Urban rho
    applies to P1 (incl. P1d09) and P3, highway rho to P2, P4 takes the larger.
Outputs: runs/s_place_<tag>/<cell>_site<k>.json (resumable per site), results/s_place_<tag>.json.
Usage: python pipeline/snapshot_placement.py --tag sens_place [--sites 0 10] [--cells P1 P2]"""
import argparse
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
import sumo_kinematics as sk  # noqa: E402
from fdoa.estimator import fdoa_from_samples  # noqa: E402
from fdoa.geometry import FC, LAMBDA, Site  # noqa: E402
from quadriga.driver import C_MAX, HEAD_Z, RAYS, VEH_Z, PassInput, build_pass, run_site, segment_starts  # noqa: E402
from pipeline.stage2_evaluate import HEAD_PPB, one_stat  # noqa: E402
from pipeline.stage1_channels import C  # noqa: E402
from pipeline.provenance import check_tag, provenance, rng_for, same_provenance  # noqa: E402

SP = C["sensitivities"]["S_place"]
STAT = "d" if C["evaluation_v4"]["lever_d"]["adopted"] else "ls"
CELLS = {c: dict(cfg=v["cfg"], v0=float(v["v0"]), scenario=v["scenario"], idx=v["idx"]) for c, v in SP["cells"].items()}
Q, PRB, WIN = C["run"]["Q"], SP["prb"], C["pipeline_constants_from_code"]["B_WINDOW"]
LAT = C["run"]["lateral_offset_m"]
BURST_S, WIN_ARC = C["window_rule"]["burst_s"], C["window_rule"]["arc_L"]


def augment(p):
    """Beacon snapshots + uniform-in-distance snapshots (spacing lambda / 2.4, QuaDRiGa's sample density 1.2) over the
    same trajectory; returns the new PassInput and the indices of the beacon snapshots (term off: nu = 0)."""
    tt = p.traj.t
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(p.traj.x), np.diff(p.traj.y)))])
    tu = np.interp(np.arange(s[0], s[-1], LAMBDA / 2.4), s, tt)
    tu = tu[(tu > p.t_snap[0]) & (tu < p.t_snap[-1])]
    t_all = np.unique(np.concatenate([p.t_snap, tu]))
    idx = np.searchsorted(t_all, p.t_snap)
    assert np.allclose(t_all[idx], p.t_snap)
    x, _, _, _ = p.traj.at(t_all)
    seg = segment_starts(x)
    return PassInput(p.traj, p.t_beacon, t_all, seg, np.zeros((2, len(seg), C_MAX, RAYS)), p.z), idx


def site_passes(cell, site, n_pass=None):
    n_pass = SP["passes"] if n_pass is None else n_pass
    cf = CELLS[cell]
    base = [SP["seed"], cf["idx"], site]
    rs = np.random.default_rng(base)
    geo = Site(D=C["geometry"]["L_m"], pole_offset=float(rs.uniform(*C["geometry"]["pole_offset_m_uniform"])),
               dh=HEAD_Z - VEH_Z)
    scen = sk.URBAN if cf["scenario"] == "urban" else sk.HIGHWAY
    trajs, _ = sk.site_passes(np.random.default_rng(base + [1]), geo, cf["v0"], n_pass + 1, scen, background=True)
    if len(trajs) < n_pass + 1:
        raise RuntimeError(f"S-place {cell} site {site}: only {len(trajs)} SUMO passes")
    passes = []
    for k, T in enumerate(trajs):
        T.y = T.y + np.random.default_rng(base + [k, 3]).uniform(-LAT, LAT)
        passes.append(build_pass(np.random.default_rng(base + [k, 0]), geo, cf["v0"], Q, 0.0, traj=T,
                                 rng_schedule=np.random.default_rng(base + [k, 1])))
    return geo, passes[:-1], passes[-1]                  # the last pass is the dummy of the null call


def b_hats(res, passes, geo, idx_map, cell, site):
    """Per pass and mapping (prof, match, included) through the stage-2 chain with common random numbers."""
    imp = copy.deepcopy(C["impairments"])
    out = {"e1": [], "lb": []}
    for i, p in enumerate(passes):
        idx = idx_map[i]
        t_b = p.t_beacon
        x, y, vx, _ = p.traj.at(t_b)
        heads = geo.heads()
        r = np.linalg.norm(heads[:, None, :] - np.stack([x, y, np.zeros_like(x)], -1)[None], axis=-1)
        u_x = (heads[:, 0][:, None] - x[None, :]) / r
        rho_true = np.hypot(geo.pole_offset - y, geo.dh)
        inw = (x >= WIN_ARC[0] * geo.D) & (x <= WIN_ARC[1] * geo.D)
        t_window = float(t_b[inw][-1] - t_b[inw][0]) if inw.sum() >= 2 else float("inf")
        sig = np.stack([itf.interp_q(itf.band_avg_q(res[(i, h)]["off"][:, idx], res[(i, h)]["dl"][:, idx], Q, PRB)) for h in range(2)])
        gain = np.stack([itf.beacon_gain(res[(i, h)]["ls"][idx], Q) for h in range(2)])
        off = np.random.default_rng([SP["seed"], CELLS[cell]["idx"], site, i, 5]).uniform(-1, 1) * HEAD_PPB * 1e-9 * FC
        for mp in ("e1", "lb"):
            snr = itf.snr_e1_db(r) if mp == "e1" else itf.snr_link_budget_db(gain)
            rf = np.random.default_rng([SP["seed"], CELLS[cell]["idx"], site, i, 6 if mp == "e1" else 7])
            yb = itf.finish(rf, sig, snr, gain=gain, head_offset_hz=(off / 2, -off / 2), t_b=t_b)
            yb = itf.apply_impairments(np.random.default_rng([SP["seed"], CELLS[cell]["idx"], site, i, 8]), yb, t_b, u_x, imp)
            st = one_stat(t_b, fdoa_from_samples(yb, 1e-3), vx, x, rho_true, geo)
            ok = bool(t_window <= BURST_S and np.isfinite(st["match"]) and np.isfinite(st["prof"]))
            ok_d = bool(t_window <= BURST_S and np.isfinite(st["match_d"]) and np.isfinite(st["prof_d"]))
            out[mp].append([float(st["prof"]), float(st["match"]), ok, float(st["prof_d"]), float(st["match_d"]), ok_d])
    return out


def use_statistic(calls, stat=None):
    """[prof, match, included] of the statistic in force: plain LS, or the Huber fit (d) if constants
    evaluation_v4.lever_d.adopted (the reader rule of stage3_criteria.use_statistic)."""
    stat = stat or STAT
    i = 0 if stat == "ls" else 3
    return {c: {mp: [x[i:i + 3] for x in v] for mp, v in d.items()} for c, d in calls.items()}


def run_site_triple(cell, site, out_dir, tag, n_pass=None):
    path = os.path.join(out_dir, f"{cell}_site{site:03d}.json")
    prov = provenance(tag, stage=2, cell=cell, site=site, n_pass=n_pass)
    if os.path.exists(path):
        if not same_provenance(json.load(open(path)).get("provenance", {}), prov):
            raise RuntimeError(f"{path} exists with a different provenance: not reused")
        return path
    geo, passes, dummy = site_passes(cell, site, n_pass)
    aug = [augment(p) for p in passes]
    seed = int((SP["seed"] + 1000 * CELLS[cell]["idx"] + site) % 10 ** 6)
    cfg = CELLS[cell]["cfg"]
    own = {i: np.arange(p.t_snap.size) for i, p in enumerate(passes)}
    calls = dict(B=b_hats(run_site(cfg, geo, passes, seed), passes, geo, own, cell, site))
    calls["A"] = b_hats(run_site(cfg, geo, [a for a, _ in aug], seed), passes, geo, {i: aug[i][1] for i in range(len(aug))}, cell, site)
    calls["N"] = b_hats(run_site(cfg, geo, passes + [dummy], seed), passes, geo, own, cell, site)
    json.dump(dict(cell=cell, site=site, calls=calls, snapshots=dict(B=int(sum(p.t_snap.size for p in passes)),
                                                                     A=int(sum(a.t_snap.size for a, _ in aug))),
                   provenance=prov), open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return path


def sigma_w(per_site, sites, key_of, ok_of):
    """sd of the within-site-centred, +-25 % clipped select statistic over the passes in ok_of[s] (the passes included
    in all three calls)."""
    e = []
    for s in sites:
        v = np.array([x[key_of[s]] for x in per_site[s]], float)[ok_of[s]]
        e.append(np.clip(v - v.mean(), -WIN, WIN))
    return float(np.std(np.concatenate(e))) if e else np.nan


def summarise(out_dir, tag):
    res = dict(tag=tag, provenance=provenance(tag, stage=2), rule=SP["rule_v4"], statistic=STAT, cells={})
    for cell in CELLS:
        recs = {}
        for f in sorted(os.listdir(out_dir)):
            if f.startswith(f"{cell}_site") and f.endswith(".json"):
                d = json.load(open(os.path.join(out_dir, f)))
                recs[d["site"]] = use_statistic(d["calls"])
        if not recs:
            continue
        row = dict(n_sites_run=len(recs))
        for mp in ("e1", "lb"):
            key_of, ok_of = {}, {}
            for s in sorted(recs):                           # common included set across B, A, N
                ok = np.logical_and.reduce([np.array([x[2] for x in recs[s][c][mp]], bool) for c in ("B", "A", "N")])
                if ok.sum() < 2:
                    continue                                 # a site enters all three calls or none
                b = recs[s]["B"][mp]                         # select on the production (B) call, kept for A and N
                pr = np.array([x[0] for x in b], float)[ok]; ma = np.array([x[1] for x in b], float)[ok]
                key_of[s], ok_of[s] = (1 if np.var(ma) <= np.var(pr) else 0), ok
            sites = sorted(key_of)
            per = {c: {s: recs[s][c][mp] for s in sites} for c in ("B", "A", "N")}
            sb, sa, sn = (sigma_w(per[c], sites, key_of, ok_of) for c in ("B", "A", "N"))
            rng = rng_for("S_place", cell, mp)
            boot_a, boot_n = [], []
            with np.errstate(invalid="ignore", divide="ignore"):
                for _ in range(C["run"]["boot"]):
                    ss = [sites[i] for i in rng.integers(0, len(sites), len(sites))]
                    b0 = sigma_w(per["B"], ss, key_of, ok_of)
                    boot_a.append(sigma_w(per["A"], ss, key_of, ok_of) / b0)
                    boot_n.append(sigma_w(per["N"], ss, key_of, ok_of) / b0)
            ci = lambda x: [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]
            nf = lambda x: float(np.mean(~np.isfinite(np.asarray(x, float))))
            rho, rci = sa / sb, ci(boot_a)
            row[mp] = dict(n_sites=len(sites), passes=int(sum(ok_of[s].sum() for s in sites)), sigma_w=dict(B=sb, A=sa, N=sn),
                           rho=rho, rho_ci=rci, rho_ci_nonfinite_share=nf(boot_a), rho_null=sn / sb, rho_null_ci=ci(boot_n),
                           correct=bool(rho > SP["threshold"]), correct_v3_rule=bool(rho > SP["threshold"] and rci[0] > 1.0))
        trig = [row[mp]["rho"] for mp in ("e1", "lb") if mp in row and row[mp]["correct"]]
        row["decision"] = dict(correct=bool(trig), factor=float(max(trig)) if trig else 1.0,
                               statement="G1 depends on QuaDRiGa's snapshot placement" if trig else None)
        res["cells"][cell] = row
    res["stratum_factor"] = dict(urban=res["cells"].get("P1", {}).get("decision", {}).get("factor", 1.0),
                                 highway=res["cells"].get("P2", {}).get("decision", {}).get("factor", 1.0))
    os.makedirs(os.path.join(ER, "..", "results"), exist_ok=True)
    path = os.path.join(ER, "..", "results", f"s_place_{tag}.json")
    json.dump(res, open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)                     # atomic
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="sens_place (pre-specified) or dev_<x>")
    ap.add_argument("--sites", type=int, nargs=2, default=[0, SP["sites"]])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--n-passes", type=int, default=None, help="development only")
    a = ap.parse_args()
    if a.n_passes is not None and not a.tag.startswith("dev_"):
        raise ValueError("--n-passes is development only")
    check_tag(a.tag, stage=2)
    if a.tag == "sens_place" and (a.sites != [0, SP["sites"]] or sorted(a.cells) != sorted(CELLS)):
        raise ValueError(f"sens_place: all cells, sites 0-{SP['sites'] - 1}")
    if a.tag.startswith("sens_") and a.tag != "sens_place":
        raise ValueError("the pre-specified S-place tag is sens_place")
    out_dir = os.path.join(ER, "runs", f"s_place_{a.tag}")
    os.makedirs(out_dir, exist_ok=True)
    for cell in a.cells:
        for s in range(*a.sites):
            print(run_site_triple(cell, s, out_dir, a.tag, a.n_passes), flush=True)
    print(summarise(out_dir, a.tag))


if __name__ == "__main__":
    main()
