"""The speed-check simulation TR 37.885 runner, stage 1 (channel records). One QuaDRiGa call per site with every track of every cell that shares the
site (so a site is never split). Per track it stores
the compact channel record that every later stage uses (no new channel draws after this stage):
  hq      [n_prb, 2 (term off, on), n_tx, n_b, Q] complex64  band average over the occupied subcarriers at the Q
                                                              snapshots of each beacon (interface.band_avg_q)
  gain    [n_tx, n_b]  large-scale gain (sum of ray powers, beacon mean)       los_pow [n_tx, n_b] |LoS path|^2
  K_band_db [n_tx, n_prb] in-band K (LoS power over the band-averaged power of the other paths, term off); the paired
          in-house arm uses it per PRB; K_realised_db (broadband) is a diagnostic
  term on uses its own merged delays (dlon), term off the term-off delays
  kf, pf  builder K per segment and term-on power factors                        traj (0.01 s), t_b, seeds
P3 (family N) also stores the per-beacon NLOSv state and blockage loss at head A (nlosv_blockage.py).
Families (sites shared across speeds; constants `grid`; `evaluation_v4`; reduced design):
  U  P1 urban LOS 10, 15 m/s (30 reference + 60 test passes each)
  H  P2 highway LOS 20, 30, 38 m/s
  N  P3 urban: head A LOS/NLOSv switch, 15 m/s (test vehicle in lane 1, blockers in lane 0)
  P0U / P0H / P0N  the P0 development pilot (constants evaluation_v4.p0): P1 10 + 15, P2 20 + 30, P3 15,
           30 reference + 30 test passes, with the disjoint development seed set evaluation_v4.p0.seeds
  Not run: R (P4 38.901 UMi LOS GR), U9 (P1 at degree of
  saturation 0.9), 5U / 5H (P5), the P6 type-3 tracks and the C5 quadratic-control tracks (run.c5_quadratic_control
  families =).
D-qual: a family listed in constants evaluation_v4.p2_dqual runs its primary sites 5-9 with the reference
passes only (plan fixed before the run). The track list of a site is part of the pre-specified design (environment
dependence): it is fixed by the constants, never by a command-line option.
Output: runs/g1_<family>/site_<k>.npz (regenerable from the seeds), site_<k>.json (meta, provenance) and
manifest.json (SHA-256 of every file).
Usage: python pipeline/stage1_channels.py --family U --sites 0 20 --tag primary [--workers 4]
       (development: --tag dev_<x>)"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
import interface as itf  # noqa: E402
import nlosv_blockage as pb  # noqa: E402
import sumo_kinematics as sk  # noqa: E402
from pipeline.provenance import check_tag, diff_provenance, provenance, same_provenance, sha256_file  # noqa: E402
from fdoa.geometry import Site, make_trajectory  # noqa: E402
from quadriga.driver import HEAD_Z, VEH_Z, build_pass, run_site  # noqa: E402

C = json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study1_constants.json"))))
PRBS = tuple(C["beacon"]["prb"]["swept"])
TRUCK_Z = C["geometry"]["vehicle_antenna_m"]["type3_truck_roof"]
VS = C["scatterer_mobility"]["v_scatt_mps"]
LAT = C["run"]["lateral_offset_m"]
REF_TEST = (("ref", 30, False), ("test", 60, True))            # survey-like references on a free road, test in traffic
TEST = (("test", 60, True),)
KX_FIELDS = ("K_realised_db", "K_band_db", "K_coef_on_db", "K_coef_off_db", "K_band_on_db", "K_band_pbn_db", "K_seg_db", "kf")
REF_ONLY = (("ref", 30, False),)                               # D-qual qualification arm
P0 = C["evaluation_v4"]["p0"]
REF_TEST_P0 = (("ref", P0["ref"], False), ("test", P0["test"], True))
P5_ROLES = (("ref", 30, False), ("traffic", 150, True), ("eval", 100, True))   # the consistency simulation per site
FAMILIES = {
    "U": dict(idx=1, cfg="3GPP_37.885_Urban_LOS", scenario="urban", v_scatt=VS["urban"],
              tracks=[("P1", 10.0, REF_TEST, VEH_Z), ("P1", 15.0, REF_TEST, VEH_Z)]),
    "H": dict(idx=2, cfg="3GPP_37.885_Highway_LOS", scenario="highway", v_scatt=VS["highway"],
              tracks=[("P2", 20.0, REF_TEST, VEH_Z), ("P2", 30.0, REF_TEST, VEH_Z), ("P2", 38.0, REF_TEST, VEH_Z)]),
    "N": dict(idx=3, cfg=["3GPP_37.885_Urban_LOS", "3GPP_37.885_Urban_LOS", "3GPP_37.885_Urban_NLOSv"], tx_heads=[0, 1, 0],
              scenario="urban", v_scatt=VS["urban"], test_lane=1, tracks=[("P3", 15.0, REF_TEST, VEH_Z)]),
    "P0U": dict(idx=11, cfg="3GPP_37.885_Urban_LOS", scenario="urban", v_scatt=VS["urban"], seedset="p0",
                tracks=[("P1", 10.0, REF_TEST_P0, VEH_Z), ("P1", 15.0, REF_TEST_P0, VEH_Z)]),
    "P0H": dict(idx=12, cfg="3GPP_37.885_Highway_LOS", scenario="highway", v_scatt=VS["highway"], seedset="p0",
                tracks=[("P2", 20.0, REF_TEST_P0, VEH_Z), ("P2", 30.0, REF_TEST_P0, VEH_Z)]),
    "P0N": dict(idx=13, cfg=["3GPP_37.885_Urban_LOS", "3GPP_37.885_Urban_LOS", "3GPP_37.885_Urban_NLOSv"], tx_heads=[0, 1, 0],
                scenario="urban", v_scatt=VS["urban"], test_lane=1, seedset="p0", tracks=[("P3", 15.0, REF_TEST_P0, VEH_Z)]),
    "R": dict(idx=4, cfg="3GPP_38.901_UMi_LOS_GR", scenario="urban", v_scatt=0.0,
              tracks=[("P4", 10.0, REF_TEST, VEH_Z), ("P4", 20.0, REF_TEST, VEH_Z), ("P4", 30.0, REF_TEST, VEH_Z),
                      ("P6", 20.0, TEST, TRUCK_Z)]),
    "U9": dict(idx=7, cfg="3GPP_37.885_Urban_LOS", scenario="urban", v_scatt=VS["urban"], demand_vc=0.9,
               tracks=[("P1d09", 15.0, REF_TEST, VEH_Z)]),
    "5U": dict(idx=5, cfg="3GPP_37.885_Urban_LOS", scenario="urban", v_scatt=VS["urban"],
               tracks=[("P5", 15.0, P5_ROLES, VEH_Z)], until_qualified=6, stratum="urban"),
    "5H": dict(idx=6, cfg="3GPP_37.885_Highway_LOS", scenario="highway", v_scatt=VS["highway"],
               tracks=[("P5", 30.0, P5_ROLES, VEH_Z)], until_qualified=6, stratum="highway"),
}
# Study 2 (L-T survey; development and held-out families). Added families only: no study-1 family, seed or track
# list changes. U2 / N2: survey (reference) passes in traffic, REF_TEST_S2; H2: the study-1 free-road survey, REF_TEST, P2-20
# only. New indices, seedset "s2" (seed_base: evaluation_v6.seeds).
# D3U (Dev-3): a copy of every FAMILIES["U"] field (same index, seeds and tracks), plus the C5 quadratic tracks appended
# after the loop over all tracks (c5_after_loop).
REF_TEST_S2 = (("ref", 30, True), ("test", 60, True))         # L-T: survey passes in traffic (urban-type sites only)
FAMILIES.update({
    "U2": dict(idx=21, cfg="3GPP_37.885_Urban_LOS", scenario="urban", v_scatt=VS["urban"], seedset="s2",
               tracks=[("P1", 10.0, REF_TEST_S2, VEH_Z), ("P1", 15.0, REF_TEST_S2, VEH_Z)]),
    "N2": dict(idx=23, cfg=["3GPP_37.885_Urban_LOS", "3GPP_37.885_Urban_LOS", "3GPP_37.885_Urban_NLOSv"], tx_heads=[0, 1, 0],
               scenario="urban", v_scatt=VS["urban"], test_lane=1, seedset="s2", tracks=[("P3", 15.0, REF_TEST_S2, VEH_Z)]),
    "H2": dict(idx=22, cfg="3GPP_37.885_Highway_LOS", scenario="highway", v_scatt=VS["highway"], seedset="s2",
               tracks=[("P2", 20.0, REF_TEST, VEH_Z)]),
    "D3U": dict(FAMILIES["U"], c5_after_loop=True),
})
S2_SEEDSET = "s2"


def _s2_options():
    from pipeline import s2_constants as S2
    return dict(S2.OPTIONS, **((C.get("evaluation_v6") or {}).get("options") or {}))




def site_junction_offset(fam, sd):
    """N15, study 2 (seedset s2, option junction = "site"): one junction offset per site, U(50, 150) m from the
    dedicated stream default_rng(sd["site"] + [77]), shared by both roles and both speed classes; None otherwise. Only urban (junction) scenarios use it (H2: none)."""
    from pipeline import s2_constants as S2
    F = FAMILIES[fam]
    if F.get("seedset") != S2_SEEDSET or F["scenario"] != "urban" or _s2_options()["junction"] != "site":
        return None
    return float(np.random.default_rng(sd["site"] + [S2.N15_JUNCTION_STREAM]).uniform(*S2.N15_JUNCTION_RANGE_M))


class SharedJunctionRng:
    """Proxy of a track's SUMO generator for one site_passes call (study 2, N15): the junction draw of the call
    (sumo_kinematics.py:155, its first draw, rng.uniform(50.0, 150.0)) is consumed as usual and replaced by the site's offset;
    every other call goes to the generator, so sumo_kinematics.py (a study-1 stage-1 source) stays unchanged and the
    draw is consumed, not skipped."""

    def __init__(self, rng, u):
        self._r, self._u, self._used = rng, float(u), False

    def uniform(self, *a, **k):
        v = self._r.uniform(*a, **k)
        if not self._used:
            if tuple(a) != (50.0, 150.0) or k:
                raise RuntimeError("SharedJunctionRng: the first draw is not the junction draw of sumo_kin.site_passes")
            self._used = True
            return self._u
        return v

    def __getattr__(self, n):
        return getattr(self._r, n)


def seed_base(fam):
    """The family's seed set: production (constants seeds) or the disjoint P0 development set (evaluation_v4.p0.seeds);
    study 2 (seedset "s2"): the study-1 production seeds with the five bases of replaced by
    evaluation_v6.seeds."""
    if FAMILIES[fam].get("seedset") == S2_SEEDSET:
        return s2_seed_base()
    return P0["seeds"] if FAMILIES[fam].get("seedset") == "p0" else C["seeds"]


def s2_seed_base(consts=None):
    ev6 = (consts or C).get("evaluation_v6") or {}
    if not ev6.get("seeds"):
        raise RuntimeError("study-2 seeds are not set (evaluation_v6.seeds): they exist only after the evaluation design "
                           "record is timestamped (make_constants.py --study2-seeds)")
    return dict((consts or C)["seeds"], **ev6["seeds"])


def is_nlosv(fam):
    return fam in ("N", "P0N", "N2")


def c5_families():
    """C5 quadratic-control families: constants run.c5_quadratic_control.families plus the Dev-3 family of
    evaluation_v6.dev3.c5_families (absent outside study 2)."""
    ev6 = C.get("evaluation_v6") or {}
    return list(C["run"]["c5_quadratic_control"]["families"]) + list((ev6.get("dev3") or {}).get("c5_families", []))


def workers_stage1(fam):
    """Stage-1 workers of a family: constants run.workers_stage1 (every study-1 family, unchanged); the Dev-3 family D3U
    reads evaluation_v6.dev3.workers_stage1 (N14; run.* is not changed, so the study-1 constants stay unchanged)."""
    if fam in C["run"]["workers_stage1"]:
        return C["run"]["workers_stage1"][fam]
    from pipeline import s2_constants as S2
    if fam == S2.DEV3["family"]:
        return int(((C.get("evaluation_v6") or {}).get("dev3") or {})["workers_stage1"])
    raise KeyError(f"no stage-1 worker count for family {fam}")


def check_s2_site(fam, site):
    """Study-2 families run only on their pre-specified sites (H2 10-14; S-place
    light 20-24 on the study-2 seed base; U2 P1-15); Dev-3 only on study-1 U sites 0-4."""
    from pipeline import s2_constants as S2
    if fam in S2.S2_FAMILIES:
        lo, hi = S2.S2_FAMILIES[fam]["sites"]
        sp = fam == "U2" and S2.S_PLACE_SITES[0] <= site <= S2.S_PLACE_SITES[1]
        if not (lo <= site < hi or sp):
            raise ValueError(f"{fam}: site {site} is outside the pre-specified sites [{lo}, {hi})")
    if fam == S2.DEV3["family"] and not (S2.DEV3["sites"][0] <= site < S2.DEV3["sites"][1]):
        raise ValueError(f"{fam}: site {site} is outside the Dev-3 sites {S2.DEV3['sites']}")


def seeds(fam, site):
    f, sb = FAMILIES[fam]["idx"], seed_base(fam)
    return dict(site=[sb["G1_site_base"], f, site], sumo=[sb["sumo"], f, site],
                octave=int(sb["G1_site_base"] % 100000) * 100 + 1000 * f + site)


def track_rngs(fam, site, k):
    """Per-track generators: channel-side draws (nu, blockers) and the beacon schedule (shared with the in-house arm)."""
    f, sb = FAMILIES[fam]["idx"], seed_base(fam)
    return (np.random.default_rng([sb["G1_pass_base"], f, site, k, 0]),
            np.random.default_rng([sb["G1_pass_base"], f, site, k, 1]))


def site_tracks(fam, site):
    """The pre-specified track list of a site: the family's tracks, or for a D-qual family (constants
    evaluation_v4.p2_dqual) the reference passes only at sites 5-9."""
    F = FAMILIES[fam]
    plan = C["evaluation_v4"]["p0"]["D_qual"]["p2_site_plan"]
    if fam in C["evaluation_v4"].get("p2_dqual", []) and plan["ref_only"][0] <= site < plan["ref_only"][1]:
        return [(cell, v0, REF_ONLY, z) for cell, v0, _, z in F["tracks"]]
    return F["tracks"]


def build_site(fam, site, Q, n_scale=1.0):
    F = FAMILIES[fam]
    check_s2_site(fam, site)
    sd = seeds(fam, site)
    rs = np.random.default_rng(sd["site"])
    geo = Site(D=C["geometry"]["L_m"], pole_offset=float(rs.uniform(*C["geometry"]["pole_offset_m_uniform"])),
               dh=HEAD_Z - VEH_Z)
    scen = sk.URBAN if F["scenario"] == "urban" else sk.HIGHWAY
    n_tx = len(F["cfg"]) if isinstance(F["cfg"], list) else 2
    c5 = C["run"]["c5_quadratic_control"]
    c5_fams = c5_families()
    after = bool(F.get("c5_after_loop"))                   # study 2 Dev-3
    s2 = F.get("seedset") == S2_SEEDSET
    passes, meta = [], []
    k = 0
    deferred = []
    u_site = site_junction_offset(fam, sd)                  # None for every study-1 family and Dev-3
    for ti, (cell, v0, roles, z) in enumerate(site_tracks(fam, site)):
        r_sumo = np.random.default_rng(sd["sumo"] + [ti])
        trajs, role_of, role_idx, info = [], [], [], {}
        traffic = []
        for role, n, background in roles:
            n = max(1, int(round(n * n_scale)))                 # n_scale < 1: development smoke only
            kw = dict(demand_vc=F["demand_vc"]) if "demand_vc" in F else {}
            rr = SharedJunctionRng(r_sumo, u_site) if u_site is not None else r_sumo
            tt, inf = sk.site_passes(rr, geo, v0, n, scen, test_lane=F.get("test_lane"), background=background, **kw)
            if len(tt) < n:
                raise RuntimeError(f"{fam} site {site} {cell}@{v0} {role}: only {len(tt)} SUMO passes")
            trajs += tt
            role_of += [role] * n
            role_idx += list(range(n))                          # index within the role
            traffic += [bool(background)] * n
            info[role] = inf
        k_of = {}
        for j, T in enumerate(trajs):
            # lateral position inside the lane (constants run.lateral_offset_m): constant per pass
            T.y = T.y + np.random.default_rng([seed_base(fam)["G1_pass_base"], F["idx"], site, k, 3]).uniform(-LAT, LAT)
            if is_nlosv(fam):                                  # P3: the test vehicle stays in lane 1
                inw = (T.x >= -0.05 * geo.D) & (T.x <= 1.05 * geo.D)
                if not np.all(T.y[inw] < -1.75):
                    raise RuntimeError(f"N site {site} track {k}: vehicle leaves lane 1 inside the window")
            r_ch, r_sched = track_rngs(fam, site, k)
            passes.append(build_pass(r_ch, geo, v0, Q, F["v_scatt"], n_tx=n_tx, traj=T, z=z, rng_schedule=r_sched))
            mm = dict(k=k, cell=cell, v0=v0, z=z, kin="sumo", j=j, jr=role_idx[j], role=role_of[j],
                      sumo=info if j == 0 else None)
            if s2:                                          # study 2: whether the pass is in traffic (L-T survey)
                mm["traffic"] = traffic[j]
            meta.append(mm)
            k_of[j] = k
            k += 1
        if fam in c5_fams and cell in ("P1", "P2"):            # C5: paired quadratic kinematics, test passes only
            if after:                                       # Dev-3: appended after the loop over all tracks
                deferred.append((ti, cell, v0, z, trajs, role_of, role_idx, k_of))
            else:
                k = _c5_block(fam, site, Q, geo, F, n_tx, sd, c5, passes, meta, k, ti, cell, v0, z, trajs, role_of, role_idx,
                              k_of)
    for args in deferred:                                   # quadratic k >= the number of SUMO tracks of the site
        k = _c5_block(fam, site, Q, geo, F, n_tx, sd, c5, passes, meta, k, *args)
    return geo, passes, meta, sd


def _c5_block(fam, site, Q, geo, F, n_tx, sd, c5, passes, meta, k, ti, cell, v0, z, trajs, role_of, role_idx, k_of):
    """The C5 quadratic-control tracks of one track ti (unchanged; moved into a function so that Dev-3 can
    run it after the loop over all tracks). Returns the next track index k."""
    rq = np.random.default_rng(sd["sumo"] + [ti, 99])
    for j, T in enumerate(trajs):
        if role_of[j] not in c5["roles"]:
            continue
        tq = make_trajectory(rq, geo, v0, x_start=-0.35 * geo.D, x_end=1.35 * geo.D)
        tq.y[:] = T.y[0]                                    # same lane as the paired SUMO pass, no lane change
        tq.vy[:] = 0.0
        r_ch, r_sched = track_rngs(fam, site, k_of[j])      # the pair's seeds
        passes.append(build_pass(r_ch, geo, v0, Q, F["v_scatt"], n_tx=n_tx, traj=tq, z=z, rng_schedule=r_sched))
        meta.append(dict(k=k, cell=cell, v0=v0, z=z, kin="quadratic", j=j, jr=role_idx[j], role=role_of[j],
                         pair_k=k_of[j], sumo=None))
        k += 1
    return k


def ckpt_dir(out_dir, site):
    """Per-builder checkpoint folder of a site's Octave call: inside the family folder, not in the TMPDIR that the run
    script empties at each start; quadriga/site.m discards it when its meta (input, code, Octave)
    differs. Deleted here once the site record is written, not by quadriga/site.m after its save: the work folder with out.mat
    is emptied at a restart, so the checkpoints must outlive it. Its files are not named site_* (never in the manifest
    or a push)."""
    return os.path.join(os.path.abspath(out_dir), "ckpt", f"site_{site:03d}")


def site_record(fam, site, Q, out_dir, tag, n_scale=1.0):
    path = os.path.join(out_dir, f"site_{site:03d}.npz")
    prov = provenance(tag, stage=1, Q=Q, n_scale=n_scale, family=fam)
    if os.path.exists(path) and not os.path.exists(os.path.join(out_dir, f"site_{site:03d}.json")):
        os.remove(path)                                 # interrupted between the npz and its json: recompute the site
    if os.path.exists(path):
        old = json.load(open(os.path.join(out_dir, f"site_{site:03d}.json")))
        if not same_provenance(old.get("provenance", {}), prov):
            raise RuntimeError(f"{path} exists with a different provenance: not reused; "
                               f"{diff_provenance(old.get('provenance', {}), prov)}")
        shutil.rmtree(ckpt_dir(out_dir, site), ignore_errors=True)   # killed after its record, before the deletion
        return path, None
    t0 = time.time()
    F = FAMILIES[fam]
    geo, passes, meta, sd = build_site(fam, site, Q, n_scale)
    res = run_site(F["cfg"], geo, passes, seed=sd["octave"], tx_heads=F.get("tx_heads"), ckpt_dir=ckpt_dir(out_dir, site))
    n_tx = len(F["cfg"]) if isinstance(F["cfg"], list) else 2
    arrays = dict(pole_offset=geo.pole_offset, Q=Q, prbs=np.array(PRBS))
    for m, p in zip(meta, passes):
        k = m["k"]
        nb = p.t_beacon.size
        hq = np.zeros((len(PRBS), 2, n_tx, nb, Q), np.complex64)
        gain, losp, Kr = np.zeros((n_tx, nb)), np.zeros((n_tx, nb)), np.zeros(n_tx)
        Kb = np.zeros((n_tx, len(PRBS)))
        Kb_on, Kb_pbn = np.zeros((n_tx, len(PRBS))), np.zeros((n_tx, len(PRBS)))
        Kc_off, Kc_on, Kseg = np.zeros(n_tx), np.zeros(n_tx), np.full((n_tx, 64), np.nan)
        kdb = lambda a, b: 10 * np.log10(np.mean(a) / max(np.mean(b), 1e-30))
        kf, pf = [], []
        for h in range(n_tx):
            r = res[(k, h)]
            for i, prb in enumerate(PRBS):
                hq[i, 0, h] = itf.band_avg_q(r["off"], r["dl"], Q, prb)
                hq[i, 1, h] = itf.band_avg_q(r["on"], r["dlon"], Q, prb)   # term-on delays
                w = itf.band_weights(r["dl"][1:] - r["dl"][0:1], prb)        # in-band K, term off
                nlos = np.sum(r["off"][1:] * w, axis=0)
                Kb[h, i] = 10 * np.log10(np.mean(np.abs(r["off"][0]) ** 2) / max(np.mean(np.abs(nlos) ** 2), 1e-30))
            gain[h] = itf.beacon_gain(r["ls"], Q)
            losp[h] = (np.abs(r["off"][0]) ** 2).reshape(nb, Q).mean(axis=1)
            Kr[h] = 10 * np.log10(np.mean(losp[h]) / max(np.mean(gain[h] - losp[h]), 1e-30))
            # term-on K and K diagnostics; never used in a decision
            lo, lon = np.abs(r["off"][0]) ** 2, np.abs(r["on"][0]) ** 2
            Kc_off[h] = kdb(lo, np.sum(np.abs(r["off"][1:]) ** 2, axis=0))
            Kc_on[h] = kdb(lon, np.sum(np.abs(r["on"][1:]) ** 2, axis=0))
            for i, prb in enumerate(PRBS):
                w_on = itf.band_weights(r["dlon"][1:] - r["dlon"][0:1], prb)
                Kb_on[h, i] = kdb(lon, np.abs(np.sum(r["on"][1:] * w_on, axis=0)) ** 2)
                w = itf.band_weights(r["dl"][1:] - r["dl"][0:1], prb)
                Kb_pbn[h, i] = kdb(lo / r["ls"], np.abs(np.sum(r["off"][1:] * w, axis=0)) ** 2 / r["ls"])
            st = list(np.asarray(p.seg_start, int) - 1) + [lo.size]
            for sg in range(min(len(st) - 1, 64)):
                a_, b_ = st[sg], st[sg + 1]
                if b_ > a_:
                    Kseg[h, sg] = kdb(lo[a_:b_], np.sum(np.abs(r["off"][1:, a_:b_]) ** 2, axis=0))
            kf.append(r["kf"]); pf.append(r["pf"])
        pre = f"t{k}_"
        arrays.update({pre + "K_coef_off_db": Kc_off, pre + "K_coef_on_db": Kc_on, pre + "K_band_on_db": Kb_on,
                       pre + "K_band_pbn_db": Kb_pbn, pre + "K_seg_db": Kseg})
        arrays.update({pre + "hq": hq, pre + "gain": gain, pre + "los_pow": losp, pre + "K_realised_db": Kr, pre + "K_band_db": Kb,
                       pre + "los_slot_equal": np.array([bool(np.allclose(res[(k, h)]["dl"][0], res[(k, h)]["dlon"][0]))
                                                         for h in range(n_tx)]),
                       pre + "t_b": p.t_beacon, pre + "traj": np.vstack([p.traj.t, p.traj.x, p.traj.y, p.traj.vx, p.traj.vy]),
                       pre + "kf": np.array([np.pad(np.atleast_1d(a), (0, 64 - np.size(a)), constant_values=np.nan) for a in kf]),
                       pre + "pf_median": np.array([np.nanmedian(np.atleast_2d(a)[:, 1:]) for a in pf])})
        if is_nlosv(fam):
            if m["role"] == "ref" and not m.get("traffic", False):   # free-road survey passes: no blockers (N2's
                # survey passes are in traffic (L-T) and are drawn like test passes)
                arrays.update({pre + "nlosv_state": np.full(nb, -1), pre + "loss_db": np.zeros(nb),
                               pre + "blk_speed": np.nan, pre + "blk_clipped": False})
            else:
                r_blk = np.random.default_rng([seed_base(fam)["G1_pass_base"], F["idx"], site, k, 2])
                st = pb.pass_states(r_blk, geo, p.traj, p.t_beacon, HEAD_Z, p.z, y_c=0.0)
                arrays.update({pre + "nlosv_state": st["state"], pre + "loss_db": st["loss_db"],
                               pre + "blk_speed": st["speed"], pre + "blk_clipped": st["clipped"]})
    np.savez_compressed(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)
    kx = os.path.join(out_dir, f"site_{site:03d}.kx.json")   # the K-report fields only: stage 3
    json.dump({k: np.asarray(v).tolist() for k, v in arrays.items() if k.split("_", 1)[-1] in KX_FIELDS},   # runs from
              open(kx + ".tmp", "w"))                                                                       # the merged
    os.replace(kx + ".tmp", kx)                                                                             # small files
    import resource                                     # kernel high-water marks:
    peak = dict(octave_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,   # max over this worker's Octave calls
                python_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    jpath = os.path.join(out_dir, f"site_{site:03d}.json")
    json.dump(dict(meta=meta, seeds=sd, seedset=F.get("seedset", "production"), seconds=time.time() - t0, n_tracks=len(passes),
                   provenance=prov, peak_rss=peak,
                   npz_sha256=sha256_file(path), kx_sha256=sha256_file(kx)),
              open(jpath + ".tmp", "w"), indent=1, default=str)
    os.replace(jpath + ".tmp", jpath)                   # atomic: a power cut never leaves a truncated record
    shutil.rmtree(ckpt_dir(out_dir, site), ignore_errors=True)       # the record is complete: checkpoints no longer needed
    return path, time.time() - t0


def write_manifest(out_dir):
    files = []
    for root, _, fs in os.walk(out_dir):
        files += [os.path.relpath(os.path.join(root, f), out_dir) for f in fs
                  if f.startswith("site_") and (f.endswith(".npz") or f.endswith(".json"))]
    man = dict(files={f: sha256_file(os.path.join(out_dir, f)) for f in sorted(files)})
    mp = os.path.join(out_dir, "manifest.json")
    json.dump(man, open(mp + ".tmp", "w"), indent=1)
    os.replace(mp + ".tmp", mp)                         # atomic


def _job(args):
    fam, site, Q, out_dir, tag, ns = args
    return site_record(fam, site, Q, out_dir, tag, ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--sites", type=int, nargs=2, default=[0, 20])
    ap.add_argument("--tag", required=True, help="'primary' or dev_<x> / sens_<x>")
    ap.add_argument("--workers", type=int, default=None,
                    help="default and primary: constants run.workers_stage1[family] (from the measured peak memory)")
    ap.add_argument("--Q", type=int, default=C["run"]["Q"], help="development only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-scale", type=float, default=C["run"]["n_scale"], help="development only (fewer passes)")
    ap.add_argument("--governing", nargs=2, default=None, metavar=("MAPPING", "PRB"),
                    help="development only; primary P5 reads results/governing_primary.json")
    a = ap.parse_args()
    check_tag(a.tag, stage=1, Q=a.Q, n_scale=a.n_scale)
    if FAMILIES[a.family].get("seedset") == "p0" and a.tag != P0["tag"]:
        raise ValueError(f"{a.family}: pilot families run only under the tag {P0['tag']!r}")
    if a.tag == P0["tag"] and FAMILIES[a.family].get("seedset") != "p0":
        raise ValueError(f"tag {P0['tag']!r} is reserved for the pilot families")
    if a.tag == "primary":                              # full site ranges, default output
        if a.family not in C["evaluation_v4"]["p2_order"]:
            raise ValueError(f"primary: family {a.family} is not run")
        lo, hi = C["evaluation_v4"]["primary_sites"]      # no extension; a run takes its
        if not (lo <= a.sites[0] < a.sites[1] <= hi) or a.out:   # sites one by one in seed order
            raise ValueError(f"primary {a.family}: --sites within [{lo}, {hi}) and no --out")
    ws = workers_stage1(a.family)
    if a.workers is None:
        a.workers = ws
    elif a.tag == "primary" and a.workers != ws:
        raise ValueError(f"primary: {a.workers} workers differ from constants run.workers_stage1[{a.family}] = {ws}")
    out = a.out or os.path.join(ER, "runs", f"g1_{a.family}" + ("" if a.tag == "primary" else f"_{a.tag}"))
    os.makedirs(out, exist_ok=True)
    F = FAMILIES[a.family]
    if F.get("until_qualified"):
        select_until_qualified(a, out, F)
    else:
        jobs = [(a.family, s, a.Q, out, a.tag, a.n_scale) for s in range(*a.sites)]
        with ProcessPoolExecutor(a.workers) as ex:
            for path, dt in ex.map(_job, jobs):
                print(path, "skipped (exists, same provenance)" if dt is None else f"{dt:.0f} s", flush=True)
    write_manifest(out)


def p5_qualified(fam, site, out, mapping, prb, tag):
    """Site qualification of a P5 site on its 30 survey passes (gen_on, governing SNR mapping and PRB): sd of the
    select statistic <= 6 %."""
    from pipeline.stage2_evaluate import evaluate_site
    from pipeline.stage3_criteria import ref_qualification
    path = evaluate_site(fam, site, out, f"qual_{tag}_{mapping}_{prb}", mappings=(mapping,), prbs=[prb], stage1_tag=tag)
    recs = [r for r in json.load(open(path))["recs"] if r["role"] == "ref" and r["arm"] == "gen_on" and r["beta"] == 0.0]
    q = ref_qualification(recs)                         # the same rule as criteria
    return q["qualified"], q["sd_ref"]


def governing_for(stratum, tag):
    """The SNR mapping and PRB that govern the stratum's headline."""
    path = os.path.join(ER, "..", "results", f"governing_{tag}.json")
    g = json.load(open(path))[stratum]
    return g["mapping"], int(g["prb"])


def select_until_qualified(a, out, F):
    """P5 site selection (constants run.p5): layouts in seed order from index 0 until `need` qualify;
    batches of `workers` sites run in parallel; the selection is the first `need` qualified sites in seed order
    (independent of the batch size) and the number drawn is the index of the last selected site + 1, or every layout drawn when fewer than `need`
    qualify (a C3 figure).
    At most max_drawn layouts: if fewer than `need` qualify, P5 for this site type is "not applicable"."""
    p5 = C["run"]["p5"]
    need, cap = p5["need"], p5["max_drawn"]
    if not a.tag == "primary":                          # development: stay inside --sites
        cap = min(cap, a.sites[1] - a.sites[0])
    if a.tag == "primary":
        if a.sites[0] != p5["start_index"] or a.governing is not None:
            raise ValueError("primary P5: start index and governing combination are fixed")
        mapping, prb = governing_for(F["stratum"], a.tag)
    else:
        mapping, prb = (a.governing[0], int(a.governing[1])) if a.governing else governing_for(F["stratum"], a.tag)
    status, s = {}, a.sites[0]
    while sum(q for q, _ in status.values()) < need and s < a.sites[0] + cap:
        ks = [k for k in range(s, min(s + a.workers, a.sites[0] + cap))]
        jobs = [(a.family, k, a.Q, out, a.tag, a.n_scale) for k in ks]
        with ProcessPoolExecutor(a.workers) as ex:
            for path, dt in ex.map(_job, jobs):
                print(path, "skipped (exists, same provenance)" if dt is None else f"{dt:.0f} s", flush=True)
        for k in ks:
            status[k] = p5_qualified(a.family, k, out, mapping, prb, a.tag)
        s += len(ks)
    sel, drawn = [], 0
    for k in sorted(status):
        if status[k][0] and len(sel) < need:
            sel.append(k)
            drawn = k - a.sites[0] + 1
    if len(sel) < need:                                 # loop ran to the cap: every layout was drawn
        drawn = len(status)
    na = len(sel) < C["run"]["p5"]["min_report"]            # 3-5 qualified: reported on those (stated)
    json.dump(dict(governing=[mapping, prb], status={str(k): dict(qualified=bool(q), sd=sd) for k, (q, sd) in status.items()},
                   selected=sel, drawn=drawn if not na else len(status), not_applicable=na,
                   provenance=provenance(a.tag, Q=a.Q, n_scale=a.n_scale, family=a.family)),
              open(os.path.join(out, "selection.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(out, "selection.json.tmp"), os.path.join(out, "selection.json"))   # atomic
    print("selected", sel, "drawn", drawn, "NOT APPLICABLE" if na else "")


if __name__ == "__main__":
    main()
