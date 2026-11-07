"""TR 37.885 driver: builds the inputs of one site for quadriga/site.m (geometry, trajectories, snapshot times, segment
starts, scatterer-mobility draws), runs QuaDRiGa (patched tree) in Octave and returns the per-pass, per-head merged
cluster coefficients (term off and term on) at the snapshot instants. Constants: config/study1_constants.json.
Snapshots: Q per beacon, centred in the 1 ms beacon (the interface interpolates to M = 128 samples)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
import antenna  # noqa: E402
import scatterer_mobility as sm  # noqa: E402
from fdoa.geometry import LAMBDA, Site, make_trajectory  # noqa: E402

VEH_Z, HEAD_Z = 1.6, 5.0
C_MAX, RAYS = 64, 20                  # cluster bound incl. QuaDRiGa sub-clusters (LOS smoke: 35 paths; NLOSv more); quadriga/site.m errors if exceeded


@dataclass
class PassInput:
    traj: object
    t_beacon: np.ndarray        # beacon centre times (s)
    t_snap: np.ndarray          # snapshot times (s), Q per beacon
    seg_start: np.ndarray       # 1-based snapshot index of each segment start
    nu: np.ndarray              # [n_tx, n_seg, C_MAX, RAYS] extra Doppler (Hz), independent per link (head)
    z: float = 1.6              # vehicle antenna height (m): type 2 roof 1.6, type 3 truck roof 3.0


QUADRIGA_PATH = os.environ.get("QUADRIGA_PATCHED_DIR", os.path.join(os.path.dirname(os.path.dirname(HERE)), json.load(open(
    os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(os.path.dirname(HERE), "config", "study1_constants.json"))))[
    "quadriga"]["patched_path"]))   # default: speedcheck_sim/quadriga_patched

def beacon_schedule(rng, traj, beacon_hz=10.0, packet_s=1e-3):
    """Same rule as fdoa.channel.simulate_pass: 10 Hz from t0 + 0.05 s with a uniform random phase."""
    t_b = np.arange(traj.t[0] + 0.05, traj.t[-1] - 0.05, 1.0 / beacon_hz)
    t_b = t_b + rng.uniform(0, 1.0 / beacon_hz)
    return t_b[t_b < traj.t[-1] - packet_s]


def snapshot_times(t_b, Q, packet_s=1e-3):
    tau = (np.arange(Q) + 0.5) / Q * packet_s - packet_s / 2
    return (t_b[:, None] + tau[None, :]).ravel()


def segment_starts(x, seg_m=20.0):
    d = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(x)))])
    starts = [0]
    for k in range(1, int(d[-1] // seg_m) + 1):
        starts.append(int(np.searchsorted(d, k * seg_m)))
    starts = sorted(set(s for s in starts if s < len(x) - 2))
    return np.array(starts) + 1


def build_pass(rng, site: Site, v0, Q, v_scatt, alpha="uniform", granularity="ray", n_tx=2, traj=None, z=VEH_Z,
               rng_schedule=None):
    """traj: given trajectory (e.g. SUMO) or None; rng_schedule: generator of
    the beacon schedule (default rng); pass a copy of the in-house arm's schedule generator for pairing."""
    if traj is None:
        traj = make_trajectory(rng, site, v0, x_start=-0.35 * site.D, x_end=1.35 * site.D)
    t_b = beacon_schedule(rng if rng_schedule is None else rng_schedule, traj)
    t_s = snapshot_times(t_b, Q)
    x, _, _, _ = traj.at(t_s)
    seg = segment_starts(x)
    nu = np.stack([np.stack([sm.draw_extra_doppler(rng, C_MAX, RAYS, v_scatt, LAMBDA, alpha=alpha, granularity=granularity)
                             for _ in seg]) for _ in range(n_tx)])
    return PassInput(traj, t_b, t_s, seg, nu, float(z))


def run_site(cfg, site: Site, passes, seed, workdir=None, veh_el=None, tx_heads=None, ckpt_dir=None):
    """Run quadriga/site.m for one site; returns dict[(pass, head)] -> dict(off, on, dl, kf, pf, nray, ls). Head element: TR 37.885
    UE-type RSU (antenna.HEAD); vehicle: iso (primary) or veh_el (e.g. antenna.VEH_TYPE2, sensitivity). cfg: scenario name
    or one per transmitter (with tx_heads); passes must be built with n_tx = number of transmitters. ckpt_dir: per-builder
    checkpoint folder of quadriga/site.m, outside the work folder so that it survives a restart;
    None: no checkpoint."""
    wd = workdir or tempfile.mkdtemp(prefix="g1_")
    try:
        fin, fout = os.path.join(wd, "in.mat"), os.path.join(wd, "out.mat")
        n_tx = write_input(fin, cfg, site, passes, seed, veh_el, tx_heads)
        if ckpt_dir:
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_dir)), exist_ok=True)
        r = subprocess.run(octave_cmd(fin, fout, ckpt_dir), capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(fout):
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        return read_output(fout, len(passes), n_tx)
    finally:                                            # own work folder (in.mat + out.mat, up to ~2.5 GB at full size):
        if workdir is None:                             # removed on success and on any Python-level failure;
            shutil.rmtree(wd, ignore_errors=True)       # a SIGKILL leaves it for the next start to clean up


def write_input(fin, cfg, site: Site, passes, seed, veh_el=None, tx_heads=None):
    """The input .mat of quadriga/site.m; returns the number of transmitters."""
    heads = site.heads().T.copy()                       # in-house frame: vehicle at z = 0
    heads[2, :] = HEAD_Z
    if tx_heads is not None:                            # e.g. P3: [0, 1, 0] = head A (LOS), head B, head A (NLOSv)
        heads = heads[:, list(tx_heads)]
    pos = []
    for p in passes:
        x, y, _, _ = p.traj.at(p.t_snap)
        pos.append(np.vstack([x, y, np.full_like(x, p.z)]))
    obj = lambda L: np.array(L + [None], dtype=object)[:-1]   # cell arrays for Octave
    sio.savemat(fin, dict(cfg=cfg if isinstance(cfg, str) else obj(list(cfg)), heads=heads, pos=obj(pos), seg_start=obj([p.seg_start.astype(float) for p in passes]),
                          t_snap=obj([p.t_snap for p in passes]), nu=obj([p.nu for p in passes]), seed=float(seed),
                          head_el=antenna.octave_vector(antenna.HEAD), quadriga_path=QUADRIGA_PATH,
                          veh_el=antenna.octave_vector(veh_el) if veh_el else np.zeros((0, 0))))
    return heads.shape[1]


def octave_cmd(fin, fout, ckpt_dir=None):
    ck = f", '{ckpt_dir}'" if ckpt_dir else ""
    return ["octave", "--no-gui", "-q", "--eval", f"addpath('{HERE}'); g1_site('{fin}', '{fout}'{ck});"]


def read_output(fout, n_pass, n_tx):
    """dict[(pass, head)] of one quadriga/site.m output file (0-based keys)."""
    O = sio.loadmat(fout, squeeze_me=True)
    out = {}
    for p in range(1, n_pass + 1):
        for h in range(1, n_tx + 1):
            k = f"{p}_{h}"
            out[(p - 1, h - 1)] = dict(off=np.atleast_2d(O["off_" + k]), on=np.atleast_2d(O["on_" + k]),
                                       dl=np.atleast_2d(O["dl_" + k]), dlon=np.atleast_2d(O["dlon_" + k]),
                                       kf=np.atleast_1d(O["kf_" + k]),
                                       pf=np.atleast_2d(O["pf_" + k]), nray=int(O["nray_" + k]),
                                       ls=np.atleast_1d(O["ls_" + k]))
    del O
    return out


if __name__ == "__main__":                              # smoke test: 2 passes, Urban LOS, 15 m/s
    rng = np.random.default_rng(1)
    site = Site(D=100.0, pole_offset=float(rng.uniform(6, 12)), dh=HEAD_Z - VEH_Z)
    passes = [build_pass(rng, site, 15.0, Q=16, v_scatt=16.7) for _ in range(2)]
    import time
    t0 = time.time()
    res = run_site("3GPP_37.885_Urban_LOS", site, passes, seed=1)
    print("site run s", round(time.time() - t0, 1))
    for k, v in res.items():
        print(k, "paths x snaps", v["off"].shape, "kf seg dB", np.round(v["kf"], 2), "pf median", np.round(np.median(v["pf"][:, 1:]), 3),
              "snaps expected", passes[k[0]].t_snap.size)
