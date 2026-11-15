"""Study 3: snapshot-placement check on the sites 40-44 (cell P1-15, 20 traffic crossings per site, three QuaDRiGa
calls per site: B beacon-only, A beacon + uniform snapshots, N null) through pipeline/snapshot_placement.py
(run_site_triple), with its module inputs replaced for the call, as pipeline/s2_snapshot_placement.py does: the site
seed base is the U3 site seed and cell P1 carries the U3 family index. Two additions:
  * the per-beacon series of every crossing, call and mapping (air times, FDOA from the periodogram peaks and from the
    pulse pair of the cross-head product, speed, position) are written to <out>/P1_site<k>_series.npz, so that the
    urban estimator is evaluated on the samples of the base estimator's statistic;
  * the head-to-head offset is one value per site, as in stage 2, instead of snapshot_placement.py's value per
    crossing, because the urban estimator takes the offset as a site constant. It is the same in the three calls, so
    the ratio remains a paired comparison.
Record order inside one site: calls B, A, N; per call the crossings in order; per crossing the mappings e1, lb.
Output: <out>/P1_site<k>.json (statistics of the base estimator, as snapshot_placement.py) and
<out>/P1_site<k>_series.npz; pipeline/s3_placement_factor.py reads both.
Usage: python pipeline/s3_snapshot_placement.py [--sites 40 45] [--out runs/s_place_dev_s3_place]
       python pipeline/s3_snapshot_placement.py --sites 90 91 --n-passes 2 --out <dir>     (smoke test)"""
import argparse
import os
import sys
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from fdoa.geometry import FC  # noqa: E402
from fdoa.urban import pulse_pair  # noqa: E402
from pipeline import s3_family as FAM  # noqa: E402
from pipeline import snapshot_placement as SPL  # noqa: E402


@contextmanager
def patched(state):
    """snapshot_placement.py with the U3 seed and family index, one head offset per site, and its FDOA and statistic
    functions wrapped so that the per-beacon series of every record are collected in state["series"]; restored."""
    G = FAM.apply()
    seed = FAM.seeds()["G1_site_base"]
    U2 = G.FAMILIES["U2"]
    old = SPL.SP, SPL.CELLS, SPL.fdoa_from_samples, SPL.one_stat, SPL.itf.finish
    fdoa0, stat0, finish0 = old[2:]

    def finish(rf, sig, snr, gain=None, head_offset_hz=(0.0, 0.0), t_b=None, **kw):
        off = np.random.default_rng([seed, FAM.IDX, state["site"], 0, 5]).uniform(-1, 1) * SPL.HEAD_PPB * 1e-9 * FC
        return finish0(rf, sig, snr, gain=gain, head_offset_hz=(off / 2, -off / 2), t_b=t_b, **kw)

    def fdoa_from_samples(yb, packet_s):
        state["fpp"] = pulse_pair(yb[0] * np.conj(yb[1]), 1, packet_s)[0]
        return fdoa0(yb, packet_s)

    def one_stat(t_b, fd, vx, x, rho_true, site):
        n = state["n"]
        state["series"].update({f"r{n}_t": t_b, f"r{n}_fd": fd.astype(np.float32), f"r{n}_fpp": state["fpp"].astype(np.float32),
                                f"r{n}_v": vx.astype(np.float32), f"r{n}_x": x.astype(np.float32)})
        state["n"], state["pole"] = n + 1, float(site.pole_offset)
        return stat0(t_b, fd, vx, x, rho_true, site)

    SPL.SP = dict(SPL.SP, seed=seed, passes=FAM.PLACE_PASSES)
    SPL.CELLS = {"P1": dict(cfg=U2["cfg"], v0=15.0, scenario="urban", idx=FAM.IDX)}
    SPL.fdoa_from_samples, SPL.one_stat, SPL.itf.finish = fdoa_from_samples, one_stat, finish
    try:
        yield SPL
    finally:
        SPL.SP, SPL.CELLS, SPL.fdoa_from_samples, SPL.one_stat, SPL.itf.finish = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, nargs=2, default=list(FAM.PLACE_SITES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-passes", type=int, default=None, help="smoke test only (sites outside 30-44)")
    a = ap.parse_args()
    smoke = a.n_passes is not None
    if not smoke and not (FAM.PLACE_SITES[0] <= a.sites[0] and a.sites[1] <= FAM.PLACE_SITES[1]):
        raise SystemExit(f"placement sites are {FAM.PLACE_SITES[0]}-{FAM.PLACE_SITES[1] - 1}")
    if smoke and a.sites[0] < FAM.PLACE_SITES[1] and a.sites[1] > FAM.SITES[0]:
        raise SystemExit("a smoke test must not draw a site of the evaluation")
    tag = FAM.PLACE_TAG + "_smoke" if smoke else FAM.PLACE_TAG
    out_dir = a.out or os.path.join(ER, "runs", f"s_place_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    with patched(state) as M:
        for s in range(*a.sites):
            ser = os.path.join(out_dir, f"P1_site{s:03d}_series.npz")
            js = os.path.join(out_dir, f"P1_site{s:03d}.json")
            if os.path.exists(js) and not os.path.exists(ser):
                os.remove(js)                            # interrupted between the two files: recompute the site
            if os.path.exists(js):
                print(js, "skipped (exists)", flush=True)
                continue
            state.update(site=s, series={}, n=0, pole=np.nan)
            M.run_site_triple("P1", s, out_dir, tag, a.n_passes)
            np.savez_compressed(ser + ".tmp.npz", n_records=state["n"], pole_offset=state["pole"], **state["series"])
            os.replace(ser + ".tmp.npz", ser)
            print(js, "done", flush=True)


if __name__ == "__main__":
    main()
