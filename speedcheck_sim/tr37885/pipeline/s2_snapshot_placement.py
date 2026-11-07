"""Study 2 S-place light (N5): the fixed study-1 sensitivity
pipeline/snapshot_placement.py (unchanged) run on the study-2 sites 20-24, cell P1-15, 20 passes per site, through its own
functions (run_site_triple, summarise), with two inputs replaced for the duration of the call:
  * its site seed base: snapshot_placement.py draws every site from [SP.seed, cell idx, site]; here [the study-2
    G1_site_base (evaluation_v6.seeds), the U2 family index, site], i.e. the study-2 site stream of
    stage1_channels.seeds("U2", site); the QuaDRiGa seed follows from it as in snapshot_placement.py;
  * its site-bootstrap seed: the study-2 'bootstrap' seed (s2_heldout.s2_bootstrap).
Everything else is snapshot_placement.py unchanged: three QuaDRiGa calls per site (B, A, N), term off, PRB 10, both mappings, nominal
impairments, the rule rho_hat > 1.10 -> factor (constants sensitivities.S_place.rule_v4), the statistic in force
(plain LS: the Huber fit (d) is not adopted; the development stage adds no estimator, s2_heldout.check_supported). The
study-2 factor enters stage 3 only through N5 (s2_criteria.s_place_factor: the larger of the study-1 and the study-2
factors). The study-2 seed base is one of the five derived study-2 seeds, none of which is snapshot_placement.py's own
seed.
Outputs (as snapshot_placement.py): runs/s_place_<tag>/P1_site<k>.json, results/s_place_<tag>.json, tag
s2_heldout.PLACE_TAG. No value is printed.
Usage: python pipeline/s2_snapshot_placement.py [--sites 20 25]"""
import argparse
import os
import sys
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import snapshot_placement as SPL  # noqa: E402
from pipeline import s2_constants as S2  # noqa: E402
from pipeline import s2_heldout as HO  # noqa: E402


def spec(C=None):
    """The study-2 S-place light design from the constants (evaluation_v6.s_place, evaluation_v6.seeds) and the U2
    family; refuses before the study-2 seeds are set."""
    from pipeline import stage1_channels as G
    C = C or HO.constants()
    ev6 = C.get("evaluation_v6") or {}
    sp = ev6.get("s_place") or {}
    if sp.get("sites") != list(S2.S_PLACE_SITES) or sp.get("passes") != S2.S_PLACE_PASSES or sp.get("cell") != "P1-15":
        raise RuntimeError(f"evaluation_v6.s_place {sp} is not the pre-specified S-place light (sites 20-24, 20 passes, P1-15)")
    seeds = ev6.get("seeds") or {}
    if "G1_site_base" not in seeds:
        raise RuntimeError("study-2 seeds missing (evaluation_v6.seeds): S-place light runs only after the seeds are set")
    F = G.FAMILIES["U2"]
    cell, v0 = sp["cell"].split("-")
    return dict(seed=int(seeds["G1_site_base"]), idx=int(F["idx"]), cell=cell, v0=float(v0), cfg=F["cfg"],
                scenario=F["scenario"], sites=(S2.S_PLACE_SITES[0], S2.S_PLACE_SITES[1] + 1), passes=S2.S_PLACE_PASSES)


@contextmanager
def patched(sp):
    """snapshot_placement.py's module inputs SP (seed, passes) and CELLS (P1 only, the U2 index) replaced for one call; restored."""
    old_sp, old_cells = SPL.SP, SPL.CELLS
    SPL.SP = dict(old_sp, seed=sp["seed"], passes=sp["passes"])
    SPL.CELLS = {sp["cell"]: dict(cfg=sp["cfg"], v0=sp["v0"], scenario=sp["scenario"], idx=sp["idx"])}
    try:
        yield SPL
    finally:
        SPL.SP, SPL.CELLS = old_sp, old_cells


def check_sites(sites, sp):
    lo, hi = sites
    if not (sp["sites"][0] <= lo < hi <= sp["sites"][1]):
        raise ValueError(f"S-place light sites {lo}-{hi - 1} outside the pre-specified {sp['sites'][0]}-{sp['sites'][1] - 1}")


def run(sites=None, tag=HO.PLACE_TAG, C=None, out_dir=None, summarise=True):
    """Run the pre-specified sites (resumable per site, as snapshot_placement.py) and write the summary; returns the summary path."""
    sp = spec(C)
    sites = tuple(sites or sp["sites"])
    check_sites(sites, sp)
    out_dir = out_dir or os.path.join(ER, "runs", f"s_place_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    with patched(sp) as M:
        for s in range(*sites):
            print(M.run_site_triple(sp["cell"], s, out_dir, tag), flush=True)
        if not summarise:
            return None
        with HO.s2_bootstrap(C):
            return M.summarise(out_dir, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, nargs=2, default=None)
    a = ap.parse_args()
    HO.guard()
    from pipeline.provenance import check_tag
    check_tag(HO.PLACE_TAG, stage=2)
    print(run(a.sites, summarise=a.sites is None))


if __name__ == "__main__":
    main()
