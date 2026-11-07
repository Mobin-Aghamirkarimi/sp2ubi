"""Lattice-estimator run for the risk-path BFV parameters, one process per (case, attack), so a slow attack cannot
block the others.
Usage: sage -python estimate_per_attack.sage.py <case> <attack>   -> results/per_attack/<case>__<attack>.json
Case: single_key_ternary (exact moduli of OpenFHE's context from params.json, uniform ternary secret, sigma 3.19).
Attacks: the attack set of LWE.estimate in malb/lattice-estimator: usvp, bdd, bdd_hybrid, bdd_mitm_hybrid, dual,
dual_hybrid, bkw, arora-gb.
Driver: run_per_attack.sh (time limit per job; a job that times out is recorded as such, never silently dropped)."""
import json, math, os, subprocess, sys
from estimator import LWE, ND
from sage.all import oo

case, attack = sys.argv[1], sys.argv[2]
cfg = json.load(open("params.json"))


def Q(ms):
    q = 1
    for m in ms:
        q *= m
    return q


cases = {"single_key_ternary": (cfg["single_key"], ND.Uniform(-1, 1))}
c, Xs = cases[case]
params = LWE.Parameters(n=c["N"], q=Q(c["moduli"]), Xs=Xs, Xe=ND.DiscreteGaussian(3.19), m=oo, tag=case)
fn = {"usvp": LWE.primal_usvp, "bdd": LWE.primal_bdd, "bdd_hybrid": LWE.primal_hybrid,
      "bdd_mitm_hybrid": lambda p: LWE.primal_hybrid(p, mitm=True, babai=True), "dual": LWE.dual,
      "dual_hybrid": LWE.dual_hybrid, "bkw": LWE.coded_bkw, "arora-gb": LWE.arora_gb}[attack]
try:
    r = fn(params)
    rop = r.get("rop", oo)
    val = None if rop == oo else round(float(math.log2(rop)), 1)
    out = {"case": case, "attack": attack, "log2_rop": val, "infeasible": bool(rop == oo),
           "detail": {k: str(v) for k, v in r.items()}}
except Exception as e:
    out = {"case": case, "attack": attack, "log2_rop": None, "error": repr(e)}
out["estimator_commit"] = subprocess.run(["git", "-C", os.environ["LATTICE_ESTIMATOR"], "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()
os.makedirs(os.path.join("results", "per_attack"), exist_ok=True)
json.dump(out, open(os.path.join("results", "per_attack", f"{case}__{attack}.json"), "w"), indent=1)
print(out)
