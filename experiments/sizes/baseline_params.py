"""Parameter sets of the baselines B1 (single-key BFV) and B3 (single-key CKKS), with the same OpenFHE calls as
experiments/timings/bench_risk_path.py (_bfv_single_key; part_baselines, B3): ring dimension, log2 Q and the requested
security level. Parameters only, no timing (hardware-independent). The ring dimensions must equal those recorded by
the timing run (experiments/timings/results/baselines_bfv_ckks.json).

Run (from the repository root):  python experiments/sizes/baseline_params.py
Output: experiments/sizes/results/baseline_params.json"""
import importlib.metadata
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import openfhe as o  # noqa: E402
from sp2ubi.params import DEFAULT as P  # noqa: E402

TIMINGS = os.path.join(ROOT, "experiments", "timings", "results")
baselines = json.load(open(os.path.join(TIMINGS, "baselines_bfv_ckks.json")))["result"]["baselines"]
risk_context = json.load(open(os.path.join(TIMINGS, "risk_path_budget.json")))["result"]["risk_context"]

prm = o.CCParamsBFVRNS()                     # B1, as bench_risk_path._bfv_single_key
prm.SetPlaintextModulus(P.p)
prm.SetMultiplicativeDepth(1)
prm.SetSecurityLevel(o.HEStd_128_classic)
b1 = o.GenCryptoContext(prm)

prm = o.CCParamsCKKSRNS()                    # B3, as bench_risk_path.part_baselines
prm.SetMultiplicativeDepth(1)
prm.SetScalingModSize(50)
prm.SetFirstModSize(60)
prm.SetBatchSize(64)
prm.SetSecurityLevel(o.HEStd_128_classic)
b3 = o.GenCryptoContext(prm)

out = {
    "openfhe": importlib.metadata.version("openfhe"),
    "source": "experiments/timings/bench_risk_path.py (_bfv_single_key; part_baselines B3), same calls",
    "B1_single_key_bfv": {"N": b1.GetRingDimension(), "log2_Q": round(math.log2(b1.GetModulus()), 2),
                          "security_level": "HEStd_128_classic", "plaintext_modulus_bits": P.p.bit_length()},
    "B3_ckks": {"N": b3.GetRingDimension(), "log2_Q": round(math.log2(b3.GetModulus()), 2),
                "security_level": "HEStd_128_classic"},
    "SP2UBI_bfv_from_risk_path_budget": {"N": risk_context["N"], "log2_Q": risk_context["log2_Q"],
                                         "security_level": risk_context["security"]},
}
assert out["B1_single_key_bfv"]["N"] == baselines["B1_single_key_bfv_encrypted_weights"]["ring_dimension"], "B1 N differs"
assert out["B3_ckks"]["N"] == baselines["B3_ckks_encrypted_weights_score_only"]["ring_dimension"], "B3 N differs"
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "baseline_params.json"), "w") as fh:
    json.dump(out, fh, indent=2)
    fh.write("\n")
print(json.dumps(out, indent=2))
