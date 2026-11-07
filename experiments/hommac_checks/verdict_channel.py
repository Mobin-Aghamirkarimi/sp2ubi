"""The per-record HomMAC verdict as a test channel for a deviating TSP.

Numeric checks only, no timing. Plaintext model of the slot-wise affine operations that the TSP can apply to the BFV
ciphertexts with public keys alone (it holds pk_Ins and the evaluation keys): add an encryption of a vector to a slot
group, or multiply a slot by a plaintext constant, before or after the product with [[w]] and the rotate-and-sum. The
tag slots are left untouched, so the aggregated tags T_R, T'_R are the honest ones and only the score slot changes.

  V1. add delta != 0 to feature slot j of the record only:   accepted  iff  w_j * delta == 0 (mod p), i.e. w_j == 0
      (one rejected or accepted record per slot reveals the zero pattern of the weights);
  V2. multiply feature slot j of the record by c != 1:        accepted  iff  w_j * x_j == 0 (mod p)
      (whether feature j of this record is zero, for a slot with w_j != 0);
  V3. add delta to slot j and -c*delta to slot k:             accepted  iff  w_j == c * w_k (mod p)
      (an equality test of a weight ratio against a guessed c);
  V4. add an encryption of v to the feature slots of [[w]]:   accepted  iff  sum_i v_i x_i == 0 (mod p)
      (e.g. v = (-c, 0, 0, 0, 1, 0): whether feature 4 of the record equals the guess c).

In every case an accepted output carries the honest score sum_i w_i x_i, so it is not a win in the integrity game
(integrity is not affected); what the TSP gains is one predicate per altered record, and every altered record that is
rejected is a deviation that an honest TSP never produces. The script asserts both facts.

Uses the reference implementation sp2ubi/hommac.py and the fitted weights of models/so2021_additive_poisson.json.

Run (from the repository root):  python experiments/hommac_checks/verdict_channel.py
Writes results/verdict_channel.json next to this file.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from sp2ubi import hommac  # noqa: E402
from sp2ubi.params import DEFAULT  # noqa: E402
from sp2ubi.risk_model import load_model  # noqa: E402

p = DEFAULT.p
EPOCH = 5
N = 500
w = [int(v) % p for v in load_model(os.path.join(ROOT, "models", "so2021_additive_poisson.json")).w]
rng = random.Random(7)


def lin(a, b):
    return sum(x * y for x, y in zip(a, b)) % p


def verdict(score, T, T2, u, K):
    """The insurer's HomMAC check on a decrypted output whose tag slots are the honest aggregated tags."""
    return hommac.verify_record(score, lin(w, T), lin(w, T2), u, w, K, EPOCH, p)


def record():
    return [1] + [rng.choice([0, rng.randrange(1, 1 << 14)]) for _ in range(3)] \
        + [rng.choice([0, rng.randrange(1, 4096)]) for _ in range(2)]


res = {"p": p, "weights": w, "records_per_check": N,
       "V1_accepted_per_slot": {j: 0 for j in range(6)},
       "V2": {"agree_with_predicate": 0, "accepted": 0},
       "V3": {"correct_ratio_accepted": 0, "wrong_ratio_accepted": 0},
       "V4": {"predicate_true": [0, 0], "predicate_false": [0, 0]}}
accepted_scores_honest = True
for _ in range(N):
    K = rng.randbytes(32)
    u = rng.randrange(1, p)
    x = record()
    honest = lin(w, x)
    T, T2 = hommac.tag_blocks(x, K, EPOCH, u, p)
    assert verdict(honest, T, T2, u, K)
    # V1
    for j in range(6):
        x1 = list(x)
        x1[j] = (x1[j] + rng.randrange(1, p)) % p
        acc = verdict(lin(w, x1), T, T2, u, K)
        res["V1_accepted_per_slot"][j] += acc
        accepted_scores_honest &= (not acc) or lin(w, x1) == honest
    # V2
    j = rng.randrange(1, 6)
    c = rng.randrange(2, p)
    x2 = list(x)
    x2[j] = x2[j] * c % p
    acc = verdict(lin(w, x2), T, T2, u, K)
    res["V2"]["accepted"] += acc
    res["V2"]["agree_with_predicate"] += (acc == ((w[j] * x[j]) % p == 0))
    accepted_scores_honest &= (not acc) or lin(w, x2) == honest
    # V3 (slots 4, 5: the true ratio w_4 / w_5 and a wrong guess 3)
    d = rng.randrange(1, p)
    for cc, key in ((w[4] * pow(w[5], -1, p) % p, "correct_ratio_accepted"), (3, "wrong_ratio_accepted")):
        x3 = list(x)
        x3[4] = (x3[4] + d) % p
        x3[5] = (x3[5] - cc * d) % p
        acc = verdict(lin(w, x3), T, T2, u, K)
        res["V3"][key] += acc
        accepted_scores_honest &= (not acc) or lin(w, x3) == honest
    # V4: v added to the feature slots of [[w]]; guess whether x_4 equals c4
    c4 = rng.choice([0, x[4]])
    v = [(-c4) % p, 0, 0, 0, 1, 0]
    score4 = sum(((a + b) % p) * xi for a, b, xi in zip(w, v, x)) % p
    acc = verdict(score4, T, T2, u, K)
    truth = x[4] == c4
    res["V4"]["predicate_true" if truth else "predicate_false"][0 if acc else 1] += 1
    accepted_scores_honest &= (not acc) or score4 == honest

ok = (all(res["V1_accepted_per_slot"][j] == (N if w[j] == 0 else 0) for j in range(6))
      and res["V2"]["agree_with_predicate"] == N
      and res["V3"]["correct_ratio_accepted"] == N and res["V3"]["wrong_ratio_accepted"] == 0
      and res["V4"]["predicate_true"][1] == 0 and res["V4"]["predicate_false"][0] == 0
      and accepted_scores_honest)
res["V4_note"] = "[accepted, rejected] counts"
res["every_accepted_output_has_the_honest_score"] = accepted_scores_honest
res["all_predicates_exact"] = ok
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "verdict_channel.json"), "w") as f:
    json.dump(res, f, indent=2, sort_keys=True)
print(json.dumps(res, indent=2, sort_keys=True))
if not ok:
    sys.exit("verdict_channel.py: a predicate did not hold exactly")
print("verdict channel: every predicate exact; no accepted output differs from the honest score")
