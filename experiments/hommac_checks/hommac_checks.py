"""Numerical checks behind the HomMAC integrity theorem (acceptance rates against the bounds).

Numeric checks only, no timing. Uses the reference implementation sp2ubi/hommac.py (the code the protocol tests run)
and the parameters of sp2ubi/params.py.

  A. p: the 41-bit prime, p = 1 (mod 2^16), and the bound q/(p-1)^2 in bits for several q.
  B. Decoding: the largest 31-day epoch score of the fitted integer model stays below p/2 (Params.check_bounds).
  C. Correctness: honest records with the intercept slot x_0 verify in both blocks.
  D. An earlier tag construction (T_i = alpha_t x_i + u r_i with alpha_t, r_i fixed for the epoch; check
     T_R == alpha_t R + u F(r)): (D1) an honest evaluation of a model with an intercept b added by the circuit fails;
     (D2) the sum of two records of the epoch verifies as a record with nonce u_1 + u_2 (a forgery that needs no key);
     (D3) the same two attacks against the current scheme.
  E. Small primes (p = 17, 13): the attack classes of the integrity game against the current scheme with an adversary
     that sees x, T, T' and w in the clear, compared with the per-output bound 1/(p-1)^2 (and 1/p^2 for an untagged
     label): (E1) altered score with guessed tag shifts, (E2) injected record under an untagged nonce, (E3) record of one
     policy moved to another policy's key, (E4) adaptive attack on one record with q distinct guesses (bound q/(p-1)^2).
     Each empirical rate is reported with a 99.9 % Wilson interval; the check fails if the interval lies above the bound.

Run (from the repository root):  python experiments/hommac_checks/hommac_checks.py
Writes results/hommac_checks.json next to this file.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from sp2ubi import hommac  # noqa: E402
from sp2ubi.params import DEFAULT, Params, is_prime  # noqa: E402

Z = 3.2905  # two-sided 99.9 % normal quantile


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def lin(w, v, p):
    return sum(a * b for a, b in zip(w, v)) % p


out = {}
fail = []

# ------------------------------------------------------------------------------------------------ A. the prime and the bound
p = DEFAULT.p
out["A"] = {
    "p": p,
    "p_minus_2^40": p - (1 << 40),
    "bits": p.bit_length(),
    "is_prime": is_prime(p),
    "p_mod_2^16": p % (1 << 16),
    "smallest_prime_above_2^40_with_p=1_mod_2^16": all(not is_prime(k * (1 << 16) + 1)
                                                       for k in range((1 << 40) // (1 << 16) + 1, (p - 1) >> 16)),
    "log2_single_block_1_over_(p-1)": -math.log2(p - 1),
    "log2_two_blocks_1_over_(p-1)^2": -2 * math.log2(p - 1),
    "log2_bound_q_over_(p-1)^2": {f"q=2^{e}": e - 2 * math.log2(p - 1) for e in (0, 20, 30, 40)},
    "nonce_bits_param": DEFAULT.nonce_bits,
}
if not (out["A"]["is_prime"] and out["A"]["p_mod_2^16"] == 1 and out["A"]["bits"] == 41
        and out["A"]["smallest_prime_above_2^40_with_p=1_mod_2^16"]):
    fail.append("A: prime properties")

# ------------------------------------------------------------------------------------------------ B. decoding bound
from sp2ubi.risk_model import load_model  # noqa: E402
w_fit = list(load_model(os.path.join(ROOT, "models", "so2021_additive_poisson.json")).w)
prm = Params()
smin, smax = prm.score_bounds(w_fit, prm.max_epoch_days)
prm.check_bounds(w_fit)
out["B"] = {"weights": w_fit, "max_31_day_score": smax, "p_half": p // 2, "below_p_half": smax < p // 2,
            "weights_below_2^22": all(0 <= v < (1 << 22) for v in w_fit)}
if not out["B"]["below_p_half"]:
    fail.append("B: epoch score bound")

# ------------------------------------------------------------------------------------------------ C. correctness with intercept
rng = random.Random(20261030)
K = rng.randbytes(32)
n_ok, N_C = 0, 10000
for _ in range(N_C):
    x = [1] + [rng.randrange(1 << 14) for _ in range(3)] + [rng.randrange(4096) for _ in range(2)]
    u = rng.randrange(1, p)
    T, T2 = hommac.tag_blocks(x, K, 7, u, p)
    R, TR, TR2 = lin(w_fit, x, p), lin(w_fit, T, p), lin(w_fit, T2, p)
    n_ok += hommac.verify_record(R, TR, TR2, u, w_fit, K, 7, p)
out["C"] = {"records": N_C, "honest_records_verified_both_blocks": n_ok}
if n_ok != N_C:
    fail.append("C: honest records rejected")


# ------------------------------------------------------------------------------------------------ D. earlier tag construction
def old_keys(K, t, m, p):
    alpha = hommac.prf(K, p - 1, "old", t, 0) + 1
    return alpha, [hommac.prf(K, p, "old", t, i + 1) for i in range(m)]


def old_tags(x, K, t, u, p):
    a, r = old_keys(K, t, len(x), p)
    return [(a * xi + u * ri) % p for xi, ri in zip(x, r)]


def old_verify(R, TR, u, w, K, t, p, b=0):
    """Verification of the earlier tag construction with r_total = F(r); F has the intercept b (added by the circuit)
    when b != 0."""
    a, r = old_keys(K, t, len(w), p)
    return TR % p == (a * R + u * ((lin(w, r, p) + b) % p)) % p


N_D = 2000
w5 = w_fit[1:]                       # earlier layout: five features, intercept b added by the circuit
b = w_fit[0]
d1_fail = d2_ok = d3_old_int = d3_comb = 0
for _ in range(N_D):
    x1 = [rng.randrange(1 << 14) for _ in range(5)]
    x2 = [rng.randrange(1 << 14) for _ in range(5)]
    u1, u2 = rng.randrange(1, p), rng.randrange(1, p)
    T1, T2o = old_tags(x1, K, 3, u1, p), old_tags(x2, K, 3, u2, p)
    # D1: honest evaluation with intercept: R = b + w.x, T_R = b + w.T (the circuit adds b to both)
    d1_fail += not old_verify((b + lin(w5, x1, p)) % p, (b + lin(w5, T1, p)) % p, u1, w5, K, 3, p, b)
    # D2: sum of two records, claimed with nonce u1 + u2 (no intercept, so that D1 does not interfere)
    xs = [(a + c) for a, c in zip(x1, x2)]
    Ts = [(a + c) % p for a, c in zip(T1, T2o)]
    d2_ok += old_verify(lin(w5, xs, p), lin(w5, Ts, p), (u1 + u2) % p, w5, K, 3, p)
    # D3: the same against the current scheme (x_0 = 1 slot carries the intercept; sum claimed under u1 + u2)
    y1, y2 = [1] + x1, [1] + x2
    A1, A1b = hommac.tag_blocks(y1, K, 3, u1, p)
    A2, A2b = hommac.tag_blocks(y2, K, 3, u2, p)
    d3_old_int += hommac.verify_record(lin(w_fit, y1, p), lin(w_fit, A1, p), lin(w_fit, A1b, p), u1, w_fit, K, 3, p)
    ys = [a + c for a, c in zip(y1, y2)]
    us = (u1 + u2) % p or 1
    d3_comb += hommac.verify_record(lin(w_fit, ys, p), lin(w_fit, [(a + c) % p for a, c in zip(A1, A2)], p),
                                    lin(w_fit, [(a + c) % p for a, c in zip(A1b, A2b)], p), us, w_fit, K, 3, p)
out["D"] = {"trials": N_D,
            "D1_earlier_honest_with_intercept_rejected": d1_fail,
            "D2_earlier_sum_of_two_records_accepted": d2_ok,
            "D3_current_honest_with_intercept_accepted": d3_old_int,
            "D3_current_sum_of_two_records_accepted": d3_comb}
if d2_ok != N_D or d3_old_int != N_D or d3_comb != 0 or d1_fail < N_D - 2:
    fail.append("D: earlier vs current construction")


# ------------------------------------------------------------------------------------------------ E. small primes
def small_prime_attacks(ps, trials, q_adapt, seed):
    r = random.Random(seed)
    m = 4
    res = {"p": ps, "trials": trials}
    e1 = e2 = e3 = e4 = 0
    for _ in range(trials):
        KA, KB = r.randbytes(16), r.randbytes(16)
        w = [r.randrange(ps) for _ in range(m)]
        if not any(w):
            w[0] = 1
        x = [1] + [r.randrange(ps) for _ in range(m - 1)]
        u = r.randrange(1, ps)
        T, T2 = hommac.tag_blocks(x, KA, 1, u, ps)
        Rs, TR, TR2 = lin(w, x, ps), lin(w, T, ps), lin(w, T2, ps)
        # E1: altered score R* + delta, tag shifts g*delta, g'*delta with guesses g, g' (the optimal attack form)
        dlt = r.randrange(1, ps)
        g, g2 = r.randrange(1, ps), r.randrange(1, ps)
        e1 += hommac.verify_record((Rs + dlt) % ps, (TR + g * dlt) % ps, (TR2 + g2 * dlt) % ps, u, w, KA, 1, ps)
        # E2: injected record under a nonce the OBU did not use
        u_new = r.choice([v for v in range(1, ps) if v != u])
        e2 += hommac.verify_record(r.randrange(ps), r.randrange(ps), r.randrange(ps), u_new, w, KA, 1, ps)
        # E3: the honest result of policy A moved to policy B's key (ID slot changed)
        e3 += hommac.verify_record(Rs, TR, TR2, u, w, KB, 1, ps)
        # E4: q adaptive attempts on the same record, each with a guess pair not tried before
        pairs = r.sample([(a, c) for a in range(1, ps) for c in range(1, ps)], q_adapt)
        dlt = r.randrange(1, ps)
        e4 += any(hommac.verify_record((Rs + dlt) % ps, (TR + a * dlt) % ps, (TR2 + c * dlt) % ps, u, w, KA, 1, ps)
                  for a, c in pairs)
    bnd = 1 / (ps - 1) ** 2
    for name, k, bound in (("E1_altered_score", e1, bnd), ("E2_injected_untagged_nonce", e2, 1 / ps ** 2),
                           ("E3_moved_to_other_policy_key", e3, 1 / ps ** 2), (f"E4_adaptive_q={q_adapt}", e4,
                                                                                q_adapt * bnd)):
        lo, hi = wilson(k, trials)
        res[name] = {"accepted": k, "rate": k / trials, "wilson99.9": [lo, hi], "bound": bound,
                     "bound_within_or_above_interval": lo <= bound}
        if lo > bound:
            fail.append(f"E p={ps} {name}: rate above bound")
    return res


out["E"] = [small_prime_attacks(17, 200000, 16, 1), small_prime_attacks(13, 200000, 12, 2)]
out["failures"] = fail
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "hommac_checks.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
if fail:
    sys.exit("FAILED: " + "; ".join(fail))
print("all checks passed")
