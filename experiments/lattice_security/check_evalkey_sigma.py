"""Error standard deviation of the OpenFHE key-switching keys versus the public key (risk-path context).

Claim checked: in OpenFHE v1.5.1 (pip openfhe==1.5.1.0.24.4) the public key and the encryptions use the context's
discrete-Gaussian sigma = 3.19, whereas the errors of the BV key-switching keys (relinearisation key, rotation keys) are
sampled with sigma = 1.0 (the sampling loop of KeySwitchBV::KeySwitchGenInternal runs under OpenMP private(dgg), which
default-constructs the generator with std = 1.0).

Method. The script builds the context of sp2ubi.fhe.RiskContext (imported unchanged), generates the key pair, the
relinearisation key and the rotation keys (1, 2, 4), serialises them to JSON and recovers each error polynomial from the
secret key. All polynomials are serialised in evaluation (NTT) form, one tower per RNS modulus q_i, with the primitive
2N-th root of unity psi_i stored in the serialised context.
  * Public key (h[0], h[1]) = (b, a):  b = e - a*s, hence  e = b + a*s  (mod q_i).
  * Key-switching key for the key pair (old, new) with digits j = 0..d-1 (one digit per RNS tower, digitSize 0):
        bk_j = g_j*old - a_j*new + e_j,  with  a_j = ak_j,  g_j = CRT idempotent (1 mod q_j, 0 mod q_i for i != j)
    hence  e_j = bk_j + ak_j*new - g_j*old  (mod q_i), where
        relinearisation key:  old = s^2, new = s;
        rotation key for the stored automorphism index k (5^r mod 2N for the slot rotation r):
                           old = s, new = s(X^(k^-1 mod 2N))  (the pairing was fixed by testing the four
                           combinations of old/new and k/k^-1; only this one gives a small error)
    The script does not assume the relation: it checks that the result is small (|e| << q) and reports the rms and
    maximum over all digits and towers; a wrong relation would give uniformly distributed values near q/2.
  * Inverse NTT: Gentleman-Sande, bit-reversed ordering of the evaluation points psi^(2*brv(i)+1). The ordering is
    validated by checking that the recovered secret key is ternary.
Values are centred modulo q_i. Output: results/check_evalkey_sigma.json next to this script. No timing is measured.
The key generation is not seeded, so the digits of the statistics differ between runs; the rms values do not.

Usage (from the repository root):  python experiments/lattice_security/check_evalkey_sigma.py
Run with OMP_NUM_THREADS=1 or any thread count: the result does not depend on it.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import openfhe as o  # noqa: E402

from sp2ubi.fhe import RiskContext  # noqa: E402
from sp2ubi.params import DEFAULT as P  # noqa: E402


def brv(k, bits):
    return int(format(k, "0{}b".format(bits))[::-1], 2)


class Ntt:
    def __init__(self, N, q, psi):
        self.N, self.q = N, q
        self.bits = N.bit_length() - 1
        pinv = pow(psi, -1, q)
        self.psi_br = [pow(psi, brv(k, self.bits), q) for k in range(N)]
        self.pinv_br = [pow(pinv, brv(k, self.bits), q) for k in range(N)]
        self.ninv = pow(N, -1, q)

    def forward(self, a):
        q, N = self.q, self.N
        a = np.array(a, dtype=object)
        t, m = N, 1
        while m < N:
            t //= 2
            v = a.reshape(m, 2, t)
            S = np.array(self.psi_br[m:2 * m], dtype=object).reshape(m, 1)
            U, V = v[:, 0, :].copy(), (v[:, 1, :] * S) % q
            v[:, 0, :], v[:, 1, :] = (U + V) % q, (U - V) % q
            m *= 2
        return a

    def inverse(self, a):
        q, N = self.q, self.N
        a = np.array(a, dtype=object)
        t, m = 1, N
        while m > 1:
            h = m // 2
            v = a.reshape(h, 2, t)
            S = np.array(self.pinv_br[h:2 * h], dtype=object).reshape(h, 1)
            U, V = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :], v[:, 1, :] = (U + V) % q, ((U - V) * S) % q
            t *= 2
            m = h
        return (a * self.ninv) % q


def centred(a, q):
    return np.array([int(x) - q if int(x) > q // 2 else int(x) for x in a], dtype=object)


def stats(vals):
    f = np.array([float(x) for x in vals])
    return {"n": int(f.size), "rms": round(float(np.sqrt(np.mean(f * f))), 4), "max_abs": int(np.max(np.abs(f))),
            "mean": round(float(np.mean(f)), 4)}


def find(x, key):
    """All values stored under `key` anywhere in a parsed JSON tree."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k == key:
                out.append(v)
            out.extend(find(v, key))
    elif isinstance(x, list):
        for v in x:
            out.extend(find(v, key))
    return out


def towers(poly):
    """Evaluation-form towers of a serialised DCRTPoly: list of coefficient lists, one per RNS modulus (N values each)."""
    return [x for x in find(poly, "v") if isinstance(x, list) and len(x) > 64 and isinstance(x[0], int)]


def main():
    risk = RiskContext(P.p, P.m)
    kp = risk.keygen()
    tag = kp.secretKey.GetKeyTag()
    N = risk.N
    ctx = json.loads(o.Serialize(risk.cc, o.JSON))
    tower_params = [t["ptr_wrapper"]["data"]["value0"] for t in find(ctx, "p")[0]]
    moduli = [t["cm"]["v"] for t in tower_params]
    roots = [t["ru"]["v"] for t in tower_params]
    sigma_ctx = find(ctx, "dp")[0]
    ntts = [Ntt(N, q, psi) for q, psi in zip(moduli, roots)]

    sk_json = json.loads(o.Serialize(kp.secretKey, o.JSON))
    s_ev = towers(sk_json)
    pk_json = json.loads(o.Serialize(kp.publicKey, o.JSON))
    h = [towers(poly) for poly in find(pk_json, "h")[0]]

    def mul(x, y, q):
        return [(int(a) * int(b)) % q for a, b in zip(x, y)]

    results = {"openfhe": "1.5.1.0.24.4", "N": N, "moduli": moduli, "context_sigma": round(sigma_ctx, 4),
               "p": P.p}

    # secret key must be ternary in coefficient form (validates the NTT ordering)
    sk_coef = [centred(ntts[i].inverse(s_ev[i]), moduli[i]) for i in range(len(moduli))]
    results["secret_key_ternary"] = all(set(int(x) for x in c) <= {-1, 0, 1} for c in sk_coef)
    if not results["secret_key_ternary"]:
        sys.exit("NTT ordering not validated: the recovered secret key is not ternary")

    # public key
    pk_err = []
    for i, q in enumerate(moduli):
        ev = [(int(b) + int(x)) % q for b, x in zip(h[0][i], mul(h[1][i], s_ev[i], q))]
        pk_err.append(centred(ntts[i].inverse(ev), q))
    results["public_key_error"] = stats(np.concatenate(pk_err))

    # relinearisation key: old = s^2
    s2 = [mul(s_ev[i], s_ev[i], moduli[i]) for i in range(len(moduli))]
    em = json.loads(o.SerializeEvalMultKeyString(o.JSON, tag))
    results["relinearisation_key_error"] = ks_error(em, s2, s_ev, moduli, ntts)

    # rotation keys: old = s(X^k)
    ea = json.loads(o.SerializeEvalAutomorphismKeyString(o.JSON, tag))
    entries = find(ea, "data")[0]
    rot = {}
    for ent in entries:
        k = pow(int(ent["key"]), -1, 2 * N)   # the key stored under index k0 switches to s(X^(k0^-1)), found by trial
        # sigma_k(s) = s(X^k), computed on the coefficients of s and transformed back
        sig = []
        for i, q in enumerate(moduli):
            c = sk_coef[i]
            r = [0] * N
            for j in range(N):
                e = (j * k) % (2 * N)
                r[e % N] = (r[e % N] + (int(c[j]) if e < N else -int(c[j]))) % q
            sig.append(ntts[i].forward(r))
        rot["automorphism_%d" % int(ent["key"])] = ks_error_one(ent["value"], s_ev, sig, moduli, ntts)
    results["rotation_key_errors"] = rot
    rot_all = [r["rms"] for r in rot.values()]
    results["verdict"] = {
        "public_key_rms_expected": 3.19, "evalkey_rms_expected": 1.0,
        "public_key_rms_measured": results["public_key_error"]["rms"],
        "relinearisation_rms_measured": results["relinearisation_key_error"]["rms"],
        "rotation_rms_measured": rot_all,
        "confirmed": bool(abs(results["public_key_error"]["rms"] - 3.19) < 0.15
                          and abs(results["relinearisation_key_error"]["rms"] - 1.0) < 0.1
                          and all(abs(v - 1.0) < 0.1 for v in rot_all))}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "check_evalkey_sigma.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("written", out)


def ks_error(em_json, old, new, moduli, ntts):
    val = find(em_json, "value")[0][0]
    return ks_error_one(val, old, new, moduli, ntts)


def ks_error_one(val, old, new, moduli, ntts):
    node = val
    ak, bk = find(node, "ak")[0], find(node, "bk")[0]
    allv, per_digit = [], []
    for j, (a_j, b_j) in enumerate(zip(ak, bk)):
        a_t, b_t = towers(a_j), towers(b_j)
        dv = []
        for i, q in enumerate(moduli):
            ev = []
            for x in range(len(new[i])):
                v = int(b_t[i][x]) + int(a_t[i][x]) * int(new[i][x])
                if i == j:
                    v -= int(old[i][x])
                ev.append(v % q)
            dv.append(centred(ntts[i].inverse(ev), q))
        dv = np.concatenate(dv)
        per_digit.append(stats(dv))
        allv.append(dv)
    res = stats(np.concatenate(allv))
    res["per_digit"] = per_digit
    return res


if __name__ == "__main__":
    main()
