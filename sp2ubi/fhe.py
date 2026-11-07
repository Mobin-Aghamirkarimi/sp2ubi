"""OpenFHE-BFV layer.

Risk path (one insurer BFV key, no threshold):
* BFV with plaintext modulus t = p (the HomMAC field), depth 1, HEStd_128_classic, a single key pair that the
  insurer generates (KeyGen), together with the relinearisation key and the rotation keys for (1, 2, 4). OpenFHE chooses the
  ring dimension and the modulus for this single-key context (see `info()`; recorded by
  experiments/sizes/risk_path_sizes.py). There is no TSP key share, no multiparty mode,
  no flooding and no erasure assumption on the risk path: the TSP holds only the public key and the public evaluation
  keys, and the insurer decrypts the shuffled outputs itself.
* After the evaluation the TSP fills every non-output slot of all N slots (both rows) with fresh uniform values mod p;
  the insurer decodes only the output slots.
Slot layout of an epoch record (block size 8):
    x_0..x_{m-1} -> slots 0..m-1,  T_0..T_{m-1} -> 8..8+m-1,  u -> 16,  ID -> 24,  T'_0..T'_{m-1} -> 32..32+m-1
(T' is the second, independent HomMAC tag block of the risk path.)
Enc(w) uses the same layout (w in the x block and in both tag blocks, 1 at slots 16 and 24): one ct x ct product
followed by rotate-and-sum (1, 2, 4) yields S in slot 0, T_S in 8, u in 16, ID in 24 and T'_S in 32. In this
single-process simulation the evaluation keys live in the shared CryptoContext; in a deployment they are serialized,
insurer-signed (entities.KeyBundle) and installed at the TSP.

The speed-verification path does not use BFV; it is in speedcheck.py (two-party exponential ElGamal on BLS12-381).
"""
import json
import math
import secrets
from typing import List, Optional, Sequence

import openfhe as o

SLOTS = 40
SLOT_X, SLOT_T, SLOT_U, SLOT_ID, SLOT_T2 = 0, 8, 16, 24, 32
OUTPUT_SLOTS = (SLOT_X, SLOT_T, SLOT_U, SLOT_ID, SLOT_T2)
ROTATIONS = (1, 2, 4)
OPENFHE_VERSION = "1.5.1.0.24.4"     # pinned (requirements/core.lock)


def uniform_mod(p: int, n: int) -> List[int]:
    """n values uniform in [0, p) from the OS CSPRNG (128-bit reduction: bias < 2^-87)."""
    raw = secrets.token_bytes(16 * n)
    return [int.from_bytes(raw[16 * i:16 * i + 16], "big") % p for i in range(n)]


class RiskContext:
    def __init__(self, p: int, m: int):
        if m > 8:
            raise ValueError("layout supports at most 8 coefficient positions")
        self.p, self.m = p, m
        prm = o.CCParamsBFVRNS()
        prm.SetPlaintextModulus(p)
        prm.SetMultiplicativeDepth(1)
        prm.SetSecurityLevel(o.HEStd_128_classic)
        self.cc = o.GenCryptoContext(prm)
        for f in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH, o.PKESchemeFeature.LEVELEDSHE):
            self.cc.Enable(f)
        self.N = self.cc.GetRingDimension()

    # --- insurer: one-off key generation
    def keygen(self):
        """The insurer's key pair, relinearisation key and rotation keys (1, 2, 4). Returns the OpenFHE key pair; the
        evaluation keys are installed in the context under the key's tag (public: signed into the key bundle and
        installed at the TSP)."""
        kp = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(kp.secretKey)
        self.cc.EvalAtIndexKeyGen(kp.secretKey, list(ROTATIONS))
        return kp

    def encrypt_weights(self, pk, w: Sequence[int]):
        v = [0] * SLOTS
        v[SLOT_X:SLOT_X + self.m] = [wi % self.p for wi in w]
        v[SLOT_T:SLOT_T + self.m] = [wi % self.p for wi in w]
        v[SLOT_T2:SLOT_T2 + self.m] = [wi % self.p for wi in w]
        v[SLOT_U] = v[SLOT_ID] = 1
        return self.cc.Encrypt(pk, self.cc.MakePackedPlaintext(v))

    def encrypt_vector(self, pk, values: Sequence[int]):
        """Encrypt an arbitrary slot vector (key confirmation, adversarial tests)."""
        return self.cc.Encrypt(pk, self.cc.MakePackedPlaintext([v % self.p for v in values]))

    # --- OBU
    def encrypt_payload(self, pk, x: Sequence[int], T: Sequence[int], u: int, ident: int,
                        T2: Optional[Sequence[int]] = None):
        """T2 is the second tag block; None leaves its slots 0."""
        v = [0] * SLOTS
        v[SLOT_X:SLOT_X + self.m] = list(x)
        v[SLOT_T:SLOT_T + self.m] = list(T)
        if T2 is not None:
            v[SLOT_T2:SLOT_T2 + self.m] = list(T2)
        v[SLOT_U], v[SLOT_ID] = u, ident
        return self.cc.Encrypt(pk, self.cc.MakePackedPlaintext(v))

    # --- TSP
    def evaluate(self, ct_payload, ct_w):
        """F_Ins applied to features and tags at once: one ct x ct product + rotate-and-sum."""
        ct = self.cc.EvalMult(ct_payload, ct_w)
        for k in ROTATIONS:
            ct = self.cc.EvalAdd(ct, self.cc.EvalRotate(ct, k))
        return ct

    def fill(self, ct, pk):
        """Add Enc(nu), nu fresh and uniform mod p in every non-output slot of all N slots, 0 in the output slots."""
        nu = uniform_mod(self.p, self.N)
        for i in OUTPUT_SLOTS:
            nu[i] = 0
        return self.cc.EvalAdd(ct, self.cc.Encrypt(pk, self.cc.MakePackedPlaintext(nu)))

    # --- insurer
    def decrypt(self, ct, sk, length: int = SLOTS) -> List[int]:
        """Insurer's decryption of a published output; `length` slots (default: the record layout)."""
        pt = self.cc.Decrypt(sk, ct)
        pt.SetLength(length)
        return [int(v) % self.p for v in pt.GetPackedValue()]

    def size(self, obj) -> int:
        return len(o.Serialize(obj, o.BINARY))

    def moduli(self) -> List[int]:
        """RNS moduli q_0, q_1, ... of the ciphertext modulus Q, read from the serialized context (GetModulus() is a
        float in the Python wrapper)."""
        found = []

        def walk(x, path):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, path + [k])
            elif isinstance(x, list):
                for v in x:
                    walk(v, path + ["#"])
            elif isinstance(x, int) and path[-4:] == ["data", "value0", "cm", "v"] and x > (1 << 40):
                found.append(x)
        walk(json.loads(o.Serialize(self.cc, o.JSON)), [])
        return found

    def info(self) -> dict:
        q = self.moduli()
        Q = 1
        for qi in q:
            Q *= qi
        return {"scheme": "BFV", "N": self.N, "log2_Q": round(math.log2(Q), 2), "moduli_bits": [round(math.log2(qi), 2) for qi in q],
                "log2_Delta": round(math.log2(Q // self.p), 2), "t_bits": self.p.bit_length(),
                "security": "HEStd_128_classic", "depth": 1, "keys": "single key pair (insurer)",
                "openfhe": OPENFHE_VERSION}

