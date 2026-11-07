"""Exponential (additively homomorphic) ElGamal over the BLS12-381 G1 group, used for the speed-verification path.

    KeyGen:  sk = x,  pk = X = x*G
    Enc(m):  (C1, C2) = (r*G, m*G + r*X)            m is a small signed integer
    Add:     (C1 + C1', C2 + C2'),   ScalarMul by integer k: (k*C1, k*C2)
    ReRand:  (C1 + r'*G, C2 + r'*X)                 the result is a uniformly random encryption of the same m, so the
                                                    output distribution is independent of how it was computed
                                                    (perfect circuit privacy, no noise flooding needed)
    Dec:     M = C2 - x*C1 = m*G, then m by baby-step giant-step over a bounded range.

Ciphertexts are two compressed G1 points (96 bytes). IND-CPA under DDH in G1; key-private (IK-CPA) since all keys
share the group (Bellare, Boldyreva, Desai, Pointcheval, ASIACRYPT 2001).
"""
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import py_arkworks_bls12381 as ark

from .groupsig_bbs04 import POINT_BYTES, rand_scalar
from .proofs import Proof, prove_dleq, prove_dlog, verify_dleq, verify_dlog

G1, Scalar = ark.G1Point, ark.Scalar
CIPHERTEXT_BYTES = 2 * POINT_BYTES


def _scalar(m: int) -> "Scalar":
    s = Scalar(abs(m))
    return -s if m < 0 else s


@dataclass(frozen=True)
class Ciphertext:
    C1: "G1"
    C2: "G1"

    def __add__(self, o: "Ciphertext") -> "Ciphertext":
        return Ciphertext(self.C1 + o.C1, self.C2 + o.C2)

    def __sub__(self, o: "Ciphertext") -> "Ciphertext":
        return Ciphertext(self.C1 - o.C1, self.C2 - o.C2)

    def mul(self, k: int) -> "Ciphertext":
        s = _scalar(k)
        return Ciphertext(self.C1 * s, self.C2 * s)

    def to_bytes(self) -> bytes:
        return self.C1.to_compressed_bytes() + self.C2.to_compressed_bytes()

    @classmethod
    def from_bytes(cls, b: bytes) -> "Ciphertext":
        if len(b) != CIPHERTEXT_BYTES:
            raise ValueError("bad ciphertext length")
        return cls(G1.from_compressed_bytes(b[:POINT_BYTES]), G1.from_compressed_bytes(b[POINT_BYTES:]))


@dataclass(frozen=True)
class PublicKey:
    G: "G1"
    X: "G1"

    def encrypt(self, m: int) -> Ciphertext:
        r = rand_scalar()
        return Ciphertext(self.G * r, self.G * _scalar(m) + self.X * r)

    def rerandomize(self, ct: Ciphertext) -> Ciphertext:
        r = rand_scalar()
        return Ciphertext(ct.C1 + self.G * r, ct.C2 + self.X * r)


class SecretKey:
    def __init__(self, bsgs_bits: int = 32, baby_bits: int = 18):
        self.G = G1()
        self._x = rand_scalar()
        self.pk = PublicKey(self.G, self.G * self._x)
        self._range_bits, self._baby_bits = bsgs_bits, baby_bits
        self._table: Optional[Dict[bytes, int]] = None

    def decrypt_point(self, ct: Ciphertext) -> "G1":
        return ct.C2 - ct.C1 * self._x

    def _build_table(self) -> None:
        self._table, P = {}, G1.identity()
        for j in range(1 << self._baby_bits):
            self._table[P.to_compressed_bytes()] = j
            P = P + self.G

    def decrypt(self, ct: Ciphertext) -> Optional[int]:
        return self._dlog(self.decrypt_point(ct))

    def _dlog(self, M) -> Optional[int]:
        """Return m with M = m*G and |m| < 2^(bsgs_bits-1), or None if outside that range.

        Two-sided baby-step giant-step: m = i*W + j with j in [0, W), W = 2^baby_bits, searching i = 0, -1, 1, -2, ...
        so the cost grows with |m| (verification results cluster near zero)."""
        if self._table is None:
            self._build_table()
        W = 1 << self._baby_bits
        half = 1 << (self._range_bits - 1)
        step = self.G * Scalar(W)
        up = down = M                                               # up = M - i*W*G (i >= 0), down = M + i*W*G
        for i in range(half // W + 1):
            j = self._table.get(up.to_compressed_bytes())
            if j is not None and -half <= i * W + j < half:
                return i * W + j
            if i > 0:
                j = self._table.get(down.to_compressed_bytes())
                if j is not None and -half <= -i * W + j < half:
                    return -i * W + j
            up, down = up - step, down + step
        return None


class ThresholdKey:
    """2-of-2 additive sharing of the ElGamal secret (x = x_ins + x_tsp, X = X_ins + X_tsp), generated distributedly:
    neither party ever holds x. Key generation (two-party DKG): each party first publishes a commitment to
    (X_k, PoK_k), then reveals; each checks the other's opening and Schnorr proof of knowledge, so no party can choose
    its share after seeing the other's (rogue-key shares). Decryption needs a partial D_k = x_k * C1 from each party,
    every partial carries a DLEQ proof against the published X_k, so a wrong partial is detected instead of silently
    shifting the plaintext. 2-of-2 has no availability margin: either party can block decryption."""

    def __init__(self, bsgs_bits: int = 32, baby_bits: int = 18, party: bytes = b""):
        self.G = G1()
        self._x = rand_scalar()
        self.share_pub = self.G * self._x
        self.party = party
        self._bsgs = SecretKey(bsgs_bits, baby_bits)          # only its BSGS table is used (shared group, same G)
        self._pok: Dict[bytes, Proof] = {}

    # --- DKG (commit, then reveal with proof of knowledge) ---
    def pok(self, session: bytes) -> Proof:
        if session not in self._pok:                          # one proof per session: the reveal must match the commitment
            self._pok[session] = prove_dlog(session + b"|" + self.party, self.G, self.share_pub, self._x)
        return self._pok[session]

    def commitment(self, session: bytes) -> bytes:
        return dkg_commitment(session, self.party, self.share_pub, self.pok(session))

    # --- decryption ---
    def partial(self, ct: Ciphertext) -> "G1":
        return ct.C1 * self._x

    def partial_with_proof(self, ct: Ciphertext, ctx: bytes = b"") -> Tuple["G1", Proof]:
        D = self.partial(ct)
        return D, prove_dleq(ctx, self.G, self.share_pub, ct.C1, D, self._x)

    def combine(self, ct: Ciphertext, other_partial: "G1") -> Optional[int]:
        M = ct.C2 - self.partial(ct) - other_partial
        return self._bsgs._dlog(M)

    def combine_verified(self, ct: Ciphertext, other_partial: "G1", other_proof: Proof, other_pub: "G1",
                         ctx: bytes = b"") -> Optional[int]:
        """Combine only if the other party's partial is proven correct; a wrong partial raises instead of decoding."""
        if not verify_dleq(ctx, self.G, other_pub, ct.C1, other_partial, other_proof):
            raise ValueError("partial decryption proof invalid")
        return self.combine(ct, other_partial)


def dkg_commitment(session: bytes, party: bytes, X: "G1", pi: Proof) -> bytes:
    return hashlib.sha256(b"SP2UBI-DKG|" + session + b"|" + party + X.to_compressed_bytes()
                          + pi.c.to_le_bytes() + b"".join(si.to_le_bytes() for si in pi.s)).digest()


def dkg_accept_peer(session: bytes, party: bytes, commitment: bytes, X: "G1", pi: Proof) -> bool:
    """Check a peer's revealed share against its earlier commitment and its proof of knowledge."""
    return (dkg_commitment(session, party, X, pi) == commitment
            and verify_dlog(session + b"|" + party, G1(), X, pi))


def joint_public_key(a: "ThresholdKey", b: "ThresholdKey") -> PublicKey:
    return PublicKey(a.G, a.share_pub + b.share_pub)
