"""BBS04 short group signatures (Boneh, Boyen, Shacham, CRYPTO 2004) over BLS12-381 (type-3 pairing).

Roles are split:
  * Issuer:   holds gamma, enrols OBUs (A_i = g1^{1/(gamma + x_i)}); in SP2UBI: the insurer.
  * Opener:   (xi1, xi2), the opening key. In SP2UBI it is generated without a dealer by the insurer and the TSP, each
              holding one additive share (opening_dkg.py); opening needs both (speedcheck.py). The single `Opener`
              below is the original BBS04 role.
  * Verifier: anyone with the group public key (RSU, TSP, insurer).

Notation follows the scheme description of the paper, with G1 written additively (P * s is scalar multiplication) and GT
multiplicatively. GT has no exponentiation in the binding, so every GT power is moved into G1 by bilinearity,
e(P, Q)^s = e(s*P, Q); signing and verifying each cost one 2-term multi-pairing.

Known property of BBS04: the issuer knows x_i, so the scheme is not exculpable against the
issuer (it is in the original BBS04 model); revocation is not included here.
"""
import hashlib
import secrets
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import py_arkworks_bls12381 as ark

from .groupsig import GroupSignature

G1, G2, GT, Scalar = ark.G1Point, ark.G2Point, ark.GT, ark.Scalar
SCALAR_BYTES = 32
POINT_BYTES = 48


def rand_scalar() -> "Scalar":
    while True:
        s = Scalar.from_be_bytes_mod_order(secrets.token_bytes(64))
        if not s.is_zero():
            return s


def _gt_bytes(e) -> bytes:
    return bytes.fromhex(str(e))          # canonical arkworks serialization of the Fq12 element (576 bytes)


def _challenge(msg: bytes, *elems) -> "Scalar":
    h = hashlib.sha512(b"SP2UBI-BBS04|" + msg)
    for e in elems:
        h.update(e.to_compressed_bytes() if hasattr(e, "to_compressed_bytes") else _gt_bytes(e))
    return Scalar.from_be_bytes_mod_order(h.digest())


G2_BYTES = 96


@dataclass(frozen=True)
class GroupPublicKey:
    g1: "G1"
    g2: "G2"
    h: "G1"
    u: "G1"
    v: "G1"
    w: "G2"

    def to_bytes(self) -> bytes:
        """g1 | h | u | v (compressed G1, 48 B each) | g2 | w (compressed G2, 96 B each): 384 bytes."""
        return b"".join(p.to_compressed_bytes() for p in (self.g1, self.h, self.u, self.v, self.g2, self.w))

    @classmethod
    def from_bytes(cls, b: bytes) -> "GroupPublicKey":
        if len(b) != GPK_BYTES:
            raise ValueError("bad group public key length")
        g1, h, u, v = (G1.from_compressed_bytes(b[i * POINT_BYTES:(i + 1) * POINT_BYTES]) for i in range(4))
        off = 4 * POINT_BYTES
        g2, w = (G2.from_compressed_bytes(b[off + i * G2_BYTES:off + (i + 1) * G2_BYTES]) for i in range(2))
        if any(p == G1.identity() for p in (g1, h, u, v)) or g2 == G2.identity() or w == G2.identity():
            raise ValueError("identity element in the group public key")
        return cls(g1, g2, h, u, v, w)


GPK_BYTES = 4 * POINT_BYTES + 2 * G2_BYTES     # 384 bytes


@dataclass(frozen=True)
class MemberKey:
    A: "G1"
    x: "Scalar"


@dataclass
class Signature:
    T1: "G1"
    T2: "G1"
    T3: "G1"
    c: "Scalar"
    s_alpha: "Scalar"
    s_beta: "Scalar"
    s_x: "Scalar"
    s_d1: "Scalar"
    s_d2: "Scalar"

    def to_bytes(self) -> bytes:
        return (self.T1.to_compressed_bytes() + self.T2.to_compressed_bytes() + self.T3.to_compressed_bytes()
                + b"".join(s.to_be_bytes() for s in (self.c, self.s_alpha, self.s_beta, self.s_x, self.s_d1,
                                                      self.s_d2)))

    @classmethod
    def from_bytes(cls, b: bytes) -> "Signature":
        if len(b) != 3 * POINT_BYTES + 6 * SCALAR_BYTES:
            raise ValueError("bad signature length")
        pts = [G1.from_compressed_bytes(b[i * POINT_BYTES:(i + 1) * POINT_BYTES]) for i in range(3)]  # subgroup-checked
        off = 3 * POINT_BYTES
        sc = [Scalar.from_be_bytes(b[off + i * SCALAR_BYTES: off + (i + 1) * SCALAR_BYTES]) for i in range(6)]
        return cls(*pts, *sc)


SIGNATURE_BYTES = 3 * POINT_BYTES + 6 * SCALAR_BYTES   # 336 bytes


class Issuer:
    def __init__(self, tracing=None):
        """tracing: the (h, u, v) of a jointly generated opening key (opening_dkg.TracingKey); the issuer then never
        knows (xi1, xi2). Without it the issuer generates (h, u, v) and (xi1, xi2) itself (original BBS04 setup).
        The generators g1, g2 are drawn at random (uniform non-identity: a random non-zero multiple of the fixed
        generator) and published in the group public key, as in the key generation of BBS04; the traceability
        reduction embeds its q-SDH instance into them (Boneh-Boyen 2008, Lemma 9)."""
        g1, g2 = G1() * rand_scalar(), G2() * rand_scalar()
        self._gamma = rand_scalar()
        if tracing is None:
            h = g1 * rand_scalar()
            self._xi1, self._xi2 = rand_scalar(), rand_scalar()
            u, v = h * self._xi1.inverse(), h * self._xi2.inverse()
        else:
            h, u, v = tracing.h, tracing.u, tracing.v
            self._xi1 = self._xi2 = None
        self.gpk = GroupPublicKey(g1, g2, h, u, v, g2 * self._gamma)

    def opener_key(self) -> "Opener":
        """Original BBS04: the opening key handed once to a single opening authority; the issuer then deletes it."""
        if self._xi1 is None:
            raise ValueError("no opening key at the issuer (jointly generated, or already handed over)")
        op = Opener(self.gpk, self._xi1, self._xi2)
        self._xi1 = self._xi2 = None
        return op

    def enrol(self) -> MemberKey:
        x = rand_scalar()
        return MemberKey(self.gpk.g1 * (self._gamma + x).inverse(), x)


class Opener:
    def __init__(self, gpk: GroupPublicKey, xi1, xi2):
        self.gpk, self._xi1, self._xi2 = gpk, xi1, xi2
        self.registry: Dict[bytes, str] = {}         # compressed A_i -> member label (filled at enrolment)

    def register(self, label: str, A: "G1") -> None:
        self.registry[A.to_compressed_bytes()] = label

    def open(self, sig: Signature) -> Optional[str]:
        return self.open_tracing(sig.T1, sig.T2, sig.T3)

    def open_tracing(self, T1: "G1", T2: "G1", T3: "G1") -> Optional[str]:
        """Decrypt a (possibly re-randomised) linear-encryption tracing ciphertext to the member label."""
        A = T3 - (T1 * self._xi1 + T2 * self._xi2)
        return self.registry.get(A.to_compressed_bytes())


class OpenerShare:
    def __init__(self, xi1_share, xi2_share, u: "G1", v: "G1", h: Optional["G1"] = None,
                 peer_Y: Optional[Tuple["G1", "G1"]] = None):
        self._s1, self._s2 = xi1_share, xi2_share
        self.u, self.v, self.h = u, v, h
        self.Y = (u * xi1_share, v * xi2_share)                    # public share commitments (A_k, B_k)
        self.peer_Y = peer_Y           # the peer's share values, checked at setup to add up with Y to (h, h)

    def partial(self, T1: "G1", T2: "G1") -> "G1":
        return T1 * self._s1 + T2 * self._s2

    def partial_with_proof(self, T1: "G1", T2: "G1", ctx: bytes = b""):
        """Returns ((xi1_k*T1, xi2_k*T2), (DLEQ(u, A_k; T1, .), DLEQ(v, B_k; T2, .)))."""
        from .proofs import prove_dleq                              # local import: proofs depends on this module
        P1, P2 = T1 * self._s1, T2 * self._s2
        return (P1, P2), (prove_dleq(ctx + b"|o1", self.u, self.Y[0], T1, P1, self._s1),
                          prove_dleq(ctx + b"|o2", self.v, self.Y[1], T2, P2, self._s2))


def combine_opening(T3: "G1", p1: "G1", p2: "G1") -> "G1":
    """Recover the member credential A from the two partial openings."""
    return T3 - (p1 + p2)


def member_key_valid(gpk: GroupPublicKey, mk: MemberKey) -> bool:
    """e(A, w + x*g2) == e(g1, g2): the OBU checks its credential at enrolment."""
    return GT.pairing(mk.A, gpk.w + gpk.g2 * mk.x) == GT.pairing(gpk.g1, gpk.g2)


class BBS04(GroupSignature):
    def __init__(self, gpk: GroupPublicKey):
        self.gpk = gpk

    def sign(self, mk: MemberKey, msg: bytes) -> bytes:
        g = self.gpk
        a, b = rand_scalar(), rand_scalar()
        T1, T2, T3 = g.u * a, g.v * b, mk.A + g.h * (a + b)
        d1, d2 = mk.x * a, mk.x * b
        ra, rb, rx, rd1, rd2 = (rand_scalar() for _ in range(5))
        R1, R2 = g.u * ra, g.v * rb
        R3 = GT.multi_pairing([T3 * rx + g.h * (-(rd1 + rd2)), g.h * (-(ra + rb))], [g.g2, g.w])
        R4, R5 = T1 * rx + g.u * (-rd1), T2 * rx + g.v * (-rd2)
        c = _challenge(msg, T1, T2, T3, R1, R2, R3, R4, R5)
        sig = Signature(T1, T2, T3, c, ra + c * a, rb + c * b, rx + c * mk.x, rd1 + c * d1, rd2 + c * d2)
        return sig.to_bytes()

    def verify(self, msg: bytes, sig_bytes: bytes) -> bool:
        try:
            s = Signature.from_bytes(sig_bytes)
        except Exception:
            return False
        g, c = self.gpk, s.c
        R1 = g.u * s.s_alpha - s.T1 * c
        R2 = g.v * s.s_beta - s.T2 * c
        R3 = GT.multi_pairing([s.T3 * s.s_x + g.h * (-(s.s_d1 + s.s_d2)) + g.g1 * (-c),
                               g.h * (-(s.s_alpha + s.s_beta)) + s.T3 * c], [g.g2, g.w])
        R4 = s.T1 * s.s_x - g.u * s.s_d1
        R5 = s.T2 * s.s_x - g.v * s.s_d2
        return _challenge(msg, s.T1, s.T2, s.T3, R1, R2, R3, R4, R5) == c
