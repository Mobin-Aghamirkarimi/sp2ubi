"""Non-interactive zero-knowledge proofs (Fiat-Shamir, SHA-512) used on the speed-verification key path: Schnorr proof of
knowledge of a key share (two-party DKG), Chaum-Pedersen DLEQ for partial decryptions, a two-base representation proof
for partial openings of the BBS04 tracing ciphertext, and the public consistency check of the opening-key shares
(generated without a dealer, opening_dkg.py).
"""
import hashlib
from dataclasses import dataclass
from typing import Tuple

import py_arkworks_bls12381 as ark

from .groupsig_bbs04 import GroupPublicKey, rand_scalar

G1, G2, GT, Scalar = ark.G1Point, ark.G2Point, ark.GT, ark.Scalar


def fs_challenge(label: bytes, *elems) -> "Scalar":
    """Fiat-Shamir challenge: SHA-512 over a domain label and the compressed group elements / bytes, reduced mod r."""
    h = hashlib.sha512(b"SP2UBI-FS|" + label)
    for e in elems:
        b = e if isinstance(e, (bytes, bytearray)) else e.to_compressed_bytes()
        h.update(len(b).to_bytes(4, "big") + b)
    return Scalar.from_be_bytes_mod_order(h.digest())


@dataclass(frozen=True)
class Proof:
    """Sigma-protocol transcript in Fiat-Shamir form: challenge c and responses s (one per witness)."""
    c: "Scalar"
    s: Tuple["Scalar", ...]


def prove_dlog(ctx: bytes, G, X, x) -> Proof:
    """Schnorr proof of knowledge of x with X = x*G (key-share registration, prevents rogue-key shares)."""
    r = rand_scalar()
    c = fs_challenge(b"pok|" + ctx, G, X, G * r)
    return Proof(c, (r + c * x,))


def verify_dlog(ctx: bytes, G, X, pi: Proof) -> bool:
    return fs_challenge(b"pok|" + ctx, G, X, G * pi.s[0] - X * pi.c) == pi.c


def prove_dleq(ctx: bytes, G, X, C, D, x) -> Proof:
    """Chaum-Pedersen proof that log_G X = log_C D (correct partial decryption D = x*C1 under share X = x*G)."""
    r = rand_scalar()
    c = fs_challenge(b"dleq|" + ctx, G, X, C, D, G * r, C * r)
    return Proof(c, (r + c * x,))


def verify_dleq(ctx: bytes, G, X, C, D, pi: Proof) -> bool:
    s = pi.s[0]
    return fs_challenge(b"dleq|" + ctx, G, X, C, D, G * s - X * pi.c, C * s - D * pi.c) == pi.c


def schnorr_sign(e, E, msg: bytes) -> Proof:
    """Schnorr signature with an ephemeral per-burst key (E = e*G): proof of knowledge of e bound to msg."""
    G = G1()
    r = rand_scalar()
    c = fs_challenge(b"esig|" + msg, G, E, G * r)
    return Proof(c, (r + c * e,))


def schnorr_verify(E, msg: bytes, sig: Proof) -> bool:
    G = G1()
    return fs_challenge(b"esig|" + msg, G, E, G * sig.s[0] - E * sig.c) == sig.c


def proof_bytes(p: Proof) -> bytes:
    return p.c.to_le_bytes() + b"".join(x.to_le_bytes() for x in p.s)


def proof_from_bytes(b: bytes, n: int = 1) -> Proof:
    k = 32
    if len(b) != k * (n + 1):
        raise ValueError("bad proof length")
    return Proof(Scalar.from_le_bytes(b[:k]), tuple(Scalar.from_le_bytes(b[k * (i + 1):k * (i + 2)]) for i in range(n)))


def verify_opening_setup(gpk: GroupPublicKey, Y_a: Tuple["G1", "G1"], Y_b: Tuple["G1", "G1"]) -> bool:
    """Opening-share commitments A_k = xi1_k * u, B_k = xi2_k * v of the two parties must add up to h (xi1 * u =
    xi2 * v = h by construction of the group key), so the shares are consistent with the group public key.
    gpk: any object with the element h (GroupPublicKey or opening_dkg.TracingKey). Called by each party at the end of
    the opening-key generation (opening_dkg.OpeningKeyParty.finish: a failure aborts before a key is output) and by
    the insurer when it registers the TSP's share values (speedcheck.InsurerSpeedVerifier)."""
    return Y_a[0] + Y_b[0] == gpk.h and Y_a[1] + Y_b[1] == gpk.h
