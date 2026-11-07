"""Homomorphic MAC with per-record one-time keys and two tag blocks.

For record nonce u in epoch t and tag block b in {0, 1}, with domain-separated PRF labels:
    block 0:  alpha_u  = PRF_K(t, u, "alpha") + 1,     r_{u,i}  = PRF_K(t, u, "r", i)
    block 1:  alpha'_u = PRF_K(t, u, "alpha", 1) + 1,  r'_{u,i} = PRF_K(t, u, "r", i, 1)
Tags (computed by the OBU):      T_i = alpha_u x_i + r_{u,i},  T'_i = alpha'_u x_i + r'_{u,i}  (mod p),  x_0 = 1
TSP evaluates the same linear F: R = sum w_i x_i,  T_R = sum w_i T_i,  T'_R = sum w_i T'_i
Insurer verifies both:           T_R == alpha_u R + sum w_i r_{u,i}  and  T'_R == alpha'_u R + sum w_i r'_{u,i}

Because r_u (r'_u) is fresh for every record it acts as a one-time pad on alpha_u (alpha'_u), so both are
information-theoretically hidden from anything the TSP can compute from (x, T, T', u). The labels of block 1 cannot
equal any label of block 0 (the joined messages "t|u|alpha|1" and "t|u|r|i|1" differ from "t|u|alpha" and "t|u|r|j"
because i, j are integers), so (alpha', r') is independent of (alpha, r) up to Adv_PRF: a forged (R', T_R, T'_R)
with R' != F(x) passes both checks with probability at most 1/(p-1)^2 ~ 2^-80 (plus Adv_PRF) per deviating record.
`verify` checks one block (the tests use it to measure the single-block rate 1/(p-1)); `verify_record` checks both
blocks and is what the insurer uses.
"""
import hashlib
import hmac
from typing import List, Sequence, Tuple

BLOCKS = (0, 1)


def prf(key: bytes, mod: int, *parts) -> int:
    """HMAC-SHA256(key, "|".join(parts)) reduced mod `mod` (bias < 2^-215 for a 41-bit modulus)."""
    msg = b"|".join(str(x).encode() for x in parts)
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest(), "big") % mod


def record_keys(K: bytes, epoch: int, u: int, p: int, m: int, block: int = 0) -> Tuple[int, List[int]]:
    """One-time keys of tag block `block`; block 1 appends the label 1 to the labels of block 0."""
    if block not in BLOCKS:
        raise ValueError("tag block must be 0 or 1")
    sfx = () if block == 0 else (block,)
    alpha = prf(K, p - 1, epoch, u, "alpha", *sfx) + 1          # alpha in Z_p^*
    r = [prf(K, p, epoch, u, "r", i, *sfx) for i in range(m)]
    return alpha, r


def tags(x: Sequence[int], K: bytes, epoch: int, u: int, p: int, block: int = 0) -> List[int]:
    alpha, r = record_keys(K, epoch, u, p, len(x), block)
    return [(alpha * xi + ri) % p for xi, ri in zip(x, r)]


def tag_blocks(x: Sequence[int], K: bytes, epoch: int, u: int, p: int) -> Tuple[List[int], List[int]]:
    """Both tag blocks (T, T') of a record."""
    return tags(x, K, epoch, u, p, 0), tags(x, K, epoch, u, p, 1)


def verify(R: int, T_R: int, u: int, w: Sequence[int], K: bytes, epoch: int, p: int, block: int = 0) -> bool:
    """One block. R and w may be signed integers or residues; everything is reduced mod p."""
    alpha, r = record_keys(K, epoch, u, p, len(w), block)
    return T_R % p == (alpha * R + sum(wi * ri for wi, ri in zip(w, r))) % p


def verify_record(R: int, T_R: int, T_R2: int, u: int, w: Sequence[int], K: bytes, epoch: int, p: int) -> bool:
    """Both blocks: the insurer accepts a record only if T_R and T'_R both verify."""
    return verify(R, T_R, u, w, K, epoch, p, 0) and verify(R, T_R2, u, w, K, epoch, p, 1)
