import random
import secrets

import pytest

from sp2ubi import hommac
from sp2ubi.params import DEFAULT, is_prime

P = DEFAULT.p
K = secrets.token_bytes(32)


def _record(rng, m=DEFAULT.m):
    x = [1] + [rng.randrange(1 << 16) for _ in range(m - 1)]
    w = [rng.randrange(-4095, 4096) for _ in range(m)]
    return x, w


def test_honest_record_verifies_with_intercept():
    rng = random.Random(1)
    for _ in range(200):
        x, w = _record(rng)
        u = rng.randrange(1, P)
        T = hommac.tags(x, K, 1, u, P)
        R = sum(a * b for a, b in zip(w, x))                       # includes intercept w_0 * 1
        T_R = sum(a * b for a, b in zip(w, T))
        assert hommac.verify(R, T_R, u, w, K, 1, P)


def test_modified_score_or_nonce_rejected():
    rng = random.Random(2)
    x, w = _record(rng)
    u = rng.randrange(1, P)
    T = hommac.tags(x, K, 1, u, P)
    R, T_R = sum(a * b for a, b in zip(w, x)), sum(a * b for a, b in zip(w, T))
    assert not hommac.verify(R + 1, T_R, u, w, K, 1, P)
    assert not hommac.verify(R, T_R, u + 1, w, K, 1, P)
    assert not hommac.verify(R, T_R, u, w, K, 2, P)                # other epoch
    assert not hommac.verify(R, T_R, u, [w[0] + 1] + w[1:], K, 1, P)  # other model


def test_model_substitution_succeeds_with_probability_about_1_over_p():
    """Small field p = 257: a TSP applying w' != w to (x, T) passes verification for ~1/p of the keys."""
    p = 257
    assert is_prime(p)
    rng = random.Random(3)
    trials, passes = 20000, 0
    for _ in range(trials):
        x = [1] + [rng.randrange(p) for _ in range(3)]
        w = [rng.randrange(p) for _ in range(4)]
        w2 = list(w)
        w2[rng.randrange(4)] += rng.randrange(1, p)
        key = rng.randbytes(16)
        u = rng.randrange(1, p)
        T = hommac.tags(x, key, 1, u, p)
        R2 = sum(a * b for a, b in zip(w2, x))
        T2 = sum(a * b for a, b in zip(w2, T))
        passes += hommac.verify(R2, T2, u, w, key, 1, p)
    rate = passes / trials
    assert abs(rate - 1 / p) < 4 * (1 / p / trials) ** 0.5 + 1e-3, rate


def test_second_tag_block_is_independent_and_both_blocks_are_checked():
    """Second tag block: block 1 uses the domain-separated labels (t, u, "alpha", 1) and (t, u, "r", i, 1); its
    keys differ from block 0; an honest record verifies in both blocks; a tag of one block never verifies as the other."""
    rng = random.Random(6)
    for _ in range(100):
        x, w = _record(rng)
        u = rng.randrange(1, P)
        (a0, r0), (a1, r1) = hommac.record_keys(K, 1, u, P, len(x), 0), hommac.record_keys(K, 1, u, P, len(x), 1)
        assert a0 != a1 and r0 != r1 and 1 <= a1 < P
        assert a1 == hommac.prf(K, P - 1, 1, u, "alpha", 1) + 1 and r1[2] == hommac.prf(K, P, 1, u, "r", 2, 1)
        T, T2 = hommac.tag_blocks(x, K, 1, u, P)
        R = sum(a * b for a, b in zip(w, x))
        T_R, T_R2 = sum(a * b for a, b in zip(w, T)), sum(a * b for a, b in zip(w, T2))
        assert hommac.verify_record(R, T_R, T_R2, u, w, K, 1, P)
        assert not hommac.verify_record(R, T_R, T_R, u, w, K, 1, P)       # block-0 tag in the block-1 slot
        assert not hommac.verify_record(R + 1, T_R, T_R2, u, w, K, 1, P)
    with pytest.raises(ValueError):
        hommac.record_keys(K, 1, 1, P, 3, 2)


def test_model_substitution_passes_both_blocks_with_probability_about_1_over_p_squared():
    """Small field p = 17: a TSP applying w' != w to (x, T, T') passes block 0 for ~1/(p-1) of the keys and both blocks
    for ~1/(p-1)^2 (independent one-time keys), i.e. ~2^-80 at p ~ 2^40."""
    p = 17
    rng = random.Random(8)
    trials, one, both = 60000, 0, 0
    for _ in range(trials):
        x = [1] + [rng.randrange(p) for _ in range(3)]
        w = [rng.randrange(p) for _ in range(4)]
        w2 = list(w)
        w2[rng.randrange(4)] += rng.randrange(1, p)
        key = rng.randbytes(16)
        u = rng.randrange(1, p)
        T, T2 = hommac.tag_blocks(x, key, 1, u, p)
        R2 = sum(a * b for a, b in zip(w2, x))
        TR, TR2 = sum(a * b for a, b in zip(w2, T)), sum(a * b for a, b in zip(w2, T2))
        ok0 = hommac.verify(R2, TR, u, w, key, 1, p)
        one += ok0
        both += ok0 and hommac.verify(R2, TR2, u, w, key, 1, p, block=1)
    r1, r2 = one / trials, both / trials
    assert abs(r1 - 1 / p) < 5 * (1 / p / trials) ** 0.5 + 2e-3, r1
    assert abs(r2 - 1 / p ** 2) < 5 * (1 / p ** 2 / trials) ** 0.5 + 5e-4, r2


def test_tags_of_another_record_or_vehicle_rejected():
    """Algebraic effect of swapping tags between records (same or different policyholder)."""
    rng = random.Random(5)
    K2 = secrets.token_bytes(32)
    for _ in range(200):
        (xa, w), (xb, _) = _record(rng), _record(rng)
        ua, ub = rng.randrange(1, P), rng.randrange(1, P)
        Ra = sum(a * b for a, b in zip(w, xa))
        Tb_same = sum(a * b for a, b in zip(w, hommac.tags(xb, K, 1, ub, P)))
        Tb_other = sum(a * b for a, b in zip(w, hommac.tags(xb, K2, 1, ub, P)))
        assert not hommac.verify(Ra, Tb_same, ua, w, K, 1, P)
        assert not hommac.verify(Ra, Tb_same, ub, w, K, 1, P)
        assert not hommac.verify(Ra, Tb_other, ub, w, K, 1, P)
