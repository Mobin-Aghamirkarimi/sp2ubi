import random

from sp2ubi import hommac
from sp2ubi.fhe import SLOT_ID, SLOT_T, SLOT_T2, SLOT_U, SLOT_X, SLOTS, RiskContext
from sp2ubi.params import DEFAULT

P, M = DEFAULT.p, DEFAULT.m


def test_parameters_are_the_selected_ones():
    """Single insurer key: OpenFHE's choice for t = p (41 bits), depth 1, HEStd_128_classic."""
    info = RiskContext(P, M).info()
    assert info["N"] == 8192 and info["log2_Q"] == 120.0 and info["t_bits"] == 41
    assert info["moduli_bits"] == [60.0] * 2 and info["security"] == "HEStd_128_classic"


def test_single_key_evaluation_matches_plaintext_and_hommac():
    rng = random.Random(4)
    risk = RiskContext(P, M)
    kp = risk.keygen()
    w = [rng.randrange(0, 1 << 22) for _ in range(M)]
    ct_w = risk.encrypt_weights(kp.publicKey, w)
    K = rng.randbytes(32)
    for _ in range(5):
        x = [rng.randrange(1, 32)] + [rng.randrange(1 << 19) for _ in range(M - 1)]
        u, ident = rng.randrange(1, P), rng.getrandbits(32)
        T, T2 = hommac.tag_blocks(x, K, 1, u, P)
        ct = risk.fill(risk.evaluate(risk.encrypt_payload(kp.publicKey, x, T, u, ident, T2), ct_w), kp.publicKey)
        d = risk.decrypt(ct, kp.secretKey)
        assert d[SLOT_X] == sum(a * b for a, b in zip(w, x)) % P
        assert d[SLOT_T] == sum(a * b for a, b in zip(w, T)) % P
        assert d[SLOT_T2] == sum(a * b for a, b in zip(w, T2)) % P
        assert (d[SLOT_U], d[SLOT_ID]) == (u, ident)
        assert hommac.verify_record(d[SLOT_X], d[SLOT_T], d[SLOT_T2], u, w, K, 1, P)
        assert not hommac.verify_record((d[SLOT_X] + 1) % P, d[SLOT_T], d[SLOT_T2], u, w, K, 1, P)


def test_another_key_does_not_decrypt():
    risk = RiskContext(P, M)
    kp, other = risk.keygen(), risk.keygen()
    v = [random.randrange(P) for _ in range(SLOTS)]
    ct = risk.encrypt_vector(kp.publicKey, v)
    assert risk.decrypt(ct, kp.secretKey) == v and risk.decrypt(ct, other.secretKey) != v

