import pytest

from sp2ubi.groupsig_bbs04 import BBS04, SIGNATURE_BYTES, Issuer, Signature, member_key_valid


def _setup(n=3):
    issuer = Issuer()
    members = [issuer.enrol() for _ in range(n)]
    opener = issuer.opener_key()
    for i, mk in enumerate(members):
        opener.register(f"OBU-{i}", mk.A)
    return issuer, opener, members, BBS04(issuer.gpk)


def test_member_credentials_valid():
    issuer, _, members, _ = _setup()
    assert all(member_key_valid(issuer.gpk, mk) for mk in members)


def test_sign_verify_and_size():
    _, _, members, gs = _setup()
    sig = gs.sign(members[0], b"ciphertext||commitment||ID_Ins")
    assert len(sig) == SIGNATURE_BYTES == 336
    assert gs.verify(b"ciphertext||commitment||ID_Ins", sig)


def test_wrong_message_or_modified_signature_rejected():
    _, _, members, gs = _setup()
    sig = gs.sign(members[1], b"m")
    assert not gs.verify(b"m'", sig)
    for pos in range(0, SIGNATURE_BYTES, 16):             # flip one byte in every point and every scalar
        bad = bytearray(sig)
        bad[pos] ^= 1
        assert not gs.verify(b"m", bytes(bad))
    assert not gs.verify(b"m", sig[:-1])


def test_signature_from_another_group_rejected():
    _, _, members_a, gs_a = _setup()
    _, _, _, gs_b = _setup()
    sig = gs_a.sign(members_a[0], b"m")
    assert gs_a.verify(b"m", sig) and not gs_b.verify(b"m", sig)
    assert not gs_b.verify(b"m", gs_b.sign(members_a[0], b"m"))    # a credential of group A under the key of group B


def test_signatures_are_randomized():
    _, _, members, gs = _setup()
    assert gs.sign(members[0], b"m") != gs.sign(members[0], b"m")


def test_opener_traces_the_signer_and_issuer_cannot():
    issuer, opener, members, gs = _setup(4)
    for i, mk in enumerate(members):
        assert opener.open(Signature.from_bytes(gs.sign(mk, b"daily record"))) == f"OBU-{i}"
    assert issuer._xi1 is None and issuer._xi2 is None    # opening key handed over and deleted


def test_non_member_cannot_sign():
    """A credential not issued by the issuer (random A, x) must not yield valid signatures."""
    from sp2ubi.groupsig_bbs04 import G1, MemberKey, rand_scalar
    issuer, _, _, gs = _setup()
    fake = MemberKey(G1() * rand_scalar(), rand_scalar())
    assert not member_key_valid(issuer.gpk, fake)
    for _ in range(20):
        assert not gs.verify(b"m", gs.sign(fake, b"m"))


# ------------------------------------------------------------------ random generators in the group public key
def test_issuer_draws_random_generators_and_publishes_them():
    """g1, g2 are uniform non-identity elements drawn by each issuer (not the fixed generators of the library) and are
    part of the group public key; credentials, signatures and openings work with them."""
    from sp2ubi.groupsig_bbs04 import G1, G2
    a, b = Issuer(), Issuer()
    for gpk in (a.gpk, b.gpk):
        assert gpk.g1 != G1.identity() and gpk.g2 != G2.identity()
        assert gpk.g1 != G1() and gpk.g2 != G2()
        assert gpk.g1.is_in_subgroup() and gpk.g2.is_in_subgroup()
    assert a.gpk.g1 != b.gpk.g1 and a.gpk.g2 != b.gpk.g2           # two issuers get different generators
    issuer, opener, members, gs = _setup(3)
    assert all(member_key_valid(issuer.gpk, mk) for mk in members)
    for i, mk in enumerate(members):
        sig = gs.sign(mk, b"record")
        assert gs.verify(b"record", sig)
        assert opener.open(Signature.from_bytes(sig)) == f"OBU-{i}"


def test_jointly_generated_tracing_key_with_random_generators():
    """Same (h, u, v) from the dealer-free setup, two issuers: different g1, g2, w; a signature of one group does not
    verify in the other; both open with the joint shares."""
    from sp2ubi.groupsig_bbs04 import combine_opening
    from sp2ubi.opening_dkg import run_two_party
    key, s_i, s_t = run_two_party()
    i1, i2 = Issuer(key), Issuer(key)
    assert (i1.gpk.h, i1.gpk.u, i1.gpk.v) == (i2.gpk.h, i2.gpk.u, i2.gpk.v) == (key.h, key.u, key.v)
    assert i1.gpk.g1 != i2.gpk.g1 and i1.gpk.g2 != i2.gpk.g2 and i1.gpk.w != i2.gpk.w
    mk = i1.enrol()
    assert member_key_valid(i1.gpk, mk) and not member_key_valid(i2.gpk, mk)
    sig = BBS04(i1.gpk).sign(mk, b"m")
    assert BBS04(i1.gpk).verify(b"m", sig) and not BBS04(i2.gpk).verify(b"m", sig)
    s = Signature.from_bytes(sig)
    assert combine_opening(s.T3, s_i.partial(s.T1, s.T2), s_t.partial(s.T1, s.T2)) == mk.A


def test_group_public_key_serialization_round_trip_and_sizes():
    from sp2ubi.groupsig_bbs04 import G1, GPK_BYTES, GroupPublicKey
    issuer, _, members, gs = _setup(1)
    b = issuer.gpk.to_bytes()
    assert len(b) == GPK_BYTES == 4 * 48 + 2 * 96 == 384
    gpk = GroupPublicKey.from_bytes(b)
    assert gpk == issuer.gpk and gpk.to_bytes() == b
    sig = gs.sign(members[0], b"m")
    assert len(sig) == SIGNATURE_BYTES == 336
    assert BBS04(gpk).verify(b"m", sig)
    with pytest.raises(ValueError):
        GroupPublicKey.from_bytes(b[:-1])
    ident = G1.identity().to_compressed_bytes()
    with pytest.raises(ValueError):                                # identity generator refused
        GroupPublicKey.from_bytes(ident + b[48:])


def test_malformed_signature_encodings_are_rejected():
    _, _, members, gs = _setup(1)
    sig = gs.sign(members[0], b"m")
    assert Signature.from_bytes(sig).to_bytes() == sig
    not_reduced = sig[:-32] + b"\xff" * 32                          # last scalar >= the group order
    no_point = b"\x9f" + b"\xff" * 47 + sig[48:]                      # x coordinate >= the field modulus
    for bad in (sig[:-1], sig + b"\0", not_reduced, no_point):
        with pytest.raises(ValueError):
            Signature.from_bytes(bad)
        assert not gs.verify(b"m", bad)
