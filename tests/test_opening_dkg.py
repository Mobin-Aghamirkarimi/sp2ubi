"""Dealer-free two-party generation of the BBS04 opening key (opening_dkg.py): key
relation, joint opening, no single party opens, every malformed contribution aborts the setup before any key is output,
and a party refuses a session identifier it has used before."""
import dataclasses

import pytest

from sp2ubi.groupsig_bbs04 import BBS04, G1, Issuer, Signature, combine_opening, rand_scalar
from sp2ubi.opening_dkg import (OpeningKeyParty, Reveal, SessionRegistry, SetupAborted, TracingKey, base_point,
                                run_two_party)
from sp2ubi.proofs import prove_dleq, prove_dlog, verify_dleq, verify_opening_setup

SESSION = b"open-dkg-test"


def parties(session=SESSION):
    ins, tsp = OpeningKeyParty(b"insurer", session), OpeningKeyParty(b"tsp", session)
    ins.receive_commitment(tsp.commitment())
    tsp.receive_commitment(ins.commitment())
    return ins, tsp


def test_key_relation_and_distribution_of_roles():
    key, s_i, s_t = run_two_party(SESSION)
    assert key.P == base_point(SESSION)
    # xi1 * u = xi2 * v = h with xi1 = a_I + a_T, xi2 = b_I + b_T (the shares are only inside the share objects)
    assert key.u * (s_i._s1 + s_t._s1) == key.h and key.v * (s_i._s2 + s_t._s2) == key.h
    assert verify_opening_setup(Issuer(key).gpk, s_i.Y, s_t.Y)
    assert not verify_opening_setup(Issuer(key).gpk, s_i.Y, s_i.Y)
    # the issuer (insurer) built from the joint key holds no opening key
    iss = Issuer(key)
    assert iss._xi1 is None and iss._xi2 is None
    with pytest.raises(ValueError):
        iss.opener_key()
    # fresh sessions give independent keys
    assert run_two_party()[0].h != run_two_party()[0].h


def test_joint_opening_traces_the_signer_and_one_share_does_not():
    key, s_i, s_t = run_two_party()
    iss = Issuer(key)
    gs = BBS04(iss.gpk)
    members = [iss.enrol() for _ in range(4)]
    reg = {mk.A.to_compressed_bytes(): f"OBU-{i}" for i, mk in enumerate(members)}
    for i, mk in enumerate(members):
        s = Signature.from_bytes(gs.sign(mk, b"record"))
        assert gs.verify(b"record", s.to_bytes())
        (P1, P2), (p1, p2) = s_t.partial_with_proof(s.T1, s.T2, b"ctx")
        assert verify_dleq(b"ctx|o1", s_t.u, s_t.Y[0], s.T1, P1, p1) and verify_dleq(b"ctx|o2", s_t.v, s_t.Y[1], s.T2, P2, p2)
        assert reg.get(combine_opening(s.T3, s_i.partial(s.T1, s.T2), P1 + P2).to_compressed_bytes()) == f"OBU-{i}"
        assert reg.get(combine_opening(s.T3, s_i.partial(s.T1, s.T2), G1.identity()).to_compressed_bytes()) is None
        assert reg.get(combine_opening(s.T3, s_t.partial(s.T1, s.T2), G1.identity()).to_compressed_bytes()) is None


def test_reveal_only_after_the_peer_commitment():
    ins = OpeningKeyParty(b"insurer", SESSION)
    with pytest.raises(PermissionError):
        ins.reveal()
    with pytest.raises(PermissionError):
        ins.h_share()


def test_rogue_share_chosen_after_seeing_the_peer_is_rejected():
    """A TSP that, after seeing A_I, replaces its share by X - A_I (so that it would know xi1 = log_P X) cannot open
    its commitment (made before A_I was revealed), nor prove knowledge of the replaced share."""
    ins, tsp = parties()
    r_i = ins.reveal()
    x = rand_scalar()
    rogue_A = ins.P * x - r_i.A
    r_t = tsp.reveal()
    with pytest.raises(SetupAborted, match="commitment"):
        ins.receive_reveal(dataclasses.replace(r_t, A=rogue_A))
    # even with a matching (new) commitment the proof of knowledge fails: the TSP does not know log_P(rogue_A)
    ins2 = OpeningKeyParty(b"insurer", SESSION)
    tsp2 = OpeningKeyParty(b"tsp", SESSION)
    tsp2.A = rogue_A
    ins2.receive_commitment(tsp2.commitment())
    tsp2.receive_commitment(ins2.commitment())
    with pytest.raises(SetupAborted, match="proof of knowledge"):
        ins2.receive_reveal(tsp2.reveal())


def test_mirrored_or_replayed_contributions_are_rejected():
    ins, tsp = parties()
    r_i = ins.reveal()
    with pytest.raises(SetupAborted):                                      # TSP echoes the insurer's own reveal
        ins.receive_reveal(r_i)
    other_ins, other_tsp = parties(b"another-session")
    with pytest.raises(SetupAborted):                                      # reveal of another session
        ins.receive_reveal(other_tsp.reveal())
    with pytest.raises(SetupAborted):                                      # second commitment
        ins.receive_commitment(b"\0" * 32)


def test_wrong_proof_of_knowledge_is_rejected():
    ins, tsp = parties()
    r_t = tsp.reveal()
    bad = dataclasses.replace(r_t, pok_b=prove_dlog(b"wrong-context", tsp.P, r_t.B, rand_scalar()))
    with pytest.raises(SetupAborted, match="proof of knowledge"):
        ins.receive_reveal(bad)


def test_wrong_h_share_is_rejected():
    """A TSP that sends H_T = target - H_I (to fix h) or any H_T other than a_T * u fails the DLEQ check."""
    ins, tsp = parties()
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    H_i = ins.h_share()[0]
    H_t, V_t, pi_t, pv_t = tsp.h_share()
    with pytest.raises(SetupAborted, match="h share"):
        ins.finish(G1() * rand_scalar() - H_i, V_t, pi_t, pv_t)
    ins2, tsp2 = parties()
    ins2.receive_reveal(tsp2.reveal())
    tsp2.receive_reveal(ins2.reveal())
    ins2.h_share()
    H2, V2, pi2, pv2 = tsp2.h_share()
    with pytest.raises(SetupAborted, match="h share"):
        ins2.finish(H2 + G1(), V2, pi2, pv2)
    key, _ = ins.finish(H_t, V_t, pi_t, pv_t)                                    # the honest share is accepted
    assert key.h == H_i + H_t


def test_degenerate_key_is_rejected():
    """A share that cancels the peer's (only possible with knowledge of the peer's secret, simulated here) gives
    xi1 = 0, i.e. v = identity: the setup aborts."""
    ins, tsp = parties()
    tsp._a = -ins._a
    tsp.A = tsp.P * tsp._a
    ins2 = OpeningKeyParty(b"insurer", SESSION)
    ins2._a, ins2.A = ins._a, ins.A
    ins2.receive_commitment(tsp.commitment())
    with pytest.raises(SetupAborted, match="degenerate"):
        ins2.receive_reveal(tsp.reveal())


def test_reveal_type_is_complete():
    assert {f.name for f in dataclasses.fields(Reveal)} == {"nonce", "A", "B", "pok_a", "pok_b"}


# ------------------------------------------------------------------ abort before any key; session identifiers
def registered(session, regs=None):
    """Both parties started through their own session registries, commitments exchanged."""
    regs = regs or (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    ins, tsp = regs[0].start(session), regs[1].start(session)
    ins.receive_commitment(tsp.commitment())
    tsp.receive_commitment(ins.commitment())
    return ins, tsp, regs


def complete(ins, tsp):
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    hi, ht = ins.h_share(), tsp.h_share()
    return ins.finish(*ht), tsp.finish(*hi)


def _second_commitment(ins, tsp, made=None):
    ins.receive_commitment(tsp.commitment())


def _malformed_commitment(ins, tsp, made):
    fresh = SessionRegistry(b"insurer").start(SESSION + b"-malformed")
    made.append(fresh)
    fresh.receive_commitment(b"\0" * 31)


def _reveal_not_opening_commitment(ins, tsp, made=None):
    r = tsp.reveal()
    ins.receive_reveal(dataclasses.replace(r, nonce=bytes(32)))


def _reveal_wrong_pok(ins, tsp, made=None):
    r = tsp.reveal()
    ins.receive_reveal(dataclasses.replace(r, pok_a=prove_dlog(b"wrong-context", tsp.P, r.A, rand_scalar())))


def _reveal_mirrored(ins, tsp, made=None):
    ins.receive_reveal(ins.reveal())


def _degenerate(ins, tsp, made):
    tsp._a = -ins._a                                         # simulated knowledge of the peer's share
    tsp.A = tsp.P * tsp._a
    ins2 = SessionRegistry(b"insurer").start(SESSION + b"-degenerate")
    ins2._a, ins2.A = ins._a, ins.A
    tsp2 = SessionRegistry(b"tsp").start(SESSION + b"-degenerate")
    tsp2._a, tsp2.A, tsp2._nonce = tsp._a, tsp.A, tsp._nonce
    made += [ins2, tsp2]
    ins2.receive_commitment(tsp2.commitment())
    tsp2.receive_commitment(ins2.commitment())
    ins2.receive_reveal(tsp2.reveal())


def _finish_before_reveal(ins, tsp, made=None):
    ins.finish(G1(), G1(), None, None)


def _wrong_h_share(ins, tsp, made=None):
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    ins.h_share()
    H_t, V_t, pi_t, pv_t = tsp.h_share()
    ins.finish(H_t + G1(), V_t, pi_t, pv_t)


def _wrong_share_value(ins, tsp, made=None):
    """Valid DLEQ for H_T but a share value V_T other than b_T * v: its DLEQ proof fails."""
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    ins.h_share()
    H_t, V_t, pi_t, pv_t = tsp.h_share()
    ins.finish(H_t, V_t + G1(), pi_t, pv_t)


def _wrong_share_value_with_proof_for_it(ins, tsp, made=None):
    """A share value V_T' = b'*v other than b_T*v, with a valid DLEQ for b' but against B_T: the proof fails."""
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    ins.h_share()
    H_t, V_t, pi_t, _ = tsp.h_share()
    b2 = rand_scalar()
    from sp2ubi.opening_dkg import _ctx
    ins.finish(H_t, tsp.v * b2, pi_t, prove_dleq(_ctx(tsp.session, b"tsp", b"v"), tsp.P, tsp.P * b2, tsp.v,
                                                  tsp.v * b2, b2))


ABORT_PATHS = [_second_commitment, _malformed_commitment, _reveal_not_opening_commitment, _reveal_wrong_pok,
               _reveal_mirrored, _degenerate, _finish_before_reveal, _wrong_h_share, _wrong_share_value,
               _wrong_share_value_with_proof_for_it]


@pytest.mark.parametrize("path", ABORT_PATHS, ids=lambda f: f.__name__.strip("_"))
def test_failed_check_aborts_before_any_key_is_output(path):
    """Every check that fails raises SetupAborted before the call returns, so no TracingKey or OpenerShare is output;
    the run of a party started through its registry is then over: an honest continuation also raises, and the party no
    longer holds its secret shares."""
    ins, tsp, _ = registered(SESSION + b"-" + path.__name__.encode())
    made, out = [], None
    with pytest.raises(SetupAborted):
        out = path(ins, tsp, made)
    assert out is None
    aborted = [p for p in [ins, tsp] + made if p.aborted]
    assert len(aborted) == 1                                 # the party whose check failed, and only that one
    for party in aborted:
        assert party._a is None and party._b is None and not party._done
        for call in (lambda: party.receive_commitment(b"\1" * 32), party.reveal, party.h_share,
                     lambda: party.finish(G1(), G1(), None, None)):
            result = None
            with pytest.raises(SetupAborted, match="aborted"):
                result = call()
            assert not isinstance(result, (TracingKey, tuple))


def test_honest_continuation_after_an_abort_outputs_no_key():
    """After a rejected h share the honest h share is not accepted on the same run (registered party)."""
    ins, tsp, _ = registered(SESSION + b"-retry")
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    ins.h_share()
    H_t, V_t, pi_t, pv_t = tsp.h_share()
    with pytest.raises(SetupAborted, match="h share"):
        ins.finish(H_t + G1(), V_t, pi_t, pv_t)
    with pytest.raises(SetupAborted, match="aborted"):
        ins.finish(H_t, V_t, pi_t, pv_t)


def test_reused_session_identifier_is_refused():
    reg_i, reg_t = SessionRegistry(b"insurer"), SessionRegistry(b"tsp")
    s1 = SESSION + b"-reuse-1"
    (key_i, share_i), (key_t, share_t) = complete(*registered(s1, (reg_i, reg_t))[:2])
    assert key_i == key_t and verify_opening_setup(Issuer(key_i).gpk, share_i.Y, share_t.Y)
    for reg in (reg_i, reg_t):                               # each party refuses its own used identifier
        with pytest.raises(SetupAborted, match="already used"):
            reg.start(s1)
    # a run that aborts also consumes its identifier: the restart must use a new one
    s2 = SESSION + b"-reuse-2"
    ins, tsp, _ = registered(s2, (reg_i, reg_t))
    with pytest.raises(SetupAborted):
        _wrong_h_share(ins, tsp)
    with pytest.raises(SetupAborted, match="already used"):
        reg_i.start(s2)
    with pytest.raises(SetupAborted, match="already used"):
        reg_t.start(s2)
    (k2, _), (k2t, _) = complete(*registered(SESSION + b"-reuse-3", (reg_i, reg_t))[:2])
    assert k2 == k2t and k2.h != key_i.h
    with pytest.raises(SetupAborted):
        SessionRegistry(b"tsp").start(b"")
    with pytest.raises(ValueError):
        SessionRegistry(b"rsu")


# ------------------------------------------------------------------ RFC 9380 base point
QUUX_DST = b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_"
# RFC 9380, App. J.9.1 (suite BLS12381G1_XMD:SHA-256_SSWU_RO_), P = (P.x, P.y) for msg "" and msg "abc"
J91 = {b"": ("052926add2207b76ca4fa57a8734416c8dc95e24501772c814278700eed6d1e4e8cf62d9c09db0fac349612b759e79a1",
             "08ba738453bfed09cb546dbb0783dbb3a5f1f566ed67bb6be0e8c67e2e81a4cc68ee29813bb7994998f3eae0c9c6a265"),
       b"abc": ("03567bc5ef9c690c2ab2ecdf6a96ef1c139cc0b2f284dca0a9a7943388a49a3aee664ba5379a7655d3c68900be2f6903",
                "0b9c15f3fe6e5cf4211f346271d7b01c8f3b28be689c8429c85b67af215533311f0b8dfaaa154fa6b88176c229f2885d")}


def test_hash_to_curve_reproduces_rfc9380_j91_and_base_point_uses_rfc_argument_order():
    """The library's G1 hash_to_curve(msg, dst) reproduces the RFC 9380 J.9.1 vectors; base_point passes the session
    as the message and the constant DST as the domain separation tag (RFC 9380 usage)."""
    from sp2ubi.opening_dkg import DST
    for msg, (x, y) in J91.items():
        assert G1.hash_to_curve(msg, QUUX_DST).to_xy_bytes_be().hex() == x + y
    assert G1.hash_to_curve(QUUX_DST, b"").to_xy_bytes_be().hex()[:96] != J91[b""][0]     # swapped order differs
    for session in (SESSION, b"another-session"):
        assert base_point(session) == G1.hash_to_curve(session, DST)
        assert base_point(session) != G1.hash_to_curve(DST, session)
    assert base_point(SESSION) != base_point(b"another-session")


# ------------------------------------------------------------------ sum check inside the setup and at registration
def test_setup_checks_share_values_and_insurer_registration_refuses_others():
    """Each party ends the setup only if the two share values add up to (h, h); the shares carry h and the peer's
    checked values, and the insurer's speed-path verifier refuses TSP share values that do not add up."""
    from sp2ubi.elgamal import ThresholdKey
    from sp2ubi.speedcheck import InsurerSpeedVerifier
    key, s_i, s_t = run_two_party()
    assert s_i.h == s_t.h == key.h
    assert s_i.peer_Y == s_t.Y and s_t.peer_Y == s_i.Y
    assert verify_opening_setup(key, s_i.Y, s_t.Y)
    k_ins, k_tsp = ThresholdKey(34, 10, b"insurer"), ThresholdKey(34, 10, b"tsp")
    InsurerSpeedVerifier(k_ins, s_i, {}, (k_tsp.share_pub, s_t.Y))             # the checked values register
    for bad in ((s_t.Y[0], s_t.Y[1] + G1()), (s_t.Y[0] + G1(), s_t.Y[1]), s_i.Y):
        with pytest.raises(ValueError, match="add up"):
            InsurerSpeedVerifier(k_ins, s_i, {}, (k_tsp.share_pub, bad))


# ------------------------------------------------------------------ authenticated setup messages
def auth_pair(session, keys=None, registries=None):
    from sp2ubi.opening_dkg import AuthenticatedParty, long_term_key
    (ki, kt) = keys or (long_term_key(), long_term_key())
    if registries:
        pi, pt = registries[0].start(session), registries[1].start(session)
    else:
        pi, pt = OpeningKeyParty(b"insurer", session), OpeningKeyParty(b"tsp", session)
    return AuthenticatedParty(pi, ki[0], ki[1], kt[1]), AuthenticatedParty(pt, kt[0], kt[1], ki[1]), (ki, kt)


def test_authenticated_setup_completes_with_the_long_term_keys():
    from sp2ubi.opening_dkg import long_term_key
    keys = (long_term_key(), long_term_key())
    key, s_i, s_t = run_two_party(SESSION + b"-auth", keys=keys)
    assert verify_opening_setup(Issuer(key).gpk, s_i.Y, s_t.Y)


def test_unsigned_or_wrongly_signed_message_aborts_and_identifies_nobody():
    """A message signed with another key, a modified signed message, a signed message presented as another step, a
    truncated signature, or a signed message replayed from another session fails the signature check: the run aborts
    before any key and no party is blamed (the message may come from the network)."""
    from sp2ubi.opening_dkg import Signed
    s = SESSION + b"-auth-bad"
    ins, tsp, keys = auth_pair(s)
    c_t = tsp.commitment()
    forger = auth_pair(s)[1]                                   # same name and session, another long-term key
    for bad in (forger.commitment(), Signed(b"commit", b"\0" * 32, c_t.sig), Signed(b"reveal", c_t.body, c_t.sig),
                Signed(b"commit", c_t.body, c_t.sig[:-1])):
        receiver = auth_pair(s, keys=keys)[0]
        with pytest.raises(SetupAborted, match="not signed"):
            receiver.receive_commitment(bad)
        assert receiver.blame == b"" and receiver.evidence is None and receiver.party.u is None
    other_tsp = auth_pair(SESSION + b"-auth-other", keys=keys)[1]     # same keys, another session: replay fails
    with pytest.raises(SetupAborted, match="not signed"):
        ins.receive_commitment(other_tsp.commitment())
    assert ins.blame == b""
    ins2 = auth_pair(s, keys=keys)[0]                          # the genuine message is accepted
    ins2.receive_commitment(c_t)


def _run_until_h(s, regs=None, keys=None):
    ins, tsp, keys = auth_pair(s, keys=keys, registries=regs)
    ci, ct = ins.commitment(), tsp.commitment()
    ins.receive_commitment(ct)
    tsp.receive_commitment(ci)
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    return ins, tsp, keys


def test_signed_deviation_aborts_and_identifies_the_signer_with_the_transcript():
    """A TSP that signs a wrong share value (valid DLEQ for H_T, V_T other than b_T * v): the insurer aborts before any
    key, blames the TSP and keeps the run's signed messages as evidence; a third party with only the public keys
    confirms the blame (verify_blame), and altered evidence identifies nobody."""
    from sp2ubi.opening_dkg import Signed, verify_blame
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    s = SESSION + b"-auth-blame"
    ins, tsp, (ki, kt) = _run_until_h(s, regs)
    ins.h_share()                                                        # the insurer's own h share is sent first
    honest = tsp.party.h_share
    tsp.party.h_share = lambda: (lambda H, V, ph, pv: (H, V + G1(), ph, pv))(*honest())
    m = tsp.h_share()
    out = None
    with pytest.raises(SetupAborted, match="share value proof"):
        out = ins.finish(m)
    assert out is None and ins.blame == b"tsp"
    ev = ins.evidence
    assert [(st, n) for st, n, _ in ev] == [(st, n) for st in (b"commit", b"reveal", b"h") for n in (b"insurer", b"tsp")]
    assert ev[-1][2] is m
    vks = {b"insurer": ki[1], b"tsp": kt[1]}
    assert verify_blame(s, ev, vks) == b"tsp"
    assert verify_blame(s, ev, {b"insurer": ki[1], b"tsp": ki[1]}) == b""          # wrong key: nobody
    assert verify_blame(s, ev[:-2] + (ev[-1], ev[-2]), vks) == b"tsp"               # order within a step is free
    assert verify_blame(s, (ev[2],) + ev[:2] + ev[3:], vks) == b""                   # a reveal before the commits
    assert verify_blame(s, ev[:-1], vks) == b""                                      # without the failing message
    st, n, m0 = ev[1]
    forged = ev[:1] + ((st, n, Signed(m0.step, bytes(32), m0.sig)),) + ev[2:]      # altered TSP commitment
    assert verify_blame(s, forged, vks) == b""
    assert verify_blame(s + b"x", ev, vks) == b""                                    # another session
    assert ins.party.aborted and ins.party._a is None                 # registered run ended for good
    with pytest.raises(SetupAborted, match="aborted"):
        ins.finish(m)


def test_honest_transcript_proves_no_deviation():
    from sp2ubi.opening_dkg import verify_blame
    s = SESSION + b"-auth-honest"
    ins, tsp, (ki, kt) = _run_until_h(s)
    hi, ht = ins.h_share(), tsp.h_share()
    ins.finish(ht)
    tsp.finish(hi)
    assert ins.blame is None and tsp.blame is None
    assert verify_blame(s, ins.transcript(), {b"insurer": ki[1], b"tsp": kt[1]}) == b""
    assert ins.transcript() == tsp.transcript()


def test_replayed_or_out_of_order_message_aborts_and_identifies_nobody():
    """A genuine signed message delivered twice (network replay), or a message before those of the earlier step,
    aborts the run as OutOfOrder before any key, and nobody is blamed."""
    from sp2ubi.opening_dkg import OutOfOrder
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    s = SESSION + b"-replay"
    ins, tsp, keys = auth_pair(s, registries=regs)
    ins.commitment()
    c_t = tsp.commitment()
    ins.receive_commitment(c_t)
    with pytest.raises(OutOfOrder):
        ins.receive_commitment(c_t)                                   # replay of the TSP's genuine commitment
    assert ins.blame == b"" and ins.evidence is None and ins.party.aborted and ins.party._a is None
    # a reveal before the commitments: no blame
    ins2, tsp2, _ = auth_pair(s + b"-2", keys=keys)
    c_i2, c_t2 = ins2.commitment(), tsp2.commitment()
    tsp2.receive_commitment(c_i2)
    r_t2 = tsp2.reveal()
    with pytest.raises(OutOfOrder):
        ins2.receive_reveal(r_t2)                                     # TSP's commitment never arrived at the insurer
    assert ins2.blame == b""
    # a replayed reveal and a replayed h share: no blame
    ins3, tsp3, _ = _run_until_h(s + b"-3", keys=keys)
    r_dup = tsp3.transcript()[3][2]                                   # the TSP's own signed reveal
    with pytest.raises(OutOfOrder):
        ins3.receive_reveal(r_dup)
    assert ins3.blame == b""
    ins4, tsp4, _ = _run_until_h(s + b"-4", keys=keys)
    ins4.h_share()
    m4 = tsp4.h_share()
    ins4.finish(m4)
    with pytest.raises(SetupAborted):
        ins4.finish(m4)
    assert ins4.blame == b""                                          # duplicate after the end: nobody identified


def test_signature_failure_of_a_registered_party_drops_its_shares():
    from sp2ubi.opening_dkg import Signed
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    ins, tsp, _ = auth_pair(SESSION + b"-sigfail", registries=regs)
    ins.commitment()
    c_t = tsp.commitment()
    with pytest.raises(SetupAborted, match="not signed"):
        ins.receive_commitment(Signed(b"commit", c_t.body, bytes(64)))
    assert ins.blame == b"" and ins.party.aborted and ins.party._a is None and ins.party._b is None


def test_run_two_party_uses_the_session_registries():
    """run_two_party starts both parties through their registries: a reused identifier is refused before any
    message, a new one completes; mismatched registries are refused."""
    from sp2ubi.opening_dkg import long_term_key
    keys = (long_term_key(), long_term_key())
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    s = SESSION + b"-registry-path"
    key, _, _ = run_two_party(s, keys=keys, registries=regs)
    with pytest.raises(SetupAborted, match="already used"):
        run_two_party(s, keys=keys, registries=regs)
    key2, _, _ = run_two_party(s + b"-new", keys=keys, registries=regs)
    assert key2.h != key.h
    with pytest.raises(SetupAborted, match="already used"):              # the TSP alone also refuses the reuse
        run_two_party(s, keys=keys, registries=(SessionRegistry(b"insurer"), regs[1]))
    with pytest.raises(ValueError):
        run_two_party(s + b"-x", keys=keys, registries=(regs[1], regs[0]))


def test_protocol_path_starts_both_parties_through_their_registries(monkeypatch):
    """protocol.run_epoch runs the opening-key generation through the insurer's and the TSP's own registries, the TSP
    object keeps the TSP's registry, and the TSP signs its setup messages with its receipt-signing key."""
    import sp2ubi.protocol as proto
    seen = {}
    real = proto.run_two_party

    def spy(*a, **kw):
        seen["registries"], seen["keys"] = kw.get("registries"), kw.get("keys")
        return real(*a, **kw)
    monkeypatch.setattr(proto, "run_two_party", spy)
    built = {}
    real_tsp = proto.TSP

    class SpyTSP(real_tsp):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built["tsp"] = self
    monkeypatch.setattr(proto, "TSP", SpyTSP)
    proto.run_epoch(n_policies=1, days=2)
    reg_i, reg_t = seen["registries"]
    assert reg_i.name == b"insurer" and reg_t.name == b"tsp" and len(reg_i._used) == len(reg_t._used) == 1
    assert built["tsp"].dkg_sessions is reg_t
    assert seen["keys"][1][1] == built["tsp"].vk                      # setup messages and receipts: the same TSP key


def test_message_sizes_per_party():
    """Commit 32 B, reveal 32 + 2*48 + 2*64 B, h share 2*48 + 2*64 B, each with a 64-B signature: 704 B per party."""
    from sp2ubi.opening_dkg import message_bytes
    ins, tsp, _ = _run_until_h(SESSION + b"-sizes")
    h = ins.h_share()
    tr = ins.transcript()
    body = {st: len(message_bytes(st, m.body)) - 4 * {b"commit": 1, b"reveal": 5, b"h": 4}[st]
            for st, n, m in tr if n == b"insurer"}
    body[b"h"] = len(message_bytes(b"h", h.body)) - 16
    assert body == {b"commit": 32, b"reveal": 256, b"h": 224}
    assert sum(body.values()) + 3 * len(h.sig) == 704


# ------------------------------------------------------------------ attribution at the authenticated layer
def _tsp_commit_31_bytes(tsp):
    tsp.party.commitment = lambda: b"\0" * 31


def _tsp_reveal_not_opening(tsp):
    honest = tsp.party.reveal
    tsp.party.reveal = lambda: dataclasses.replace(honest(), nonce=bytes(32))


def _tsp_reveal_wrong_pok(tsp):
    honest = tsp.party.reveal

    def bad():
        r = honest()
        return dataclasses.replace(r, pok_a=prove_dlog(b"wrong-context", tsp.party.P, r.A, rand_scalar()))
    tsp.party.reveal = bad


def _tsp_wrong_h_share(tsp):
    honest = tsp.party.h_share
    tsp.party.h_share = lambda: (lambda H, V, ph, pv: (H + G1(), V, ph, pv))(*honest())


@pytest.mark.parametrize("deviation, step", [(_tsp_commit_31_bytes, b"commit"), (_tsp_reveal_not_opening, b"reveal"),
                                             (_tsp_reveal_wrong_pok, b"reveal"), (_tsp_wrong_h_share, b"h")],
                         ids=lambda x: x.__name__.strip("_") if callable(x) else x.decode())
def test_signed_deviation_at_each_step_is_blamed_locally_and_by_a_third_party(deviation, step):
    """A TSP that signs a commitment of 31 B, a reveal that does not open its commitment, a reveal with a wrong proof of
    knowledge, or an H_T without a valid DLEQ proof (with a valid V_T): the insurer aborts before any key and blames
    the TSP, and `verify_blame` on the insurer's evidence returns the same party."""
    from sp2ubi.opening_dkg import verify_blame
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    s = SESSION + b"-auth-step-" + deviation.__name__.encode()
    ins, tsp, (ki, kt) = auth_pair(s, registries=regs)
    deviation(tsp)
    out = None
    with pytest.raises(SetupAborted) as exc:
        ci, ct = ins.commitment(), tsp.commitment()
        ins.receive_commitment(ct)
        tsp.receive_commitment(ci)
        ins.receive_reveal(tsp.reveal())
        tsp.receive_reveal(ins.reveal())
        ins.h_share()
        out = ins.finish(tsp.h_share())
    assert out is None and type(exc.value) is SetupAborted
    assert ins.blame == b"tsp" and ins.evidence[-1][:2] == (step, b"tsp")
    assert verify_blame(s, ins.evidence, {b"insurer": ki[1], b"tsp": kt[1]}) == b"tsp"
    assert ins.party.aborted and ins.party._a is None


def test_degenerate_key_aborts_and_identifies_nobody_locally_or_by_a_third_party():
    """A TSP that sets b_T = -b_I before committing (simulated knowledge of the insurer's share) makes u the identity.
    Both shares were fixed by their commitments, so the degeneracy is attributable to nobody: the insurer aborts before
    any key with blame b"" and no evidence, and `verify_blame` on the run's signed messages also returns b""."""
    from sp2ubi.opening_dkg import Degenerate, verify_blame
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    s = SESSION + b"-auth-degenerate"
    ins, tsp, (ki, kt) = auth_pair(s, registries=regs)
    tsp.party._b = -ins.party._b
    tsp.party.B = tsp.party.P * tsp.party._b
    ci, ct = ins.commitment(), tsp.commitment()
    ins.receive_commitment(ct)
    tsp.receive_commitment(ci)
    out = None
    with pytest.raises(Degenerate):
        out = ins.receive_reveal(tsp.reveal())
    assert out is None and ins.party.u is None
    assert ins.blame == b"" and ins.evidence is None
    assert verify_blame(s, ins.transcript(), {b"insurer": ki[1], b"tsp": kt[1]}) == b""
    assert ins.party.aborted and ins.party._a is None and ins.party._b is None


def test_finish_is_refused_before_the_party_has_sent_its_own_h_share():
    """A party that receives the peer's h share before sending its own may not output a key yet (it could no longer
    send its share, and the peer would never finish): `finish` raises PermissionError, nothing is recorded or aborted,
    and after `h_share` both parties finish with the same key (plain and authenticated parties)."""
    ins, tsp = parties(SESSION + b"-finish-order")
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    ht = tsp.h_share()
    with pytest.raises(PermissionError, match="own h share"):
        ins.finish(*ht)
    assert ins._a is not None and not ins._done
    hi = ins.h_share()
    (key_i, _), (key_t, _) = ins.finish(*ht), tsp.finish(*hi)
    assert key_i == key_t
    regs = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    a_ins, a_tsp, _ = _run_until_h(SESSION + b"-finish-order-auth", regs)
    m_t = a_tsp.h_share()
    with pytest.raises(PermissionError, match="own h share"):
        a_ins.finish(m_t)
    assert b"tsp" not in a_ins._msgs[b"h"] and a_ins.blame is None and not a_ins.party.aborted
    m_i = a_ins.h_share()
    (k_i, _), (k_t, _) = a_ins.finish(m_t), a_tsp.finish(m_i)
    assert k_i == k_t
