"""Dealer-free two-party generation of the BBS04 opening (tracing) key by the insurer and the TSP.

BBS04 needs h, u, v in G1 and the opening key (xi1, xi2) with xi1*u = xi2*v = h. Additive shares of xi1 and xi2 cannot
give u = h/xi1 directly, so the key is built from a base point P instead:

    P = hash_to_curve(session)                    (nobody knows a discrete logarithm of P)
    xi1 = a_I + a_T,  xi2 = b_I + b_T             (party k draws a_k, b_k; never sent)
    v = xi1*P = A_I + A_T,  u = xi2*P = B_I + B_T  (A_k = a_k*P, B_k = b_k*P)
    h = xi1*u = a_I*u + a_T*u = H_I + H_T          (and then xi2*v = xi1*xi2*P = h as well)

so (h, u, v, xi1, xi2) has the BBS04 key relation and the distribution of an honestly generated key, while each party
holds only (a_k, b_k). Opening stays the parallel 2-of-2 opening of speedcheck.py: party k returns a_k*T1 and b_k*T2
with DLEQ proofs against its share commitments (a_k*u, b_k*v) = (H_k, b_k*v), and the two commitments add up to h
(proofs.verify_opening_setup).

Messages (each party runs the same code; both must finish before the group key is used):
  1. commit    c_k = SHA-256(session | name_k | nonce_k | A_k | B_k)
  2. reveal    (nonce_k, A_k, B_k, PoK(a_k: A_k = a_k*P), PoK(b_k: B_k = b_k*P)), sent only after the peer's c_k arrived
  3. h share   (H_k = a_k*u, V_k = b_k*v, DLEQ(log_P A_k = log_u H_k), DLEQ(log_P B_k = log_v V_k)); (H_k, V_k) are
               party k's public share values
Checks by the receiver: the peer's reveal opens its commitment, both PoKs verify, u, v, h are not the identity, both
of the peer's DLEQ proofs verify (so H_k = a_k*u and V_k = b_k*v for the committed shares), and the two parties' share
values add up to (h, h) (proofs.verify_opening_setup; implied by the proofs, kept as the registration check). Any
failure aborts the setup before a key is output (a detected deviation).

Authentication (`AuthenticatedParty`, used by `run_two_party`): every message is signed by its sender with the
sender's long-term Schnorr signing key (the insurer's bundle-signing key, the TSP's receipt-signing key) over
(domain tag, session, sender, step, transcript hash, message), where the transcript hash covers both parties' messages
of every earlier step; the receiver verifies the signature before any other check. A second message for the same
step, or a message before the messages of an earlier step (a duplicate, a network replay, a reordering), aborts the
run as `OutOfOrder` and identifies nobody; a message whose signature fails aborts the run and identifies nobody (it may
come from the network); a degenerate key (u, v or h the identity; negligible probability with one honest party) aborts
the run as `Degenerate` and identifies nobody; a validly signed message that fails a check on its commitment or on one
of its proofs aborts the run and identifies its signer, and the signed messages of the run
(`AuthenticatedParty.evidence`, each bound to the transcript before it) are the evidence, which any third party can
check with `verify_blame` (every such check, the commitment length and opening, the proofs of knowledge and both DLEQ
proofs, is recomputable from the signed messages and the public keys). A party outputs a key only after it has sent
its own h share (`finish` before `h_share` is refused), so neither party can end the run while the other still waits
for a share.

Session identifiers: a party starts every run through its own `SessionRegistry`, which refuses a session identifier
the party has used before (a restart after an abort uses a new identifier) and returns a party whose first failed
check is final: after `SetupAborted` every further call of that run raises `SetupAborted`, so an aborted run never
outputs a key. `run_two_party` starts both parties through their registries (fresh ones unless the caller passes the
parties' own, as `protocol.run_epoch` does).
"""
import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import py_arkworks_bls12381 as ark

from .groupsig_bbs04 import OpenerShare, rand_scalar
from .proofs import (Proof, proof_bytes, proof_from_bytes, prove_dleq, prove_dlog, schnorr_sign, schnorr_verify,
                     verify_dleq, verify_dlog, verify_opening_setup)

G1 = ark.G1Point
DST = b"SP2UBI-BBS04-OPENING-BASE-V1"
NAMES = (b"insurer", b"tsp")


class SetupAborted(ValueError):
    """A peer's contribution failed a check; the setup produces no key."""


class OutOfOrder(SetupAborted):
    """A second message for the same step, or a message before its predecessor (a duplicate, a network replay or a
    reordering): the setup produces no key, and nobody is identified as deviating."""


class Degenerate(SetupAborted):
    """The key would be degenerate (u, v or h the identity). Both parties' shares were fixed by their commitments, so
    no third party can attribute it to either party; with one honest party it has negligible probability. The setup
    produces no key, and nobody is identified as deviating."""


def base_point(session: bytes) -> "G1":
    """P = hash_to_curve(msg = session, DST) of RFC 9380, suite BLS12381G1_XMD:SHA-256_SSWU_RO_ (constant domain
    separation tag, variable message)."""
    return G1.hash_to_curve(session, DST)


def _commit(session: bytes, name: bytes, nonce: bytes, A: "G1", B: "G1") -> bytes:
    h = hashlib.sha256(b"SP2UBI-open-dkg-commit|")
    for part in (session, name, nonce, A.to_compressed_bytes(), B.to_compressed_bytes()):
        h.update(len(part).to_bytes(4, "big") + part)
    return h.digest()


def _ctx(session: bytes, name: bytes, what: bytes) -> bytes:
    return b"open-dkg|" + session + b"|" + name + b"|" + what


@dataclass(frozen=True)
class Reveal:
    nonce: bytes
    A: "G1"
    B: "G1"
    pok_a: Proof
    pok_b: Proof


@dataclass(frozen=True)
class TracingKey:
    """Public part of the opening key: the BBS04 elements (h, u, v) and the base P they were built from."""
    P: "G1"
    h: "G1"
    u: "G1"
    v: "G1"


class OpeningKeyParty:
    """One party (insurer or TSP) of the two-party opening-key generation."""

    def __init__(self, name: bytes, session: bytes):
        if name not in NAMES:
            raise ValueError("party name must be b'insurer' or b'tsp'")
        self.name, self.peer = name, NAMES[1 - NAMES.index(name)]
        self.session = session
        self.P = base_point(session)
        self._a, self._b = rand_scalar(), rand_scalar()
        self._nonce = secrets.token_bytes(32)
        self.A, self.B = self.P * self._a, self.P * self._b
        self._peer_commit: Optional[bytes] = None
        self.u = self.v = None
        self._h_sent = False
        self._done = False

    # --- message 1
    def commitment(self) -> bytes:
        return _commit(self.session, self.name, self._nonce, self.A, self.B)

    def receive_commitment(self, c: bytes) -> None:
        if self._peer_commit is not None:
            raise OutOfOrder("second commitment")
        if len(c) != 32:
            raise SetupAborted("malformed commitment")
        self._peer_commit = c

    # --- message 2
    def reveal(self) -> Reveal:
        if self._peer_commit is None:
            raise PermissionError("reveal only after the peer's commitment has arrived")
        return Reveal(self._nonce, self.A, self.B,
                      prove_dlog(_ctx(self.session, self.name, b"a"), self.P, self.A, self._a),
                      prove_dlog(_ctx(self.session, self.name, b"b"), self.P, self.B, self._b))

    def receive_reveal(self, r: Reveal) -> None:
        if self._peer_commit is None or self.u is not None:
            raise OutOfOrder("reveal out of order")
        if _commit(self.session, self.peer, r.nonce, r.A, r.B) != self._peer_commit:
            raise SetupAborted("reveal does not open the peer's commitment")
        if not (verify_dlog(_ctx(self.session, self.peer, b"a"), self.P, r.A, r.pok_a)
                and verify_dlog(_ctx(self.session, self.peer, b"b"), self.P, r.B, r.pok_b)):
            raise SetupAborted("proof of knowledge of a peer share fails")
        u, v = self.B + r.B, self.A + r.A
        if u == G1.identity() or v == G1.identity():
            raise Degenerate("degenerate key (xi1 or xi2 = 0)")
        self.u, self.v, self._peer_A, self._peer_B = u, v, r.A, r.B

    # --- message 3
    def h_share(self) -> Tuple["G1", "G1", Proof, Proof]:
        """(H_k, V_k, DLEQ_h, DLEQ_v): the public share values H_k = a_k*u and V_k = b_k*v, with Chaum-Pedersen proofs
        that log_P A_k = log_u H_k and log_P B_k = log_v V_k."""
        if self.u is None:
            raise PermissionError("h share only after the peer's reveal")
        H, V = self.u * self._a, self.v * self._b
        self._h_sent = True
        return (H, V, prove_dleq(_ctx(self.session, self.name, b"h"), self.P, self.A, self.u, H, self._a),
                prove_dleq(_ctx(self.session, self.name, b"v"), self.P, self.B, self.v, V, self._b))

    def finish(self, H_peer: "G1", V_peer: "G1", proof_h: Proof, proof_v: Proof) -> Tuple[TracingKey, OpenerShare]:
        if self.u is None or self._done:
            raise OutOfOrder("h share out of order")
        if not self._h_sent:
            raise PermissionError("finish only after the party's own h share was produced")
        if not verify_dleq(_ctx(self.session, self.peer, b"h"), self.P, self._peer_A, self.u, H_peer, proof_h):
            raise SetupAborted("peer h share proof fails")
        if not verify_dleq(_ctx(self.session, self.peer, b"v"), self.P, self._peer_B, self.v, V_peer, proof_v):
            raise SetupAborted("peer share value proof fails")
        h = self.u * self._a + H_peer
        if h == G1.identity():
            raise Degenerate("degenerate key (h = identity)")
        key = TracingKey(self.P, h, self.u, self.v)
        own_Y, peer_Y = (self.u * self._a, self.v * self._b), (H_peer, V_peer)
        if not verify_opening_setup(key, own_Y, peer_Y):
            raise SetupAborted("share values do not add up to (h, h)")
        self._done = True
        share = OpenerShare(self._a, self._b, self.u, self.v, h, peer_Y)
        self._a = self._b = None                      # the share object keeps them; no second copy here
        return key, share


class _RegisteredParty(OpeningKeyParty):
    """`OpeningKeyParty` started through a `SessionRegistry`: the first `SetupAborted` ends the run for good."""

    def __init__(self, name: bytes, session: bytes):
        super().__init__(name, session)
        self.aborted = False


def _final_on_abort(step: str):
    base = getattr(OpeningKeyParty, step)

    def guarded(self, *args, **kwargs):
        if self.aborted:
            raise SetupAborted("this run was aborted; a restart needs a new session identifier")
        try:
            return base(self, *args, **kwargs)
        except SetupAborted:
            self.aborted = True
            self._a = self._b = None                      # the run's secret shares are discarded with the run
            raise
    guarded.__name__ = guarded.__qualname__ = step
    guarded.__doc__ = base.__doc__
    return guarded


for _step in ("commitment", "receive_commitment", "reveal", "receive_reveal", "h_share", "finish"):
    setattr(_RegisteredParty, _step, _final_on_abort(_step))
del _step


class SessionRegistry:
    """The session identifiers one party (insurer or TSP) has used for the opening-key generation. Each party keeps its
    own registry; an identifier is recorded when a run starts, whether the run later finishes or aborts."""

    def __init__(self, name: bytes):
        if name not in NAMES:
            raise ValueError("party name must be b'insurer' or b'tsp'")
        self.name = name
        self._used: Set[bytes] = set()

    def start(self, session: bytes) -> OpeningKeyParty:
        if not session:
            raise SetupAborted("empty session identifier")
        if session in self._used:
            raise SetupAborted("session identifier already used by this party")
        self._used.add(session)
        return _RegisteredParty(self.name, session)


# ------------------------------------------------------------------ authenticated messages (long-term signing keys)
AUTH_TAG = b"SP2UBI-open-dkg-msg|"
TRANSCRIPT_TAG = b"SP2UBI-open-dkg-transcript|"
SIG_BYTES = 64
STEPS = (b"commit", b"reveal", b"h")


def _enc(*parts) -> bytes:
    out = b""
    for p in parts:
        b = p if isinstance(p, (bytes, bytearray)) else p.to_compressed_bytes()
        out += len(b).to_bytes(4, "big") + b
    return out


def message_bytes(step: bytes, body) -> bytes:
    """Canonical bytes of a setup message body: commitment (32 B), reveal (nonce, A, B, two proofs), h share
    (H, V, two DLEQ proofs)."""
    if step == b"commit":
        return _enc(body)
    if step == b"reveal":
        return _enc(body.nonce, body.A, body.B, proof_bytes(body.pok_a), proof_bytes(body.pok_b))
    if step == b"h":
        H, V, pi_h, pi_v = body
        return _enc(H, V, proof_bytes(pi_h), proof_bytes(pi_v))
    raise ValueError("unknown step")


def transcript_hash(session: bytes, msgs, step: bytes) -> bytes:
    """SHA-256 over both parties' messages of every step before `step` (protocol order; within a step the insurer's
    first). msgs: {step: {name: Signed}}. Raises OutOfOrder if a message of an earlier step is missing."""
    h = hashlib.sha256(TRANSCRIPT_TAG + _enc(session))
    for s in STEPS[:STEPS.index(step)]:
        for name in NAMES:
            m = msgs.get(s, {}).get(name)
            if m is None:
                raise OutOfOrder("a message of an earlier step is missing")
            h.update(_enc(s, name, message_bytes(s, m.body)))
    return h.digest()


def _signed_message(session: bytes, sender: bytes, step: bytes, th: bytes, body) -> bytes:
    return AUTH_TAG + _enc(session, sender, step, th, message_bytes(step, body))


@dataclass(frozen=True)
class Signed:
    """A setup message with its sender's Schnorr signature (64 B) over the message and the transcript before it."""
    step: bytes
    body: object
    sig: bytes


def _sig_ok(vk, session: bytes, sender: bytes, step: bytes, th: bytes, m: Signed) -> bool:
    try:
        return (m.step == step and len(m.sig) == SIG_BYTES and schnorr_verify(
            vk, _signed_message(session, sender, step, th, m.body), proof_from_bytes(m.sig)))
    except Exception:
        return False


class AuthenticatedParty:
    """A party of the opening-key generation whose messages are signed with its long-term signing key (sk, vk) and
    whose received messages must carry a valid signature under the peer's long-term key peer_vk, bound to the
    transcript of the earlier steps.
    `blame` is None while no check failed; after a failed check it is the peer's name if a validly signed peer message
    failed a check on its commitment or on one of its proofs (`evidence` is then the run's signed messages,
    `transcript()`), and b"" if the message was a second message for the same step or out of order, its signature
    failed, or the key was degenerate (nobody identified)."""

    def __init__(self, party: OpeningKeyParty, sk, vk, peer_vk):
        self.party, self._sk, self.vk, self.peer_vk = party, sk, vk, peer_vk
        self.blame: Optional[bytes] = None
        self.evidence: Optional[Tuple[Tuple[bytes, bytes, Signed], ...]] = None
        self._msgs = {s: {} for s in STEPS}

    def transcript(self) -> Tuple[Tuple[bytes, bytes, Signed], ...]:
        """The run's signed messages so far, (step, sender, message), in protocol order."""
        return tuple((s, n, self._msgs[s][n]) for s in STEPS for n in NAMES if n in self._msgs[s])

    def _sign(self, step: bytes, body) -> Signed:
        try:
            th = transcript_hash(self.party.session, self._msgs, step)
        except OutOfOrder:
            raise PermissionError("the messages of the earlier steps are not complete")
        msg = _signed_message(self.party.session, self.party.name, step, th, body)
        m = Signed(step, body, proof_bytes(schnorr_sign(self._sk, self.vk, msg)))
        self._msgs[step][self.party.name] = m
        return m

    def _abort(self, blame: bytes, why: str, exc=SetupAborted):
        self.blame = blame
        if isinstance(self.party, _RegisteredParty):
            self.party.aborted = True
            self.party._a = self.party._b = None
        raise exc(why)

    def _receive(self, step: bytes, m: Signed, handler):
        if getattr(self.party, "aborted", False):
            raise SetupAborted("this run was aborted; a restart needs a new session identifier")
        peer = self.party.peer
        if peer in self._msgs[step]:
            self._abort(b"", "duplicate message (replay)", OutOfOrder)
        try:
            th = transcript_hash(self.party.session, self._msgs, step)
        except OutOfOrder as e:
            self._abort(b"", str(e), OutOfOrder)
        if not _sig_ok(self.peer_vk, self.party.session, peer, step, th, m):
            self._abort(b"", "message not signed by the peer's long-term key")
        self._msgs[step][peer] = m
        try:
            return handler(*m.body) if step == b"h" else handler(m.body)
        except OutOfOrder as e:
            self._abort(b"", str(e), OutOfOrder)
        except Degenerate as e:
            self._abort(b"", str(e), Degenerate)
        except SetupAborted as e:
            self.evidence = self.transcript()
            self._abort(peer, str(e))

    def commitment(self) -> Signed:
        return self._sign(b"commit", self.party.commitment())

    def receive_commitment(self, m: Signed) -> None:
        self._receive(b"commit", m, self.party.receive_commitment)

    def reveal(self) -> Signed:
        return self._sign(b"reveal", self.party.reveal())

    def receive_reveal(self, m: Signed) -> None:
        self._receive(b"reveal", m, self.party.receive_reveal)

    def h_share(self) -> Signed:
        return self._sign(b"h", self.party.h_share())

    def finish(self, m: Signed) -> Tuple[TracingKey, OpenerShare]:
        if not getattr(self.party, "aborted", False) and self.party.name not in self._msgs[b"h"]:
            raise PermissionError("finish only after the party's own h share was sent")
        return self._receive(b"h", m, self.party.finish)


def verify_blame(session: bytes, evidence, vks) -> bytes:
    """Third-party check of the evidence of an aborted run. evidence: (step, sender, Signed) triples in protocol order
    (`AuthenticatedParty.evidence`); vks: {name: long-term public key}. Every signature is verified against the
    transcript the evidence itself defines; the first validly signed message that fails a public check (commitment
    length or opening, proofs of knowledge, DLEQ of H_k or of V_k) identifies its sender, whose name is returned.
    Returns b"" if the evidence is not validly signed throughout, is out of protocol order, or proves no deviation
    (a degenerate key, u, v or h the identity, is attributable to nobody: with one honest party it has negligible
    probability)."""
    P = base_point(session)
    msgs = {s: {} for s in STEPS}
    for step, sender, m in evidence:
        if step not in STEPS or sender not in vks or sender in msgs[step]:
            return b""
        try:
            th = transcript_hash(session, msgs, step)
        except OutOfOrder:
            return b""
        if not _sig_ok(vks[sender], session, sender, step, th, m):
            return b""
        msgs[step][sender] = m
        try:
            if step == b"commit":
                if not isinstance(m.body, (bytes, bytearray)) or len(m.body) != 32:
                    return sender
            elif step == b"reveal":
                r = m.body
                if (_commit(session, sender, r.nonce, r.A, r.B) != msgs[b"commit"][sender].body
                        or not verify_dlog(_ctx(session, sender, b"a"), P, r.A, r.pok_a)
                        or not verify_dlog(_ctx(session, sender, b"b"), P, r.B, r.pok_b)):
                    return sender
            else:
                rv = [msgs[b"reveal"][n].body for n in NAMES]
                u, v = rv[0].B + rv[1].B, rv[0].A + rv[1].A
                if u == G1.identity() or v == G1.identity():
                    return b""
                own = msgs[b"reveal"][sender].body
                H, V, pi_h, pi_v = m.body
                if not (verify_dleq(_ctx(session, sender, b"h"), P, own.A, u, H, pi_h)
                        and verify_dleq(_ctx(session, sender, b"v"), P, own.B, v, V, pi_v)):
                    return sender
        except Exception:
            return sender
    return b""


def long_term_key():
    """A fresh Schnorr signing key pair (sk, vk = sk*G); stands in for a party's long-term key in simulations."""
    sk = rand_scalar()
    return sk, G1() * sk


def run_two_party(session: Optional[bytes] = None, keys=None, registries=None
                  ) -> Tuple[TracingKey, OpenerShare, OpenerShare]:
    """Both parties in one process (simulation, tests, benchmarks): authenticated messages exchanged in the protocol
    order. keys = ((sk_I, vk_I), (sk_T, vk_T)), the insurer's and the TSP's long-term signing keys (fresh pairs if
    omitted). registries = (SessionRegistry(b"insurer"), SessionRegistry(b"tsp")), each party's own record of used
    session identifiers (fresh registries if omitted): both parties are started through them, so a reused identifier
    is refused before any message. Returns (tracing key, insurer share, TSP share); both parties computed the same
    tracing key."""
    session = session or secrets.token_bytes(16)
    (sk_i, vk_i), (sk_t, vk_t) = keys or (long_term_key(), long_term_key())
    reg_i, reg_t = registries or (SessionRegistry(b"insurer"), SessionRegistry(b"tsp"))
    if reg_i.name != b"insurer" or reg_t.name != b"tsp":
        raise ValueError("registries = (insurer's, TSP's)")
    ins = AuthenticatedParty(reg_i.start(session), sk_i, vk_i, vk_t)
    tsp = AuthenticatedParty(reg_t.start(session), sk_t, vk_t, vk_i)
    ci, ct = ins.commitment(), tsp.commitment()
    ins.receive_commitment(ct)
    tsp.receive_commitment(ci)
    ins.receive_reveal(tsp.reveal())
    tsp.receive_reveal(ins.reveal())
    hi, ht = ins.h_share(), tsp.h_share()
    key_i, share_i = ins.finish(ht)
    key_t, share_t = tsp.finish(hi)
    if key_i != key_t:
        raise SetupAborted("the parties computed different keys")
    return key_i, share_i, share_t
