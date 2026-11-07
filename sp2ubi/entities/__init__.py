"""The four SP2UBI entities on the risk path: OBU, RSU, TSP and insurer. Each object holds only the keys that the key
distribution of the scheme assigns to it.

* OBU: holds the insurer-signed public key pk_Ins (`PublicKeyCert`: key, key tag, version, epoch range) and receives neither Enc(w) nor
  the evaluation keys. It produces one encrypted record per aggregation interval (Params.interval_days; 1 = daily, 31 = monthly) with
  x_0 = insured days covered, a random nonce (redrawn if already sealed in the epoch) and a dispute salt per record in sealed storage
  (identical re-send), and two HomMAC tag blocks. Each record goes on its own over the vehicle's cellular (Uu) link, a fresh connection per
  attempt, at an independent random time, and is re-sent until the TSP's receipt or the settlement deadline. The cellular link is an
  untrusted pipe: the record is encrypted and group-signed, and the OBU verifies the receipt.
* RSU: not on the risk path (RSUs do not relay risk records); holds no FHE key. Its role on the speed-verification path is
  `speedcheck.SensingRSU`.
* TSP: holds the public key, the public evaluation keys and Enc(w) (inside the insurer-signed bundle) and no secret FHE key and no policy.
  It evaluates, fills every non-output slot, signs a receipt for every accepted record, shuffles and publishes its output to the insurer.
  The insurer receives only the published, shuffled output.
* Insurer: holds the risk-path secret key, w and one HomMAC key K per policy; decrypts the published outputs, checks HomMAC before the range
  check and gives one identical outcome for every rejected record. It sends the TSP no verdict; the only outcome toward the TSP is one bit per
  epoch (`Insurer.deviation_notice`).

Settlement: the TSP gives no receipt for a record of an epoch whose batch it has published or that no key bundle it holds covers, so a
receipted record that is not accepted is always a deviation of the TSP (`TSP.receive`). Records of an ended policy are discarded, not rejected
(`Insurer.end_policy`). The dispute commitment cm_r commits to the record without its interval index and capped days (kept in sealed storage), so a
dispute discloses an undated feature vector. Settlement and dispute run between the insurer and the OBU (`Insurer.settlement_message`,
`OBU.settlement_upload`); the TSP's receipt signs the record digest, cm_r, cm_b and t (`receipt_message`), the OBU keeps the receipt with these
values (`Receipted`) and the dispute opens cm_r against the receipted value (`Insurer.dispute_record`). In a conflict epoch (more distinct
valid records than expected) the policy-epoch yields no score and the settlement message requests no evidence (`NO_EVIDENCE`). A record that
decrypts to an ended policy is verified under that policy's key first: only a record that verifies is discarded, one that fails is rejected. Settlement messages and
uploads are authenticated and encrypted end to end between the insurer and the OBU's TEE (channel not modelled), carry no padding and are not sent at fixed times.
"""
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import openfhe as o

from .. import hommac
from ..commitment import commit, encode_record, open_commitment
from ..fhe import SLOT_ID, SLOT_T, SLOT_T2, SLOT_U, SLOT_X, RiskContext
from ..groupsig import GroupSignature
from ..groupsig_bbs04 import rand_scalar
from ..opening_dkg import SessionRegistry
from ..params import Params
from ..proofs import proof_bytes, proof_from_bytes, schnorr_sign, schnorr_verify

import py_arkworks_bls12381 as ark

G1 = ark.G1Point
REJECTED = "rejected"          # the single externally visible outcome for every rejected record
DISCARDED = "discarded"        # a record of an ended policy: no deviation, internal only


def _ser(obj) -> bytes:
    return o.Serialize(obj, o.BINARY)


# ------------------------------------------------------------------------------------------------ signed key bundle
@dataclass
class KeyBundle:
    """Insurer-signed risk-path key material: public key, evaluation keys
    and Enc(w), bound to the key tag, a version label and the epoch range. OBUs and the TSP accept only a valid
    bundle. The bundle goes to the TSP only; the OBUs receive `PublicKeyCert`."""
    pk: object
    ct_w: object
    key_tag: str
    version: int
    epochs: Tuple[int, int]
    evk_digest: bytes
    sig: bytes = b""

    def message(self) -> bytes:
        h = hashlib.sha256(b"SP2UBI-risk-keys|")
        for part in (_ser(self.pk), _ser(self.ct_w), self.key_tag.encode(), self.version.to_bytes(4, "big"),
                     self.epochs[0].to_bytes(4, "big"), self.epochs[1].to_bytes(4, "big"), self.evk_digest):
            h.update(len(part).to_bytes(8, "big") + part)
        return h.digest()

    def verify(self, insurer_vk, epoch: int, cc=None) -> bool:
        if not self.epochs[0] <= epoch <= self.epochs[1]:
            return False
        if cc is not None and evk_digest(cc, self.key_tag) != self.evk_digest:
            return False                                   # evaluation keys installed at the TSP differ from the signed ones
        try:
            return schnorr_verify(insurer_vk, self.message(), proof_from_bytes(self.sig))
        except ValueError:
            return False


@dataclass
class PublicKeyCert:
    """Insurer-signed public key of the risk path for the OBUs: pk_Ins bound to its key tag, version and epoch range.
    It carries no Enc(w), no evaluation key and no digest of either; the OBU needs only pk_Ins to encrypt
    and the signature to accept it. Signed under its own domain label, so it is never confused with the TSP's bundle."""
    pk: object
    key_tag: str
    version: int
    epochs: Tuple[int, int]
    sig: bytes = b""

    def message(self) -> bytes:
        h = hashlib.sha256(b"SP2UBI-risk-pk|")
        for part in (_ser(self.pk), self.key_tag.encode(), self.version.to_bytes(4, "big"),
                     self.epochs[0].to_bytes(4, "big"), self.epochs[1].to_bytes(4, "big")):
            h.update(len(part).to_bytes(8, "big") + part)
        return h.digest()

    def verify(self, insurer_vk, epoch: int) -> bool:
        if not self.epochs[0] <= epoch <= self.epochs[1]:
            return False
        try:
            return schnorr_verify(insurer_vk, self.message(), proof_from_bytes(self.sig))
        except ValueError:
            return False


def evk_digest(cc, key_tag: str) -> bytes:
    mult = o.SerializeEvalMultKeyString(o.BINARY, key_tag)
    rot = o.SerializeEvalAutomorphismKeyString(o.BINARY, key_tag)
    return hashlib.sha256(b"SP2UBI-evk|" + hashlib.sha256(mult).digest() + hashlib.sha256(rot).digest()).digest()


# ------------------------------------------------------------------------------------------------ OBU
@dataclass
class EpochMessage:
    """C^(t): encrypted epoch record, dispute commitment c (cm_r), policy-epoch binding c' (cm_b), the
    cleartext epoch index t, public routing identifier, group signature (over all of them). A record
    of epoch t may arrive during epoch t+1 (upload window + grace), so the TSP pools it by its signed t; t is public
    metadata (every record of the epoch carries the same value) and dates no record within the epoch."""
    ct_payload: object
    commitment: bytes
    id_ins: bytes
    signature: bytes
    binding: bytes = b""
    epoch: int = 0

    def signed_bytes(self) -> bytes:
        return (_ser(self.ct_payload) + self.commitment + self.binding + self.epoch.to_bytes(4, "big")
                + self.id_ins)

    def digest(self) -> bytes:
        return hashlib.sha256(b"SP2UBI-record|" + self.signed_bytes()).digest()


def binding_commitment(ident: int, epoch: int, u: int, salt: bytes) -> bytes:
    """c' = H("SP2UBI-bind" | ID | t | u | salt'): binds a record to (policy, epoch,
    nonce) without x or the interval index; opened (and only it) to prove a receipted record missing at settlement."""
    h = hashlib.sha256(b"SP2UBI-bind|")
    for part in (str(ident).encode(), str(epoch).encode(), str(u).encode(), salt):
        h.update(len(part).to_bytes(8, "big") + part)
    return h.digest()


def receipt_message(digest: bytes, commitment: bytes, binding: bytes, epoch: int) -> bytes:
    """What the TSP receipt signs: the record digest and, explicitly, the dispute
    commitment c (cm_r), the binding c' (cm_b) and the signed epoch t. The dispute then opens cm_r against the receipt,
    so the OBU need not keep the record message (a ciphertext of about 1 MB)."""
    return b"SP2UBI-receipt|" + digest + commitment + binding + epoch.to_bytes(4, "big")


@dataclass(frozen=True)
class Receipted:
    """What the OBU keeps of a receipted record besides the sealed record, its salts and the receipt: the
    values the receipt signs. No ciphertext."""
    digest: bytes
    commitment: bytes
    binding: bytes
    epoch: int

    @staticmethod
    def of(msg: "EpochMessage") -> "Receipted":
        return Receipted(msg.digest(), msg.commitment, msg.binding, msg.epoch)


NO_EVIDENCE = (1 << 64) - 1    # settlement message of a conflict epoch: no evidence requested (never a nonce)


def fresh_nonce(p: int) -> int:
    """Nonce of an epoch record: uniform in [1, p), drawn when the record is first issued and kept in sealed storage
    with the record, so a re-send is identical. It is not derived from K: a derived nonce
    would give two different records of the same epoch (an OBU that lost its sealed storage) the same one-time HomMAC
    keys, and their combination would verify; it is not derived from the record either, because the insurer knows K
    and could then test candidate feature vectors against u. The OBU redraws a value already sealed in the same epoch
    (OBU.record_message), so two records of one OBU never share one-time keys."""
    return secrets.randbelow(p - 1) + 1


class OBU:
    """Holds the insurer-signed public key pk_Ins (`PublicKeyCert`), the HomMAC key K, its group signing key and its
    private ID (TEE / HSM).
    One record per aggregation interval (Params.interval_days; default daily); every record has its own random nonce and
    dispute salt in sealed storage, so a re-send is identical; records are uploaded after independent random delays,
    each on its own over the cellular (Uu) link (per-record scheduler: `next_attempt`, `attempt`, `acknowledge`,
    `retry`)."""

    def __init__(self, ident: int, K: bytes, keys: PublicKeyCert, insurer_vk, risk: RiskContext,
                 gs: GroupSignature, gs_key, id_ins: bytes, params: Params, epoch: int):
        if not isinstance(keys, PublicKeyCert):
            raise TypeError("the OBU receives only the signed public key pk_Ins (PublicKeyCert), not the TSP's bundle")
        if not keys.verify(insurer_vk, epoch):
            raise ValueError("unsigned or expired risk-path keys")
        self.ident, self.K, self.keys = ident, K, keys
        self.risk, self.gs, self.gs_key = risk, gs, gs_key
        self.id_ins, self.prm, self.epoch = id_ins, params, epoch
        self._days: Dict[int, List[int]] = {}
        self.capped_days: List[int] = []                    # days with a capped value (kept for disputes, not sent)
        self._sealed: Dict[int, Tuple[dict, bytes]] = {}   # interval k -> (committed record with nonce, salt) until settled
        self._meta: Dict[int, dict] = {}                    # interval k -> {interval, capped days}: sealed, not committed
        self._due: Dict[int, float] = {}                    # interval k -> upload time (drawn in the TEE at sealing)
        self._next: Dict[int, float] = {}                   # interval k -> time of its next upload attempt
        self._acked: Dict[int, bytes] = {}                  # interval k -> TSP receipt
        self._receipted: Dict[int, Receipted] = {}          # interval k -> the values the receipt signs (no ciphertext)

    def add_day(self, day: int, features: List[int]) -> None:
        """Daily feature vector (off-peak, rush, weekend miles in 0.1-mile units, hard brakes, hard accelerations)."""
        if len(features) != self.prm.n_features:
            raise ValueError("wrong number of features")
        caps = [self.prm.daily_mile_cap] * 3 + [self.prm.daily_event_cap] * 2
        if any(f < 0 for f in features):
            raise ValueError("negative daily feature")
        if day in self._days:
            raise ValueError("one daily vector per day")
        if any(f > c for f, c in zip(features, caps)):           # cap and flag, do not drop the day
            self.capped_days.append(day)
        self._days[day] = [min(f, c) for f, c in zip(features, caps)]

    def interval_record(self, k: int, D: int) -> List[int]:
        """Record of interval k (days k*a .. min((k+1)*a, D) - 1 of the epoch): (insured days covered, feature sums);
        days without a vector (no driving) count as insured days with zero features."""
        a = self.prm.interval_days
        if not 1 <= D <= self.prm.max_epoch_days or not 0 <= k < self.prm.records_per_epoch(D):
            raise ValueError("D outside [1, 31] or interval index outside the epoch")
        days = range(k * a, min((k + 1) * a, D))
        sums = [sum(self._days.get(d, [0] * self.prm.n_features)[i] for d in days) for i in range(self.prm.n_features)]
        return [len(days)] + sums

    def epoch_record(self, D: int) -> List[int]:
        """Sum over the whole epoch (the monthly configuration's single record; used by tests)."""
        sums = [sum(v[i] for d, v in self._days.items() if d < D) for i in range(self.prm.n_features)]
        return [D] + sums

    def record_message(self, k: int, D: int) -> EpochMessage:
        if k not in self._sealed:
            x = self.interval_record(k, D)
            days = range(k * self.prm.interval_days, k * self.prm.interval_days + x[0])
            sealed_u = {rec["u"] for rec, _ in self._sealed.values()}
            u = fresh_nonce(self.prm.p)
            while u in sealed_u:                               # redraw a nonce already sealed in this epoch
                u = fresh_nonce(self.prm.p)
            # cm_r commits to the record without its date (no interval index, no capped
            # day indices); the opening of a dispute therefore discloses an undated feature vector
            record = {"id": self.ident, "epoch": self.epoch, "x": x, "u": u,
                      "bind_salt": secrets.token_bytes(32).hex()}          # salt' of c'
            c, salt = commit(record)
            self._sealed[k] = (record, salt)
            self._meta[k] = {"interval": k, "capped_days": [d for d in self.capped_days if d in days]}
        return self._message_from_sealed(k)

    def schedule(self, D: int, rng=None) -> Dict[int, float]:
        """Seal every record of the epoch and draw its upload time (seconds from the epoch start): end of the record's
        interval plus an independent delay uniform in [0, W). In a deployment a record is sealed and its
        delay drawn when its interval ends."""
        rng = rng or secrets.SystemRandom()
        for k in range(self.prm.records_per_epoch(D)):
            self.record_message(k, D)
            if k not in self._due:
                end = min((k + 1) * self.prm.interval_days, D) * 86400
                self._due[k] = end + rng.random() * self.prm.upload_window_s
                self._next[k] = self._due[k]
        return dict(self._due)

    def messages(self, D: int, rng=None) -> List[Tuple[float, EpochMessage]]:
        """All records of the epoch with their upload times (no outage model; tests and the monthly configuration)."""
        due = self.schedule(D, rng)
        return [(due[k], self._message_from_sealed(k)) for k in sorted(due)]

    def next_attempt(self) -> Optional[Tuple[float, int]]:
        """Per-record Uu scheduler: (time, k) of the earliest pending upload attempt, or None. Each
        record has its own time (its drawn upload time, then its own retry times), so records are never sent together."""
        pending = [(t, k) for k, t in self._next.items() if k not in self._acked]
        return min(pending) if pending else None

    def attempt(self, k: int) -> EpochMessage:
        """One upload attempt of record k over a fresh cellular connection (fresh encryption of the sealed record)."""
        if k in self._acked or k not in self._next:
            raise ValueError("record not pending")
        return self._message_from_sealed(k)

    def retry(self, k: int, now: float, rng=None) -> None:
        """No valid receipt (link outage, TSP silent): the next attempt of record k is drawn independently, uniform in
        [now, now + retry_window_s), so retries after an outage do not arrive together."""
        rng = rng or secrets.SystemRandom()
        self._next[k] = now + rng.random() * self.prm.retry_window_s

    def acknowledge(self, k: int, msg: EpochMessage, receipt: Optional[bytes], tsp_vk) -> bool:
        """A record stays pending (and is re-sent at its next attempt) until a TSP receipt arrives that is valid over the
        values of the message the OBU sent for record k (digest, cm_r, cm_b, t). The OBU then keeps the receipt and these
        values, not the message."""
        if receipt is None or k not in self._sealed:
            return False
        record, salt = self._sealed[k]
        if msg.epoch != record["epoch"] or msg.commitment != hashlib.sha256(salt + encode_record(record)).digest() or \
                msg.binding != binding_commitment(record["id"], record["epoch"], record["u"],
                                                  bytes.fromhex(record["bind_salt"])):
            return False                                    # not a message of record k
        if verify_receipt(tsp_vk, msg, receipt):
            self._acked[k] = receipt
            self._receipted[k] = Receipted.of(msg)
            return True
        return False

    def receipt_evidence(self, accepted_nonces) -> List[Tuple[bytes, bytes, bytes, bytes, Tuple[int, int, int, bytes]]]:
        """At settlement the insurer returns the set of nonces u it accepted for this
        policy-epoch; the OBU presents evidence only for receipted records whose u is not in that set: (record digest,
        c (cm_r), c' (cm_b), TSP receipt over these and t, opening of c' = (ID, t, u, salt')). No x, no interval index
        and no date is revealed; c is opened only in the dispute procedure. An honest settlement opens nothing."""
        return [self._evidence(k) for k in sorted(self._acked) if self._sealed[k][0]["u"] not in accepted_nonces]

    def _evidence(self, k: int) -> Tuple[bytes, bytes, bytes, bytes, Tuple[int, int, int, bytes]]:
        record, r = self._sealed[k][0], self._receipted[k]
        return (r.digest, r.commitment, r.binding, self._acked[k],
                (record["id"], record["epoch"], record["u"], bytes.fromhex(record["bind_salt"])))

    def settlement_upload(self, accepted_nonces) -> List[tuple]:
        """The OBU's settlement upload, authenticated and encrypted end to end to the insurer
        (channel not modelled): one item for every receipted record whose nonce was not accepted, namely the evidence
        (`receipt_evidence`) and the dispute (opening of cm_r: the committed record and its salt); the OBU runs the
        dispute for every such record. Empty if every receipted record was accepted, and empty when the settlement
        message requests no evidence (`accepted_nonces` is None: conflict epoch)."""
        if accepted_nonces is None:
            return []
        return [(self._evidence(k), self._sealed[k]) for k in sorted(self._acked)
                if self._sealed[k][0]["u"] not in accepted_nonces]

    def unacknowledged(self) -> List[int]:
        return sorted(k for k in self._due if k not in self._acked)

    def epoch_message(self, D: int) -> EpochMessage:
        """Monthly configuration (interval_days >= D): the single record of the epoch."""
        if self.prm.records_per_epoch(D) != 1:
            raise ValueError("epoch_message needs interval_days >= D; use messages()")
        return self.record_message(0, D)

    def resend(self, k: int = 0) -> EpochMessage:
        """Re-send of a sealed record (same plaintext, nonce and commitment; fresh encryption)."""
        if k not in self._sealed:
            raise ValueError("nothing to re-send")
        return self._message_from_sealed(k)

    def _message_from_sealed(self, k: int) -> EpochMessage:
        record, salt = self._sealed[k]
        x, u = record["x"], record["u"]
        T, T2 = hommac.tag_blocks(x, self.K, self.epoch, u, self.prm.p)
        ct = self.risk.encrypt_payload(self.keys.pk, x, T, u, self.ident, T2)
        c = hashlib.sha256(salt + encode_record(record)).digest()
        cb = binding_commitment(record["id"], record["epoch"], u, bytes.fromhex(record["bind_salt"]))
        msg = EpochMessage(ct, c, self.id_ins, b"", cb, record["epoch"])
        msg.signature = self.gs.sign(self.gs_key, msg.signed_bytes())
        return msg

    def open_dispute(self, k: int = 0) -> Tuple[dict, bytes]:
        """The opening of cm_r: the committed record (policy, epoch, features, nonce, salt' of cm_b; no interval index and
        no day index) and its salt."""
        return self._sealed[k]

    def sealed_state(self) -> Dict[int, Tuple[dict, bytes]]:
        """Sealed storage content (records with their nonces, dispute salts) that survives an OBU reset."""
        return dict(self._sealed)

    def restore_sealed(self, sealed: Dict[int, Tuple[dict, bytes]]) -> None:
        """After a reset with sealed storage intact: re-sends are identical."""
        self._sealed = dict(sealed)
        self._meta = {k: self._meta.get(k, {"interval": k, "capped_days": []}) for k in self._sealed}


# ------------------------------------------------------------------------------------------------ TSP
def ct_id(ct) -> bytes:
    return hashlib.sha256(b"SP2UBI-out|" + _ser(ct)).digest()


class TSP:
    """Public key, evaluation keys and Enc(w) (inside the signed bundle); no secret FHE key; never sees a plaintext
    weight."""

    def __init__(self, gs: GroupSignature, risk: RiskContext, keys: KeyBundle, insurer_vk, epoch: int,
                 signing_key=None, dkg_sessions: Optional[SessionRegistry] = None):
        """signing_key: the TSP's long-term signing key if it was drawn before (it also signs the TSP's messages of
        the opening-key generation, opening_dkg.py); a fresh one otherwise. dkg_sessions: the TSP's record of the
        session identifiers of the opening-key generation it has used (opening_dkg.SessionRegistry), if the generation
        ran before the TSP object was built; a fresh one otherwise."""
        if not keys.verify(insurer_vk, epoch, risk.cc):
            raise ValueError("unsigned, expired or mismatching key bundle")
        self.gs, self.risk, self.keys = gs, risk, keys
        self.ct_w = keys.ct_w                          # the signed Enc(w)
        self.epoch = epoch                             # default epoch of end_epoch()
        self.pools: Dict[int, List[object]] = {}       # signed epoch t -> evaluated results of epoch t
        self.closed: set = set()                       # epochs whose batch is published (settlement deadline passed)
        self.rejected = 0
        self.refused_closed = 0                        # validly signed records of a closed epoch: no receipt
        self._sk = signing_key if signing_key is not None else rand_scalar()   # TSP receipt-signing key (vk below)
        self.dkg_sessions = dkg_sessions if dkg_sessions is not None else SessionRegistry(b"tsp")
        self.vk = G1() * self._sk
        self._setup_open = True                        # key confirmation window (closed before the first epoch)

    def confirmation_evaluate(self, ct_payload):
        """Key confirmation (setup only): the TSP's own pipeline on an insurer-made test record (installed Enc(w),
        evaluation with the installed relinearisation and rotation keys, slot fill)."""
        if not self._setup_open:
            raise PermissionError("key confirmation window closed")
        return self.risk.fill(self.risk.evaluate(ct_payload, self.ct_w), self.keys.pk)

    def close_setup(self) -> None:
        self._setup_open = False

    @property
    def pool(self) -> List[object]:
        """Pool L_t of the TSP's default epoch (the pool of any other epoch is `pools[t]`)."""
        return self.pools.setdefault(self.epoch, [])

    def receive(self, msg: EpochMessage) -> Optional[bytes]:
        """Verify, evaluate, fill, store in the pool of the record's signed epoch t; returns a TSP-signed receipt
        over the record digest, the commitments c (cm_r) and c' (cm_b) and t (or None). Admission rule:
        no receipt for an invalid signature, for a t outside the epoch range of the held key bundle (the prototype holds
        one bundle; a deployment selects the bundle whose range contains t), or for a t whose batch is already published
        (its settlement deadline has passed). A receipt is thus the TSP's commitment to include the record's result in
        the batch of epoch t, and a receipted record that is not accepted (outside a conflict epoch) is always a deviation
        of the TSP."""
        if not self.gs.verify(msg.signed_bytes(), msg.signature) or \
                not self.keys.epochs[0] <= msg.epoch <= self.keys.epochs[1]:
            self.rejected += 1
            return None
        if msg.epoch in self.closed:
            self.refused_closed += 1
            return None
        ct = self.risk.evaluate(msg.ct_payload, self.ct_w)
        self.pools.setdefault(msg.epoch, []).append(self.risk.fill(ct, self.keys.pk))
        return proof_bytes(schnorr_sign(self._sk, self.vk, receipt_message(msg.digest(), msg.commitment, msg.binding,
                                                                           msg.epoch)))

    def end_epoch(self, epoch: Optional[int] = None) -> List[object]:
        """After the settlement deadline of epoch t (default: the TSP's epoch): global shuffle over all records of all
        policyholders whose signed epoch is t; the output is published to the insurer and the epoch is closed: `receive`
        gives no receipt for a record of t any more. The TSP keeps no copy of the batch: once it is returned (sent), the
        TSP holds no result of epoch t."""
        t = self.epoch if epoch is None else epoch
        self.closed.add(t)
        batch = self.pools.pop(t, [])
        secrets.SystemRandom().shuffle(batch)
        return batch


def verify_receipt(tsp_vk, msg: EpochMessage, receipt: bytes) -> bool:
    return verify_receipt_digest(tsp_vk, msg.digest(), msg.commitment, msg.binding, receipt, msg.epoch)


def verify_receipt_digest(tsp_vk, digest: bytes, commitment: bytes, binding: bytes, receipt: bytes, epoch: int) -> bool:
    try:
        return schnorr_verify(tsp_vk, receipt_message(digest, commitment, binding, epoch), proof_from_bytes(receipt))
    except ValueError:
        return False


# ------------------------------------------------------------------------------------------------ settlement sizes
RECEIPT_BYTES = 64             # Schnorr proof over the receipt message (challenge + one response, 32 B each)


def max_committed_record(prm: Params) -> dict:
    """The committed record (opening of cm_r) at the largest values it can take: identity and nonce below p, a 32-bit
    epoch label, every feature of an interval at its cap, a 32-byte salt' in hex."""
    a = prm.interval_days
    return {"id": prm.p - 1, "epoch": 2 ** 32 - 1,
            "x": [a] + [a * prm.daily_mile_cap] * 3 + [a * prm.daily_event_cap] * 2,
            "u": prm.p - 1, "bind_salt": "f" * 64}


def settlement_item_bytes_max(prm: Params) -> int:
    """Largest byte length of one item of the OBU's settlement upload (`encode_settlement_upload`): evidence = digest,
    cm_r, cm_b (32 B each), receipt (64 B), opening of cm_b = ID (8 B), t (4 B), u (8 B), salt' (32 B); dispute = length
    (2 B) and JSON encoding of the committed record (at most `max_committed_record`), and its salt (32 B). Items are not padded; this is an upper bound for the size figures."""
    return 3 * 32 + RECEIPT_BYTES + 8 + 4 + 8 + 32 + 2 + len(encode_record(max_committed_record(prm))) + 32


def encode_settlement_upload(upload: List[tuple]) -> bytes:
    """Byte form of `OBU.settlement_upload` (before the end-to-end authenticated encryption): the items one after the
    other, each with a fixed header, the length of the committed record's encoding, the encoding and the salt of cm_r."""
    out = []
    for (digest, cm_r, cm_b, receipt, (rid, t, u, salt_b)), (record, salt_r) in upload:
        enc = encode_record(record)
        if len(receipt) != RECEIPT_BYTES or len(enc) >= 1 << 16:
            raise ValueError("malformed settlement item")
        out.append(digest + cm_r + cm_b + receipt + rid.to_bytes(8, "big") + t.to_bytes(4, "big") + u.to_bytes(8, "big")
                   + salt_b + len(enc).to_bytes(2, "big") + enc + salt_r)
    return b"".join(out)


def decode_settlement_upload(data: bytes) -> List[tuple]:
    import json
    out, i = [], 0
    while i < len(data):
        b = data[i:]
        if len(b) < 214:
            raise ValueError("truncated settlement item")
        dg, cr, cb, rc = b[0:32], b[32:64], b[64:96], b[96:160]
        rid, t, u = int.from_bytes(b[160:168], "big"), int.from_bytes(b[168:172], "big"), int.from_bytes(b[172:180], "big")
        sb, ln = b[180:212], int.from_bytes(b[212:214], "big")
        if len(b) < 214 + ln + 32:
            raise ValueError("truncated settlement item")
        record = json.loads(b[214:214 + ln])
        out.append(((dg, cr, cb, rc, (rid, t, u, sb)), (record, b[214 + ln:214 + ln + 32])))
        i += 214 + ln + 32
    return out


# ------------------------------------------------------------------------------------------------ insurer
@dataclass
class PolicyReport:
    expected_records: int = 1
    accepted_records: int = 0
    identical_duplicates: int = 0
    scores: List[int] = field(default_factory=list)     # unordered multiset of record scores (daily by default)
    status: str = "missing"            # "ok" | "missing" (fewer distinct records) | "conflict" (more)
    _entries: set = field(default_factory=set)

    @property
    def accepted(self) -> bool:
        return self.status == "ok"

    @property
    def score_sum(self) -> Optional[int]:
        return sum(self.scores) if self.status == "ok" else None


@dataclass
class BatchResult:
    reports: Dict[int, PolicyReport]
    verdicts: List[str]                # per batch member: "accepted", REJECTED or DISCARDED; kept by the insurer, never sent to the TSP
    rejected: int = 0
    discarded: int = 0                 # records of ended policies: no score, not a deviation
    diagnostics: List[str] = field(default_factory=list)   # internal only, never sent to the TSP


class Insurer:
    """Holds the risk-path secret key, w and one HomMAC key K per policy; issuer of the signed key bundle (to the TSP)
    and of the signed public key (to the OBUs)."""

    def __init__(self, params: Params, risk: RiskContext, weights: List[int], epoch: int):
        if len(weights) != params.m:
            raise ValueError("weights must include w_0 (day weight)")
        if any(not 0 <= wi < (1 << params.w_bits) for wi in weights):
            raise ValueError("weight outside [0, 2^w_bits)")
        params.check_bounds(weights)
        self.prm, self.risk, self.w, self.epoch = params, risk, list(weights), epoch
        self._sigk = rand_scalar()
        self.vk = G1() * self._sigk
        self.dkg_sessions = SessionRegistry(b"insurer")   # session identifiers of the opening-key generation used
        self.K: Dict[int, bytes] = {}
        self.D: Dict[int, int] = {}
        self.ended: set = set()                        # policies not insured in the current epoch (membership ended)
        self._sk = None

    def setup(self, version: int = 1, epochs: Tuple[int, int] = (1, 12)) -> KeyBundle:
        """One-off key generation: the insurer's key pair, relinearisation and rotation keys, Enc(w); returns the signed
        bundle (public: sent to the TSP) and keeps in `obu_keys` the signed public key pk_Ins with the same key tag,
        version and epoch range (public: sent to the OBUs). The secret key never leaves the insurer."""
        kp = self.risk.keygen()
        self._sk = kp.secretKey
        tag = kp.publicKey.GetKeyTag()
        b = KeyBundle(kp.publicKey, self.risk.encrypt_weights(kp.publicKey, self.w), tag, version, epochs,
                      evk_digest(self.risk.cc, tag))
        b.sig = proof_bytes(schnorr_sign(self._sigk, self.vk, b.message()))
        c = PublicKeyCert(kp.publicKey, tag, version, epochs)
        c.sig = proof_bytes(schnorr_sign(self._sigk, self.vk, c.message()))
        self.keys, self.obu_keys = b, c
        return b

    def enroll(self, D: int) -> Tuple[int, bytes]:
        """New policy; D = insured days of the policy in this epoch (from the signed contract dates)."""
        while True:
            ident = secrets.randbits(32)
            if ident not in self.K:
                break
        self.K[ident] = secrets.token_bytes(32)
        self.D[ident] = D
        return ident, self.K[ident]

    def end_policy(self, ident: int) -> None:
        """Membership end: from the current epoch on the policy is not insured; the OBU erases its
        signing and HomMAC keys on the insurer's signed instruction (SP2UBI does not use the revocation mechanism of
        BBS04). A record that decrypts to an ended policy is checked under that policy's key (the insurer keeps it); if
        it verifies it is discarded, not rejected, so it is no detected deviation; if it fails it is rejected. A TSP that
        moves a receipted record into such an identity is therefore detected: the moved record fails the check, and its
        own policy holds a receipt for a record that was not accepted."""
        if ident not in self.K:
            raise ValueError("unknown policy")
        self.ended.add(ident)

    def decrypt(self, ct) -> List[int]:
        return self.risk.decrypt(ct, self._sk)

    def key_confirmation(self, tsp: TSP) -> bool:
        """Key confirmation: one insurer-made test record through the TSP's full pipeline (installed Enc(w),
        relinearisation and rotation keys, fill), decrypted by the insurer and checked with HomMAC."""
        p = self.prm.p
        K, u = secrets.token_bytes(32), 1
        x = [1] + [secrets.randbelow(1000) for _ in range(self.prm.n_features)]
        T, T2 = hommac.tag_blocks(x, K, 0, u, p)
        ct = tsp.confirmation_evaluate(self.risk.encrypt_payload(self.keys.pk, x, T, u, 0, T2))
        d = self.decrypt(ct)
        return d[SLOT_X] == sum(a * b for a, b in zip(self.w, x)) % p and \
            hommac.verify_record(d[SLOT_X], d[SLOT_T], d[SLOT_T2], u, self.w, K, 0, p)

    def process_batch(self, batch: List[object]) -> BatchResult:
        """Per record: decrypt; unknown ID, HomMAC failure (checked first) or range failure give the same outward outcome
        REJECTED. Per policy-epoch exactly records_per_epoch(D) distinct valid records are expected: an identical
        duplicate (same u, S, T_S) is accepted once and has no effect; fewer distinct records -> 'missing' (re-send of
        the sealed records until the settlement deadline; receipted records that were not accepted are handled by
        `settle`); more -> 'conflict' (an OBU that lost its sealed storage re-issued records under fresh nonces): the
        policy-epoch yields no score and no record is chosen (its settlement requests no evidence and opens nothing).

        No verdict reaches the TSP. The insurer decrypts the published batch itself (single
        key), so the TSP sees no per-record request at all; `verdicts` and `diagnostics` stay with the insurer, and the
        only outcome toward the TSP is `deviation_notice` (one bit per epoch)."""
        p = self.prm.p
        lo, hi = self.prm.record_bounds(self.w)
        active = [i for i in self.K if i not in self.ended]
        reports = {i: PolicyReport(self.prm.records_per_epoch(self.D[i])) for i in active}
        # monthly configuration: a single record covers exactly D insured days
        lo_single = {i: self.w[0] * self.D[i] for i in active if reports[i].expected_records == 1}
        res = BatchResult(reports, [])
        for ct in batch:
            d = self.decrypt(ct)
            S, T_S, T_S2, u, ident = d[SLOT_X], d[SLOT_T], d[SLOT_T2], d[SLOT_U], d[SLOT_ID]
            ok = ident in self.K and hommac.verify_record(S, T_S, T_S2, u, self.w, self.K[ident], self.epoch, p) and \
                lo_single.get(ident, lo) <= S <= hi
            if ok and ident in self.ended:                        # ended policy: a record that verifies is discarded,
                res.verdicts.append(DISCARDED)                    # not a detected deviation (verify
                res.discarded += 1                                # first; an altered one is rejected below)
                continue
            if not ok:
                res.verdicts.append(REJECTED)
                res.rejected += 1
                res.diagnostics.append("record rejected")
                continue
            rep = reports[ident]
            if (u, S, T_S) in rep._entries:
                rep.identical_duplicates += 1
            else:
                rep._entries.add((u, S, T_S))
                rep.scores.append(S)
                rep.accepted_records += 1
            res.verdicts.append("accepted")
        for rep in reports.values():
            n = rep.accepted_records
            rep.status = "ok" if n == rep.expected_records else ("missing" if n < rep.expected_records else "conflict")
        return res

    @staticmethod
    def deviation_notice(res: "BatchResult", receipted_missing: int = 0) -> bool:
        """The only outcome of the insurer's checks that reaches the TSP, once per epoch: True
        iff the epoch contained a detected deviation, i.e. a rejected record or a receipted record that is not accepted
        (outside a conflict epoch) (`receipted_missing`, the sum of `receipted_missing_days` over the policies). Under the trust assumptions only
        the TSP can cause either (OBUs are trusted, honest results always verify, the TSP receipts only records of an
        epoch it can still publish (`TSP.receive`), records of ended policies are discarded, not rejected). It is sent
        at a fixed time after the settlement of the epoch and names no record or receipt, also after a termination. It depends on no per-record verdict
        beyond this disjunction, so over n epochs the TSP receives at most n bits."""
        return res.rejected > 0 or receipted_missing > 0

    @staticmethod
    def accepted_nonces(rep: "PolicyReport") -> set:
        """The nonces u of the records accepted for a policy-epoch, returned to the OBU at settlement."""
        return {u for u, _, _ in rep._entries}

    def receipted_missing_days(self, ident: int, rep: "PolicyReport", evidence, tsp_vk) -> int:
        """Number of missing days for which the OBU proves a TSP receipt, without opening
        any record. Each evidence item (digest, c, c', receipt, (ID, t, u, salt')) counts once per distinct u, and only
        if the receipt verifies under the TSP key over (digest, c, c', t), c' opens to (ID, t, u, salt') with ID = this policy
        and t = this epoch, and u is not an accepted nonce; the count (records x days per record: 1 in the daily
        configuration, D in the monthly one) is capped at D - D_r."""
        D = self.D[ident]
        accepted = self.accepted_nonces(rep)
        seen = set()
        for digest, commitment, binding, receipt, (rid, t, u, salt) in evidence:
            if rid != ident or t != self.epoch or u in accepted or u in seen:
                continue
            if binding_commitment(rid, t, u, salt) != binding or \
                    not verify_receipt_digest(tsp_vk, digest, commitment, binding, receipt, t):
                continue
            seen.add(u)
        if rep.status == "ok" or rep.status == "conflict":
            return 0
        if self.prm.interval_days == 1:
            D_r, per = rep.accepted_records, 1
        elif rep.expected_records == 1:
            D_r, per = 0, D
        else:
            raise ValueError("defined for the daily (a = 1) and monthly (a >= D) configurations only")
        return min(len(seen) * per, D - D_r)

    def settlement_message(self, ident: int, rep: "PolicyReport") -> bytes:
        """The insurer's settlement message to the OBU, authenticated and encrypted end to end to its TEE (channel not
        modelled): the accepted nonces of the policy-epoch as sorted 8-byte values (no padding). In a conflict epoch
        (more distinct valid records than expected, e.g. records of a replacement OBU beside the old one's) the
        policy-epoch yields no score whatever the evidence, so the message requests none: it is the single value
        NO_EVIDENCE, which is never a nonce."""
        if rep.status == "conflict":
            return NO_EVIDENCE.to_bytes(8, "big")
        return b"".join(u.to_bytes(8, "big") for u in sorted(self.accepted_nonces(rep)))

    @staticmethod
    def nonces_from_settlement(message: bytes) -> Optional[set]:
        """The accepted nonces, or None if the message requests no evidence (conflict epoch)."""
        if len(message) % 8:
            raise ValueError("not a whole number of nonces")
        if message == NO_EVIDENCE.to_bytes(8, "big"):
            return None
        return {int.from_bytes(message[i:i + 8], "big") for i in range(0, len(message), 8)}

    def settle(self, ident: int, rep: "PolicyReport", upload, tsp_vk, opened_nonces: Optional[set] = None):
        """Process an OBU's settlement upload (`OBU.settlement_upload`). Returns
        (receipted_missing_days, dispute_scores): the provisional count of `receipted_missing_days` and the scores of
        the records settled by the dispute procedure. A conflict epoch opens nothing (its settlement message requested
        no evidence): (0, [])."""
        if rep.status == "conflict":
            return 0, []
        opened = set() if opened_nonces is None else opened_nonces
        items = list(upload)
        n_rc = self.receipted_missing_days(ident, rep, [ev for ev, _ in items], tsp_vk)
        accepted = self.accepted_nonces(rep)
        scores = []
        for (digest, cm_r, cm_b, receipt, (_, t, _, _)), (record, salt) in items:
            sc = self.dispute_record(ident, Receipted(digest, cm_r, cm_b, t), receipt, tsp_vk, record, salt, accepted,
                                     opened)
            if sc is not None:
                scores.append(sc)
        return n_rc, scores

    @staticmethod
    def verify_dispute(msg: EpochMessage, record: dict, salt: bytes) -> bool:
        return open_commitment(msg.commitment, record, salt)

    def dispute_record(self, ident: int, receipted: Receipted, receipt: bytes, tsp_vk, record: dict, salt: bytes,
                       accepted_nonces, opened_nonces: set) -> Optional[int]:
        """Dispute procedure: the OBU presents the receipt with the values it
        signs (digest, c = cm_r, c' = cm_b, t) and the opening of c. The insurer checks the receipt over these values,
        that c opens to the disclosed record, that (ID, t, u) is this policy, this epoch and the receipted t, that c'
        opens to the same (ID, t, u), and that u was neither accepted in B nor already opened; it then records u as
        opened and returns the score of the disclosed features (None if any check fails). No message or ciphertext is
        needed: the TSP's receipt binds cm_r directly."""
        u = record.get("u")
        if record.get("id") != ident or ident not in self.K or record.get("epoch") != self.epoch \
                or receipted.epoch != self.epoch or u in accepted_nonces or u in opened_nonces:
            return None
        if not verify_receipt_digest(tsp_vk, receipted.digest, receipted.commitment, receipted.binding, receipt,
                                     receipted.epoch) or not open_commitment(receipted.commitment, record, salt) or \
                binding_commitment(ident, self.epoch, u, bytes.fromhex(record.get("bind_salt", ""))) != receipted.binding:
            return None
        opened_nonces.add(u)
        x = record["x"]
        return sum(wi * xi for wi, xi in zip(self.w, x)) % self.prm.p
