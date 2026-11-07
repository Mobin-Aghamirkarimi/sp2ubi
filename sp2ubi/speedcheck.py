"""Speed-verification path of SP2UBI.

Entities are those of the paper (OBU, RSU, TSP, insurer); no additional party.
  OBU (trusted TEE)  drives in consecutive, non-overlapping bursts of BURST_S seconds (random phase). A burst has an
                     ephemeral Schnorr key E certified once by a BBS04 group signature (certificate repeated every
                     second); beacon j = (burst nonce, j, transmit time t_ms stamped at air time, Enc_SV(v_sen,j));
                     every second a rolling trailer signs the digests of the beacons sent since the previous one with
                     the ephemeral key. The certificate's tracing ciphertext binds a result to the signer.
  RSU                one entry per crossing: from the burst whose certificate verifies and whose beacons cover the
                     crossing window given by the physical layer, it accepts the beacons in that window only if each is
                     fresh (|arrival - t_ms| <= FRESH_MS) and matches the digest signed by a verified rolling trailer
                     of that burst, the beacons span >= SPAN_MIN of the window and >= MIN_COVER of the indices between
                     the first and last one arrived; invalid or foreign trailers, contradicting beacons and stale copies
                     are ignored (their beacons count as missing) and of several fresh copies of a beacon the earliest
                     is used, so an injected message cannot suppress a crossing; a
                     burst gives at most one entry per RSU, and only crossings in the site's surveyed direction are used. It
                     evaluates Enc(B) = sum c w_i Enc(v_sen,i) - Enc(c W') + Enc(e_s), W' = W (1 + site bias), with
                     Enc(e_s) the smudging term of the signed trailer covering the first used beacon (drawn by the
                     OBU's TEE, never known to an RSU; one-term form) and no entry if c > C_MAX; it attaches
                     Enc(level), Enc(z_q). It stores the entry unsigned with only a release time drawn uniformly in
                     [0, FORWARD_WINDOW_S) after the end of the billing epoch (no crossing time, no crossing order), and
                     at the release time re-randomises every component, signs with the RSU group key (RSU identity
                     hidden), forwards and erases it. It sends Enc(number of entries it formed and stored for the epoch;
                     an entry lost from storage before its release therefore fails the count check),
                     group-signed, at a time uniform in [FORWARD_WINDOW_S, FORWARD_WINDOW_S + COUNT_REPORT_SPREAD_S)
                     after the epoch end, independent of its entries.
  TSP                verifies RSU signatures, stores the epoch's entries, shuffles (re-randomise + permute), publishes
                     the shuffle output and answers partial decryptions/openings only for exact members of it (membership check).
  Insurer + TSP      2-of-2 shares of the speed-verification key (two-party DKG with proofs of knowledge) and of the
                     opening key (generated jointly without a dealer, opening_dkg.py; share commitments checked
                     against the group key); every partial carries a DLEQ proof, so a wrong partial is detected.
Bounds (check_bounds): v in [0, 70] m/s; <= K_MAX beacons per entry; c >= C_MIN; |b_hat| > B_WINDOW is reported as
out of window (None) and counts as a saturated residual in the decision rules.
"""
import bisect
import hashlib
import random
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import py_arkworks_bls12381 as ark

from .elgamal import Ciphertext, PublicKey, ThresholdKey, _scalar
from .groupsig_bbs04 import BBS04, MemberKey, OpenerShare, Signature, combine_opening, rand_scalar
from .proofs import (Proof, proof_bytes, proof_from_bytes, schnorr_sign, schnorr_verify, verify_dleq,
                     verify_opening_setup)

G1, Scalar = ark.G1Point, ark.Scalar

UNIT_MPS = 0.1           # integer speed unit (m/s)
V_MAX = 700              # 70 m/s
K_MAX = 400              # beacons per entry
S_BITS = 34
C_MIN = 64
C_MAX = 1 << 20          # no entry if c > C_MAX: the smudging sd is >= 16 c for every entry the code can emit (the
                         # privacy argument rests on this guard alone). Constant-speed crossings of the evaluated geometry
                         # under the 90 % span rule stay far below it: c <= 9.4e4, sigma/c >= 183, for site biases of
                         # +-6 % (experiments/sizes/smudging_scale_range.py).
B_WINDOW = 0.25          # |b_hat| decodable without a flag; larger results are reported as None (out of window)
SMUDGE_FRAC = 0.001      # smudge std as a fraction of 2^S: breaks divisibility of B by c; drawn by the
                         # OBU's TEE per rolling trailer and sent encrypted inside the trailer's signed content
BSGS_BITS = 34
N_LEVELS = 8             # disclosed sigma level (3 bits); 8 levels double the power of 4 levels
Z_STEP, Z_CLIP = 3.0, 3.0  # consistency value z disclosed as -1 / 0 / +1 (z < -1.5, |z| <= 1.5, z > 1.5): the only
                           # option meeting the privacy target with 8 levels (matching simulation) and the epoch rule
                           # least sensitive to site calibration error (consistency simulation); z is standardised by
                           # the RSU with the median and MAD of its recent traffic
DIGEST_BYTES = 16
NONCE_BYTES = 16
BEACON_PERIOD_MS = 100   # 10 Hz
BURST_MS = 40_000        # burst length (non-overlapping bursts; random phase)
ROLL_PERIOD_MS = 1_000   # rolling trailer and certificate repetition
FRESH_MS = 2             # |arrival - air-time stamp| per beacon (excludes stored replays and slow relays, not a fast
                         # dedicated live relay; stated limitation)
MIN_COVER = 0.8          # received share of the beacon indices spanned inside the crossing window
SPAN_MIN = 0.9           # beacons must span >= SPAN_MIN of the crossing window
USED_HORIZON_MS = 2 * BURST_MS  # memory of bursts already used at this RSU (bounded)
FORWARD_DELAY_S = 86_400  # release time of an entry: uniform in [0, FORWARD_DELAY_S) after the end of its billing epoch
                          # (copied by make_constants.py into the constants file; no simulation stage reads it)
FORWARD_WINDOW_S = FORWARD_DELAY_S  # alias used by the forwarding code, so the two cannot diverge
COUNT_REPORT_SPREAD_S = 3_600  # count report: uniform in [FORWARD_WINDOW_S, + this) after the end of the epoch

W_MAX_WEIGHT = V_MAX      # largest plaintext weight (profile: v_fit <= V_MAX; matched: G g^2 <= 4 G, G = 64)


def check_bounds() -> None:
    worst_sum = K_MAX * W_MAX_WEIGHT * V_MAX
    assert (1 << S_BITS) / worst_sum >= C_MIN, "S too small for the worst-case pass"
    assert B_WINDOW * (1 << S_BITS) + 8 * SMUDGE_FRAC * (1 << S_BITS) < (1 << (BSGS_BITS - 1)), "BSGS range too small"
    assert 16 * C_MAX <= SMUDGE_FRAC * (1 << S_BITS), "C_MAX too large for the smudging term"


def speed_to_units(v_mps: float) -> int:
    return max(0, min(V_MAX, round(v_mps / UNIT_MPS)))


def z_to_units(z: float) -> int:
    """RSU-side disclosure of the standardised consistency value: clip to +-Z_CLIP, quantise to Z_STEP."""
    return round(max(-Z_CLIP, min(Z_CLIP, z)) / Z_STEP)


# ---------------------------------------------------------------------------------------------------- OBU messages
@dataclass(frozen=True)
class Certificate:
    """Burst certificate: ephemeral key E, burst start, group signature (repeated every ROLL_PERIOD_MS)."""
    burst: bytes
    E: "G1"
    t0_ms: int
    sig: bytes

    def message(self) -> bytes:
        return (b"SP2UBI-cert|" + self.burst + self.E.to_compressed_bytes() + self.t0_ms.to_bytes(8, "big")
                + BURST_MS.to_bytes(4, "big"))

    def size(self) -> int:
        return len(self.burst) + 48 + 8 + len(self.sig)


@dataclass(frozen=True)
class Beacon:
    burst: bytes
    j: int
    t_ms: int                 # absolute GNSS time of transmission (ms), stamped at air time
    ct: Ciphertext

    def body(self) -> bytes:
        return self.burst + self.j.to_bytes(2, "big") + self.t_ms.to_bytes(8, "big") + self.ct.to_bytes()

    def digest(self) -> bytes:
        return hashlib.sha256(b"SP2UBI-beacon|" + self.body()).digest()[:DIGEST_BYTES]

    def size(self) -> int:
        return len(self.body())


@dataclass(frozen=True)
class Rolling:
    """Rolling trailer: digests of beacons j0 .. j0 + n - 1 and Enc(e_s), a fresh smudging term drawn in the TEE, all
    signed with the burst's ephemeral key. The ciphertext must be inside message(): pk_SV is public and ElGamal is
    malleable, so an unsigned term could be replaced over the air (e.g. by Enc(+0.1 * 2^S), cancelling a -10 % VSS)."""
    burst: bytes
    j0: int
    digests: Tuple[bytes, ...]
    ct_s: Ciphertext
    esig: bytes

    def message(self) -> bytes:
        return (b"SP2UBI-roll|" + self.burst + self.j0.to_bytes(2, "big") + len(self.digests).to_bytes(2, "big")
                + b"".join(self.digests) + self.ct_s.to_bytes())

    def size(self) -> int:
        return len(self.burst) + 4 + DIGEST_BYTES * len(self.digests) + len(self.ct_s.to_bytes()) + len(self.esig)

    def covers(self, j: int) -> bool:
        return self.j0 <= j < self.j0 + len(self.digests)


@dataclass
class BurstRecord:
    cert: Certificate
    beacons: List[Beacon] = field(default_factory=list)
    rollings: List[Rolling] = field(default_factory=list)


class OBU:
    """OBU side (TEE + modem). `pk_sv` is the joint ElGamal key; `gs` the OBU-group BBS04 instance; `mk` the credential.
    drive() emits consecutive non-overlapping bursts of BURST_MS with a random phase. `rng` draws the smudging terms;
    it defaults to the system CSPRNG (the TEE's RNG) and is replaced only by tests."""

    def __init__(self, gs: BBS04, mk: MemberKey, pk_sv: PublicKey, phase_ms: Optional[int] = None,
                 rng: Optional[random.Random] = None):
        self.gs, self.mk, self.pk = gs, mk, pk_sv
        self.phase_ms = secrets.randbelow(BURST_MS) if phase_ms is None else phase_ms
        self._rng = rng or random.SystemRandom()

    def _new_burst(self, t0_ms: int):
        e = rand_scalar()
        E = G1() * e
        nonce = secrets.token_bytes(NONCE_BYTES)
        cert = Certificate(nonce, E, t0_ms, b"")
        return e, BurstRecord(Certificate(nonce, E, t0_ms, self.gs.sign(self.mk, cert.message())))

    def _roll(self, e, rec: BurstRecord, upto: int) -> None:
        j0 = sum(len(r.digests) for r in rec.rollings)
        if upto > j0:
            dig = tuple(b.digest() for b in rec.beacons[j0:upto])
            # the TEE draws the whole smudging term and sends only its encryption, inside the signed content; the
            # plaintext is a local that is discarded here (not stored, not among the values sealed for disputes)
            ct_s = self.pk.encrypt(round(self._rng.gauss(0.0, SMUDGE_FRAC * (1 << S_BITS))))
            r = Rolling(rec.cert.burst, j0, dig, ct_s, b"")
            rec.rollings.append(Rolling(r.burst, j0, dig, ct_s, proof_bytes(schnorr_sign(e, rec.cert.E, r.message()))))

    def drive(self, v_sen_units: Sequence[int], t_ms: Sequence[int]) -> List[BurstRecord]:
        out: List[BurstRecord] = []
        e = rec = None
        for v, t in zip(v_sen_units, t_ms):
            k = (t - self.phase_ms) // BURST_MS
            t0 = self.phase_ms + k * BURST_MS
            if rec is None or rec.cert.t0_ms != t0:
                if rec is not None:
                    self._roll(e, rec, len(rec.beacons))
                e, rec = self._new_burst(t0)
                out.append(rec)
            if rec.beacons and t // ROLL_PERIOD_MS != rec.beacons[-1].t_ms // ROLL_PERIOD_MS:
                self._roll(e, rec, len(rec.beacons))
            rec.beacons.append(Beacon(rec.cert.burst, len(rec.beacons), t, self.pk.encrypt(v)))
        if rec is not None:
            self._roll(e, rec, len(rec.beacons))
        return out


# ---------------------------------------------------------------------------------------------------- RSU
@dataclass(frozen=True)
class Entry:
    """RSU output for one crossing: re-randomised tracing ciphertext, Enc(B), Enc(sigma level), Enc(consistency z),
    the billing epoch of the crossing (signed, so an entry cannot be moved into another epoch), and the RSU's group
    signature over these (the signer is anonymous among RSUs)."""
    T1: "G1"
    T2: "G1"
    T3: "G1"
    B: Ciphertext
    level: Ciphertext
    z: Ciphertext
    epoch: int = 0
    rsu_sig: bytes = b""

    def body(self) -> bytes:
        return (self.epoch.to_bytes(4, "big") + self.T1.to_compressed_bytes() + self.T2.to_compressed_bytes()
                + self.T3.to_compressed_bytes() + self.B.to_bytes() + self.level.to_bytes() + self.z.to_bytes())


@dataclass(frozen=True)
class CountReport:
    """Report of one RSU for one billing epoch: Enc(number of entries of that epoch it formed and stored; an entry lost
    from storage before its release therefore makes the count exceed the published entries), epoch label and group
    signature. Every RSU sends one report per epoch (also for zero entries), so the number of reports equals the
    published RSU-group size; only the homomorphic total over all reports is ever decrypted, so per-RSU counts (and
    undisclosed-site activity) stay hidden."""
    count: Ciphertext
    epoch: int
    rsu_sig: bytes

    def message(self) -> bytes:
        return b"SP2UBI-count|" + self.epoch.to_bytes(4, "big") + self.count.to_bytes()


def _rerand_tracing(gs: BBS04, T1, T2, T3):
    g = gs.gpk
    r1, r2 = rand_scalar(), rand_scalar()
    return T1 + g.u * r1, T2 + g.v * r2, T3 + g.h * (r1 + r2)


class SensingRSU:
    """Roadside verifier: authenticates the crossing, evaluates the speed check homomorphically, stores the entry
    unsigned with only its release time (after the end of the billing epoch) and at that time re-randomises, signs
    with the RSU group key, forwards and erases it. `surveyed_direction` (+1 / -1) is the direction of travel for which
    the site's calibration was surveyed; crossings in the other direction yield no entry."""

    NO_CROSSING: Tuple[List[Beacon], Optional[Ciphertext]] = ([], None)

    def __init__(self, gs: BBS04, pk_sv: PublicKey, rsu_gs: BBS04, rsu_key: MemberKey,
                 rng: Optional[random.Random] = None, surveyed_direction: int = +1):
        self.gs, self.pk, self.rsu_gs, self.rsu_key = gs, pk_sv, rsu_gs, rsu_key
        self.surveyed_direction = surveyed_direction
        self._used: Dict[bytes, int] = {}
        self._rng = rng or random.SystemRandom()       # release times and count-report times only (no smudging)
        self._store: List[Tuple[float, int, Entry]] = []   # (release time, tie-breaker, unsigned entry); no crossing time
        self._counted: Dict[int, int] = {}               # entries formed and stored per epoch (non-volatile, with the store)
        self.epoch = 0                                   # billing epoch of crossings evaluated from now on

    def accept(self, rec: BurstRecord, arrival_ms: Sequence[Optional[int]], window: Tuple[int, int],
               rollings_heard: Optional[Sequence[Rolling]] = None, direction: int = +1,
               copies: Sequence[Tuple[Beacon, int]] = ()) -> Tuple[List[Beacon], Optional[Ciphertext]]:
        """(beacons of one burst inside the crossing window [w0, w1], Enc(e_s) of the verified trailer covering the
        first of them), or ([], None). arrival_ms[i] is the arrival time of rec.beacons[i] (None: not received);
        `copies` are further received copies (beacon, arrival time), genuine, delayed or injected.
        Invalid messages are ignored, never a reason to discard the crossing: a trailer that fails verification or names another burst is ignored and the beacons it would cover
        count as missing; a copy that is foreign, unsigned, contradicts its signed digest or is not fresh is ignored; a
        beacon received in several fresh copies is used once, from the copy that arrived first by the RSU's own clock
        (a later copy can only be a re-radiation or a blind retransmission; a copy that matches the signed digest is
        byte-identical, so the choice cannot change the entry). Only two
        verified trailers of the burst that differ for one beacon (digest or term), or a signed beacon outside the
        certificate's interval, reject the crossing: only the TEE holds e, so these indicate a fault, not an injection.
        Every distinct heard trailer is verified; there is no cap that drops trailers (a cap would re-open suppression).
        Then: span >= SPAN_MIN of the window, >= MIN_COVER of the index range, burst not used before, surveyed
        direction (from the sign of the FDOA, given by the physical layer with the window)."""
        if direction != self.surveyed_direction:
            return self.NO_CROSSING                      # calibration is per carriageway; one term enters one entry
        cert = rec.cert
        heard = [t for t in arrival_ms if t is not None] + [t for _, t in copies if t is not None]
        if len(arrival_ms) != len(rec.beacons) or not heard or not self.gs.verify(cert.message(), cert.sig):
            return self.NO_CROSSING
        now = max(heard)
        self._used = {k: t for k, t in self._used.items() if now - t <= USED_HORIZON_MS}
        if cert.burst in self._used:
            return self.NO_CROSSING
        signed: Dict[int, Tuple[bytes, bytes]] = {}     # index -> (digest, term) signed by a verified trailer
        trailer_of: Dict[int, Rolling] = {}
        seen = set()
        for r in (rec.rollings if rollings_heard is None else rollings_heard):
            if r.burst != cert.burst:
                continue                                 # another burst's trailer: not "of this burst", ignored
            key = r.message() + r.esig
            if key in seen:
                continue                                 # an identical copy of a trailer already verified
            seen.add(key)
            try:
                ok = schnorr_verify(cert.E, r.message(), proof_from_bytes(r.esig))
            except ValueError:
                ok = False
            if not ok:
                continue                                 # invalid trailer: ignored, its beacons count as missing
            term = r.ct_s.to_bytes()
            for i, d in enumerate(r.digests):
                if signed.get(r.j0 + i, (d, term)) != (d, term):
                    return self.NO_CROSSING              # two verified trailers differ: only the TEE signs (fault)
                signed[r.j0 + i], trailer_of[r.j0 + i] = (d, term), r
        w0, w1 = window
        fresh: Dict[int, List[Tuple[int, Beacon]]] = {}
        for b, t_rx in list(zip(rec.beacons, arrival_ms)) + list(copies):
            if t_rx is None or b.burst != cert.burst or not w0 <= b.t_ms <= w1:
                continue                                 # not received / foreign / outside the window
            if b.j not in signed or signed[b.j][0] != b.digest():
                continue                                 # no verified trailer, or contradicts the signed digest: ignored
            if abs(t_rx - b.t_ms) > FRESH_MS:
                continue                                 # stale copy: ignored (a fresh matching copy is used)
            if not cert.t0_ms <= b.t_ms < cert.t0_ms + BURST_MS:
                return self.NO_CROSSING                  # a signed beacon outside its certificate's interval: TEE fault
            fresh.setdefault(b.j, []).append((t_rx, b))
        sel = [min(cs, key=lambda x: x[0])[1] for j, cs in sorted(fresh.items())]   # earliest fresh copy is used
        if len(sel) < 3:
            return self.NO_CROSSING
        js = [b.j for b in sel]
        ts = sorted(b.t_ms for b in sel)
        if ts[-1] - ts[0] < SPAN_MIN * (w1 - w0) or len(sel) < MIN_COVER * (js[-1] - js[0] + 1):
            return self.NO_CROSSING
        self._used[cert.burst] = now
        return sel, trailer_of[sel[0].j].ct_s           # the term of the (verified) trailer covering the first used beacon

    def evaluate(self, beacons: Sequence[Beacon], cert: Certificate, v_rsu_units: Sequence[int],
                 smudge_ct: Optional[Ciphertext], level: int = 0, z_units: int = 0) -> Optional[Entry]:
        """Profile statistic with the per-beacon roadside speeds as weights (w_i = v_rsu,i, W = sum v_rsu,i^2)."""
        return self.evaluate_weighted(beacons, cert, v_rsu_units, sum(v * v for v in v_rsu_units), smudge_ct,
                                      level, z_units)

    def evaluate_weighted(self, beacons: Sequence[Beacon], cert: Certificate, weights: Sequence[int], W: int,
                          smudge_ct: Optional[Ciphertext], level: int = 0, z_units: int = 0,
                          site_bias: float = 0.0) -> Optional[Entry]:
        """General linear statistic b_hat = sum w_i v_sen,i / (W (1 + site_bias)) - 1 with plaintext integer weights;
        site_bias is the calibrated relative site bias (applied by the RSU, the only party that knows the site).
        smudge_ct is Enc(e_s) from accept(); the RSU draws no smudging term itself. Returns the unsigned entry; it is
        signed only at release (store / release)."""
        if smudge_ct is None or not beacons or len(beacons) != len(weights) or len(beacons) > K_MAX or W <= 0:
            return None
        if any(w < 0 or w > W_MAX_WEIGHT for w in weights):
            return None
        if not 0 <= level < N_LEVELS or abs(z_units) > round(Z_CLIP / Z_STEP):
            return None                                  # nothing finer than the disclosed resolution leaves the RSU
        Wc = round(W * (1.0 + site_bias))
        c = round((1 << S_BITS) / Wc)
        if c > C_MAX:
            return None                                  # guard: B mod c stays uniform for every entry emitted
        acc = Ciphertext(G1.identity(), G1.identity())
        for b, w in zip(beacons, weights):
            if w:
                acc = acc + b.ct.mul(c * w)
        acc = acc + smudge_ct                            # the OBU's term, unknown to every RSU (one-term form)
        acc = Ciphertext(acc.C1, acc.C2 - self.pk.G * _scalar(c * Wc))
        s = Signature.from_bytes(cert.sig)
        T1, T2, T3 = _rerand_tracing(self.gs, s.T1, s.T2, s.T3)
        # Re-randomisation is required: acc is a public linear combination of broadcast ciphertexts (the
        # beacons and the trailer term); without it a recorder colluding with an RSU that knows (c, w_i) could
        # recompute Enc(B) and find it among the TSP's published inputs. release() re-randomises again.
        return Entry(T1, T2, T3, self.pk.rerandomize(acc), self.pk.encrypt(level), self.pk.encrypt(z_units),
                     epoch=self.epoch)

    # --- storage until after the epoch; release, sign and erase; count report independent of the entries ---
    def store(self, e: Entry, epoch_end_s: float) -> None:
        """Keep the unsigned entry with only its release time, drawn now, uniform in [0, FORWARD_WINDOW_S) after the
        end of its billing epoch; ordered by release time, so neither the crossing time nor the crossing order is
        kept (models the RSU's non-volatile storage). The entry is counted here, when it is formed and stored, not at
        release: an entry lost from storage before its release still counts, so the epoch's count check fails."""
        t_rel = epoch_end_s + self._rng.random() * FORWARD_WINDOW_S
        bisect.insort(self._store, (t_rel, self._rng.getrandbits(64), e), key=lambda x: (x[0], x[1]))
        self._counted[e.epoch] = self._counted.get(e.epoch, 0) + 1

    def stored(self) -> List[Tuple[float, Entry]]:
        """The stored state (release time, entry), e.g. as a seizure of the RSU would expose it."""
        return [(t, e) for t, _, e in self._store]

    def _finalize(self, e: Entry) -> Entry:
        """Re-randomise every component (as TSPShuffle does, public key only) and group-sign: the forwarded entry is
        unlinkable to the stored one under DDH (the signature must be made after the re-randomisation)."""
        T1, T2, T3 = _rerand_tracing(self.gs, e.T1, e.T2, e.T3)
        f = Entry(T1, T2, T3, self.pk.rerandomize(e.B), self.pk.rerandomize(e.level), self.pk.rerandomize(e.z),
                  epoch=e.epoch)
        return Entry(f.T1, f.T2, f.T3, f.B, f.level, f.z, epoch=f.epoch,
                     rsu_sig=self.rsu_gs.sign(self.rsu_key, b"SP2UBI-entry|" + f.body()))

    def release(self, now_s: float) -> List[Entry]:
        """Entries whose release time has come: re-randomised, signed, forwarded and erased from the store."""
        k = bisect.bisect_right([t for t, _, _ in self._store], now_s)
        due, self._store = [e for _, _, e in self._store[:k]], self._store[k:]
        return [self._finalize(e) for e in due]

    def count_report_time(self, epoch_end_s: float) -> float:
        """Send time of the epoch's count report: uniform in [FORWARD_WINDOW_S, FORWARD_WINDOW_S +
        COUNT_REPORT_SPREAD_S) after the end of the epoch, drawn independently of the entries (an n_r-dependent time
        would leak the RSU's count)."""
        return epoch_end_s + FORWARD_WINDOW_S + self._rng.random() * COUNT_REPORT_SPREAD_S

    def count_report(self, epoch: int) -> CountReport:
        """Sent at count_report_time(), after every entry of `epoch` has been released: the cut-off is the epoch
        label of the crossing, not the forwarding time, so count and entry list cover the same set. It encrypts the
        number of entries of the epoch formed and stored (counted at store())."""
        if any(e.epoch == epoch for _, _, e in self._store):
            raise RuntimeError("entries of this epoch are still stored; the count report is sent after the window")
        ct = self.pk.encrypt(self._counted.pop(epoch, 0))
        r = CountReport(ct, epoch, b"")
        return CountReport(ct, epoch, self.rsu_gs.sign(self.rsu_key, r.message()))


def verify_entry(rsu_gs: BBS04, e: Entry, epoch: Optional[int] = None) -> bool:
    """TSP-side (and public) check: the entry was produced by some RSU of the group (which one stays hidden) and, if
    `epoch` is given, carries that epoch label."""
    return (epoch is None or e.epoch == epoch) and rsu_gs.verify(b"SP2UBI-entry|" + e.body(), e.rsu_sig)


def aggregate_counts(rsu_gs: BBS04, reports: Sequence[CountReport], epoch: int, n_rsus: int) -> Optional[Ciphertext]:
    """Homomorphic total of the RSU-signed counts of `epoch`; None unless there is exactly one valid, distinct report
    per RSU of the published group (n_rsus) for that epoch, so a report cannot be dropped, replayed or moved."""
    if len(reports) != n_rsus or len({r.message() for r in reports}) != len(reports):
        return None
    total = Ciphertext(G1.identity(), G1.identity())
    for r in reports:
        if r.epoch != epoch or not rsu_gs.verify(r.message(), r.rsu_sig):
            return None
        total = total + r.count
    return total


# ---------------------------------------------------------------------------------------------------- TSP / insurer
class TSPShuffle:
    """End-of-epoch shuffle at the TSP: re-randomise every component and apply a secret permutation. Verifiability
    (Bayer-Groth over the product of BBS linear encryption and ElGamal) is cited, not implemented."""

    def __init__(self, gs: BBS04, pk_sv: PublicKey):
        self.gs, self.pk = gs, pk_sv

    def shuffle(self, entries: Sequence[Entry]) -> List[Entry]:
        out = [Entry(*_rerand_tracing(self.gs, e.T1, e.T2, e.T3), self.pk.rerandomize(e.B),
                     self.pk.rerandomize(e.level), self.pk.rerandomize(e.z), epoch=e.epoch) for e in entries]
        secrets.SystemRandom().shuffle(out)
        return out


def entry_id(e: Entry) -> bytes:
    """Binding context of every proof on an entry (a proof cannot be reused for another entry)."""
    return hashlib.sha256(b"SP2UBI-entry-id|" + e.body()).digest()


@dataclass(frozen=True)
class Partials:
    """One party's partial opening (xi1_k*T1, xi2_k*T2) and partial decryptions of B, level, z, each with a DLEQ proof
    (order: opening 1, opening 2, B, level, z)."""
    open_parts: Tuple["G1", "G1"]
    B: "G1"
    level: "G1"
    z: "G1"
    proofs: Tuple[Proof, ...] = ()


class KeyHolder:
    """Insurer or TSP: holds one share of the speed-verification key and one share of the opening key."""

    def __init__(self, sv_share: ThresholdKey, open_share: OpenerShare):
        self.sv, self.op = sv_share, open_share

    def public(self):
        """(ElGamal share public key, opening-share commitments (A_k, B_k)) as registered at setup."""
        return self.sv.share_pub, self.op.Y

    def partials(self, e: Entry) -> Partials:
        ctx = entry_id(e)
        (P1, P2), (po1, po2) = self.op.partial_with_proof(e.T1, e.T2, ctx)
        dec = [self.sv.partial_with_proof(ct, ctx) for ct in (e.B, e.level, e.z)]
        return Partials((P1, P2), dec[0][0], dec[1][0], dec[2][0], (po1, po2, dec[0][1], dec[1][1], dec[2][1]))


class TSPKeyHolder(KeyHolder):
    """TSP side with a membership check: partials only for exact members of its own published shuffle output, and a
    partial decryption of the epoch's count total only for the total it published. Without the check the two shares would
    also open raw beacons, raw certificates or trailer terms (Enc(e_s)) that an RSU could hand to the insurer."""

    def __init__(self, sv_share: ThresholdKey, open_share: OpenerShare):
        super().__init__(sv_share, open_share)
        self._allowed, self._totals = set(), set()

    def publish(self, shuffled: Sequence[Entry]) -> bytes:
        """Register the epoch's shuffle output; returns its digest (published with the shuffle proof)."""
        ids = [entry_id(e) for e in shuffled]
        self._allowed |= set(ids)
        return hashlib.sha256(b"SP2UBI-epoch|" + b"".join(ids)).digest()

    def publish_total(self, total: Ciphertext) -> None:
        self._totals.add(total.to_bytes())

    def partials(self, e: Entry) -> Partials:
        if entry_id(e) not in self._allowed:
            raise PermissionError("entry is not a member of the published shuffle output")
        return super().partials(e)

    def partial_total(self, total: Ciphertext):
        if total.to_bytes() not in self._totals:
            raise PermissionError("not the published count total")
        return self.sv.partial_with_proof(total, b"count-total")


class InsurerSpeedVerifier(KeyHolder):
    """Insurer: combines its own partials with the TSP's to obtain (policy, b_hat, level, z) per shuffled entry.
    Every TSP partial is checked against the TSP's registered public share values; a wrong partial raises. The
    registered opening-share values must be those checked in the opening-key generation: with the insurer's own they
    add up to (h, h) (proofs.verify_opening_setup), otherwise the registration raises."""

    def __init__(self, sv_share: ThresholdKey, open_share: OpenerShare, registry: Dict[bytes, str], tsp_public):
        super().__init__(sv_share, open_share)
        self.registry = registry
        self.tsp_X, self.tsp_Y = tsp_public
        if open_share.h is None or not verify_opening_setup(open_share, open_share.Y, tuple(self.tsp_Y)):
            raise ValueError("TSP opening-share values do not add up with the insurer's to (h, h)")

    def open_entry(self, e: Entry, tsp: Partials):
        ctx = entry_id(e)
        if len(tsp.proofs) != 5:
            raise ValueError("missing proofs")
        P1, P2 = tsp.open_parts
        if not (verify_dleq(ctx + b"|o1", self.op.u, self.tsp_Y[0], e.T1, P1, tsp.proofs[0])
                and verify_dleq(ctx + b"|o2", self.op.v, self.tsp_Y[1], e.T2, P2, tsp.proofs[1])):
            raise ValueError("partial opening proof invalid")
        A = combine_opening(e.T3, self.op.partial(e.T1, e.T2), P1 + P2)
        B = self.sv.combine_verified(e.B, tsp.B, tsp.proofs[2], self.tsp_X, ctx)
        level = self.sv.combine_verified(e.level, tsp.level, tsp.proofs[3], self.tsp_X, ctx)
        z = self.sv.combine_verified(e.z, tsp.z, tsp.proofs[4], self.tsp_X, ctx)
        b = None if B is None or abs(B) > B_WINDOW * (1 << S_BITS) else B / (1 << S_BITS)   # out of window -> None
        return self.registry.get(A.to_compressed_bytes()), b, level, z

    def open_total(self, total: Ciphertext, tsp_partial: "G1", tsp_proof: Proof) -> Optional[int]:
        return self.sv.combine_verified(total, tsp_partial, tsp_proof, self.tsp_X, b"count-total")
