"""Risk-path tests (single insurer key): slot fill over all N slots, decryption noise margin, deviations with one
outward outcome, duplicate / nonce / re-send rules, bounds, key setup and confirmation, forwarding rule, receipts,
epoch equivalence, the second HomMAC tag block, nonce redraw, the per-record Uu scheduler and the settlement of
receipted records that were not accepted."""
import math
import os
import random
import secrets

import openfhe as o
import pytest

from sp2ubi import hommac
from sp2ubi.entities import (OBU, REJECTED, TSP, Insurer, KeyBundle, PublicKeyCert, ct_id, verify_receipt)
from sp2ubi.fhe import OUTPUT_SLOTS, SLOT_ID, SLOT_T, SLOT_T2, SLOT_U, SLOT_X, RiskContext
from sp2ubi.groupsig_bbs04 import BBS04, Issuer as GroupIssuer
from sp2ubi.opening_dkg import run_two_party
from sp2ubi.params import DEFAULT, Params

MONTHLY = Params(interval_days=31)          # monthly configuration of the aggregation interval
from sp2ubi.risk_model import load_model
from sp2ubi.workload import daily_records, iter_policies
from sp2ubi.protocol import DATASET

P = DEFAULT.p
MARGIN_BITS_MIN = 6           # decryption margin of a result ciphertext at the feature caps (bits): the measured value is
                              # 9 at N = 8192, log2 Q = 120 (recorded by experiments/sizes/risk_path_sizes.py);
                              # 2^6 = 64 times the largest noise coefficient of this ciphertext still decrypts


class ScenarioTSP(TSP):
    """The tests of this module run independent scenarios that all use epoch 1 on one TSP object, so publishing a
    scenario's batch does not close epoch 1 here. The admission rule (no receipt for a record of a published epoch)
    is tested in test_epoch_label.py."""

    def end_epoch(self, epoch=None):
        out = super().end_epoch(epoch)
        self.closed.clear()
        return out


@pytest.fixture(scope="module")
def world():
    risk = RiskContext(P, DEFAULT.m)
    model = load_model()
    ins = Insurer(MONTHLY, risk, model.w, epoch=1)
    keys = ins.setup(epochs=(1, 12))
    gi = GroupIssuer(run_two_party()[0])           # opening key of insurer and TSP (not used here)
    gs = BBS04(gi.gpk)
    tsp = ScenarioTSP(gs, risk, keys, ins.vk, 1)
    assert ins.key_confirmation(tsp)
    tsp.close_setup()
    return {"risk": risk, "ins": ins, "keys": keys, "obu_keys": ins.obu_keys, "tsp": tsp, "gs": gs, "gi": gi, "w": model.w}


def new_obu(world, D=31, days=None, seed=0):
    ident, K = world["ins"].enroll(D)
    obu = OBU(ident, K, world["obu_keys"], world["ins"].vk, world["risk"], world["gs"], world["gi"].enrol(),
              b"INS-0001", MONTHLY, 1)
    rng = random.Random(seed)
    for d in range(D if days is None else days):
        obu.add_day(d, [rng.randrange(0, 900), rng.randrange(0, 300), rng.randrange(0, 400), rng.randrange(0, 3),
                        rng.randrange(0, 3)])
    return obu


def published(world, msg):
    """TSP pipeline for one message: returns the published output ciphertext."""
    tsp = world["tsp"]
    assert tsp.receive(msg) is not None
    out = tsp.end_epoch()
    assert len(out) == 1
    return out[0]


def all_slots(world, ct):
    """The insurer's decryption of all N slots."""
    return world["risk"].decrypt(ct, world["ins"]._sk, world["risk"].N)


# ---------------------------------------------------------------------------------------- all-N-slot test
def test_all_slots_uniform_outputs_exact_and_negative_control(world):
    risk, keys, w = world["risk"], world["keys"], world["w"]
    N, B, n_rec = risk.N, 64, 100
    counts, fills = [0] * B, []
    for j in range(n_rec):
        x = [31] + [random.randrange(0, 31 * 16383) for _ in range(3)] + [random.randrange(0, 31 * 4095)] * 2
        u, ident, K = secrets.randbelow(P - 1) + 1, secrets.randbits(32), secrets.token_bytes(32)
        T, T2 = hommac.tag_blocks(x, K, 1, u, P)
        ct = risk.fill(risk.evaluate(risk.encrypt_payload(keys.pk, x, T, u, ident, T2), keys.ct_w), keys.pk)
        d = all_slots(world, ct)
        assert d[SLOT_X] == sum(a * b for a, b in zip(w, x)) % P and d[SLOT_U] == u and d[SLOT_ID] == ident
        assert hommac.verify_record(d[SLOT_X], d[SLOT_T], d[SLOT_T2], u, w, K, 1, P)
        assert d[SLOT_T2] == sum(a * b for a, b in zip(w, T2)) % P          # second block output slot
        rest = [d[i] for i in range(N) if i not in OUTPUT_SLOTS]
        for v in rest:
            counts[v * B // P] += 1
        if j < 2:
            fills.append(rest)
    n = sum(counts)
    chi2 = sum((c - n / B) ** 2 / (n / B) for c in counts)
    assert chi2 < 103.4                                            # chi^2(63) at 0.001
    # mask freshness: the fills of two records are uncorrelated (|r| < 4 / sqrt(N))
    a, b = fills
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    r = cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    assert abs(r) < 4 / math.sqrt(len(a))
    # negative control: without the fill the wrap-around prefix sums leak the weighted features
    x = [31, 2370, 610, 0, 2, 3]
    T = hommac.tags(x, b"k" * 32, 1, 5, P)
    ct = risk.evaluate(risk.encrypt_payload(keys.pk, x, T, 5, 77), keys.ct_w)
    d = all_slots(world, ct)
    h = N // 2
    prefix = [sum(w[i] * x[i] for i in range(j + 1)) % P for j in range(len(x))]
    assert [d[h - 7 + i] for i in range(len(x))] == prefix


# ---------------------------------------------------------------------------------------- noise margin
def test_noise_margin_of_the_result_ciphertext(world):
    """Scaling method: decryption of 2^e * ct fails once 2^e |e_ct| exceeds Delta/2, so the largest e that
    still decrypts is the decryption margin in bits of the result ciphertext (records at the feature caps)."""
    risk, keys = world["risk"], world["keys"]
    info = risk.info()
    x = [31] + [31 * 16383] * 3 + [31 * 4095] * 2                 # feature caps
    T, T2 = hommac.tag_blocks(x, b"k" * 32, 1, 9, P)
    ct = risk.fill(risk.evaluate(risk.encrypt_payload(keys.pk, x, T, 9, 1, T2), keys.ct_w), keys.pk)

    def dec(c):
        return all_slots(world, c)
    ref = dec(ct)

    def scaled(e):
        c = ct
        while e > 0:
            s = min(30, e)
            c = risk.cc.EvalMult(c, risk.cc.MakePackedPlaintext([1 << s] * risk.N))
            e -= s
        return c
    lo, hi = 0, 200
    while lo < hi:
        mid = (lo + hi + 1) // 2
        d = dec(scaled(mid))
        if all(d[i] == (ref[i] << mid) % P for i in range(0, risk.N, 97)):
            lo = mid
        else:
            hi = mid - 1
    assert lo >= MARGIN_BITS_MIN, (lo, info)


# ---------------------------------------------------------------------------------------- deviations, one outcome
def test_deviations_rejected_with_one_outcome(world):
    risk, ins, tsp, keys = world["risk"], world["ins"], world["tsp"], world["keys"]
    obu = new_obu(world, seed=3)
    out = published(world, obu.epoch_message(31))
    honest = ins.process_batch([out])
    assert honest.verdicts == ["accepted"]

    def delta(slots):
        v = [0] * 40
        for i, d in slots.items():
            v[i] = d
        return risk.encrypt_vector(keys.pk, v)
    cases = {
        "shift_S": risk.cc.EvalAdd(out, delta({SLOT_X: 1})),
        "shift_S_and_T_guessed_alpha": risk.cc.EvalAdd(out, delta({SLOT_X: 1, SLOT_T: secrets.randbelow(P)})),
        "shift_S_and_both_T_guessed": risk.cc.EvalAdd(out, delta({SLOT_X: 1, SLOT_T: secrets.randbelow(P),
                                                                 SLOT_T2: secrets.randbelow(P)})),
        "change_u": risk.cc.EvalAdd(out, delta({SLOT_U: 1})),
        "change_id": risk.cc.EvalAdd(out, delta({SLOT_ID: 1})),
        "skip_evaluation": obu.resend().ct_payload,
        "wrong_Enc_w": risk.fill(risk.evaluate(obu.resend().ct_payload,
                                               risk.encrypt_weights(keys.pk, [wi + 1 for wi in world["w"]])), keys.pk),
    }
    for name, c in cases.items():
        r = ins.process_batch([c])
        assert r.verdicts == [REJECTED], name


# ---------------------------------------------------------------------------------------- duplicates and re-send
def test_duplicate_nonce_and_resend_rules(world):
    ins, tsp = world["ins"], world["tsp"]
    obu = new_obu(world, seed=4)
    m1 = obu.epoch_message(31)
    out1 = published(world, m1)
    out2 = published(world, obu.resend())                          # re-send: identical plaintext, fresh encryption
    r = ins.process_batch([out1, out2])
    rep = r.reports[obu.ident]
    assert rep.status == "ok" and rep.identical_duplicates == 1 and r.verdicts == ["accepted", "accepted"]
    # a dropped record leaves the policy 'missing'; the re-send then completes it
    obu2 = new_obu(world, seed=5)
    obu2.epoch_message(31)
    r = ins.process_batch([])
    assert r.reports[obu2.ident].status == "missing"
    r = ins.process_batch([published(world, obu2.resend())])
    assert r.reports[obu2.ident].status == "ok"
    # OBU reset with sealed storage intact: the restored OBU re-sends the identical record
    clone = OBU(obu.ident, obu.K, world["obu_keys"], ins.vk, world["risk"], world["gs"], world["gi"].enrol(),
                b"INS-0001", MONTHLY, 1)
    clone.restore_sealed(obu.sealed_state())
    a, b = all_slots(world, out1), all_slots(world, published(world, clone.epoch_message(31)))
    assert [a[i] for i in OUTPUT_SLOTS] == [b[i] for i in OUTPUT_SLOTS]
    # OBU data loss: a replacement OBU with partly lost logs issues a second, different record under a
    # fresh nonce; (i) a TSP combination 2*ct_a - ct_b of the two payloads is rejected (different one-time keys);
    # (ii) if both records arrive, the policy is a 'conflict' and the second record is rejected
    risk, keys = world["risk"], world["keys"]
    days = list(obu._days.items())
    b = OBU(obu.ident, obu.K, world["obu_keys"], ins.vk, risk, world["gs"], world["gi"].enrol(), b"INS-0001", MONTHLY, 1)
    for d, v in days[:20]:
        b.add_day(d, v)
    ma, mb = obu.resend(), b.epoch_message(31)
    cc = risk.cc
    comb = risk.fill(risk.evaluate(cc.EvalSub(cc.EvalAdd(ma.ct_payload, ma.ct_payload), mb.ct_payload), keys.ct_w),
                     keys.pk)
    r = ins.process_batch([comb])
    assert r.verdicts == [REJECTED] and r.reports[obu.ident].status == "missing"
    r = ins.process_batch([published(world, ma), published(world, mb)])
    assert r.verdicts == ["accepted", "accepted"] and r.reports[obu.ident].status == "conflict"


# ---------------------------------------------------------------------------------------- bounds
def test_bounds_caps_and_model_update(world):
    w = world["w"]
    lo, hi = MONTHLY.score_bounds(w, 31)
    assert hi < P // 2 and lo == w[0] * 31
    DEFAULT.check_bounds(w)
    with pytest.raises(ValueError):
        DEFAULT.check_bounds([wi * 4 for wi in w])                 # a model update that breaks the bound is refused
    obu = new_obu(world, days=0)
    obu.add_day(0, [DEFAULT.daily_mile_cap + 1, 0, 0, 0, 0])          # over-cap values are capped and flagged
    obu.add_day(1, [0, 0, 0, DEFAULT.daily_event_cap + 5, 0])
    assert obu._days[0][0] == DEFAULT.daily_mile_cap and obu._days[1][3] == DEFAULT.daily_event_cap
    assert obu.capped_days == [0, 1]
    with pytest.raises(ValueError):
        obu.add_day(2, [-1, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        obu.interval_record(0, 32)
    assert (1 << DEFAULT.x_bits) > 31 * DEFAULT.daily_mile_cap


@pytest.mark.skipif(not os.path.exists(DATASET), reason="So et al. dataset not available")
def test_epoch_record_equals_sum_of_daily_scores():
    """The epoch score of the OBU's aggregated record equals the sum of the daily scores (So et al. policies)."""
    model = load_model()
    rows = [r for _, r in zip(range(300), iter_policies(DATASET))]
    for k, row in enumerate(rows):
        recs = daily_records(row, 31, seed=k)
        x = [31] + [sum(r[i] for r in recs) for i in range(1, 6)]
        assert model.daily_score(x) == sum(model.daily_score(r) for r in recs)
        assert all(v <= c for v, c in zip(x, DEFAULT.epoch_caps(31)))


# ---------------------------------------------------------------------------------------- key setup
def _ser_pk(pk) -> bytes:
    return o.Serialize(pk, o.BINARY)


def test_single_key_setup_confirmation_and_signed_bundle(world):
    """One insurer key: the TSP holds no secret FHE key; key confirmation runs one test record through the
    TSP's pipeline and fails for a wrong installed Enc(w); a substituted bundle does not verify. The OBU receives only
    the signed public key pk_Ins (no Enc(w), no evaluation key, no digest of them) and refuses the TSP's bundle."""
    risk, ins, keys, tsp = world["risk"], world["ins"], world["keys"], world["tsp"]
    assert not any(isinstance(v, o.PrivateKey) for v in vars(tsp).values())
    assert sum(isinstance(v, o.PrivateKey) for v in vars(ins).values()) == 1
    bad = TSP(world["gs"], risk, keys, ins.vk, 1)
    bad.ct_w = risk.encrypt_weights(keys.pk, [wi + 1 for wi in world["w"]])   # TSP installs a different Enc(w)
    assert not ins.key_confirmation(bad)
    assert ins.key_confirmation(TSP(world["gs"], risk, keys, ins.vk, 1))
    # substituted public key / evaluation-key digest / epoch range: the signed bundle no longer verifies
    other = RiskContext(P, DEFAULT.m).keygen()
    sub = KeyBundle(other.publicKey, keys.ct_w, keys.key_tag, keys.version, keys.epochs, keys.evk_digest, keys.sig)
    assert not sub.verify(ins.vk, 1)
    sub = KeyBundle(keys.pk, keys.ct_w, keys.key_tag, keys.version, keys.epochs, b"\0" * 32, keys.sig)
    assert not sub.verify(ins.vk, 1)
    assert not keys.verify(ins.vk, 13)
    with pytest.raises(ValueError):
        TSP(world["gs"], risk, sub, ins.vk, 1)
    # the OBU receives only pk_Ins: not the TSP's bundle, so neither Enc(w) nor the evaluation keys nor a digest
    pkc = world["obu_keys"]
    assert pkc.verify(ins.vk, 1) and not pkc.verify(ins.vk, 13)
    assert set(vars(pkc)) == {"pk", "key_tag", "version", "epochs", "sig"}
    assert _held_ciphertext_ids(vars(pkc)) == set()
    assert _ser_pk(pkc.pk) == _ser_pk(keys.pk) and pkc.key_tag == keys.key_tag
    obu = OBU(1, b"k" * 32, pkc, ins.vk, risk, world["gs"], world["gi"].enrol(), b"I", DEFAULT, 1)
    assert not any(isinstance(v, (KeyBundle, o.Ciphertext)) for v in vars(obu).values())
    with pytest.raises(TypeError):
        OBU(1, b"k" * 32, keys, ins.vk, risk, world["gs"], world["gi"].enrol(), b"I", DEFAULT, 1)
    # substituted public key / epoch range, or the bundle's signature moved onto the certificate: rejected
    for bad_c in (PublicKeyCert(other.publicKey, pkc.key_tag, pkc.version, pkc.epochs, pkc.sig),
                  PublicKeyCert(pkc.pk, pkc.key_tag, pkc.version, (1, 13), pkc.sig),
                  PublicKeyCert(pkc.pk, pkc.key_tag, pkc.version, pkc.epochs, keys.sig)):
        assert not bad_c.verify(ins.vk, 1)
        with pytest.raises(ValueError):
            OBU(1, b"k" * 32, bad_c, ins.vk, risk, world["gs"], world["gi"].enrol(), b"I", DEFAULT, 1)
    # only the insurer's key decrypts
    v = [secrets.randbelow(P) for _ in range(32)]
    ct = risk.encrypt_vector(keys.pk, v)
    assert risk.decrypt(ct, ins._sk, 32) == v and risk.decrypt(ct, other.secretKey, 32) != v


# ---------------------------------------------------------------------------------------- receipts, setup window
def test_receipts_setup_window_and_published_output(world):
    ins, tsp = world["ins"], world["tsp"]
    obu = new_obu(world, seed=7)
    msg = obu.epoch_message(31)
    with pytest.raises(PermissionError):
        tsp.confirmation_evaluate(msg.ct_payload)                  # confirmation window is closed
    receipt = tsp.receive(msg)
    assert verify_receipt(tsp.vk, msg, receipt) and not verify_receipt(tsp.vk, obu.resend(), receipt)
    out = tsp.end_epoch()
    assert len(out) == 1                                           # the insurer receives the batch output only
    assert ins.process_batch(out).reports[obu.ident].status == "ok"


def _held_ciphertext_ids(obj, seen=None) -> set:
    """ct_id of every OpenFHE ciphertext reachable from the attributes of `obj` (containers walked recursively)."""
    seen = set() if seen is None else seen
    out = set()
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, o.Ciphertext):
        out.add(ct_id(obj))
    elif isinstance(obj, dict):
        for v in list(obj.keys()) + list(obj.values()):
            out |= _held_ciphertext_ids(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            out |= _held_ciphertext_ids(v, seen)
    return out


def test_tsp_retention_no_record_ciphertext_no_result_after_end_epoch(world):
    """Retention after `receive` the TSP holds no record ciphertext (only the evaluated, slot-filled
    result); after `end_epoch` it holds no result of that epoch. The signed bundle (public key, evaluation keys, Enc(w))
    is the only ciphertext state the TSP keeps across epochs."""
    tsp = TSP(world["gs"], world["risk"], world["keys"], world["ins"].vk, 1)
    assert world["ins"].key_confirmation(tsp)
    tsp.close_setup()
    standing = {ct_id(tsp.ct_w)}
    assert _held_ciphertext_ids(vars(tsp)) == standing
    obu = new_obu(world, seed=11)
    msgs = [obu.epoch_message(31), new_obu(world, seed=12).epoch_message(31)]
    for m in msgs:
        assert tsp.receive(m) is not None
    held = _held_ciphertext_ids(vars(tsp))
    assert not held & {ct_id(m.ct_payload) for m in msgs}          # no record ciphertext after receive
    assert len(held - standing) == 2                               # the two evaluated, slot-filled results of epoch 1
    out = tsp.end_epoch(1)
    assert len(out) == 2 and 1 not in tsp.pools                    # the batch is handed over
    assert _held_ciphertext_ids(vars(tsp)) == standing              # no result of epoch 1 is held after end_epoch
    assert not _held_ciphertext_ids(vars(tsp)) & {ct_id(c) for c in out}


# ---------------- daily records
@pytest.fixture(scope="module")
def daily(world):
    """Daily configuration (default Params) on the same keys."""
    ins = Insurer(DEFAULT, world["risk"], world["w"], epoch=1)
    ins._sk, ins.keys, ins._sigk, ins.vk = world["ins"]._sk, world["keys"], world["ins"]._sigk, world["ins"].vk
    return dict(world, ins=ins)


def daily_obu(daily, D=6, seed=0):
    ident, K = daily["ins"].enroll(D)
    obu = OBU(ident, K, daily["obu_keys"], daily["ins"].vk, daily["risk"], daily["gs"], daily["gi"].enrol(),
              b"INS-0001", DEFAULT, 1)
    rng = random.Random(seed)
    for d in range(D):
        if d != 2:                                                  # day 2: no driving
            obu.add_day(d, [rng.randrange(0, 900), rng.randrange(0, 300), 0, rng.randrange(0, 3), 0])
    return obu


def test_daily_records_multiset_and_rules(daily):
    """Exactly D distinct records; identical duplicate once; replay of one day + drop of another -> missing;
    more than D distinct valid records (lost sealed storage) -> conflict; scores = daily scores (multiset)."""
    ins, tsp, w = daily["ins"], daily["tsp"], daily["w"]
    obu = daily_obu(daily, D=6)
    msgs = [m for _, m in obu.messages(6)]
    outs = [published(daily, m) for m in msgs]
    r = ins.process_batch(outs + [outs[0]])                   # identical duplicate of day 0
    rep = r.reports[obu.ident]
    assert rep.status == "ok" and rep.identical_duplicates == 1 and rep.expected_records == 6
    daily_scores = sorted(sum(a * b for a, b in zip(w, obu.interval_record(k, 6))) for k in range(6))
    assert sorted(rep.scores) == daily_scores and w[0] in rep.scores  # the non-driving day decrypts to w_0
    r = ins.process_batch(outs[:3] + [outs[0]] + outs[4:])     # replay day 0, drop day 3
    assert r.reports[obu.ident].status == "missing"
    b = OBU(obu.ident, obu.K, daily["obu_keys"], ins.vk, daily["risk"], daily["gs"], daily["gi"].enrol(),
            b"INS-0001", DEFAULT, 1)                                 # replacement unit, sealed storage lost
    b.add_day(0, [1, 1, 0, 0, 0])
    extra = published(daily, b.record_message(0, 6))
    r = ins.process_batch(outs + [extra])
    assert r.reports[obu.ident].status == "conflict"


def test_uu_per_record_schedule_receipts_retry_and_deadline(daily):
    """Independent upload times in
    [k-th interval end, + W); each record is attempted at its own time over its own connection (attempts are ordered by
    the per-record times, never batched); a record stays pending until a valid receipt and is retried at an independently
    drawn later time; records not acknowledged by the settlement deadline are missing."""
    from sp2ubi.protocol import uu_upload_loop
    tsp = daily["tsp"]
    obu = daily_obu(daily, D=4, seed=9)
    due = obu.schedule(4, random.Random(1))
    W = DEFAULT.upload_window_s
    assert all((k + 1) * 86400 <= t < (k + 1) * 86400 + W for k, t in due.items())
    t1, k1 = obu.next_attempt()
    assert (t1, k1) == min((t, k) for k, t in due.items())          # earliest drawn time first, one record
    m1 = obu.attempt(k1)
    assert not obu.acknowledge(k1, m1, None, tsp.vk) and obu.unacknowledged() == [0, 1, 2, 3]
    rng = random.Random(2)
    obu.retry(k1, t1, rng)                                         # failed attempt: own retry time in [t1, t1 + 1 h)
    assert t1 <= obu._next[k1] < t1 + DEFAULT.retry_window_s
    assert obu.acknowledge(k1, obu.attempt(k1), tsp.receive(obu.attempt(k1)), tsp.vk) is False  # receipt of another encryption
    m1 = obu.attempt(k1)
    assert obu.acknowledge(k1, m1, tsp.receive(m1), tsp.vk) and len(obu.unacknowledged()) == 3
    assert obu.next_attempt()[1] != k1
    with pytest.raises(ValueError):
        obu.attempt(k1)                                            # acknowledged: never sent again
    # outage covering the remaining records' times until after the deadline: they stay pending (missing)
    deadline = DEFAULT.settlement_deadline_s(31)
    assert deadline == 31 * 86400 + W + DEFAULT.settlement_grace_s
    times = []

    def send(k, t):
        times.append((t, k))
        m = obu.attempt(k)
        return obu.acknowledge(k, m, tsp.receive(m), tsp.vk)
    # outage until day 12 (after every drawn time): every pending record is retried after it, each at its own time
    uu_upload_loop(obu, deadline, [(0.0, 12 * 86400)], rng, send)
    assert obu.unacknowledged() == [] and len({t for t, _ in times}) == len(times) == 3
    assert all(12 * 86400 <= t < 12 * 86400 + DEFAULT.retry_window_s for t, _ in times)
    # another vehicle whose link is down from day 0 until after the settlement deadline: nothing reaches the TSP, every
    # record stays pending and is 'missing' at settlement
    obu = daily_obu(daily, D=4, seed=10)
    obu.schedule(4, random.Random(3))
    times.clear()
    uu_upload_loop(obu, deadline, [(0.0, deadline + 1)], rng, send)
    assert times == [] and obu.unacknowledged() == [0, 1, 2, 3] and obu.next_attempt()[0] > deadline
    tsp.end_epoch()


def test_nonce_already_sealed_in_the_epoch_is_redrawn(daily, monkeypatch):
    """A nonce value already sealed for another record of the same epoch is redrawn, so two records of one
    OBU never share one-time HomMAC keys."""
    import sp2ubi.entities as ent
    draws = iter([5, 5, 5, 7, 9])
    monkeypatch.setattr(ent, "fresh_nonce", lambda p: next(draws))
    obu = daily_obu(daily, D=3, seed=1)
    for k in range(3):
        obu.record_message(k, 3)
    us = [obu.open_dispute(k)[0]["u"] for k in range(3)]
    assert us == [5, 7, 9]
    assert obu.resend(0) is not None and obu.open_dispute(0)[0]["u"] == 5   # re-send keeps the sealed nonce


def test_late_record_is_not_counted_in_the_next_epoch(daily):
    """A record of epoch 1 presented to the settlement of epoch 2 fails HomMAC (epoch label in
    the one-time keys) and is rejected; epoch 1 of that policy stays 'missing'."""
    ins1, tsp = daily["ins"], daily["tsp"]
    obu = daily_obu(daily, D=2, seed=4)
    out = published(daily, obu.record_message(0, 2))
    ins2 = Insurer(DEFAULT, daily["risk"], daily["w"], epoch=2)
    ins2._sk, ins2.keys, ins2.K, ins2.D = ins1._sk, ins1.keys, {obu.ident: obu.K}, {obu.ident: 2}
    assert ins2.process_batch([out]).verdicts == [REJECTED]
    assert ins1.process_batch([]).reports[obu.ident].status == "missing"


def test_receipted_record_dropped_by_the_tsp_is_counted_without_opening(daily):
    """A record the TSP receipted and then dropped is counted as a receipted record that the insurer did not accept
    (the TSP is accountable for it). The evidence is the receipt plus the opening of c' = H(ID|t|u|salt') and reveals
    no x and no interval index."""
    ins, tsp = daily["ins"], daily["tsp"]
    tsp.end_epoch()                                                # flush records left by earlier tests
    obu = daily_obu(daily, D=6, seed=31)
    for k, (_, m) in enumerate(obu.messages(6)):
        assert obu.acknowledge(k, m, tsp.receive(m), tsp.vk)       # every record receipted by the TSP
    outs = tsp.end_epoch()
    assert len(outs) == 6
    rep = ins.process_batch(outs[1:]).reports[obu.ident]     # the TSP drops one receipted record
    assert rep.status == "missing" and rep.accepted_records == 5
    accepted = Insurer.accepted_nonces(rep)                        # returned to the OBU at settlement
    evidence = obu.receipt_evidence(accepted)
    assert len(evidence) == 1                                      # only the dropped record
    digest, commitment, binding, receipt, opening = evidence[0]
    sealed = {k: obu.open_dispute(k)[0] for k in range(6)}
    dropped = next(r for r in sealed.values() if r["u"] not in accepted)
    # the evidence reveals no feature vector and no interval: only (ID, t, u, salt'), three hashes and a signature
    assert opening[:3] == (obu.ident, 1, dropped["u"]) and len(opening) == 4 and isinstance(opening[3], bytes)
    assert all(isinstance(v, (bytes, int)) for v in (digest, commitment, binding, receipt) + opening)
    assert {v for v in opening if isinstance(v, int)} == {obu.ident, 1, dropped["u"]}      # no x component, no interval
    assert not any(isinstance(v, (dict, list)) for v in (digest, commitment, binding, receipt) + opening)
    assert ins.receipted_missing_days(obu.ident, rep, evidence, tsp.vk) == 1
    assert Insurer.deviation_notice(ins.process_batch(outs[1:]), receipted_missing=1) is True
    # not counted: evidence for an accepted nonce, a duplicate, a wrong salt', another policy, a wrong digest or cm_r
    every = obu.receipt_evidence(set())                            # what an OBU would send for all receipted records
    assert len(every) == 6
    assert ins.receipted_missing_days(obu.ident, rep, every, tsp.vk) == 1          # accepted u are ignored
    assert ins.receipted_missing_days(obu.ident, rep, evidence * 3, tsp.vk) == 1   # duplicates count once
    bad = [(digest, commitment, binding, receipt, opening[:3] + (b"\0" * 32,)),
           (digest, commitment, binding, receipt, (obu.ident + 1,) + opening[1:]),
           (b"\0" * 32, commitment, binding, receipt, opening),
           (digest, b"\0" * 32, binding, receipt, opening)]                 # the receipt also signs cm_r
    assert ins.receipted_missing_days(obu.ident, rep, bad, tsp.vk) == 0


def test_honest_settlement_opens_nothing(daily):
    """When every record is accepted, the OBU presents no evidence and the insurer opens nothing (neither
    c nor c'); the insurer's view stays the multiset of scores."""
    ins, tsp = daily["ins"], daily["tsp"]
    tsp.end_epoch()
    obu = daily_obu(daily, D=4, seed=33)
    for k, (_, m) in enumerate(obu.messages(4)):
        assert obu.acknowledge(k, m, tsp.receive(m), tsp.vk)
    rep = ins.process_batch(tsp.end_epoch()).reports[obu.ident]
    assert rep.status == "ok"
    assert obu.receipt_evidence(Insurer.accepted_nonces(rep)) == []
    assert ins.receipted_missing_days(obu.ident, rep, [], tsp.vk) == 0


def test_conflict_yields_no_score_and_requests_no_evidence(daily):
    """A 'conflict' (attributable to the TSP or to a lost-storage OBU) yields no score for the policy-epoch; its
    settlement requests no evidence and opens nothing (see
    test_epoch_label.test_conflict_epoch_of_a_full_month_settles_without_evidence)."""
    ins, tsp = daily["ins"], daily["tsp"]
    tsp.end_epoch()
    obu = daily_obu(daily, D=6, seed=32)
    outs = [published(daily, m) for _, m in obu.messages(6)]
    b = OBU(obu.ident, obu.K, daily["obu_keys"], ins.vk, daily["risk"], daily["gs"], daily["gi"].enrol(),
            b"INS-0001", DEFAULT, 1)                                 # replacement unit, sealed storage lost
    b.add_day(0, [1, 1, 0, 0, 0])
    conf = ins.process_batch(outs + [published(daily, b.record_message(0, 6))]).reports[obu.ident]
    assert conf.status == "conflict" and conf.score_sum is None
    assert Insurer.nonces_from_settlement(ins.settlement_message(obu.ident, conf)) is None
    assert ins.settle(obu.ident, conf, [], tsp.vk) == (0, [])
