"""The cleartext epoch label, the TSP's admission rule, the dispute checks, the undated dispute commitment, the end of
a membership, and the settlement of a dropped or altered record by the dispute procedure.

A record of epoch t may reach the TSP during epoch t+1 (upload delay in [0, W_u) plus re-sends), so the group-signed
message carries the epoch index t in clear, the TSP pools each record by its signed t and its receipt signs t. Without
the label a record of epoch t pooled with epoch t+1 would fail the HomMAC under (ID, t+1, u) and look like a deviation of
the TSP. The TSP gives no receipt for a record of an epoch whose batch it has published, so a receipted record that is
not accepted is always a deviation of the TSP. The dispute procedure opens a receipted record only if its nonce was
neither accepted nor opened before, and the opening carries no date. The
receipt signs cm_r as well, the OBU keeps no ciphertext, the dispute opens cm_r against the receipt, a conflict epoch of
a full month settles without evidence, and a record of an ended policy is verified before it is discarded. The
settlement messages carry no padding (the TSP holds no policy). Each test uses its own TSP, so its epochs are its own."""
import copy
import random

import pytest

from sp2ubi.entities import (DISCARDED, OBU, REJECTED, TSP, EpochMessage, Insurer, Receipted,
                              decode_settlement_upload, encode_settlement_upload, settlement_item_bytes_max,
                              verify_receipt)
from sp2ubi.fhe import SLOT_ID, SLOT_X, RiskContext
from sp2ubi.groupsig_bbs04 import BBS04, Issuer as GroupIssuer
from sp2ubi.params import DEFAULT, Params
from sp2ubi.risk_model import load_model

P = DEFAULT.p
MONTHLY = Params(interval_days=31)
DAILY = Params(interval_days=1)


def make_world(params):
    risk = RiskContext(P, DEFAULT.m)
    model = load_model()
    ins = Insurer(params, risk, model.w, epoch=1)
    keys = ins.setup(epochs=(1, 12))
    gi = GroupIssuer()
    gs = BBS04(gi.gpk)
    w = {"risk": risk, "ins": ins, "keys": keys, "obu_keys": ins.obu_keys, "gs": gs, "gi": gi, "w": model.w, "params": params}
    tsp = new_tsp(w, confirm=True)
    assert ins.key_confirmation(tsp)
    return w


def new_tsp(world, confirm=False):
    tsp = TSP(world["gs"], world["risk"], world["keys"], world["ins"].vk, 1)
    if not confirm:
        tsp.close_setup()
    return tsp


@pytest.fixture(scope="module")
def world():
    return make_world(MONTHLY)


@pytest.fixture(scope="module")
def daily():
    return make_world(DAILY)


def obu_for(world, ident, K, epoch, seed, days=31):
    gkey = world["gi"].enrol()
    obu = OBU(ident, K, world["obu_keys"], world["ins"].vk, world["risk"], world["gs"], gkey, b"INS-0001",
              world["params"], epoch)
    rng = random.Random(seed)
    for d in range(days):
        obu.add_day(d, [rng.randrange(0, 900), rng.randrange(0, 300), rng.randrange(0, 400), rng.randrange(0, 3),
                        rng.randrange(0, 3)])
    return obu, gkey


def altered(world, ct, slot, delta):
    """A deviating TSP adds Enc(delta) to one slot of a result (public-key operation, tags untouched)."""
    v = [0] * 40
    v[slot] = delta % P
    return world["risk"].cc.EvalAdd(ct, world["risk"].encrypt_vector(world["keys"].pk, v))


def test_records_pooled_by_signed_epoch(world):
    ins, tsp = world["ins"], new_tsp(world)
    pols = [ins.enroll(31) for _ in range(2)]
    late = [obu_for(world, i, K, 1, seed=10 + j)[0].epoch_message(31) for j, (i, K) in enumerate(pols)]
    early = [obu_for(world, i, K, 2, seed=20 + j)[0].epoch_message(31) for j, (i, K) in enumerate(pols)]
    assert [m.epoch for m in late] == [1, 1] and [m.epoch for m in early] == [2, 2]
    for m in (early[0], late[0], early[1], late[1]):                 # epoch-1 records arrive during epoch 2
        assert tsp.receive(m) is not None
    assert len(tsp.pools[1]) == 2 and len(tsp.pools[2]) == 2
    ins.epoch = 1
    res1 = ins.process_batch(tsp.end_epoch(1))
    assert res1.rejected == 0 and all(res1.reports[i].accepted for i, _ in pols)
    ins.epoch = 2
    res2 = ins.process_batch(tsp.end_epoch(2))
    assert res2.rejected == 0 and all(res2.reports[i].accepted for i, _ in pols)
    ins.epoch = 1


def test_epoch_label_is_signed_and_receipted(world):
    ins, tsp, gs = world["ins"], new_tsp(world), world["gs"]
    ident, K = ins.enroll(31)
    obu, gkey = obu_for(world, ident, K, 1, seed=30)
    msg = obu.epoch_message(31)
    moved = copy.copy(msg)
    moved.epoch = 2                                                  # relabelled on the link: signature fails
    assert tsp.receive(moved) is None
    outside = obu.resend()
    outside.epoch = 13                                               # outside the key bundle's epoch range
    outside.signature = gs.sign(gkey, outside.signed_bytes())
    assert tsp.receive(outside) is None
    receipt = tsp.receive(msg)
    assert receipt is not None and verify_receipt(tsp.vk, msg, receipt)
    assert not verify_receipt(tsp.vk, moved, receipt)               # the receipt names the signed epoch


def test_no_receipt_after_publication_and_no_false_deviation(world):
    """Late receipt, with the admission rule: a validly signed record of epoch 1 that reaches an
    honest TSP after it has published the batch of epoch 1 gets no receipt, so its OBU holds no evidence, the record
    counts as not uploaded, and the deviation notice of the honest run stays False."""
    ins, tsp = world["ins"], new_tsp(world)
    ins.epoch = 1
    (a, Ka), (b, Kb) = ins.enroll(31), ins.enroll(31)
    m_a = obu_for(world, a, Ka, 1, seed=1)[0].epoch_message(31)
    assert tsp.receive(m_a) is not None
    res = ins.process_batch(tsp.end_epoch(1))                   # settlement deadline of epoch 1: batch published
    obu_b = obu_for(world, b, Kb, 1, seed=2)[0]
    late = obu_b.epoch_message(31)                                   # last re-send, delayed in transit
    r = tsp.receive(late)
    assert r is None and tsp.refused_closed == 1 and not tsp.pools.get(1)
    assert not obu_b.acknowledge(0, late, r, tsp.vk)
    ev = obu_b.receipt_evidence(Insurer.accepted_nonces(res.reports[b]))
    n_rc = ins.receipted_missing_days(b, res.reports[b], ev, tsp.vk)
    assert ev == [] and n_rc == 0
    assert res.reports[a].accepted and res.reports[b].status == "missing"
    assert Insurer.deviation_notice(res, receipted_missing=n_rc) is False
    m2 = obu_for(world, a, Ka, 2, seed=3)[0].epoch_message(31)      # the next epoch stays open
    assert tsp.receive(m2) is not None
    assert tsp.end_epoch(1) == []                                    # a closed epoch publishes nothing further


def test_dispute_opens_only_a_receipted_record_neither_accepted_nor_opened(world):
    ins, tsp = world["ins"], new_tsp(world)
    ins.epoch = 1
    ident, K = ins.enroll(31)
    obu, _ = obu_for(world, ident, K, 1, seed=40)
    msg = obu.epoch_message(31)
    receipt = tsp.receive(msg)
    assert obu.acknowledge(0, msg, receipt, tsp.vk)
    rcp = obu._receipted[0]                                          # what the OBU keeps: the receipted values only
    assert rcp == Receipted.of(msg) and not any(isinstance(v, EpochMessage) for v in vars(obu).values())
    assert all(isinstance(getattr(rcp, f), (bytes, int)) for f in ("digest", "commitment", "binding", "epoch"))
    tsp.end_epoch(1)                                                 # suppose the TSP dropped it from the batch
    record, salt = obu.open_dispute(0)
    # the opening of cm_r carries no date: no interval index and no day index
    assert set(record) == {"id", "epoch", "x", "u", "bind_salt"}
    assert obu._meta[0] == {"interval": 0, "capped_days": []}       # kept in sealed storage, outside the commitment
    expected = sum(wi * xi for wi, xi in zip(world["w"], record["x"])) % P
    assert ins.dispute_record(ident, rcp, receipt, tsp.vk, record, salt, {record["u"]}, set()) is None   # accepted
    opened = set()
    assert ins.dispute_record(ident, rcp, receipt, tsp.vk, record, salt, set(), opened) == expected
    assert ins.dispute_record(ident, rcp, receipt, tsp.vk, record, salt, set(), opened) is None          # opened
    other, _ = ins.enroll(31)
    assert ins.dispute_record(other, rcp, receipt, tsp.vk, record, salt, set(), set()) is None           # policy
    bad = dict(record, x=[record["x"][0], record["x"][1] + 1] + record["x"][2:])
    assert ins.dispute_record(ident, rcp, receipt, tsp.vk, bad, salt, set(), set()) is None              # opening
    dated = dict(record, interval=0)
    assert ins.dispute_record(ident, rcp, receipt, tsp.vk, dated, salt, set(), set()) is None            # no date
    # the receipt signs cm_r: a cm_r that the TSP did not receipt is refused, even if it opens
    other_obu = obu_for(world, ident, K, 1, seed=41)[0]
    m2 = other_obu.epoch_message(31)
    rec2, salt2 = other_obu.open_dispute(0)
    swapped = Receipted(rcp.digest, m2.commitment, rcp.binding, rcp.epoch)
    assert ins.dispute_record(ident, swapped, receipt, tsp.vk, rec2, salt2, set(), set()) is None       # not receipted
    assert not other_obu.acknowledge(0, m2, receipt, tsp.vk)        # a receipt of another record is not accepted


def test_records_of_an_ended_policy_are_discarded_not_a_deviation(world):
    """An OBU of an ended policy that keeps uploading (e.g. lost with its keys) does not set the deviation bit;
    a TSP that moves a live record into an ended identity is still detected through the live policy's receipt."""
    ins, tsp = world["ins"], new_tsp(world)
    ins.epoch = 1
    (a, Ka), (b, Kb) = ins.enroll(31), ins.enroll(31)
    obu_a = obu_for(world, a, Ka, 1, seed=50)[0]
    obu_b = obu_for(world, b, Kb, 1, seed=51)[0]
    ins.end_policy(b)
    m_a, m_b = obu_a.epoch_message(31), obu_b.epoch_message(31)
    assert obu_a.acknowledge(0, m_a, tsp.receive(m_a), tsp.vk) and tsp.receive(m_b) is not None
    batch = tsp.end_epoch(1)
    res = ins.process_batch(batch)
    assert res.discarded == 1 and res.rejected == 0 and b not in res.reports and res.reports[a].accepted
    assert sorted(res.verdicts) == sorted(["accepted", DISCARDED])
    assert Insurer.deviation_notice(res) is False
    # deviating TSP: the live record of a relabelled to the ended identity b
    tsp2 = new_tsp(world)
    obu_a2 = obu_for(world, a, Ka, 1, seed=52)[0]
    m = obu_a2.epoch_message(31)
    assert obu_a2.acknowledge(0, m, tsp2.receive(m), tsp2.vk)
    out = tsp2.end_epoch(1)
    moved = altered(world, out[0], SLOT_ID, b - a)
    res2 = ins.process_batch([moved])
    assert res2.discarded == 0 and res2.rejected == 1 and res2.reports[a].status == "missing"   # verified first
    ev = obu_a2.receipt_evidence(Insurer.accepted_nonces(res2.reports[a]))
    n_rc = ins.receipted_missing_days(a, res2.reports[a], ev, tsp2.vk)
    assert n_rc == 31 and Insurer.deviation_notice(res2, receipted_missing=n_rc) is True
    # an altered record of the ended policy itself is rejected, not discarded (verify before discarding)
    tsp3 = new_tsp(world)
    m_b3 = obu_for(world, b, Kb, 1, seed=53)[0].epoch_message(31)
    assert tsp3.receive(m_b3) is not None
    out3 = tsp3.end_epoch(1)
    bad = altered(world, out3[0], SLOT_X, 1)
    res3 = ins.process_batch([bad])
    assert res3.verdicts == [REJECTED] and res3.discarded == 0 and Insurer.deviation_notice(res3) is True
    res4 = ins.process_batch(out3)
    assert res4.verdicts == [DISCARDED] and Insurer.deviation_notice(res4) is False
    ins.ended.discard(b)


def _settled_run(world, ident, K, seed, handling, days):
    """One epoch of a policy whose record the TSP accepts, alters (score slot shifted: rejected) or drops; the OBU
    disputes every receipted record that was not accepted. Returns the number of settlement items, the multiset of
    record scores after the dispute procedure and the deviation bit."""
    ins, tsp = world["ins"], new_tsp(world)
    obu = obu_for(world, ident, K, 1, seed=seed, days=days)[0]
    for k, (_, m) in enumerate(obu.messages(days)):
        assert obu.acknowledge(k, m, tsp.receive(m), tsp.vk)
    out = tsp.end_epoch(1)
    if handling == "alter":
        bad = altered(world, out[0], SLOT_X, 1)
        out = [bad] + out[1:]
    elif handling == "drop":
        out = out[1:]
    res = ins.process_batch(out)
    rep = res.reports[ident]
    to_obu = ins.settlement_message(ident, rep)                      # authenticated and encrypted to the TEE
    upload = obu.settlement_upload(Insurer.nonces_from_settlement(to_obu))
    wire = encode_settlement_upload(upload)
    assert len(wire) <= len(upload) * settlement_item_bytes_max(world["params"])
    assert decode_settlement_upload(wire) == upload
    n_rc, disputed = ins.settle(ident, rep, decode_settlement_upload(wire), tsp.vk)
    scores = tuple(sorted(list(rep.scores) + list(disputed)))
    notice = Insurer.deviation_notice(res, receipted_missing=n_rc)
    return len(upload), scores, notice


@pytest.mark.parametrize("cfg", ["monthly", "daily"])
def test_dispute_settles_an_altered_or_dropped_record_at_its_score(cfg, world, daily):
    """A receipted record that the TSP alters (rejected) or drops is disputed by the OBU
    and enters with the score of its features, as if it had been accepted; the TSP's deviation is detected (one bit for
    the epoch). An honest epoch needs no settlement item."""
    w, days = (world, 31) if cfg == "monthly" else (daily, 4)
    ident, K = w["ins"].enroll(days)
    runs = {h: _settled_run(w, ident, K, 60, h, days) for h in ("accept", "alter", "drop")}
    assert [runs[h][0] for h in ("accept", "alter", "drop")] == [0, 1, 1]  # one item per disputed record, no padding
    assert len({r[1] for r in runs.values()}) == 1                   # same scores after the dispute procedure
    assert [runs[h][2] for h in ("accept", "alter", "drop")] == [False, True, True]


def test_conflict_epoch_of_a_full_month_settles_without_evidence(daily):
    """Daily configuration, 31 days, an OBU and its replacement (sealed storage lost, fresh
    nonces) upload 62 receipted records that all verify, so the policy-epoch is a conflict with more valid records
    than days. The settlement message requests no evidence (the NO_EVIDENCE marker), each OBU's upload is empty, the
    insurer opens nothing and the policy-epoch yields no score."""
    w = daily
    ins, tsp = w["ins"], new_tsp(w)
    ins.epoch = 1
    ident, K = ins.enroll(31)
    obus = [obu_for(w, ident, K, 1, seed=70 + j)[0] for j in range(2)]
    for obu in obus:
        for k, (_, m) in enumerate(obu.messages(31)):
            assert obu.acknowledge(k, m, tsp.receive(m), tsp.vk)
    res = ins.process_batch(tsp.end_epoch(1))
    rep = res.reports[ident]
    assert rep.status == "conflict" and rep.accepted_records == 62
    to_obu = ins.settlement_message(ident, rep)
    assert len(to_obu) == 8 and Insurer.nonces_from_settlement(to_obu) is None
    opened = set()
    for obu in obus:
        upload = obu.settlement_upload(Insurer.nonces_from_settlement(to_obu))
        assert upload == [] and encode_settlement_upload(upload) == b""
        assert ins.settle(ident, rep, upload, tsp.vk, opened) == (0, [])
    assert opened == set()
    # An honest conflict epoch is no deviation of the TSP, so no deviation notice is sent
    receipted_missing = ins.receipted_missing_days(
        ident, rep, [e for obu in obus for e in obu.settlement_upload(Insurer.nonces_from_settlement(to_obu))], tsp.vk)
    assert receipted_missing == 0 and Insurer.deviation_notice(res, receipted_missing) is False
    assert rep.score_sum is None
    # an ordinary epoch is unaffected: its message carries the accepted nonces (no NO_EVIDENCE marker)
    ident2, K2 = ins.enroll(31)
    tsp2 = new_tsp(w)
    obu2 = obu_for(w, ident2, K2, 1, seed=72)[0]
    for k, (_, m) in enumerate(obu2.messages(31)):
        assert obu2.acknowledge(k, m, tsp2.receive(m), tsp2.vk)
    rep2 = ins.process_batch(tsp2.end_epoch(1)).reports[ident2]
    msg2 = ins.settlement_message(ident2, rep2)
    assert rep2.status == "ok" and len(msg2) == 248 and len(Insurer.nonces_from_settlement(msg2)) == 31
    assert Insurer.nonces_from_settlement(msg2) == Insurer.accepted_nonces(rep2)
