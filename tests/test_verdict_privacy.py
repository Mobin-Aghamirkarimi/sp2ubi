"""No verdict of the insurer's checks reaches the TSP.

Single insurer key: the insurer decrypts the published batch itself and makes no call to the TSP while it
processes the batch; the per-record verdicts stay with the insurer, and the only outcome toward the TSP is one bit per
epoch (`Insurer.deviation_notice`). The probes are those of
`experiments/hommac_checks/verdict_channel.py`, run on real BFV ciphertexts: a feature shift in a
zero-weight slot (accepted, correct score) and a shifted score (rejected)."""
import random

import pytest

from sp2ubi import hommac
import sp2ubi.entities as ent
from sp2ubi.entities import OBU, REJECTED, TSP, Insurer
from sp2ubi.fhe import SLOT_X, RiskContext
from sp2ubi.groupsig_bbs04 import BBS04, Issuer as GroupIssuer
from sp2ubi.params import DEFAULT, Params
from sp2ubi.risk_model import load_model

P = DEFAULT.p
MONTHLY = Params(interval_days=31)


class SpyTSP(TSP):
    """Records every public attribute of the TSP that is touched while `log` is a list (any call from the insurer)."""
    log = None

    def __getattribute__(self, name):
        log = SpyTSP.log
        if log is not None and not name.startswith("_"):
            log.append((name, None))
        return super().__getattribute__(name)

    def end_epoch(self, epoch=None):
        """Each test scenario uses epoch 1 anew, so publication does not close it here (the admission rule is tested in test_epoch_label.py)."""
        out = super().end_epoch(epoch)
        self.closed.clear()
        return out


@pytest.fixture(scope="module")
def world():
    risk = RiskContext(P, DEFAULT.m)
    model = load_model()
    ins = Insurer(MONTHLY, risk, model.w, epoch=1)
    keys = ins.setup(epochs=(1, 12))
    gi = GroupIssuer()
    gs = BBS04(gi.gpk)
    tsp = SpyTSP(gs, risk, keys, ins.vk, 1)
    assert ins.key_confirmation(tsp)
    tsp.close_setup()
    return {"risk": risk, "ins": ins, "keys": keys, "obu_keys": ins.obu_keys, "tsp": tsp, "gs": gs, "gi": gi, "w": model.w}


def zero_weight_slot(w):
    zs = [i for i, wi in enumerate(w) if i > 0 and wi % P == 0]
    if not zs:
        pytest.skip("fitted model has no zero weight")
    return zs[0]


def outputs(world, seed):
    """Two policies' monthly records through the TSP pipeline, plus one accepted and one rejected probe."""
    risk, keys, ins, tsp = world["risk"], world["keys"], world["ins"], world["tsp"]
    rng = random.Random(seed)
    msgs = []
    for _ in range(2):
        ident, K = ins.enroll(31)
        obu = OBU(ident, K, world["obu_keys"], ins.vk, risk, world["gs"], world["gi"].enrol(), b"INS-0001", MONTHLY, 1)
        for d in range(31):
            obu.add_day(d, [rng.randrange(0, 900), rng.randrange(0, 300), rng.randrange(0, 400), rng.randrange(0, 3),
                            rng.randrange(0, 3)])
        msgs.append(obu.epoch_message(31))
    for m in msgs:
        assert tsp.receive(m) is not None
    honest = tsp.end_epoch()
    j = zero_weight_slot(world["w"])
    shift = [0] * 40
    shift[j] = 7                                                     # feature slot with weight 0: test "w_j = 0"
    accepted_probe = risk.fill(risk.evaluate(risk.cc.EvalAdd(msgs[0].ct_payload, risk.encrypt_vector(keys.pk, shift)),
                                             tsp.ct_w), keys.pk)
    bad = [0] * 40
    bad[SLOT_X] = 1                                                  # shifted score: rejected
    rejected_probe = risk.cc.EvalAdd(honest[1], risk.encrypt_vector(keys.pk, bad))
    SpyTSP.log = None
    return honest, accepted_probe, rejected_probe


def run(world, batch, monkeypatch):
    calls = []
    SpyTSP.log = calls
    real = hommac.verify_record

    def spy_verify(*a, **k):
        calls.append(("verify", None))
        return real(*a, **k)
    monkeypatch.setattr(ent.hommac, "verify_record", spy_verify)
    res = world["ins"].process_batch(batch)
    SpyTSP.log = None
    tsp_calls = [c for c in calls if c[0] != "verify"]
    assert len(calls) - len(tsp_calls) == len(batch)                 # every member was verified by the insurer
    return res, tsp_calls


def test_tsp_receives_no_verdict_only_one_bit_per_epoch(world, monkeypatch):
    honest, acc, rej = outputs(world, seed=8)
    # batch A: the probe's condition holds (accepted, correct score); batch B: it fails (rejected record)
    res_a, calls_a = run(world, [honest[0], honest[1], acc], monkeypatch)
    res_b, calls_b = run(world, [honest[0], rej, acc], monkeypatch)
    assert res_a.verdicts == ["accepted", "accepted", "accepted"]
    assert res_b.verdicts == ["accepted", REJECTED, "accepted"]
    # the TSP is asked nothing while the insurer processes the batch, whatever the verdicts
    assert calls_a == calls_b == []
    assert Insurer.deviation_notice(res_a) is False
    assert Insurer.deviation_notice(res_b) is True
    assert Insurer.deviation_notice(res_a, receipted_missing=1) is True     # a receipted record not accepted
    # an accepted probe carries the correct score: it is an identical duplicate of the honest record (no effect on the scores)
    rep = res_a.reports
    assert sum(r.identical_duplicates for r in rep.values()) == 1


def test_verdict_independent_requests_also_for_an_empty_and_an_all_rejected_batch(world, monkeypatch):
    honest, _, rej = outputs(world, seed=9)
    res, calls = run(world, [rej, honest[0]], monkeypatch)
    assert res.verdicts == [REJECTED, "accepted"] and calls == []
    SpyTSP.log = []
    empty = world["ins"].process_batch([])
    assert SpyTSP.log == [] and Insurer.deviation_notice(empty) is False
    SpyTSP.log = None
