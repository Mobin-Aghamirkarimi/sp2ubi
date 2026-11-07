"""End-to-end simulation of one billing epoch on the risk path, honest or with an actively deviating TSP.

One encrypted record per policy and aggregation interval (daily by default) is uploaded at an independent random time
over the vehicle's cellular (Uu) link to the TSP, a fresh connection per attempt, with an outage model and re-sends
until the TSP's receipt or the settlement deadline; RSUs do not relay risk records. BFV under one insurer key, slot
fill at the TSP, decryption by the insurer.

Group signatures: BBS04 over BLS12-381 (groupsig_bbs04.py); the insurer is the issuer; the insurer and the TSP generate the
opening key without a dealer (opening_dkg.py) and an opening needs both shares (as in speedcheck.py).
Risk model: the integer additive Poisson model (risk_model.py). Daily features: simulated from So et al. (2021)
policies (workload.py) and summed inside the OBU.
"""
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .entities import OBU, TSP, BatchResult, Insurer
from .fhe import SLOT_ID, SLOT_U, SLOT_X, RiskContext
from .groupsig_bbs04 import (G1, BBS04, Issuer as GroupIssuer, Signature, combine_opening, member_key_valid,
                             rand_scalar)
from .opening_dkg import SessionRegistry, run_two_party
from .params import DEFAULT, Params
from .risk_model import load_model
from .workload import daily_records, iter_policies

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "telematics_syn-032021.csv")

# Deviations of a TSP holding the public key, the evaluation keys and Enc(w). Integrity deviations must be
# detected by the insurer (HomMAC, record counts); "replay" of an identical output must have no effect; "drop" makes a
# policy 'missing' (re-send, attributable through the OBU's receipt).
ATTACKS = ("substitute_model", "add_constant", "combine_records", "tamper_nonce", "tamper_id", "replay", "drop",
           "replay_and_drop")


class MaliciousTSP(TSP):
    def __init__(self, *args, attack: str, w_true: List[int], **kw):
        super().__init__(*args, **kw)
        if attack not in ATTACKS:
            raise ValueError(attack)
        self.attack = attack
        if attack == "substitute_model":            # TSP evaluates its own model, encrypted under the public key
            w_fake = list(w_true)
            w_fake[1] += 1
            self.ct_w = self.risk.encrypt_weights(self.keys.pk, w_fake)

    def _delta(self, slots: Dict[int, int]):
        v = [0] * 32
        for i, d in slots.items():
            v[i] = d
        return self.risk.encrypt_vector(self.keys.pk, v)

    def end_epoch(self):
        cc = self.risk.cc
        if self.attack == "combine_records":
            self.pool[0] = cc.EvalAdd(self.pool[0], self.pool.pop(1))
        elif self.attack == "add_constant":
            self.pool[0] = cc.EvalAdd(self.pool[0], self._delta({SLOT_X: -1000}))
        elif self.attack == "tamper_nonce":
            self.pool[0] = cc.EvalAdd(self.pool[0], self._delta({SLOT_U: 1}))
        elif self.attack == "tamper_id":
            self.pool[0] = cc.EvalAdd(self.pool[0], self._delta({SLOT_ID: 1}))
        elif self.attack == "replay":
            self.pool.append(self.pool[0])
        elif self.attack == "drop":
            self.pool.pop(0)
        elif self.attack == "replay_and_drop":          # keep the count: duplicate one record, drop another
            self.pool.append(self.pool[0])
            self.pool.pop(1)
        return super().end_epoch()


@dataclass
class EpochResult:
    batch: BatchResult
    detected: bool
    sizes_B: Dict[str, int] = field(default_factory=dict)
    receipts_ok: Optional[bool] = None
    confirmation_ok: Optional[bool] = None
    disputes_ok: Optional[bool] = None
    traced_ok: Optional[bool] = None
    outsider_rejected: Optional[bool] = None
    expected_claims: Optional[Dict[int, float]] = None
    plain_expected_claims: Optional[Dict[int, float]] = None
    used_dataset: Optional[bool] = None


def sample_policies(n: int, seed: int, path: str = DATASET):
    """n policies drawn uniformly (reservoir sampling) from the So et al. dataset, or None if it is absent."""
    if not os.path.exists(path):
        return None
    rng, chosen = random.Random(seed), []
    for i, row in enumerate(iter_policies(path)):
        if i < n:
            chosen.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < n:
                chosen[j] = row
    return chosen


Outages = Sequence[Tuple[float, float]]


def link_up(t: float, outages: Outages) -> bool:
    """Outage model of the vehicle's cellular link: down inside each [start, end) interval (seconds from the epoch
    start), up otherwise; an attempt made while the link is up reaches the TSP (TCP/TLS provides ARQ within it)."""
    return not any(a <= t < b for a, b in outages)


def uu_upload_loop(obu, deadline: float, outages: Outages, rng, send: Callable[[int, float], bool]) -> None:
    """Per-record Uu scheduler: take the earliest pending attempt of any record; stop once it lies
    after the settlement deadline; if the link is down at that time or `send(k, t)` returns no valid receipt, the
    record's next attempt is drawn independently (`OBU.retry`). `obu` needs only next_attempt() and retry()."""
    while True:
        nxt = obu.next_attempt()
        if nxt is None or nxt[0] > deadline:
            return
        t, k = nxt
        if link_up(t, outages) and send(k, t):
            continue
        obu.retry(k, t, rng)


def run_epoch(n_policies: int = 3, days: int = 31, attack: Optional[str] = None, params: Params = DEFAULT,
              seed: int = 7, epoch: int = 1, insured_days: Optional[List[int]] = None,
              outages: Outages = ()) -> EpochResult:
    """insured_days[k] (default: all `days`) models contract start/end inside the epoch (D < days). Every record is
    uploaded at its own drawn time over a fresh cellular connection; `outages` (seconds from the epoch start) are the
    intervals in which every vehicle's link is down; a record without a valid receipt is re-sent at an independently
    drawn later time until the settlement deadline (epoch end + W + grace), when the TSP closes the epoch."""
    rng = random.Random(seed)
    risk = RiskContext(params.p, params.m)
    model = load_model()
    insurer = Insurer(params, risk, model.w, epoch)
    keys = insurer.setup(epochs=(epoch, epoch + 11))
    tsp_sk = rand_scalar()                                 # TSP long-term signing key (receipts, opening-key setup)
    tsp_sessions = SessionRegistry(b"tsp")                 # the TSP's record of used session identifiers
    tracing, open_ins, open_tsp = run_two_party(           # opening key: insurer and TSP, no dealer; messages signed
        keys=((insurer._sigk, insurer.vk), (tsp_sk, G1() * tsp_sk)),   # with the two long-term signing keys, each
        registries=(insurer.dkg_sessions, tsp_sessions))   # party started through its own session registry
    group_issuer = GroupIssuer(tracing)
    registry = {}
    gs = BBS04(group_issuer.gpk)
    id_ins = b"INS-0001"
    tsp_cls = MaliciousTSP if attack else TSP
    kw = {"attack": attack, "w_true": insurer.w} if attack else {}
    tsp = tsp_cls(gs, risk, keys, insurer.vk, epoch, signing_key=tsp_sk, dkg_sessions=tsp_sessions, **kw)
    confirmation_ok = insurer.key_confirmation(tsp)
    tsp.close_setup()
    D = insured_days or [days] * n_policies
    obus = []
    for k in range(n_policies):
        ident, K = insurer.enroll(D[k])
        mk = group_issuer.enrol()
        assert member_key_valid(group_issuer.gpk, mk)
        registry[mk.A.to_compressed_bytes()] = str(ident)
        obus.append(OBU(ident, K, insurer.obu_keys, insurer.vk, risk, gs, mk, id_ins, params, epoch))

    policies = sample_policies(n_policies, seed)
    if policies is not None:
        logs = [daily_records(row, D[k], seed=seed + k) for k, row in enumerate(policies)]
    else:
        logs = [[[1] + [rng.randrange(0, 2000) for _ in range(3)] + [rng.randrange(0, 5) for _ in range(2)]
                 for _ in range(D[k])] for k in range(n_policies)]
    plain_expected = {obu.ident: model.expected_claims(sum(model.daily_score(x) for x in log))
                      for obu, log in zip(obus, logs)}
    sent, receipts_ok, sizes = {}, True, {}
    deadline = params.settlement_deadline_s(days)
    for k, (obu, log) in enumerate(zip(obus, logs)):
        for d, x in enumerate(log):
            obu.add_day(d, x[1:])
        obu.schedule(D[k], rng)

        def send(j, now, obu=obu):
            nonlocal sizes
            msg = obu.attempt(j)                       # fresh connection and fresh encryption per attempt
            sent[(obu.ident, j)] = msg
            if not sizes:
                sizes = {"ct_payload": risk.size(msg.ct_payload), "commitment": len(msg.commitment),
                         "group_signature": len(msg.signature)}
            return obu.acknowledge(j, msg, tsp.receive(msg), tsp.vk)
        uu_upload_loop(obu, deadline, outages, rng, send)
        receipts_ok &= not obu.unacknowledged()

    batch = tsp.end_epoch()
    if batch:
        sizes["ct_result"] = risk.size(batch[0])
    result = insurer.process_batch(batch)

    obu0 = obus[0]
    rec, salt = obu0.open_dispute(0)
    msg0 = sent.get((obu0.ident, 0)) or obu0.resend(0)          # never sent (outage until the deadline): the sealed one
    disputes_ok = Insurer.verify_dispute(msg0, rec, salt)
    rec_bad = dict(rec, x=[rec["x"][0], rec["x"][1] + 1] + rec["x"][2:])
    disputes_ok &= not Insurer.verify_dispute(msg0, rec_bad, salt)
    sig0 = Signature.from_bytes(msg0.signature)
    A0 = combine_opening(sig0.T3, open_ins.partial(sig0.T1, sig0.T2), open_tsp.partial(sig0.T1, sig0.T2))
    traced_ok = registry.get(A0.to_compressed_bytes()) == str(obu0.ident)
    outsider = GroupIssuer().enrol()
    forged = obu0.resend(0)
    forged.signature = gs.sign(outsider, forged.signed_bytes())
    r0 = tsp.rejected                                   # the epoch is closed now: check that the signature fails
    outsider_rejected = tsp.receive(forged) is None and tsp.rejected == r0 + 1

    detected = result.rejected > 0 or any(not r.accepted for r in result.reports.values())
    res = EpochResult(result, detected, sizes, receipts_ok, confirmation_ok, disputes_ok, traced_ok,
                      outsider_rejected)
    res.expected_claims = {i: model.expected_claims(r.score_sum) for i, r in result.reports.items()
                           if r.score_sum is not None}
    res.plain_expected_claims = plain_expected
    res.used_dataset = policies is not None
    return res
