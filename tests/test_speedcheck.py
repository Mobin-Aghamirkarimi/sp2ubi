"""Speed-verification path: correctness; two-party DKG with proofs of knowledge; DLEQ proofs on every
partial decryption and partial opening; public opening-setup check; TSP membership check; threshold split (no single party
decrypts or opens); non-overlapping certified bursts with rolling ephemeral signatures; one entry per crossing with
freshness, span and completeness checks; storage until after the epoch with release-time order, re-randomisation and
signing at release; count report independent of the entries; RSU-sourced encrypted count total; disclosure resolution;
decode window; one-term smudging (term drawn by the OBU inside the signed trailer; substituted term rejected; c
guard; worst-case c over the acceptance envelope; surveyed-direction rule); invalid messages ignored, never a reason to
discard a crossing; count at storage (a lost stored entry fails the count check)."""
import math
import random

import pytest

from sp2ubi.opening_dkg import run_two_party
from sp2ubi.elgamal import Ciphertext, ThresholdKey, dkg_accept_peer, dkg_commitment, joint_public_key
from sp2ubi.groupsig_bbs04 import BBS04, Issuer, combine_opening
from sp2ubi.proofs import prove_dlog, verify_opening_setup
from sp2ubi.speedcheck import (BEACON_PERIOD_MS, Beacon, BURST_MS, C_MAX, COUNT_REPORT_SPREAD_S, FORWARD_WINDOW_S,
                               FRESH_MS, K_MAX, MIN_COVER, N_LEVELS, S_BITS, SMUDGE_FRAC, SPAN_MIN, UNIT_MPS, V_MAX,
                               Z_CLIP, Z_STEP, OBU, BurstRecord, InsurerSpeedVerifier, Partials, Rolling, SensingRSU,
                               TSPKeyHolder, TSPShuffle, aggregate_counts, check_bounds, speed_to_units, verify_entry,
                               z_to_units)

SESSION = b"SP2UBI-test-key-epoch"


class NoSmudge(random.Random):
    """OBU rng for the exactness tests: every smudging term is `value` (0 by default)."""
    def __init__(self, value=0.0):
        super().__init__(1)
        self.value = value

    def gauss(self, mu=0.0, sigma=1.0):
        return self.value


@pytest.fixture(scope="module")
def world():
    check_bounds()
    k_ins, k_tsp = ThresholdKey(34, 18, b"insurer"), ThresholdKey(34, 18, b"tsp")
    c_ins, c_tsp = k_ins.commitment(SESSION), k_tsp.commitment(SESSION)          # 1. exchange commitments
    assert dkg_accept_peer(SESSION, b"tsp", c_tsp, k_tsp.share_pub, k_tsp.pok(SESSION))      # 2. reveal and check
    assert dkg_accept_peer(SESSION, b"insurer", c_ins, k_ins.share_pub, k_ins.pok(SESSION))
    pk = joint_public_key(k_ins, k_tsp)
    tracing, sh_ins, sh_tsp = run_two_party(SESSION)            # opening key: insurer and TSP, no dealer
    issuer = Issuer(tracing); gs = BBS04(issuer.gpk)
    a, b = issuer.enrol(), issuer.enrol()
    registry = {a.A.to_compressed_bytes(): "policy-a", b.A.to_compressed_bytes(): "policy-b"}
    rsu_issuer = Issuer(); rsu_gs = BBS04(rsu_issuer.gpk); rsu_key = rsu_issuer.enrol()
    return dict(pk=pk, gs=gs, a=a, b=b, rsu_gs=rsu_gs, rsu_key=rsu_key, issuer=issuer,
                sh_ins=sh_ins, sh_tsp=sh_tsp, k_ins=k_ins, k_tsp=k_tsp,
                ins=InsurerSpeedVerifier(k_ins, sh_ins, registry, (k_tsp.share_pub, sh_tsp.Y)),
                tsp=TSPKeyHolder(k_tsp, sh_tsp))


def _rsu(w, rng=None):
    return SensingRSU(w["gs"], w["pk"], w["rsu_gs"], w["rsu_key"], rng=rng)


def _drive(w, mk="a", n=200, v=20.0, bias=0.0, t0=1_000_000, phase=0, smudge=None):
    """n beacons at 10 Hz; returns (records, v_rsu per beacon time, v_sen). smudge=None: no smudging (exact tests);
    a number: every term equals it; "real": the OBU's CSPRNG."""
    t = [t0 + BEACON_PERIOD_MS * i for i in range(n)]
    v_rsu = [speed_to_units(v + 0.3 * (i % 5 - 2)) for i in range(n)]
    v_sen = [round(x * (1 + bias)) for x in v_rsu]
    rng = None if smudge == "real" else NoSmudge(smudge or 0.0)
    recs = OBU(w["gs"], w[mk], w["pk"], phase_ms=phase, rng=rng).drive(v_sen, t)
    return recs, dict(zip(t, v_rsu)), dict(zip(t, v_sen))


def _evaluate(w, rec, vmap, window, rsu=None, delay=1, level=1, z_units=0):
    """Accept and evaluate one crossing: the unsigned entry, or None."""
    rsu = rsu or _rsu(w)
    arr = [b.t_ms + delay for b in rec.beacons]
    ok, ct_s = rsu.accept(rec, arr, window)
    if not ok:
        return None
    return rsu.evaluate(ok, rec.cert, [vmap[b.t_ms] for b in ok], ct_s, level=level, z_units=z_units)


def _cross(w, rec, vmap, window, rsu=None, delay=1, level=1, z_units=0):
    """One crossing through the whole RSU path: evaluate, store, release after the window (signed entry), or None."""
    rsu = rsu or _rsu(w)
    e = _evaluate(w, rec, vmap, window, rsu=rsu, delay=delay, level=level, z_units=z_units)
    if e is None:
        return None
    rsu.store(e, epoch_end_s=0.0)
    out = rsu.release(FORWARD_WINDOW_S)
    assert len(out) == 1
    return out[0]


def _release(w, entries):
    """TSP shuffles the epoch's entries and publishes the output (the only entries it answers for)."""
    out = TSPShuffle(w["gs"], w["pk"]).shuffle(entries)
    w["tsp"].publish(out)
    return out


# ------------------------------------------------------------------------------------------------ keys and proofs
def test_threshold_elgamal_needs_both_shares(world):
    pk = world["pk"]
    ct = pk.encrypt(4242)
    ins, tsp = world["ins"], world["tsp"]
    assert ins.sv.combine(ct, tsp.sv.partial(ct)) == 4242
    assert ins.sv._bsgs._dlog(ct.C2 - ins.sv.partial(ct)) != 4242         # one share alone does not decrypt
    assert Ciphertext.from_bytes(ct.to_bytes()) == ct


def test_dkg_rejects_changed_or_unproven_share(world):
    w = world
    other = ThresholdKey(34, 18, b"tsp")
    c = w["k_tsp"].commitment(SESSION)
    assert not dkg_accept_peer(SESSION, b"tsp", c, other.share_pub, other.pok(SESSION))   # not the committed share
    rogue = other.share_pub - w["k_ins"].share_pub                       # chosen from the peer's share
    fake = prove_dlog(SESSION + b"|tsp", other.G, rogue, other._x)       # no witness for it
    assert not dkg_accept_peer(SESSION, b"tsp", dkg_commitment(SESSION, b"tsp", rogue, fake), rogue, fake)


def test_opening_setup_is_publicly_consistent(world):
    w = world
    assert verify_opening_setup(w["issuer"].gpk, w["sh_ins"].Y, w["sh_tsp"].Y)
    assert not verify_opening_setup(w["issuer"].gpk, w["sh_ins"].Y, w["sh_ins"].Y)


# ------------------------------------------------------------------------------------------------ end to end
def test_end_to_end_exact_with_threshold_opening(world):
    w = world
    recs, vmap, smap = _drive(w, bias=-0.10, n=100)
    rec = recs[0]
    window = (rec.beacons[10].t_ms, rec.beacons[90].t_ms)
    entry = _cross(w, rec, vmap, window, level=2, z_units=z_to_units(-1.6))
    assert entry is not None and verify_entry(w["rsu_gs"], entry)
    used = [b for b in rec.beacons if window[0] <= b.t_ms <= window[1]]
    vi = [vmap[b.t_ms] for b in used]; vs = [smap[b.t_ms] for b in used]
    den = sum(x * x for x in vi); c = round((1 << S_BITS) / den)
    expect = c * (sum(p * q for p, q in zip(vi, vs)) - den) / (1 << S_BITS)
    out = _release(w, [entry])[0]
    assert w["ins"].open_entry(out, w["tsp"].partials(out)) == ("policy-a", expect, 2, z_to_units(-1.6))


def test_weighted_statistic_with_site_bias(world):
    w = world
    recs, vmap, smap = _drive(w, bias=-0.07, n=60)
    rec = recs[0]
    rsu = _rsu(w)
    ok, ct_s = rsu.accept(rec, [b.t_ms for b in rec.beacons], (rec.beacons[0].t_ms, rec.beacons[-1].t_ms))
    weights = [round(64 * (1.2 + 0.7 * (i % 7) / 6.0) ** 2) for i in range(len(ok))]
    vi = [vmap[b.t_ms] for b in ok]; vs = [smap[b.t_ms] for b in ok]
    W = sum(p * q for p, q in zip(weights, vi))
    entry = rsu.evaluate_weighted(ok, rec.cert, weights, W, ct_s, site_bias=0.01)
    Wc = round(W * 1.01); c = round((1 << S_BITS) / Wc)
    expect = c * (sum(p * q for p, q in zip(weights, vs)) - Wc) / (1 << S_BITS)
    out = _release(w, [entry])[0]
    assert w["ins"].open_entry(out, w["tsp"].partials(out))[1] == expect


def test_shuffle_preserves_results_and_unlinks(world):
    w = world
    entries = []
    for mk, bias in (("a", 0.0), ("b", -0.2)):
        recs, vmap, _ = _drive(w, mk=mk, bias=bias, n=40)
        rec = recs[0]
        entries.append(_cross(w, rec, vmap, (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)))
    out = _release(w, entries)
    before = {e.B.to_bytes() for e in entries} | {e.T3.to_compressed_bytes() for e in entries}
    assert not any(e.B.to_bytes() in before or e.T3.to_compressed_bytes() in before for e in out)
    opened = [w["ins"].open_entry(e, w["tsp"].partials(e)) for e in out]
    assert sorted((p, round(b, 2)) for p, b, _, _ in opened) == [("policy-a", 0.0), ("policy-b", -0.2)]


# ------------------------------------------------------------------------------------------------ bursts and RSU checks
def test_bursts_are_consecutive_and_non_overlapping(world):
    w = world
    recs, _, _ = _drive(w, n=1000, phase=12_345)                    # 100 s of driving
    assert len(recs) >= 3
    for r1, r2 in zip(recs, recs[1:]):
        assert r2.cert.t0_ms == r1.cert.t0_ms + BURST_MS and r1.beacons[-1].t_ms < r2.beacons[0].t_ms
        assert r1.cert.burst != r2.cert.burst and r1.cert.E != r2.cert.E
    for r in recs:
        assert all(r.cert.t0_ms <= b.t_ms < r.cert.t0_ms + BURST_MS for b in r.beacons)
        assert sum(len(x.digests) for x in r.rollings) == len(r.beacons)


def test_rsu_rejects_stale_altered_forged_or_reused(world):
    w = world
    recs, vmap, _ = _drive(w, n=80)
    rec = recs[0]
    window = (rec.beacons[5].t_ms, rec.beacons[75].t_ms)
    arr = [b.t_ms + 1 for b in rec.beacons]
    NO = SensingRSU.NO_CROSSING
    assert _rsu(w).accept(rec, [b.t_ms + FRESH_MS + 1 for b in rec.beacons], window) == NO       # not fresh
    altered = BurstRecord(rec.cert, list(rec.beacons), rec.rollings)
    b3 = rec.beacons[30]
    altered.beacons[30] = type(b3)(b3.burst, b3.j, b3.t_ms, w["pk"].encrypt(1))
    ok = _rsu(w).accept(altered, arr, window)[0]                        # contradicts its signed digest: ignored,
    assert ok and 30 not in {b.j for b in ok} and len(ok) == 70          # beacon 30 counts as missing
    forged_roll = [Rolling(r.burst, r.j0, r.digests, r.ct_s, bytes(len(r.esig))) for r in rec.rollings]
    assert _rsu(w).accept(BurstRecord(rec.cert, rec.beacons, forged_roll), arr, window) == NO   # all trailers invalid:
                                                                                                # every beacon missing
    other_cert = _drive(w, n=20)[0][0].cert
    assert _rsu(w).accept(BurstRecord(other_cert, rec.beacons, rec.rollings), arr, window) == NO  # wrong certificate
    rsu = _rsu(w)
    assert rsu.accept(rec, arr, window)[0] and rsu.accept(rec, arr, window) == NO               # one entry per burst


def test_rsu_span_and_completeness(world):
    w = world
    recs, _, _ = _drive(w, n=100)
    rec = recs[0]
    window = (rec.beacons[10].t_ms, rec.beacons[90].t_ms)
    arr = [b.t_ms for b in rec.beacons]
    half = [t if i < 50 else None for i, t in enumerate(arr)]
    NO = SensingRSU.NO_CROSSING
    assert _rsu(w).accept(rec, half, window) == NO                       # beacons do not span the window
    holes = [t if i % 3 else None for i, t in enumerate(arr)]
    assert _rsu(w).accept(rec, holes, window) == NO                      # 67 % received < MIN_COVER
    few_holes = [t if i % 10 else None for i, t in enumerate(arr)]
    assert len(_rsu(w).accept(rec, few_holes, window)[0]) >= 70          # 90 % received: accepted
    unheard = rec.rollings[2:]                                           # first two trailers not heard
    assert _rsu(w).accept(rec, arr, (rec.beacons[0].t_ms, rec.beacons[90].t_ms), rollings_heard=unheard) == NO


def test_post_epoch_forwarding_and_count_total(world):
    w = world
    rsu = _rsu(w)
    epoch_end = 2_592_000.0                                              # end of the billing epoch (s)
    stored_B = []
    for k in range(3):
        recs, vmap, _ = _drive(w, n=30, t0=2_000_000 + 100_000 * k, smudge="real")
        rec = recs[0]
        e = _evaluate(w, rec, vmap, (rec.beacons[0].t_ms, rec.beacons[-1].t_ms), rsu=rsu)
        rsu.store(e, epoch_end_s=epoch_end)
        stored_B.append(e.B.to_bytes())
    st = rsu.stored()
    # the store keeps only (release time, unsigned entry), ordered by release time, all after the epoch end
    assert [t for t, _ in st] == sorted(t for t, _ in st)
    assert all(epoch_end <= t < epoch_end + FORWARD_WINDOW_S for t, _ in st)
    assert all(e.rsu_sig == b"" for _, e in st)
    assert rsu.release(epoch_end - 1) == []                              # nothing leaves before the epoch has ended
    with pytest.raises(RuntimeError):
        rsu.count_report(0)                                              # not before every entry has been released
    sent = rsu.release(epoch_end + FORWARD_WINDOW_S / 2)                 # some may be due early, none twice
    sent += rsu.release(epoch_end + FORWARD_WINDOW_S)
    assert len(sent) == 3 and rsu.release(epoch_end + 10 * FORWARD_WINDOW_S) == [] and rsu.stored() == []
    assert all(verify_entry(w["rsu_gs"], e) for e in sent)               # signed at release
    # re-randomised at release: a seized store image does not match the forwarded (published) entries
    assert not ({e.B.to_bytes() for e in sent} & set(stored_B))
    assert not ({e.T3.to_compressed_bytes() for e in sent} & {e.T3.to_compressed_bytes() for _, e in st})
    for _ in range(50):                                                  # count-report time: [24, 25) h after the epoch
        assert epoch_end + FORWARD_WINDOW_S <= rsu.count_report_time(epoch_end) < (
            epoch_end + FORWARD_WINDOW_S + COUNT_REPORT_SPREAD_S)
    assert FORWARD_WINDOW_S == 24 * 3600 and COUNT_REPORT_SPREAD_S == 3600
    reports = [rsu.count_report(0), _rsu(w).count_report(0)]           # one report per RSU, also for zero entries
    total = aggregate_counts(w["rsu_gs"], reports, epoch=0, n_rsus=2)
    w["tsp"].publish_total(total)
    D, pi = w["tsp"].partial_total(total)
    assert w["ins"].open_total(total, D, pi) == 3
    with pytest.raises(PermissionError):
        w["tsp"].partial_total(reports[0].count)                         # a single RSU's count is never decrypted
    assert aggregate_counts(w["rsu_gs"], reports[:1], epoch=0, n_rsus=2) is None          # a report dropped
    assert aggregate_counts(w["rsu_gs"], [reports[0]] * 2, epoch=0, n_rsus=2) is None     # a report duplicated
    assert aggregate_counts(w["rsu_gs"], reports, epoch=1, n_rsus=2) is None              # wrong epoch
    other = _rsu(w); other.epoch = 1
    assert aggregate_counts(w["rsu_gs"], [reports[0], other.count_report(1)], epoch=0, n_rsus=2) is None


def test_epoch_label_is_signed_and_checked(world):
    w = world
    rsu = _rsu(w); rsu.epoch = 7
    recs, vmap, _ = _drive(w, n=30, t0=3_000_000)
    rec = recs[0]
    e = _cross(w, rec, vmap, (rec.beacons[0].t_ms, rec.beacons[-1].t_ms), rsu=rsu)
    assert e.epoch == 7 and verify_entry(w["rsu_gs"], e, epoch=7) and not verify_entry(w["rsu_gs"], e, epoch=8)
    moved = type(e)(e.T1, e.T2, e.T3, e.B, e.level, e.z, epoch=8, rsu_sig=e.rsu_sig)
    assert not verify_entry(w["rsu_gs"], moved)                          # relabelling breaks the RSU signature
    out = TSPShuffle(w["gs"], w["pk"]).shuffle([e])[0]
    assert out.epoch == 7


def test_beacon_outside_its_certificate_interval_is_rejected(world):
    """Valid certificate and valid rolling signatures, but beacon times outside the certified 40 s interval."""
    w = world
    obu = OBU(w["gs"], w["a"], w["pk"], phase_ms=0)
    e, rec = obu._new_burst(1_000_000)                                  # certified interval [1 000 000, 1 040 000)
    t = [1_050_000 + BEACON_PERIOD_MS * i for i in range(30)]           # 10 s after the interval ends
    for i, ti in enumerate(t):
        rec.beacons.append(Beacon(rec.cert.burst, i, ti, w["pk"].encrypt(200)))
    obu._roll(e, rec, len(rec.beacons))
    assert w["gs"].verify(rec.cert.message(), rec.cert.sig)             # the certificate itself is valid
    assert _rsu(w).accept(rec, t, (t[0], t[-1])) == SensingRSU.NO_CROSSING
    good_e, good = obu._new_burst(1_050_000 - 5_000)                    # same beacons under a covering certificate
    for i, ti in enumerate(t):
        good.beacons.append(Beacon(good.cert.burst, i, ti, w["pk"].encrypt(200)))
    obu._roll(good_e, good, len(good.beacons))
    assert len(_rsu(w).accept(good, t, (t[0], t[-1]))[0]) == 30


# ------------------------------------------------------------------------------------------------ membership check, proofs, window
def _one_entry(w, bias=-0.05, mk="a"):
    recs, vmap, _ = _drive(w, mk=mk, bias=bias, n=40)
    rec = recs[0]
    return _cross(w, rec, vmap, (rec.beacons[0].t_ms, rec.beacons[-1].t_ms))


def test_wrong_tsp_partial_is_detected_not_decoded(world):
    w = world
    out = _release(w, [_one_entry(w)])[0]
    good = w["tsp"].partials(out)
    shifted = Partials(good.open_parts, good.B + w["pk"].G, good.level, good.z, good.proofs)
    with pytest.raises(ValueError):
        w["ins"].open_entry(out, shifted)
    wrong_open = Partials((good.open_parts[0] + w["pk"].G, good.open_parts[1]), good.B, good.level, good.z, good.proofs)
    with pytest.raises(ValueError):
        w["ins"].open_entry(out, wrong_open)
    assert w["ins"].open_entry(out, good)[0] == "policy-a"


def test_tsp_gate_answers_only_published_entries(world):
    w = world
    e = _one_entry(w)
    with pytest.raises(PermissionError):
        w["tsp"].partials(e)                                    # not a member of a published shuffle output
    out = _release(w, [e])[0]
    assert w["ins"].open_entry(out, w["tsp"].partials(out))[0] == "policy-a"


def test_single_opening_share_does_not_open(world):
    w = world
    e = _one_entry(w)
    A_one = combine_opening(e.T3, w["sh_ins"].partial(e.T1, e.T2), type(w["pk"].G).identity())
    assert A_one != w["a"].A
    A_both = combine_opening(e.T3, w["sh_ins"].partial(e.T1, e.T2), w["sh_tsp"].partial(e.T1, e.T2))
    assert A_both == w["a"].A


def test_out_of_window_result_is_none(world):
    w = world
    out = _release(w, [_one_entry(w, bias=-0.30, mk="b")])[0]
    policy, b_hat, _, _ = w["ins"].open_entry(out, w["tsp"].partials(out))
    assert policy == "policy-b" and b_hat is None


def test_disclosure_resolution_of_level_and_z(world):
    """Only the level (N_LEVELS values) and the clipped, quantised z can leave the RSU."""
    zmax = round(Z_CLIP / Z_STEP)
    assert [z_to_units(x) for x in (-9.0, 0.0, 7.5)] == [-zmax, 0, zmax]
    w = world
    recs, vmap, _ = _drive(w, n=30)
    rec = recs[0]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    assert _cross(w, rec, vmap, window, level=N_LEVELS) is None
    assert _cross(w, rec, vmap, window, z_units=zmax + 1) is None
    assert _cross(w, rec, vmap, window, level=N_LEVELS - 1, z_units=-zmax) is not None


def test_sizes(world):
    w = world
    recs, _, _ = _drive(w, n=40)
    rec = recs[0]
    assert rec.beacons[0].size() == 16 + 2 + 8 + 96
    assert rec.cert.size() == 16 + 48 + 8 + 336
    assert rec.rollings[0].size() == 16 + 4 + 16 * len(rec.rollings[0].digests) + 96 + 64      # + Enc(e_s)
    assert max(r.size() for r in rec.rollings) == 340                    # 10 digests per second


# ------------------------------------------------------------------------------------------------ one-term smudging
def test_smudging_term_is_signed_and_substitution_rejected(world):
    """The trailer's Enc(e_s) is inside the signed content: a substituted term fails verification of the trailer, so
    the crossing yields no entry (an unsigned term could be replaced by Enc(+0.1 * 2^S), cancelling a -10 % VSS)."""
    w = world
    recs, vmap, _ = _drive(w, n=60)
    rec = recs[0]
    arr = [b.t_ms + 1 for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    r0 = rec.rollings[0]
    big = w["pk"].encrypt(round(0.1 * (1 << S_BITS)))
    assert Rolling(r0.burst, r0.j0, r0.digests, big, r0.esig).message() != r0.message()
    swapped = [Rolling(r.burst, r.j0, r.digests, big, r.esig) for r in rec.rollings]
    assert _rsu(w).accept(BurstRecord(rec.cert, rec.beacons, swapped), arr, window) == SensingRSU.NO_CROSSING
    one = [Rolling(r0.burst, r0.j0, r0.digests, w["pk"].encrypt(10 ** 9), r0.esig)] + list(rec.rollings[1:])
    assert _rsu(w).accept(BurstRecord(rec.cert, rec.beacons, one), arr, window) == SensingRSU.NO_CROSSING
    ok, ct_s = _rsu(w).accept(rec, arr, window)
    assert ok and ct_s == rec.rollings[0].ct_s                          # the term of the trailer covering beacon 0


def test_term_from_trailer_covering_first_used_beacon(world):
    """Exact decryption B = c * Delta + e_s with the OBU's term (all terms set to a known value), taken from the trailer
    that covers the first used beacon; the RSU adds no term of its own."""
    w = world
    e_s = 123_457
    recs, vmap, smap = _drive(w, bias=-0.04, n=100, smudge=float(e_s))
    rec = recs[0]
    window = (rec.beacons[15].t_ms, rec.beacons[95].t_ms)               # first used beacon in the second trailer
    ok, ct_s = _rsu(w).accept(rec, [b.t_ms + 1 for b in rec.beacons], window)
    assert ok[0].j == 15 and ct_s == next(r for r in rec.rollings if r.covers(15)).ct_s
    entry = _cross(w, rec, vmap, window, level=3)
    used = [b for b in rec.beacons if window[0] <= b.t_ms <= window[1]]
    vi = [vmap[b.t_ms] for b in used]; vs = [smap[b.t_ms] for b in used]
    den = sum(x * x for x in vi); c = round((1 << S_BITS) / den)
    expect = (c * (sum(p * q for p, q in zip(vi, vs)) - den) + e_s) / (1 << S_BITS)
    out = _release(w, [entry])[0]
    assert w["ins"].open_entry(out, w["tsp"].partials(out))[:3] == ("policy-a", expect, 3)


def test_real_smudging_stays_within_eight_sigma(world):
    w = world
    recs, vmap, smap = _drive(w, bias=-0.06, n=60, smudge="real")
    rec = recs[0]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    entry = _cross(w, rec, vmap, window)
    vi = [vmap[b.t_ms] for b in rec.beacons]; vs = [smap[b.t_ms] for b in rec.beacons]
    den = sum(x * x for x in vi); c = round((1 << S_BITS) / den)
    exact = c * (sum(p * q for p, q in zip(vi, vs)) - den) / (1 << S_BITS)
    out = _release(w, [entry])[0]
    b = w["ins"].open_entry(out, w["tsp"].partials(out))[1]
    assert b != exact and abs(b - exact) <= 8 * SMUDGE_FRAC


def test_c_guard(world):
    """No entry when c > C_MAX (a tiny W'), so B mod c is uniform for every entry the code can emit."""
    w = world
    check_bounds()
    assert 16 * C_MAX <= SMUDGE_FRAC * (1 << S_BITS)
    recs, vmap, _ = _drive(w, n=30)
    rec = recs[0]
    rsu = _rsu(w)
    ok, ct_s = rsu.accept(rec, [b.t_ms for b in rec.beacons], (rec.beacons[0].t_ms, rec.beacons[-1].t_ms))
    W_small = round((1 << S_BITS) / (C_MAX + 1)) - 1                     # c just above C_MAX
    assert round((1 << S_BITS) / W_small) > C_MAX
    assert rsu.evaluate_weighted(ok, rec.cert, [1] * len(ok), W_small, ct_s) is None
    W_ok = round((1 << S_BITS) / C_MAX) + 1
    assert rsu.evaluate_weighted(ok, rec.cert, [1] * len(ok), W_ok, ct_s) is not None
    assert rsu.evaluate_weighted(ok, rec.cert, [1] * len(ok), W_ok, None) is None   # no trailer term, no entry


def _envelope_c_max(site_bias=0.0):
    """Largest c over the constant-speed crossings the RSU's code path accepts (window [-0.05L, 1.05L], L = 100 m,
    10 Hz; the used beacons span >= SPAN_MIN of the window inside one burst, so v >= SPAN_MIN * 110 m / 39.9 s =
    2.48 m/s, and >= MIN_COVER of the spanned index range is kept; the kept beacons chosen adversarially with the
    smallest weights; rho 1-50 m; W' = W (1 + site_bias)). Same model as experiments/sizes/
    smudging_scale_range.py (rule "code")."""
    L = 100.0
    lo, hi = -0.05 * L, 1.05 * L
    dt = BEACON_PERIOD_MS / 1000
    v_min = SPAN_MIN * (hi - lo) / ((BURST_MS - BEACON_PERIOD_MS) / 1000)
    speeds = [v_min] + [(u + 0.5) * UNIT_MPS - 1e-6 for u in range(25, 50)] + [5, 7.5, 10, 15, 20, 30, 38, 50,
                                                                              V_MAX * UNIT_MPS]
    c_max = 0
    for v in speeds:
        t_w = (hi - lo) / v
        n = int(math.floor(t_w / dt + 1e-9)) + 1
        m = math.ceil(SPAN_MIN * t_w / dt - 1e-9) + 1
        if m > BURST_MS // BEACON_PERIOD_MS:
            continue                                                    # cannot span 90 % of the window in one burst
        keep = min(K_MAX, max(3, math.ceil(MIN_COVER * m - 1e-9)))
        xs = [lo + i * v * dt for i in range(n)]
        vu = speed_to_units(v)
        for rho in (1, 2, 3, 5, 10, 20, 30, 50):
            g2 = sorted(round(64 * (x / math.hypot(x, rho) + (L - x) / math.hypot(L - x, rho)) ** 2) for x in xs)
            for W in (keep * vu * vu, sum(g2[:keep]) * vu):                # profile, matched
                c_max = max(c_max, round((1 << S_BITS) / round(W * (1 + site_bias))))
    return c_max


def test_worst_case_c_over_acceptance_envelope():
    sigma = SMUDGE_FRAC * (1 << S_BITS)
    assert sigma / C_MAX >= 16                                          # the guard: the paper's claim rests on it
    assert _envelope_c_max(0.0) == 88_102                               # slow_sweep value (2.546 m/s)
    for bias in (0.0, -0.03, -0.06, 0.06):
        c_max = _envelope_c_max(bias)
        assert c_max <= C_MAX and sigma / c_max > 180                   # the paper's "more than 180 times"


def test_surveyed_direction_only(world):
    w = world
    recs, vmap, _ = _drive(w, n=40)
    rec = recs[0]
    arr = [b.t_ms for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    assert _rsu(w).accept(rec, arr, window, direction=-1) == SensingRSU.NO_CROSSING
    rsu = SensingRSU(w["gs"], w["pk"], w["rsu_gs"], w["rsu_key"], surveyed_direction=-1)
    assert rsu.accept(rec, arr, window, direction=-1)[0]


def test_rsu_rerandomises_enc_b(world):
    """Enc(B) is a public linear combination of broadcast ciphertexts; the RSU's output is not that combination
    (re-randomisation is load-bearing against a recorder colluding with an RSU that knows c and w_i)."""
    w = world
    recs, vmap, _ = _drive(w, n=40, smudge="real")
    rec = recs[0]
    rsu = _rsu(w)
    ok, ct_s = rsu.accept(rec, [b.t_ms for b in rec.beacons], (rec.beacons[0].t_ms, rec.beacons[-1].t_ms))
    weights = [vmap[b.t_ms] for b in ok]
    W = sum(x * x for x in weights); c = round((1 << S_BITS) / W)
    from sp2ubi.elgamal import Ciphertext, _scalar
    acc = Ciphertext(type(w["pk"].G).identity(), type(w["pk"].G).identity())
    for b, x in zip(ok, weights):
        acc = acc + b.ct.mul(c * x)
    acc = acc + ct_s
    det = Ciphertext(acc.C1, acc.C2 - w["pk"].G * _scalar(c * W))
    e = rsu.evaluate(ok, rec.cert, weights, ct_s)
    assert e.B.to_bytes() != det.to_bytes()


# ------------------------------------------------------------------------------------------------ invalid messages ignored
def _burst_with_key(w, n=60, t0=4_000_000):
    """A burst built through the OBU with its ephemeral secret e kept (to sign conflicting trailers); terms are 0."""
    obu = OBU(w["gs"], w["a"], w["pk"], phase_ms=0, rng=NoSmudge())
    e, rec = obu._new_burst(t0)
    for i in range(n):
        rec.beacons.append(Beacon(rec.cert.burst, i, t0 + BEACON_PERIOD_MS * i, w["pk"].encrypt(200)))
        if (i + 1) % 10 == 0:
            obu._roll(e, rec, i + 1)
    return e, rec


def test_injected_junk_trailer_is_ignored(world):
    """A trailer with the burst nonce and a garbage signature (sent by the driver or anyone in range) does not make the
    RSU discard the crossing: same beacons, same term."""
    w = world
    recs, vmap, _ = _drive(w, n=60)
    rec = recs[0]
    arr = [b.t_ms + 1 for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    ref, ref_term = _rsu(w).accept(rec, arr, window)
    junk = Rolling(rec.cert.burst, 300, tuple(bytes(16) for _ in range(10)), w["pk"].encrypt(5), bytes(64))
    junk2 = Rolling(rec.cert.burst, 0, tuple(bytes(16) for _ in range(10)), w["pk"].encrypt(5), b"\x01" * 64)
    junk3 = Rolling(rec.cert.burst, 10, rec.rollings[1].digests, w["pk"].encrypt(5), b"\xff" * 64)   # non-canonical
    junk4 = Rolling(rec.cert.burst, 20, rec.rollings[2].digests, rec.rollings[2].ct_s, b"\x00" * 7)  # wrong length
    for heard in ([junk] + list(rec.rollings), list(rec.rollings) + [junk2, junk] * 3,
                  [junk3, junk4, junk2] + list(rec.rollings)):
        ok, term = _rsu(w).accept(rec, arr, window, rollings_heard=heard)
        assert [b.j for b in ok] == [b.j for b in ref] and term == ref_term


def test_foreign_burst_trailer_is_ignored(world):
    """A genuine trailer of another vehicle's burst (mixed traffic) leaves the entry unchanged."""
    w = world
    recs, vmap, _ = _drive(w, n=60)
    rec = recs[0]
    other = _drive(w, mk="b", n=30, t0=1_000_000)[0][0]
    arr = [b.t_ms + 1 for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    ref, ref_term = _rsu(w).accept(rec, arr, window)
    ok, term = _rsu(w).accept(rec, arr, window, rollings_heard=list(other.rollings) + list(rec.rollings))
    assert [b.j for b in ok] == [b.j for b in ref] and term == ref_term and len(ok) == 60


def test_contradicting_and_stale_copies_are_ignored(world):
    """A copy that contradicts its signed digest, a stale (delayed) copy of a genuine beacon and a foreign beacon, each
    next to the fresh genuine copy, are ignored: same entry."""
    w = world
    recs, vmap, _ = _drive(w, n=60)
    rec = recs[0]
    arr = [b.t_ms + 1 for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    ref, ref_term = _rsu(w).accept(rec, arr, window)
    b30 = rec.beacons[30]
    altered = Beacon(b30.burst, b30.j, b30.t_ms, w["pk"].encrypt(1))
    foreign = _drive(w, mk="b", n=40, t0=rec.beacons[0].t_ms)[0][0].beacons[30]
    for copies in ([(altered, b30.t_ms + 1)], [(b30, b30.t_ms + 5)], [(rec.beacons[j], rec.beacons[j].t_ms + 7)
                                                                      for j in range(60)], [(foreign, foreign.t_ms)]):
        ok, term = _rsu(w).accept(rec, arr, window, copies=copies)
        assert [b.j for b in ok] == [b.j for b in ref] and term == ref_term
    # copies alone do not create an entry: a burst received only in stale copies still yields none (coverage)
    stale = [b.t_ms + FRESH_MS + 1 for b in rec.beacons]
    assert _rsu(w).accept(rec, stale, window, copies=[(b, b.t_ms + 9) for b in rec.beacons]) == SensingRSU.NO_CROSSING


def test_earliest_fresh_copy_is_used(world):
    """Of several fresh copies of a beacon the earliest is used: a 1-ms re-sent copy of every beacon, a Rel-14 blind
    retransmission 1-2 subframes later and a later copy of one beacon leave the entry unchanged."""
    w = world
    recs, vmap, _ = _drive(w, n=60)
    rec = recs[0]
    arr = [b.t_ms + 1 for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    ref, ref_term = _rsu(w).accept(rec, arr, window)
    assert len(ref) == 60
    b30 = rec.beacons[30]
    variants = [
        [(b30, b30.t_ms + 2)],                                           # a later fresh copy of one beacon
        [(b, b.t_ms + 2) for b in rec.beacons],                          # 1-ms re-sent copy of every beacon
        [(b, b.t_ms + 1 + g) for b in rec.beacons for g in (1, 2)],      # blind retransmission 1 and 2 subframes later
        [(b, b.t_ms) for b in rec.beacons[5:55]],                        # an earlier-arriving copy of every middle beacon
    ]
    for copies in variants:
        ok, term = _rsu(w).accept(rec, arr, window, copies=copies)
        assert [b.j for b in ok] == [b.j for b in ref] and term == ref_term and len(ok) == 60
    # the copy that arrives first is the one selected
    early = (b30, b30.t_ms - 1)
    ok, _ = _rsu(w).accept(rec, [None if i == 30 else a for i, a in enumerate(arr)], window, copies=[early, (b30, b30.t_ms + 2)])
    assert len(ok) == 60


def test_two_verified_trailers_that_differ_reject(world):
    """Two trailers that both verify under the burst's ephemeral key but differ for one beacon (digest or term) can
    only come from the TEE (a fault): the crossing is rejected."""
    from sp2ubi.proofs import proof_bytes, schnorr_sign
    w = world
    e, rec = _burst_with_key(w)
    arr = [b.t_ms + 1 for b in rec.beacons]
    window = (rec.beacons[0].t_ms, rec.beacons[-1].t_ms)
    assert len(_rsu(w).accept(rec, arr, window)[0]) == 60
    r1 = rec.rollings[1]

    def signed(digests, ct_s):
        r = Rolling(r1.burst, r1.j0, digests, ct_s, b"")
        return Rolling(r.burst, r.j0, digests, ct_s, proof_bytes(schnorr_sign(e, rec.cert.E, r.message())))
    other_digest = signed((bytes(16),) + r1.digests[1:], r1.ct_s)
    other_term = signed(r1.digests, w["pk"].encrypt(7))
    for extra in (other_digest, other_term):
        assert _rsu(w).accept(rec, arr, window, rollings_heard=list(rec.rollings) + [extra]) == SensingRSU.NO_CROSSING
    assert len(_rsu(w).accept(rec, arr, window, rollings_heard=list(rec.rollings) + [r1])[0]) == 60   # replay: same


def test_lost_stored_entry_fails_the_count_check(world):
    """The count is taken when an entry is formed and stored, not at release: an entry lost from storage before its
    release is still counted, so the epoch's opened count total exceeds the published entries."""
    w = world
    rsu = _rsu(w); rsu.epoch = 5
    epoch_end = 9_000_000.0
    for k in range(3):
        recs, vmap, _ = _drive(w, n=30, t0=5_000_000 + 100_000 * k, smudge="real")
        rec = recs[0]
        rsu.store(_evaluate(w, rec, vmap, (rec.beacons[0].t_ms, rec.beacons[-1].t_ms), rsu=rsu), epoch_end)
    rsu._store.pop()                                                     # one stored entry lost before its release
    sent = rsu.release(epoch_end + FORWARD_WINDOW_S)
    assert len(sent) == 2
    other = _rsu(w); other.epoch = 5
    total = aggregate_counts(w["rsu_gs"], [rsu.count_report(5), other.count_report(5)], epoch=5, n_rsus=2)
    w["tsp"].publish_total(total)
    assert w["ins"].open_total(total, *w["tsp"].partial_total(total)) == 3 != len(sent)   # the count check fails
