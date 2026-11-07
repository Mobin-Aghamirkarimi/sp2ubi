"""Unit test of simpler_designs_sizes.py: the SP2UBI numbers equal the risk-path sizes JSON, the hybrid sizes follow the stated construction,
and the primitives' sizes agree with the implementation."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simpler_designs_sizes as S  # noqa: E402
from sp2ubi.groupsig_bbs04 import BBS04, G1, Issuer, rand_scalar  # noqa: E402


def load():
    with open(S.SINGLE_KEY_JSON) as f:
        return json.load(f)


def test_sp2ubi_matches_json():
    js, r = load(), S.compute(load())
    rec = js["per_record_B"]
    sp = r["designs"]["SP2UBI"]
    assert sp["upload_B"] == rec["upload_total"] == 263423
    assert sp["tsp_stored_per_record_B"] == rec["result_ciphertext_after_fill"]
    assert r["inputs"]["id_ins_B"] == 8


def test_hybrid_sizes():
    r = S.compute(load())
    a, b = r["designs"]["A"], r["designs"]["B"]
    assert a["plaintext_B"] == 4 + 6 + 5 + 32 and b["plaintext_B"] == 4 + (1 + 5 * 2) + 5 + 32
    assert r["inputs"]["x_coefficient_widths_B"] == [1, 2, 2, 2, 2, 2]
    assert a["ciphertext_B"] == 48 + a["plaintext_B"] + 16
    assert a["upload_B"] == a["ciphertext_B"] + 8 + 4 + 64 + 336
    assert b["upload_B"] - a["upload_B"] == b["plaintext_B"] - a["plaintext_B"]
    assert (a["upload_B"], b["upload_B"]) == (523, 528)
    assert a["tsp_storage_10e6_policies_B"] == a["tsp_stored_per_record_B"] * 31 * 10 ** 6


def test_widths_follow_the_code():
    """x widths from the daily caps of params.py (one daily record: interval_days = 1); the MAC is HMAC-SHA-256."""
    import hashlib
    import hmac
    from sp2ubi.params import DEFAULT as P
    assert P.interval_days == 1 and P.epoch_caps(1) == [1, 16383, 16383, 16383, 4095, 4095]
    assert S.MAC_BYTES == len(hmac.new(b"k" * 32, b"m", hashlib.sha256).digest()) == 32
    assert S.COMMITMENT_BYTES == hashlib.sha256().digest_size


def test_primitive_sizes_from_implementation():
    pt = G1() * rand_scalar()
    assert len(pt.to_compressed_bytes()) == S.POINT_BYTES == 48
    gi = Issuer()
    gs = BBS04(gi.gpk)
    sig = gs.sign(gi.enrol(), b"m")
    assert len(sig) == S.SIGNATURE_BYTES == 336


def test_ratios_are_computed_not_typed():
    r = S.compute(load())
    assert r["ratios_SP2UBI_over"]["A"]["upload_per_record"] == round(263423 / r["designs"]["A"]["upload_B"], 1)
