"""Emission of the dedicated SP2UBI speed beacons: bytes per second from the code's message layouts, and the number of
PSSCH sub-channels each message needs at the highest MCS of a non-RSU station.

Messages per second of driving (sp2ubi/speedcheck.py): 10 beacons (BEACON_PERIOD_MS = 100), one burst certificate
(repeated every ROLL_PERIOD_MS) and one rolling trailer with 10 digests; the trailer also carries the smudging term
Enc(e_s) (one compressed ElGamal ciphertext in G1, 96 B) inside its signed content.

Sub-channels:
  * ETSI EN 303 798 V2.1.1 Table A.1.2: 10 MHz pool = 5 sub-channels of 10 PRB; Table A.1.4: maxMCS-PSSCH 11 for a
    non-RSU station below 160 km/h; Table A.1.5: maxMCS-PSSCH 8 at or above 160 km/h;
  * 3GPP TS 36.213 V14.4.0 Table 8.6.1-1: I_MCS 11 -> I_TBS 10, I_MCS 8 -> I_TBS 8; Table 7.1.7.2.1-1 rows I_TBS 10 and
    I_TBS 8: TBS for N_PRB = 8, 18, 28, 38, 48;
  * n sub-channels carry N_PRB = 10 n - 2 PSSCH PRB (2 PRB of the adjacent PSCCH).
Not modelled: MAC / RLC / PDCP and any lower-layer header bytes, packing of several messages into one transport block,
retransmissions; the sub-channel count is therefore a payload-level figure, and the margin to the TBS is reported.
Sizes only, no timing.

Run (from the repository root):  python experiments/sizes/speed_beacon_emission.py
Output: experiments/sizes/results/speed_beacon_emission.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from sp2ubi import speedcheck as sc  # noqa: E402
from sp2ubi.elgamal import ThresholdKey, joint_public_key  # noqa: E402
from sp2ubi.groupsig_bbs04 import BBS04, Issuer  # noqa: E402

# TS 36.213 V14.4.0 Table 7.1.7.2.1-1, bits; key: N_PRB. Row I_TBS 10 = I_MCS 11, row I_TBS 8 = I_MCS 8 (Table 8.6.1-1)
TBS = {11: {8: 1384, 18: 3112, 28: 4968, 38: 6712, 48: 8504},
       8: {8: 1096, 18: 2536, 28: 3880, 38: 5352, 48: 6712}}
SUBCHANNELS_10MHZ = 5
PRB_PER_SUBCHANNEL = 10
PSCCH_PRB = 2


def subchannels(nbytes, mcs=11):
    for n in range(1, SUBCHANNELS_10MHZ + 1):
        tbs = TBS[mcs][PRB_PER_SUBCHANNEL * n - PSCCH_PRB] // 8
        if nbytes <= tbs:
            return n, tbs - nbytes
    return None, None


def main():
    k1, k2 = ThresholdKey(34, 18, b"a"), ThresholdKey(34, 18, b"b")
    pk = joint_public_key(k1, k2)
    issuer = Issuer()
    gs = BBS04(issuer.gpk)
    obu = sc.OBU(gs, issuer.enrol(), pk, phase_ms=0)
    t = [1_000_000 + sc.BEACON_PERIOD_MS * i for i in range(40)]
    rec = obu.drive([200] * len(t), t)[0]
    beacon = rec.beacons[0].size()
    cert = rec.cert.size()
    per_s = 1000 // sc.BEACON_PERIOD_MS
    trailer = max(r.size() for r in rec.rollings)
    assert all(len(r.digests) == per_s for r in rec.rollings)
    ct = len(rec.rollings[0].ct_s.to_bytes())
    trailer_pre = trailer - ct
    rate = per_s * beacon + cert + trailer
    rate_pre = per_s * beacon + cert + trailer_pre
    msgs = {"beacon": beacon, "certificate": cert, "trailer": trailer, "trailer_without_smudging_term": trailer_pre}
    sub = {k: dict(zip(("subchannels", "spare_bytes_to_tbs"), subchannels(v))) for k, v in msgs.items()}
    sub8 = {k: dict(zip(("subchannels", "spare_bytes_to_tbs"), subchannels(v, 8))) for k, v in msgs.items()}
    occ = per_s * sub["beacon"]["subchannels"] + sub["certificate"]["subchannels"] + sub["trailer"]["subchannels"]
    res = {
        "message_bytes": msgs, "enc_e_s_bytes": ct,
        "bytes_per_second": {"total": rate, "without_smudging_term": rate_pre, "total_kB_per_s": round(rate / 1000, 1),
                             "without_smudging_term_kB_per_s": round(rate_pre / 1000, 1)},
        "subchannels_at_mcs11_10MHz": sub,
        "subchannels_at_mcs8_10MHz_at_or_above_160kmh": sub8,
        "trailer_subchannel_count_unchanged_by_smudging_term_mcs11": (
            sub["trailer"]["subchannels"] == sub["trailer_without_smudging_term"]["subchannels"]),
        "trailer_subchannel_count_unchanged_by_smudging_term_mcs8": (
            sub8["trailer"]["subchannels"] == sub8["trailer_without_smudging_term"]["subchannels"]),
        "subchannel_subframes_per_second": occ,
        "share_of_10MHz_pool_per_second_single_transmission": occ / (SUBCHANNELS_10MHZ * 1000),
        "share_of_10MHz_pool_per_second_one_blind_retransmission": 2 * occ / (SUBCHANNELS_10MHZ * 1000),
        "caveats": ["payload level: MAC/RLC/PDCP and lower-layer headers are not included",
                    "one transport block per message (no packing), PSSCH PRB = 10 n - 2 per n sub-channels",
                    "pool shares are given at MCS 11 (below 160 km/h); MCS 8 (at or above 160 km/h) is given per message",
                    "the pool shares are not the CR of TS 103 574 for a given PPPP; they are channel-use figures for the "
                    "congestion-control discussion"],
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "speed_beacon_emission.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
