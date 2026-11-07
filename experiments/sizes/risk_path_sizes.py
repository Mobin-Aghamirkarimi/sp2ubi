"""Single insurer key on the risk path: parameters chosen by OpenFHE, byte sizes of every key, ciphertext and message,
the decryption noise margin of a result ciphertext, and the message sizes of the dealer-free opening-key generation.
Sizes and parameters only; the timings are in experiments/timings.

Usage (from the repository root):  python experiments/sizes/risk_path_sizes.py
Output: experiments/sizes/results/risk_path_sizes.json"""
import json
import os
import secrets
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import openfhe as o  # noqa: E402

from sp2ubi import hommac  # noqa: E402
from sp2ubi.entities import OBU, TSP, Insurer  # noqa: E402
from sp2ubi.fhe import SLOT_X, RiskContext  # noqa: E402
from sp2ubi.groupsig_bbs04 import BBS04, SIGNATURE_BYTES, Issuer  # noqa: E402
from sp2ubi.opening_dkg import AuthenticatedParty, SessionRegistry, long_term_key  # noqa: E402
from sp2ubi.groupsig_bbs04 import GPK_BYTES  # noqa: E402
from sp2ubi.params import DEFAULT as P  # noqa: E402
from sp2ubi.proofs import proof_bytes  # noqa: E402
from sp2ubi.risk_model import load_model  # noqa: E402


def ser(x) -> int:
    return len(o.Serialize(x, o.BINARY))


def margin_bits(risk, sk, ct) -> int:
    """Largest e such that 2^e * ct still decrypts to 2^e * m (scaling method of the noise-margin test)."""
    ref = risk.decrypt(ct, sk, risk.N)

    def scaled(e):
        c = ct
        while e > 0:
            s = min(30, e)
            c = risk.cc.EvalMult(c, risk.cc.MakePackedPlaintext([1 << s] * risk.N))
            e -= s
        return c
    lo, hi = 0, 120
    while lo < hi:
        mid = (lo + hi + 1) // 2
        d = risk.decrypt(scaled(mid), sk, risk.N)
        if all(d[i] == (ref[i] << mid) % P.p for i in range(0, risk.N, 97)):
            lo = mid
        else:
            hi = mid - 1
    return lo


def main():
    w = load_model().w
    risk = RiskContext(P.p, P.m)
    ins = Insurer(P, risk, w, 1)
    bundle = ins.setup()
    tag = bundle.key_tag
    # opening key (dealer-free) and the OBU group
    (ska, vka), (skb, vkb) = long_term_key(), long_term_key()   # messages signed with the long-term keys
    a = AuthenticatedParty(SessionRegistry(b"insurer").start(b"sizes"), ska, vka, vkb)
    b = AuthenticatedParty(SessionRegistry(b"tsp").start(b"sizes"), skb, vkb, vka)
    c_a, c_b = a.commitment(), b.commitment()
    a.receive_commitment(c_b)
    b.receive_commitment(c_a)
    r_a, r_b = a.reveal(), b.reveal()
    a.receive_reveal(r_b)
    b.receive_reveal(r_a)
    h_a, h_b = a.h_share(), b.h_share()
    gs_key_tracing, share_a = a.finish(h_b)
    b.finish(h_a)
    gi = Issuer(gs_key_tracing)
    gs = BBS04(gi.gpk)
    tsp = TSP(gs, risk, bundle, ins.vk, 1)
    assert ins.key_confirmation(tsp)
    tsp.close_setup()
    ident, K = ins.enroll(31)
    obu = OBU(ident, K, ins.obu_keys, ins.vk, risk, gs, gi.enrol(), b"INS-0001", P, 1)
    for d in range(31):
        obu.add_day(d, [1200, 300, 0, 2, 1])
    msg = obu.record_message(0, 31)
    assert tsp.receive(msg) is not None
    out = tsp.end_epoch()[0]
    d = ins.decrypt(out)
    rec = obu.open_dispute(0)[0]
    assert d[SLOT_X] == sum(x * y for x, y in zip(w, rec["x"])) % P.p
    # result ciphertext at the feature caps (worst case for the noise)
    x = [31] + [31 * 16383] * 3 + [31 * 4095] * 2
    T, T2 = hommac.tag_blocks(x, secrets.token_bytes(32), 1, 9, P.p)
    worst = risk.fill(risk.evaluate(risk.encrypt_payload(bundle.pk, x, T, 9, 1, T2), bundle.ct_w), bundle.pk)
    G1B = 48
    pok = len(proof_bytes(r_a.body.pok_a))
    res = {
        "what": "single insurer BFV key on the risk path; sizes in bytes, OpenFHE BINARY serialization",
        "context": risk.info(),
        "security_note": "HEStd_128_classic: OpenFHE selects N and log2 Q from the HE-standard table for 128-bit "
                         "classical security (N = 8192 admits log2 Q <= 218)",
        "keys_B": {
            "public_key": ser(bundle.pk),
            "secret_key_insurer_only": ser(ins._sk),
            "relinearisation_key": len(o.SerializeEvalMultKeyString(o.BINARY, tag)),
            "rotation_keys_1_2_4": len(o.SerializeEvalAutomorphismKeyString(o.BINARY, tag)),
            "enc_w": ser(bundle.ct_w),
            "signed_bundle_to_tsp": ser(bundle.pk) + ser(bundle.ct_w) + len(o.SerializeEvalMultKeyString(o.BINARY, tag))
            + len(o.SerializeEvalAutomorphismKeyString(o.BINARY, tag)) + len(bundle.sig) + 4 + 8 + 32 + len(tag),
            # to the OBU: pk_Ins with its key tag, version (4 B) and epoch range (2 x 4 B) under the insurer's signature;
            # no Enc(w), no evaluation key, no digest of either (OBU key provisioning)
            "signed_public_key_to_obu": ser(ins.obu_keys.pk) + len(ins.obu_keys.sig) + 4 + 8 + len(ins.obu_keys.key_tag),
        },
        "per_record_B": {
            "record_ciphertext": ser(msg.ct_payload),
            "signed_record_message": len(msg.signed_bytes()),
            "group_signature": len(msg.signature),
            "upload_total": len(msg.signed_bytes()) + len(msg.signature),
            "result_ciphertext_after_fill": ser(out),
            "tsp_receipt": 64,
        },
        "decryption_margin_bits_result_at_caps": margin_bits(risk, ins._sk, worst),
        "opening_key_generation_B_per_party": {
            "commitment": len(c_a.body),
            "reveal_nonce_A_B_two_pok": len(r_a.body.nonce) + 2 * G1B + 2 * pok,
            "h_share_H_V_with_two_dleq": 2 * G1B + len(proof_bytes(h_a.body[2])) + len(proof_bytes(h_a.body[3])),
            "signature_per_message_long_term_key": len(c_a.sig),
            "messages_total_signed": len(c_a.body) + (len(r_a.body.nonce) + 2 * G1B + 2 * pok)
            + (2 * G1B + len(proof_bytes(h_a.body[2])) + len(proof_bytes(h_a.body[3]))) + 3 * len(c_a.sig),
            "share_secret_kept": 64, "share_commitments_public": 2 * G1B,
            "group_public_key": GPK_BYTES,
            "group_signature": SIGNATURE_BYTES,
        },
    }
    path = os.path.join(HERE, "results", "risk_path_sizes.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps(res, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
