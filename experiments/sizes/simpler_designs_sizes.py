"""Byte sizes and operation counts of SP2UBI and two simpler designs, per daily record, per policy-epoch and for
the TSP's storage per 10^6 policies. Sizes and counts only, no timings.

  SP2UBI   record = BFV ciphertext (single insurer key, N = 8192, log2 Q = 120) + commitments + group signature;
           numbers read from results/risk_path_sizes.json (OpenFHE BINARY serialisation).
  Design A the OBU's TEE computes R = w.x, encrypts (ID, R, u, MAC) under the insurer's key with a hybrid scheme and
           group-signs the message; the TSP shuffles; the insurer decrypts and verifies the MAC. The model w sits in
           every TEE.
  Design B the TEE encrypts (ID, x, u, MAC) under the insurer's key with the same hybrid scheme and group-signs; the
           TSP shuffles; the insurer decrypts, verifies the MAC and computes R.
  Like for like with SP2UBI: inside the encryption, a MAC under the policy's key K (the key of SP2UBI's
  HomMAC, which the OBU already holds) over (ID, t, R, u) in A and (ID, t, x, u) in B authenticates the record to the
  insurer and binds it to the epoch t; the AEAD's associated data is (t, ID_Ins), known to the insurer per batch, so it
  is not transmitted; the message carries the same two 32-byte dispute commitments (c, c') and epoch label as
  SP2UBI's. MAC = HMAC-SHA-256 (32-byte tag; size taken from Python's hmac module below).

Hybrid encryption (stated construction, both designs): KEM = ephemeral Diffie-Hellman in G1 of BLS12-381 (one
compressed point, POINT_BYTES of the implementation's groupsig_bbs04.py) with a hash-based key derivation; DEM =
AES-256-GCM with a 16-byte tag; the 12-byte GCM nonce is derived from the KDF together with the key (the key is fresh
for every message), so it is not transmitted. Ciphertext = KEM point + plaintext + tag. The implementation does not
contain this construction; its sizes follow from the stated primitives and are computed here, not measured.

Field widths (fixed-width encoding, each field rounded up to whole bytes): ID 32 bits (entities.Insurer.enroll), nonce
u 40 bits (Params.nonce_bits), R reduced modulo the 41-bit prime p (Params.p); coefficient i of x of a daily record at
most its daily cap (Params.epoch_caps(interval_days): x_0 = d <= 1, three mileage features <= daily_mile_cap = 2^14 - 1,
two event counts <= daily_event_cap = 4095), i.e. 1 + 5 x 2 bytes.

Usage (from the repository root):  python experiments/sizes/simpler_designs_sizes.py
Output: experiments/sizes/results/simpler_designs_sizes.json"""
import hashlib
import hmac
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from sp2ubi.groupsig_bbs04 import POINT_BYTES, SIGNATURE_BYTES  # noqa: E402
from sp2ubi.params import DEFAULT as P  # noqa: E402

SINGLE_KEY_JSON = os.path.join(HERE, "results", "risk_path_sizes.json")
POLICIES = 10 ** 6
GCM_TAG_BYTES = 16          # AES-GCM authentication tag (NIST SP 800-38D, tag length 128 bits)
COMMITMENT_BYTES = 32       # SHA-256 digest (entities.EpochMessage.commitment, .binding)
EPOCH_BYTES = 4             # EpochMessage.signed_bytes: epoch.to_bytes(4)
MAC_BYTES = hmac.new(bytes(32), b"", hashlib.sha256).digest_size   # HMAC-SHA-256 tag under the policy's key K: 32

# Operation counts, read from the code:
#   BBS04 sign  (groupsig_bbs04.BBS04.sign): G1 scalar multiplications 12, one 2-term multi-pairing
#   BBS04 verify (groupsig_bbs04.BBS04.verify): G1 scalar multiplications 13, one 2-term multi-pairing
#   SP2UBI TSP evaluation (fhe.RiskContext.evaluate): 1 ciphertext-ciphertext product, len(ROTATIONS) = 3 rotations,
#     3 additions; fill (fhe.RiskContext.fill): 1 encryption, 1 addition.
BBS_SIGN = {"g1_scalar_mult": 12, "multi_pairing_2": 1}
BBS_VERIFY = {"g1_scalar_mult": 13, "multi_pairing_2": 1}


def nbytes(bits: int) -> int:
    return -(-bits // 8)


def hybrid_ct(plaintext_bytes: int) -> int:
    return POINT_BYTES + plaintext_bytes + GCM_TAG_BYTES


def compute(js: dict) -> dict:
    rec = js["per_record_B"]
    ct, signed, sig = rec["record_ciphertext"], rec["signed_record_message"], rec["group_signature"]
    upload, result = rec["upload_total"], rec["result_ciphertext_after_fill"]
    assert sig == SIGNATURE_BYTES and upload == signed + sig
    id_ins = signed - ct - 2 * COMMITMENT_BYTES - EPOCH_BYTES       # routing identifier (EpochMessage.id_ins)
    D = P.epoch_days // P.interval_days                             # records per policy-epoch, 31 daily records
    f_id, f_u, f_R = 4, nbytes(P.nonce_bits), nbytes(P.p.bit_length())
    f_x = [nbytes(c.bit_length()) for c in P.epoch_caps(P.interval_days)]   # daily caps, one width per coefficient
    pt_a = f_id + f_R + f_u + MAC_BYTES
    pt_b = f_id + sum(f_x) + f_u + MAC_BYTES
    out = {
        "inputs": {"records_per_policy_epoch": D, "policies": POLICIES, "single_key_json": os.path.relpath(
            SINGLE_KEY_JSON, ROOT), "N": js["context"]["N"], "log2_Q": js["context"]["log2_Q"],
            "id_ins_B": id_ins, "id_B": f_id, "nonce_B": f_u, "R_B": f_R, "x_coefficient_widths_B": f_x,
            "x_B": sum(f_x), "x_daily_caps": P.epoch_caps(P.interval_days), "x_coefficients": P.m,
            "mac_B": MAC_BYTES, "dispute_commitments_B": 2 * COMMITMENT_BYTES, "epoch_B": EPOCH_BYTES,
            "kem_point_B": POINT_BYTES, "aead_tag_B": GCM_TAG_BYTES, "group_signature_B": sig,
            "p_bits": P.p.bit_length(), "interval_days": P.interval_days},
        "hybrid_construction": "ephemeral DH in G1 of BLS12-381 (compressed point) + hash KDF + AES-256-GCM "
                               "(16-byte tag, nonce derived from the KDF, not transmitted); associated data (t, ID_Ins) "
                               "not transmitted; inside the plaintext an HMAC-SHA-256 tag under the policy's key K over "
                               "(ID, t, R, u) in A and (ID, t, x, u) in B",
        "tsp_storage_rule": "A and B: the signed message without its group signature (an upper bound: the insurer "
                            "needs only the ciphertext); SP2UBI: the result ciphertext after the fill",
    }
    designs = {}
    sp = {"plaintext_B": None, "ciphertext_B": ct, "signed_message_B": signed, "upload_B": upload,
          "tsp_stored_per_record_B": result, "tsp_to_insurer_per_record_B": result}
    for name, pt in (("A", pt_a), ("B", pt_b)):
        c = hybrid_ct(pt)
        msg = c + id_ins + EPOCH_BYTES + 2 * COMMITMENT_BYTES          # same envelope as SP2UBI's signed message
        designs[name] = {"plaintext_B": pt, "ciphertext_B": c, "signed_message_B": msg, "upload_B": msg + sig,
                         "tsp_stored_per_record_B": msg, "tsp_stored_with_signature_per_record_B": msg + sig,
                         "tsp_to_insurer_per_record_B": c}
    designs["SP2UBI"] = sp
    for d in designs.values():
        d["upload_per_policy_epoch_B"] = d["upload_B"] * D
        d["tsp_storage_per_policy_epoch_B"] = d["tsp_stored_per_record_B"] * D
        d["tsp_storage_10e6_policies_B"] = d["tsp_stored_per_record_B"] * D * POLICIES
        d["tsp_to_insurer_per_policy_epoch_B"] = d["tsp_to_insurer_per_record_B"] * D
        d["tsp_storage_10e6_policies_TB"] = round(d["tsp_storage_10e6_policies_B"] / 10 ** 12, 3)   # decimal TB
        d["tsp_storage_10e6_policies_GB"] = round(d["tsp_storage_10e6_policies_B"] / 10 ** 9, 3)
    out["designs"] = designs
    s = designs["SP2UBI"]
    out["ratios_SP2UBI_over"] = {
        k: {"upload_per_record": round(s["upload_B"] / designs[k]["upload_B"], 1),
            "tsp_storage": round(s["tsp_stored_per_record_B"] / designs[k]["tsp_stored_per_record_B"], 1)}
        for k in ("A", "B")}
    out["operations_per_daily_record"] = {
        "SP2UBI": {
            "OBU": {"bfv_encrypt": 1, "hommac_tag_blocks": 2, "sha256_commitments": 2, "bbs04_sign": 1},
            "TSP": {"bbs04_verify": 1, "bfv_ct_ct_product": 1, "bfv_rotation": 3, "bfv_add": 3 + 1,
                    "bfv_encrypt_fill": 1, "shuffle_entries": 1},
            "insurer": {"bfv_decrypt": 1, "hommac_verify_blocks": 2}},
        "A": {"OBU": {"dot_product_mults": P.m, "hmac_sha256": 1, "sha256_commitments": 2, "kem_g1_scalar_mult": 2,
                      "kdf": 1, "aead_encrypt": 1, "bbs04_sign": 1},
              "TSP": {"bbs04_verify": 1, "shuffle_entries": 1},
              "insurer": {"kem_g1_scalar_mult": 1, "kdf": 1, "aead_decrypt": 1, "hmac_sha256": 1}},
        "B": {"OBU": {"hmac_sha256": 1, "sha256_commitments": 2, "kem_g1_scalar_mult": 2, "kdf": 1, "aead_encrypt": 1,
                      "bbs04_sign": 1},
              "TSP": {"bbs04_verify": 1, "shuffle_entries": 1},
              "insurer": {"kem_g1_scalar_mult": 1, "kdf": 1, "aead_decrypt": 1, "hmac_sha256": 1,
                          "dot_product_mults": P.m}},
        "bbs04_sign_cost": BBS_SIGN, "bbs04_verify_cost": BBS_VERIFY,
    }
    return out


def main():
    with open(SINGLE_KEY_JSON) as f:
        js = json.load(f)
    res = compute(js)
    path = os.path.join(HERE, "results", "simpler_designs_sizes.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(json.dumps(res, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
