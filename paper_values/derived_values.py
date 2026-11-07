"""Every derived size, storage, channel and timing value of the evaluation, computed from the stored result files.

Inputs (none edited here):
  experiments/sizes/results/risk_path_sizes.json         OpenFHE object sizes and context of the risk path
  experiments/sizes/results/simpler_designs_sizes.json   per policy-month, fleet storage, simpler designs A and B
  experiments/sizes/results/settlement_sizes.json        OBU dispute state and settlement traffic
  experiments/sizes/results/baseline_params.json         CKKS parameters of baseline B3
  experiments/sizes/results/speed_beacon_emission.json   speed-beacon bytes per second
  experiments/timings/results/*.json                     timing items (sizes, context and timings)
Channel times are bytes divided by the peak uplink rate of a UE category (ETSI TS 136 306 V14.4.0, Table 4.1-2:
Category 1, 5160 UL-SCH bits per 1-ms TTI; Table 4.1A-2: UL Category M1, 1000 bits, or 2984 bits with
ce-PUSCH-NB-MaxTBS-r14); they are lower bounds, rounded down to 0.01 s.

Timing values are the medians over all timed repetitions after the discarded warm-up runs, with the interquartile
range (q1, q3) and the number of repetitions n; "sum" values add the medians of their parts, "derived" values are the
stated arithmetic on other values. A timing item is used only if its file says result_status "final".

Output: paper_values/paper_values.csv with the columns key, value, printed, unit, q1, q3, n, source, description;
`printed` is the value with the rounding the paper uses. The script exits with status 1 if a consistency check of
the inputs fails.

Usage (from the repository root):  python paper_values/derived_values.py [--out paper_values/paper_values.csv]
"""
import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIZES = os.path.join(ROOT, "experiments", "sizes", "results")
TIMINGS = os.path.join(ROOT, "experiments", "timings", "results")
ITEMS = ("speed_path", "risk_path_budget", "baselines_bfv_ckks", "baseline_paillier", "throughput")
RATE_BITS_PER_TTI = {"cat1": 5160, "m1": 1000, "m1_tbs2984": 2984}
TTI_S = 1e-3
POLICIES = 10 ** 6
DAYS = 31


def load_json(*parts):
    with open(os.path.join(*parts)) as f:
        return json.load(f)


def floor2(x):
    return math.floor(x * 100) / 100


def group(s):
    """Thousands separator for the integer part of a decimal string."""
    neg = s.startswith("-")
    ip, _, fp = s.lstrip("-").partition(".")
    out = ""
    while len(ip) > 3:
        out = "," + ip[-3:] + out
        ip = ip[:-3]
    return ("-" if neg else "") + ip + out + ("." + fp if fp else "")


def ms_digits(v):
    """Display precision of a time in ms: 3 decimals below 1 ms, 2 below 100 ms, 1 above."""
    return 3 if abs(v) < 1 else (2 if abs(v) < 100 else 1)


def printed(v, cls):
    if cls == "int":
        return group(str(int(round(v))))
    if cls == "plain":
        return str(int(v))
    if cls in ("dec1", "dec2"):
        return group(f"{v:.{int(cls[3:])}f}")
    if cls == "ms":
        return group(f"{v:.{ms_digits(v)}f}")
    if cls == "rps":
        return group(f"{v:.1f}")
    if cls == "ratio":
        return f"{v:.1f}"
    if cls == "ratio2":
        return f"{v:.2f}"
    if cls == "sci":
        e = int(f"{v:e}".split("e")[1])
        return f"{v / 10 ** e:.1f}e{e}"
    raise ValueError(cls)


# ---------------------------------------------------------------------------------------------------- sizes
def size_rows(errors):
    def check(cond, msg):
        if not cond:
            errors.append(msg)

    SK = load_json(SIZES, "risk_path_sizes.json")
    CI = load_json(SIZES, "simpler_designs_sizes.json")
    SET = load_json(SIZES, "settlement_sizes.json")
    BP = load_json(SIZES, "baseline_params.json")
    EMIT = load_json(SIZES, "speed_beacon_emission.json")["bytes_per_second"]
    budget = load_json(TIMINGS, "risk_path_budget.json")["result"]
    base = load_json(TIMINGS, "baselines_bfv_ckks.json")["result"]["baselines"]
    pail = load_json(TIMINGS, "baseline_paillier.json")["result"]["baselines"]
    speed = load_json(TIMINGS, "speed_path.json")["result"]["sizes_K40"]
    ctx, k, r = SK["context"], SK["keys_B"], SK["per_record_B"]
    ci, sp = CI["inputs"], CI["designs"]["SP2UBI"]
    day, mon = SET["configurations"]["daily"], SET["configurations"]["monthly"]

    # consistency of the recorded inputs
    for f in ("N", "log2_Q", "moduli_bits", "t_bits", "security", "depth", "keys", "openfhe"):
        check(budget["risk_context"].get(f) == ctx.get(f), f"risk_path_sizes context.{f} != timing risk_context")
    check(budget["record_ciphertext_B"] == r["record_ciphertext"], "record ciphertext size differs from the timing run")
    check(budget["record_signed_message_B"] == r["signed_record_message"], "signed message size differs from the timing run")
    check(k["enc_w"] == r["record_ciphertext"], "Enc(w) size != record ciphertext size")
    msg = r["record_ciphertext"] + ci["dispute_commitments_B"] + ci["epoch_B"] + ci["id_ins_B"] + r["group_signature"]
    check(msg == r["upload_total"], "record message parts != upload_total")
    check(r["upload_total"] - r["group_signature"] == r["signed_record_message"], "signed message != upload - signature")
    check(r["group_signature"] == ci["group_signature_B"], "group signature size differs between the size files")
    check(sp["upload_per_policy_epoch_B"] == DAYS * r["upload_total"], "upload per policy-month != 31 x upload")
    check(sp["tsp_to_insurer_per_policy_epoch_B"] == DAYS * r["result_ciphertext_after_fill"], "TSP->insurer != 31 x result")
    check(sp["tsp_storage_10e6_policies_B"] == POLICIES * DAYS * r["result_ciphertext_after_fill"], "fleet storage")
    check(ci["single_key_json"] == "experiments/sizes/results/risk_path_sizes.json", "simpler designs not from risk_path_sizes")
    check(BP["B3_ckks"]["N"] == base["B3_ckks_encrypted_weights_score_only"]["ring_dimension"], "B3 N differs")
    check(day["records_per_epoch"] == DAYS, "settlement daily epoch is not 31 records")
    check(day["settlement_message_insurer_to_obu_B_honest_epoch"] == 8 * DAYS, "settlement message != 8 B per nonce")
    check(day["settlement_upload_obu_to_insurer_B_worst_case_all_disputed"] == DAYS * day["settlement_upload_item_B_max"],
          "worst-case settlement upload != 31 x item")
    # designs A and B recomputed from the field widths
    dA, dB = CI["designs"]["A"], CI["designs"]["B"]
    ab = {}
    for nm, d, body in (("A", dA, ci["R_B"]), ("B", dB, ci["x_B"])):
        pt = ci["id_B"] + body + ci["nonce_B"] + ci["mac_B"]
        ct = ci["kem_point_B"] + pt + ci["aead_tag_B"]
        signed = ct + ci["id_ins_B"] + ci["epoch_B"] + ci["dispute_commitments_B"]
        up = signed + ci["group_signature_B"]
        ab[nm] = dict(signed=signed, up=up)
        for f, v in (("plaintext_B", pt), ("ciphertext_B", ct), ("signed_message_B", signed), ("upload_B", up),
                     ("tsp_stored_per_record_B", signed), ("tsp_storage_10e6_policies_B", POLICIES * DAYS * signed)):
            check(d[f] == v, f"design {nm}.{f} {d[f]} != recomputed {v}")
    ratios = {}
    for nm in ("A", "B"):
        ratios[(nm, "upload")] = round(r["upload_total"] / ab[nm]["up"], 1)
        ratios[(nm, "storage")] = round(sp["tsp_storage_per_policy_epoch_B"] / (DAYS * ab[nm]["signed"]), 1)
        check(CI["ratios_SP2UBI_over"][nm]["upload_per_record"] == ratios[(nm, "upload")], f"ratio {nm} upload")
        check(CI["ratios_SP2UBI_over"][nm]["tsp_storage"] == ratios[(nm, "storage")], f"ratio {nm} storage")
    about = {}
    for kind in ("upload", "storage"):
        exact = ratios[("A", kind)]
        about[kind] = int(round(exact / 100.0) * 100)
        check(abs(about[kind] - exact) / exact < 0.01, f"rounded A {kind} ratio not within 1 % of {exact}")
    ok = SK["opening_key_generation_B_per_party"]
    check(ok["commitment"] + ok["reveal_nonce_A_B_two_pok"] + ok["h_share_H_V_with_two_dleq"]
          + 3 * ok["signature_per_message_long_term_key"] == ok["messages_total_signed"], "opening-key messages")
    check(10 * speed["beacon_B"] + speed["certificate_B"] + speed["rolling_B"] == EMIT["total"],
          "10 x beacon + certificate + trailer != speed-beacon bytes per second")

    up_bits = 8 * r["upload_total"]
    ch = {n: floor2(up_bits / (b / TTI_S)) for n, b in RATE_BITS_PER_TTI.items()}
    stor_month_B = POLICIES * r["result_ciphertext_after_fill"]
    share = day["settlement_upload_obu_to_insurer_B_worst_case_all_disputed"] / sp["upload_per_policy_epoch_B"]
    SKf, CIf, SETf = "risk_path_sizes.json", "simpler_designs_sizes.json", "settlement_sizes.json"
    SPf, B3f, B4f = "speed_path.json", "baselines_bfv_ckks.json", "baseline_paillier.json"
    rows = [  # key, value, class, unit, source, description
        ("sk|N", ctx["N"], "plain", "", f"{SKf} context.N", "ring dimension chosen by OpenFHE (HEStd_128_classic)"),
        ("sk|log2Q", ctx["log2_Q"], "plain", "bit", f"{SKf} context.log2_Q", "ciphertext modulus"),
        ("sk|modulus_bits", ctx["moduli_bits"][0], "plain", "bit", f"{SKf} context.moduli_bits", "each of the two RNS moduli"),
        ("sk|p_bits", ctx["t_bits"], "plain", "bit", f"{SKf} context.t_bits", "plaintext modulus = HomMAC prime"),
        ("sk|margin_bits", SK["decryption_margin_bits_result_at_caps"], "plain", "bit",
         f"{SKf} decryption_margin_bits_result_at_caps", "decryption margin of a result at the feature caps"),
        ("sk|ct_record", r["record_ciphertext"], "int", "B", f"{SKf} per_record_B.record_ciphertext", "record ciphertext"),
        ("sk|signed_msg", r["signed_record_message"], "int", "B", f"{SKf} per_record_B.signed_record_message",
         "group-signed record message"),
        ("sk|groupsig", r["group_signature"], "int", "B", f"{SKf} per_record_B.group_signature", "BBS04 signature"),
        ("sk|upload", r["upload_total"], "int", "B", f"{SKf} per_record_B.upload_total",
         "record upload (payload, without TLS framing and handshake)"),
        ("sk|result_after_fill", r["result_ciphertext_after_fill"], "int", "B",
         f"{SKf} per_record_B.result_ciphertext_after_fill", "result ciphertext after the slot fill"),
        ("sk|receipt", r["tsp_receipt"], "int", "B", f"{SKf} per_record_B.tsp_receipt", "TSP receipt"),
        ("sk|pk", k["public_key"], "int", "B", f"{SKf} keys_B.public_key", "public key"),
        ("sk|enc_w", k["enc_w"], "int", "B", f"{SKf} keys_B.enc_w", "Enc(w)"),
        ("sk|relin", k["relinearisation_key"], "int", "B", f"{SKf} keys_B.relinearisation_key", "relinearisation key"),
        ("sk|rot", k["rotation_keys_1_2_4"], "int", "B", f"{SKf} keys_B.rotation_keys_1_2_4", "rotation keys (1, 2, 4)"),
        ("sk|evalkeys", k["relinearisation_key"] + k["rotation_keys_1_2_4"], "int", "B",
         f"{SKf} keys_B.relinearisation_key + rotation_keys_1_2_4", "evaluation keys (sum)"),
        ("msg|commitment", ci["dispute_commitments_B"] // 2, "int", "B", f"{CIf} inputs.dispute_commitments_B / 2",
         "one commitment (SHA-256)"),
        ("msg|epoch", ci["epoch_B"], "int", "B", f"{CIf} inputs.epoch_B", "epoch index"),
        ("msg|id_ins", ci["id_ins_B"], "int", "B", f"{CIf} inputs.id_ins_B", "routing identifier ID_Ins"),
        ("sk|opening_setup", ok["messages_total_signed"], "int", "B",
         f"{SKf} opening_key_generation_B_per_party.messages_total_signed", "signed opening-key setup messages per party"),
        ("simpler|A|upload", dA["upload_B"], "int", "B", f"{CIf} designs.A.upload_B", "design A upload (computed)"),
        ("simpler|B|upload", dB["upload_B"], "int", "B", f"{CIf} designs.B.upload_B", "design B upload (computed)"),
        ("simpler|A|stored_record", dA["tsp_stored_per_record_B"], "int", "B", f"{CIf} designs.A.tsp_stored_per_record_B",
         "design A record stored at the TSP (upper bound)"),
        ("simpler|B|stored_record", dB["tsp_stored_per_record_B"], "int", "B", f"{CIf} designs.B.tsp_stored_per_record_B",
         "design B record stored at the TSP (upper bound)"),
        ("simpler|A|storage_GB", round(dA["tsp_storage_10e6_policies_GB"], 2), "dec2", "GB",
         f"{CIf} designs.A.tsp_storage_10e6_policies_GB", "design A TSP storage, 10^6 policies, 31 daily records"),
        ("simpler|B|storage_GB", round(dB["tsp_storage_10e6_policies_GB"], 2), "dec2", "GB",
         f"{CIf} designs.B.tsp_storage_10e6_policies_GB", "design B TSP storage, 10^6 policies, 31 daily records"),
        ("simpler|ratio_A_upload", ratios[("A", "upload")], "dec1", "", f"{CIf} ratios_SP2UBI_over.A.upload_per_record",
         "SP2UBI upload / design A upload"),
        ("simpler|ratio_B_upload", ratios[("B", "upload")], "dec1", "", f"{CIf} ratios_SP2UBI_over.B.upload_per_record",
         "SP2UBI upload / design B upload"),
        ("simpler|ratio_A_storage", ratios[("A", "storage")], "dec1", "", f"{CIf} ratios_SP2UBI_over.A.tsp_storage",
         "SP2UBI / design A TSP storage per policy-epoch"),
        ("simpler|ratio_B_storage", ratios[("B", "storage")], "dec1", "", f"{CIf} ratios_SP2UBI_over.B.tsp_storage",
         "SP2UBI / design B TSP storage per policy-epoch"),
        ("simpler|ratio_A_upload_about", about["upload"], "int", "", "simpler|ratio_A_upload rounded to the nearest 100",
         "printed as an approximate ratio"),
        ("simpler|ratio_A_storage_about", about["storage"], "int", "", "simpler|ratio_A_storage rounded to the nearest 100",
         "printed as an approximate ratio"),
        ("month|records", DAYS, "plain", "", "definition", "records of a 31-day epoch"),
        ("month|upload", sp["upload_per_policy_epoch_B"], "int", "B", f"{CIf} designs.SP2UBI.upload_per_policy_epoch_B",
         "upload per policy-month (31 x sk|upload)"),
        ("month|upload_MB", round(sp["upload_per_policy_epoch_B"] / 1e6, 2), "dec2", "MB",
         f"{CIf} designs.SP2UBI.upload_per_policy_epoch_B / 10^6", "upload per policy-month"),
        ("month|tsp_to_insurer", sp["tsp_to_insurer_per_policy_epoch_B"], "int", "B",
         f"{CIf} designs.SP2UBI.tsp_to_insurer_per_policy_epoch_B", "TSP to insurer per policy-month"),
        ("tsp|storage_daily_TB", sp["tsp_storage_10e6_policies_TB"], "dec2", "TB",
         f"{CIf} designs.SP2UBI.tsp_storage_10e6_policies_TB", "TSP storage, 10^6 policies, 31 daily records"),
        ("tsp|storage_monthly_TB", round(stor_month_B / 1e12, 2), "dec2", "TB",
         f"10^6 x {SKf} per_record_B.result_ciphertext_after_fill / 10^12", "TSP storage, one record per policy-month"),
        ("ch|cat1_s", ch["cat1"], "dec2", "s", "8 x sk|upload / (5160 bit per 1-ms TTI)",
         "channel time, UE Category 1 (lower bound, rounded down)"),
        ("ch|m1_s", ch["m1"], "dec2", "s", "8 x sk|upload / (1000 bit per 1-ms TTI)",
         "channel time, UE Category M1 (lower bound, rounded down)"),
        ("ch|m1_tbs2984_s", ch["m1_tbs2984"], "dec2", "s", "8 x sk|upload / (2984 bit per 1-ms TTI)",
         "channel time, Category M1 with the larger transport block (lower bound, rounded down)"),
        ("set|obu_state_daily", day["obu_dispute_state_per_receipted_record_B_max"], "int", "B",
         f"{SETf} configurations.daily.obu_dispute_state_per_receipted_record_B_max",
         "OBU dispute state per receipted record, daily (upper bound)"),
        ("set|obu_state_monthly", mon["obu_dispute_state_per_receipted_record_B_max"], "int", "B",
         f"{SETf} configurations.monthly.obu_dispute_state_per_receipted_record_B_max",
         "OBU dispute state per receipted record, monthly (upper bound)"),
        ("set|obu_state_epoch", day["obu_dispute_state_per_31d_epoch_B_max"], "int", "B",
         f"{SETf} configurations.daily.obu_dispute_state_per_31d_epoch_B_max", "OBU dispute state per 31-day epoch"),
        ("set|msg_per_nonce", day["settlement_message_insurer_to_obu_B_honest_epoch"] // day["records_per_epoch"], "int",
         "B", f"{SETf} daily.settlement_message_insurer_to_obu_B_honest_epoch / records_per_epoch",
         "settlement message per accepted nonce"),
        ("set|msg_daily", day["settlement_message_insurer_to_obu_B_honest_epoch"], "int", "B",
         f"{SETf} daily.settlement_message_insurer_to_obu_B_honest_epoch", "settlement message, all records accepted"),
        ("set|msg_conflict", day["settlement_message_insurer_to_obu_B_conflict_epoch"], "int", "B",
         f"{SETf} daily.settlement_message_insurer_to_obu_B_conflict_epoch", "settlement message, conflict epoch"),
        ("set|upload_honest", day["settlement_upload_obu_to_insurer_B_honest_epoch"], "int", "B",
         f"{SETf} daily.settlement_upload_obu_to_insurer_B_honest_epoch", "settlement upload, honest epoch"),
        ("set|item_daily", day["settlement_upload_item_B_max"], "int", "B", f"{SETf} daily.settlement_upload_item_B_max",
         "settlement upload item (upper bound)"),
        ("set|upload_worst_daily", day["settlement_upload_obu_to_insurer_B_worst_case_all_disputed"], "int", "B",
         f"{SETf} daily.settlement_upload_obu_to_insurer_B_worst_case_all_disputed",
         "settlement upload, every record disputed (upper bound)"),
        ("set|share_pct", round(100 * share, 2), "dec2", "%", "set|upload_worst_daily / month|upload",
         "worst-case settlement upload, share of the monthly record uploads"),
        ("sp|beacon", speed["beacon_B"], "int", "B", f"{SPf} result.sizes_K40.beacon_B", "speed beacon"),
        ("sp|cert", speed["certificate_B"], "int", "B", f"{SPf} result.sizes_K40.certificate_B", "burst certificate"),
        ("sp|trailer", speed["rolling_B"], "int", "B", f"{SPf} result.sizes_K40.rolling_B", "rolling trailer"),
        ("sp|air40", speed["air_bytes_per_crossing"], "int", "B", f"{SPf} result.sizes_K40.air_bytes_per_crossing",
         "air bytes of a crossing with 40 beacons"),
        ("sp|entry", speed["entry_B"], "int", "B", f"{SPf} result.sizes_K40.entry_B", "RSU entry"),
        ("sp|Bps", EMIT["total"], "int", "B/s", "speed_beacon_emission.json bytes_per_second.total",
         "speed-beacon emission (10 x sp|beacon + sp|cert + sp|trailer)"),
        ("sp|kBps", EMIT["total_kB_per_s"], "dec1", "kB/s", "speed_beacon_emission.json bytes_per_second.total_kB_per_s",
         "speed-beacon emission"),
        ("b3|N", base["B3_ckks_encrypted_weights_score_only"]["ring_dimension"], "plain", "",
         f"{B3f} result.baselines.B3_ckks_encrypted_weights_score_only.ring_dimension", "baseline B3 ring dimension"),
        ("b3|log2Q", BP["B3_ckks"]["log2_Q"], "dec2", "bit", "baseline_params.json B3_ckks.log2_Q", "baseline B3 log2 Q"),
        ("b3|ct", base["B3_ckks_encrypted_weights_score_only"]["ciphertext_B"], "int", "B",
         f"{B3f} result.baselines.B3_ckks_encrypted_weights_score_only.ciphertext_B", "baseline B3 ciphertext"),
        ("b4|modulus_bits", pail["B4_paillier_clear_weights"]["modulus_bits"], "plain", "bit",
         f"{B4f} result.baselines.B4_paillier_clear_weights.modulus_bits", "baseline B4 Paillier modulus"),
        ("b4|ct_value", pail["B4_paillier_clear_weights"]["ciphertext_one_value_B"], "int", "B",
         f"{B4f} result.baselines.B4_paillier_clear_weights.ciphertext_one_value_B", "baseline B4 ciphertext of one value"),
        ("b4|upload", pail["B4_paillier_clear_weights"]["record_upload_B"], "int", "B",
         f"{B4f} result.baselines.B4_paillier_clear_weights.record_upload_B", "baseline B4 record upload"),
    ]
    return [dict(key=key, value=v, printed=printed(v, cls), unit=unit, q1="", q3="", n="", source="sizes: " + src,
                 description=desc) for key, v, cls, unit, src, desc in rows]


# ---------------------------------------------------------------------------------------------------- timings
BUDGET = {"obu": ("risk_path_budget|obu_record_message_total", "risk_path_budget|obu_verify_receipt"),
          "tsp": ("risk_path_budget|tsp_receive_total", "risk_path_budget|tsp_end_epoch_publish"),
          "insurer": ("risk_path_budget|insurer_decrypt", "risk_path_budget|insurer_hommac_verify_both_blocks")}
T5A = ("result", "timings")
TB1 = ("result", "baselines", "B1_single_key_bfv_encrypted_weights", "timings")
TB2 = ("result", "baselines", "B2_sp2ubi_context_clear_weights", "timings")
TB3 = ("result", "baselines", "B3_ckks_encrypted_weights_score_only", "timings")
TB4 = ("result", "baselines", "B4_paillier_clear_weights", "timings")


def stat(item, *path):
    return ("stat", item, path)


TIMING_KEYS = {
    **{f"risk_path_budget|{op}": stat("risk_path_budget", *T5A, op) for op in (
        "obu_bfv_encrypt", "obu_group_sign", "obu_hommac_tags_both_blocks", "obu_record_message_total",
        "tsp_receive_total", "tsp_group_verify", "tsp_deserialize_ciphertext", "tsp_evaluate", "tsp_fill",
        "tsp_receipt_sign", "obu_verify_receipt", "tsp_end_epoch_publish", "insurer_decrypt",
        "insurer_hommac_verify_both_blocks", "chain_end_to_end_compute")},
    "risk_path_budget|obu_enc_plus_sign": ("sum", ("risk_path_budget|obu_bfv_encrypt", "risk_path_budget|obu_group_sign")),
    **{f"risk_path_budget|budget_{p}": ("num", "risk_path_budget", ("result", "per_record_budget_ms_sum_of_medians", p),
                                        "ms") for p in ("obu", "tsp", "insurer", "total")},
    "baselines|B1|obu_encrypt": stat("baselines_bfv_ckks", *TB1, "obu_encrypt"),
    "baselines|B1|tsp_evaluate": stat("baselines_bfv_ckks", *TB1, "tsp_evaluate"),
    "baselines|B1|insurer_decrypt": stat("baselines_bfv_ckks", *TB1, "insurer_decrypt"),
    "baselines|B2|tsp_evaluate": stat("baselines_bfv_ckks", *TB2, "tsp_evaluate_clear_weights"),
    "baselines|B2|tsp_fill": stat("baselines_bfv_ckks", *TB2, "tsp_fill"),
    "baselines|B3|obu_encrypt": stat("baselines_bfv_ckks", *TB3, "obu_encrypt"),
    "baselines|B3|tsp_evaluate": stat("baselines_bfv_ckks", *TB3, "tsp_evaluate"),
    "baselines|B3|insurer_decrypt": stat("baselines_bfv_ckks", *TB3, "insurer_decrypt"),
    "baselines|B3|max_rel_err": ("num", "baselines_bfv_ckks", ("result", "baselines",
                                 "B3_ckks_encrypted_weights_score_only", "max_relative_error"), "sci"),
    "baselines|B4|obu_encrypt": stat("baseline_paillier", *TB4, "obu_encrypt_20_values"),
    "baselines|B4|tsp_evaluate": stat("baseline_paillier", *TB4, "tsp_evaluate_clear_weights"),
    "baselines|B4|insurer_decrypt": stat("baseline_paillier", *TB4, "insurer_decrypt_5_values"),
    **{f"throughput|workers{w}": ("num", "throughput", ("result", "throughput", str(w), "records_per_s_median"), "rps")
       for w in range(1, 9)},
    **{f"throughput|workers{w}|{e}": ("num", "throughput", ("result", "throughput", str(w), f"records_per_s_{e}"), "rps")
       for w in range(1, 9) for e in ("min", "max")},
    **{f"speed_path|{op}": stat("speed_path", "result", op) for op in (
        "tee_encrypt_per_beacon", "tee_trailer_smudging_term_encrypt", "tee_burst_certificate_bbs04_sign",
        "tee_rolling_trailer_schnorr_sign", "rsu_accept_crossing_K40", "rsu_evaluate_K40",
        "rsu_release_rerandomise_and_sign_K40", "rsu_accept_crossing_K160", "tsp_verify_entry",
        "tsp_shuffle_rerandomize_per_entry", "tsp_partials_with_proofs_per_entry",
        "insurer_verify_open_and_decrypt_per_entry", "insurer_decrypt_b+0.00", "insurer_decrypt_b+0.10")},
}
DERIVED = [  # key, function of g, class, unit, description
    ("derived|ratio_eval_B1", lambda g: g("risk_path_budget|tsp_evaluate") / g("baselines|B1|tsp_evaluate"), "ratio", "",
     "TSP evaluation, SP2UBI / B1 (single-key BFV)"),
    ("derived|speedup_8workers", lambda g: g("throughput|workers8") / g("throughput|workers1"), "ratio", "",
     "throughput with 8 workers / with 1 worker"),
    ("derived|core_h_daily", lambda g: 31 * 1e6 * g("risk_path_budget|budget_tsp") / 3.6e6, "int", "core-h",
     "TSP core-hours per 31-day month per 10^6 policies, daily records"),
    ("derived|core_h_monthly", lambda g: 1e6 * g("risk_path_budget|budget_tsp") / 3.6e6, "int", "core-h",
     "TSP core-hours per month per 10^6 policies, one record per policy-month"),
    ("derived|machine_days_daily_8workers", lambda g: 31e6 / g("throughput|workers8") / 86400, "dec1", "d",
     "days of the measurement machine (8 workers) per month for 10^6 policies, daily records"),
    ("derived|speedup_4workers", lambda g: g("throughput|workers4") / g("throughput|workers1"), "ratio2", "",
     "throughput with 4 workers / with 1 worker"),
    ("derived|eff_4workers", lambda g: 100 * g("throughput|workers4") / (4 * g("throughput|workers1")), "int", "%",
     "per cent of linear scaling at 4 workers"),
    ("derived|core_ms_8workers", lambda g: 4 * 1000 / g("throughput|workers8"), "ms", "ms",
     "physical-core time per record at 8 workers (4 physical cores)"),
    ("derived|chain_minus_budget", lambda g: g("risk_path_budget|chain_end_to_end_compute")
     - g("risk_path_budget|budget_total"), "ms", "ms", "whole chain timed as one minus the sum of the budget medians"),
    ("derived|workers5to8_min", lambda g: min(g(f"throughput|workers{w}") for w in (5, 6, 7, 8)), "rps", "records/s",
     "smallest throughput median with 5 to 8 workers"),
    ("derived|workers5to8_max", lambda g: max(g(f"throughput|workers{w}") for w in (5, 6, 7, 8)), "rps", "records/s",
     "largest throughput median with 5 to 8 workers"),
    ("derived|deser_pct_tsp", lambda g: 100 * g("risk_path_budget|tsp_deserialize_ciphertext")
     / g("risk_path_budget|budget_tsp"), "int", "%", "deserialisation of an uploaded ciphertext, share of the TSP budget"),
]


def node(data, item, path):
    x = data[item]
    for k in path:
        x = x[k]
    return x


def timing_rows(errors):
    data = {}
    for it in ITEMS:
        d = load_json(TIMINGS, f"{it}.json")
        if d.get("result_status") != "final" or not d.get("final_rule", {}).get("at_start", {}).get("ok"):
            errors.append(f"timing item {it} is not final")
        data[it] = d
    vals, rows = {}, []
    for key, spec in TIMING_KEYS.items():
        if spec[0] == "stat":
            n = node(data, spec[1], spec[2])
            med, q1, q3, reps = float(n["median_ms"]), float(n["q1_ms"]), float(n["q3_ms"]), n.get("n", n.get("reps"))
            d = ms_digits(med)
            vals[key] = med
            rows.append(dict(key=key, value=med, printed=printed(med, "ms"), unit="ms", q1=group(f"{q1:.{d}f}"),
                             q3=group(f"{q3:.{d}f}"), n=reps, source=f"{spec[1]}.json {'.'.join(spec[2])}.median_ms",
                             description="median over the timed repetitions"))
        elif spec[0] == "sum":
            v = sum(vals[k] for k in spec[1])
            vals[key] = v
            rows.append(dict(key=key, value=v, printed=printed(v, "ms"), unit="ms", q1="", q3="", n="",
                             source=" + ".join(spec[1]), description="sum of the medians"))
        else:
            _, item, path, cls = spec
            v = float(node(data, item, path))
            vals[key] = v
            rows.append(dict(key=key, value=v, printed=printed(v, cls), unit={"ms": "ms", "rps": "records/s"}.get(cls, ""),
                             q1="", q3="", n="", source=f"{item}.json {'.'.join(path)}", description=""))
    t = data["risk_path_budget"]["result"]["timings"]
    b = data["risk_path_budget"]["result"]["per_record_budget_ms_sum_of_medians"]
    sums = {p: sum(float(t[k.split("|")[1]]["median_ms"]) for k in parts) for p, parts in BUDGET.items()}
    sums["total"] = sum(sums.values())
    for p, v in sums.items():
        if abs(float(b[p]) - v) > 1e-9 * max(1.0, abs(v)):
            errors.append(f"budget {p}: stored {b[p]} != sum of the listed medians {v}")
    for key, fn, cls, unit, desc in DERIVED:
        v = float(fn(vals.__getitem__))
        vals[key] = v
        rows.append(dict(key=key, value=v, printed=printed(v, cls), unit=unit, q1="", q3="", n="", source="derived",
                         description=desc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "paper_values.csv"))
    a = ap.parse_args()
    errors = []
    rows = size_rows(errors) + timing_rows(errors)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value", "printed", "unit", "q1", "q3", "n", "source", "description"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} values written to {os.path.relpath(a.out, ROOT)}")
    if errors:
        print("consistency checks failed:\n  " + "\n  ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
