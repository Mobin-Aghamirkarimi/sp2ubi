"""Sizes in bytes of the risk path's settlement traffic and of the OBU's dispute state per receipted record, computed
by the implementation itself (no timing). The OBU keeps no ciphertext of a receipted record; settlement messages and
uploads carry no padding (the TSP holds no policy).

* insurer -> OBU settlement message: 8 B per accepted nonce (`Insurer.settlement_message`; at most one per possible
  record of the epoch; 8 B in a conflict epoch);
* OBU -> insurer settlement upload: one item (evidence + dispute; at most `entities.settlement_item_bytes_max`) per
  receipted record that was not accepted (`entities.encode_settlement_upload`), 0 B in an honest epoch, before the
  end-to-end authenticated encryption; the worst case has every record of the epoch disputed;
* OBU sealed state per receipted record: committed record (JSON, at its maximum, `entities.max_committed_record`) +
  salt of cm_r + interval/capped-day metadata (JSON, at its maximum) + `entities.Receipted` (digest, cm_r, cm_b, t) +
  the TSP receipt; no ciphertext. The OBU could recompute cm_r and cm_b from the sealed values; the
  implementation keeps them.
* the record upload per record (results/risk_path_sizes.json, OpenFHE 1.5.1 objects) for the ratio.

Run (from the repository root):  python experiments/sizes/settlement_sizes.py
Output: experiments/sizes/results/settlement_sizes.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from sp2ubi.commitment import SALT_BYTES, encode_record  # noqa: E402
from sp2ubi.entities import (RECEIPT_BYTES, max_committed_record, settlement_item_bytes_max)  # noqa: E402
from sp2ubi.params import Params  # noqa: E402

record_upload = json.load(open(os.path.join(HERE, "results", "risk_path_sizes.json")))["per_record_B"]["upload_total"]
out = {"record_upload_per_record_B": record_upload, "configurations": {}}
for name, prm in (("daily", Params(interval_days=1)), ("monthly", Params(interval_days=31))):
    D = prm.max_epoch_days
    slots = prm.records_per_epoch(D)                               # possible records of a 31-day epoch
    meta = {"interval": prm.records_per_epoch(D) - 1, "capped_days": list(range(D - prm.interval_days, D))
            if prm.interval_days < D else list(range(D))}
    committed = len(encode_record(max_committed_record(prm)))
    receipted = 32 + 32 + 32 + 4                                   # Receipted: digest, cm_r, cm_b, t
    per_record = committed + SALT_BYTES + len(encode_record(meta)) + receipted + RECEIPT_BYTES
    up = slots * settlement_item_bytes_max(prm)
    out["configurations"][name] = {
        "records_per_epoch": slots,
        "settlement_message_insurer_to_obu_B_honest_epoch": slots * 8,
        "settlement_message_insurer_to_obu_B_conflict_epoch": 8,
        "settlement_upload_item_B_max": settlement_item_bytes_max(prm),
        "settlement_upload_obu_to_insurer_B_honest_epoch": 0,
        "settlement_upload_obu_to_insurer_B_worst_case_all_disputed": up,
        "record_uploads_per_policy_month_B": slots * record_upload,
        "settlement_upload_worst_case_share_of_record_uploads": up / (slots * record_upload),
        "obu_committed_record_encoding_B_max": committed,
        "obu_dispute_state_per_receipted_record_B_max": per_record,
        "obu_dispute_state_per_31d_epoch_B_max": slots * per_record,
    }
out["note"] = ("Sizes before the end-to-end authenticated encryption of the settlement channel; JSON record encodings are "
               "upper bounds (a binary encoding is smaller). No padding: the settlement message carries the accepted "
               "nonces, the upload one item per receipted record that was not accepted. Keeping the receipted record "
               "message instead would need one record message (record_upload_per_record_B) per disputed record.")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "settlement_sizes.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
