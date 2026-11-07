"""Hiding, binding commitment to a daily record, used for dispute resolution.

OpenFHE does not expose the encryption randomness, so a dispute cannot re-encrypt a stored plaintext. The OBU instead
places c = SHA-256(salt || record) inside the group-signed message and keeps only (record, salt) in its TEE.
In a dispute it opens (record, salt) and anyone checks c against the signed message.
"""
import hashlib
import json
import secrets
from typing import Tuple

SALT_BYTES = 32


def encode_record(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def commit(record: dict) -> Tuple[bytes, bytes]:
    salt = secrets.token_bytes(SALT_BYTES)
    return hashlib.sha256(salt + encode_record(record)).digest(), salt


def open_commitment(c: bytes, record: dict, salt: bytes) -> bool:
    return secrets.compare_digest(c, hashlib.sha256(salt + encode_record(record)).digest())
