"""Study 2 held-out seeds. They are fixed by the timestamped design-record bundle and by the Bitcoin block that holds its
OpenTimestamps proof, so that the bundle's bytes cannot be varied in search of favourable sites (the block hash does not
exist when the bundle is built).

  bundle SHA = SHA-256 of the bundle manifest (the manifest lists every bundle file with its SHA-256)
  H          = hash of the Bitcoin block that timestamps the manifest, as block explorers show it (SHA256d of the
               80-byte header, byte-reversed, lower-case hex). The upgraded .ots proof gives the block height and the
               commitment (the block's Merkle root); the seed record (config/study2_seed_record.json) holds the
               80-byte header. Checks:
               (1) the header's Merkle-root field = the proof's commitment;
               (2) SHA256d(header) = the recorded H;
               (3) proof of work: SHA256d(header) as a little-endian integer <= the target encoded in the header's
                   nBits, and that target <= POW_MAX_TARGET = 4 x the target of block 972524 (nBits 0x17020093;
                   difficulty 1.406e14), so a header with the right Merkle root cannot be produced at low cost;
               (4) no calendar is left pending: every pending attestation of the proof must be resolved, otherwise a
                   later upgrade could reveal a lower block and a second H. The one exception is the pre-specified
                   48-h rule (`pending_deadline`, `check_pending_deadline`): a calendar still pending 48 h after the
                   first Bitcoin attestation is listed and ignored.
               Checks (1) and (4) need the manifest and the .ots proof; (2) and (3) need only the seed record.
  seed(name) = int(SHA-256(ascii(bundle SHA hex) + ascii(H hex) + "study2" + ascii(name)).hexdigest[:8], 16)
               for name in G1_site_base, G1_pass_base, sumo, impairments, bootstrap; the other seeds keep their
               study-1 values; a collision with any study-1 or pilot seed base is refused.
The .ots parser below reads the OpenTimestamps serialisation (magic; version; file-hash op).
Usage: python make_constants.py --study2 ... --study2-seeds config/study2_seed_record.json"""
import hashlib
import json
import os

from pipeline import s2_constants as S2

MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
BITCOIN_TAG = bytes.fromhex("0588960d73d71901")
PENDING_TAG = bytes.fromhex("83dfe30d2ef90c8e")
HASH_OPS = {0x02: "sha1", 0x03: "ripemd160", 0x08: "sha256", 0x67: "keccak256"}
OP_APPEND, OP_PREPEND, OP_REVERSE, OP_HEXLIFY = 0xF0, 0xF1, 0xF2, 0xF3


class _R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def byte(self):
        if self.i >= len(self.b):
            raise ValueError("truncated .ots proof")
        v = self.b[self.i]
        self.i += 1
        return v

    def bytes(self, n):
        if self.i + n > len(self.b):
            raise ValueError("truncated .ots proof")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def varuint(self):
        v, s = 0, 0
        while True:
            c = self.byte()
            v |= (c & 0x7F) << s
            s += 7
            if not c & 0x80:
                return v

    def varbytes(self):
        return self.bytes(self.varuint())


def _apply(op, arg, msg):
    if op == OP_APPEND:
        return msg + arg
    if op == OP_PREPEND:
        return arg + msg
    if op == OP_REVERSE:
        return msg[::-1]
    if op == OP_HEXLIFY:
        return msg.hex().encode()
    name = HASH_OPS[op]
    if name == "keccak256":
        raise ValueError("keccak256 operations are not supported")
    return hashlib.new(name, msg).digest()


def _timestamp(r, msg, out):
    """Walk one timestamp node (the serialisation of python-opentimestamps Timestamp.serialize): 0xff precedes every
    entry but the last; an entry is an attestation (0x00, 8-byte tag, varbytes payload) or an operation followed by the
    child timestamp of the operation's result. Returns whether the node's subtree holds a Bitcoin attestation; a pending
    attestation is 'resolved' iff its own node does."""
    has_btc, pend = False, []
    while True:
        tag = r.byte()
        more = tag == 0xFF
        if more:
            tag = r.byte()
        if tag == 0x00:
            atag, payload = r.bytes(8), r.varbytes()
            if atag == BITCOIN_TAG:
                out.append(dict(kind="bitcoin", height=_R(payload).varuint(), commitment=msg.hex()))
                has_btc = True
            elif atag == PENDING_TAG:
                pend.append(dict(kind="pending", uri=_R(payload).varbytes().decode(errors="replace")))
                out.append(pend[-1])
            else:
                out.append(dict(kind="unknown", tag=atag.hex()))
        else:
            arg = r.varbytes() if tag in (OP_APPEND, OP_PREPEND) else None
            if tag not in (OP_APPEND, OP_PREPEND, OP_REVERSE, OP_HEXLIFY) and tag not in HASH_OPS:
                raise ValueError(f"unknown .ots operation 0x{tag:02x}")
            has_btc |= _timestamp(r, _apply(tag, arg, msg), out)
        if not more:
            for p in pend:
                p["resolved"] = has_btc
            return has_btc


def parse_ots(data):
    """Returns dict(file_hash_op, file_digest, attestations=[...]) of a detached .ots proof (bytes)."""
    r = _R(data)
    if r.bytes(len(MAGIC)) != MAGIC:
        raise ValueError("not an OpenTimestamps proof")
    if r.varuint() != 1:
        raise ValueError("unsupported .ots version")
    op = r.byte()
    if op not in HASH_OPS:
        raise ValueError(f"unknown file hash op 0x{op:02x}")
    n = {"sha1": 20, "ripemd160": 20, "sha256": 32, "keccak256": 32}[HASH_OPS[op]]
    digest = r.bytes(n)
    out = []
    _timestamp(r, digest, out)
    return dict(file_hash_op=HASH_OPS[op], file_digest=digest.hex(), attestations=out)


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def block_hash_from_header(header_hex):
    h = bytes.fromhex(header_hex)
    if len(h) != 80:
        raise ValueError("a Bitcoin block header has 80 bytes")
    return hashlib.sha256(hashlib.sha256(h).digest()).digest()[::-1].hex()


def merkle_root_of_header(header_hex):
    return bytes.fromhex(header_hex)[36:68].hex()


def target_from_bits(bits):
    """The proof-of-work target encoded in a header's compact nBits (mantissa x 256^(exponent - 3))."""
    exp, mant = bits >> 24, bits & 0x007FFFFF
    if bits & 0x00800000:
        raise ValueError("negative nBits target")
    return mant * (1 << (8 * (exp - 3))) if exp >= 3 else mant >> (8 * (3 - exp))


def header_bits(header_hex):
    return int.from_bytes(bytes.fromhex(header_hex)[72:76], "little")


P0_BITS_972524 = 0x17020093           # nBits of block 972524
POW_MAX_TARGET = 4 * target_from_bits(P0_BITS_972524)   # Bitcoin retargets by at most 4x per 2016-block period


def pow_ok(header_hex, max_target=None):
    """(3) proof of work: SHA256d(header) <= its own nBits target, and that target <= max_target (POW_MAX_TARGET)."""
    h = bytes.fromhex(header_hex)
    if len(h) != 80:
        raise ValueError("a Bitcoin block header has 80 bytes")
    t = target_from_bits(header_bits(header_hex))
    work = int.from_bytes(hashlib.sha256(hashlib.sha256(h).digest()).digest(), "little")
    return bool(0 < t <= (POW_MAX_TARGET if max_target is None else max_target) and work <= t)


def bitcoin_attestations(proof):
    return [a for a in proof["attestations"] if a["kind"] == "bitcoin"]


def unresolved_pending(proof):
    """The calendars whose pending attestation has no Bitcoin attestation in its node (not yet upgraded)."""
    return sorted(a["uri"] for a in proof["attestations"] if a["kind"] == "pending" and not a.get("resolved"))


PENDING_DEADLINE_H = 48                # a calendar still pending 48 h after the first attestation is ignored


def _utc(t):
    """An ISO-8601 time with an explicit UTC offset ('Z' or '+00:00') -> datetime (refuses a naive time)."""
    import datetime as _dt
    d = _dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    if d.tzinfo is None or d.utcoffset() != _dt.timedelta(0):
        raise ValueError(f"pending_deadline: time {t!r} must be UTC with an explicit offset (a trailing Z or +00:00)")
    return d


def header_time(header_hex):
    """The header's own nTime (bytes 68-72, little endian) as a UTC datetime."""
    import datetime as _dt
    return _dt.datetime.fromtimestamp(int.from_bytes(bytes.fromhex(header_hex)[68:72], "little"), _dt.timezone.utc)


def check_pending_deadline(pd, pend, header_hex=None):
    """If a calendar is still pending 48 h after the first Bitcoin attestation, H is the block of lowest height among
    the Bitcoin attestations present in the proof as upgraded at that time, the still-pending calendars are listed in
    the seed record and ignored, and no later upgrade changes H. `pd` is the seed-record field `pending_deadline` =
    {first_attestation_utc, final_upgrade_utc, ignored_pending}; accepted only if (a) ignored_pending lists exactly the
    proof's unresolved pending calendars (a non-empty list: a later upgrade that resolves one is refused, so it cannot
    move H), and (b) final_upgrade_utc - max(first attestation, header time) >= 48 h. Returns the record that goes into
    the constants."""
    for k in ("first_attestation_utc", "final_upgrade_utc", "ignored_pending"):
        if k not in pd:
            raise ValueError(f"pending_deadline: field {k!r} missing")
    if not pend:
        raise ValueError("pending_deadline given but no calendar is pending in the proof: remove the field")
    if sorted(pd["ignored_pending"]) != sorted(pend):
        raise ValueError(f"pending_deadline: ignored_pending {sorted(pd['ignored_pending'])} differs from the proof's "
                         f"pending calendars {sorted(pend)}")
    first, last = _utc(pd["first_attestation_utc"]), _utc(pd["final_upgrade_utc"])
    start = max(first, header_time(header_hex)) if header_hex else first
    span_h = (last - start).total_seconds() / 3600.0
    if not span_h >= PENDING_DEADLINE_H:
        raise ValueError(f"pending_deadline: the final upgrade is {span_h:.2f} h after the first Bitcoin attestation "
                         f"(< {PENDING_DEADLINE_H} h): upgrade again and wait")
    return dict(first_attestation_utc=pd["first_attestation_utc"], final_upgrade_utc=pd["final_upgrade_utc"],
                ignored_pending=sorted(pend), span_h=round(span_h, 3))


def lowest_bitcoin_block(proof, manifest_sha, pending_deadline=None, header_hex=None):
    """The Bitcoin attestation of lowest height; checks that the proof timestamps the manifest and that no calendar is
    still pending (4): the seeds are derived only from a fully upgraded proof. The only exception is the pre-specified
    48-h rule (`check_pending_deadline`): with `pending_deadline` recorded and valid, the still-pending calendars are
    ignored and H is the lowest-height attestation present."""
    if proof["file_digest"] != manifest_sha:
        raise ValueError("the .ots proof does not timestamp this manifest (digest differs from the bundle SHA)")
    btc = bitcoin_attestations(proof)
    if not btc:
        raise ValueError("the .ots proof has no Bitcoin attestation (run `ots upgrade`)")
    pend = unresolved_pending(proof)
    if pending_deadline is not None:
        check_pending_deadline(pending_deadline, pend, header_hex)
    elif pend:
        raise ValueError(f"the .ots proof still has pending calendar attestations {pend}: upgrade again; if one is still "
                         "pending 48 h after the first Bitcoin attestation, record `pending_deadline` in "
                         "the seed record")
    return min(btc, key=lambda a: a["height"])


def derive(bundle_sha, block_hash, names=S2.SEED_NAMES, domain=S2.SEED_DOMAIN):
    """seed(name) = int(sha256(bundle_sha + block_hash + domain + name)[:8 hex], 16)."""
    for x in (bundle_sha, block_hash):
        if len(x) != 64 or any(c not in "0123456789abcdef" for c in x):
            raise ValueError("bundle SHA and block hash are 64 lower-case hex digits")
    return {n: int(hashlib.sha256((bundle_sha + block_hash + domain + n).encode("ascii")).hexdigest()[:8], 16)
            for n in names}


def existing_bases(C):
    """Every study-1 production and pilot seed base (ints, lists flattened)."""
    vals = []
    for src in (C["seeds"], C["evaluation_v4"]["p0"]["seeds"]):
        for v in src.values():
            vals += list(v) if isinstance(v, list) else [v]
    return set(int(v) for v in vals)


def check_collisions(seeds, C):
    clash = sorted(n for n, v in seeds.items() if v in existing_bases(C))
    if clash or len(set(seeds.values())) != len(seeds):
        raise ValueError(f"study-2 seed collision with a study-1 / pilot base or within the set: {clash}")
    return True


def seeds_from_record(rec_path, C):
    """Verify the seed record and derive the seeds. Always: the header meets its own proof of work with a target
    <= POW_MAX_TARGET (3) and H = SHA256d(header) = the recorded block hash (2). If the record names a manifest and a
    proof that exist: bundle SHA = SHA-256(manifest) = the proof's digest, no pending calendar left (4), the
    lowest-height Bitcoin attestation's commitment = the header's Merkle root (1) and its height = the recorded height."""
    reg = json.load(open(rec_path))
    base = os.path.dirname(os.path.abspath(rec_path))
    bundle_sha = reg["bundle_sha256"]
    pd = reg.get("pending_deadline")
    have_proof = all(reg.get(k) and os.path.exists(os.path.join(base, reg[k])) for k in ("manifest", "proof"))
    if have_proof:
        if sha256_file(os.path.join(base, reg["manifest"])) != bundle_sha:
            raise ValueError("seed record: bundle SHA differs from SHA-256(manifest)")
        proof = parse_ots(open(os.path.join(base, reg["proof"]), "rb").read())
        att = lowest_bitcoin_block(proof, bundle_sha, pending_deadline=pd, header_hex=reg["block_header_hex"])
        if att["height"] != reg["block_height"]:
            raise ValueError("seed record: block height differs from the lowest attested height")
        if merkle_root_of_header(reg["block_header_hex"]) != att["commitment"]:
            raise ValueError("seed record: the block header does not carry the proof's commitment")
    elif merkle_root_of_header(reg["block_header_hex"]) != bytes.fromhex(reg.get("block_merkle_root", ""))[::-1].hex():
        raise ValueError("seed record: the block header does not carry the recorded Merkle root (explorer byte order)")
    if not pow_ok(reg["block_header_hex"]):
        raise ValueError("seed record: the block header fails its proof of work or its target exceeds "
                         "POW_MAX_TARGET (4 x the target of block 972524)")
    H = block_hash_from_header(reg["block_header_hex"])
    if H != reg["block_hash"]:
        raise ValueError("seed record: block hash differs from SHA256d(header)")
    seeds = derive(bundle_sha, H)
    check_collisions(seeds, C)
    rec = dict(bundle_sha256=bundle_sha, block_height=reg["block_height"], block_hash=H)
    if pd is not None and have_proof:
        rec["pending_deadline"] = check_pending_deadline(pd, unresolved_pending(proof), reg["block_header_hex"])
    return seeds, rec
