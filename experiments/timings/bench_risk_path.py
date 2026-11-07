"""Per-record compute budget, TSP throughput and baselines on the risk-path workload (one insurer BFV key), using the
classes of the reference implementation unchanged (entities.OBU / TSP / Insurer, fhe.RiskContext, groupsig_bbs04,
hommac).

Parts (`--part`; run_timings.sh runs each as its own item):
  budget      One daily record through the whole risk path, serially, single thread, `--reps` times (>= 1000):
              OBU record_message (seal, both HomMAC tag blocks, BFV encryption, commitments, group signature)
              -> TSP receive (group-signature check, evaluation with Enc(w), slot fill, receipt) -> OBU receipt check
              -> TSP end of epoch (shuffle, publish) -> insurer decryption, HomMAC check.
              Each component is timed on the same record, and the chain is also timed as a whole; the sum is the
              compute part of the per-record latency budget. The channel time is not measured here (it follows from
              the record size, experiments/sizes/results/risk_path_sizes.json).
              Also timed: serialization of the record ciphertext (OBU upload) and its deserialization (TSP).
  throughput  TSP records/s with W = 1..8 worker processes (`--workers`), OMP_NUM_THREADS=1 per worker, `--records`
              records per worker, `--repeats` repetitions per W. Each worker is one insurer tenant with its own key
              context, keys and Enc(w) (multi-tenant load); a worker runs TSP.receive on pre-made signed records (OBU
              cost excluded) and, every `--batch` records, end_epoch (the TSP's whole per-record work; the TSP
              decrypts nothing). Throughput = W * records / (last end - first start) after a common start barrier.
  baselines   Same record, same weights, same slot layout, single thread, `--reps` each (OpenFHE 1.5.1):
              B1 single-key BFV, encrypted weights, no slot fill (N chosen by OpenFHE);
              B2 the SP2UBI context with clear weights (the TSP sees w: the model-revealing ablation);
              B3 CKKS, encrypted weights, single key, depth 1 (approximate: score only, no mod-p HomMAC).
  paillier    B4 Paillier-3072 (python-phe 1.5.0 + gmpy2), clear weights (additive HE cannot multiply two ciphertexts),
              one ciphertext per value, both HomMAC tag blocks; runs in the Paillier environment (no OpenFHE import).

Usage (from the repository root; run_timings.sh does this):
  OMP_NUM_THREADS=1 .venv/bin/python experiments/timings/bench_risk_path.py --part budget --reps 1000 --out OUT.json
  (budget and baselines with OMP_NUM_THREADS=1; paillier with .venv-paillier/bin/python; throughput without an OMP
  setting, each worker sets OMP_NUM_THREADS=1). `--smoke` runs every part with tiny repetition counts (functional
  check only; its numbers are not measurements).
"""
import argparse
import os
import secrets
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_common as tc  # noqa: E402

sys.path.insert(0, tc.ROOT)
from sp2ubi import hommac  # noqa: E402  (pure Python; no OpenFHE import)
from sp2ubi.params import DEFAULT as P  # noqa: E402

D_EPOCH = 31
DAILY = [(1200, 300, 0, 2, 1), (800, 0, 450, 0, 0), (0, 0, 0, 0, 0), (2500, 900, 1300, 5, 3)]  # fixed daily vectors
ID_INS = b"INS-0001"
SCRIPTS = [os.path.abspath(__file__), os.path.join(HERE, "timing_common.py")]


def model_weights():
    from sp2ubi.risk_model import load_model
    return load_model().w


def sample_record(p):
    """A daily record x = (1 day, off-peak, rush, weekend miles in 0.1 mi, hard brakes, hard accelerations)."""
    return [1] + list(DAILY[secrets.randbelow(len(DAILY))])


# ------------------------------------------------------------------------------------------------ shared setup
def setup_parties():
    """Insurer key generation and signed key bundle, opening key of insurer and TSP (no dealer), TSP, one
    enrolled OBU; key confirmation as in the protocol."""
    from sp2ubi.entities import OBU, TSP, Insurer
    from sp2ubi.fhe import RiskContext
    from sp2ubi.groupsig_bbs04 import BBS04, Issuer
    from sp2ubi.opening_dkg import run_two_party
    w = model_weights()
    risk = RiskContext(P.p, P.m)
    ins = Insurer(P, risk, w, 1)
    bundle = ins.setup()
    gi = Issuer(run_two_party()[0])
    gs = BBS04(gi.gpk)
    mk = gi.enrol()
    tsp = TSP(gs, risk, bundle, ins.vk, 1)
    if not ins.key_confirmation(tsp):
        raise RuntimeError("key confirmation failed")
    tsp.close_setup()
    ident, K = ins.enroll(D_EPOCH)
    obu = OBU(ident, K, ins.obu_keys, ins.vk, risk, gs, mk, ID_INS, P, 1)
    for d in range(D_EPOCH):
        obu.add_day(d, list(DAILY[d % len(DAILY)]))
    return dict(w=w, risk=risk, ins=ins, tsp=tsp, obu=obu, gs=gs, ident=ident, K=K, bundle=bundle)


def fresh_message(obu, k):
    """A fresh record of interval k (the sealed state is cleared so every call seals, tags, encrypts and signs anew)."""
    obu._sealed.pop(k, None)
    return obu.record_message(k, D_EPOCH)


def reopen_epoch(tsp):
    """Reopen the TSP's benchmark epoch between benchmark records (memory only; not a protocol step). The TSP keeps no
    copy of a sent batch, so only the closed-epoch set is reset."""
    tsp.closed.clear()          # end_epoch closes the epoch; every benchmark record reuses epoch 1


# ------------------------------------------------------------------------------------------------ part: budget
def part_budget(a, res):
    import openfhe as o
    from sp2ubi.entities import verify_receipt, receipt_message
    from sp2ubi.fhe import SLOT_ID, SLOT_T, SLOT_T2, SLOT_U, SLOT_X
    from sp2ubi.proofs import proof_bytes, schnorr_sign
    S = setup_parties()
    risk, ins, tsp, obu, gs, w = S["risk"], S["ins"], S["tsp"], S["obu"], S["gs"], S["w"]
    R = tc.Recorder()
    reps, warm = a.reps, a.warmup
    deser = getattr(o, "DeserializeCiphertextString", None)
    checks = 0
    for i in range(warm + reps):
        rec = i >= warm
        k = i % D_EPOCH
        t_chain = time.perf_counter()
        msg = R.time("obu_record_message_total", fresh_message, obu, k, record=rec)
        receipt = R.time("tsp_receive_total", tsp.receive, msg, record=rec)
        okr = R.time("obu_verify_receipt", verify_receipt, tsp.vk, msg, receipt, record=rec)
        batch = R.time("tsp_end_epoch_publish", tsp.end_epoch, record=rec)
        ct = batch[0]
        d = R.time("insurer_decrypt", ins.decrypt, ct, record=rec)
        okm = R.time("insurer_hommac_verify_both_blocks", hommac.verify_record, d[SLOT_X], d[SLOT_T], d[SLOT_T2],
                     d[SLOT_U], w, S["K"], 1, P.p, record=rec)
        if rec:
            R.add("chain_end_to_end_compute", (time.perf_counter() - t_chain) * 1e3)
        if not (receipt and okr and okm and d[SLOT_ID] == S["ident"]):
            raise RuntimeError("risk-path chain failed (receipt, HomMAC or identity)")
        checks += 1
        # components of the OBU and TSP totals, on the same record (outside the chain timing)
        if rec:
            rec_x = obu._sealed[k][0]
            x, u = rec_x["x"], rec_x["u"]
            T, T2 = R.time("obu_hommac_tags_both_blocks", hommac.tag_blocks, x, S["K"], 1, u, P.p)
            ctp = R.time("obu_bfv_encrypt", risk.encrypt_payload, S["bundle"].pk, x, T, u, S["ident"], T2)
            sb = R.time("obu_serialize_signed_bytes", msg.signed_bytes)
            R.time("obu_group_sign", gs.sign, obu.gs_key, sb)
            blob = R.time("obu_serialize_ciphertext_upload", o.Serialize, ctp, o.BINARY)
            if deser is not None:
                R.time("tsp_deserialize_ciphertext", deser, blob, o.BINARY)
            R.time("tsp_group_verify", gs.verify, sb, msg.signature)
            ev = R.time("tsp_evaluate", risk.evaluate, msg.ct_payload, tsp.ct_w)
            R.time("tsp_fill", risk.fill, ev, S["bundle"].pk)
            R.time("tsp_receipt_sign", lambda: proof_bytes(schnorr_sign(
                tsp._sk, tsp.vk, receipt_message(msg.digest(), msg.commitment, msg.binding, msg.epoch))))
        reopen_epoch(tsp)
    t = R.summary()
    med = {k: v["median_ms"] for k, v in t.items()}
    obu_keys = ["obu_record_message_total"]
    tsp_keys = ["tsp_receive_total", "tsp_end_epoch_publish"]
    ins_keys = ["insurer_decrypt", "insurer_hommac_verify_both_blocks"]
    res["per_record_budget_ms_sum_of_medians"] = {
        "obu": sum(med[k] for k in obu_keys + ["obu_verify_receipt"]),
        "tsp": sum(med[k] for k in tsp_keys), "insurer": sum(med[k] for k in ins_keys),
        "total": sum(med[k] for k in obu_keys + ["obu_verify_receipt"] + tsp_keys + ins_keys),
        "chain_end_to_end_median": med["chain_end_to_end_compute"],
        "note": "compute only, one thread, measurement CPU (not OBU hardware); channel time from the record size; the "
                "OBU and TSP totals include their components listed separately"}
    res["design_note"] = "single insurer key: the insurer decrypts each published result itself"
    res["record_ciphertext_B"] = len(o.Serialize(msg.ct_payload, o.BINARY))
    res["record_signed_message_B"] = len(msg.signed_bytes())
    res["chains_checked"] = checks
    res["risk_context"] = risk.info()
    res["timings"] = t


# ------------------------------------------------------------------------------------------------ part: throughput
def _worker(idx, n_records, batch, barrier, q):
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        S = setup_parties()
        tsp, obu = S["tsp"], S["obu"]
        pool = [fresh_message(obu, k) for k in range(min(8, n_records))]   # distinct signed records (OBU not timed)
        barrier.wait()
        t0 = time.monotonic()
        lat = []
        for r in range(n_records):
            s = time.perf_counter()
            if tsp.receive(pool[r % len(pool)]) is None:
                raise RuntimeError("TSP rejected a valid record")
            lat.append((time.perf_counter() - s) * 1e3)
            if (r + 1) % batch == 0 or r + 1 == n_records:
                tsp.end_epoch()
                reopen_epoch(tsp)
        t1 = time.monotonic()
        q.put((idx, t0, t1, lat, None))
    except Exception as e:                              # report, never hang the barrier
        try:
            barrier.abort()
        except Exception:
            pass
        q.put((idx, 0.0, 0.0, [], repr(e)))


def part_throughput(a, res):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    os.environ["OMP_NUM_THREADS"] = "1"                 # inherited by every spawned worker before it imports OpenFHE
    out = {}
    for W in a.workers:
        runs = []
        for rep in range(a.repeats):
            barrier, q = ctx.Barrier(W), ctx.Queue()
            procs = [ctx.Process(target=_worker, args=(i, a.records, a.batch, barrier, q)) for i in range(W)]
            for p_ in procs:
                p_.start()
            got = [q.get() for _ in range(W)]
            for p_ in procs:
                p_.join()
            errs = [g[4] for g in got if g[4]]
            if errs:
                raise RuntimeError(f"worker failed (W={W}): {errs[0]}")
            start, end = min(g[1] for g in got), max(g[2] for g in got)
            lat = sorted(x for g in got for x in g[3])
            runs.append({"records_total": W * a.records, "wall_s": end - start,
                         "records_per_s": W * a.records / (end - start),
                         "per_worker_s": [g[2] - g[1] for g in sorted(got)],
                         "tsp_receive_latency": tc.summarize(lat)})
            print(f"W={W} rep={rep}: {runs[-1]['records_per_s']:.2f} records/s", flush=True)
        rps = sorted(r["records_per_s"] for r in runs)
        out[str(W)] = {"runs": runs, "records_per_s_median": rps[len(rps) // 2], "records_per_s_min": rps[0],
                       "records_per_s_max": rps[-1]}
    res["throughput"] = out
    res["design"] = {"workers": a.workers, "records_per_worker": a.records, "repeats": a.repeats, "batch": a.batch,
                     "omp_threads_per_worker": 1, "tenant_per_worker": True,
                     "timed_work": "TSP.receive (verify, evaluate, fill, receipt) + end_epoch"}


# ------------------------------------------------------------------------------------------------ part: baselines
def _bfv_single_key():
    import openfhe as o
    prm = o.CCParamsBFVRNS()
    prm.SetPlaintextModulus(P.p)
    prm.SetMultiplicativeDepth(1)
    prm.SetSecurityLevel(o.HEStd_128_classic)
    cc = o.GenCryptoContext(prm)
    for f in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH, o.PKESchemeFeature.LEVELEDSHE):
        cc.Enable(f)
    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    cc.EvalAtIndexKeyGen(kp.secretKey, [1, 2, 4])
    return cc, kp


def part_baselines(a, res):
    import openfhe as o
    from sp2ubi.fhe import ROTATIONS, SLOTS, SLOT_ID, SLOT_T, SLOT_T2, SLOT_U, SLOT_X, RiskContext
    w = model_weights()
    K = secrets.token_bytes(32)
    ident = 4242
    reps, warm = a.reps, a.warmup
    out = {}

    def layout(x, T, T2, u):
        v = [0] * SLOTS
        v[SLOT_X:SLOT_X + len(x)], v[SLOT_T:SLOT_T + len(T)], v[SLOT_T2:SLOT_T2 + len(T2)] = x, T, T2
        v[SLOT_U], v[SLOT_ID] = u, ident
        return v
    wv = [0] * SLOTS
    for s0 in (SLOT_X, SLOT_T, SLOT_T2):
        wv[s0:s0 + len(w)] = w
    wv[SLOT_U] = wv[SLOT_ID] = 1

    def new_record():
        x = sample_record(P.p)
        u = secrets.randbelow(P.p - 1) + 1
        T, T2 = hommac.tag_blocks(x, K, 1, u, P.p)
        return x, u, layout(x, T, T2, u)

    # B1: single-key BFV, encrypted weights
    cc, kp = _bfv_single_key()
    ctw = cc.Encrypt(kp.publicKey, cc.MakePackedPlaintext(wv))
    R = tc.Recorder()
    for i in range(warm + reps):
        rec = i >= warm
        x, u, v = new_record()
        ct = R.time("obu_encrypt", lambda: cc.Encrypt(kp.publicKey, cc.MakePackedPlaintext(v)), record=rec)

        def ev():
            c = cc.EvalMult(ct, ctw)
            for k in ROTATIONS:
                c = cc.EvalAdd(c, cc.EvalRotate(c, k))
            return c
        r = R.time("tsp_evaluate", ev, record=rec)

        def dec():
            pt = cc.Decrypt(kp.secretKey, r)
            pt.SetLength(SLOTS)
            return [int(z) % P.p for z in pt.GetPackedValue()]
        d = R.time("insurer_decrypt", dec, record=rec)
        ok = R.time("insurer_hommac_verify_both_blocks", hommac.verify_record, d[SLOT_X], d[SLOT_T], d[SLOT_T2],
                    d[SLOT_U], w, K, 1, P.p, record=rec)
        if not ok or d[SLOT_X] != sum(p_ * q_ for p_, q_ in zip(w, x)) % P.p:
            raise RuntimeError("B1 check failed")
    out["B1_single_key_bfv_encrypted_weights"] = {"ring_dimension": cc.GetRingDimension(),
                                                  "ciphertext_B": len(o.Serialize(ct, o.BINARY)),
                                                  "timings": R.summary()}

    # B2: the SP2UBI context (one insurer key), clear weights at the TSP
    risk = RiskContext(P.p, P.m)
    kp2 = risk.keygen()
    ptw = risk.cc.MakePackedPlaintext([z % P.p for z in wv])
    R = tc.Recorder()
    for i in range(warm + reps):
        rec = i >= warm
        x, u, v = new_record()
        ct = risk.cc.Encrypt(kp2.publicKey, risk.cc.MakePackedPlaintext(v))

        def ev_cw():
            c = risk.cc.EvalMult(ct, ptw)
            for k in ROTATIONS:
                c = risk.cc.EvalAdd(c, risk.cc.EvalRotate(c, k))
            return c
        r = R.time("tsp_evaluate_clear_weights", ev_cw, record=rec)
        r = R.time("tsp_fill", risk.fill, r, kp2.publicKey, record=rec)
        d = risk.decrypt(r, kp2.secretKey)
        if not hommac.verify_record(d[SLOT_X], d[SLOT_T], d[SLOT_T2], d[SLOT_U], w, K, 1, P.p):
            raise RuntimeError("B2 check failed")
    out["B2_sp2ubi_context_clear_weights"] = {"ring_dimension": risk.N, "timings": R.summary()}

    # B3: CKKS, encrypted weights, single key (inputs scaled to [0, 1): x / 2^x_bits, w / 2^w_bits)
    prm = o.CCParamsCKKSRNS()
    prm.SetMultiplicativeDepth(1)
    prm.SetScalingModSize(50)
    prm.SetFirstModSize(60)
    prm.SetBatchSize(64)
    prm.SetSecurityLevel(o.HEStd_128_classic)
    ck = o.GenCryptoContext(prm)
    for f in (o.PKESchemeFeature.PKE, o.PKESchemeFeature.KEYSWITCH, o.PKESchemeFeature.LEVELEDSHE):
        ck.Enable(f)
    kpc = ck.KeyGen()
    ck.EvalMultKeyGen(kpc.secretKey)
    ck.EvalAtIndexKeyGen(kpc.secretKey, [1, 2, 4])
    sx, sw = float(1 << P.x_bits), float(1 << P.w_bits)
    cwc = ck.Encrypt(kpc.publicKey, ck.MakeCKKSPackedPlaintext([z / sw for z in w] + [0.0] * (8 - len(w))))
    R = tc.Recorder()
    max_rel = 0.0
    for i in range(warm + reps):
        rec = i >= warm
        x = sample_record(P.p)
        ctx_ = R.time("obu_encrypt", lambda: ck.Encrypt(kpc.publicKey, ck.MakeCKKSPackedPlaintext(
            [z / sx for z in x] + [0.0] * (8 - len(x)))), record=rec)

        def evc():
            c = ck.EvalMult(ctx_, cwc)
            for k in ROTATIONS:
                c = ck.EvalAdd(c, ck.EvalRotate(c, k))
            return c
        r = R.time("tsp_evaluate", evc, record=rec)

        def decc():
            pt = ck.Decrypt(kpc.secretKey, r)
            pt.SetLength(1)
            return pt.GetRealPackedValue()[0] * sx * sw
        s_hat = R.time("insurer_decrypt", decc, record=rec)
        exact = sum(p_ * q_ for p_, q_ in zip(w, x))
        max_rel = max(max_rel, abs(s_hat - exact) / max(1, exact))
    out["B3_ckks_encrypted_weights_score_only"] = {"ring_dimension": ck.GetRingDimension(),
                                                   "ciphertext_B": len(o.Serialize(ctx_, o.BINARY)),
                                                   "max_relative_error": max_rel,
                                                   "hommac": "not applicable (approximate arithmetic)",
                                                   "timings": R.summary()}
    res["baselines"] = out


# ------------------------------------------------------------------------------------------------ part: paillier
def part_paillier(a, res):
    from phe import paillier
    import phe
    w = model_weights()
    K = secrets.token_bytes(32)
    ident = 4242
    reps, warm = a.reps, a.warmup
    R = tc.Recorder()
    pk, sk = R.time("insurer_keygen_once", paillier.generate_paillier_keypair, n_length=a.paillier_bits)
    for i in range(warm + reps):
        rec = i >= warm
        x = sample_record(P.p)
        u = secrets.randbelow(P.p - 1) + 1
        T, T2 = hommac.tag_blocks(x, K, 1, u, P.p)
        vals = x + T + T2 + [u, ident]
        cts = R.time("obu_encrypt_20_values", lambda: [pk.encrypt(v) for v in vals], record=rec)

        def ev():
            m = len(x)
            dot = (lambda off: sum((cts[off + j] * w[j] for j in range(1, m)), cts[off] * w[0]))
            return [dot(0), dot(m), dot(2 * m), cts[3 * m], cts[3 * m + 1]]
        outc = R.time("tsp_evaluate_clear_weights", ev, record=rec)
        d = R.time("insurer_decrypt_5_values", lambda: [sk.decrypt(c) for c in outc], record=rec)
        ok = R.time("insurer_hommac_verify_both_blocks", hommac.verify_record, d[0] % P.p, d[1] % P.p, d[2] % P.p,
                    d[3], w, K, 1, P.p, record=rec)
        if not ok or d[4] != ident:
            raise RuntimeError("Paillier check failed")
    nsq = (pk.nsquare.bit_length() + 7) // 8
    res["baselines"] = {"B4_paillier_clear_weights": {
        "library": f"python-phe {getattr(phe, '__version__', '?')}", "modulus_bits": a.paillier_bits,
        "ciphertext_one_value_B": nsq, "record_upload_B": 20 * nsq, "timings": R.summary()}}
    try:
        import gmpy2
        res["environment"]["gmpy2"] = gmpy2.version()
    except ImportError:
        res["environment"]["gmpy2"] = "MISSING (python-phe falls back to slow Python integers)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["budget", "throughput", "baselines", "paillier"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--records", type=int, default=200, help="records per worker (throughput)")
    ap.add_argument("--repeats", type=int, default=3, help="repetitions per worker count (throughput)")
    ap.add_argument("--batch", type=int, default=25, help="records between end_epoch calls (throughput)")
    ap.add_argument("--paillier-bits", type=int, default=3072)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-load", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.smoke:
        a.reps, a.warmup, a.records, a.repeats, a.batch = 3, 1, 4, 1, 2
        a.workers = a.workers if a.workers != [1, 2, 3, 4, 5, 6, 7, 8] else [1, 2]
    if a.part in ("budget", "baselines") and os.environ.get("OMP_NUM_THREADS") != "1":
        print("warning: OMP_NUM_THREADS is not 1; the budget and baselines are defined single-threaded")
    extra = {"openfhe": "1.5.1.0.24.4 (requirements/core.lock)"} if a.part != "paillier" else {}
    res = tc.header(f"risk_path_{a.part}", SCRIPTS, a.smoke, a.force, a.max_load, extra)
    res.update({"reps": a.reps, "warmup": a.warmup, "params": {"p_bits": P.p.bit_length(), "m": P.m,
                                                                 "interval_days": P.interval_days}})
    {"budget": part_budget, "throughput": part_throughput, "baselines": part_baselines,
     "paillier": part_paillier}[a.part](a, res)
    tc.write_json(a.out, tc.finish(res))
    print("status:", res["status"], "| written", a.out)


if __name__ == "__main__":
    main()
