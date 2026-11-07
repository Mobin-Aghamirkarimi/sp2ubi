"""Computation and message costs of the speed-verification path (single thread).

Measures, with the reference implementation (py_arkworks_bls12381, BLS12-381 G1):
    OBU/TEE   Enc_SV(v_sen) per beacon; burst certificate (BBS04 signature) per 40 s burst; rolling trailer
              (ephemeral Schnorr signature) per second
    RSU       crossing acceptance (certificate verify + rolling-signature verifies + digest/freshness checks);
              Enc(B) evaluation for K beacons; entry signature (RSU group, BBS04)
    TSP       entry-signature verification; shuffle re-randomisation per entry; partials with DLEQ proofs per entry
    Insurer   proof checks + threshold opening + threshold decryption + BSGS
    Setup     DKG proof of knowledge
and the beacon / certificate / rolling-trailer / entry sizes. Run on an idle machine with the CPU governor set to
`performance`; the script does not mark the result final if the 1-minute load average exceeds --max-load (override
with --force).

Usage (from the repository root, environment requirements/speed.lock; run_timings.sh does this):
  .venv-speed/bin/python experiments/timings/bench_speed_path.py [--reps 200] [--smoke] [--out OUT.json]
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from sp2ubi.elgamal import ThresholdKey, joint_public_key  # noqa: E402
from sp2ubi.groupsig_bbs04 import BBS04, Issuer  # noqa: E402
from sp2ubi.opening_dkg import run_two_party  # noqa: E402
from sp2ubi.speedcheck import (OBU, InsurerSpeedVerifier, KeyHolder, SensingRSU, TSPShuffle,  # noqa: E402
                               speed_to_units, verify_entry)
from sp2ubi.proofs import schnorr_sign  # noqa: E402


def timed(fn, reps, warmup=5):
    for _ in range(min(warmup, reps)):          # discarded warm-up runs
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); out.append((time.perf_counter() - t0) * 1e3)
    s = sorted(out)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0], statistics.median(s), s[-1]]
    return dict(median_ms=statistics.median(s), p95_ms=s[int(0.95 * (len(s) - 1))], reps=reps,
                warmup=min(warmup, reps), q1_ms=q[0], q3_ms=q[-1], mean_ms=statistics.fmean(s),
                std_ms=statistics.stdev(s) if len(s) > 1 else 0.0, min_ms=s[0], max_ms=s[-1],
                raw_ms=[round(x, 4) for x in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-load", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None, help="result file (default: experiments/timings/out/speed_path_<status>.json)")
    a = ap.parse_args()
    reps = 5 if a.smoke else a.reps
    load = os.getloadavg()[0]
    gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip() if os.path.exists(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") else "unknown"
    final = not a.smoke and (a.force or (load <= a.max_load and gov == "performance"))
    k_ins, k_tsp = ThresholdKey(34, 20, b"insurer"), ThresholdKey(34, 20, b"tsp")
    pk = joint_public_key(k_ins, k_tsp)
    tracing, sh_ins, sh_tsp = run_two_party()   # opening key: insurer and TSP, no dealer (setup not timed)
    issuer = Issuer(tracing); gs = BBS04(issuer.gpk)
    mk = issuer.enrol()
    rsu_issuer = Issuer(); rsu_gs = BBS04(rsu_issuer.gpk); rsu_key = rsu_issuer.enrol()
    ins = InsurerSpeedVerifier(k_ins, sh_ins, {mk.A.to_compressed_bytes(): "policy"}, (k_tsp.share_pub, sh_tsp.Y))
    tsp = KeyHolder(k_tsp, sh_tsp)          # TSP partials incl. proofs; the membership check (set lookup) is not timed
    obu = OBU(gs, mk, pk, phase_ms=0)
    t0 = time.perf_counter(); ct0 = pk.encrypt(0); k_ins.combine(ct0, k_tsp.partial(ct0)); table_s = time.perf_counter() - t0
    res = dict(load_1min=load, governor=gov, final=final, cpu=platform.processor() or platform.machine(),
               python=platform.python_version(), bsgs_table_build_s=table_s)
    res["tee_encrypt_per_beacon"] = timed(lambda: pk.encrypt(200), reps)
    from sp2ubi.groupsig_bbs04 import rand_scalar
    cert_msg = b"x" * 90
    res["tee_burst_certificate_bbs04_sign"] = timed(lambda: gs.sign(mk, cert_msg), max(reps // 4, 3))
    e = rand_scalar(); E = pk.G * e
    res["tee_rolling_trailer_schnorr_sign"] = timed(lambda: schnorr_sign(e, E, b"y" * 200), reps)
    res["tee_trailer_smudging_term_encrypt"] = timed(lambda: pk.encrypt(123_456), reps)   # one per trailer
    for K in (40, 80, 160):                                       # beacons in the crossing window (4 / 8 / 16 s)
        v = [speed_to_units(20.0)] * K
        t = [10_000 + 100 * i for i in range(K)]
        rec = obu.drive(v, t)[0]
        window = (t[0], t[-1])
        mk_rsu = lambda: SensingRSU(gs, pk, rsu_gs, rsu_key)
        res[f"rsu_accept_crossing_K{K}"] = timed(lambda: mk_rsu().accept(rec, t, window), max(reps // 4, 3))
        rsu = mk_rsu(); ok, ct_s = rsu.accept(rec, t, window)
        res[f"rsu_evaluate_K{K}"] = timed(lambda: rsu.evaluate(ok, rec.cert, v, ct_s), max(reps // 4, 3))
        unsigned = rsu.evaluate(ok, rec.cert, v, ct_s)
        # release: re-randomise every component, group-sign, forward, erase
        res[f"rsu_release_rerandomise_and_sign_K{K}"] = timed(lambda: rsu._finalize(unsigned), max(reps // 4, 3))
        entry = rsu._finalize(unsigned)
        air = K * rec.beacons[0].size() + sum(r.size() for r in rec.rollings) + (K // 10) * rec.cert.size()
        res[f"sizes_K{K}"] = dict(beacon_B=rec.beacons[0].size(), certificate_B=rec.cert.size(),
                                  rolling_B=rec.rollings[0].size(), entry_B=len(entry.body()) + len(entry.rsu_sig),
                                  air_bytes_per_crossing=air)
    res["tsp_verify_entry"] = timed(lambda: verify_entry(rsu_gs, entry), max(reps // 4, 3))
    shuf = TSPShuffle(gs, pk)
    res["tsp_shuffle_rerandomize_per_entry"] = timed(lambda: shuf.shuffle([entry]), reps)
    out = shuf.shuffle([entry])[0]
    res["tsp_partials_with_proofs_per_entry"] = timed(lambda: tsp.partials(out), reps)
    P = tsp.partials(out)
    res["insurer_verify_open_and_decrypt_per_entry"] = timed(lambda: ins.open_entry(out, P), max(reps // 4, 3))
    res["dkg_pok_and_check"] = timed(lambda: (ThresholdKey(34, 20, b"x").pok(b"s")), max(reps // 10, 3))
    from sp2ubi.speedcheck import S_BITS
    for b in (0.0, 0.01, 0.03, 0.10):
        ct = pk.encrypt(round(b * (1 << S_BITS)))
        part = k_tsp.partial(ct)
        res[f"insurer_decrypt_b{b:+.2f}"] = timed(lambda: k_ins.combine(ct, part), max(reps // 10, 3))
    tag = "final" if final else ("smoke" if a.smoke else "provisional")
    out = a.out or os.path.join(HERE, "out", f"speed_path_{tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not final and not a.smoke:
        print(f"not final (load {load:.2f}, governor {gov}); rerun idle with governor performance")


if __name__ == "__main__":
    main()
