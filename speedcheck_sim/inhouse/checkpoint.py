"""Resumable job runner for the long simulations.

Every job has a stable name; its result is pickled to <ckdir>/<name>.pkl atomically (temporary file, fsync, rename) as
soon as the job finishes. A rerun skips every job whose file exists, so a crash or reboot costs at most the jobs that
were running at that moment. Jobs are small (one (cell, site) unit, minutes), seeded from their own name, so results
do not depend on execution order or on how often the run was interrupted. Workers run at the lowest CPU priority so
that the desktop stays responsive.
"""
import glob
import hashlib
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def job_seed(*parts) -> int:
    """Deterministic 63-bit seed from the job's identity."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") >> 1


def atomic_pickle(obj, path: str) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _low_priority():
    try:
        os.nice(19)
    except OSError:
        pass


def run_jobs(fn, jobs, ckdir: str, workers: int, log=print):
    """jobs: list of (name, args). Runs fn(args) for every job without a checkpoint, saves each result as it
    completes, and returns {name: result} for all jobs (loaded from the checkpoints)."""
    os.makedirs(ckdir, exist_ok=True)
    for t in glob.glob(os.path.join(ckdir, "*.tmp")):
        os.remove(t)                                          # leftovers of an interrupted write
    path = lambda n: os.path.join(ckdir, n + ".pkl")
    pending = [(n, a) for n, a in jobs if not os.path.exists(path(n))]
    log(f"[ckpt] {ckdir}: {len(jobs) - len(pending)} of {len(jobs)} jobs already done, {len(pending)} to run", flush=True)
    t0 = time.time()
    if pending:
        with ProcessPoolExecutor(workers, initializer=_low_priority) as ex:
            futs = {ex.submit(fn, a): n for n, a in pending}
            for k, f in enumerate(as_completed(futs), 1):
                n = futs[f]
                atomic_pickle(f.result(), path(n))
                el = time.time() - t0
                log(f"[ckpt] done {n} ({k}/{len(pending)}, {el / 60:.1f} min, ETA {el / k * (len(pending) - k) / 60:.0f} min)",
                    flush=True)
    out = {}
    for n, _ in jobs:
        with open(path(n), "rb") as f:
            out[n] = pickle.load(f)
    return out
