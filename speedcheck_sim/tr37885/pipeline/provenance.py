"""Provenance records and run settings. Every stage-1/2/3 output carries `provenance`: the SHA-256 of the constants
file and of the sources, per-stage source hashes, tool versions and the run knobs. Tags: 'primary', 'sens_*' (the
sensitivity runs) and 'dev_*' (development runs). `rng_for(*key)` gives one bootstrap generator per key, independent of
call order."""
import hashlib
import json
import os
import subprocess
import sys

ER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTANTS = os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study1_constants.json"))
C = json.load(open(CONSTANTS))
# Per-stage source sets: a stage-k record depends on the sources of stages <= k only. Stage 1 = the channel generator
# path (stage-1 records), stage 2 = the estimator path (stage-2 records, snapshot placement), stage 3 = criteria,
# coverage check, decision adapter and reports.
STAGE_SOURCES = {
    1: ["interface.py", "antenna.py", "nlosv_blockage.py", "sumo_kinematics.py", "scatterer_mobility.py",
        "quadriga/driver.py", "quadriga/site.m", "quadriga/checkpoint.m", "quadriga/tr37885_element.m",
        "pipeline/stage1_channels.py", "quadriga/patch/get_channels.diff", "pipeline/null_test.py", "pipeline/provenance.py"],
    2: ["pipeline/stage2_evaluate.py", "huber_fit.py", "pipeline/snapshot_placement.py", "pipeline/coverage_sim.py"],
    3: ["pipeline/stage3_criteria.py", "pipeline/stage3_extra_checks.py", "pipeline/coverage_check.py",
        "pipeline/decision_adapter.py", "pipeline/consistency_report.py", "reports/make_reports.py",
        "reports/number_macros.py", "reports/select_texts.py", "pipeline/pilot_decision.py", "pipeline/inhouse_sweeps.py",
        "../inhouse/decisions_base.py", "../inhouse/decisions_levels.py", "../inhouse/decisions_monitor.py",
        "../inhouse/decisions.py"],
}
# Study-2 modules of each stage.
S2_STAGE_SOURCES = {1: ["pipeline/s2_constants.py", "pipeline/s2_dev3.py", "run_study2.sh"],
                    2: ["pipeline/s2_fields.py", "pipeline/s2_evaluate.py", "../fdoa/estimator.py"],
                    3: ["pipeline/s2_criteria.py", "pipeline/s2_selection.py", "pipeline/s2_selection_run.py",
                        "pipeline/s2_calibrate.py", "pipeline/s2_seeds.py"]}
SOURCES = sorted({f for fs in STAGE_SOURCES.values() for f in fs} | {f for fs in S2_STAGE_SOURCES.values() for f in fs}
                 | {"run_study1.sh", "pipeline/s2_heldout.py", "pipeline/s2_stage3.py", "reports/s2_texts.py"})
ENV_KNOBS = ("SPEEDCHECK_BOOT", "SPEEDCHECK_RUNS", "SPEEDCHECK_RESULTS", "SPEEDCHECK_ADAPTER_OUT")
QUADRIGA_DIR = os.environ.get("QUADRIGA_PATCHED_DIR", os.path.join(os.path.dirname(ER), C["quadriga"]["patched_path"]))


_TREE = {}


def quadriga_tree_sha256(root=None):
    """SHA-256 over the sorted relative paths and contents of the patched QuaDRiGa source tree (environment variable
    QUADRIGA_PATCHED_DIR); None if the tree is not installed. Cached per process."""
    root = root or os.path.join(QUADRIGA_DIR, "quadriga_src")
    if not os.path.isdir(root):
        return None
    if root not in _TREE:
        h = hashlib.sha256()
        for d, _, fs in sorted(os.walk(root)):
            for f in sorted(fs):
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).encode())
                h.update(open(p, "rb").read())
        _TREE[root] = h.hexdigest()
    return _TREE[root]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _versions():
    out = {}
    try:
        out["octave"] = subprocess.run(["octave", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:                                   # noqa: BLE001
        out["octave"] = None
    out["quadriga_commit"] = C["quadriga"]["commit"]
    for mod in ("numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                               # noqa: BLE001
            out[mod] = None
    out["sumo"] = C["sumo"]["version"]
    try:                                                # the installed binary, not the constant
        exe = os.path.join(os.path.dirname(sys.executable), "sumo")
        out["sumo_binary"] = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:                                   # noqa: BLE001
        out["sumo_binary"] = None
    return out


def _with_s2(k, files):
    """Stage k's sources plus the study-2 modules of that stage."""
    return files + S2_STAGE_SOURCES.get(k, [])


def stage_hashes():
    """SHA-256 per stage over that stage's sources (concatenated file hashes in list order)."""
    out = {}
    for k, files in ((k, _with_s2(k, f)) for k, f in STAGE_SOURCES.items()):
        h = hashlib.sha256()
        for f in files:
            p = os.path.join(ER, f)
            h.update(f.encode() + (sha256_file(p).encode() if os.path.exists(p) else b"absent"))
        out[str(k)] = h.hexdigest()
    return out


def provenance(tag, stage=None, **knobs):
    """stage: 1, 2 or 3 for the staged records (resume compares the sources of stages <= stage only)."""
    return dict(tag=tag, stage=stage, stage_sha256=stage_hashes(), constants_sha256=sha256_file(CONSTANTS),
                sources_sha256={s: sha256_file(os.path.join(ER, s)) for s in SOURCES if os.path.exists(os.path.join(ER, s))},
                versions=_versions(), quadriga_tree_sha256=quadriga_tree_sha256(), machine=_machine(), knobs=knobs)


def _machine():
    import platform
    try:
        mem = int(open("/proc/meminfo").read().split("MemTotal:")[1].split()[0]) // 1024 ** 2
    except Exception:                                   # noqa: BLE001
        mem = None
    return dict(os=platform.platform(), cpus=os.cpu_count(), mem_gb=mem)


def fixed_knobs():
    r = C["run"]
    return dict(Q=r["Q"], n_scale=r["n_scale"], boot=r["boot"], e2_vehicles=r["e2_vehicles"])


def check_tag(tag, stage=None, **knobs):
    """Tags: 'primary', 'sens_*' or 'dev_*'. A primary run uses the fixed knobs of constants `run`."""
    if tag.startswith("dev_"):
        return
    if tag != "primary" and not tag.startswith("sens_"):
        raise ValueError(f"tag must be 'primary' or start with dev_/sens_ (got {tag!r})")
    fk = fixed_knobs() if tag == "primary" else {}
    for k, v in knobs.items():
        if k in fk and v != fk[k]:
            raise ValueError(f"primary run: {k} = {v!r} differs from the fixed value {fk[k]!r}")


def same_provenance(a, b):
    """Resume check: the parts of the provenance that must not change between a stored output and the current run: the
    code, the constants, the knobs, the patched QuaDRiGa tree and the installed tool versions. The machine may differ."""
    if a.get("stage") and a.get("stage") == b.get("stage"):   # staged record: sources of stages <= stage only
        k = int(a["stage"])
        keys = ("constants_sha256", "knobs", "quadriga_tree_sha256", "versions")
        return (all(a.get(x) == b.get(x) for x in keys) and
                all((a.get("stage_sha256") or {}).get(str(j)) == (b.get("stage_sha256") or {}).get(str(j))
                    for j in range(1, k + 1)))
    keys = ("constants_sha256", "sources_sha256", "knobs", "quadriga_tree_sha256", "versions")
    return all(a.get(k) == b.get(k) for k in keys)


def same_stage_sources(a, b, upto):
    """The sources of stages 1..upto agree (a stage-2 run reading stage-1 records)."""
    return all((a.get("stage_sha256") or {}).get(str(j)) == (b.get("stage_sha256") or {}).get(str(j)) and
               (a.get("stage_sha256") or {}).get(str(j)) is not None for j in range(1, upto + 1))


def diff_provenance(a, b):
    """The keys on which two provenances differ (for the refusal message of a resume)."""
    keys = ("constants_sha256", "sources_sha256", "knobs", "quadriga_tree_sha256", "versions")
    out = {}
    for k in keys:
        if a.get(k) != b.get(k):
            if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
                out[k] = {x: (a[k].get(x), b[k].get(x)) for x in set(a[k]) | set(b[k]) if a[k].get(x) != b[k].get(x)}
            else:
                out[k] = (a.get(k), b.get(k))
    return out


def rng_for(*key):
    import numpy as np
    h = int.from_bytes(hashlib.sha256(json.dumps(key, default=str).encode()).digest()[:8], "little")
    return np.random.default_rng([C["seeds"]["bootstrap"], h])
