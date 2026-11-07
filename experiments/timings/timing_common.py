"""Shared helpers of the timing scripts (experiments/timings/).

Statistics: warm-up runs are discarded, then every timed run is kept (no outlier removal); median, IQR, mean, standard
deviation, min, max, p95 and the raw samples are stored. One `final` rule holds for every item (`final_rule` below):
the CPU governor `performance` on every core, a 1-minute load average <= MAX_LOAD (0.5) at the start and, if the
environment variable TIMING_REQUIRED_CPU is set, a CPU model name containing it. The scripts apply it in `header`;
run_timings.sh applies the same function before every item and records it in the item's result file. A result that
fails the rule is `provisional` (or `smoke` for a functional check) and is not a measurement.

Command line (used by run_timings.sh): `python timing_common.py env` prints environment() as JSON;
`python timing_common.py rule '<environment JSON>'` prints the final_rule verdict as JSON (exit 0 iff it holds);
`python timing_common.py code SCRIPT...` prints the code identity of the scripts and the implementation as JSON.
"""
import glob
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REQUIRED_CPU = os.environ.get("TIMING_REQUIRED_CPU", "")
MAX_LOAD = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def governors() -> list:
    out = set()
    for f in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"):
        try:
            out.add(open(f).read().strip())
        except OSError:
            pass
    return sorted(out) or ["unknown"]


def no_turbo():
    try:
        return open("/sys/devices/system/cpu/intel_pstate/no_turbo").read().strip()
    except OSError:
        return "unknown"


def on_ac():
    """True / False when a mains supply reports its state, None when not detectable."""
    seen = None
    for d in glob.glob("/sys/class/power_supply/*"):
        try:
            if open(os.path.join(d, "type")).read().strip() == "Mains":
                seen = (seen or False) or open(os.path.join(d, "online")).read().strip() == "1"
        except OSError:
            pass
    return seen


def load1() -> float:
    return os.getloadavg()[0]


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def code_identity(script_paths) -> dict:
    """SHA-256 of each timed script and of the implementation (every file of sp2ubi/ and models/, in path order), so
    a result file names the exact code that produced it."""
    impl = sorted(glob.glob(os.path.join(ROOT, "sp2ubi", "**", "*.py"), recursive=True)
                  + glob.glob(os.path.join(ROOT, "models", "*.json")))
    h = hashlib.sha256()
    for p in impl:
        h.update(os.path.relpath(p, ROOT).encode() + b"\0" + bytes.fromhex(sha256_file(p)))
    return {"implementation_sha256": h.hexdigest(), "implementation_files": len(impl),
            "scripts": {os.path.relpath(p, ROOT): sha256_file(p) for p in script_paths}}


def cpu_freq_and_temp() -> dict:
    """CPU frequencies (kHz, distinct values over all cores) and thermal-zone temperatures (milli-degC) at this moment,
    so that a throttling question can be answered afterwards. Never raises."""
    out = {}
    try:
        out["cur_freq_khz"] = sorted({open(f).read().strip() for f in glob.glob(
            "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")})
    except OSError:
        out["cur_freq_khz"] = "unknown"
    try:
        out["thermal_mC"] = [open(f).read().strip() for f in glob.glob("/sys/class/thermal/thermal_zone*/temp")]
    except OSError:
        out["thermal_mC"] = "unknown"
    return out


def peak_rss_kib() -> dict:
    """Peak resident set size in KiB (Linux ru_maxrss) of this process and of its waited-for children."""
    try:
        import resource
        return {"self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    except Exception:
        return {"self": "unknown", "children": "unknown"}


def environment(extra=None) -> dict:
    env = {"cpu": cpu_model(), "governors": governors(), "intel_no_turbo": no_turbo(), "on_ac": on_ac(),
           "load_1min": load1(), "nproc": os.cpu_count(), "python": platform.python_version(),
           "kernel": platform.release(),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "default"),
           "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", "default")}
    env.update(cpu_freq_and_temp())
    env.update(extra or {})
    return env


def final_rule(env: dict, max_load: float = MAX_LOAD) -> dict:
    """The one `final` rule, evaluated on an environment() record: governor `performance` on every core, 1-minute
    load <= max_load and, if TIMING_REQUIRED_CPU is set, that CPU model. Returns {"ok": bool, "reasons": [...],
    "max_load": ...}."""
    reasons = []
    if REQUIRED_CPU and REQUIRED_CPU not in str(env.get("cpu", "")):
        reasons.append(f"CPU is not the {REQUIRED_CPU}")
    if env.get("governors") != ["performance"]:
        reasons.append("governor not 'performance' on every core: " + ",".join(map(str, env.get("governors") or [])))
    load = env.get("load_1min")
    if not isinstance(load, (int, float)) or load > max_load:
        reasons.append(f"1-min load {load} > {max_load}")
    return {"ok": not reasons, "reasons": reasons, "max_load": max_load}


def is_final(smoke: bool, force: bool, max_load: float, load: float) -> bool:
    if smoke:
        return False
    if force:
        return True
    return final_rule({"cpu": cpu_model(), "governors": governors(), "load_1min": load}, max_load)["ok"]


def summarize(samples) -> dict:
    s = sorted(samples)
    n = len(s)
    q = statistics.quantiles(s, n=4) if n >= 4 else [s[0], statistics.median(s), s[-1]]
    return {"n": n, "median_ms": statistics.median(s), "q1_ms": q[0], "q3_ms": q[-1],
            "mean_ms": statistics.fmean(s), "std_ms": statistics.stdev(s) if n > 1 else 0.0,
            "min_ms": s[0], "max_ms": s[-1], "p95_ms": s[min(n - 1, int(round(0.95 * (n - 1))))]}


class Recorder:
    """Times labelled operations; keeps raw samples (ms) per label."""

    def __init__(self):
        self.samples = {}

    def time(self, label, fn, *a, record=True, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        if record:
            self.samples.setdefault(label, []).append((time.perf_counter() - t0) * 1e3)
        return out

    def repeat(self, label, fn, reps, warmup):
        for i in range(warmup + reps):
            self.time(label, fn, record=i >= warmup)

    def add(self, label, ms):
        self.samples.setdefault(label, []).append(ms)

    def summary(self, raw=True) -> dict:
        out = {}
        for k, v in self.samples.items():
            out[k] = summarize(v)
            if raw:
                out[k]["raw_ms"] = [round(x, 4) for x in v]
        return out


def write_json(path, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def header(name, script_paths, smoke, force, max_load, extra_env=None) -> dict:
    load = load1()
    final = is_final(smoke, force, max_load, load)
    return {"benchmark": name, "status": "final" if final else ("smoke" if smoke else "provisional"),
            "final": final, "smoke": smoke, "forced": bool(force and not smoke), "start_utc": utc_now(),
            "environment": environment(extra_env), "code": code_identity(script_paths),
            "argv": sys.argv[1:]}


def finish(res: dict) -> dict:
    res["end_utc"] = utc_now()
    res["load_1min_at_end"] = load1()
    res["peak_rss_kib"] = peak_rss_kib()
    res.update({"at_end_" + k: v for k, v in cpu_freq_and_temp().items()})
    if not res["final"] and not res["smoke"]:
        res["note"] = ("not final: needs the governor performance on every core, an idle machine and, if set, the CPU "
                       "named by TIMING_REQUIRED_CPU; these numbers are not a measurement")
    if res["smoke"]:
        res["note"] = "smoke run: checks that the script runs end to end; these numbers are not a measurement"
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["env"]:
        print(json.dumps(environment()))
    elif sys.argv[1:2] == ["rule"] and len(sys.argv) == 3:
        v = final_rule(json.loads(sys.argv[2]))
        print(json.dumps(v))
        sys.exit(0 if v["ok"] else 1)
    elif sys.argv[1:2] == ["code"]:
        print(json.dumps(code_identity(sys.argv[2:])))
    elif sys.argv[1:2] == ["wrap"] and len(sys.argv) >= 4:
        # wrap RSSFILE CMD...: run CMD, then write the peak RSS (KiB) of CMD and its descendants to RSSFILE
        rc = subprocess.call(sys.argv[3:])
        try:
            open(sys.argv[2], "w").write(str(peak_rss_kib()["children"]))
        except OSError:
            pass
        sys.exit(rc)
    else:
        sys.exit("usage: timing_common.py env | rule '<environment JSON>' | code SCRIPT... | wrap RSSFILE CMD...")
