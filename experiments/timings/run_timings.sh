#!/usr/bin/env bash
# Runs every timing of the evaluation on this machine, one item after the other (never in parallel: each item is a
# measurement of a machine at rest, and L3 cache, memory bandwidth and turbo frequency are shared):
#   speed_path          bench_speed_path.py --reps 200                                      (.venv-speed)
#   risk_path_budget    bench_risk_path.py --part budget --reps 1000                        (.venv, OMP_NUM_THREADS=1)
#   baselines_bfv_ckks  bench_risk_path.py --part baselines --reps 1000                     (.venv, OMP_NUM_THREADS=1)
#   baseline_paillier   bench_risk_path.py --part paillier --reps 1000                      (.venv-paillier)
#   throughput          bench_risk_path.py --part throughput, 1..8 workers, 200 records, 3 repeats   (.venv)
#
# Usage (from anywhere; run setup_envs.sh first):
#   bash experiments/timings/run_timings.sh [--out DIR] [--smoke] [ITEM ...]
#     --out DIR   directory of the item files (default experiments/timings/out); each item writes DIR/<item>.json
#     --smoke     tiny repetition counts: a functional check whose files are marked "smoke" (not a measurement)
#     ITEM ...    a subset of the items above (default: all, in the order above)
#
# Before each item the script waits up to 15 minutes for a 1-minute load <= 0.5 and then applies the final rule of
# timing_common.final_rule (governor `performance` on every core, load <= 0.5 and, if TIMING_REQUIRED_CPU is set,
# that CPU model); an item whose rule fails is not started. An item is final only if the rule held at its start and
# the script reports a final result. The stored files in results/ were measured on an Intel Core i7-1195G7; to
# repeat them on that machine: TIMING_REQUIRED_CPU=i7-1195G7 bash experiments/timings/run_timings.sh --out DIR
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/out"
SMOKE=0
SEL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) SEL+=("$1"); shift ;;
  esac
done
ITEMS=(speed_path risk_path_budget baselines_bfv_ckks baseline_paillier throughput)
[ ${#SEL[@]} -gt 0 ] && ITEMS=("${SEL[@]}")

IMPL_PY="$ROOT/.venv/bin/python"
SPEED_PY="$ROOT/.venv-speed/bin/python"
PAILLIER_PY="$ROOT/.venv-paillier/bin/python"
REPS_SPEED=200
REPS_RISK=1000
TP_RECORDS=200; TP_REPEATS=3; TP_WORKERS="1 2 3 4 5 6 7 8"
SCRIPT_MAX_LOAD=0.75   # the scripts' own load guard at their start; the binding rule (<= 0.5) is checked here first
for py in "$IMPL_PY" "$SPEED_PY" "$PAILLIER_PY"; do
  [ -x "$py" ] || { echo "missing $py: run bash experiments/timings/setup_envs.sh first"; exit 1; }
done
mkdir -p "$OUT/raw"

item_def() {        # sets PY, ENVV, ARGS, SCRIPTS for one item
  local id=$1 sm=()
  [ $SMOKE = 1 ] && sm=(--smoke)
  ENVV=""
  case "$id" in
    speed_path) PY="$SPEED_PY"
      ARGS=("$HERE/bench_speed_path.py" --reps "$REPS_SPEED" --max-load "$SCRIPT_MAX_LOAD" "${sm[@]}" --out "$OUT/raw/$id.json")
      SCRIPTS="$HERE/bench_speed_path.py" ;;
    risk_path_budget) PY="$IMPL_PY"; ENVV="OMP_NUM_THREADS=1"
      ARGS=("$HERE/bench_risk_path.py" --part budget --reps "$REPS_RISK" --max-load "$SCRIPT_MAX_LOAD" "${sm[@]}" --out "$OUT/raw/$id.json")
      SCRIPTS="$HERE/bench_risk_path.py $HERE/timing_common.py" ;;
    baselines_bfv_ckks) PY="$IMPL_PY"; ENVV="OMP_NUM_THREADS=1"
      ARGS=("$HERE/bench_risk_path.py" --part baselines --reps "$REPS_RISK" --max-load "$SCRIPT_MAX_LOAD" "${sm[@]}" --out "$OUT/raw/$id.json")
      SCRIPTS="$HERE/bench_risk_path.py $HERE/timing_common.py" ;;
    baseline_paillier) PY="$PAILLIER_PY"
      ARGS=("$HERE/bench_risk_path.py" --part paillier --reps "$REPS_RISK" --max-load "$SCRIPT_MAX_LOAD" "${sm[@]}" --out "$OUT/raw/$id.json")
      SCRIPTS="$HERE/bench_risk_path.py $HERE/timing_common.py" ;;
    throughput) PY="$IMPL_PY"
      # shellcheck disable=SC2206
      local w=($TP_WORKERS); [ $SMOKE = 1 ] && w=(1 2)
      ARGS=("$HERE/bench_risk_path.py" --part throughput --workers "${w[@]}" --records "$TP_RECORDS" --repeats "$TP_REPEATS"
            --max-load "$SCRIPT_MAX_LOAD" "${sm[@]}" --out "$OUT/raw/$id.json")
      SCRIPTS="$HERE/bench_risk_path.py $HERE/timing_common.py" ;;
    *) echo "unknown item $id"; exit 1 ;;
  esac
}

wait_idle() {       # up to 15 minutes for a 1-minute load <= 0.5 (smoke runs do not wait)
  [ $SMOKE = 1 ] && return 0
  local i
  for i in $(seq 1 90); do
    awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l <= 0.5)}' && return 0
    sleep 10
  done
  echo "load still $(cut -d' ' -f1 /proc/loadavg) after 15 min"
}

run_item() {
  local id=$1 st rc before after rule rule_ok=1
  item_def "$id"
  wait_idle
  before=$("$IMPL_PY" "$HERE/timing_common.py" env)
  rule=$("$IMPL_PY" "$HERE/timing_common.py" rule "$before") || rule_ok=0
  if [ $rule_ok = 0 ] && [ $SMOKE = 0 ]; then
    echo "[$id] not started: the final rule fails ($rule)"
    return 2
  fi
  st=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$id] start $st | $ENVV ${PY#"$ROOT"/} ${ARGS[*]//$ROOT\//}"
  rm -f "$OUT/raw/$id.json" "$OUT/raw/$id.rss"
  ( cd "$ROOT" && env $ENVV "$IMPL_PY" "$HERE/timing_common.py" wrap "$OUT/raw/$id.rss" "$PY" "${ARGS[@]}" ) \
    > "$OUT/raw/$id.stdout" 2>&1; rc=$?
  after=$("$IMPL_PY" "$HERE/timing_common.py" env)
  tail -n 3 "$OUT/raw/$id.stdout"
  [ $rc = 0 ] && [ ! -f "$OUT/raw/$id.json" ] && { echo "[$id] failed: no output file"; rc=97; }
  "$IMPL_PY" - "$id" "$rc" "$st" "$OUT/raw/$id.json" "$before" "$after" "$OUT/$id.json" "$HERE" "$SCRIPTS" \
      "$ENVV ${ARGS[*]//$ROOT\//}" "$rule" "$SMOKE" "$OUT/raw/$id.rss" <<'EOF'
import json, os, sys
id_, rc, st, prod, before, after, out, here, scripts, cmd, rule, smoke, rssf = sys.argv[1:14]
sys.path.insert(0, here)
import timing_common as tc
res = json.load(open(prod)) if os.path.exists(prod) else None
if res is None:
    script_status = None
elif "status" in res:
    script_status = res["status"]
else:
    script_status = "final" if res.get("final") else "provisional"
rule = json.loads(rule)
if smoke == "1":
    eff = "smoke"
elif int(rc) != 0:
    eff = "failed"
elif script_status == "final" and rule["ok"]:
    eff = "final"
else:
    eff = "provisional"
cmd = cmd.replace(os.path.join(tc.ROOT, "experiments", "timings", "out") + "/", "")
doc = {"item": id_, "exit_code": int(rc), "start_utc": st, "end_utc": tc.utc_now(), "command": cmd,
       "code": tc.code_identity(scripts.split()), "machine_before": json.loads(before),
       "machine_after": json.loads(after),
       "peak_rss_kib_item_and_descendants": (open(rssf).read().strip() if os.path.exists(rssf) else "unknown"),
       "final_rule": {"rule": "timing_common.final_rule: governor performance on every core, 1-min load <= %s%s, "
                              "checked by run_timings.sh before the item; final = rule held AND script status final"
                              % (tc.MAX_LOAD, ", CPU " + tc.REQUIRED_CPU if tc.REQUIRED_CPU else ""),
                      "at_start": rule, "at_end_for_information": tc.final_rule(json.loads(after)),
                      "script_status": script_status},
       "result_status": eff, "result": res}
tc.write_json(out, doc)
print(f"[{id_}] {eff} -> {out}")
EOF
  [ $rc = 0 ] || echo "[$id] failed (exit $rc); see $OUT/raw/$id.stdout"
  return $rc
}

fail=0
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(sed -n 's/^model name[[:space:]]*: *//p' /proc/cpuinfo | head -1) ====="
for id in "${ITEMS[@]}"; do run_item "$id" || fail=1; done
exit $fail
