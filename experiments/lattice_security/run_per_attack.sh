#!/usr/bin/env bash
# Runs every attack of estimate_per_attack.sage.py for the risk-path parameters with a per-job time limit (LIMIT
# seconds, default 5400), 3 jobs at a time, nice 10. Rerun = resume (finished jobs are skipped). A timed-out job is
# recorded as such.
# Environment: SAGE (the sage executable) and LATTICE_ESTIMATOR (a checkout of malb/lattice-estimator, commit
# 53da5982597709ba0fdf94ea37a84d822310fd84).
set -u
cd "$(dirname "$0")"
: "${SAGE:?set SAGE to the sage executable}"
: "${LATTICE_ESTIMATOR:?set LATTICE_ESTIMATOR to the lattice-estimator checkout}"
export SAGE LATTICE_ESTIMATOR
export PYTHONPATH="$LATTICE_ESTIMATOR"
export LIMIT=${LIMIT:-5400}
mkdir -p results/per_attack
for c in single_key_ternary; do
  for a in usvp bdd dual dual_hybrid bkw arora-gb bdd_hybrid bdd_mitm_hybrid; do
    [ -f "results/per_attack/${c}__${a}.json" ] || echo "$c $a"
  done
done | xargs -P 3 -L 1 bash -c '
  if ! timeout "$LIMIT" nice -n 10 "$SAGE" -python estimate_per_attack.sage.py "$0" "$1" > "results/per_attack/$0__$1.log" 2>&1; then
    [ -f "results/per_attack/$0__$1.json" ] || echo "{\"case\": \"$0\", \"attack\": \"$1\", \"log2_rop\": null, \"error\": \"timeout ($LIMIT s) or failure\"}" > "results/per_attack/$0__$1.json"
  fi'
echo done
