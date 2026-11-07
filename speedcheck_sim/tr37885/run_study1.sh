#!/usr/bin/env bash
# Study 1, full run: null test, coverage check, stage 1 (QuaDRiGa channels), snapshot placement, stage 2, stage 3,
# decision adapter and reports. Every stage is resumable (outputs are reused only with the same provenance).
# Needs Octave, the patched QuaDRiGa tree (QUADRIGA_PATCHED_DIR, see quadriga/patch/) and eclipse-sumo.
# Outputs: runs/ (stage-1 and stage-2 records) and ../results/ (stage-3 outputs and reports).
# Usage: bash run_study1.sh            (PY=<python> to choose the interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
TAG=primary; PTAG=sens_place; LTAG=sens_ls; CTAG=sens_coverage; ETAG=sens_erasure
SITES="0 10"
MINGB=$($PY -c "import json; print(json.load(open('config/study1_constants.json'))['run']['min_free_disk_gb'])")
need_disk() { local a; a=$(df -BG --output=avail . | tail -1 | tr -dc 0-9); [ "$a" -ge "$MINGB" ] || { echo "only ${a} GB free (< ${MINGB} GB) before $1"; exit 5; }; }

# 0. null test of the interface (500 passes at 10/20/50 PRB) and of the QuaDRiGa driver (100 passes at 10/50 PRB)
for PRB in 10 20 50; do $PY pipeline/null_test.py --tag $TAG --passes 500 --prb "$PRB"; done
for PRB in 10 50; do $PY pipeline/null_test.py --tag $TAG --arm g1 --passes 100 --prb "$PRB"; done
$PY - "$TAG" <<'PYEOF'
import glob, json, sys
fs = sorted(glob.glob(f"../results/c8_*_{sys.argv[1]}.json"))
assert len(fs) == 5, fs
sys.exit(0 if all(json.load(open(f))["C8_pass"] for f in fs) else 3)
PYEOF

# 1. coverage check (in-house channel only): fixes the widening factor of the optimism check before stage 3
$PY pipeline/coverage_check.py --tag $CTAG

# 2. stage 1, families H, U, N (one QuaDRiGa call per site; workers from constants run.workers_stage1)
for FAM in H U N; do need_disk "stage 1 $FAM"; $PY pipeline/stage1_channels.py --family "$FAM" --sites $SITES --tag $TAG; done

# 3. snapshot placement (5 sites x 20 passes, P1-15 and P2-38)
$PY pipeline/snapshot_placement.py --tag $PTAG

# 4. stage 2: primary; link-budget sensitivity on U and H; beacon erasures on U and H (iid and Gilbert-Elliott, 10 % and 30 %)
for FAM in H U N; do $PY pipeline/stage2_evaluate.py --family "$FAM" --sites $SITES --tag $TAG; done
for FAM in H U; do $PY pipeline/stage2_evaluate.py --family "$FAM" --sites $SITES --tag $LTAG --lb-gain-shift-db -1.42; done
for FAM in H U; do
  $PY pipeline/stage2_evaluate.py --family "$FAM" --sites $SITES --tag ${ETAG}_iid10 --erasure iid:0.1
  $PY pipeline/stage2_evaluate.py --family "$FAM" --sites $SITES --tag ${ETAG}_ge10 --erasure ge:0.1:3
  $PY pipeline/stage2_evaluate.py --family "$FAM" --sites $SITES --tag ${ETAG}_iid30 --erasure iid:0.3
  $PY pipeline/stage2_evaluate.py --family "$FAM" --sites $SITES --tag ${ETAG}_ge30 --erasure ge:0.3:3
done

# 5. stage 3 (optimism check, scope, headline, consistency, governing variant per stratum)
$PY pipeline/stage3_criteria.py --tag $TAG

# 6. decision simulation (fixed level grid; data-derived grid as the sensitivity row) and matching through the adapter
( cd ../inhouse && $PY ../tr37885/pipeline/decision_adapter.py e2 --tag $TAG )
( cd ../inhouse && $PY ../tr37885/pipeline/decision_adapter.py e2 --tag $TAG --grid data )
( cd ../inhouse && $PY ../tr37885/pipeline/decision_adapter.py e5 --tag $TAG )

# 7. in-house sweeps of Figs. C and E (descriptive)
$PY pipeline/inhouse_sweeps.py k4 --tag sens_inhouse
$PY pipeline/inhouse_sweeps.py fige --tag sens_inhouse

# 8. reports: Table A, Table B, figures, number and branch files
$PY reports/make_reports.py --tag $TAG
$PY reports/number_macros.py --tag $TAG
$PY reports/select_texts.py --tag $TAG
echo "$TAG run complete"
