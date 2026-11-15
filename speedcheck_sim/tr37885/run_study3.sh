#!/usr/bin/env bash
# Study 3, full run: the urban estimator (fdoa/urban.py) against the base estimator on the ten newly simulated urban
# LOS sites U3 30-39 (survey in traffic and free-road survey, TR 37.885 and our channel model), the snapshot-placement
# factor on the sites 40-44, and the detection rules with the urban estimator (five decision-level runs).
# Run after run_study1.sh: the decision-level runs read the study-1 stage-2 records runs/g1_H/eval_primary,
# runs/g1_H/eval_sens_ls and runs/g1_N/eval_primary and the stored study-1 results of the coverage check and of the
# snapshot placement. Needs Octave, the patched QuaDRiGa tree (QUADRIGA_PATCHED_DIR, see quadriga/patch/) and
# eclipse-sumo. Stages 1 and 2 are resumable; a decision-level run starts from an empty folder.
# Outputs: runs/ (records and per-beacon series) and $OUT (default runs/s3_out): heldout_*.json, per_lane_new_sites.txt,
# s_place_A.json, urban_rows.csv, detection_runs.json (stored copies: ../results/study3/).
# Usage: bash run_study3.sh            (PY=<python> to choose the interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export SPEEDCHECK_CONSTANTS="$PWD/config/study2_constants.json"
TAG=dev_s3; RUN=runs/g1_U3_$TAG; SER=runs/s3_series; PLACE=runs/s_place_${TAG}_place; DEC=runs/s3_decision
OUT=${OUT:-runs/s3_out}
URB=h_pp_now_c20
mkdir -p "$OUT"

# 1. stage 1 (QuaDRiGa channels) of the ten sites and the snapshot-placement sites; stage 2 with the base estimator
for K in 30 31 32 33 34 35 36 37 38 39; do $PY pipeline/s3_stage1.py --site "$K" --out "$RUN"; done
$PY pipeline/s3_snapshot_placement.py --out "$PLACE"
$PY pipeline/s3_stage2.py --run-dir "$RUN"

# 2. per-beacon series under both channel models; the stored base-estimator statistics must equal those of stage 2
$PY pipeline/s3_series.py --family U3 --run-dir "$RUN" --sites 30 40 --out "$SER" --check-against "$RUN/eval_$TAG"
$PY pipeline/s3_series.py --family U3 --run-dir "$RUN" --sites 30 40 --out "$SER" --arm inhouse --check-against "$RUN/eval_$TAG"

# 3. urban estimator and evaluation, per survey design (ref: survey in traffic; reff: free-road survey)
: > "$OUT/per_lane_new_sites.txt"
for ROLE in ref reff; do
  for LABEL in U3 U3in; do
    $PY pipeline/s3_estimate.py --series "$SER" --family $LABEL --sites 30 40 --out "runs/s3_res_$ROLE" --survey-role $ROLE
  done
  $PY pipeline/s3_evaluate.py --res "runs/s3_res_$ROLE" --series "$SER" --family U3 --variants $URB --survey-role $ROLE \
    --out "$OUT/heldout_$ROLE.json"
  $PY pipeline/s3_evaluate.py --res "runs/s3_res_$ROLE" --series "$SER" --family U3in --variants $URB --survey-role $ROLE \
    --out "$OUT/heldout_in_$ROLE.json"
  echo "== TR survey role $ROLE" >> "$OUT/per_lane_new_sites.txt"
  $PY pipeline/s3_per_lane.py --res "runs/s3_res_$ROLE" --series "$SER" --family U3 --variant $URB --survey-role $ROLE \
    >> "$OUT/per_lane_new_sites.txt"
done

# 4. snapshot-placement factor of the urban estimator
$PY pipeline/s3_placement_factor.py --dir "$PLACE" --out "$OUT/s_place_A.json"

# 5. detection rules: five runs (survey, channel model, estimator), each through stage 3 and the decision adapter
decision_run() {
  local R=$1 SURVEY=$2 CH=$3 EST=$4 T=${TAG}_$1 D=$PWD/$DEC/$1 RES=runs/s3_res_reff
  [ "$SURVEY" = traffic ] && RES=runs/s3_res_ref
  rm -rf "$D"; mkdir -p "$D/ptr/g1_H" "$D/ptr/g1_N" "$D/res"
  $PY pipeline/s3_decision_records.py --base "$RUN/eval_$TAG" --res "$RES" --survey "$SURVEY" --channel "$CH" \
    --estimator "$EST" --out "$D/ptr/g1_U/eval_$T"
  cp -r runs/g1_H/eval_primary "$D/ptr/g1_H/eval_$T"
  cp -r runs/g1_H/eval_sens_ls "$D/ptr/g1_H/eval_${T}_ls"
  cp -r runs/g1_N/eval_primary "$D/ptr/g1_N/eval_$T"
  cp ../results/coverage_check_sens_coverage.json "$D/res/coverage_check_${T}_cov.json"
  cp ../results/s_place_sens_place.json "$D/res/s_place_${T}_place.json"
  (
    export SPEEDCHECK_RUNS="$D/ptr" SPEEDCHECK_RESULTS="$D/res" SPEEDCHECK_ADAPTER_OUT="$D/res/adapter"
    $PY pipeline/stage3_criteria.py --tag "$T"
    ( cd ../inhouse && $PY ../tr37885/pipeline/decision_adapter.py e2 --tag "$T" )
    if [ "$CH" = tr ] && [ "$EST" = urban ]; then
      $PY reports/make_reports.py --tag "$T" --results "$D/res" --out "$D/res/reports"
    fi
  )
}
decision_run trfree free tr urban
decision_run trtraf traffic tr urban
decision_run ctl free tr base
decision_run infree free inhouse urban
decision_run intraf traffic inhouse urban
$PY pipeline/s3_decision_readout.py --runs "$DEC" --pub ../results --out "$OUT"
echo "study 3 run complete"
