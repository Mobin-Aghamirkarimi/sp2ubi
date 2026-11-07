#!/usr/bin/env bash
# Study 2, full run: pilot records, calibration, development stage (Dev-1, Dev-2, Dev-3 and the decision of
# step0_decision.json), study-2 seeds, held-out families U2 and N2 with the study-2 S-place check, stage 3 and texts.
# Run after run_study1.sh: Dev-2 reads the study-1 records runs/g1_U and runs/g1_N, and stage 3 reads the stored
# study-1 S-place and K7 results. Needs Octave, the patched QuaDRiGa tree (QUADRIGA_PATCHED_DIR, see quadriga/patch/)
# and eclipse-sumo. Every stage is resumable (outputs are reused only with the same provenance).
# The seeds of the held-out families are re-derived from config/study2_seed_record.json (pipeline/s2_seeds.py).
# Usage: bash run_study2.sh            (PY=<python> to choose the interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export SPEEDCHECK_CONSTANTS="$PWD/config/study2_constants.json"
P0TAG=dev_p0; CTAG=dev_s2_cal; DTAG=dev_s2
site_range() { $PY -c "import json, sys; print(json.load(open(sys.argv[1]))['evaluation_v4']['p0']['families'][sys.argv[2]]['sites'])" \
  "$SPEEDCHECK_CONSTANTS" "$1"; }

# 0. pilot records (stage 1 and the study-1 stage 2 under the pilot tag)
for FAM in P0U P0H P0N; do
  N=$(site_range "$FAM")
  $PY pipeline/stage1_channels.py --family "$FAM" --sites 0 "$N" --tag $P0TAG
  $PY pipeline/stage2_evaluate.py --family "$FAM" --sites 0 "$N" --tag $P0TAG --run-dir "runs/g1_${FAM}_$P0TAG"
done

# 1. calibration reads (Dev-1 survey passes of P0U and P0N only) and the development-stage constants
for FAM in P0U P0N; do
  $PY pipeline/s2_evaluate.py --family "$FAM" --tag $CTAG --run-dir "runs/g1_${FAM}_$P0TAG" \
    --check-against "runs/g1_${FAM}_$P0TAG/eval_$P0TAG"
done
$PY pipeline/s2_calibrate.py --eval "P0U=runs/g1_P0U_$P0TAG/eval_$CTAG" --eval "P0N=runs/g1_P0N_$P0TAG/eval_$CTAG" \
  --out config/study2_calibration.json
$PY make_constants.py --study2 --study2-calibration config/study2_calibration.json --study2-timestamped \
  --study2-merge-done --out "$SPEEDCHECK_CONSTANTS"

# 2. development records: Dev-1 (pilot families), Dev-2 (study-1 U and N records), Dev-3 (family D3U, two parts)
for FAM in P0U P0N P0H; do
  $PY pipeline/s2_evaluate.py --family "$FAM" --tag $DTAG --run-dir "runs/g1_${FAM}_$P0TAG" \
    --check-against "runs/g1_${FAM}_$P0TAG/eval_$P0TAG"
done
for FAM in U N; do
  $PY pipeline/s2_evaluate.py --family "$FAM" --tag $DTAG --run-dir "runs/g1_$FAM" --check-against "runs/g1_$FAM/eval_primary" \
    --sites 0 10
done
$PY pipeline/s2_dev3.py --guard
W=$($PY pipeline/s2_dev3.py --setting workers_stage1)
for PART in a b; do
  read -r -a PL <<< "$($PY pipeline/s2_dev3.py --part-sites $PART)"
  S0=${PL[0]}; S1=$(( ${PL[-1]} + 1 ))
  $PY pipeline/s2_dev3.py --identity $PART --out "runs/identity_dev3_$PART.json"
  $PY pipeline/stage1_channels.py --family D3U --sites "$S0" "$S1" --tag $DTAG --workers "$W" --out "runs/g1_D3U_$DTAG"
  $PY pipeline/stage2_evaluate.py --family D3U --sites "$S0" "$S1" --tag $DTAG --run-dir "runs/g1_D3U_$DTAG"
done

# 3. development-stage decision (step0_decision.json, step0_report.json), then the selected configuration and the seeds
$PY pipeline/s2_selection_run.py --dev1 "P0U=runs/g1_P0U_$P0TAG/eval_$DTAG" --dev1 "P0N=runs/g1_P0N_$P0TAG/eval_$DTAG" \
  --dev1 "P0H=runs/g1_P0H_$P0TAG/eval_$DTAG" --dev2 "U=runs/g1_U/eval_$DTAG" --dev2 "N=runs/g1_N/eval_$DTAG" \
  --dev3 "runs/g1_D3U_$DTAG/eval_$DTAG" --out-dir ../results/study2
$PY make_constants.py --study2 --study2-calibration config/study2_calibration.json --study2-timestamped \
  --study2-merge-done --study2-selected ../results/study2/step0_decision.json \
  --study2-seeds config/study2_seed_record.json --out "$SPEEDCHECK_CONSTANTS"

# 4. held-out families: stage 1 per site, then the stage-2 runs of s2_heldout.stage2_plan
$PY pipeline/s2_heldout.py --guard
TAG=$($PY -c "import sys; sys.path.insert(0, '.'); from pipeline.s2_heldout import TAG; print(TAG)")
for ROLE in U2-a U2-b N2; do
  FAM=$($PY pipeline/s2_heldout.py --role-info $ROLE family)
  read -r -a SITES <<< "$($PY pipeline/s2_heldout.py --role-info $ROLE sites)"
  OUT="runs/g1_${FAM}_$TAG"
  for K in "${SITES[@]}"; do $PY pipeline/s2_heldout.py --stage1 "$FAM" "$K" --out "$OUT"; done
  S0=${SITES[0]}; S1=$(( ${SITES[-1]} + 1 ))
  $PY pipeline/s2_heldout.py --stage2-plan "$FAM" | while read -r T2 EXTRA; do
    [ -n "$T2" ] || continue
    # shellcheck disable=SC2086
    $PY pipeline/stage2_evaluate.py --family "$FAM" --sites "$S0" "$S1" --tag "$T2" --run-dir "$OUT" $EXTRA
  done
done

# 5. study-2 S-place light check (role S2); its final state is read by stage 3
if $PY pipeline/s2_snapshot_placement.py; then echo "DONE s2 S2" > runs/STATE_FINAL_s2_S2
else echo "FAILED s2 S2" > runs/STATE_FINAL_s2_S2; fi

# 6. stage 3, numbers and texts
$PY pipeline/s2_stage3.py --study1-s-place ../results/s_place_sens_place.json \
  --study1-k7 ../results/coverage_check_sens_coverage.json --out ../results/study2/heldout
$PY reports/s2_texts.py --criteria "../results/study2/heldout/criteria_s2_$TAG.json" \
  --outcome "../results/study2/heldout/outcome_s2_$TAG.json" --decision ../results/study2/step0_decision.json \
  --adapter-dir ../results/study2/heldout/adapter --out ../results/study2/heldout/reports --tag "$TAG"
$PY reports/study2_derived_values.py
echo "study 2 run complete"
