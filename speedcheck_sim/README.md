# Speed-check simulation

Code and stored outputs of the roadside speed-check evaluation: study 1 (the speed check with the base estimator),
study 2 (a second survey design for urban street sites) and study 3 (the urban estimator on newly simulated urban LOS
sites, and the detection rules with it).

## Layout

| Folder | Content |
|---|---|
| `fdoa/` | FDOA geometry, in-house channel model, the base estimator (`estimator.py`) and the urban estimator (`urban.py`) |
| `inhouse/` | simulations on the in-house channel model: site accuracy, residual monitor, consistency test, decision rules, matching; `inhouse/results/` holds the stored outputs that the reports read (site-accuracy passes, consistency simulation and analysis, residual monitor, decision simulation with 8 levels, matching); the base and sensitivity scripts (`*_base.py`, `decisions_levels.py`, `decisions_monitor.py`, `levels_qualification.py`, `residual_scale.py`) have no stored output |
| `tr37885/` | TR 37.885 pipeline: interface, antenna, NLOSv blockage, SUMO kinematics, scatterer mobility, Huber fit, QuaDRiGa driver (`quadriga/`), stages 1 to 3 (`pipeline/`; study-2 modules `s2_*.py`, study-3 modules `s3_*.py`), report generators (`reports/`), constants and design records (`config/`), `run_study1.sh`, `run_study2.sh`, `run_study3.sh` |
| `results/` | stored study-1 outputs (stage 3, decision adapter, reports), `results/study2/` (study 2) and `results/study3/` (study 3) |
| `tests/` | unit tests of the pipeline components |

## Requirements

Python 3.12 with `requirements/speedcheck_sim.lock` (includes eclipse-sumo 1.28.0). A full run also needs Octave 8.4.0
and QuaDRiGa v2.8.1 (commit 277866650eb115adb5b3e8ac252b0d1df073596d), patched by
`tr37885/quadriga/patch/build_patched_tree.sh`; the patched tree is found through the environment variable
`QUADRIGA_PATCHED_DIR` (default `speedcheck_sim/quadriga_patched`). The report generators need numpy, scipy and
matplotlib only.

## Large stored inputs

The stage-2 records of study 1 (`results/stage2/*.tar.gz`, 11 files) and the extra site-accuracy passes
(`inhouse/results/site_accuracy_passes_extra.json`) are the largest files of the repository. Their SHA-256 sums are in
`SHA256SUMS` (check with `cd speedcheck_sim && sha256sum -c SHA256SUMS`). They are needed for the erasure
rows of the number file (`A|erasure|...`; Supplement Table S4), for the error-CDF figure `fig_S1` and for the in-house
decision simulation (`inhouse/decisions.py`, which reads the extra passes). Every other command below runs without them.
`SHA256SUMS` also lists the 14 stored study-3 outputs in `results/study3/`.

## Commands

Environment: `python3.12 -m venv .venv-sim && .venv-sim/bin/pip install -r requirements/speedcheck_sim.lock` at the
repository root; `python` below is `.venv-sim/bin/python`. Commands are run from `speedcheck_sim/`.

| Result | Command | Runtime |
|---|---|---|
| Study 1 from the stored outputs: `table_A` (Table 8), `table_B` (highway cells of Fig. 6; every row: Supplement S4), the number file of every study-1 value of Section VI-D, Table 12 (matching) and the supplement, the report figures `fig_C` and `fig_D` (data of Figs. 7 and 8) and the diagnostic figures `fig_E`, `fig_S1` | `mkdir -p runs && for f in results/stage2/*.tar.gz; do tar xzf "$f" -C runs; done`, then `cd tr37885`, `SPEEDCHECK_RUNS=../runs python reports/make_reports.py --tag primary --out ../out`, `python reports/number_macros.py --tag primary --numbers ../out/numbers_primary.json --out ../out/speedpath-numbers.tex`, `python reports/select_texts.py --tag primary --out ../out` | short |
| Study 2 (Supplement S4) from the stored stage-3 outputs | `cd tr37885`, `python reports/s2_texts.py --criteria ../results/study2/heldout/criteria_s2_dev_s2_heldout.json --outcome ../results/study2/heldout/outcome_s2_dev_s2_heldout.json --decision ../results/study2/step0_decision.json --out ../out2 --tag dev_s2_heldout`; `python reports/study2_derived_values.py --check` | short |
| Constants files | build the patched QuaDRiGa tree (`QUADRIGA_DIR=<QuaDRiGa checkout> QUADRIGA_PATCHED_DIR=$PWD/quadriga_patched bash tr37885/quadriga/patch/build_patched_tree.sh`), then `cd tr37885`, `python make_constants.py` (study 1); `python make_constants.py --study2 --study2-calibration config/study2_calibration.json --study2-timestamped --study2-merge-done --study2-selected ../results/study2/step0_decision.json --study2-seeds config/study2_seed_record.json --out config/study2_constants.json` (study 2) | short |
| Study 1, full run | `bash tr37885/run_study1.sh` | very long |
| Study 2, full run (after study 1) | `bash tr37885/run_study2.sh` | very long |
| Study 3 from the stored outputs: the number file of Table 9 and of Supplement S4, and every cell of Fig. 6 | `python tr37885/reports/study3_values.py --check`, `python tr37885/reports/detection_figure.py --check` | short |
| Study 3, full run (after study 1) | `bash tr37885/run_study3.sh` | very long |
| Decision simulation on the in-house channel | `python inhouse/decisions.py --levels 8` (writes `inhouse/results/decisions_L8_full.json` from `inhouse/results/site_accuracy_passes_extra.json`; equal to the stored file except the run-time field `seconds_last_run`) | long |
| Matching simulation on the in-house channel | `python inhouse/matching.py --out-tag _trf` (writes `inhouse/results/matching_trf.json`, byte-identical) | medium |
| Consistency analysis | `python inhouse/consistency_analysis.py` (writes `inhouse/results/consistency_analysis_full.json`, byte-identical) | short |
| Unit tests | `python -m pytest -q speedcheck_sim/tests` (from the repository root) | short |

Expected results: every table, number and text file in `out/` and `out2/` is byte-identical to its stored copy in
`results/reports/primary/` and `results/study2/heldout/reports/`, except the first line of `speedpath-numbers.tex`
(it names the input path), the provenance block of `numbers_primary.json` (a fresh run records the source and constants
hashes) and the creation date inside the PDF figures. The two constants commands
rewrite `config/study1_constants.json` and `config/study2_constants.json` byte for byte. The printed versions of
Figs. 7 and 8 and of the supplement's Fig. S1 (Fisher bound) are drawn by `paper_values/` (top-level README).

Without the stage-2 records, `make_reports.py` still writes every value except the erasure rows and `fig_S1`. The
study-2 stages read `tr37885/config/study2_constants.json` (`run_study2.sh` sets `SPEEDCHECK_CONSTANTS`).
`config/design_constants.json` holds the design constants of Table 3 of the paper.
`config/presentation_constants.json` holds presentation values of the paper, each taken from the source named in its
`origin` field; it is stored as data, and no script of this repository writes it.

## Study 3: the urban estimator

Study 3 evaluates the urban estimator (`fdoa/urban.py`) against the base estimator on the same crossings of ten newly
simulated urban LOS sites (family `U3`, sites 30 to 39; per speed 30 survey crossings in traffic, 30 free-road survey
crossings and 60 traffic crossings), under TR 37.885 and under our channel model, and then runs the detection rules
with the urban estimator. Sites 40 to 44 serve the snapshot-placement check. The seeds are derived in
`tr37885/pipeline/s3_family.py`; the study-2 constants file is used. `run_study3.sh` runs every step in order.

| Step | Module (`tr37885/pipeline/`) | Output |
|---|---|---|
| Family and seeds | `s3_family.py` | registers `U3` in `stage1_channels.py` at run time |
| Stage 1 and stage 2 (base estimator) | `s3_stage1.py`, `s3_stage2.py`, `s3_snapshot_placement.py` | `runs/g1_U3_dev_s3/`, `runs/s_place_dev_s3_place/` |
| Per-beacon series, both channel models | `s3_series.py` | `runs/s3_series/` (the base-estimator statistics stored there must equal those of stage 2) |
| Urban estimator per crossing | `s3_estimate.py` | `runs/s3_res_ref/` (survey in traffic), `runs/s3_res_reff/` (free-road survey) |
| Evaluation with the stage-3 site rules | `s3_rules.py`, `s3_evaluate.py`, `s3_per_lane.py` | `heldout_ref.json`, `heldout_reff.json` (TR 37.885), `heldout_in_ref.json`, `heldout_in_reff.json` (our channel model), `per_lane_new_sites.txt` |
| Snapshot-placement factor | `s3_placement_factor.py` | `s_place_A.json` |
| Detection rules: five runs through `stage3_criteria.py` and `decision_adapter.py` | `s3_decision_records.py`, `s3_decision_readout.py` | `urban_rows.csv`, `detection_runs.json` |
| Numbers of the paper | `reports/study3_values.py`, `reports/detection_figure.py` | `speedpath-s3.tex`, `detection_figure.csv` |

The other stored files of `results/study3/` come from the same modules applied to the stage-1 records of earlier
families: `heldout_U2.json` and `heldout_U2_inhouse.json` (urban LOS sites 10 to 19 of study 2, family `U2`, run
directory `runs/g1_U2_dev_s2_heldout`; the first file also holds four comparison variants of `s3_estimate.py`),
`per_lane_A.txt` (families `U2` and `U`) and `highway_A_sidecheck.json` (highway LOS sites of study 1, family `H`,
`s3_evaluate.py --speeds 20 30 38`), for example

```
python tr37885/pipeline/s3_series.py --family U2 --run-dir tr37885/runs/g1_U2_dev_s2_heldout --sites 10 20 --out <series>
python tr37885/pipeline/s3_estimate.py --series <series> --family U2 --sites 10 20 --out <res> --variants h_pp_now h_pp_now_quad h_peak_now g_s30_survey h_peak_now_fofree
python tr37885/pipeline/s3_evaluate.py --res <res> --series <series> --family U2 --out heldout_U2.json --variants h_pp_now_c20 h_pp_now_quad_c20 h_peak_now_c20 g_s30_survey_c20 h_pp_now h_peak_now_fofree_c20
```

(with `SPEEDCHECK_CONSTANTS=tr37885/config/study2_constants.json` for `U2` and `U3`).

What is stored and what is not:

- Stored: every output that the paper and the supplement read (`results/study3/`, listed in `SHA256SUMS`).
  `study3_values.py --check` and `detection_figure.py --check` regenerate the numbers of Table 9, of Supplement S4
  and of Fig. 6 from them.
- Not stored: the stage-1 channel records of the `U3` sites (about 60 MB per site), the per-beacon series, the
  per-crossing outputs of the estimators and the outputs of the five decision-level runs. They regenerate with
  `run_study3.sh`, which needs Octave, the patched QuaDRiGa tree and SUMO, and, for the decision-level runs, the
  stage-2 records of a full study-1 run (`runs/g1_H/eval_primary`, `runs/g1_H/eval_sens_ls`, `runs/g1_N/eval_primary`).
- From the stage-1 records of the original run, `s3_series.py`, `s3_estimate.py`, `s3_evaluate.py` and `s3_per_lane.py`
  give the stored `heldout_ref.json`, `heldout_reff.json`, `heldout_in_ref.json` and `heldout_in_reff.json` byte for
  byte, and the per-crossing outputs of the urban estimator and of its four comparison variants equal those of the
  original code (checked on site 30 with the survey in traffic); `s3_placement_factor.py` gives the stored
  `s_place_A.json` byte for byte from the stored per-beacon series of the placement sites. `per_lane_new_sites.txt`
  differs only in its two header lines, which carry a clock time in the stored file. Stage 1 itself (QuaDRiGa) was not
  rerun for these checks. `detection_runs.json` from a fresh run differs in `input_sha256_urban`, the hash of the record
  files, which contain a provenance block.
- `urban_rows.csv` is the output of the original runs. 1,488 of its 2,832 data lines (below a header line) belong to two
  attack classes (`coverage`, `signmixed`) or to nine metrics (`billed_*`, `bill_median`, `overbilled_share_gt2pct`,
  `gain_*`, `f_min`, `f_min_union`) that the decision simulation of this repository does not compute; the paper and
  the supplement print none of them. A fresh run writes the other 1,344 lines. Read from the stored outputs of the five
  original decision-level runs, `s3_decision_readout.py` writes `detection_runs.json` byte for byte and all 2,832 lines
  of `urban_rows.csv` with the same per-channel values; the 1,344 lines are byte-identical, and in the other 1,488 the
  columns `direction`, `*_worse`, `*_tableB_governing` and `published_suburban` differ, because this repository's code
  holds no direction and no published value for those classes and metrics.

Names in the study-3 files: `ls` is the base estimator and `h_pp_now_c20` the urban estimator; `U3in` is `U3` under
our channel model; roles `ref` (survey in traffic), `reff` (free-road survey), `test`; decision-level runs `trfree`,
`trtraf`, `infree`, `intraf` (TR 37.885 or our channel model, free-road survey or survey in traffic) and `ctl` (base
estimator, TR 37.885, free-road survey); in `s_place_A.json`, the calls `B`, `A`, `N` are those of the snapshot
placement, the key `A` is the urban estimator and `registered_recomputed` the base estimator.

## Notes on the stored data

- The constants (`tr37885/config/study1_constants.json`, `study2_constants.json`) and the design constants were fixed
  before the evaluation; the timestamped design records are described in Supplement S2. The copies in this
  repository carry the same values with shorter note fields, so their SHA-256 differs from that of the timestamped
  files. Both constants files are the output of `make_constants.py`.
- `quadriga.patched_tree_sha256` in the constants is the hash of the tree that `build_patched_tree.sh` builds from this
  repository's patch. It differs from the hash recorded at the time of the runs in one comment line of the patch;
  the code of the patch is the same.
- Study-2 seeds: `tr37885/config/study2_seed_record.json` holds the bundle hash and the Bitcoin block header of the
  timestamp; `pipeline/s2_seeds.py` checks the header (proof of work, block hash, Merkle root) and derives the seeds.
- Provenance blocks of the stored outputs keep the tag, the tool versions and the run knobs. Where a stored output
  records the SHA-256 of another stored file, the hash refers to the copy in this repository.
- The `origin` fields of `config/design_constants.json` and `config/presentation_constants.json` and one note of the
  constants files use their own numbering of paper sections, tables and figures: their "Section V-C" is the speed
  verification, Section V-D of the paper; "Table 9" in `presentation_constants.json` is the decision-level table,
  whose values are in Fig. 6 and Supplement S4, and its "Fig. 6" is Fig. 7 (per-beacon error against SNR); "Table 11"
  in the constants files and in `make_constants.py` is Table 12 (coalitions, with the matching numbers).
- The stored `results/study3/speedpath-s3.tex` and the file written by `reports/study3_values.py` differ in the first
  (comment) line only; every key and value is the same (`study3_values.py --check`).
- Random seeds are part of the constants and of the code. Seed strings inside the code (for example
  `job_seed("e3v5", ...)`, `seed("p5", ...)`, `seed("h5", ...)`, the label `SEED_LABEL` of `pipeline/s3_family.py`) are
  kept unchanged so that every stored output can be regenerated.

## Key legend

Keys and labels of the stored data form a fixed schema. Some are internal labels of the runs; they are kept because
they seed the random generators or name stored outputs.

- Cells `P1-10` ... `P3-15`: P1 urban LOS, P2 highway LOS, P3 urban with blocked crossings (NLOSv); the number is the
  speed in m/s. Families `H`, `U`, `N` (highway, urban, NLOSv sites) of study 1; pilot families `P0U`, `P0H`, `P0N`;
  study-2 families `U2`, `N2`, `H2` (held-out sites) and `D3U` (development family Dev-3); `5U`, `5H` (P5, not run).
- Generators: `G1` (TR 37.885 channel generated with QuaDRiGa) and `G2` (ray tracing, not run); run folders
  `runs/g1_<family>/eval_<tag>/`; arm names `gen_on` / `gen_off` (TR 37.885 channel with / without the scatterer-mobility
  term) and `inhouse` (our channel model).
- Tags: `primary`; `sens_*` (pre-specified sensitivities); `dev_*` (development runs; `dev_p0` the pilot,
  `dev_s2`, `dev_s2_cal`, `dev_s2_heldout` study 2).
- Check labels of the constants: `C1` ... `C9`, `K2` ... `K7`, `R8` (qualification validity); study-2 thresholds
  `N1` ... `N18`, decision rules `D0-R1` ... `D0-R5`, diagnostics `D0.1` ... `D0.6`, ladder rungs `L0` ... `L3`, gate `G`.
- Detection rules of Section V-D5: `R1` notice rule (its flag in consecutive epochs is the `escalation`), `R2` residual
  rule, `R3` consistency rule (keys such as `r1_thr`, `r2_epoch`, `r3_inhouse`).
- Constants blocks `evaluation_v4`, `evaluation_v5` (study 1) and `evaluation_v6` (study 2); `step0` (constants block
  and the file names `step0_decision.json`, `step0_report.json`) is the development stage of study 2.
- Key prefixes of the number files: `A|` (`table_A`, Table 8), `B|` (`table_B`, Fig. 6 and Supplement S4), `C|`, `D|`, `E|` (figures `fig_C` = Fig. 7, `fig_D` = Fig. 8, `fig_E`), `K|` (realized K),
  `S|` (sensitivities), `E5|` (matching simulation), `reg|` (study-1 design record), `S2|`, `A2|`, `S2b|`, `S2d|`
  (study 2). Experiment labels `E1` (site-accuracy simulation), `E2` (decision simulation), `E3` (consistency
  simulation), `E5` (matching simulation) appear in key names and module aliases.
- `lever_d` is the Huber fit (d) of `tr37885/huber_fit.py`, which the pilot did not adopt; `scales*` keys hold the
  scale estimates 1.4826 MAD and (q84 - q16) / 2.
- The quality gate (`gate`, `gate_discard`, `gate_ratio`) is the per-crossing residual gate of the paper.
- `S-place` (tag `sens_place`, files `s_place_*.json`) is the snapshot-placement sensitivity; `K7` is the coverage check
  (`coverage_check_*.json`) with its widening factors `c_K7` and `c(n)`; `P0` is the pilot; `P4` ... `P7`, `U9`, `R` are
  cells and families of the original design that were not run.
- Design items of the constants: `k2` (decision and matching simulations on the TR 37.885 records), `k3_c4_light`
  (frequency-only separation, `C4-light`), `k5_erasures` (beacon erasures).
- Study 2: `Dev-1` (pilot families), `Dev-2` (study-1 U and N records), `Dev-3` (family D3U) are the development data;
  estimator candidates `E-a` (window statistic), `E-b` (matched statistic forced), `E-c` (acceleration gate), `E-d`
  (stop detector), `E-db` (stop detector with the core statistic), `E-e` (two-segment fit), `E-f` (core statistic);
  `L-T` survey in traffic; `L-B`, `L-B′` rules for blocked crossings; readings `F` (free-road survey) and `T`
  (surrogate survey in traffic).
- Keys of the branch files: `T2|` ... `T8|`, `T3N|` (outcome texts written for Section VI-D), `fnb|` (texts naming the
  site types of the frequency-only detection claim), `Bstratum|`, `Bnote|` (`table_B` stratum status and note). The
  paper reads only the switches `Bstratum|`, `esc|` and `K|` from these files; the other texts are written by the
  report generators and kept in the stored files.
