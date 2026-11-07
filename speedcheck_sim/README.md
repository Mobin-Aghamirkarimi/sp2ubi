# Speed-check simulation

Code and stored outputs of the roadside speed-check evaluation (study 1 and study 2 of the supplement).

## Layout

| Folder | Content |
|---|---|
| `fdoa/` | FDOA geometry, in-house channel model and pass estimators |
| `inhouse/` | simulations on the in-house channel model: site accuracy, residual monitor, consistency test, decision rules, matching; `inhouse/results/` holds their stored outputs |
| `tr37885/` | TR 37.885 pipeline: interface, antenna, NLOSv blockage, SUMO kinematics, scatterer mobility, Huber fit, QuaDRiGa driver (`quadriga/`), stages 1 to 3 (`pipeline/`), report generators (`reports/`), constants and design records (`config/`), `run_study1.sh`, `run_study2.sh` |
| `results/` | stored study-1 outputs (stage 3, decision adapter, reports) and `results/study2/` (study 2) |
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
rows of the number file (`A|erasure|...`; Supplement Table S19), for the error-CDF figure `fig_S1` and for the in-house
decision simulation (`inhouse/decisions.py`, which reads the extra passes). Every other command below runs without them.

## Commands

Environment: `python3.12 -m venv .venv-sim && .venv-sim/bin/pip install -r requirements/speedcheck_sim.lock` at the
repository root; `python` below is `.venv-sim/bin/python`. Commands are run from `speedcheck_sim/`.

| Result | Command | Runtime |
|---|---|---|
| Study 1 from the stored outputs: `table_A` (Table 8), `table_B` (Table 9; every row: Supplement S7), the number file of every study-1 value of Section VI-D, Table 11 (matching) and the supplement, the report figures `fig_C` and `fig_D` (data of Figs. 6 and 7) and the diagnostic figures `fig_E`, `fig_S1` | `mkdir -p runs && for f in results/stage2/*.tar.gz; do tar xzf "$f" -C runs; done`, then `cd tr37885`, `SPEEDCHECK_RUNS=../runs python reports/make_reports.py --tag primary --out ../out`, `python reports/number_macros.py --tag primary --numbers ../out/numbers_primary.json --out ../out/speedpath-numbers.tex`, `python reports/select_texts.py --tag primary --out ../out` | short |
| Study 2 (Section VI-D, Supplement S8) from the stored stage-3 outputs | `cd tr37885`, `python reports/s2_texts.py --criteria ../results/study2/heldout/criteria_s2_dev_s2_heldout.json --outcome ../results/study2/heldout/outcome_s2_dev_s2_heldout.json --decision ../results/study2/step0_decision.json --out ../out2 --tag dev_s2_heldout`; `python reports/study2_derived_values.py --check` | short |
| Constants files | build the patched QuaDRiGa tree (`QUADRIGA_DIR=<QuaDRiGa checkout> QUADRIGA_PATCHED_DIR=$PWD/quadriga_patched bash tr37885/quadriga/patch/build_patched_tree.sh`), then `cd tr37885`, `python make_constants.py` (study 1); `python make_constants.py --study2 --study2-calibration config/study2_calibration.json --study2-timestamped --study2-merge-done --study2-selected ../results/study2/step0_decision.json --study2-seeds config/study2_seed_record.json --out config/study2_constants.json` (study 2) | short |
| Study 1, full run | `bash tr37885/run_study1.sh` | very long |
| Study 2, full run (after study 1) | `bash tr37885/run_study2.sh` | very long |
| Decision simulation on the in-house channel | `python inhouse/decisions.py --levels 8` (writes `inhouse/results/decisions_L8_full.json`; needs the archive) | long |
| Unit tests | `python -m pytest -q speedcheck_sim/tests` (from the repository root) | short |

Expected results: every table, number and text file in `out/` and `out2/` is byte-identical to its stored copy in
`results/reports/primary/` and `results/study2/heldout/reports/`, except the first line of `speedpath-numbers.tex`
(it names the input path), the provenance block of `numbers_primary.json` (a fresh run records the source and constants
hashes) and the creation date inside the PDF figures (the PNG copies are identical). The two constants commands
rewrite `config/study1_constants.json` and `config/study2_constants.json` byte for byte. The printed versions of
Figs. 6 and 7 and of the supplement's Fig. S1 (Fisher bound) are drawn by `paper_values/` (top-level README).

Without the stage-2 records, `make_reports.py` still writes every value except the erasure rows and `fig_S1`. The
study-2 stages read `tr37885/config/study2_constants.json` (`run_study2.sh` sets `SPEEDCHECK_CONSTANTS`).
`config/design_constants.json` holds the design constants of Table 3 and Supplement Table S20.
`config/presentation_constants.json` holds presentation values of the paper, each taken from the source named in its
`origin` field; it is stored as data (the script that collected them read files outside this repository).

## Notes on the stored data

- The constants (`tr37885/config/study1_constants.json`, `study2_constants.json`) and the design constants were fixed
  before the evaluation; the timestamped design records are described in Supplement S2. The release copies carry the
  same values. Their notes were shortened, and the note fields that described only the internal run organisation were
  removed, so their SHA-256 differs from the timestamped originals. Both constants files are the output of
  `make_constants.py` as shipped.
- `quadriga.patched_tree_sha256` in the constants is the hash of the tree that `build_patched_tree.sh` builds from this
  repository's patch. It differs from the hash recorded at the time of the runs, because one comment line of the patch
  was changed; the code of the patch is unchanged.
- Study-2 seeds: `tr37885/config/study2_seed_record.json` holds the bundle hash and the Bitcoin block header of the
  timestamp; `pipeline/s2_seeds.py` checks the header (proof of work, block hash, Merkle root) and derives the seeds.
- Provenance blocks of the stored outputs keep the tag, the tool versions and the run knobs. Where a stored output
  records the SHA-256 of another stored file, the hash refers to the release copy.
- Random seeds are part of the constants and of the code. Seed strings inside the code (for example
  `job_seed("e3v5", ...)`, `seed("p5", ...)`, `seed("h5", ...)`) are kept unchanged so that every stored output can be
  regenerated.

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
- Detection rules of Section V-C5: `R1` notice rule (its flag in consecutive epochs is the `escalation`), `R2` residual
  rule, `R3` consistency rule (keys such as `r1_thr`, `r2_epoch`, `r3_inhouse`).
- Constants blocks `evaluation_v4`, `evaluation_v5` (study 1) and `evaluation_v6` (study 2); `step0` (constants block
  and the file names `step0_decision.json`, `step0_report.json`) is the development stage of study 2.
- Key prefixes of the number files: `A|` (`table_A`, Table 8), `B|` (`table_B`, Table 9), `C|`, `D|`, `E|` (figures `fig_C` = Fig. 6, `fig_D` = Fig. 7, `fig_E`), `K|` (realised K),
  `S|` (sensitivities), `E5|` (matching simulation), `reg|` (study-1 design record), `S2|`, `A2|`, `S2b|`, `S2d|`
  (study 2). Experiment labels `E1` (site-accuracy simulation), `E2` (decision simulation), `E3` (consistency
  simulation), `E5` (matching simulation) appear in key names and module aliases.
- `lever_d` is the Huber fit (d) of `tr37885/huber_fit.py`, which the pilot did not adopt; `scales*` keys hold the
  scale estimates 1.4826 MAD and (q84 - q16) / 2.
- The quality gate (`gate`, `gate_discard`, `gate_ratio`) is the per-crossing residual gate of the paper.
