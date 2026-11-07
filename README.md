# SP2UBI

Reference implementation and evaluation code of **SP2UBI: Secure and Privacy-Preserving Usage-Based Insurance**
(IEEE Access). The repository contains the protocol of the four entities (OBU, RSU, TSP, insurer) with its risk path
(BFV under one insurer key, homomorphic MAC with two tag blocks, BBS04 group signatures with a dealer-free opening key,
receipts, settlement and dispute) and its speed-verification path (roadside FDOA speed check with two-party
exponential ElGamal), the unit and protocol tests, the scripts that produce every computed number, table and figure
of the evaluation, the speed-check simulation, and the stored results of these scripts.

## Requirements

- Linux x86-64 and **Python 3.12**. The `openfhe` wheel (1.5.1.0.24.4) contains a CPython 3.12 extension only; on
  another Python version it installs but fails at import.
- `curl` and `sha256sum` for the download scripts.
- Optional, for single parts:
  - SageMath and the lattice estimator (malb/lattice-estimator, commit 53da5982597709ba0fdf94ea37a84d822310fd84) for
    the lattice-security estimate;
  - Octave 8.4.0 and QuaDRiGa v2.8.1 (git 277866650eb115adb5b3e8ac252b0d1df073596d) with the patch in `speedcheck_sim/`
    for a full rerun of the speed-check simulation (eclipse-sumo 1.28.0 is in `requirements/speedcheck_sim.lock`).

## Installation

All commands run from the repository root.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements/core.lock
bash data/fetch_so2021.sh          # synthetic telematics data set of So et al. (2021), checked by SHA-256
```

The other parts use their own pinned environments, created the same way:

| Environment | Lock file | Used for |
|---|---|---|
| `.venv` | `requirements/core.lock` | implementation, tests, HomMAC checks, dating, sizes, evaluation-key check, derived values, risk-path timings |
| `.venv-riskmodel` | `requirements/riskmodel.lock` | risk-model fit |
| `.venv-speed` | `requirements/speed.lock` | Fig. 2, timing figures, speed-path timings |
| `.venv-paillier` | `requirements/paillier.lock` | Paillier baseline of the timings (created by `experiments/timings/setup_envs.sh`) |
| `.venv-sim` | `requirements/speedcheck_sim.lock` | speed-check simulation, Figs. 6 and 7, supplement Fig. S1 |

For example: `python3.12 -m venv .venv-riskmodel && .venv-riskmodel/bin/pip install -r requirements/riskmodel.lock`.

The data set is not redistributed. `data/fetch_so2021.sh` downloads it from the page of its authors (override the
address with `SO2021_URL`, or put the file at `data/telematics_syn-032021.csv` by hand) and checks its SHA-256. The
stage-2 records of the speed-check simulation are included; see `speedcheck_sim/README.md`.

## Tests

```
.venv/bin/python -m pytest -q tests
```

133 tests (short). Without the data set the five tests that need it are skipped (128 passed, 5 skipped). The tests
exercise the protocol and the deviations of a TSP named in the paper: a substituted model, and altered, replayed,
dropped or injected records. `.venv/bin/python scripts/run_epoch_demo.py` runs one honest epoch and every deviation
and prints what the insurer detects. Further tests: `.venv/bin/python -m pytest -q experiments/sizes` (sizes of the
simpler designs) and `.venv-sim/bin/python -m pytest -q speedcheck_sim/tests` (speed-check pipeline components; the
QuaDRiGa tests are skipped without Octave).

## Results and how to reproduce them

Each script writes its result into the `results/` folder next to it; the stored files there are the results the paper
reports. Runtime classes: short (seconds to a few minutes), medium (up to about an hour), long (hours), very long
(several days on many cores). Table, figure and section numbers are those of the published article and its
supplement.

| Paper item | Command | Expected result | Runtime |
|---|---|---|---|
| Protocol tests, deviations of the TSP (Section VI) | `.venv/bin/python -m pytest -q tests` | all tests pass | short |
| HomMAC numerical check (Section VII-B; Supplement S1, Table S1) | `.venv/bin/python experiments/hommac_checks/hommac_checks.py`, then `.venv/bin/python experiments/hommac_checks/verdict_channel.py` | `hommac_checks.json`, `verdict_channel.json`, byte-identical | medium |
| Dating of a single record (Table 10, Section VII-A) | `.venv/bin/python experiments/dating/record_classes.py`, then `insurer_alone_dating.py`, then `coalition_dating.py` (same folder) | the three result files, byte-identical | medium |
| Risk model (Section VI, data and risk model; Supplement S11) | `.venv-riskmodel/bin/python experiments/risk_model/fit_risk_model.py` | `integer_model.json` and the integer weights byte-identical; the other CSV files equal up to the last floating-point digits (D² 0.136, Gini 0.495 as printed) | medium |
| Parameters and sizes (Tables 5, 7 and 14; Section VI-B) | `.venv/bin/python experiments/sizes/<script>.py` for `risk_path_sizes`, `simpler_designs_sizes`, `settlement_sizes`, `baseline_params`, `speed_beacon_emission`, `smudging_scale_range` (in this order) | the six files of `experiments/sizes/results/`, byte-identical | short |
| Lattice security (Table 5) | `SAGE=<sage> LATTICE_ESTIMATOR=<checkout> bash experiments/lattice_security/run_per_attack.sh`, then `.venv/bin/python experiments/lattice_security/summarize_per_attack.py` | `estimate_summary.json`, minimum 242.6 bits (dual hybrid); the Arora-GB job does not finish within the time limit | long |
| Error of the evaluation keys (Table 5) | `.venv/bin/python experiments/lattice_security/check_evalkey_sigma.py` | rms about 3.19 (public key) and 1.0 (evaluation keys), `confirmed` true; key generation is not seeded, so the digits vary | short |
| Fig. 2 (FDOA principle) | `.venv-speed/bin/python experiments/figures/plot_fdoa_principle.py` | `experiments/figures/results/fig_fdoa_principle.pdf` | short |
| Derived sizes, channel times, storage, timings, timing sums and ratios (Tables 5 to 7 and 14, Section VI) | `.venv/bin/python paper_values/derived_values.py` | `paper_values/paper_values.csv` (146 values, column `printed` as in the paper), byte-identical | short |
| Figs. 3 to 5 from the stored timings | `.venv-speed/bin/python paper_values/plot_timing_figures.py` | the three timing figures, byte-identical | short |
| New timings on your machine | `bash experiments/timings/setup_envs.sh`, then `bash experiments/timings/run_timings.sh --out <dir>` | timing files of your machine (not comparable with the stored ones on other hardware) | hardware-dependent |
| Figs. 6 and 7, supplement Fig. S1 (Fisher bound) | `.venv-sim/bin/python paper_values/plot_fdoa_error.py`, `plot_separation.py`, `plot_fisher.py` (same folder) | the three figures, drawn from the stored speed-check results, byte-identical | short |
| Table 3, Tables 8, 9 and 11 (matching), Section VI-D and Supplements S3 to S9 | see `speedcheck_sim/` below | stored report files, byte-identical | short (from stored outputs) to very long (full runs) |

Baseline labels: the timing code and its result files label the baselines B1 to B4. B2 (SP2UBI context with clear
weights), B3 (CKKS) and B4 (Paillier-3072) are the baselines B1, B2 and B3 of the paper (Fig. 4); B1 of the code
(single-key BFV with encrypted weights, without the slot fill) is an extra reference row that the paper does not print.

The figure scripts write `paper_values/figures/` (`timing_operations.pdf`, `timing_baselines.pdf`,
`timing_throughput.pdf` = Figs. 3 to 5; `fdoa_error_vs_snr.pdf` = Fig. 6; `separation.pdf` = Fig. 7;
`fisher_bound.pdf` = supplement Fig. S1).

## Speed-check simulation

`speedcheck_sim/` holds the roadside speed-check evaluation: the FDOA geometry and estimators (`fdoa/`), the
simulations on our channel model (`inhouse/`) and the TR 37.885 pipeline with QuaDRiGa and SUMO (`tr37885/`), with the
stored outputs that the paper and the supplement print (`results/`). Tables 8 and 9, the data of Figs. 6 and 7, the
matching numbers of Table 11 and every study-1 number of Section VI-D and of the supplement regenerate in minutes from
the stored stage-3 outputs. The erasure rows (Supplement Table S19) also need the stage-2 records in
`speedcheck_sim/results/stage2/` (SHA-256 sums in `speedcheck_sim/SHA256SUMS`). The
study-2 numbers and texts (Section VI-D, Supplement S8) regenerate from `speedcheck_sim/results/study2/`. Full runs
(`speedcheck_sim/tr37885/run_study1.sh`, then `run_study2.sh`) need Octave 8.4.0, QuaDRiGa v2.8.1 patched by
`speedcheck_sim/tr37885/quadriga/patch/build_patched_tree.sh` (QuaDRiGa has its own non-commercial licence and is not
included) and eclipse-sumo 1.28.0, and take several days on many cores. Environment: `requirements/speedcheck_sim.lock`
(`.venv-sim`). Commands, expected outputs and the legend of the keys and labels of the stored data:
`speedcheck_sim/README.md`.

## Layout

| Folder | Content |
|---|---|
| `sp2ubi/` | reference implementation: entities, BFV layer, HomMAC, BBS04, opening-key generation, commitments, proofs, speed-verification path, risk model, simulated daily records |
| `models/` | the integer risk model used by the implementation |
| `tests/` | unit and protocol tests |
| `scripts/` | epoch demonstration |
| `data/` | download script and checksum of the data set |
| `experiments/risk_model/` | fit of the additive Poisson risk model and its quantization |
| `experiments/hommac_checks/` | numerical checks of the HomMAC and of the verdict channel |
| `experiments/dating/` | record classes and dating of a single record by the insurer and by an insurer and TSP coalition |
| `experiments/sizes/` | object and message sizes, simpler designs, settlement, speed beacons, smudging scale, baseline parameters |
| `experiments/lattice_security/` | lattice-estimator run and the evaluation-key error check |
| `experiments/timings/` | timing scripts of the risk path, the baselines, the TSP throughput and the speed path; stored timings |
| `experiments/figures/` | FDOA principle figure |
| `paper_values/` | derived values and the figures of the evaluation |
| `speedcheck_sim/` | speed-check simulation (FDOA channel models, TR 37.885 pipeline, stored results of studies 1 and 2) |

## Reproducibility notes

- Every random experiment whose numbers the paper prints is seeded; the seeds are in the scripts or in the stored
  configuration files. The exceptions are `check_evalkey_sigma.py`, whose key generation draws fresh randomness, and
  the tests, which draw fresh keys (their pass or fail outcome does not depend on the keys).
- The timings are hardware-dependent. The stored timing files in `experiments/timings/results/` were measured on the
  authors' laptop, an Intel Core i7-1195G7 (governor `performance`, idle machine; `machine.txt`), and are the values the
  paper reports. `run_timings.sh` records the CPU, governor, load and the SHA-256 of the timed code with every result,
  and `check_machine.sh` checks the machine before a run.
- The data set of So et al. (2021) is not redistributed; `data/fetch_so2021.sh` checks the downloaded file against the
  SHA-256 every result was computed from.
- Where a regenerated file is not byte-identical to the stored one, the table above or `speedcheck_sim/README.md`
  states the difference (floating-point digits of the risk-model CSV files, creation dates of the report PDFs, input
  paths and provenance records).

## Licence

Apache License 2.0 (`LICENSE`). Third-party components and their terms are listed in `NOTICE`: the QuaDRiGa patch in
`speedcheck_sim/` stays under the QuaDRiGa Software License, and the data set stays under the terms of its authors.

## Citation

See `CITATION.cff`.
