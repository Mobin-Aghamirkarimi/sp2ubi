# Provenance of the supplement's tables and figures

Every number of the supplementary material of the SP2UBI paper is read by a script from a stored data file.
This file lists, for each table and figure of the supplement, its data file and the keys and printing rule
of its values (section 1), the further constants of the speed-check evaluation with their sources
(section 2), and the SHA-256 sum of every data file that the supplement reads (section 3). "Study 1" is the
evaluation behind the per-cell table of the paper, "study 2" the variant with a survey in traffic and
"study 3" the evaluation of the urban estimator.

## 1. Tables and figures

| Item | Data file | Keys and printing rule |
|---|---|---|
| Table S1 | study 1, number file (speedpath-numbers.tex) | S\|<cell>\|{gate,sigma_q,window}\|<level>\|{k,n,sigma,q99,oow,gate_discard,n_kept} |
| Table S2 | study 1, number file (speedpath-numbers.tex) | S\|<cell>\|snr_offset_db\|<offset>\|{k,n,sigma,oow} |
| Table S3 | study 1, number file (speedpath-numbers.tex) | S\|<cell>\|z_cut\|<cut>\|{Z,p_zq_minus1_honest,p_zq_minus1_beta10} |
| Table S4 | study 1, number file (speedpath-numbers.tex) | A\|erasure\|<model>\|<cell>\|{entry_yield,sigma} |
| Table S5 | study1_constants.json; decision simulation, decisions_base.py; study 1, evaluation code (stage2_evaluate.py) | sumo, sumo_implementation, impairments, HONEST, HEAD_PPB |
| Table S6 | risk path, hommac_checks.json | E[*].{E1..E4}.accepted, rate, wilson99.9, bound (rates and bounds to 5 decimals) |
| Table S7 | study 1, number file (speedpath-numbers.tex) | A\|<cell>\|k, n, k_inhouse, sigma_g1, sigma_inhouse_equalK (the paper's number formatter) |
| Table S8 | study 1, branch file (speedpath-branches.tex); presentation_constants.json | T4\|<cell> (verdict), T3cell\|<cell> (rating); A\|<cell>\|R_splace (ratio, as stored) |
| Table S9 | study 1, number file (speedpath-numbers.tex) | A\|<cell>\|{sigma_g1, sigma_g1_ci_hi, sigma_unqualified, qualified_over_margin_k, k, qualified_over_margin_cp_lo, qualified_over_margin_cp_hi, spearman, q99_ratio, p_holm} (the paper's number formatter) |
| Table S10 | study 2, number file (speedpath-s2.tex); study 2, derived number file (speedpath-s2-derived.tex) | keys A2\|<cell>\|..., S2\|<cell>\|..., S2d\|<cell>\|... as named in the table |
| Table S11 | study 3, heldout_ref.json (new sites, survey in traffic, TR 37.885); study 3, heldout_reff.json (new sites, free-road survey, TR 37.885) | <variant>\|<speed> with variant ls (base estimator) and h_pp_now_c20 (urban estimator): k, n, cp, sd_ref_med, sigma_qg, sigma_qg_ci, sigma_all, sigma_all_ratio, sigma_all_ratio_ci (2 decimals), q99, oow, gate, entry_yield, bias_uncal, bias_cal, bias_cal_ci, site_mean_sd (x 100, one decimal), worst_variant |
| Table S12 | study 3, heldout_in_ref.json (new sites, survey in traffic, our channel model); study 3, heldout_in_reff.json (new sites, free-road survey, our channel model) | <variant>\|<speed> (variants as for tab_s3.tex): k, n, sigma_qg, sigma_qg_ci, sigma_all, sigma_all_ratio, sigma_all_ratio_ci (2 decimals), bias_cal, bias_cal_ci (x 100, one decimal) |
| Table S13 | study 3, highway_A_sidecheck.json (highway LOS sites of study 1) | <variant>\|<speed> (variants as for tab_s3.tex): k, n, sigma_qg, sigma_all_ratio, sigma_all_ratio_ci (2 decimals), q99, gate, entry_yield, bias_cal (x 100, one decimal) |
| Table S14 | study 3, detection_runs.json (detection rules with the urban estimator: the five runs) | runs.<run> with run = trfree, trtraf, ctl, infree, intraf (rows in this order): status_urban, qualified_combinations, qualified_share_per_class (x 10 sites), n_eval_urban.n_eval, n_eval_urban.rows_below_min_eval, q_urban_profile (x 100), excess_between_site_sd (x 100, one decimal) |
| Table S15 | decision simulation, decisions_levels.py | PROFILES (literal, read with ast) |
| Table S16 | study 1, number file (speedpath-numbers.tex) | keys B\|sens_tyre\|tyre\|<profile>\|<nu>\|[inhouse\|]<metric>, printed as the paper prints them |
| Table S17 | study 1, table_B.csv | rows of sections honest and badsites and the rows of section margin with a non-zero rate; columns governing_<metric> for punitive_per_year, punitive_ub95, punitive_union_per_year, punitive_union_ub95, escalation_per_year; <metric> and inhouse_<metric> for entries_per_epoch_mean, low_m_share (the paper's number formatter, key B\|<section>\|<row>\|[inhouse\|governing\|]<metric>) |
| Table S18 | study 1, number file (speedpath-numbers.tex) | A\|<cell>\|{over_governing_k, k, over_governing_cp_lo, over_governing_cp_hi} |
| Fig. S1 | sweep of our channel model, inhouse_k4_sens_inhouse.json | rows with impairments off and finite pass_sigma_pct: snr_db, K_db, pass_sigma_pct (solid), fisher_pass_pct (dashed); plotted as stored |

## 2. Further constants of the speed check and of its evaluation

The constants of Table 3 of the paper are in the constants table of Section S11 of the supplement. The
constants below are used by the evaluation and are stated in the supplement where they are used.

| Constant | Value | Source |
|---|---|---|
| samples per subframe of the frequency estimate | 128 | design choice; registered; unambiguous range +- 64 kHz |
| traffic crossings per site and speed class (evaluation) | 60 | evaluation design |
| simulated sites per site type | 10 | evaluation design |
| falsification and channel-agreement margin | 1.5 | evaluation design |
| speed classes | 10, 15, 20, 30, 38 m/s | evaluation design |
| roadside head height | 5 m | TR 37.885 Table 6.1.4-1 |
| carrier frequency | 5.9 GHz | Section V-D1 of the paper |
| standard and tool versions | TR 37.885 V15.3.0; TR 38.901; QuaDRiGa v2.8.1 | 3GPP (TR 38.901 in version V16.1.0); QuaDRiGa release |
| beacon bandwidths (PRBs) | 10, 20, 50 | evaluation design |
| coverage rating of a cell (Section S4.1 of the supplement): share of qualified draws for covered, partly covered | 0.7, 0.3 | evaluation design |
| significance level of the channel-agreement check | 0.025 | evaluation design |
| Holm family size of the channel-agreement check, study 1 (study 2: three cells) | 6 | evaluation design |
| bootstrap resamples | 10,000 | evaluation design |
| line-of-sight probability constants | 1.05, 0.0114 | TR 37.885 Table 6.2-1 |
| honest VSS offsets (Section S2 of the supplement) | U(-3, +6) %, 0.5 % epoch term | honest VSS model of the decision simulation, an assumption |
| urban line-of-sight speed classes | 10–15 m/s | evaluation design |
| highway line-of-sight speed classes | 20–38 m/s | evaluation design |
| urban blocked-crossing speed | 15 m/s | evaluation design |
| traffic-error flag of a qualified site (margin x calibration limit) | 9 % | evaluation design |
| frequency-only manipulation tested | −10 % | evaluation design |
| separation threshold of the frequency-only claim | 1 | evaluation design |
| share of draws a speed class must qualify in for the decision-level claim | 30 % | evaluation design |
| oversized-tire sensor scale | −5 % | evaluation design |
| planned minimum of gated test crossings per site and speed class, decision-level simulation | 50 | evaluation design |
| floor of gated test crossings per site and speed class (below it: not covered) | 20 | evaluation design |
| entry keep probability at burst boundaries, decision-level simulation | 1 − 110 m/(v · 40 s) | evaluation design |
| largest honest share of z_q=-1 with a consistency-rule critical region at five entries (eps_fl^{1/5}) | 11 % | derived from the false-flag level of Section V-D5 of the paper |

## 3. Data files read by the supplement

All files are in this repository, except the design records with their bundles, the decision file of the
development pilot and the data set of So et al., which `data/fetch_so2021.sh` downloads from its authors
and checks against the SHA-256 sum below. The timestamp proofs are in the folder `stamps/` of the
supplementary material.

| Data file | SHA-256 |
|---|---|
| data set of So et al., telematics_syn-032021.csv | `558a9f7a5d1833a153dcc5f3c32218a4882ead6f3ecaca3338b3cbf4b13610c4` |
| decision simulation, decisions_base.py | `53883ed3c86a4976912fbefc7c856a372abaaef71c78e82c8c2d22385b930802` |
| decision simulation, decisions_levels.py | `c57632d0bcb169b4980c60b95ec4eeff8906fea93d6df329a0082143e20a6bac` |
| design_constants.json | `6e4121d5c1eec27bc5b4a0fffc6ea09a33f2f15d421b196d4da3e05a93bc6440` |
| presentation_constants.json | `a7ff1981b081cd149a33bf469264518070b8c3757a840e280365e9d7a239b310` |
| risk model, cv_metrics.csv | `15ea877570c7592750d1a1b739704fb6b6cdbe15fe387812f06d685adf427f8d` |
| risk model, quantization.csv | `15c6d2756dbd748c448399cd6ac4cc66d6adbc9cdac2b6cb35c01dff81838c96` |
| risk path, hommac_checks.json | `2d42f99fe4dc3bea9f55a061252ee3e105716b69d69aa710b7e531cadb3a59a1` |
| risk path, so2021_additive_poisson.json | `01a0dc6a1d30e65bbe0f2cd8b339cf81d1f2143ca06d617e22fa12b9cc66ead0` |
| risk path, verdict_channel.json | `9e150ff161774c8dd5cbd2580b8e3323480fdd42780c46eea6014ed63c2e186f` |
| simulated daily records, workload.py | `a631d1866b3fb32714547c06471135c731899cb0661e0db9a8e2060c20e6900f` |
| study 1, branch file (speedpath-branches.tex) | `dcdb176daf3bec4e62f3ea6cbbbf1fbf9d4a188ad44aa56c75bd9a75e85323b8` |
| study 1, criteria_primary.json | `39084284a81db42ca57d0bd048a8d8f567ce9430f78b73436f5e309907a98274` |
| study 1, design record (freeze) | `c1a533f34eb88a5506041b8a88c411587dad463dd809f0751453bdb373e21f7b` |
| study 1, design record (rules and pilot) | `2fa9a6fa5eb37658fd674e98426d10e5ccf7b2858c815bb7c7413def24c5fb4a` |
| study 1, development-pilot decision file | `3536a8b3f52e65374a63b4e9f87f7530c0dc34b719c9e3f56ae5af0cee69b805` |
| study 1, evaluation code (stage2_evaluate.py) | `ecaa46b8e1add885d353ca9eeb432e7014207d68b2126f2a7d1219512cac9a03` |
| study 1, freeze: proof (stamps/study1-freeze.ots) | `bbdaff09e17b4e90e58edeb050abdf77e2a6f3f3ef27750c5d3c15bb553a77b9` |
| study 1, number file (speedpath-numbers.tex) | `1a5d17b25668c7eeb2c80de409015d97a70f0ccd8a6ca2452573918eae40e494` |
| study 1, numbers_primary.json | `d195eb7586d2d5a33bcca767c24c2b46ce55ef1d1964efd4190881145e865c85` |
| study 1, rules and pilot: proof (stamps/study1-rules-and-pilot.ots) | `25d3a7c100c2b56fbc6a7dd17b4bc040a78bd4a8e380c8122a3795c20f165209` |
| study 1, table_B.csv | `8f00ad6858b9c949694fa9b5d05c98a100b013416065f071636d7352d5ac127b` |
| study 1, table_B_notes.txt | `38466cd1880e7f359f1d0677233816a92b67ac8a85b04a9297080e789fdc36f0` |
| study 2, criteria_s2_dev_s2_heldout.json | `be59225bc151f396244a4c54a3670c375b83bfaa4f7b7bd9286889984f8bba23` |
| study 2, derived number file (speedpath-s2-derived.tex) | `fe83f279dda52d8e39b3268617d84884690dd8ea179f20f2a793178a4af5189e` |
| study 2, design record (freeze) | `e19e2aa7851f4767ff65101dc7090e804c44ee19341475d7466b1084b51bdaeb` |
| study 2, design record (step 0) | `cd2b4690e0ff194dc994817634d4ac74f9b35afa53fb6110fa6205c53c822e70` |
| study 2, freeze: bundle | `61cd6f2f2f19d5f68873ed257a8c9091af5d10c5a9b94247f9d51ae1aafb9e3c` |
| study 2, freeze: proof (stamps/study2-freeze.ots) | `44ce136f95505a4d218d8b8542f3a7494ebc77ff8b62cf3a311adc88115988a8` |
| study 2, number file (speedpath-s2.tex) | `d6c5efafe3e2c7849dcf6f1598e55afc101cd84d9a6472e03987084e83b7ff69` |
| study 2, outcome_s2_dev_s2_heldout.json | `62d189aaae3db866d83034069a5acb726682b4f9c96c08b33513ad6ff06d9c22` |
| study 2, s_place_dev_s2_heldout_place.json | `94abfdd1b0c9dc459402446099065396500941e811616184164b10b7e31be7a2` |
| study 2, step 0: bundle | `18c5a9d0c5ff56c0eb73eff05e604de3eaa1073fbaed11e2ae1d514668c60b74` |
| study 2, step 0: proof (stamps/study2-step0.ots) | `8e96c55ba7a2499fa9ff157c23400fa6a4c8e1824dbc17e69cacb908e410a497` |
| study 2, step0_decision.json | `c0d75a8c1214c2dea2fa92c856e0072bda27fe0d1f63cc899b0b80f6f1cead3a` |
| study 2, step0_report.json | `e3d260a9930a0328d29e573a0f5f16577e430bf28eeff0183723b4d3d2a419d2` |
| study 3, detection_runs.json (detection rules with the urban estimator: the five runs) | `15540abb987eff375d2faf5a74e99418782c178ec139ce41c076d573c23f2059` |
| study 3, heldout_U2.json (urban LOS sites of study 2, TR 37.885) | `66db2413b62c81d22b6a39f70979d691cac9983511a63122e04a87df1dfec2a3` |
| study 3, heldout_U2_inhouse.json (urban LOS sites of study 2, our channel model) | `a98518939feb69f1765fcc5a5e3542d99ffc1d7e9edbbabcf070e2fca5fe220f` |
| study 3, heldout_in_ref.json (new sites, survey in traffic, our channel model) | `babc30524c4b21dfce91e2d080cee2d1d6f9037ed52d2d645e170d5a6e254d26` |
| study 3, heldout_in_reff.json (new sites, free-road survey, our channel model) | `1a2a56853cf32fdcdaac3034233eac9eae91a0a0824f04f37f411d4c6e194eb9` |
| study 3, heldout_ref.json (new sites, survey in traffic, TR 37.885) | `ddae6cd33bb47afe1a2f4d66acc6fe86cccea4602ba60aa126d12b5c2c284d17` |
| study 3, heldout_reff.json (new sites, free-road survey, TR 37.885) | `41d2be6afcc17b61b9beed5b4981203dab879c6584be36f3c09fd3065b35ece8` |
| study 3, highway_A_sidecheck.json (highway LOS sites of study 1) | `01277832a321e28f7c2353643379a8b3b27b352de9ac0ce106288c9216b38151` |
| study 3, number file (speedpath-s3.tex) | `0c2e2abb909724e8f32724217ef3c7aba90a3d31ddf7ee59c8909c76c26046f1` |
| study 3, per_lane_A.txt (mean error per lane, sites of studies 1 and 2) | `a4eb5977ece9f0d295c9c2a572e5f015c875b3d2f91913823a82733ff9a2e8c8` |
| study 3, per_lane_new_sites.txt (mean error per lane, new sites) | `41e7058af675671310bba4013f66e9f7988440821713e933460e3070dad6c8fc` |
| study 3, s_place_A.json (snapshot placement, five further sites) | `5ecf6a733a7a04337f810d2d4bfc023f054ca6cae8e61baa00cf58fbacd538a9` |
| study 3, urban_rows.csv (detection rules with the urban estimator: every urban and suburban row) | `a8526e4a9855df3cb9664d45f0dd6bd43ff0f83522a2239446593f6c5334b191` |
| study1_constants.json | `5bd63f356854c60a2f81977badb3fe2d27dc2b04c2f2faccf903fa62caa82238` |
| sweep of our channel model, inhouse_k4_sens_inhouse.json | `41e0a5a801f8c14c241f5fab4a0e933362b58555ed2fdefc2f25e3fe8a08ebfe` |
| sweep of our channel model, inhouse_sweeps.py | `daabfd1c5e0291ee927d4eeda3a30df882e73d8536ac46667cd6abdf687141ea` |
