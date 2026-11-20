# Speed check with a survey in traffic (study 2): full record

This file is the full record of the variant of the speed check with a survey in traffic ("study 2" of the supplementary
material of the SP2UBI paper). The paper reports the variant in Section VI-D, and Section S8.1 of the supplement
summarizes it. The variant was specified in advance, evaluated once on held-out simulated sites and did not meet its own
criteria; the paper's urban results use the urban estimator instead (Section S8.2 of the supplement).

Terms. Study 1 is the evaluation behind the per-cell table of the paper (Table 8). Cells are named by scenario and
speed: P1 is an urban street site with line of sight (LOS), P2 a highway site with line of sight, P3 an urban street
site where vehicles in the nearer lane block one head; the number is the speed in m/s. An evaluation variant combines a
signal-to-noise mapping (e1, our own link model, or lb, the TR 37.885 link budget) with a beacon bandwidth of 10, 20 or
50 physical resource blocks (PRBs). "Registered" refers to the timestamped design records (Section S2 of the
supplement). The development data are development set 1 (the development-pilot records of study 1), development set 2
(the urban and blocked-crossing records of study 1) and development set 3 (five study-1 urban sites simulated again with
paired free-flow control crossings). Step 0 is the development step; the labels D0-R1 to D0-R4, D0.1 to D0.6, E-a to
E-e, L-B and L0 to L3 are those of the design record and of the stored files
`speedcheck_sim/results/study2/step0_decision.json` and `step0_report.json`. The numbers of the section on step 0 are
development values, not held-out results. The stored outputs are in `speedcheck_sim/results/study2/`.

### Design

Study 2 was designed after the results of study 1 had been seen. It replaces the free-road survey of urban street sites
by a survey in traffic. Its levers (estimator candidates and rules for blocked crossings) were chosen on development
data by rules fixed before any lever statistic was computed (the section on step 0), and it was evaluated once, on
held-out sites whose seeds are derived from the stamped freeze bundle of study 2 and the Bitcoin block that attests it
(the table of timestamps in Section S2 of the supplement). The held-out evaluation was prescribed for every outcome of
step 0 and was run with that outcome known; how the development outputs were read and how the held-out run was carried
out are stated in Section S2 of the supplement. The evaluation covers 10 held-out urban LOS sites at 10 and 15 m/s and
10 held-out urban sites with blocked crossings at 15 m/s, under the coverage criteria of study 1; as there, each count
$k$ of qualified sites is that of the least favorable of six evaluation variants (the section on the checks per
evaluation variant).

Each urban LOS cell is rated on a ladder. The validity rung (L1) requires at least one qualified site, a calibrated bias
shown to lie within ±3.0 % in every variant by a Holm-adjusted equivalence test, no detected excess of the bias between
sites, and at most one qualified site with a test error standard deviation above 9 %. The precision rung (L2) adds a
qualified share of at least 0.3 and a governing per-crossing error standard deviation of at most 6 %. The decision level
(L3) requires, in addition, that the urban stratum meets the unchanged decision-level rule of study 1, with the honest
bound of the detection rules (Fig. 6 of the paper), and that every cell it counts as covered reaches L2. A cell that
fails the validity rung is at the lowest rung, L0. An error is the deviation $\hat b$ of Eq. (9) of the paper for a
speed sensor that reports the true speed, so a negative error means that the roadside estimate exceeds the true speed.

| Site type              | $v$ | $k/n$ \[CP\]       | $\sigma_{\mathrm{TR}}$ | $\sigma_{\mathrm{in}}$ | $\sigma_{\mathrm{gov}}$ | q99 $|e|$          | OOW             | gate            | yield             | bias, all sites \[CI\] | bias, qual.       | optimism, $R$                  | rating             |
|:-----------------------|:----|:-------------------|:-----------------------|:-----------------------|:------------------------|:-------------------|:----------------|:----------------|:------------------|:-----------------------|:------------------|:-------------------------------|:-------------------|
| urban LOS              | 10  | 0/10 \[0.0–30.8\]  | 5.9<sup>a</sup>        | 8.1<sup>a</sup>        | 9.1<sup>a</sup>         | 38.8<sup>a</sup>   | 4.0<sup>a</sup> | 2.1<sup>a</sup> | 100.0<sup>a</sup> | 7.0 \[−1.7, 17.4\]     | −1.0<sup>a</sup>  | not run                        | not covered; L0    |
| urban LOS              | 15  | 4/10 \[12.2–73.8\] | 7.3                    | 6.0                    | 11.2                    | 68.3               | 1.7             | 2.9             | 98.7              | 7.8 \[−3.2, 27.5\]     | 2.1 \[−2.0, 6.4\] | contradicted, 1.84<sup>b</sup> | partly covered; L0 |
| urban, blocked (NLOSv) | 15  | 0/10 \[0.0–30.8\]  | 9.6<sup>a</sup>        | 7.3<sup>a</sup>        | 14.8<sup>a</sup>        | 1002.0<sup>a</sup> | 6.7<sup>a</sup> | 5.0<sup>a</sup> | 100.0<sup>a</sup> | −6.8 \[−15.0, −0.6\]   | −35.3<sup>a</sup> | not run                        | not covered        |

**Table 1.** Study 2: speed check under the TR 37.885 channel at the held-out urban sites (simulation). Columns named as
in Table 8 of the paper have the same meaning; $v$ in m/s, all other values in %. Optimism, $R$: verdict and ratio of
the channel-agreement check of Section S3 of the supplement, called the optimism check in this section. Bias, all sites:
the calibrated bias of the validity rung (between-site check over all sites with a valid survey), largest in absolute
value over the variants, with its 95 % site-bootstrap interval; bias, qual.: the calibrated bias at qualified sites with
gated crossings; rating: coverage rating and rung (the blocked crossings are not rated on the ladder).

<sup>a</sup>No site qualified at the least favorable variant. The columns computed on qualified sites take the worst
value over the variants in which a site qualified (yield and calibrated bias: the variant of $\sigma_{\mathrm{TR}}$), so
each marked value rests on the sites that qualified at another variant: one site, except the gate value of the 10 m/s
row (4 sites); no interval is given. In both marked rows the optimism check was not run (too few qualified sites with
test crossings). q99 is the worst value over the variants and both channel models; in the blocked row it comes from the
crossings of one site. <sup>b</sup>Includes the governing urban snapshot-placement factor (1.54, study 1, the larger of
the two studies' factors by the registered rule; the section on the snapshot-placement check); without it $R = 1.20$ (95
% CI 0.67–2.79, 5 sites) and the check is undetermined.

### Held-out results

Table 1 gives the held-out results. The bias statistic of the validity rung is the calibrated bias of the between-site
check over all sites with a valid survey, with its site-bootstrap interval and its Holm-adjusted equivalence test on the
same draws; it differs from the statistic of the mean error after the survey calibration in Section VI-D of the paper,
which uses qualified sites and gated crossings, and both are printed.

*10 m/s.* The survey in traffic qualified no site (0/10, 95 % CI 0.0–30.8 %; 0–4 sites over the six variants). Its rank
correlation with the standard deviation of the test crossings is 0.99 over 10 sites (0.77–0.99 over the variants), so
its rejection of every site at the least favorable variant, where the per-crossing error at the unqualified sites is
16.0 %, is a valid rejection, not a failure of qualification validity. The cell is at L0 because no site qualified; in
addition, its largest between-site bias in absolute value is 7.0 % (95 % CI −1.7 to 17.4 %), and its equivalence test
passed in 0 of the 6 variants.

*15 m/s.* The survey in traffic qualified 4 of 10 sites (95 % CI 12.2–73.8 %; 4–5 over the variants), a share that
reaches the rating "partly covered"; 0 of them had a test error standard deviation above 9 %, and the rank correlation
is 0.83 over 10 sites (0.83–0.95 over the variants). The cell nevertheless stays at L0: the equivalence test passed in 0
of the 6 variants (largest between-site bias in absolute value 7.8 %, 95 % CI −3.2 to 27.5 %), an excess of the bias
between sites was detected in 4 of them, and the governing per-crossing error standard deviation, 11.2 %, exceeds 6 %,
as does its value before the snapshot-placement factor ($\sigma_{\mathrm{TR}} = 7.3$ %), so the precision rung would
fail as well. The optimism check reads "contradicted" ($R = 1.84$) only through the study-1 snapshot-placement factor;
without it the check is undetermined in 6 of 6 variants, so study 2 neither confirms nor refutes the optimism verdict of
study 1 at this cell. No cell reaches L2, so the decision level is not reached and the urban stratum stays not covered.

*Blocked crossings.* They are reported as in study 1 and are not on the ladder. At these held-out sites, 59.3 % of the
traffic crossings had a head blocked by a vehicle in the nearer lane for part of the crossing (the survey in traffic
includes such crossings), and no site qualified under the TR 37.885 channel (0/10, 95 % CI 0.0–30.8 %). At most 1 site
qualified in each of the six variants, so the qualified-site values of this row each rest on a single qualified site,
not necessarily the same one.

*Step 0 in brief.* On development data, step 0 supports a kinematic cause of the urban bias under one traffic model
(SUMO, Krauss car following, degree of saturation 0.7): it concerns the mean error, not its spread, was not tested on
held-out sites and changed no part of the configuration (the section on step 0). None of the seven estimator candidates
and neither of the two rules for blocked crossings was adopted, so highway processing is unchanged and the highway check
of study 2 was not run. The dependence of the within-site error on the Ricean $K$-factor is descriptive (our channel
model) and enters no rule of study 2.

## Checks per evaluation variant

Table 2 gives, for every cell and variant, the qualified sites $k$, the rank correlation $\rho_s$ over all sites between
the survey's $\hat\sigma$ and the $\sigma$ of the test crossings (a registered check of qualification validity), the
optimism check and the bias tests of the validity rung. For the blocked-crossing cell, $\rho_s$ is 0.36 at the variant
that governs the cell in the validity record (lb, 10 PRBs). The optimism check runs only with at least four qualified
sites with test crossings; its ratios in the table are unscaled, and with the snapshot-placement factor of the
registered rule (1.54) the 15 m/s cell reads "contradicted" ($R = 1.84$), as in Table 1. The equivalence test of the
validity rung, with the $\pm 3.0$ % band and Holm over the three cells per variant, passed in none of the variants
(smallest Holm-adjusted $p$ 0.243 at 10 m/s, 0.197 at 15 m/s and 0.317 for blocked crossings). The registered test with
the $\pm 0.5$ % band, reported and not gating, also failed in every cell and variant (Holm-adjusted $p = 1.000$ in all
18).

| Cell  | Variant | $k$ | $\rho_s$ | Sites | Optimism, $R$      |     Bias \[95 % CI\] | $p_{\mathrm{Holm}}$ | Excess |
|:------|:--------|----:|---------:|------:|:-------------------|---------------------:|--------------------:|:------:|
| P1-10 | e1/10   |   0 |     0.92 |     0 | not run            |    2.4 \[−1.7, 6.7\] |               0.774 |   no   |
| P1-10 | e1/20   |   1 |     0.95 |     1 | not run            |   7.0 \[−1.7, 17.4\] |               1.000 |  yes   |
| P1-10 | e1/50   |   4 |     0.85 |     4 | undetermined, 1.34 |   −0.4 \[−4.3, 2.9\] |               0.243 |   no   |
| P1-10 | lb/10   |   0 |     0.99 |     0 | not run            |    4.0 \[−1.9, 8.8\] |               1.000 |   no   |
| P1-10 | lb/20   |   1 |     0.95 |     1 | not run            |    4.2 \[−1.1, 9.2\] |               1.000 |  yes   |
| P1-10 | lb/50   |   4 |     0.77 |     4 | undetermined, 1.36 |   −1.7 \[−6.2, 1.6\] |               0.497 |   no   |
| P1-15 | e1/10   |   5 |     0.83 |     5 | undetermined, 1.20 |    1.5 \[−0.7, 3.8\] |               0.308 |   no   |
| P1-15 | e1/20   |   4 |     0.83 |     4 | undetermined, 1.06 |   6.6 \[−5.0, 25.8\] |               1.000 |  yes   |
| P1-15 | e1/50   |   4 |     0.95 |     4 | undetermined, 0.82 |    0.9 \[−2.1, 3.9\] |               0.243 |  yes   |
| P1-15 | lb/10   |   4 |     0.83 |     4 | undetermined, 1.14 |   −0.7 \[−3.8, 2.0\] |               0.197 |   no   |
| P1-15 | lb/20   |   4 |     0.83 |     4 | undetermined, 1.09 |   7.8 \[−3.2, 27.5\] |               1.000 |  yes   |
| P1-15 | lb/50   |   4 |     0.94 |     4 | undetermined, 0.82 |    1.0 \[−1.7, 3.8\] |               0.219 |  yes   |
| P3-15 | e1/10   |   1 |     0.27 |     1 | not run            |  −6.1 \[−15.1, 1.4\] |               0.774 |   no   |
| P3-15 | e1/20   |   0 |     0.81 |     0 | not run            |  −4.3 \[−12.6, 1.4\] |               1.000 |   no   |
| P3-15 | e1/50   |   1 |     0.83 |     1 | not run            |   −0.8 \[−7.6, 8.1\] |               0.317 |   no   |
| P3-15 | lb/10   |   1 |     0.36 |     1 | not run            | −6.8 \[−15.0, −0.6\] |               1.000 |   no   |
| P3-15 | lb/20   |   0 |     0.33 |     0 | not run            |  −2.9 \[−10.6, 2.9\] |               1.000 |   no   |
| P3-15 | lb/50   |   1 |     0.83 |     1 | not run            |   −0.9 \[−7.5, 7.3\] |               0.497 |   no   |

**Table 2.** Study 2, held-out sites ($n = 10$ per cell): checks per cell and variant (mapping/PRBs). $k$: qualified
sites; $\rho_s$: rank correlation; optimism: sites with test crossings, verdict and unscaled ratio $R$; bias:
between-site bias (%) with its 95 % interval; $p_{\mathrm{Holm}}$: Holm-adjusted $p$ of the $\pm 3.0$ % equivalence
test; excess: a detected excess of the bias between sites.

## Step 0 and the estimator candidates

### Cause (registered rule D0-R1)

Each simulated traffic crossing of the five sites of development set 3 was paired with a free-flow control crossing in
the same lane. (In the registration and the code this free-flow control track is called the "quadratic control".) The
rule holds ("kinematic cause supported") if, in at least one urban cell, the mean paired difference (traffic minus
control) is at least half of the excess bias with an interval excluding zero, read at the variant with the largest bias.
At that variant the mean paired difference is −14.4 % at 10 m/s (excess bias −13.9 %) and −10.0 % at 15 m/s (excess bias
−11.0 %); at the two 50-PRB variants the intervals of the paired difference include zero (Table 3).

| Cell  | Variant     | Excess bias | Mean paired difference | 95 % interval | Pairs |
|:------|:------------|------------:|-----------------------:|:-------------:|------:|
| P1-10 | e1/10       |       −12.8 |                  −15.2 | −28.5 to −4.2 |   295 |
| P1-10 | e1/20       |       −13.4 |                  −14.1 | −23.8 to −4.6 |   294 |
| P1-10 | e1/50       |        −1.0 |                   −0.9 | −19.0 to 19.2 |   284 |
| P1-10 | lb/10       |       −12.5 |                  −14.9 | −28.0 to −3.9 |   295 |
| P1-10 | lb/20$^{*}$ |       −13.9 |                  −14.4 | −24.4 to −4.6 |   296 |
| P1-10 | lb/50       |        −0.5 |                   −0.7 | −18.0 to 19.5 |   283 |
| P1-15 | e1/10       |       −10.2 |                   −9.9 | −19.1 to −3.9 |   297 |
| P1-15 | e1/20       |       −10.9 |                   −9.9 | −21.1 to −2.8 |   286 |
| P1-15 | e1/50       |         0.6 |                    0.8 | −7.1 to 14.3  |   275 |
| P1-15 | lb/10       |       −10.1 |                   −9.8 | −18.7 to −3.8 |   297 |
| P1-15 | lb/20$^{*}$ |       −11.0 |                  −10.0 | −21.3 to −2.8 |   286 |
| P1-15 | lb/50       |         0.7 |                    0.9 | −7.3 to 14.6  |   274 |

**Table 3.** Step 0, rule D0-R1 per variant ($^{*}$ variant with the largest bias). Excess bias: mean error of the gated
traffic crossings minus that of the free-road reference crossings. All values in %.

### Estimator candidates and rules for blocked crossings

The registration admitted at most one estimator candidate, judged in a fixed order (the letters are the registration's
labels): E-a, a window-restricted statistic ($\pm 4$ s around the crossing midpoint); E-b, the matched statistic forced
at every site; E-c, a fitted-acceleration gate ($a_{\max} = 1.35$ m/s$^2$, the 99th percentile over development survey
crossings); E-f, a spatial-core statistic over the beacons of the central part of the crossing; E-d, a roadside detector
of a stop within the crossing, "no entry" variant ($k_{\mathrm{stop}} = 0.69$); E-db, E-d combined with E-f; and E-e, a
two-segment speed profile with one fitted knot. Rule D0-R2 admitted E-e only if the crossings without a stop carry at
least half of the excess bias in development sets 1 and 2; it holds in both cells (`step0_decision.json`, key `D0_R2`);
in development set 1 at 15 m/s no gated crossing stopped between the heads, so the share there is 100 % by construction.
A candidate is eligible if, on the worse of both channel models, at both urban cells, in development set 2 under both
readings, its absolute excess bias falls by at least 50 % or is at most 0.5 %, its error spread falls by at least 20 %,
its 99th-percentile error rises by at most 5 %, its entry yield falls by at most 30 % and its non-finite share stays at
most 5 %; development set 1 is a non-regression guard. None of the seven was eligible (Table 4), so no estimator change
was adopted.

| Candidate or rule | Reduction (%) | Failed | Failing clauses                                                                            |
|:------------------|--------------:|-------:|:-------------------------------------------------------------------------------------------|
| E-a               |         −18.2 |      6 | error spread (6); excess bias (5); q99 (4)                                                 |
| E-b               |           3.7 |      5 | error spread (5); excess bias (4); q99 (4)                                                 |
| E-c               |          27.6 |      4 | excess bias (4); error spread (3)                                                          |
| E-f               |          34.7 |      3 | q99 (2); excess bias (2)                                                                   |
| E-d               |           7.7 |      4 | excess bias (4); error spread (4); entry yield (1)                                         |
| E-db              |          36.2 |      4 | q99 (2); entry yield (1); excess bias (2)                                                  |
| E-e               |          −6.0 |     10 | error spread (10); excess bias (5); q99 (4); non-finite share (4); out-of-window share (2) |
| L-B               |           7.7 |      6 | excess bias (6); calibrated bias (2)                                                       |
| L-B$^{\prime}$    |          13.8 |      6 | excess bias (6); calibrated bias (2)                                                       |

**Table 4.** Estimator candidates (registered rule D0-R3) and rules for blocked crossings (rule D0-R4). Reduction: for a
candidate, the mean reduction of the governing error spread over the cells of development set 2 (survey reading); for a
rule, $1 -$ its mean absolute excess bias over that of least squares. Failed: entries failing at least one clause;
clauses with the number of failing entries.

For blocked crossings two rules were registered: L-B, no entry when a head's per-beacon power drops by more than
$\Delta_{\mathrm{blk}} = 16.9$ dB below the crossing's fitted large-scale trend, and L-B$'$, no entry when the fitted
lateral distance $\hat\eta$ (Eq. (8) of the paper) departs from the survey median of every surveyed lane by more than
$\Delta_\eta = 19.1$ m (both limits set for at most 1 % false flags on development survey crossings). A rule is adopted
if its absolute excess bias falls by at least 50 % or is at most 0.5 % at a yield loss of at most 30 %, on every
registered entry (development sets 1 and 2, each channel model; in the survey-in-traffic reading also the calibrated
bias). Neither qualified (Table 4). The four limits $a_{\max}$, $k_{\mathrm{stop}}$, $\Delta_{\mathrm{blk}}$ and
$\Delta_\eta$ are development calibration values: the held-out configuration does not use them, because no candidate or
rule that uses them was adopted.

### Step-0 diagnostics

Besides rule D0-R1 (diagnostic D0.2), the registration lists five step-0 diagnostics: bias and spread by stratum (D0.1),
coverage of the $\pm 4$ s window (D0.3), the stop detector of E-d (D0.4), the traffic survey on development set 2 (D0.5)
and blocked crossings (D0.6). They are descriptive, enter no rule and are given in full in `step0_report.json`. Table 5
gives D0.5, the development evidence for the survey in traffic: for the free-road and the traffic survey, the qualified
sites, the $\sigma$ of the test crossings at qualified and unqualified sites, the qualified sites whose traffic error
exceeds 9 % and the rank correlation $\rho_s$.

| Survey    | Cell  | Variant | $k/n$ | $\sigma$ qualified | $\sigma$ unqualified | over flag | $\rho_s$ |
|:----------|:------|:--------|------:|-------------------:|---------------------:|----------:|---------:|
| Survey    | Cell  | Variant | $k/n$ | $\sigma$ qualified | $\sigma$ unqualified | over flag | $\rho_s$ |
| free road | P1-10 | e1/10   |  5/10 |               12.2 |                 14.3 |       3/5 |    0.770 |
| free road | P1-10 | e1/20   |  5/10 |                9.7 |                 14.1 |       2/5 |    0.442 |
| free road | P1-10 | e1/50   |  9/10 |               11.4 |                 12.8 |       6/9 |    0.406 |
| free road | P1-10 | lb/10   |  4/10 |               12.4 |                 13.9 |       2/4 |    0.770 |
| free road | P1-10 | lb/20   |  5/10 |                9.8 |                 14.0 |       2/5 |    0.442 |
| free road | P1-10 | lb/50   |  9/10 |               11.6 |                 12.9 |       6/9 |    0.358 |
| free road | P1-15 | e1/10   |  6/10 |                7.9 |                 13.7 |       2/6 |    0.697 |
| free road | P1-15 | e1/20   |  6/10 |                8.5 |                 11.3 |       2/6 |    0.600 |
| free road | P1-15 | e1/50   | 10/10 |                7.0 |                  n/a |      2/10 |    0.345 |
| free road | P1-15 | lb/10   |  6/10 |                7.9 |                 13.8 |       2/6 |    0.697 |
| free road | P1-15 | lb/20   |  6/10 |                8.5 |                 11.3 |       2/6 |    0.600 |
| free road | P1-15 | lb/50   | 10/10 |                7.0 |                  n/a |      2/10 |    0.345 |
| traffic   | P1-10 | e1/10   |  1/10 |                6.0 |                 16.8 |       0/1 |    0.976 |
| traffic   | P1-10 | e1/20   |  1/10 |                4.8 |                 16.9 |       0/1 |    0.867 |
| traffic   | P1-10 | e1/50   |  4/10 |                6.7 |                 17.7 |       1/4 |    0.952 |
| traffic   | P1-10 | lb/10   |  1/10 |                5.9 |                 17.2 |       0/1 |    0.939 |
| traffic   | P1-10 | lb/20   |  1/10 |                4.8 |                 16.9 |       0/1 |    0.842 |
| traffic   | P1-10 | lb/50   |  4/10 |                6.7 |                 16.8 |       1/4 |    0.915 |
| traffic   | P1-15 | e1/10   |  1/10 |                4.4 |                 12.0 |       0/1 |    0.661 |
| traffic   | P1-15 | e1/20   |  4/10 |                5.4 |                 12.5 |       0/4 |    0.806 |
| traffic   | P1-15 | e1/50   |  8/10 |                3.8 |                 14.9 |       0/8 |    0.903 |
| traffic   | P1-15 | lb/10   |  1/10 |                4.4 |                 11.8 |       0/1 |    0.624 |
| traffic   | P1-15 | lb/20   |  4/10 |                5.4 |                 12.4 |       0/4 |    0.806 |
| traffic   | P1-15 | lb/50   |  8/10 |                3.8 |                 15.0 |       0/8 |    0.903 |

**Table 5.** D0.5: qualification validity of the two surveys (development set 2; $\sigma$ in %).

## Snapshot-placement check of study 2

The registered light check ran on 5 sites at 15 m/s urban line of sight (99 of its 100 registered passes entered in both
mappings; one failed the registered window rule). Its statistic $\hat\rho$ is the ratio of the error standard deviation
with the channel sampled between the beacons as well as at them to that with the channel sampled at the beacons only. It
gave $\hat\rho = 0.98$ (95 % CI 0.90–1.11) under the e1 mapping and 0.96 (0.87–1.16) under the link-budget mapping; the
point estimates lie below the registered trigger 1.10, so its factor is 1.00 (no correction). The intervals reach 1.11
and 1.16, so a placement factor of that size is not excluded, and the larger study-1 factor, 1.54, governs the urban
values by the registered rule.

## How the outputs were read

In study 2, the development outputs of step 0 were read as they arrived, not once after all inputs were complete; this
was decided before the first read, applied to development data only, and changed no rule, threshold or statistic. A read
with three of the five sites of development set 3 found the 15 m/s criterion of rule D0-R1 unmet and the 10 m/s result
carried by one site; with all five it holds in both cells, and the decision file was written only from the complete
inputs. No held-out value was read before every held-out run had finished its final stage. Interrupted runs were
recomputed from their seeds and gave identical outputs; no site was dropped, selected or re-drawn. After the held-out
results were read, only texts and the scripts that format stored values for printing changed; no stored output, design
constant, rule, rung or branch changed.
