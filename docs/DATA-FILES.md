# Data files of the decision-level evaluation

This file describes the columns of the data files behind the detection results of the SP2UBI paper (Fig. 6 and Section
VI-D). The supplementary material ships byte copies of them in its folder `data/`; in this repository they are
`speedcheck_sim/results/reports/primary/table_B.csv`, `speedcheck_sim/results/study3/urban_rows.csv` and
`speedcheck_sim/results/study3/detection_runs.json`. "Study 1" is the evaluation behind the per-cell table of the paper,
"study 3" the evaluation of the urban estimator. The column names are those of the code and imply no sanction:
"punitive" marks a flag of the residual rule and "escalation" a flag of the notice rule.

### Every row of study 1: the data file

The file `table_B.csv` holds all 129 rows of the decision-level evaluation of study 1, which used the base estimator at
every site: 12 rows of honest vehicles, 26 error-model rows, 4 rows with sites at twice their disclosed $\sigma$ and 87
attack rows (column `section`: honest, margin, badsites, power; column `key`: the row, with $\nu$ after its last bar).
With the base estimator no urban site and speed class met the stricter site condition of the detection rules, so the
urban stratum is not covered in this file: it holds no urban-profile row, and its suburban-profile rows rest on the
highway sites alone. The urban- and suburban-profile rows with the urban estimator are in the table of the values behind
the detection figure in Section S4 of the supplement and in the file `urban_rows.csv` (below). Each metric has three
columns: `<metric>` under TR 37.885, `inhouse_<metric>` under our channel model on its own site population, and
`governing_<metric>`, the worse of the two, taken for each metric separately. An empty field holds no value for that
row; the per-epoch columns, for example, are filled only in the honest-vehicle rows and only per channel. The columns
`source`, `inhouse_source` and `governing_source` label the origin of each group (site stratum, site population of our
channel model, worse of both).

The paper uses these metrics, each per vehicle-year: `punitive_per_year` and `punitive_ub95`, the share of vehicle-years
in which the residual rule flags, with its 95 % upper confidence bound (UB95); `escalation_per_year`, the share with at
least one notice-rule flag; and `punitive_union_per_year` and `punitive_union_ub95`, the share in which the residual or
the notice rule flags, with its UB95 (the detection and false-flag rates of Fig. 6 of the paper). The per-epoch columns
are `entries_per_epoch_mean`, `crossings_per_epoch_mean` and `low_m_share` (share of epochs with fewer than five
entries). The paper does not use the other columns; one of them, `inspections_per_year` (mean number of notice-rule
flags), depends on the example insurer reaction used in the simulation (Section S2 of the supplement, notice-rule
threshold). The consistency rule is not simulated in these rows (Section S4 of the supplement). The framing rows (key
`framing|`$n$`|-0.15|<profile>|<`$\nu$`>`) hold the flag rates of an honest vehicle framed by live relay: in every
epoch, $n$ relayed entries with a result of $-15$ % are attributed to it.

### Every urban and suburban row with the urban estimator: the data file

The file `urban_rows.csv` has 2,832 lines with every metric of every urban- and suburban-profile row of the four runs
with the urban estimator, one line per row and metric. The columns are: `section` (honest, margin, badsites, power as in
`table_B.csv`, and sens_tyre, the honest-sensor run of Section S4 of the supplement); `key`, the row, with the profile
and $\nu$ after its last two bars; `metric`, the metric name (ten are the names of `table_B.csv`; the other thirteen are
further outputs of the decision simulation, of which the paper uses only `low_m_epoch_share`, the share of urban epochs
at $\nu=5$ with too few entries); `direction`, high if a higher value is worse, low if a lower value is worse, and empty
for a descriptive metric, which has no worse value; `free_tr` and `free_inhouse`, the values with the free-road survey
under TR 37.885 and under our channel model, and `free_worse`, the worse of the two; `traffic_tr`, `traffic_inhouse` and
`traffic_worse`, the same with the survey in traffic; and `published_suburban`, filled in the suburban rows only, the
study-1 value under TR 37.885 with the highway sites alone (column `<metric>` of `table_B.csv`). The columns
`free_tableB_governing` and `traffic_tableB_governing` are by-products of the report script, which combines the TR
37.885 value with the earlier evaluation under our channel model with the base estimator; they are not used. The file
`detection_runs.json` gives, for each run, its status, the number of qualified combinations of site and speed class and
the number of test crossings (Section S4 of the supplement).
