# QuaDRiGa: third-party modified file

Modified file: `quadriga_src/@qd_builder/get_channels.m` of QuaDRiGa v2.8.1 (git 277866650eb115adb5b3e8ac252b0d1df073596d,
https://github.com/fraunhoferhhi/QuaDRiGa), changed on 2026-10-10 by the SP2UBI authors.

Change: an optional fifth input `ray_out`. When it is set, every output channel also carries `par.ray_coeff` (per-ray
coefficients with the per-cluster factors applied) and `par.ray_cluster` (cluster index of each ray). With `ray_out = 0`
the function is unchanged: every addition is inside an `if ray_out` block (see `get_channels.diff`). Purpose: the
per-ray scatterer-mobility term of 3GPP TR 37.885 V15.3.0, clause 6.2.3.

QuaDRiGa is used under its Software License (non-commercial scientific use). The licence text is reproduced in
`QuaDRiGa_License.txt` next to this notice. This repository does not contain QuaDRiGa itself: `build_patched_tree.sh`
copies an unmodified QuaDRiGa checkout and applies `get_channels.diff` to the copy.
