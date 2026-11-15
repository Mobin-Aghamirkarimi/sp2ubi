r"""Study 3: the number file of the urban-estimator evaluation, from the stored outputs in results/study3/.

Reads heldout_ref.json and heldout_reff.json (TR 37.885; survey in traffic and free-road survey), heldout_in_ref.json
and heldout_in_reff.json (our channel model), s_place_A.json (snapshot placement), per_lane_new_sites.txt (mean error
per lane), highway_A_sidecheck.json (highway LOS sites of study 1) and the study-1 snapshot-placement factor of the
urban LOS stratum (key A|P1-15|s_place of results/reports/primary/speedpath-numbers.tex), and writes the keys S3|...
of speedpath-s3.tex. Every value is read or is a minimum, a maximum or a product of read values; none is recomputed
from records. Percentages carry one decimal.

Keys S3|<cell>|...: cell = survey (T in traffic, F free road) and speed class (a = 10 m/s, b = 15 m/s);
suffix base = base estimator (ls in the data files), urb = urban estimator (h_pp_now_c20).
  k_*, n, cp_lo_*, cp_hi_*          qualified sites and the Clopper-Pearson interval (TR 37.885); k_in_*, n_in: our model
  sigma_tr_*, sigma_in_*            error at the qualified sites, gated crossings, per channel model
  bias_cal_*, bias_lo_*, bias_hi_*  mean error after calibration and its interval (bias_in_*: our channel model)
  oow_*, q99_*                      out-of-window share and 99 % quantile of the absolute error
  sc_one, sc_gov                    the larger of sigma_tr and sigma_in, without and with the study-1 placement factor
Keys S3|<name>_min / _max are the range of a quantity over the four cells; S3|place_* the snapshot-placement ratios;
S3|lane_min / lane_max the far-lane minus near-lane mean error; S3|hw|... the highway side check.
Usage: python tr37885/reports/study3_values.py --check          (fails if the stored file differs after its first line)
       python tr37885/reports/study3_values.py --out <file>     (writes the number file)"""
import argparse
import json
import os
import re
import sys

ER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ER, ".."))
from fdoa import urban as U  # noqa: E402

S3 = os.path.join(ER, "..", "results", "study3")
OUT_TEX = os.path.join(S3, "speedpath-s3.tex")
NUMBERS_1 = os.path.join(ER, "..", "results", "reports", "primary", "speedpath-numbers.tex")
HEADER = "% written by reports/study3_values.py from the stored study-3 outputs; do not edit"
BASE, URBAN = "ls", "h_pp_now_c20"
HIGHWAY_SPEEDS = (20, 30, 38)


def pct(x, signed=False):
    v = round(100 * x + 0.0, 1)
    v = 0.0 if v == 0 else v
    t = f"{abs(v):.1f}"
    if v < 0:
        return r"\ensuremath{-}" + t
    return (r"\ensuremath{+}" + t) if signed and v > 0 else t


def study1_placement_factor():
    m = re.search(r"\\expandafter\\def\\csname spn@A\|P1-15\|s_place\\endcsname\{([0-9.]+)\}", open(NUMBERS_1).read())
    return float(m.group(1))


def build():
    load = lambda name: json.load(open(os.path.join(S3, name)))  # noqa: E731
    K, ranges = {}, {}
    s_gov = study1_placement_factor()
    for d, role in (("T", "ref"), ("F", "reff")):
        tr, ih = load(f"heldout_{role}.json"), load(f"heldout_in_{role}.json")
        for v, suf in ((10, "a"), (15, "b")):
            c = f"S3|{d}{suf}|"
            K[c + "v"] = str(v)
            for tag, var in (("base", BASE), ("urb", URBAN)):
                a, b = tr[f"{var}|{v}"], ih[f"{var}|{v}"]
                K[c + f"k_{tag}"], K[c + "n"] = str(a["k"]), str(a["n"])
                K[c + f"sigma_tr_{tag}"] = pct(a["sigma_qg"])
                K[c + f"sigma_in_{tag}"] = pct(b["sigma_qg"])
                K[c + f"k_in_{tag}"], K[c + "n_in"] = str(b["k"]), str(b["n"])
                K[c + f"bias_cal_{tag}"] = pct(a["bias_cal"], True)
                K[c + f"bias_lo_{tag}"], K[c + f"bias_hi_{tag}"] = (pct(x, True) for x in a["bias_cal_ci"])
                K[c + f"bias_in_lo_{tag}"], K[c + f"bias_in_hi_{tag}"] = (pct(x, True) for x in b["bias_cal_ci"])
                K[c + f"cp_lo_{tag}"], K[c + f"cp_hi_{tag}"] = (pct(x) for x in a["cp"])
                K[c + f"oow_{tag}"], K[c + f"q99_{tag}"] = pct(a["oow"]), pct(a["q99"])
                for name, val in ((f"sigma_tr_{tag}", a["sigma_qg"]), (f"bias_lo_{tag}", a["bias_cal_ci"][0]),
                                  (f"bias_hi_{tag}", a["bias_cal_ci"][1]), (f"oow_{tag}", a["oow"]),
                                  (f"sigma_all_{tag}", a["sigma_all"]), (f"q99_{tag}", a["q99"]),
                                  (f"scg_{tag}", max(a["sigma_qg"] * s_gov, b["sigma_qg"]))):
                    ranges.setdefault(name, []).append(val)
                if tag == "urb":
                    K[c + "sc_gov"] = pct(max(a["sigma_qg"] * s_gov, b["sigma_qg"]))
                    K[c + "sc_one"] = pct(max(a["sigma_qg"], b["sigma_qg"]))
                    ranges.setdefault("sc_gov", []).append(max(a["sigma_qg"] * s_gov, b["sigma_qg"]))
                    ranges.setdefault("sc_one", []).append(max(a["sigma_qg"], b["sigma_qg"]))
    for name, vals in ranges.items():
        if name.startswith("q99"):                       # printed without decimals (values above 100 % occur)
            K[f"S3|{name}_min"], K[f"S3|{name}_max"] = f"{100 * min(vals):.0f}", f"{100 * max(vals):.0f}"
        else:
            signed = name.startswith("bias")
            K[f"S3|{name}_min"], K[f"S3|{name}_max"] = pct(min(vals), signed), pct(max(vals), signed)
    K["S3|n_sites"] = K["S3|Ta|n"]
    sp = load("s_place_A.json")
    K["S3|place_n"] = str(len(sp["sites"]))
    K["S3|place_base"] = f'{max(sp[m]["registered_recomputed"]["rho"] for m in ("e1", "lb")):.2f}'
    mu = max(("e1", "lb"), key=lambda m: sp[m]["A"]["rho"])
    K["S3|place_urb"] = f'{sp[mu]["A"]["rho"]:.2f}'
    K["S3|place_urb_lo"], K["S3|place_urb_hi"] = (f"{x:.2f}" for x in sp[mu]["A"]["rho_ci"])
    lanes = [float(m.group(1)) for m in re.finditer(r"lane [01]: n =\s*\d+ \(crossing x variant\), mean ([+-][0-9.]+) %",
                                                    open(os.path.join(S3, "per_lane_new_sites.txt")).read())]
    if len(lanes) != 8:
        sys.exit("study3_values: per_lane_new_sites.txt does not hold two surveys x two speeds x two lanes")
    diff = [lanes[i + 1] - lanes[i] for i in range(0, 8, 2)]
    K["S3|lane_min"], K["S3|lane_max"] = f"{min(diff):.1f}", f"{max(diff):.1f}"
    K["S3|core_pct"] = f"{100 * (U.CORE[1] - U.CORE[0]):.0f}"
    K["S3|node_s"] = f"{U.NODE_S:g}"
    hw = load("highway_A_sidecheck.json")
    for tag, var in (("base", BASE), ("urb", URBAN)):
        vals = [hw[f"{var}|{v}"]["sigma_qg"] for v in HIGHWAY_SPEEDS]
        K[f"S3|hw|sigma_{tag}_min"], K[f"S3|hw|sigma_{tag}_max"] = pct(min(vals)), pct(max(vals))
    return K


def render(K):
    return (HEADER + "\n\\makeatletter\n" + "".join(f"\\expandafter\\def\\csname spn@{k}\\endcsname{{{K[k]}}}\n" for k in sorted(K))
            + "\\makeatother\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--out", default=None)
    a = ap.parse_args()
    tex = render(build())
    if a.check:
        ok = os.path.exists(OUT_TEX) and open(OUT_TEX).read().split("\n", 1)[1] == tex.split("\n", 1)[1]
        print("study3_values: stored number file", "matches" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)
    open(a.out, "w").write(tex)


if __name__ == "__main__":
    main()
