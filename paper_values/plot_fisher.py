"""Supplement figure: pass sigma of the in-house channel model against the raw-sample SNR, with the Fisher bound.

Plots the stored rows of inhouse_k4_sens_inhouse.json (speedcheck_sim, in-house sweep at v0 = 15 m/s) with the
impairments off and a finite pass_sigma_pct: one solid curve (pass sigma) and one dashed curve (Fisher bound) per
Rice factor K. The values are plotted as stored.

Output: paper_values/figures/fisher_bound.pdf
Usage (from the repository root, environment requirements/speedcheck_sim.lock):  python paper_values/plot_fisher.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import speedcheck_paths as SP  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "figures", "fisher_bound.pdf")
ORDER = [0.0, 3.5, 6.0, 9.0, 15.0, None]          # Rice factor K in dB; None: K = infinity


def main():
    k4 = json.load(open(SP.find("inhouse_k4_sens_inhouse.json")))
    assert k4["kind"] == "k4" and k4["v0"] == 15.0
    curves = {}
    for r in k4["rows"]:
        if r.get("impairments") not in (None, "off"):
            continue
        ps = r.get("pass_sigma_pct")
        if ps is None or not math.isfinite(ps):
            continue
        curves.setdefault(r["K_db"], []).append((r["snr_db"], ps, r["fisher_pass_pct"]))
    assert sorted(curves, key=lambda k: (k is None, k)) == ORDER and sum(map(len, curves.values())) == 66
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = plt.get_cmap("viridis")
    for i, K in enumerate(ORDER):
        pts = sorted(curves[K])
        lab = r"$K = \infty$" if K is None else f"$K = {K:g}$ dB"
        col = cols(i / (len(ORDER) - 1) * 0.9)
        ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], "-o", ms=3, lw=1.2, color=col, label=lab)
        ax.plot([x for x, _, _ in pts], [f for _, _, f in pts], "--", lw=1.0, color=col)
    ax.set_yscale("log")
    ax.set_xlabel("raw-sample SNR (dB)")
    ax.set_ylabel(r"pass $\sigma$ (%)")
    ax.plot([], [], "k-", label=r"pass $\sigma$ (solid)")
    ax.plot([], [], "k--", label="Fisher bound (dashed)")
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print("written", os.path.relpath(OUT, SP.ROOT))


if __name__ == "__main__":
    main()
