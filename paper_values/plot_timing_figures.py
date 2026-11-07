"""The three timing figures of the evaluation, from the stored timing results (experiments/timings/results/):
operation times of the risk path and the speed path, the risk path against the baselines, and the TSP throughput.

Every plotted value and printed label is the value of paper_values/derived_values.py for the same key.
Output: paper_values/figures/timing_operations.pdf, timing_baselines.pdf, timing_throughput.pdf

Usage (from the repository root, environment requirements/speed.lock):  python paper_values/plot_timing_figures.py
"""
import json
import os
import sys

import matplotlib
from matplotlib import font_manager
matplotlib.use("Agg")
for _f in ("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derived_values as DV  # noqa: E402

OUT = os.path.join(HERE, "figures")
META = {"Creator": None, "Producer": None, "CreationDate": None}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#b9b8b2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "legend.fontsize": 7, "legend.frameon": False,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
DATA = {it: json.load(open(os.path.join(DV.TIMINGS, f"{it}.json"))) for it in DV.ITEMS}


def stat(key):
    """(median, q1, q3) of a timing key."""
    _, item, path = DV.TIMING_KEYS[key]
    n = DV.node(DATA, item, path)
    return float(n["median_ms"]), float(n["q1_ms"]), float(n["q3_ms"])


def num(key):
    _, item, path, _ = DV.TIMING_KEYS[key]
    return float(DV.node(DATA, item, path))


def label_ms(v):
    return DV.printed(v, "ms").replace("-", "−")


def style(ax, axis="x"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


OPS_ROWS = [  # (label, key, colour) top to bottom
    ("TSP: group-signature check", "risk_path_budget|tsp_group_verify", BLUE),
    ("TSP: evaluation of the risk model", "risk_path_budget|tsp_evaluate", BLUE),
    ("TSP: slot fill", "risk_path_budget|tsp_fill", BLUE),
    ("TSP: receipt", "risk_path_budget|tsp_receipt_sign", BLUE),
    ("Insurer: decryption", "risk_path_budget|insurer_decrypt", BLUE),
    ("Insurer: HomMAC check", "risk_path_budget|insurer_hommac_verify_both_blocks", BLUE),
    ("RSU: crossing acceptance", "speed_path|rsu_accept_crossing_K40", ORANGE),
    (r"RSU: evaluation of Enc($\tilde b$)", "speed_path|rsu_evaluate_K40", ORANGE),
    ("RSU: release (re-randomize, sign)", "speed_path|rsu_release_rerandomise_and_sign_K40", ORANGE),
    ("TSP: entry-signature check", "speed_path|tsp_verify_entry", ORANGE),
    ("TSP: shuffle re-randomization", "speed_path|tsp_shuffle_rerandomize_per_entry", ORANGE),
    ("TSP: partial decryptions with proofs", "speed_path|tsp_partials_with_proofs_per_entry", ORANGE),
    ("Insurer: proofs, opening, decryption", "speed_path|insurer_verify_open_and_decrypt_per_entry", ORANGE),
    (r"Insurer: decoding, $\hat b=0$", "speed_path|insurer_decrypt_b+0.00", ORANGE),
    (r"Insurer: decoding, $|\hat b|=0.10$", "speed_path|insurer_decrypt_b+0.10", ORANGE),
]
N_RISK = sum(1 for r in OPS_ROWS if r[2] == BLUE)


def fig_ops(out):
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    fig.subplots_adjust(left=0.49, right=0.97, top=0.89, bottom=0.13)
    y = list(range(len(OPS_ROWS)))[::-1]
    for yi, (lab, key, col) in zip(y, OPS_ROWS):
        m, q1, q3 = stat(key)
        ax.barh(yi, m, height=0.62, color=col, zorder=2, lw=0)
        ax.plot([q1, q3], [yi, yi], color=INK, lw=0.7, zorder=3)
        ax.text(m * 1.12, yi, label_ms(m), va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in OPS_ROWS])
    ax.set_xscale("log")
    ax.set_xlim(0.03, 400)
    ax.set_xlabel("Time (ms), median; line: interquartile range")
    ax.xaxis.set_label_coords(0.33, -0.1)
    ax.axhline(len(OPS_ROWS) - N_RISK - 0.5, color=MUTED, lw=0.5, ls=(0, (3, 3)))
    style(ax)
    ax.legend(handles=[Patch(color=BLUE, label="Risk path, per record"),
                       Patch(color=ORANGE, label="Speed path, per crossing (RSU, 40 beacons) or per entry")],
              loc="lower center", bbox_to_anchor=(0.30, 1.0), ncol=1, handlelength=1.0)
    fig.savefig(out, metadata=META)
    plt.close(fig)


CONFIGS = ["SP2UBI: BFV, insurer's key,\nencrypted weights", "B1: BFV, same key,\nclear weights",
           "B2: CKKS,\nencrypted weights", "B3: Paillier-3072,\nclear weights"]
SUBPLOTS = [("OBU encryption", ["risk_path_budget|obu_bfv_encrypt", None, "baselines|B3|obu_encrypt",
                              "baselines|B4|obu_encrypt"]),
          ("TSP evaluation", ["risk_path_budget|tsp_evaluate", "baselines|B2|tsp_evaluate", "baselines|B3|tsp_evaluate",
                              "baselines|B4|tsp_evaluate"]),
          ("Decryption", ["risk_path_budget|insurer_decrypt", None, "baselines|B3|insurer_decrypt",
                          "baselines|B4|insurer_decrypt"])]


def fig_baselines(out):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.6), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.85, bottom=0.23, wspace=0.12)
    y = list(range(len(CONFIGS)))[::-1]
    for ax, (title, keys) in zip(axes, SUBPLOTS):
        for i, (yi, key) in enumerate(zip(y, keys)):
            if key is None:
                ax.text(0.04, yi, "as SP2UBI (same key and ciphertext)", va="center", ha="left", fontsize=6.3,
                        color=MUTED, style="italic")
                continue
            v = stat(key)[0]
            ax.barh(yi, v, height=0.6, color=BLUE if i == 0 else BASE, zorder=2, lw=0)
            ax.text(v * 1.15, yi, label_ms(v), va="center", ha="left", fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.03, 8000)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Time (ms), median")
        style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(CONFIGS, fontsize=6.5)
    fig.savefig(out, metadata=META)
    plt.close(fig)


def fig_throughput(out):
    W = sorted(int(w) for w in DATA["throughput"]["result"]["throughput"])
    assert W == list(range(1, 9)), W
    med = [num(f"throughput|workers{w}") for w in W]
    lo = [num(f"throughput|workers{w}|min") for w in W]
    hi = [num(f"throughput|workers{w}|max") for w in W]
    lbl_hi, lbl_lo = DV.printed(med[-1], "rps"), DV.printed(med[0], "rps")
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.2)
    ax.plot(W, [w * med[0] for w in W], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(5.3, 5.3 * med[0] + 2, "workers × one-worker rate", color=INK2, fontsize=6.5, ha="right",
            va="bottom")
    ax.axvline(4, color=GRID, lw=1.0, zorder=1)
    ax.text(4.08, 3, "4 physical cores", color=INK2, fontsize=6.5, ha="left", va="bottom")
    ax.vlines(W, lo, hi, color=BLUE, lw=1.0, zorder=3)
    ax.plot(W, med, color=BLUE, lw=2.0, marker="o", ms=4.5, zorder=4, mec="white", mew=0.8)
    top = max(W) * med[0] * 1.02
    ax.text(W[-1], med[-1] + 0.04 * top, lbl_hi, color=INK2, fontsize=6.5, ha="center", va="bottom")
    ax.text(W[0] + 0.12, med[0] - 0.05 * top, lbl_lo, color=INK2, fontsize=6.5, ha="left", va="top")
    ax.set_xticks(W)
    ax.set_xlabel("TSP worker processes (one insurer tenant each, one thread)")
    ax.set_ylabel("Records per second")
    ax.set_ylim(0, top)
    style(ax, "y")
    fig.savefig(out, metadata=META)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("timing_operations.pdf", fig_ops), ("timing_baselines.pdf", fig_baselines),
                     ("timing_throughput.pdf", fig_throughput)):
        fn(os.path.join(OUT, name))
        print("written", os.path.relpath(os.path.join(OUT, name), DV.ROOT))


if __name__ == "__main__":
    main()
