"""Figure: separation of a frequency-only manipulation in honest standard deviations (study 1), one subplot per cell.

The points are those of the study-1 report generator (speedcheck_sim, make_reports.fig_d) on the stored stage-3 output
criteria_primary.json (c4_light): the script runs fig_d, takes its plotted points and threshold line, and draws the same
points with presentation changes only (beacon bandwidth on the x-axis, channel model by marker and colour, SNR mapping
by a filled or an open marker, threshold labelled in the plot, legend below the axes, 85.29 mm wide). It asserts that
the multiset of plotted separations per (cell, channel model) and the threshold equal the report figure's exactly.

Output: paper_values/figures/separation.pdf
Usage (from the repository root, environment requirements/speedcheck_sim.lock):
  python paper_values/plot_separation.py [--check]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import speedcheck_paths as SP  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = os.path.join(HERE, "figures", "separation.pdf")
FIG_W_IN, FIG_H_IN = 3.36, 2.9       # one column of the journal template (85.29 mm), room for the legend
FS_LABEL, FS_TICK = 8, 7
ARMS = (("gen_on", "o", "C0", "TR 37.885"), ("inhouse", "s", "C1", "our channel model"))
MAPS = (("e1", "in-house link model", True), ("lb", "TR 37.885 link budget", False))


def report_figure(MR, cr):
    """Run fig_d into a temporary directory and keep its figure (plt.close is suspended for the call)."""
    figs, close = [], plt.close
    real_subplots = plt.subplots

    def keep(*a, **k):
        f, ax = real_subplots(*a, **k)
        figs.append(f)
        return f, ax
    plt.subplots, plt.close = keep, (lambda *a, **k: None)
    try:
        with tempfile.TemporaryDirectory() as td:
            MR.fig_d(cr, os.path.join(td, "fig_D"))
    finally:
        plt.subplots, plt.close = real_subplots, close
    assert len(figs) == 1
    return figs[0]


def report_data(fig):
    """Points per (tick label, arm) and the threshold of the report figure."""
    ax = fig.axes[0]
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    pts = {}
    for c in ax.collections:
        off = np.asarray(c.get_offsets(), float)
        if off.size == 0:
            continue
        xc = off[:, 0]
        i = int(round(float(np.mean(xc))))
        arm = "inhouse" if float(np.mean(xc)) > i else "gen_on"
        pts.setdefault((ticks[i], arm), []).extend(off[:, 1].tolist())
    hl = [l for l in ax.get_lines() if np.allclose(np.asarray(l.get_ydata(), float), l.get_ydata()[0])]
    assert len(hl) == 1, "expected one threshold line"
    return {k: sorted(v) for k, v in pts.items()}, float(hl[0].get_ydata()[0]), ax.get_ylabel(), ticks


def redraw(cr, ticks, thr, ylabel):
    rows = sorted((cr or {}).get("c4_light", {}).items())
    assert len(rows) == len(ticks)
    prbs = sorted({r["prb"] for _, d in rows for r in d["variants"]})
    fig, axes = plt.subplots(1, len(rows), figsize=(FIG_W_IN, FIG_H_IN), sharey=True)
    drawn = {}
    for ax, (key, d), title in zip(axes, rows, ticks):
        for arm, mk, col, _ in ARMS:
            for mp, _, filled in MAPS:
                vs = [r for r in d["variants"] if r["arm"] == arm and r["mapping"] == mp and np.isfinite(r["Z"])]
                x = [prbs.index(r["prb"]) + (0.17 if arm == "inhouse" else -0.17) + (-0.07 if filled else 0.07)
                     for r in vs]
                z = [r["Z"] for r in vs]
                ax.scatter(x, z, marker=mk, s=20, linewidths=1.0, edgecolors=col,
                           facecolors=col if filled else "none", zorder=3)
                drawn.setdefault((title, arm), []).extend(z)
        ax.axhline(thr, color="grey", ls=":", lw=1.0, zorder=1)
        ax.text(len(prbs) - 0.45, thr - 0.08, f"threshold {thr:g}", fontsize=FS_TICK, color="0.3", ha="right", va="top")
        ax.set_xticks(range(len(prbs)), [str(p) for p in prbs])
        ax.set_xlim(-0.6, len(prbs) - 0.4)
        ax.set_title(title, fontsize=FS_TICK + 0.5, pad=3)
        ax.set_xlabel("beacon bandwidth (PRBs)", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.0, float(np.ceil(max(max(v) for v in drawn.values()) + 0.5)))
    axes[0].set_ylabel(ylabel, fontsize=FS_LABEL)
    handles = [Line2D([], [], ls="none", marker=mk, color=col, markersize=5, label=lab) for _, mk, col, lab in ARMS]
    handles += [Line2D([], [], ls="none", marker="o", markersize=5, markeredgecolor="0.25",
                       markerfacecolor="0.25" if filled else "none", label=f"SNR mapping: {lab}") for _, lab, filled in MAPS]
    fig.tight_layout(pad=0.2, rect=(0, 0.17, 1, 1))
    fig.legend(handles=handles, fontsize=FS_TICK, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, handletextpad=0.3, columnspacing=0.8, labelspacing=0.3)
    return fig, {k: sorted(v) for k, v in drawn.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assertions only, no figure")
    a = ap.parse_args()
    MR = SP.import_make_reports()
    cr = json.load(open(SP.find("criteria_primary.json")))
    ref = report_figure(MR, cr)
    pts, thr, ylabel, ticks = report_data(ref)
    new, drawn = redraw(cr, ticks, thr, ylabel)
    assert set(drawn) == set(pts), (sorted(drawn), sorted(pts))
    for k in pts:
        assert np.array_equal(np.asarray(drawn[k]), np.asarray(pts[k])), f"points differ for {k}"
    n = sum(len(v) for v in pts.values())
    if a.check:
        print(f"plot_separation: OK ({n} points and the threshold equal to the report figure)")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new.savefig(OUT, metadata={"Creator": None, "Producer": None, "CreationDate": None}, bbox_inches="tight",
                pad_inches=0.02)
    print(f"written {os.path.relpath(OUT, SP.ROOT)} ({n} points; data equal to the report figure)")


if __name__ == "__main__":
    main()
