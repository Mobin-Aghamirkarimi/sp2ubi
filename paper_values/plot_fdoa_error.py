"""Figure: per-beacon FDOA error against SNR (study 1), drawn at the printed one-column size.

The data are those of the study-1 report generator (speedcheck_sim, make_reports.fig_c) on the stored stage-3 outputs
criteria_primary.json and inhouse_k4_sens_inhouse.json: the script runs fig_c, takes its plotted data, and draws the
same data with presentation changes only (legend below the axes, channel-model labels, x-axis from -7 dB to the last
populated SNR bin, 85.29 mm wide). It asserts that every plotted array (curves, q10-q90 bands, SNR histograms, decoding
lines) equals the report figure's exactly and that the numbers fig_c computes equal the stored numbers_primary.json.

Output: paper_values/figures/fdoa_error_vs_snr.pdf
Usage (from the repository root, environment requirements/speedcheck_sim.lock):
  python paper_values/plot_fdoa_error.py [--check]
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
import matplotlib.transforms  # noqa: E402

OUT = os.path.join(HERE, "figures", "fdoa_error_vs_snr.pdf")
FIG_W_IN, FIG_H_IN = 3.36, 4.1       # one column of the journal template (85.29 mm), room for the legend
FS_LABEL, FS_TICK, FS_LEGEND = 8, 7, 7


def report_figure(MR, cr, k4):
    """Run fig_c into a temporary directory and keep its figure (plt.close is suspended for the call)."""
    figs, close = [], plt.close
    real_subplots = plt.subplots

    def keep(*a, **k):
        f, ax = real_subplots(*a, **k)
        figs.append(f)
        return f, ax
    plt.subplots, plt.close = keep, (lambda *a, **k: None)
    try:
        nums = {}
        with tempfile.TemporaryDirectory() as td:
            MR.fig_c(cr, k4, os.path.join(td, "fig_C"), nums)
    finally:
        plt.subplots, plt.close = real_subplots, close
    assert len(figs) == 1
    return figs[0], nums


def data_of(fig):
    """Plotted data: lines (x, y), bands (polygon vertices), histograms (values, edges)."""
    ax, ax2 = fig.axes[0], fig.axes[1]
    lines = [(np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)) for l in ax.get_lines()]
    bands = [np.asarray(p.get_paths()[0].vertices, float) for p in ax.collections]
    hists = [(np.asarray(p.get_data().values, float), np.asarray(p.get_data().edges, float)) for p in ax2.patches]
    return lines, bands, hists


def relabel(lab):
    if lab.startswith("in-house K = "):
        return r"our channel model, $K_R$ = " + lab[len("in-house K = "):]
    return {"single-tone CRLB (2 heads)": "single-tone Cramér–Rao bound (2 heads)"}.get(lab, lab)


def redraw(ref):
    """The same artists as the report figure, in the same order and with the same data."""
    rax, rax2 = ref.axes[0], ref.axes[1]
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax2 = ax.twinx()
    for l in rax.get_lines():
        x, y = l.get_xdata(), l.get_ydata()
        lab = l.get_label()
        ax.plot(x, y, ls=l.get_linestyle(), marker=l.get_marker(), ms=l.get_markersize(), lw=l.get_linewidth(),
                color=l.get_color(), label=None if lab.startswith("_") else relabel(lab))
    ax.set_yscale("log")
    for p in rax.collections:
        ax.add_collection(matplotlib.collections.PolyCollection([p.get_paths()[0].vertices], closed=False,
                                                                alpha=p.get_alpha(), facecolors=p.get_facecolor()))
    for p in rax2.patches:
        d = p.get_data()
        ax2.stairs(d.values, d.edges, alpha=p.get_alpha(), color=p.get_edgecolor())
    edges = [(d.values, d.edges) for d in (p.get_data() for p in rax2.patches)]
    hi = max([e[np.nonzero(v)[0][-1] + 1] for v, e in edges if np.any(v)] + [50.0])
    ax.set_xlim(-7.0, float(np.ceil(hi)))
    ax2.set_ylim(rax2.get_ylim())
    ax2.set_ylabel(rax2.get_ylabel(), fontsize=FS_LABEL)
    ax.set_xlabel(rax.get_xlabel(), fontsize=FS_LABEL)
    ax.set_ylabel(rax.get_ylabel(), fontsize=FS_LABEL)
    for a in (ax, ax2):
        a.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3)
    below_xlabel = matplotlib.transforms.offset_copy(ax.transAxes, fig=fig, y=-(FS_TICK + FS_LABEL + 14), units="points")
    ax.legend(fontsize=FS_LEGEND, ncol=1, loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=below_xlabel,
              frameon=False, handlelength=2.2, labelspacing=0.25, borderaxespad=0.0)
    fig.tight_layout(pad=0.2)
    return fig


def same(a, b):
    la, ba, ha = a
    lb, bb, hb = b
    assert len(la) == len(lb) and len(ba) == len(bb) and len(ha) == len(hb), "artist counts differ"
    for (xa, ya), (xb, yb) in zip(la, lb):
        assert np.array_equal(xa, xb, equal_nan=True) and np.array_equal(ya, yb, equal_nan=True), "curve data differ"
    for pa, pb in zip(ba, bb):
        assert np.array_equal(pa, pb), "band data differ"
    for (va, ea), (vb, eb) in zip(ha, hb):
        assert np.array_equal(va, vb) and np.array_equal(ea, eb), "histogram data differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assertions only, no figure")
    a = ap.parse_args()
    MR = SP.import_make_reports()
    cr = json.load(open(SP.find("criteria_primary.json")))
    k4 = json.load(open(SP.find("inhouse_k4_sens_inhouse.json")))
    ref, nums = report_figure(MR, cr, k4)
    stored = json.load(open(SP.find("numbers_primary.json")))["numbers"]
    for k, v in nums.items():
        assert k in stored and stored[k] == v, f"{k}: {v} != numbers_primary.json {stored.get(k)}"
    new = redraw(ref)
    same(data_of(ref), data_of(new))
    n = len(data_of(new)[0])
    if a.check:
        print(f"plot_fdoa_error: OK ({n} curves, {len(nums)} numbers equal to the stored report)")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new.savefig(OUT, metadata={"Creator": None, "Producer": None, "CreationDate": None}, bbox_inches="tight",
                pad_inches=0.02)
    print(f"written {os.path.relpath(OUT, SP.ROOT)} ({n} curves; data equal to the report figure)")


if __name__ == "__main__":
    main()
