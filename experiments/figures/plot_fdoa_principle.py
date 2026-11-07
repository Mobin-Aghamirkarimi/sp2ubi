"""Figure: measurement principle of the passive multistatic Doppler (FDOA) speed check.

(a) site geometry (top view): two receive heads 1, 2 on the roadside line, baseline L = 100 m (not to scale), the
    vehicle's beacon positions along one crossing; (b) per-beacon FDOA measured at the two heads (simulated channel,
    K = 10 dB, 10 Hz beacons) with the tied crossing fit of the FDOA model including the head offset f_o
    (fit_pass_offset); (c) the roadside speed profile from the fit against the per-beacon VSS speeds of an honest
    vehicle and of a VSS scaled by -10 % (the speed check compares them homomorphically; b = profile statistic).
Simulation-conditional illustration of one pass (fixed seed); the numbers of the evaluation come from the simulations
in speedcheck_sim/, not from this pass.

Run (from the repository root, environment requirements/speed.lock):  python experiments/figures/plot_fdoa_principle.py
Output: experiments/figures/results/fig_fdoa_principle.pdf
"""
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
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "speedcheck_sim"))
from fdoa.channel import ChannelConfig, simulate_pass  # noqa: E402
from fdoa.estimator import fdoa_from_samples, fit_pass_offset, fitted_speed, model  # noqa: E402
from fdoa.geometry import Site, make_trajectory  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
OUT_PDF = os.path.join(HERE, "results", "fig_fdoa_principle.pdf")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "legend.fontsize": 7, "legend.frameon": False,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def fmt(x):
    """Signed one-decimal number with a typographic minus."""
    return f"{x:+.1f}".replace("-", "\u2212")


def style(ax):
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    rng = np.random.default_rng(20261005)
    site = Site(D=100.0)
    cfg = ChannelConfig(K_db=10.0)
    traj = make_trajectory(rng, site, 20.0, x_start=-0.3 * site.D, x_end=1.3 * site.D)
    t_b, y, _ = simulate_pass(rng, site, traj, cfg)
    fd = fdoa_from_samples(y, cfg.packet_s)
    p, t_mid, _, ok = fit_pass_offset(t_b, fd, site)       # p = (x_mid, v_mid, acc, rho, delta)
    x_b, y_b, vx_b, _ = traj.at(t_b)
    v_fit = fitted_speed(p, t_b, t_mid)
    vsen_honest = vx_b
    vsen_scaled = 0.90 * vx_b
    b_h = np.sum(v_fit * vsen_honest) / np.sum(v_fit * v_fit) - 1
    b_s = np.sum(v_fit * vsen_scaled) / np.sum(v_fit * v_fit) - 1

    fig = plt.figure(figsize=(7.16, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.15], wspace=0.42, left=0.03, right=0.985, top=0.86,
                          bottom=0.2)

    # (a) geometry
    ax = fig.add_subplot(gs[0])
    ax.add_patch(plt.Rectangle((-35, -5.25), 170, 10.5, color="#efeeea", zorder=0, lw=0))
    ax.plot([-35, 135], [0, 0], color="white", lw=1.2, ls=(0, (6, 5)), zorder=1)
    ax.plot([-35, 135], [site.pole_offset] * 2, color=MUTED, lw=0.6, zorder=1)
    sel = slice(None, None, 4)
    ax.scatter(x_b[sel], y_b[sel], s=7, color=BLUE, zorder=3, lw=0)
    ax.annotate("", xy=(x_b[sel][-1] + 8, y_b[0]), xytext=(x_b[sel][-1], y_b[0]),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8))
    for hx, name in ((0.0, "1"), (site.D, "2")):
        ax.scatter([hx], [site.pole_offset], marker="^", s=46, color=INK, zorder=4)
        ax.text(hx, site.pole_offset + 2.4, f"head {name}", ha="center", va="bottom", fontsize=7, color=INK)
    k = int(np.argmin(np.abs(x_b - 30)))
    for hx in (0.0, site.D):
        ax.plot([x_b[k], hx], [y_b[k], site.pole_offset], color=ORANGE, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
    ax.annotate("", xy=(0, -9.5), xytext=(site.D, -9.5), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text(site.D / 2, -11.2, r"$L = 100$ m (not to scale)", ha="center", va="top", fontsize=7, color=INK2)
    ax.text(x_b[k] + 2, y_b[k] - 2.2, "beacon", fontsize=6.5, color=ORANGE, va="top")
    ax.text(118, -3.6, "vehicle", fontsize=6.5, color=BLUE, ha="center")
    ax.set_xlim(-38, 138); ax.set_ylim(-16, 16)
    ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) site (top view)", loc="left", color=INK)

    # (b) FDOA per beacon and the tied fit
    ax = fig.add_subplot(gs[1]); style(ax)
    tau = t_b - t_mid
    xx = np.linspace(x_b[0], x_b[-1], 400)
    tt = np.interp(xx, x_b, tau)
    ax.scatter(x_b, fd, s=6, color=BLUE, lw=0, zorder=3, label="measured FDOA")
    ax.plot(xx, model(p[:4], tt, site.D, False) + p[4], color=ORANGE, lw=1.4, zorder=4, label=r"crossing fit with $f_{\mathrm{o}}$")
    ax.axvline(0, color=MUTED, lw=0.5, ls=":"); ax.axvline(site.D, color=MUTED, lw=0.5, ls=":")
    ax.set_xlabel(r"vehicle position $\xi$ (m)"); ax.set_ylabel("FDOA (Hz)")
    ax.legend(loc="upper center", handletextpad=0.3, borderaxespad=0.1, markerscale=1.6, ncol=1)
    ax.set_title("(b) frequency difference of arrival", loc="left", color=INK)

    # (c) roadside speed profile vs VSS speeds
    ax = fig.add_subplot(gs[2]); style(ax)
    ax.plot(x_b, v_fit, color=ORANGE, lw=1.4, zorder=4, label="roadside speed (fit)")
    ax.scatter(x_b, vsen_honest, s=6, color=BLUE, lw=0, zorder=3, label=f"VSS, honest ($\\hat b$ = {fmt(100*b_h)} %)")
    ax.scatter(x_b, vsen_scaled, s=6, color=AQUA, lw=0, zorder=3, marker="s",
               label=f"VSS scaled by −10 % ($\\hat b$ = {fmt(100*b_s)} %)")
    ax.set_xlabel(r"vehicle position $\xi$ (m)"); ax.set_ylabel("speed (m/s)")
    lo = min(vsen_scaled.min(), v_fit.min()); hi = max(vsen_honest.max(), v_fit.max())
    ax.set_ylim(lo - 3.0, hi + 0.6)
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.1, markerscale=1.6)
    ax.set_title("(c) roadside speed vs. VSS", loc="left", color=INK)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    print("fit ok", ok, "b honest", round(b_h, 4), "b scaled", round(b_s, 4), "delta Hz", round(p[4], 3), "->",
          os.path.relpath(OUT_PDF, ROOT))


if __name__ == "__main__":
    main()
