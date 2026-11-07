"""Frequency-of-arrival per packet, FDOA, joint pass fit with nuisance parameters, chi-square gate, Fisher bound.

Pass model (both heads on the roadside line, perpendicular distance rho to the vehicle antenna, unknown):
    x(tau) = x_mid + v_mid tau + acc tau^2/2,   v(tau) = v_mid + acc tau
    FDOA(tau) = (a / v_mid) * v(tau) / lambda * [ (0 - x)/r_A - (D - x)/r_B ],   r = sqrt((x_head - x)^2 + rho^2)
Tied (honest physics): a = v_mid, parameters (x_mid, v_mid, acc, rho).
Free (consistency test): a separate Doppler-amplitude speed, parameters (x_mid, v_mid, acc, rho, a);
v_mid is then the kinematic (transit time-scale) speed and a - v_mid tests frequency-only manipulation.
A cubic term (jerk) is available but off by default: it is almost unidentifiable from one pass and, in the full
Monte Carlo, inflates the per-pass error from ~1 % to ~3.5 %.
"""
import numpy as np
from scipy.optimize import least_squares

from .geometry import LAMBDA, Site

NFFT = 1 << 13


def foa(y, packet_s):
    """Frequency (Hz) of the dominant tone in each row of y (decimated, packet-centred), by zero-padded
    periodogram peak plus parabolic interpolation on the log-magnitude."""
    M = y.shape[-1]
    fs = M / packet_s
    X = np.abs(np.fft.fft(y, NFFT, axis=-1)) ** 2
    k = np.argmax(X, axis=-1)
    idx = np.arange(X.shape[0])
    a = np.log(X[idx, (k - 1) % NFFT] + 1e-300)
    b = np.log(X[idx, k] + 1e-300)
    c = np.log(X[idx, (k + 1) % NFFT] + 1e-300)
    d = 0.5 * (a - c) / (a - 2 * b + c)
    f = (k + d) * fs / NFFT
    return np.where(f > fs / 2, f - fs, f)


def fdoa_from_samples(y, packet_s):
    """FDOA = FOA_A - FOA_B per beacon; the transmitter offset is common to both heads and cancels."""
    return foa(y[0], packet_s) - foa(y[1], packet_s)


def _g(x, D, rho):
    rA = np.sqrt(x * x + rho * rho); rB = np.sqrt((D - x) ** 2 + rho * rho)
    return (-x) / rA - (D - x) / rB


def model(p, tau, D, free, jerk=False):
    """Parameters: (x_mid, v_mid, acc, rho[jerk][a])."""
    xm, vm, acc, rho = p[:4]
    j = p[4] if jerk else 0.0
    a = p[-1] if free else vm
    x = xm + vm * tau + 0.5 * acc * tau ** 2 + j * tau ** 3 / 6
    v = vm + acc * tau + 0.5 * j * tau ** 2
    return (a / vm) * v / LAMBDA * _g(x, D, rho)


def fit_pass_free(t_b, fdoa, site: Site):
    """Consistency fit: tied fit first, then the free model (separate Doppler-amplitude speed a) started at the tied
    solution with a bounded to [0.5, 1.5] x v_mid (the unconstrained free fit can diverge at low speed)."""
    p, t_mid, _, ok = fit_pass(t_b, fdoa, site)
    tau = t_b - t_mid
    lo = [-np.inf, 0.5, -10.0, 0.5, 0.5 * p[1]]; hi = [np.inf, 80.0, 10.0, 40.0, 1.5 * p[1]]
    p0 = np.clip(np.concatenate([p, [p[1]]]), np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    res = least_squares(lambda q: model(q, tau, site.D, True) - fdoa, p0, bounds=(lo, hi), x_scale="jac")
    return res.x, t_mid, res.fun, res.success and ok


def fit_pass_offset(t_b, fdoa, site: Site):
    """Tied fit with a constant FDOA offset delta (head-to-head reference offset) as a nuisance parameter
    (F1). Two starts (delta = 0 and delta = mean residual of the no-offset fit); the lower cost wins.
    Returns (params incl. delta as last element, t_mid, residuals, success)."""
    p, t_mid, r, ok = fit_pass(t_b, fdoa, site)
    tau = t_b - t_mid
    lo = [-np.inf, 0.5, -10.0, 0.5, -5000.0]; hi = [np.inf, 80.0, 10.0, 40.0, 5000.0]
    best = None
    for d0 in (0.0, float(np.mean(r))):
        p0 = np.clip(np.concatenate([p, [d0]]), np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        res = least_squares(lambda q: model(q[:4], tau, site.D, False) + q[4] - fdoa, p0, bounds=(lo, hi), x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    return best.x, t_mid, best.fun, best.success and ok


def fit_pass_free_offset(t_b, fdoa, site: Site):
    """Consistency fit with the head-to-head offset delta as a nuisance parameter: offset fit first, then
    the free model (Doppler-amplitude speed a separate from the kinematic speed) plus delta, started at the offset
    solution with a bounded to [0.5, 1.5] x v_mid. Returns (x_mid, v, acc, rho, a, delta), t_mid, residuals, success."""
    q, t_mid, _, ok = fit_pass_offset(t_b, fdoa, site)
    tau = t_b - t_mid
    lo = [-np.inf, 0.5, -10.0, 0.5, 0.5 * q[1], -5000.0]; hi = [np.inf, 80.0, 10.0, 40.0, 1.5 * q[1], 5000.0]
    p0 = np.clip(np.concatenate([q[:4], [q[1]], [q[4]]]), np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    res = least_squares(lambda r: model(r[:5], tau, site.D, True) + r[5] - fdoa, p0, bounds=(lo, hi), x_scale="jac")
    return res.x, t_mid, res.fun, res.success and ok


def fit_pass(t_b, fdoa, site: Site, *, free=False, jerk=False, rho0=None):
    """Least-squares fit of a pass. Returns (params, t_mid, residuals, success)."""
    t_mid = 0.5 * (t_b[0] + t_b[-1])
    tau = t_b - t_mid
    rho0 = np.hypot(site.pole_offset, site.dh) if rho0 is None else rho0
    k = int(np.argmax(np.abs(fdoa)))
    g_peak = np.abs(_g(np.array([site.D / 2]), site.D, rho0))[0]
    v0 = max(np.abs(fdoa[k]) * LAMBDA / g_peak, 1.0)
    x0 = site.D / 2 - v0 * tau[k]
    p0 = [x0, v0, 0.0, rho0] + ([0.0] if jerk else []) + ([v0] if free else [])
    lo = [-np.inf, 0.5, -10.0, 0.5] + ([-5.0] if jerk else []) + ([0.1] if free else [])
    hi = [np.inf, 80.0, 10.0, 40.0] + ([5.0] if jerk else []) + ([100.0] if free else [])
    lo_a, hi_a = np.array(lo, float), np.array(hi, float)
    p0 = np.clip(np.array(p0, float), np.where(np.isfinite(lo_a), lo_a + 1e-6, -1e9), np.where(np.isfinite(hi_a), hi_a - 1e-6, 1e9))
    res = least_squares(lambda p: model(p, tau, site.D, free, jerk) - fdoa, p0, bounds=(lo, hi), x_scale="jac")
    return res.x, t_mid, res.fun, res.success


def fitted_speed(p, t, t_mid, jerk=False):
    tau = t - t_mid
    return p[1] + p[2] * tau + (0.5 * p[4] * tau ** 2 if jerk else 0.0)


def fisher_sigma_vmid(t_b, site: Site, p, sigma_fdoa, free=False, jerk=False):
    """Fisher-bound standard deviation of v_mid (and of a - v_mid when free) for white per-packet FDOA noise."""
    t_mid = 0.5 * (t_b[0] + t_b[-1]); tau = t_b - t_mid
    n = len(p); e = 1e-6 * np.maximum(np.abs(p), 1.0)
    f = lambda q: model(q, tau, site.D, free, jerk)
    J = np.column_stack([(f(p + e[i] * np.eye(n)[i]) - f(p - e[i] * np.eye(n)[i])) / (2 * e[i]) for i in range(n)])
    Cov = np.linalg.inv(J.T @ J) * sigma_fdoa ** 2
    if not free:
        return np.sqrt(Cov[1, 1])
    w = np.zeros(n); w[1] = -1; w[-1] = 1
    return np.sqrt(Cov[1, 1]), np.sqrt(w @ Cov @ w)


def model_two_segment(q, tau, D, rho):
    """Study 2, candidate E-e: tied model with rho fixed and the speed piecewise linear in time with
    one knot; parameters (x_k, v_k, acc_1, acc_2, tau_k, delta): position x_k and speed v_k at the knot tau_k, acceleration
    acc_1 before and acc_2 after it, and the head-to-head offset delta (as fit_pass_offset). No jerk."""
    xk, vk, a1, a2, tk, delta = q
    s = tau - tk
    acc = np.where(s < 0, a1, a2)
    x = xk + vk * s + 0.5 * acc * s * s
    v = vk + acc * s
    return v / LAMBDA * _g(x, D, rho) + delta


def two_segment_track(q, tau):
    """Fitted position and speed of model_two_segment at tau."""
    xk, vk, a1, a2, tk, _ = q
    s = tau - tk
    acc = np.where(s < 0, a1, a2)
    return xk + vk * s + 0.5 * acc * s * s, vk + acc * s


def fit_pass_two_segment(t_b, fdoa, site: Site, rho, min_seg_s=0.5):
    """Study 2, candidate E-e: least-squares fit of model_two_segment with rho fixed (per lane, from the survey). Started
    from the tied offset fit (fit_pass_offset) with the knot at -T/4, 0 and +T/4 of the pass span T (lowest cost wins);
    the knot is bounded to the pass span minus min_seg_s at each end. Returns (params, t_mid, residuals, success)."""
    p, t_mid, _, ok = fit_pass_offset(t_b, fdoa, site)
    tau = t_b - t_mid
    T = float(tau[-1] - tau[0])
    lo = [-np.inf, 0.0, -10.0, -10.0, tau[0] + min_seg_s, -5000.0]
    hi = [np.inf, 80.0, 10.0, 10.0, tau[-1] - min_seg_s, 5000.0]
    if not lo[4] < hi[4]:
        return np.full(6, np.nan), t_mid, np.full(t_b.size, np.nan), False
    best = None
    for tk in (-T / 4, 0.0, T / 4):
        tk = float(np.clip(tk, lo[4] + 1e-6, hi[4] - 1e-6))
        q0 = np.array([p[0] + p[1] * tk + 0.5 * p[2] * tk * tk, p[1] + p[2] * tk, p[2], p[2], tk, p[4]])
        q0 = np.clip(q0, np.where(np.isfinite(lo), np.array(lo) + 1e-6, -1e9), np.where(np.isfinite(hi), np.array(hi) - 1e-6, 1e9))
        res = least_squares(lambda q: model_two_segment(q, tau, site.D, rho) - fdoa, q0, bounds=(lo, hi), x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    return best.x, t_mid, best.fun, bool(best.success and ok)
