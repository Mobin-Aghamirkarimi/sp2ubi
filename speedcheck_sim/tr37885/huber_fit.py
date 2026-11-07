"""Huber pass fit (constants evaluation_v4.lever_d), one fixed variant.

Iteratively reweighted least squares with Huber psi (c = 1.345) on the per-beacon FDOA residuals, standardised by
1.4826 * MAD of the pass's residuals (re-estimated each iteration), started at the plain LS solution, at most 20
iterations (stop when the largest relative parameter change is < 1e-6), no other per-beacon quality weight. The final
Huber weights r_i enter both statistics b_hat of the speed check and the free fit of the consistency test:
    profile  b = sum r_i v_fit_i v_sen_i / sum r_i v_fit_i^2 - 1
    matched  b = sum r_i g_i^2 v_sen_i / (lambda sum r_i g_i (f_FDOA,i - delta)) - 1
The weights use only the FDOA data, never the VSS speed, so b stays linear in v_sen.
The quality gate and the monitor keep the residual RMS of the plain LS fit (returned by the LS statistics).
Pure functions; the in-house site-accuracy code is not modified."""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "inhouse")]
from fdoa.estimator import _g, fit_pass, fit_pass_free, fit_pass_offset, fitted_speed, model  # noqa: E402
from fdoa.geometry import LAMBDA  # noqa: E402

C_HUBER, MAX_ITER, TOL = 1.345, 20, 1e-6
_LO_OFF, _HI_OFF = [-np.inf, 0.5, -10.0, 0.5, -5000.0], [np.inf, 80.0, 10.0, 40.0, 5000.0]


def huber_weights(r, c=C_HUBER):
    """Huber IRLS weights for residuals r standardised by 1.4826 * MAD."""
    r = np.asarray(r, float)
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(r), s
    u = np.abs(r) / s
    return np.where(u <= c, 1.0, c / np.maximum(u, 1e-300)), s


def _irls(resid_fn, q0, lo, hi):
    """Generic IRLS: resid_fn(q) returns the unweighted residuals; returns (q, residuals, weights, iterations, ok)."""
    q = np.array(q0, float)
    r = resid_fn(q)
    ok, it = True, 0
    for it in range(1, MAX_ITER + 1):
        w, _ = huber_weights(r)
        sw = np.sqrt(w)
        p0 = np.clip(q, np.where(np.isfinite(lo), np.array(lo) + 1e-9, -np.inf), np.where(np.isfinite(hi), np.array(hi) - 1e-9, np.inf))
        res = least_squares(lambda x: sw * resid_fn(x), p0, bounds=(lo, hi), x_scale="jac")
        ok = ok and bool(res.success)
        rel = float(np.max(np.abs(res.x - q) / np.maximum(np.abs(q), 1.0)))
        q = res.x
        r = resid_fn(q)
        if rel < TOL:
            break
    w, _ = huber_weights(r)
    return q, r, w, it, ok


def fit_offset_huber(t_b, fd, site):
    """Huber tied fit with the head-to-head offset delta (reweighted). Returns
    (params incl. delta, t_mid, residuals, weights, iterations, success)."""
    q, t_mid, _, ok = fit_pass_offset(t_b, fd, site)
    tau = t_b - t_mid
    fn = lambda x: model(x[:4], tau, site.D, False) + x[4] - fd
    q, r, w, it, ok2 = _irls(fn, q, _LO_OFF, _HI_OFF)
    return q, t_mid, r, w, it, bool(ok and ok2)


def statistics_huber(t_b, fd, vx, site):
    """(profile_d, matched_d, v_fit_d, iterations, ok, weight_share_below_1) of the Huber fit, for the same beacons as
    the LS statistics (offset fit)."""
    q, tm, _, w, it, ok = fit_offset_huber(t_b, fd, site)
    p, delta = q[:4], q[4]
    tau = t_b - tm
    vf = fitted_speed(p, t_b, tm)
    g = _g(p[0] + p[1] * tau + 0.5 * p[2] * tau ** 2, site.D, p[3])
    prof = float(np.sum(w * vf * vx) / np.sum(w * vf * vf) - 1)
    match = float(np.sum(w * g * g * vx) / (LAMBDA * np.sum(w * g * (fd - delta))) - 1)
    return prof, match, float(p[1]), it, ok, float(np.mean(w < 1.0))


def d_huber(t_b, fd, site):
    """Consistency statistic d = (a - v_mid) / v_mid of the Huber free fit (started at fit_pass_free)."""
    pf, t_mid, _, ok = fit_pass_free(t_b, fd, site)
    pt = fit_pass(t_b, fd, site)[0]                     # the bounds of fit_pass_free: a in [0.5, 1.5] x tied v_mid
    tau = t_b - t_mid
    lo = [-np.inf, 0.5, -10.0, 0.5, 0.5 * pt[1]]
    hi = [np.inf, 80.0, 10.0, 40.0, 1.5 * pt[1]]
    fn = lambda x: model(x, tau, site.D, True) - fd
    q, _, _, _, ok2 = _irls(fn, pf, lo, hi)
    return float((q[-1] - q[1]) / q[1]) if (ok and ok2) else float("nan")
