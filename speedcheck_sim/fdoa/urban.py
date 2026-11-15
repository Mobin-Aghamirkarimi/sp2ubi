"""Urban estimator of the roadside speed statistic (urban street sites; the base estimator is fdoa/estimator.py).

It differs from the base estimator in four ways:
  1. the FDOA of a beacon is the lag-one pulse-pair estimate of the product of the two heads' samples (pulse_pair), in
     which the transmitter phase cancels sample by sample, instead of the difference of two periodogram peaks;
  2. the head offset f_o and the perpendicular distance rho are fixed to the medians of the site's survey fits;
  3. the speed profile is piecewise linear, with a node every NODE_S seconds (at most MAX_NODES) and a penalty of
     PENALTY_HZ per m/s^2 on each change of the acceleration between adjacent segments;
  4. the statistics use only the beacons whose fitted position lies in CORE (shares of the head spacing D); a crossing
     with fewer than MIN_CORE_BEACONS such beacons gives no entry.
Every input is observable at the RSU (per-beacon FDOA, received power, air times, the site survey); the reported speed
enters only the numerator of the statistics, linearly.

Pass model:
    constant acceleration   x(tau) = x_mid + v_mid tau + acc tau^2 / 2,   v(tau) = v_mid + acc tau
    speed nodes             v(tau) piecewise linear through the nodes, x(tau) its integral
    FDOA(tau) = v(tau) / lambda * g(x(tau); rho) + f_o
Statistics (the estimate of the scale error b of a crossing):
    profile  sum v_hat v_sen / sum v_hat^2 - 1
    matched  sum q g^2 v_sen / (lambda sum q g (f - f_o)) - 1,   q = 1 or the per-beacon weight

envelope_weights and the free parameters of fit (w, and f_o or rho left free with speed nodes) serve only the
comparison variants of pipeline/s3_estimate.py; the urban estimator of the paper uses no weights.
"""
import numpy as np
from scipy.optimize import least_squares

from .estimator import _g
from .geometry import LAMBDA

NODE_S = 2.0              # spacing of the speed nodes (s)
MAX_NODES = 12            # cap of the speed nodes of one crossing (long, stopped crossings get wider segments)
PENALTY_HZ = 30.0         # penalty on a change of the acceleration between adjacent segments (Hz per m/s^2)
CORE = (0.20, 0.80)       # fitted positions that enter the statistics (shares of D)
MIN_CORE_BEACONS = 12


def pulse_pair(z, lag, packet_s=1e-3):
    """Lag-`lag` autocorrelation estimate per row of z: frequency (Hz) and coherence |R(lag)| / R(0)."""
    M = z.shape[-1]
    r = np.sum(z[:, lag:] * np.conj(z[:, :-lag]), axis=1)
    r0 = np.sqrt(np.sum(np.abs(z[:, lag:]) ** 2, axis=1) * np.sum(np.abs(z[:, :-lag]) ** 2, axis=1))
    return np.angle(r) / (2 * np.pi * lag * packet_s / M), np.abs(r) / np.maximum(r0, 1e-300)


def running_median(a, k=21):
    p = np.pad(np.asarray(a, float), (k // 2, k // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=1)


def envelope_weights(pa_db, pb_db):
    """Per-beacon weight 1 / (1 + 1/P_A + 1/P_B), P the head's received power relative to its running median (linear),
    normalised to mean 1. The conditional variance of the random FM of a fading signal given its envelope R falls as
    1/R^2 (Rice 1948), so this is an inverse-variance weight with a floor. Only the difference of the two heads' fades
    enters, so that a change of the transmit power, which is common to both heads, cannot move a weight."""
    da = np.asarray(pa_db, float) - running_median(pa_db)
    db = np.asarray(pb_db, float) - running_median(pb_db)
    da, db = 0.5 * (da - db), 0.5 * (db - da)
    w = 1.0 / (1.0 + 10 ** (-da / 10) + 10 ** (-db / 10))
    return w / np.mean(w)


def _node_profile(p, tau, nodes):
    """Position and speed at tau for p = (x_mid, speed at each node, ...); x is the exact integral of the speed."""
    vn = p[1:1 + nodes.size]
    v = np.interp(tau, nodes, vn)
    xn = np.concatenate([[0.0], np.cumsum(0.5 * (vn[1:] + vn[:-1]) * np.diff(nodes))])
    i = np.clip(np.searchsorted(nodes, tau, side="right") - 1, 0, nodes.size - 2)
    dt = tau - nodes[i]
    slope = (vn[i + 1] - vn[i]) / (nodes[i + 1] - nodes[i])
    xi = xn[i] + vn[i] * dt + 0.5 * slope * dt ** 2
    i0 = int(np.clip(np.searchsorted(nodes, 0.0, side="right") - 1, 0, nodes.size - 2))
    d0 = 0.0 - nodes[i0]
    x_at0 = xn[i0] + vn[i0] * d0 + 0.5 * (vn[i0 + 1] - vn[i0]) / (nodes[i0 + 1] - nodes[i0]) * d0 ** 2
    return p[0] + xi - x_at0, v


def fit(t_b, fd, site, *, w=None, fo=None, rho=None, node_s=None, penalty=PENALTY_HZ):
    """Least-squares fit of one crossing in three stages: (A) the constant-acceleration model without a free offset,
    started at the largest |FDOA|; (B) if fo is None, the offset f_o freed, two starts (0 and the mean residual of A),
    the lower cost wins; (C) if node_s is given, the speed-node profile started at the solution of B.
    fo, rho: fixed values, or None to fit the parameter; w: per-beacon weights or None.
    With w, fo, rho and node_s all None this is the fit of the base estimator (fdoa.estimator.fit_pass_offset).
    Returns dict(x, v, g, fo, rho, resid (unweighted, Hz), ok, p)."""
    t_mid = 0.5 * (t_b[0] + t_b[-1])
    tau = t_b - t_mid
    sw = np.ones_like(fd, dtype=float) if w is None else np.sqrt(np.asarray(w, float))
    free_rho, free_fo = rho is None, fo is None
    rho0 = float(np.hypot(site.pole_offset, site.dh)) if free_rho else float(rho)
    fo0 = 0.0 if free_fo else float(fo)
    k = int(np.argmax(np.abs(fd - fo0)))
    g_peak = abs(_g(np.array([site.D / 2]), site.D, rho0)[0])
    v0 = max(abs(fd[k] - fo0) * LAMBDA / g_peak, 1.0)
    x0 = site.D / 2 - v0 * tau[k]

    def solve(q0, lo, hi, fun, max_nfev=None):
        lo_a, hi_a = np.array(lo, float), np.array(hi, float)
        q0 = np.clip(np.array(q0, float), np.where(np.isfinite(lo_a), lo_a + 1e-6, -1e9), np.where(np.isfinite(hi_a), hi_a - 1e-6, 1e9))
        return least_squares(fun, q0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev)

    def quad(q, r, o):
        x = q[0] + q[1] * tau + 0.5 * q[2] * tau ** 2
        return (q[1] + q[2] * tau) / LAMBDA * _g(x, site.D, r) + o - fd

    # (A) constant acceleration, offset fixed
    lo, hi = [-np.inf, 0.5, -10.0], [np.inf, 80.0, 10.0]
    if free_rho:
        A = solve([x0, v0, 0.0, rho0], lo + [0.5], hi + [40.0], lambda q: sw * quad(q, q[3], fo0))
        pA, rA = A.x[:3], A.x[3]
    else:
        A = solve([x0, v0, 0.0], lo, hi, lambda q: sw * quad(q, rho0, fo0))
        pA, rA = A.x[:3], rho0
    ok = bool(A.success)
    p3, r_hat, o_hat = pA, float(rA), fo0
    # (B) offset freed
    if free_fo:
        best = None
        for d0 in (0.0, float(np.mean(quad(pA, rA, fo0)))):
            if free_rho:
                B = solve(list(pA) + [rA, d0], lo + [0.5, -5000.0], hi + [40.0, 5000.0], lambda q: sw * quad(q, q[3], q[4]))
            else:
                B = solve(list(pA) + [d0], lo + [-5000.0], hi + [5000.0], lambda q: sw * quad(q, rho0, q[3]))
            if best is None or B.cost < best.cost:
                best = B
        p3 = best.x[:3]
        r_hat = float(best.x[3]) if free_rho else rho0
        o_hat = float(best.x[-1])
        ok = ok and bool(best.success)
    x, v = p3[0] + p3[1] * tau + 0.5 * p3[2] * tau ** 2, p3[1] + p3[2] * tau
    p = p3
    # (C) speed nodes
    inner = 0 if node_s is None else min(max(1, int(np.ceil((tau[-1] - tau[0]) / node_s)) - 1), MAX_NODES - 2)
    if inner > 0:
        nodes = np.linspace(tau[0], tau[-1], inner + 2)
        h = nodes[1] - nodes[0]
        n_v = nodes.size

        def prof(q):
            r = q[1 + n_v] if free_rho else rho0
            o = q[-1] if free_fo else fo0
            xx, vv = _node_profile(q, tau, nodes)
            return xx, vv, r, o

        def cost(q):
            xx, vv, r, o = prof(q)
            e = sw * (vv / LAMBDA * _g(xx, site.D, r) + o - fd)
            return np.concatenate([e, penalty * np.diff(q[1:1 + n_v], 2) / h])

        q0 = [p3[0]] + list(np.maximum(p3[1] + p3[2] * nodes, 1e-3)) + ([r_hat] if free_rho else []) + ([o_hat] if free_fo else [])
        Cc = solve(q0, [-np.inf] + [0.0] * n_v + ([0.5] if free_rho else []) + ([-5000.0] if free_fo else []),
                   [np.inf] + [80.0] * n_v + ([40.0] if free_rho else []) + ([5000.0] if free_fo else []), cost, max_nfev=40)
        x, v, r_hat, o_hat = prof(Cc.x)
        r_hat, o_hat, p = float(r_hat), float(o_hat), Cc.x      # reaching the evaluation cap of stage C is not a failure
    g = _g(x, site.D, r_hat)
    return dict(x=x, v=v, g=g, fo=o_hat, rho=r_hat, resid=v / LAMBDA * g + o_hat - fd, ok=ok, p=p)


def statistics(F, fd, v_sen, q=None, core=None, D=100.0):
    """(profile, matched, mean squared residual) of a fit F for the reported speeds v_sen. q: per-beacon weights of the
    matched statistic. core = (a, b): both sums run only over the beacons whose fitted position lies in [a D, b D];
    with fewer than MIN_CORE_BEACONS of them both statistics are NaN."""
    q = np.ones_like(fd, dtype=float) if q is None else np.asarray(q, float)
    mse = float(np.mean(F["resid"] ** 2))
    if core is None:
        m = np.ones(fd.shape, dtype=bool)
    else:
        m = (F["x"] >= core[0] * D) & (F["x"] <= core[1] * D)
        if m.sum() < MIN_CORE_BEACONS:
            return float("nan"), float("nan"), mse
    q = q * m
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = float(np.sum(m * F["v"] * v_sen) / np.sum(m * F["v"] ** 2) - 1)
        match = float(np.sum(q * F["g"] ** 2 * v_sen) / (LAMBDA * np.sum(q * F["g"] * (fd - F["fo"]))) - 1)
    return prof, match, mse
