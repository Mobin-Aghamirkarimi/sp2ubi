"""Site layout, vehicle trajectories and per-head Doppler.

Coordinates: the road runs along +x. The vehicle antenna moves in the plane z = 0 at lateral position y(t)
(y = 0 is the nominal lane centre). Both receive heads stand on the roadside line y = pole_offset at height dh
above the vehicle antenna, at x = 0 (head A) and x = D (head B).
"""
from dataclasses import dataclass

import numpy as np

C = 299_792_458.0
FC = 5.9e9
LAMBDA = C / FC


@dataclass(frozen=True)
class Site:
    D: float = 100.0            # head spacing along the road (m)
    pole_offset: float = 8.0    # lateral distance from the nominal lane centre to the head line (m)
    dh: float = 5.5             # head height above the vehicle antenna (m)

    def heads(self) -> np.ndarray:
        return np.array([[0.0, self.pole_offset, self.dh], [self.D, self.pole_offset, self.dh]])


@dataclass
class Trajectory:
    """Dense vehicle trajectory sampled on a uniform grid; positions/velocities are interpolated at query times."""
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    def at(self, tq):
        f = lambda a: np.interp(tq, self.t, a)
        return f(self.x), f(self.y), f(self.vx), f(self.vy)


def make_trajectory(rng, site: Site, v0: float, *, sigma_acc=0.3, tau_acc=2.0, tau_speed=10.0, lane_halfwidth=3.5,
                    p_lane_change=0.0, mean_acc=0.0, dt=0.01, x_start=None, x_end=None) -> Trajectory:
    """Cruising pass at nominal speed v0: Ornstein-Uhlenbeck acceleration (std sigma_acc, time constant tau_acc) with
    reversion of the speed towards v0 (time constant tau_speed), speed floored at 1 m/s; random lane offset in
    [-lane_halfwidth, lane_halfwidth]; with probability p_lane_change one 3.5 m lane change over 4 s. The trajectory is
    generated until the vehicle has left the window [x_start, x_end]. mean_acc (m/s^2) moves the reversion target
    linearly in time (e.g. -0.5 for a vehicle slowing towards a junction), floored at 3 m/s."""
    x_start = -0.3 * site.D if x_start is None else x_start
    x_end = 1.3 * site.D if x_end is None else x_end
    phi = np.exp(-dt / tau_acc); s = sigma_acc * np.sqrt(1 - phi * phi)
    v_init = max(v0 + rng.normal(0, sigma_acc * np.sqrt(tau_acc * tau_speed / 2)), 1.0)
    a_k, v_k, x_k = rng.normal(0, sigma_acc), v_init, x_start
    xs, vs = [x_k], [v_k]
    while x_k < x_end:
        a_k = phi * a_k + rng.normal(0, s)
        target = v0 + mean_acc * (len(xs) - 1) * dt
        drift = mean_acc if target > 3.0 else 0.0
        v_k = max(v_k + (a_k + drift - (v_k - max(target, 3.0)) / tau_speed) * dt, 1.0)
        x_k += v_k * dt
        xs.append(x_k); vs.append(v_k)
    x, vx = np.array(xs), np.array(vs)
    t = np.arange(len(x)) * dt
    stop = len(x)
    y = np.full(stop, rng.uniform(-lane_halfwidth, lane_halfwidth))
    vy = np.zeros(stop)
    if rng.random() < p_lane_change:
        dur = 4.0; t0 = rng.uniform(0, max(t[-1] - dur, 0.1)); amp = rng.choice([-3.5, 3.5])
        m = (t >= t0) & (t <= t0 + dur)
        ph = (t[m] - t0) / dur
        y[m] += amp * 0.5 * (1 - np.cos(np.pi * ph)); y[t > t0 + dur] += amp
        vy[m] = amp * 0.5 * np.pi / dur * np.sin(np.pi * ph)
    return Trajectory(t, x, y, vx, vy)


def unit_to_heads(site: Site, x, y):
    """Unit vectors from the vehicle antenna to each head, shape (2, len(x), 3)."""
    h = site.heads()
    pos = np.stack([x, y, np.zeros_like(x)], axis=-1)
    d = h[:, None, :] - pos[None, :, :]
    return d / np.linalg.norm(d, axis=-1, keepdims=True), np.linalg.norm(d, axis=-1)


def doppler(site: Site, x, y, vx, vy):
    """One-way Doppler (Hz) at each head, shape (2, n); positive while approaching the head."""
    u, _ = unit_to_heads(site, x, y)
    vel = np.stack([vx, vy, np.zeros_like(vx)], axis=-1)
    return np.einsum("hnk,nk->hn", u, vel) / LAMBDA
