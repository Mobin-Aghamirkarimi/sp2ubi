"""P3 NLOSv mechanics (cell P3): blockers in the lane between the vehicle and head A, dropped with the
TR 37.885 V15.3.0 clause 6.1.2 rule (types 1-3: 5 x 2.0 x 1.6 m, 5 x 2.0 x 1.6 m, 13 x 2.6 x 3 m; bumper gap
max{2 m, Exp(mean = speed * 2 s)}; all vehicles in a lane at the same speed); a beacon is NLOSv at head A when a
blocker box intersects the 3D segment vehicle antenna -> head A; the state is set per beacon; the blockage loss is the
clause 6.2.1 additional loss max{0 dB, N(mean, sd) in dB}, case by antenna and blocker heights, one draw per blockage
event (a run of consecutive blocked beacons by the same blocker) with the distance-dependent mean per beacon.
Implementation readings (constants `nlosv_blockage.py`): type mix of urban option B
(20/60/20 %), blocker-lane speed per pass uniform over the urban option B lane speeds {60, 50, 25, 15} km/h, blocker
lane centre 3.5 m (TR Annex A urban lane width) from the vehicle's lane centre towards the heads, same direction."""
from dataclasses import dataclass

import numpy as np

TYPES = {1: (5.0, 2.0, 1.6), 2: (5.0, 2.0, 1.6), 3: (13.0, 2.6, 3.0)}          # length, width, height (m)


@dataclass
class Blockers:
    x0: np.ndarray        # centre x of each box at t = t_ref (m)
    length: np.ndarray
    width: np.ndarray
    height: np.ndarray
    vtype: np.ndarray
    y_c: float            # lane centre (m)
    speed: float          # m/s, +x
    t_ref: float


def drop_blockers(rng, x_lo, x_hi, y_c, speed, t_ref, mix=(0.2, 0.6, 0.2)):
    """Drop vehicles in one lane over [x_lo, x_hi] at time t_ref (TR 37.885 clause 6.1.2 gap rule), types by `mix`."""
    xs, L, W, H, T = [], [], [], [], []
    x = x_lo + rng.uniform(0, max(2.0, speed * 2.0))       # random phase of the first rear bumper
    while x < x_hi:
        vt = int(rng.choice([1, 2, 3], p=mix))
        l, w, h = TYPES[vt]
        xs.append(x + l / 2); L.append(l); W.append(w); H.append(h); T.append(vt)
        x = x + l + max(2.0, rng.exponential(speed * 2.0))
    return Blockers(np.array(xs), np.array(L), np.array(W), np.array(H), np.array(T), float(y_c), float(speed), float(t_ref))


def segment_hits_box(p0, p1, lo, hi):
    """Slab test: does the segment p0 -> p1 (3) intersect the axis-aligned box [lo, hi] (3)?"""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if p0[k] < lo[k] or p0[k] > hi[k]:
                return False
            continue
        a, b = (lo[k] - p0[k]) / d[k], (hi[k] - p0[k]) / d[k]
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def beacon_states(blk: Blockers, t_b, xv, yv, zv, head):
    """Per beacon: index of the blocker that cuts the vehicle -> head segment (first in list order), or -1."""
    out = np.full(len(t_b), -1, int)
    for i, t in enumerate(t_b):
        xc = blk.x0 + blk.speed * (t - blk.t_ref)
        p0, p1 = np.array([xv[i], yv[i], zv]), np.asarray(head, float)
        xmin, xmax = min(p0[0], p1[0]), max(p0[0], p1[0])
        cand = np.where((xc + blk.length / 2 >= xmin) & (xc - blk.length / 2 <= xmax))[0]
        for j in cand:
            lo = np.array([xc[j] - blk.length[j] / 2, blk.y_c - blk.width[j] / 2, 0.0])
            hi = np.array([xc[j] + blk.length[j] / 2, blk.y_c + blk.width[j] / 2, blk.height[j]])
            if segment_hits_box(p0, p1, lo, hi):
                out[i] = j
                break
    return out


def blockage_case(h_tx, h_rx, h_blk):
    if min(h_tx, h_rx) > h_blk:
        return 1
    if max(h_tx, h_rx) < h_blk:
        return 2
    return 3


def blockage_loss_db(rng, state, d3d, h_veh, h_head, heights):
    """TR 37.885 clause 6.2.1 additional loss per beacon (0 where not blocked); one standard-normal draw per event."""
    loss = np.zeros(len(state))
    i = 0
    while i < len(state):
        j = state[i]
        if j < 0:
            i += 1
            continue
        k = i
        while k + 1 < len(state) and state[k + 1] == j:
            k += 1
        case = blockage_case(h_veh, h_head, heights[j])
        z = rng.standard_normal()
        if case != 1:
            base, sd = (9.0, 4.5) if case == 2 else (5.0, 4.0)
            mean = base + np.maximum(0.0, 15 * np.log10(d3d[i:k + 1]) - 41)
            loss[i:k + 1] = np.maximum(0.0, mean + sd * z)
        i = k + 1
    return loss


LANE_SPEEDS_KMH = (60.0, 50.0, 25.0, 15.0)   # TR 37.885 clause 6.1.2 urban option B, East-West lanes
MIX_URBAN_B = (0.2, 0.6, 0.2)
LANE_WIDTH = 3.5                               # TR 37.885 Annex A, urban


def pass_states(rng, site, traj, t_b, head_z, veh_z, mix=MIX_URBAN_B, speeds_kmh=LANE_SPEEDS_KMH, clearance=0.5,
                y_c=None):
    """Blockers and per-beacon NLOSv state at head A (index 0 of site.heads) for one pass. Returns a dict with the
    state per beacon (blocker index or -1), loss_db per beacon, d3d, lane speed, lane centre and whether the lane
    centre had to be moved towards the vehicle to keep `clearance` between the widest box and the head line.
    y_c: blocker-lane centre in the in-house frame (None: vehicle's mean y + 3.5 m towards head A)."""
    x, y, _, _ = traj.at(t_b)
    head = np.array([site.heads()[0][0], site.heads()[0][1], head_z])
    side = np.sign(head[1] - np.mean(y)) or 1.0
    if y_c is None:                                       # the site-accuracy simulation kinematics: adjacent lane towards head A
        y_c = float(np.mean(y)) + side * LANE_WIDTH       # (SUMO kinematics: test vehicle in lane 1, y_c = 0 = lane 0)
    gap = np.min(np.abs(y - y_c))
    if gap < LANE_WIDTH / 2 + TYPES[3][1] / 2:           # never a vehicle inside the blocker lane
        raise ValueError(f"p3: vehicle {gap:.2f} m from the blocker-lane centre (< {LANE_WIDTH / 2 + TYPES[3][1] / 2:.2f} m)")
    lim = head[1] - side * (TYPES[3][1] / 2 + clearance)
    clipped = bool(side * (y_c - lim) > 0)
    if clipped:
        y_c = float(lim)
    speed = float(rng.choice(speeds_kmh)) / 3.6
    span = abs(x[-1] - x[0]) + 2 * site.D
    blk = drop_blockers(rng, min(x[0], x[-1]) - span - speed * (t_b[-1] - t_b[0]), max(x[0], x[-1]) + span,
                        y_c, speed, t_b[0], mix)
    st = beacon_states(blk, t_b, x, y, veh_z, head)
    d3d = np.sqrt((head[0] - x) ** 2 + (head[1] - y) ** 2 + (head_z - veh_z) ** 2)
    loss = blockage_loss_db(rng, st, d3d, veh_z, head_z, blk.height)
    return dict(state=st, loss_db=loss, d3d=d3d, speed=speed, y_c=y_c, clipped=clipped, blockers=blk)


def assemble_head_a(sig_los, sig_nlosv, gain_los, gain_nlosv, state, loss_db):
    """Per-beacon switch for head A: LOS configuration where not blocked, NLOSv configuration attenuated by the
    blockage loss where blocked. sig_*: [n_b, M]; gain_*: [n_b] (linear large-scale gain). Returns sig, gain."""
    b = state >= 0
    att = 10 ** (-loss_db / 20)
    sig = np.where(b[:, None], sig_nlosv * att[:, None], sig_los)
    gain = np.where(b, gain_nlosv * att ** 2, gain_los)
    return sig, gain
