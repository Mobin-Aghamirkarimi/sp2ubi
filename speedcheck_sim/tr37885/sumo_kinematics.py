"""SUMO kinematics for the speed-check simulation: one SUMO 1.28 run per site on a straight road, Krauss
car following with the default sigma = 0.5 (IDM as a sensitivity), ballistic position update, 0.1-s time step; background
traffic with the TR 37.885 clause 6.1.2 drop statistics (bumper gap max{2 m, Exp(mean = speed * 2 s)}, same speed in a
lane, type 2 cars, urban option A / highway option A); urban: 2 lanes per direction of 3.5 m and one signal-controlled
junction 50-150 m downstream of head B (TR Annex A lane width); highway: 3 lanes per direction of 4 m.
Test vehicles are inserted one after another (one pass each) and their FCD is cut to the pass window
x in [-0.35 D, 1.35 D] (as `quadriga/driver.py.build_pass`), resampled to 0.01 s with SUMO's ballistic kinematics (speed
linear per step, position its integral, checked against SUMO's positions to 5 cm) and returned as `fdoa.geometry.Trajectory` in the in-house frame (x along the road
with head A at 0, y to the right of travel from the centre of the outer lane, heads at y = pole offset).
Plausibility: |a| <= 4.5 m/s^2 (SUMO's default passenger-car decel, an assumption, not a TR value) is checked and
the number of violating samples is returned."""
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ER = os.path.dirname(os.path.abspath(__file__))
STEP = 0.1
A_MAX = 4.5

import sys  # noqa: E402
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
from fdoa.geometry import Site, Trajectory  # noqa: E402

BIN = os.environ.get("SUMO_BIN_DIR", os.path.dirname(os.path.abspath(sys.executable)))   # eclipse-sumo (pip)


@dataclass
class Scenario:
    name: str               # "urban" or "highway"
    lanes: int
    lane_width: float
    junction: bool


URBAN = Scenario("urban", 2, 3.5, True)
HIGHWAY = Scenario("highway", 3, 4.0, False)


def _write_net(wd, sc: Scenario, x_lo, x_hi, x_j, v0):
    nodes = [f'<node id="W" x="{x_lo:.2f}" y="0" type="priority"/>', f'<node id="E" x="{x_hi:.2f}" y="0" type="priority"/>']
    if sc.junction:
        nodes += [f'<node id="J" x="{x_j:.2f}" y="0" type="traffic_light"/>',
                  f'<node id="N" x="{x_j:.2f}" y="200" type="priority"/>', f'<node id="S" x="{x_j:.2f}" y="-200" type="priority"/>']
        edges = [f'<edge id="WJ" from="W" to="J" numLanes="{sc.lanes}" speed="{v0:.3f}" width="{sc.lane_width}"/>',
                 f'<edge id="JE" from="J" to="E" numLanes="{sc.lanes}" speed="{v0:.3f}" width="{sc.lane_width}"/>',
                 f'<edge id="NJ" from="N" to="J" numLanes="{sc.lanes}" speed="{v0:.3f}" width="{sc.lane_width}"/>',
                 f'<edge id="JS" from="J" to="S" numLanes="{sc.lanes}" speed="{v0:.3f}" width="{sc.lane_width}"/>']
        route = "WJ JE"
    else:
        edges = [f'<edge id="WE" from="W" to="E" numLanes="{sc.lanes}" speed="{v0:.3f}" width="{sc.lane_width}"/>']
        route = "WE"
    open(os.path.join(wd, "n.nod.xml"), "w").write("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n")
    open(os.path.join(wd, "e.edg.xml"), "w").write("<edges>\n" + "\n".join(edges) + "\n</edges>\n")
    r = subprocess.run([os.path.join(BIN, "netconvert"), "-n", "n.nod.xml", "-e", "e.edg.xml", "-o", "net.xml",
                        "--no-turnarounds", "true", "--tls.default-type", "static"], cwd=wd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return route


def _write_routes(rng, wd, route, v0, n_test, t_first, t_gap, horizon, model, lanes, test_lane, mean_gap, background=True,
                  keep_lane=False):
    """Background flow per lane, headway (max{2, Exp(mean_gap)} + 5 m) / v0 (TR clause 6.1.2 gap rule in time with
    mean_gap = 2 s * v0; thinned at a signal, see site_passes); test vehicles every t_gap s plus a uniform jitter over one
    90 s signal cycle (desired speed exactly v0)."""
    cf = 'carFollowModel="IDM"' if model == "IDM" else 'carFollowModel="Krauss" sigma="0.5"'
    # keep_lane: the test vehicle never changes lane (strategic, keep-right, speed-gain and cooperative changes off;
    # otherwise SUMO's keep-right moves the P3 test vehicle into the blocker lane)
    keep = (' laneChangeModel="LC2013" lcStrategic="-1" lcKeepRight="0" lcSpeedGain="0" lcCooperative="0"'
            if keep_lane else "")
    vt = [f'<vType id="bg" vClass="passenger" length="5" minGap="2" {cf} speedFactor="1" speedDev="0"/>',
          f'<vType id="test" vClass="passenger" length="5" minGap="2" {cf} speedFactor="1" speedDev="0"{keep}/>']
    ev = []
    for lane in range(lanes if background else 0):
        t = rng.uniform(0, 2.0)
        k = 0
        while t < horizon:
            ev.append((t, f'<vehicle id="bg{lane}_{k}" type="bg" route="r" depart="{t:.2f}" departLane="{lane}" departSpeed="desired"/>'))
            gap = max(2.0, rng.exponential(mean_gap))
            t += (gap + 5.0) / v0
            k += 1
    lane_attr = "random" if test_lane is None else str(test_lane)
    for i in range(n_test):
        t = t_first + i * t_gap + rng.uniform(0.0, 90.0)
        ev.append((t, f'<vehicle id="test{i}" type="test" route="r" depart="{t:.2f}" departLane="{lane_attr}" departSpeed="desired"/>'))
    ev.sort(key=lambda e: e[0])
    txt = "<routes>\n" + "\n".join(vt) + f'\n<route id="r" edges="{route}"/>\n' + "\n".join(e[1] for e in ev) + "\n</routes>\n"
    open(os.path.join(wd, "r.rou.xml"), "w").write(txt)


def _lane_y(net, lane_id):
    for el in ET.parse(net).getroot().iter("lane"):
        if el.get("id") == lane_id:
            ys = [float(p.split(",")[1]) for p in el.get("shape").split()]
            return float(np.mean(ys))
    raise KeyError(lane_id)


def _read_fcd(path, prefix="test"):
    out = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "timestep":
            t = float(el.get("time"))
            for v in el.findall("vehicle"):
                vid = v.get("id")
                if vid.startswith(prefix):
                    out.setdefault(vid, []).append((t, float(v.get("x")), float(v.get("y")), float(v.get("speed")),
                                                    float(v.get("acceleration", "nan")), float(v.get("angle"))))
            el.clear()
    return {k: np.array(v) for k, v in out.items()}


def _to_trajectory(a, site: Site, x_head_a, y_outer, x_start, x_end, dt=0.01):
    """FCD rows (t, x, y, speed, acc, angle) -> Trajectory in the in-house frame over the pass window, resampled."""
    t, x, y, v = a[:, 0], a[:, 1] - x_head_a, -(a[:, 2] - y_outer), a[:, 3]
    vx = v                  # SUMO speed = longitudinal speed along the straight +x lane (the FCD angle is only the
                            # drawn heading during a lane change; lateral speed comes from y)
    keep = (x >= x_start - 5.0) & (x <= x_end + 5.0)
    if keep.sum() < 4 or x[keep][0] > x_start or x[keep][-1] < x_end:
        return None
    t, x, y, vx = t[keep], x[keep], y[keep], vx[keep]
    # SUMO's own ballistic kinematics: speed linear within each 0.1 s step (|a| = |dv/dt| as SUMO reports), position
    # = its integral from the first sample (piecewise quadratic); y linear (lane changes are continuous over 4 s)
    tt = np.arange(t[0], t[-1], dt)
    vxs = np.interp(tt, t, vx)
    xs = x[0] + np.concatenate([[0.0], np.cumsum(0.5 * (vxs[1:] + vxs[:-1]) * dt)])
    ys = np.interp(tt, t, y)
    k = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, len(t) - 2)
    vys = (np.diff(y) / np.diff(t))[k]                    # lateral speed: derivative of the linear y
    drift = float(np.max(np.abs(xs - np.interp(tt, t, x))))
    if drift > 0.05:
        raise RuntimeError(f"sumo_kin: integrated position deviates from SUMO's by {drift:.3f} m")
    m = (xs >= x_start) & (xs <= x_end)
    i0, i1 = np.argmax(m), len(m) - np.argmax(m[::-1])
    tt = tt[i0:i1] - tt[i0]
    return Trajectory(tt, xs[i0:i1], ys[i0:i1], vxs[i0:i1], vys[i0:i1])


SAT_HEADWAY_S = 2.0      # assumed saturation headway at the signal (1800 veh/h/lane)
GREEN_RATIO = 0.5        # netconvert static program: 42 s green + 3 s yellow of a 90 s cycle per approach
DEMAND_VC = 0.7          # degree of saturation of the background demand at the signal


def site_passes(rng, site: Site, v0, n_pass, scenario: Scenario = URBAN, model="Krauss", test_lane=None,
                t_gap=None, workdir=None, keep=False, demand_vc=DEMAND_VC, background=True, keep_lane=None):
    """Run SUMO once for a site and return (trajectories, info). background=False: free road (survey-like reference
    passes, C2 (iii)); the signal still operates. info: junction position, number of FCD samples with
    |a| > A_MAX, stopped passes (speed < 0.5 m/s inside the window), lane-change passes, passes dropped (not
    crossing the window)."""
    D = site.D
    x_head_a = 1000.0                                     # SUMO x of head A
    x_j = x_head_a + D + rng.uniform(50.0, 150.0) if scenario.junction else None
    x_lo, x_hi = 0.0, x_head_a + D + 600.0
    wd = workdir or tempfile.mkdtemp(prefix="sumo_")
    route = _write_net(wd, scenario, x_lo, x_hi, x_j, v0)
    if t_gap is None:
        t_gap = max(15.0, 1.5 * (x_hi - x_lo) / v0 / 4)
    t_first = 120.0                                       # warm-up of the background flow
    horizon = t_first + (n_pass + 4) * (t_gap + 90.0) + (x_hi - x_lo) / v0 * 3
    if scenario.junction:     # the TR density (headway ~ 2 s + 5 m / v0) exceeds the signal's capacity (g/C = 0.5):
        q = demand_vc * GREEN_RATIO / SAT_HEADWAY_S          # thinned to degree of saturation demand_vc (veh/s/lane)
        mean_gap = max(2.0, v0 / q - 5.0)
    else:
        mean_gap = 2.0 * v0
    _write_routes(rng, wd, route, v0, n_pass + 4, t_first, t_gap + 90.0, horizon, model, scenario.lanes, test_lane,
                  mean_gap, background, keep_lane=(test_lane is not None) if keep_lane is None else keep_lane)
    cmd = [os.path.join(BIN, "sumo"), "-n", "net.xml", "-r", "r.rou.xml", "--step-length", str(STEP),
           "--step-method.ballistic", "true", "--lanechange.duration", "4", "--fcd-output", "fcd.xml",
           "--device.fcd.explicit", ",".join(f"test{i}" for i in range(n_pass + 4)), "--fcd-output.acceleration", "true",
           "--seed", str(int(rng.integers(1, 2 ** 31 - 1))), "--end", f"{horizon:.1f}", "--no-step-log", "true",
           "--no-warnings", "true", "--time-to-teleport", "-1", "--precision", "6"]
    r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    fcd = _read_fcd(os.path.join(wd, "fcd.xml"))
    y_outer = _lane_y(os.path.join(wd, "net.xml"), route.split()[0] + "_0")   # SUMO y of the outer lane centre
    trajs, n_acc, stopped, lc, dropped = [], 0, 0, 0, 0
    for i in range(n_pass + 4):
        a = fcd.get(f"test{i}")
        if a is None:
            dropped += 1
            continue
        tr = _to_trajectory(a, site, x_head_a, y_outer, -0.35 * D, 1.35 * D)
        if tr is None:
            dropped += 1
            continue
        xw = a[:, 1] - x_head_a
        inw = (xw >= -0.35 * D) & (xw <= 1.35 * D)
        n_acc += int(np.sum(np.abs(a[inw, 4]) > A_MAX + 1e-9))
        stopped += int(np.any(a[inw, 3] < 0.5))
        lc += int(np.ptp(tr.y) > 1.0)
        trajs.append(tr)
        if len(trajs) == n_pass:
            break
    info = dict(junction_x=None if x_j is None else x_j - x_head_a, acc_violations=n_acc, stopped_passes=stopped,
                lane_change_passes=lc, dropped=dropped, n=len(trajs), workdir=wd if keep else None)
    if not keep and workdir is None:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)
    return trajs, info
