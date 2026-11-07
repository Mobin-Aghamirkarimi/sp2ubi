import os
import sys

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885"), os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tr37885"), "..")]
import nlosv_blockage as pb  # noqa: E402


def test_drop_gaps_and_types():
    rng = np.random.default_rng(0)
    b = pb.drop_blockers(rng, 0, 500_000, 3.5, 15.0, 0.0)       # ~13,000 vehicles
    rear = b.x0 - b.length / 2
    front = b.x0 + b.length / 2
    gaps = rear[1:] - front[:-1]
    assert gaps.min() >= 2.0 - 1e-9
    assert abs(np.mean(gaps) - 30.07) < 1.0                    # E max{2, Exp(30)} = 2 + 30 e^(-1/15)
    share = np.bincount(b.vtype, minlength=4)[1:] / len(b.vtype)
    assert np.allclose(share, [0.2, 0.6, 0.2], atol=0.015)


def test_segment_box():
    lo, hi = np.array([-1.0, 3.0, 0.0]), np.array([1.0, 4.0, 3.0])
    assert pb.segment_hits_box(np.array([0.0, 0.0, 1.6]), np.array([0.0, 8.0, 5.0]), lo, hi)       # z at y=3..4: 2.9..3.3
    assert not pb.segment_hits_box(np.array([0.0, 0.0, 1.6]), np.array([0.0, 8.0, 5.0]), lo, hi * [1, 1, 0.9])
    assert not pb.segment_hits_box(np.array([5.0, 0.0, 1.6]), np.array([5.0, 8.0, 5.0]), lo, hi)


def test_car_never_blocks_roof_antenna_truck_does():
    # a 1.6 m car cannot cut the segment from a 1.6 m antenna rising to 5 m; a 3 m truck can
    head = np.array([0.0, 8.0, 5.0])
    for h, expect in ((1.6, -1), (3.0, 0)):
        b = pb.Blockers(np.array([0.0]), np.array([13.0]), np.array([2.6]), np.array([h]), np.array([3]), 3.5, 0.0, 0.0)
        s = pb.beacon_states(b, np.array([0.0]), np.array([0.0]), np.array([0.0]), 1.6, head)
        assert s[0] == expect


def test_loss_cases_and_events():
    assert pb.blockage_case(1.6, 5.0, 3.0) == 3 and pb.blockage_case(1.6, 5.0, 1.6) == 3
    assert pb.blockage_case(3.5, 5.0, 3.0) == 1 and pb.blockage_case(1.0, 2.0, 3.0) == 2
    rng = np.random.default_rng(1)
    st = np.array([-1, 0, 0, 0, -1, 2, 2])
    loss = pb.blockage_loss_db(rng, st, np.full(7, 50.0), 1.6, 5.0, np.array([3.0, 3.0, 3.0]))
    assert loss[0] == 0 and loss[4] == 0
    assert len(set(loss[1:4])) == 1 and len(set(loss[5:7])) == 1   # one draw per event (mean constant below 548 m)
    rng = np.random.default_rng(2)
    z = np.array([pb.blockage_loss_db(rng, np.array([0]), np.array([50.0]), 1.6, 5.0, np.array([3.0]))[0]
                  for _ in range(20000)])
    # max{0, N(5, 4)}: mean = 5 Phi(5/4) + 4 phi(5/4) = 5.2795
    assert abs(z.mean() - 5.2795) < 0.1


def test_assemble_switch():
    n, M = 5, 4
    sl, sn = np.ones((n, M)), 2 * np.ones((n, M))
    st = np.array([-1, 0, 0, -1, 1]); loss = np.array([0, 6.0, 6.0, 0, 0.0])
    sig, g = pb.assemble_head_a(sl, sn, np.ones(n), 4 * np.ones(n), st, loss)
    assert np.allclose(sig[0], 1) and np.allclose(sig[1], 2 * 10 ** (-0.3)) and np.allclose(sig[4], 2)
    assert np.allclose(g, [1, 4 * 10 ** (-0.6), 4 * 10 ** (-0.6), 1, 4])
