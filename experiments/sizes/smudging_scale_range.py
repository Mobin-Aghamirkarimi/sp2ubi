"""Range of the scale factor c = round(2^S / W') over the crossings the RSU's code path accepts, and its ratio to the
smudging standard deviation (one-term smudging: the whole term is drawn by the OBU's TEE).

The privacy statement "an entry is formed only if c <= 2^20, at least 16 times below the term's standard deviation, so
B mod c is statistically uniform" rests on this guard, which holds for every entry by construction. The script also
gives the evaluated-geometry figure "for constant-speed crossings of the evaluated geometry c <= <c_max_code_path>,
more than <floor(sigma/c)> times below it". SensingRSU.accept accepts a burst that spans only SPAN_MIN of the window,
and W' = W (1 + site bias) shrinks with a negative bias; both are included below.

Model (sp2ubi/speedcheck.py SensingRSU.accept / evaluate_weighted; the statistic b_hat and Enc(B) of the scheme):
  * W' = round(W (1 + beta_s)) carries the speed units: profile W = sum_i v_i^2, matched W = sum_i w_i v_i with
    w_i = round(64 g_i^2), speeds in UNIT_MPS = 0.1 m/s; site bias beta_s over SITE_BIASES (see below);
  * crossing window [-0.05 L, 1.05 L], L = 100 m (T_w = 110 m / v), beacons every BEACON_PERIOD_MS, at most K_MAX per
    entry; constant speed v over the window;
  * acceptance as in the code: the used beacons span >= SPAN_MIN of the window, i.e. an index range of
    m >= ceil(SPAN_MIN T_w / 0.1 s) + 1 beacons, which must fit one burst (m <= BURST_MS / BEACON_PERIOD_MS), and at
    least ceil(MIN_COVER m) of them are kept. Speeds therefore start at
    V_MIN = SPAN_MIN * 110 m / ((BURST_MS - BEACON_PERIOD_MS) / 1 s) (= 2.481 m/s), not at 110 m / 40 s;
  * the kept beacons are chosen adversarially: the ones with the smallest weights anywhere in the window (a relaxation
    of the contiguous-span constraint, so the reported c is an upper bound for the code path);
  * speeds: a 0.005 m/s grid from V_MIN to 5 m/s, the top of every 0.1 m/s speed unit there (where c peaks within a
    unit), and a coarse grid to V_MAX; rho (closest approach) 1-50 m;
  * SITE_BIASES: 0, -3 %, -6 % and +6 %. The largest site biases of the site-accuracy simulation are -1.35 % ... +1.6 %,
    so +-6 % is a conservative envelope. Only a negative bias raises c.
  A stop-and-go crossing whose fast beacons are lost can have a smaller W' than any constant-speed crossing; for it,
  and for any site bias, only the guard C_MAX applies (sigma / c >= 16.4). The privacy statement rests on that guard.
The wrapped-Gaussian distance from uniform on Z/cZ is <= 2 sum_k exp(-2 pi^2 k^2 (sigma/c)^2); its log10 is reported
for k = 1.
Run (from the repository root; no network, CPU seconds):  python experiments/sizes/smudging_scale_range.py
Output: experiments/sizes/results/smudging_scale_range.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from sp2ubi import speedcheck as sc  # noqa: E402

L = 100.0
LO, HI = -0.05 * L, 1.05 * L
SIGMA = sc.SMUDGE_FRAC * (1 << sc.S_BITS)                 # one-term: the whole term is drawn by the OBU's TEE
BURST_BEACONS = sc.BURST_MS // sc.BEACON_PERIOD_MS          # 400 beacons per burst (indices 0 .. 399)
V_MIN = sc.SPAN_MIN * (HI - LO) / ((sc.BURST_MS - sc.BEACON_PERIOD_MS) / 1000.0)   # 2.481 m/s: 90 % span in one burst
V_FIT = (HI - LO) / (sc.BURST_MS / 1000.0)                 # 2.75 m/s: the whole window fits one burst
SPEEDS = sorted({round(V_MIN + 0.005 * i, 6) for i in range(int((5.0 - V_MIN) / 0.005) + 1)}
                | {round((u + 0.5) * sc.UNIT_MPS - 1e-6, 7) for u in range(25, 50)}   # top of each speed unit (worst)
                | {V_MIN, V_FIT, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 38, 45, 50, 60, sc.V_MAX * sc.UNIT_MPS})
RHOS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
SITE_BIASES = [0.0, -0.03, -0.06, 0.06]
EVAL_CLASSES = (10, 38)                                      # speed classes of the evaluation (m/s)


def g2(x, rho):
    return (x / math.hypot(x, rho) + (L - x) / math.hypot(L - x, rho)) ** 2


def crossing(v, rho, rule):
    """W (before the site bias) of the adversarial accepted crossing at speed v; None if no crossing is accepted.
    rule "code": the code's span rule (SPAN_MIN of the window, MIN_COVER of the spanned index range, one burst);
    rule "all": every beacon of the window kept (window inside one burst)."""
    dt = sc.BEACON_PERIOD_MS / 1000.0
    t_w = (HI - LO) / v
    n = int(math.floor(t_w / dt + 1e-9)) + 1                 # beacons inside the window
    if rule == "all":
        if n > BURST_BEACONS:
            return None
        keep = min(n, sc.K_MAX)
    else:
        m = math.ceil(sc.SPAN_MIN * t_w / dt - 1e-9) + 1     # smallest index range that spans SPAN_MIN of the window
        if m > BURST_BEACONS:
            return None
        keep = min(sc.K_MAX, max(3, math.ceil(sc.MIN_COVER * m - 1e-9)))
    xs = [LO + i * v * dt for i in range(n)]
    vu = sc.speed_to_units(v)
    w = sorted(round(64 * g2(x, rho)) for x in xs)[:keep]
    return {"profile": keep * vu * vu, "matched": sum(w) * vu}, n, keep


def c_of(W, bias):
    return round((1 << sc.S_BITS) / round(W * (1.0 + bias)))


def tv_log10(c):
    return -2 * math.pi ** 2 * (SIGMA / c) ** 2 / math.log(10)


def main():
    res = {"model": __doc__.split("Model")[1].split("Run")[0].strip(), "sigma_one_term": SIGMA, "C_MAX": sc.C_MAX,
           "C_MIN": sc.C_MIN, "v_min_mps": V_MIN, "v_window_fits_burst_mps": V_FIT, "n_speeds": len(SPEEDS),
           "rho_m": RHOS, "site_biases": SITE_BIASES, "cases": {}}
    for rule in ("code", "all"):
        for bias in SITE_BIASES:
            case = {}
            for stat in ("profile", "matched"):
                best = None
                for v in SPEEDS:
                    for rho in RHOS:
                        cr = crossing(v, rho, rule)
                        if cr is None:
                            continue
                        W, n, keep = cr
                        c = c_of(W[stat], bias)
                        if best is None or c > best["c"]:
                            best = dict(c=c, W=W[stat], v_mps=v, rho_m=rho, n_window=n, n_kept=keep)
                best["sigma_over_c"] = SIGMA / best["c"]
                best["log10_tv_bound_k1"] = tv_log10(best["c"])
                case[stat] = best
            case["evaluated_classes_c_max"] = max(
                c_of(crossing(v, rho, rule)[0][stat], bias) for v in SPEEDS if EVAL_CLASSES[0] <= v <= EVAL_CLASSES[1]
                for rho in RHOS for stat in ("profile", "matched"))
            res["cases"][f"{rule}|bias={bias:+.2f}"] = case
    code = [v for k, v in res["cases"].items() if k.startswith("code|")]
    c_code = max(max(c["profile"]["c"], c["matched"]["c"]) for c in code)
    c_code0 = max(res["cases"]["code|bias=+0.00"]["profile"]["c"], res["cases"]["code|bias=+0.00"]["matched"]["c"])
    worst = sc.K_MAX * sc.W_MAX_WEIGHT * sc.V_MAX
    res["summary"] = {
        "c_max_code_path": c_code,
        "sigma_over_c_min_code_path": SIGMA / c_code,
        "c_max_code_path_bias0": c_code0,
        "sigma_over_c_min_code_path_bias0": SIGMA / c_code0,
        "c_max_evaluated_classes_code_path": max(c["evaluated_classes_c_max"] for c in code),
        "more_than_150_times": SIGMA / c_code > 150,
        "sigma_over_c_at_guard": SIGMA / sc.C_MAX,
        "log10_tv_bound_at_guard": tv_log10(sc.C_MAX),
        "guard_ever_reached_by_constant_speed_crossing": c_code > sc.C_MAX,
        "c_lower_bound_check_bounds": round((1 << sc.S_BITS) / worst),
        "note": "the privacy statement rests on the guard c <= C_MAX (sigma/c >= 16, any crossing, any site bias); "
                "c_max_code_path is the evaluated-geometry figure (constant speed, site bias in SITE_BIASES)",
    }
    out = os.path.join(HERE, "results", "smudging_scale_range.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    for k, v in res["cases"].items():
        print(k, {s: (v[s]["c"], v[s]["v_mps"], v[s]["rho_m"], round(v[s]["sigma_over_c"], 1)) for s in ("profile", "matched")})


if __name__ == "__main__":
    main()
