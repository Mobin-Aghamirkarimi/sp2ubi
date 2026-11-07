"""Decision simulation of the paper: the monitor variant (decisions_monitor.py) plus the R1 escalation. Every other
rule, input and seed is that of the monitor variant.

Escalation:
  * per epoch, flag_e = M >= M_MIN and m < thr - z_0.999 * SE_eff, with m, SE the winsorised inverse-variance mean of
    the level variant (same code path), SE_eff = sqrt(SE^2 + SE_cal^2) and SE_cal = sum w_i sigma_level,i /
    (sqrt(N_REF) sum w_i) (the survey calibration error treated as common-mode);
  * thr starts at S_RULE = -4 % (= the honest band edge -3 % minus 1 %); flags in ESC_K = 3 consecutive epochs escalate:
    the case follows the R2 / R3 inspection path (the insurer inspects the VSS);
  * after the inspection the case is closed and the counter restarts; for the rest of the policy year the policy's
    threshold is min(s_insp, -3 %) - 1 %, with s_insp the VSS scale measured at the inspection. In the simulation the
    inspection measures the vehicle's true honest scale s: a manipulating vehicle is assumed to remove the device for
    the announced inspection and to reinstall it;
  * the punitive outcome per vehicle-year is the union (R2 punitive) OR (any escalation), taken per vehicle.
    (R3, the consistency rule, is not simulated here.)
Added outputs per row: escalation_per_year (share of vehicle-years with >= 1 escalation), escalation_ub95,
inspections_per_year (mean number of escalations), punitive_union_per_year, punitive_union_ub95. Added section
sens_tyre: honest vehicles with s = -5 % exactly (an oversized tyre), reported, never part of the honest fallback test.
Usage: python inhouse/decisions.py --levels 8 [--vehicles 20000] [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import stats

import decisions_base as DB
import decisions_levels as E
from levels_qualification import requalify, scale_to_disclosed
from decisions_base import ALPHA_R1, C_WIN, EPOCHS, M_MIN, N_REF, S_RULE, cp_upper, seed
from decisions_levels import B_WINDOW
from decisions_monitor import (ARC_M, BURST_S, HONEST, SIGMA_MAX, _sim_view, atomic_json, decide_popcal, keep_prob,
                            simulate)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
ESC_K = 3                                            # consecutive flagged epochs that escalate (the R1 notice rule)
S_EDGE = HONEST["S_LO"]                              # -3 %: the pre-specified honest band edge
S_GAP = S_RULE - S_EDGE                              # -1 %
TYRE_S = -0.05                                       # sens_tyre stratum


def epoch_stats(sim, b):
    """m, SE (winsorised at C_WIN se_i) and SE_cal per vehicle-epoch."""
    key, se_i, n_veh = sim["key"], sim["se_i"], sim["n_veh"]
    nE = n_veh * EPOCHS
    oow = np.abs(b) > B_WINDOW
    b = np.where(oow, -B_WINDOW, b)
    w = 1.0 / se_i ** 2
    W = np.bincount(key, w, nE)
    sig_level = np.sqrt(np.maximum(se_i ** 2 - HONEST["SIGMA0"] ** 2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.bincount(key, w * b, nE) / W
        bw = m0[key] + np.clip(b - m0[key], -C_WIN * se_i, C_WIN * se_i)
        m = np.bincount(key, w * bw, nE) / W
        SE = 1.0 / np.sqrt(W)
        SE_cal = np.bincount(key, w * sig_level, nE) / (np.sqrt(N_REF) * W)
    ok = sim["M"].ravel() >= M_MIN
    sh = (n_veh, EPOCHS)
    return m.reshape(sh), SE.reshape(sh), SE_cal.reshape(sh), ok.reshape(sh)


def escalation(sim, b, s_insp, guard=True, recentre=True):
    """Per vehicle: number of escalations in the policy year (sequential over the epochs)."""
    m, SE, SE_cal, ok = epoch_stats(sim, b)
    SE_eff = np.sqrt(SE ** 2 + SE_cal ** 2) if guard else SE
    z = stats.norm.ppf(1 - ALPHA_R1)
    n_veh = sim["n_veh"]
    thr = np.full(n_veh, S_RULE)
    run = np.zeros(n_veh, int)
    n_esc = np.zeros(n_veh, int)
    for e in range(EPOCHS):
        with np.errstate(invalid="ignore"):
            flag = ok[:, e] & (m[:, e] < thr - z * SE_eff[:, e])
        run = np.where(flag, run + 1, 0)
        fire = run >= ESC_K
        n_esc += fire
        if recentre:
            thr = np.where(fire, np.minimum(s_insp, S_EDGE) + S_GAP, thr)
        run = np.where(fire, 0, run)
    return n_esc


def add_union(out, punitive, n_esc, n):
    esc = n_esc > 0
    union = punitive | esc
    out.update(escalation_per_year=float(esc.mean()), escalation_ub95=cp_upper(int(esc.sum()), n),
               inspections_per_year=float(n_esc.mean()), punitive_union_per_year=float(union.mean()),
               punitive_union_ub95=cp_upper(int(union.sum()), n))
    return out


def honest_run(tag, A, sw, qual, nv, lam, vt, qtab, nl, honest=HONEST, cell_mult=None, stop_mask=None):
    sim = simulate(seed("h5", tag, vt, lam), A, sw, qual, nv, lam, vt, honest=honest, cell_mult=cell_mult,
                   stop_mask=stop_mask)
    d, f = decide_popcal(sim, sim["b_hon"], qtab, nl)
    h = E.honest_summary(d)
    h.update(entries_per_epoch_mean=float(sim["M"].mean()), crossings_per_epoch_mean=float(sim["M0"].mean()),
             low_m_share=float((sim["M"] < DB.M_MIN).mean()), popcal_scale=f.round(3).tolist())
    return add_union(h, d["punitive"], escalation(sim, sim["b_hon"], sim["s"]), nv)


def attack_run(att, A, sw, qual, nv, lam, vt, qtab, nl, honest=HONEST):
    sim = simulate(seed("p5", att, vt, lam), A, sw, qual, nv, lam, vt, honest=honest, attack=att,
                   rng_att=seed("a5", att, vt, lam))
    _, f = decide_popcal(sim, sim["b_hon"], qtab, nl)
    va = _sim_view(sim, True)
    da = E.decide(va, sim["b_att"], qtab, r2_scale=f[va["lvl"]])
    P = float(da["punitive"].mean())
    r = dict(punitive_per_year=P, punitive_ub95=cp_upper(int(da["punitive"].sum()), nv),
             notice_per_year=float(da["notice"].mean()))
    add_union(r, da["punitive"], escalation(va, sim["b_att"], sim["s"]), nv)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, choices=(4, 8), required=True)
    ap.add_argument("--vehicles", type=int, default=20_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--monitor", default=os.path.join(HERE, "results", "residual_monitor_full.json"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    nv = 400 if a.smoke else a.vehicles
    nl = a.levels
    t0 = time.time()
    qtab = DB.r2_quantiles(10_000_000, 1)
    A0 = E.load_sites(["extra"])
    sw = E.site_weights(A0, "floor")
    A, qual = requalify(A0, SIGMA_MAX, nl)
    log = lambda *x: print(*x, flush=True)
    log(f"levels {np.round(100 * A['levels'], 2)} %, qualified share of (site, speed) cells {qual.mean():.2f}")
    tag = "smoke" if a.smoke else "full"
    part = os.path.join(OUT, f"decisions_L{nl}_{tag}.partial.json")
    rep = json.load(open(part)) if os.path.exists(part) else dict(
        params=dict(levels=A["levels"].tolist(), n_levels=nl, SIGMA_MAX=SIGMA_MAX, BURST_S=BURST_S, ARC_M=ARC_M,
                    keep_prob={str(v): float(keep_prob(v)) for v in A["speeds"]},
                    escalation=dict(k=ESC_K, s_rule=S_RULE, s_edge=S_EDGE, alpha=ALPHA_R1, n_ref=N_REF,
                                    guard="SE_eff = sqrt(SE^2 + SE_cal^2)", recentre="min(s_insp, -3 %) - 1 %")),
        honest={}, margin={}, badsites={}, power={}, sens_tyre={})
    log(f"resuming: {sum(len(rep[k]) for k in ('honest', 'margin', 'badsites', 'power', 'sens_tyre'))} results in {part}")

    def save():
        atomic_json(rep, part)
    lams = (5, 15, 40)
    for cv in (0.0, 1.0):
        hon = dict(HONEST, LAM_CV=cv)
        for vt in E.PROFILES:
            for lam in lams:
                if f"cv{cv}|{vt}|{lam}" in rep["honest"]:
                    continue
                h = honest_run(f"cv{cv}", A, sw, qual, nv, lam, vt, qtab, nl, honest=hon)
                rep["honest"][f"cv{cv}|{vt}|{lam}"] = h; save()
                log(f"honest cv={cv} {vt:8s} lam={lam:2d}: punitive/yr {h['punitive_per_year']:.1e} union/yr "
                    f"{h['punitive_union_per_year']:.1e} (UB {h['punitive_union_ub95']:.1e}) esc/yr {h['escalation_per_year']:.1e}")
    for vt in E.PROFILES:                                    # sens_tyre: reported, never in the fallback test
        for lam in lams:
            if f"tyre|{vt}|{lam}" in rep["sens_tyre"]:
                continue
            h = honest_run("tyre", A, sw, qual, nv // 2, lam, vt, qtab, nl, honest=dict(HONEST, S_LO=TYRE_S, S_HI=TYRE_S))
            rep["sens_tyre"][f"tyre|{vt}|{lam}"] = h; save()
            log(f"tyre s={TYRE_S} {vt:8s} lam={lam:2d}: esc/yr {h['escalation_per_year']:.2f} inspections/yr "
                f"{h['inspections_per_year']:.2f}")
    margins = ([("DISCLOSED", k) for k in (1.0, 1.2, 1.5)] + [("SBIAS", x) for x in (0.003, 0.006, 0.010)]
               + [("SMULT", x) for x in (0.2, 0.35, 0.5)] + [("VOFF", x) for x in (0.002, 0.006)]
               + [("DRIFT", 0.010), ("COMBINED", None)])
    for k, x in margins:
        cm = scale_to_disclosed(A, x) if k == "DISCLOSED" else None
        hon = (dict(HONEST, SBIAS=0.003, SMULT=0.2, VOFF=0.002) if k == "COMBINED" else
               HONEST if k == "DISCLOSED" else dict(HONEST, **{k: x}))
        if k == "COMBINED":
            cm = scale_to_disclosed(A, 1.2)
        for vt, lam in (("urban", 15), ("highway", 40), ("suburban", 5)):
            if f"{k}|{x}|{vt}|{lam}" in rep["margin"]:
                continue
            h = honest_run(f"m|{k}|{x}", A, sw, qual, nv // 2, lam, vt, qtab, nl, honest=hon, cell_mult=cm)
            rep["margin"][f"{k}|{x}|{vt}|{lam}"] = h; save()
            log(f"margin {k:9s} {str(x):6s} {vt:8s} lam={lam:2d}: union/yr {h['punitive_union_per_year']:.1e} "
                f"(UB {h['punitive_union_ub95']:.1e})")
    mon = json.load(open(a.monitor)) if os.path.exists(a.monitor) else None
    p_stop = None
    if mon is not None:
        ps = [c["p_stop"] for c in mon["cells"] if c["change"] == "both"]
        p_stop = float(np.mean(ps)) if ps else None
    for share in (0.10, 0.25):
        rng = np.random.default_rng(int(1000 * share))
        bad = rng.random(A["nS"]) < share
        cm = scale_to_disclosed(A, 1.0) * np.where(bad[:, None], 2.0, 1.0)
        variants = [("no monitor", None)]
        if p_stop is not None:
            stop = bad[:, None] & (rng.random(A["sg"].shape) < p_stop)
            variants.append((f"monitor stop p={p_stop:.2f}", stop))
        for name, stop in variants:
            for vt, lam in (("urban", 15), ("highway", 40)):
                if f"{share}|{name}|{vt}|{lam}" in rep["badsites"]:
                    continue
                h = honest_run(f"bad|{share}|{name}", A, sw, qual, nv // 2, lam, vt, qtab, nl, cell_mult=cm, stop_mask=stop)
                rep["badsites"][f"{share}|{name}|{vt}|{lam}"] = h; save()
                log(f"bad sites {share:.2f} x2, {name}: {vt:8s} lam={lam:2d}: union/yr {h['punitive_union_per_year']:.1e}")
    attacks = ([("persistent", x) for x in (-0.05, -0.10, -0.30)] + [("days", -0.15, f) for f in (0.05, 0.1, 0.2, 0.3, 0.5)]
               + [("days", -0.30, 0.1)] + [("cap", 35.0)]
               + [("framing", n, -0.15) for n in (1, 2, 3, 5, 8)])
    for vt in E.PROFILES:
        for lam in lams:
            for att in attacks:
                if att[0] == "cap" and vt != "highway":
                    continue
                if f"{'|'.join(map(str, att))}|{vt}|{lam}" in rep["power"]:
                    continue
                r = attack_run(att, A, sw, qual, max(nv // 5, 200), lam, vt, qtab, nl)
                rep["power"][f"{'|'.join(map(str, att))}|{vt}|{lam}"] = r; save()
                log(f"class {'|'.join(map(str, att)):24s} {vt:8s} lam={lam:2d}: punitive/yr {r['punitive_per_year']:.2f} "
                    f"union/yr {r['punitive_union_per_year']:.2f}")
    rep["seconds_last_run"] = time.time() - t0
    name = f"decisions_L{nl}_{tag}.json"
    json.dump(rep, open(os.path.join(OUT, name), "w"), indent=1, default=float)
    log(f"elapsed {time.time() - t0:.0f} s -> {name}")


if __name__ == "__main__":
    main()
