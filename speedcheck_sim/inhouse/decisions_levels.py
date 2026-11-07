"""Decision simulation, level variant: insurer decision rules on per-pass speed checks with disclosed sigma levels;
the R2 null-quantile table, the rules and their constants are those of decisions_base.py.

  * site population: each site's Rice factor K is drawn from the 3GPP TR 38.901 UMi LOS distribution N(9, 5^2) dB;
    the site survey excludes sites with K < 5 dB; a drawn K is mapped down to the next simulated grid value, which is
    conservative for accuracy; scattering and specular facade 50/50;
  * speed classes from the site-accuracy simulation data; the highway profile uses it;
  * out-of-window results (|b| > B_WINDOW, the insurer's decryption returns None) enter as b = -B_WINDOW in the
    epoch mean and as a saturated residual (C_WIN^2) in R2;
  * margin analysis: actual noise larger than the disclosed sigma level, site-common bias drift, site-epoch noise
    non-stationarity, static vehicle x site offset, and a combined case;
  * threat classes evaluated for power: persistent scaling, day-level scaling and a speed cap (VSS reports
    min(v, 35 m/s), i.e. only the 38 m/s class is affected).
Usage: python inhouse/decisions_levels.py --tag extra [--extra ...] [--vehicles 20000] [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import stats

import decisions_base as DB
from decisions_base import (ALPHA_R1, C_WIN, EPOCHS, DAYS, M_MIN, N_LEVELS, N_REF, S_RULE, SIGMA_INFL, cp_upper,
                            r2_threshold, runs, seed)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
B_WINDOW = 0.25                                       # = speedcheck.B_WINDOW
K_MEAN, K_SD, K_MIN_SURVEY = 9.0, 5.0, 5.0             # TR 38.901 UMi LOS K-factor (dB) and survey exclusion
PROFILES = {"urban": {10: 0.6, 15: 0.3, 20: 0.1}, "suburban": {10: 0.1, 15: 0.3, 20: 0.4, 30: 0.2},
            "highway": {15: 0.1, 20: 0.3, 30: 0.4, 38: 0.2}}
HONEST = dict(DB.HONEST, SIG_UNDER=1.0, SBIAS=0.0, SMULT=0.0, VOFF=0.0)


# ------------------------------------------------------------------------------------------- sites from the site-accuracy simulation
def k_weights(k_grid, mapping="floor"):
    """Probability of each simulated K value for a surveyed site (K ~ N(9, 5^2) dB, K >= 5 dB)."""
    g = np.array(sorted(k_grid))
    lo = stats.norm.cdf((K_MIN_SURVEY - K_MEAN) / K_SD)
    if mapping == "floor":                                   # [g_i, g_{i+1}) -> g_i ; [5, g_0) -> g_0
        edges = np.concatenate([[K_MIN_SURVEY], g[1:], [np.inf]])
    else:                                                    # nearest grid value
        edges = np.concatenate([[K_MIN_SURVEY], (g[1:] + g[:-1]) / 2, [np.inf]])
    p = np.diff(stats.norm.cdf((edges - K_MEAN) / K_SD)) / (1 - lo)
    return dict(zip(g.tolist(), p.tolist()))


def load_sites(tags):
    data = []
    for tag in tags:
        data += json.load(open(os.path.join(HERE, "results", f"site_accuracy_passes_{tag}.json")))
    data = [d for d in data if d["cell"].get("mean_acc", 0.0) == 0.0 and d["cell"].get("D", 100.0) == 100.0]
    speeds = tuple(sorted({d["cell"]["v0"] for d in data}))
    rows = []
    for item in data:
        c = item["cell"]
        cc = (c["K_db"], c["scattering"], c["specular_db"])
        recs = item["recs"]
        em = np.array([r["match"] for r in recs], float); ep = np.array([r["prof"] for r in recs], float)
        sid = np.array([r["site"] for r in recs])
        for st in np.unique(sid):
            sel = (sid == st) & np.isfinite(em) & np.isfinite(ep)
            a_m, a_p = em[sel][:100], ep[sel][:100]
            e = a_m if np.var(a_m[:N_REF]) <= np.var(a_p[:N_REF]) else a_p          # "select" on reference passes
            rows.append(dict(cc=cc, site=int(st), v=c["v0"], ref=e[:N_REF], ev=e[N_REF:], all=e))
    keys = sorted({(r["cc"], r["site"]) for r in rows}, key=str)
    kidx = {k: i for i, k in enumerate(keys)}
    nS, nV = len(keys), len(speeds)
    n_eval = min(len(r["ev"]) for r in rows)
    A = dict(ev=np.full((nS, nV, n_eval), np.nan), dh=np.full((nS, nV), np.nan), dt=np.full((nS, nV), np.nan),
             sraw=np.full((nS, nV), np.nan), swithin=np.full((nS, nV), np.nan), nu=np.full((nS, nV), np.nan),
             nS=nS, n_eval=n_eval, keys=keys, speeds=speeds)
    cell_resid = {}
    for r in rows:
        i, j = kidx[(r["cc"], r["site"])], speeds.index(r["v"])
        A["ev"][i, j] = r["ev"][:n_eval]; A["dh"][i, j] = r["ref"].mean(); A["dt"][i, j] = r["all"].mean()
        A["sraw"][i, j] = r["ref"].std(ddof=1); A["swithin"][i, j] = r["all"].std(ddof=1)
        cell_resid.setdefault((r["cc"], r["v"]), []).append(r["ref"] - r["ref"].mean())
    assert np.isfinite(A["dh"]).all(), "every (site, speed) needs E1 passes"
    nu_cell = {k: max(stats.t.fit(np.concatenate(v), floc=0.0)[0], 2.5) for k, v in cell_resid.items()}
    for r in rows:
        A["nu"][kidx[(r["cc"], r["site"])], speeds.index(r["v"])] = nu_cell[(r["cc"], r["v"])]
    s_infl = SIGMA_INFL * A["sraw"]
    levels = np.exp(np.linspace(np.log(s_infl.min()), np.log(s_infl.max()), N_LEVELS))
    lvl = np.searchsorted(levels, s_infl - 1e-15)
    A["levels"], A["lvl"], A["sg"] = levels, lvl, levels[np.minimum(lvl, N_LEVELS - 1)]
    A["K"] = np.array([k[0][0] for k in keys])
    return A


def site_weights(A, mapping="floor"):
    kw = k_weights(np.unique(A["K"]), mapping)
    w = np.array([kw[k] / np.sum(A["K"] == k) for k in A["K"]])
    return w / w.sum()


def profile(A, vtype):
    """Speed-class probabilities on the simulated grid; mass of a missing class goes to the nearest simulated one."""
    sp = np.array(A["speeds"]); p = np.zeros(len(sp))
    for v, q in PROFILES[vtype].items():
        p[np.argmin(np.abs(sp - v))] += q
    return p


def subset(A, idx):
    B = dict(A)
    for k in ("ev", "dh", "dt", "sraw", "swithin", "nu", "lvl", "sg", "K"):
        B[k] = A[k][idx]
    B["nS"] = len(idx); B["keys"] = [A["keys"][i] for i in idx]
    return B


# ------------------------------------------------------------------------------------------- simulation
def simulate(rng, A, sw, n_veh, lam, vtype, mode, attack=None, honest=HONEST, rng_att=None, qual=None):
    """Returns per-pass arrays for the honest vehicle and (if attack) for the same vehicle under the manipulation;
    every honest draw is identical in both (common random numbers). qual: optional boolean (site, speed) mask of
    qualified cells; passes in unqualified cells produce no entry (site qualification, analysis only)."""
    probs = profile(A, vtype)
    nS, nVmax = A["nS"], min(10, A["nS"])
    n_sites = rng.integers(2, nVmax + 1, n_veh)
    gumbel = np.log(sw)[None, :] - np.log(-np.log(rng.random((n_veh, nS))))   # weighted sampling w/o replacement
    vs = np.argsort(-gumbel, axis=1)[:, :nVmax]
    s = rng.uniform(honest["S_LO"], honest["S_HI"], n_veh)
    M = rng.poisson(lam, (n_veh, EPOCHS))
    key = np.repeat(np.arange(n_veh * EPOCHS), M.ravel())
    veh = key // EPOCHS; ep = key % EPOCHS
    P = len(key)
    si = (rng.random(P) * n_sites[veh]).astype(int)
    site = vs[veh, si]
    vi = rng.choice(len(A["speeds"]), P, p=probs)
    if mode == "empirical":
        e = A["ev"][site, vi, rng.integers(0, A["n_eval"], P)]
    else:
        nu = A["nu"][site, vi]
        e = A["dt"][site, vi] + A["swithin"][site, vi] * np.sqrt((nu - 2) / nu) * rng.standard_t(nu)
    # margin cases: actual noise k x disclosed (same disclosed level), site-epoch noise multiplier, site-common bias
    # random walk reset at recalibration, static vehicle x site offset
    dev = (e - A["dt"][site, vi]) * honest["SIG_UNDER"]
    if honest["SMULT"] > 0:
        dev = dev * np.exp(rng.normal(0, honest["SMULT"], (nS, EPOCHS)))[site, ep]
    e = A["dt"][site, vi] + dev
    if honest["SBIAS"] > 0:
        st = rng.normal(0, honest["SBIAS"], (nS, EPOCHS)); walk = np.zeros_like(st)
        for k in range(EPOCHS):
            walk[:, k] = st[:, k] + (walk[:, k - 1] if k % honest["RECAL"] != 0 else 0.0)
        e = e + walk[site, ep]
    if honest["VOFF"] > 0:
        e = e + rng.normal(0, honest["VOFF"], (n_veh, nVmax))[veh, si]
    if honest["DRIFT"] > 0:
        steps = rng.normal(0, honest["DRIFT"] / np.sqrt(EPOCHS), (n_veh, nVmax, EPOCHS))
        seg = np.arange(EPOCHS) // honest["RECAL"]
        walk = np.zeros_like(steps)
        for k in range(EPOCHS):
            walk[:, :, k] = steps[:, :, k] + (walk[:, :, k - 1] if k > 0 and seg[k] == seg[k - 1] else 0.0)
        e = e + walk[veh, si, ep]
    shock = rng.normal(0, honest["SHOCK"], n_veh * EPOCHS)[key]
    v_arr = np.array(A["speeds"])[vi]
    b_vss = (s[veh] + shock - honest["RAD"] * (v_arr / 30.0) ** 2 + rng.normal(0, honest["SLIP"], P)
             + rng.normal(0, honest["LAT"], P)
             + np.where(rng.random(P) < honest["BURST_P"], rng.normal(honest["BURST_MU"], honest["BURST_SD"], P), 0.0))
    noise = e + rng.normal(0, honest["SMUDGE"], P) - A["dh"][site, vi]
    se_i = np.sqrt(A["sg"][site, vi] ** 2 + honest["SIGMA0"] ** 2)
    out = dict(key=key, M=M, s=s, se_i=se_i, n_veh=n_veh, b_hon=b_vss + noise,
               lvl=A["lvl"][site, vi])
    if attack is not None:
        ra = rng_att or np.random.default_rng(12345)
        kind, a, f = attack
        if kind == "persistent":
            hit = np.ones(P, bool)
        elif kind == "days":
            bad = ra.random((n_veh * EPOCHS, DAYS)) < f
            hit = bad[key, ra.integers(0, DAYS, P)]
        elif kind == "cap":                                   # VSS reports min(v, a): only faster passes are scaled
            ratio = np.minimum(1.0, a / v_arr)
            out["b_att"] = (1 + b_vss) * ratio - 1 + noise
            return _qualify(out, qual, site, vi) if qual is not None else out
        out["b_att"] = np.where(hit, (1 + b_vss) * (1 + a) - 1, b_vss) + noise
    return _qualify(out, qual, site, vi) if qual is not None else out


def _qualify(out, qual, site, vi):
    keep = qual[site, vi]
    for k in ("key", "se_i", "b_hon", "b_att", "lvl"):
        if k in out:
            out[k] = out[k][keep]
    out["M"] = np.bincount(out["key"], minlength=out["M"].size).reshape(out["M"].shape)
    return out


def decide(sim, b, qtab, r2_scale=None):
    key, se_i, n_veh = sim["key"], sim["se_i"], sim["n_veh"]
    nE = n_veh * EPOCHS
    # r2_scale (analysis option): per-pass factor >= 1 applied to sigma_eff in the R2 dispersion statistic only
    r2s = np.ones_like(se_i) if r2_scale is None else r2_scale
    oow = np.abs(b) > B_WINDOW                                 # decryption returns None
    b = np.where(oow, -B_WINDOW, b)
    w = 1.0 / se_i ** 2

    def stat(k, nK):
        W = np.bincount(k, w, nK)
        with np.errstate(invalid="ignore", divide="ignore"):
            m0 = np.bincount(k, w * b, nK) / W
            bw = m0[k] + np.clip(b - m0[k], -C_WIN * se_i, C_WIN * se_i)
            m = np.bincount(k, w * bw, nK) / W
            r = np.where(oow, C_WIN, np.clip((b - m[k]) / (se_i * r2s), -C_WIN, C_WIN))
            Q = np.bincount(k, r * r, nK)
        return m, 1.0 / np.sqrt(W), Q

    m, SE, Q = stat(key, nE)
    Mf = sim["M"].ravel().astype(float)
    ok = Mf >= M_MIN
    thr = np.where(ok, r2_threshold(np.maximum(Mf, M_MIN), qtab), np.inf)
    r2 = ok & (Q > thr)
    r1 = ok & (m < S_RULE - stats.norm.ppf(1 - ALPHA_R1) * SE)
    return dict(ok=ok.reshape(n_veh, EPOCHS), r2=r2.reshape(n_veh, EPOCHS), r1=r1.reshape(n_veh, EPOCHS),
                punitive=runs(r2.reshape(n_veh, EPOCHS), 2, 3), notice=runs(r1.reshape(n_veh, EPOCHS), 3, 3),
                oow_share=float(oow.mean()))


def honest_summary(d):
    n = len(d["punitive"]); k = int(d["punitive"].sum())
    return dict(r2_epoch=float(d["r2"][d["ok"]].mean()), punitive_per_year=k / n, punitive_ub95=cp_upper(k, n),
                notice_per_year=float(d["notice"].mean()), oow_share=d["oow_share"],
                low_m_epoch_share=float(1 - d["ok"].mean()))


def run_honest(tag, A, sw, nv, lam, vt, mode, qtab, honest=HONEST):
    sim = simulate(seed("h4", tag, vt, lam, mode), A, sw, nv, lam, vt, mode, honest=honest)
    return honest_summary(decide(sim, sim["b_hon"], qtab))


def run_attack(att, A, sw, nv, lam, vt, qtab):
    sim = simulate(seed("p4", att, vt, lam), A, sw, nv, lam, vt, "empirical", attack=att, rng_att=seed("a4", att, vt, lam))
    da = decide(sim, sim["b_att"], qtab)
    P = float(da["punitive"].mean())
    return dict(punitive_per_year=P, punitive_ub95=cp_upper(int(da["punitive"].sum()), nv),
                r2_epoch=float(da["r2"][da["ok"]].mean()), notice_per_year=float(da["notice"].mean()),
                oow_share=da["oow_share"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["extra"], help="site-accuracy pass files (site_accuracy_passes_<tag>.json)")
    ap.add_argument("--vehicles", type=int, default=20_000)
    ap.add_argument("--qdraws", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    nv = 400 if a.smoke else a.vehicles
    t0 = time.time()
    qtab = DB.r2_quantiles(a.qdraws, a.workers)              # cached table of decisions_base.py (same alpha and C_WIN)
    A = load_sites(a.tag)
    sw = site_weights(A, "floor")
    kw = k_weights(np.unique(A["K"]), "floor")
    log = lambda *x: print(*x, flush=True)
    log(f"sites {A['nS']}, speeds {A['speeds']}, K weights (floor) {kw}, levels {np.round(100 * A['levels'], 2)} %")
    rep = dict(params=dict(HONEST=HONEST, C_WIN=C_WIN, SIGMA_INFL=SIGMA_INFL, N_LEVELS=N_LEVELS, B_WINDOW=B_WINDOW,
                           levels=A["levels"].tolist(), S_RULE=S_RULE, ALPHA_R2=DB.ALPHA_R2, K_weights=kw,
                           speeds=A["speeds"], tags=a.tag, profiles=PROFILES),
               honest={}, worst={}, margin={}, sens={}, power={})
    lams = (5, 15, 40)
    # 1. honest, all strata (K population, floor mapping) and the nearest-K mapping as a check
    for mapping in ("floor", "nearest"):
        swm = site_weights(A, mapping)
        for mode in (("empirical", "student-t") if mapping == "floor" else ("empirical",)):
            for vt in PROFILES:
                for lam in lams:
                    h = run_honest(f"{mapping}", A, swm, nv, lam, vt, mode, qtab)
                    rep["honest"][f"{mapping}|{mode}|{vt}|{lam}"] = h
                    log(f"honest {mapping:7s} {mode:9s} {vt:8s} lam={lam:2d}: R2/ep {h['r2_epoch']:.1e} punitive/yr "
                        f"{h['punitive_per_year']:.1e} (UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e} "
                        f"oow {h['oow_share']:.1e} lowM {h['low_m_epoch_share']:.2f}")
    # 2. worst-site confinement: each vehicle only at the sites with the largest within-site sd / disclosed level
    for vt in ("urban", "highway"):
        score = np.sqrt(((A["swithin"] / A["sg"]) ** 2 * profile(A, vt)).sum(1))
        for nworst in (10, 30):
            idx = np.argsort(-score)[:nworst]
            B = subset(A, idx); swb = np.ones(nworst) / nworst
            for lam in (15, 40):
                h = run_honest(f"worst{nworst}", B, swb, nv, lam, vt, "empirical", qtab)
                rep["worst"][f"{vt}|worst{nworst}|{lam}"] = h
                log(f"worst{nworst:2d} {vt:8s} lam={lam:2d}: R2/ep {h['r2_epoch']:.1e} punitive/yr {h['punitive_per_year']:.1e} "
                    f"(UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
    # 3. margin analysis (honest population; where does the false-punitive target 1e-3 per year break?)
    margins = ([("SIG_UNDER", x) for x in (1.2, 1.5, 2.0)] + [("SBIAS", x) for x in (0.003, 0.006, 0.010)]
               + [("SMULT", x) for x in (0.2, 0.35, 0.5)] + [("VOFF", x) for x in (0.002, 0.004, 0.006)]
               + [("DRIFT", x) for x in (0.006, 0.010)] + [("COMBINED", None)])
    for k, x in margins:
        hon = (dict(HONEST, SIG_UNDER=1.2, SBIAS=0.003, SMULT=0.2, VOFF=0.002) if k == "COMBINED"
               else dict(HONEST, **{k: x}))
        for vt, lam in (("urban", 15), ("highway", 40), ("suburban", 5)):
            h = run_honest(f"m|{k}|{x}", A, sw, nv // 2, lam, vt, "empirical", qtab, honest=hon)
            rep["margin"][f"{k}|{x}|{vt}|{lam}"] = h
            log(f"margin {k:9s} {str(x):6s} {vt:8s} lam={lam:2d}: R2/ep {h['r2_epoch']:.1e} punitive/yr "
                f"{h['punitive_per_year']:.1e} (UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
    # 4. one-at-a-time sensitivity of the honest constants (x0.5 / x2) and a wider honest band
    changes = [(k, f) for k in ("SHOCK", "RAD", "SLIP", "BURST_P", "BURST_MU", "BURST_SD", "SIGMA0") for f in (0.5, 2.0)]
    for k, f in changes + [("BAND", (-0.05, 0.08))]:
        hon = dict(HONEST, S_LO=f[0], S_HI=f[1]) if k == "BAND" else dict(HONEST, **{k: HONEST[k] * f})
        for vt, lam in (("urban", 15), ("highway", 40)):
            h = run_honest(f"s|{k}|{f}", A, sw, nv // 2, lam, vt, "empirical", qtab, honest=hon)
            rep["sens"][f"{k}x{f}|{vt}|{lam}"] = h
            log(f"sens {k:8s} x{f}: {vt:8s} lam={lam}: R2/ep {h['r2_epoch']:.1e} punitive/yr {h['punitive_per_year']:.1e} "
                f"(UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
    # 5. power per threat class
    attacks = ([("persistent", x, 1.0) for x in (-0.05, -0.10, -0.30)] + [("days", -0.15, f) for f in (0.05, 0.1, 0.2, 0.3, 0.5)]
               + [("days", -0.30, 0.1)])
    if max(A["speeds"]) > 35:
        attacks.append(("cap", 35.0, 1.0))
    for vt in PROFILES:
        for lam in lams:
            for att in attacks:
                if att[0] == "cap" and vt != "highway":
                    continue
                r = run_attack(att, A, sw, max(nv // 5, 200), lam, vt, qtab)
                rep["power"][f"{att[0]}|{att[1]}|{att[2]}|{vt}|{lam}"] = r
                log(f"class {att[0]:10s} {att[1]:+6.2f} {att[2]:<5} {vt:8s} lam={lam:2d}: punitive/yr "
                    f"{r['punitive_per_year']:.2f} notice/yr {r['notice_per_year']:.2f}")
    rep["seconds"] = time.time() - t0
    name = f"decisions_levels_report_{'smoke' if a.smoke else 'full'}_{'_'.join(a.tag)}{a.out_tag}.json"
    json.dump(rep, open(os.path.join(OUT, name), "w"), indent=1, default=float)
    log(f"elapsed {time.time() - t0:.0f} s -> {name}")


if __name__ == "__main__":
    main()
