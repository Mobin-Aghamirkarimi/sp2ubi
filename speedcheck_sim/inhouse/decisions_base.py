"""Base decision simulation: insurer decision rules on per-pass speed checks.

  * sigma_hat per (site, speed class) from the 30 reference passes, inflated x SIGMA_INFL, rounded up to N_LEVELS
    log-spaced levels; only the level index (2 bits) is disclosed with each pass (no site fingerprint);
  * Student-t tail mode uses the per-site within-site scale (no artificial site homogeneity);
  * R2 null quantiles tabulated with >= 1e7 Gaussian draws per M (cached in results/r2_null_quantiles.json);
  * site-bias drift: per-site random walk per epoch with annual sd DRIFT, reset by recalibration every RECAL epochs;
  * consequence ladder: a single R2 flag has no consequence; R2 in >= 2 of 3 consecutive epochs raises a flag
    ("punitive"); R1 notice only after 3 consecutive R1 epochs, rule edge
    S_RULE = -4 % (honest band edge -3 % less radius and shock allowance);
  * worst-site confinement test, one-at-a-time sensitivity of every honest constant (x0.5 / x2), minimal-profitable-
    attack curves (day-level fraction f).
Usage: python inhouse/decisions_base.py [--vehicles 20000] [--smoke]
"""
import argparse
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SPEEDS = (10.0, 15.0, 20.0, 30.0)
N_REF, EPOCHS, DAYS, M_MIN = 30, 12, 30, 5
TARGET_YEAR = 1e-3
ALPHA_R2 = TARGET_YEAR / EPOCHS / 5
ALPHA_R1 = 1e-3
C_WIN = 2.5
SIGMA_INFL, N_LEVELS = 1.15, 4
S_RULE = -0.04
HONEST = dict(S_LO=-0.03, S_HI=0.06, SHOCK=0.005, RAD=0.005, SLIP=0.003, LAT=0.0005, BURST_P=0.01, BURST_MU=-0.03,
              BURST_SD=0.02, SMUDGE=0.001, SIGMA0=0.005, DRIFT=0.0, RECAL=3)
VEHICLE_TYPES = {"urban": (0.6, 0.3, 0.1, 0.0), "suburban": (0.1, 0.3, 0.4, 0.2), "highway": (0.0, 0.1, 0.3, 0.6)}
M_GRID = (5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200)


# ------------------------------------------------------------------------------------------- R2 null quantile table
def _q_one(args):
    M, n_draws, seed = args
    rng = np.random.default_rng(seed)
    chunk = max(10_000_000 // M, 1000)
    vals, done = [], 0
    while done < n_draws:
        n = min(chunk, n_draws - done)
        z = rng.standard_normal((n, M))
        r = np.clip(z - z.mean(1, keepdims=True), -C_WIN, C_WIN)
        q = (r * r).sum(1)
        keep = np.quantile(q, 1 - 2000 / n) if n > 4000 else q.min()
        vals.append(q[q >= keep]); done += n
    tail = np.sort(np.concatenate(vals))[::-1]
    k = int(round(ALPHA_R2 * n_draws))
    return M, float(tail[k - 1]), k


def r2_quantiles(n_draws, workers):
    path = os.path.join(OUT, "r2_null_quantiles.json")
    if os.path.exists(path):
        d = json.load(open(path))
        if d["n_draws"] >= n_draws and d["alpha"] == ALPHA_R2 and d["c_win"] == C_WIN:
            return {int(k): v for k, v in d["q"].items()}
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_q_one, [(M, n_draws, 1000 + M) for M in M_GRID]))
    q = {M: v for M, v, _ in res}
    json.dump(dict(n_draws=n_draws, alpha=ALPHA_R2, c_win=C_WIN, q=q, exceedances_used={M: k for M, _, k in res}),
              open(path, "w"), indent=1)
    return q


def r2_threshold(M, qtab):
    Ms = np.array(sorted(qtab)); qs = np.array([qtab[m] for m in Ms])
    M = np.asarray(M, float)
    slope = (qs[-1] - qs[-2]) / (Ms[-1] - Ms[-2])
    return np.where(M > Ms[-1], qs[-1] + (M - Ms[-1]) * slope, np.interp(M, Ms, qs))


# ------------------------------------------------------------------------------------------- sites from the site-accuracy simulation
def load_sites(extra_tags=()):
    """Site table from the site-accuracy simulation plus optional extra cells (e.g. extra_tags=("extra",) reads
    site_accuracy_passes_extra.json)."""
    data = json.load(open(os.path.join(HERE, "results", "site_accuracy_passes_full.json")))
    for tag in extra_tags:
        data += json.load(open(os.path.join(HERE, "results", f"site_accuracy_passes_{tag}.json")))
    rows = []
    for item in data:
        c = item["cell"]
        if c["mean_acc"] != 0.0:
            continue
        cc = (c["K_db"], c["scattering"], c["specular_db"])
        recs = item["recs"]
        em = np.array([r["match"] for r in recs], float); ep = np.array([r["prof"] for r in recs], float)
        sid = np.array([r["site"] for r in recs])
        for st in np.unique(sid):
            sel = (sid == st) & np.isfinite(em) & np.isfinite(ep)
            a_m, a_p = em[sel][:100], ep[sel][:100]
            e = a_m if np.var(a_m[:N_REF]) <= np.var(a_p[:N_REF]) else a_p          # "select" on reference passes
            rows.append(dict(cc=cc, site=int(st), v=c["v0"], ref=e[:N_REF], ev=e[N_REF:], all=e))
    # site table indexed (site_id, speed); site_id = (cc, site)
    keys = sorted({(r["cc"], r["site"]) for r in rows}, key=str)
    kidx = {k: i for i, k in enumerate(keys)}
    nS, nV = len(keys), len(SPEEDS)
    n_eval = min(len(r["ev"]) for r in rows)
    A = dict(ev=np.empty((nS, nV, n_eval)), dh=np.empty((nS, nV)), dt=np.empty((nS, nV)), sraw=np.empty((nS, nV)),
             swithin=np.empty((nS, nV)), nu=np.empty((nS, nV)), nS=nS, n_eval=n_eval, keys=keys)
    cell_resid = {}
    for r in rows:
        i, j = kidx[(r["cc"], r["site"])], SPEEDS.index(r["v"])
        A["ev"][i, j] = r["ev"][:n_eval]; A["dh"][i, j] = r["ref"].mean(); A["dt"][i, j] = r["all"].mean()
        A["sraw"][i, j] = r["ref"].std(ddof=1); A["swithin"][i, j] = r["all"].std(ddof=1)
        cell_resid.setdefault((r["cc"], r["v"]), []).append(r["ref"] - r["ref"].mean())
    nu_cell = {k: max(stats.t.fit(np.concatenate(v), floc=0.0)[0], 2.5) for k, v in cell_resid.items()}
    for r in rows:
        A["nu"][kidx[(r["cc"], r["site"])], SPEEDS.index(r["v"])] = nu_cell[(r["cc"], r["v"])]
    s_infl = SIGMA_INFL * A["sraw"]
    levels = np.exp(np.linspace(np.log(s_infl.min()), np.log(s_infl.max()), N_LEVELS))
    lvl = np.searchsorted(levels, s_infl - 1e-15)                     # round UP to the next level
    A["levels"], A["lvl"], A["sg"] = levels, lvl, levels[np.minimum(lvl, N_LEVELS - 1)]
    return A


def subset(A, idx):
    B = dict(A)
    for k in ("ev", "dh", "dt", "sraw", "swithin", "nu", "lvl", "sg"):
        B[k] = A[k][idx]
    B["nS"] = len(idx)
    return B


# ------------------------------------------------------------------------------------------- simulation
def simulate(rng, A, n_veh, lam, vtype, mode, attack=None, honest=HONEST):
    probs = np.array(VEHICLE_TYPES[vtype])
    nS = A["nS"]
    n_sites = rng.integers(2, min(10, nS) + 1, n_veh)
    vs = np.argsort(rng.random((n_veh, nS)), axis=1)[:, :min(10, nS)]
    s = rng.uniform(honest["S_LO"], honest["S_HI"], n_veh)
    rng.random((n_veh, min(10, nS)))                          # one draw per vehicle and site; keeps the random stream
    M = rng.poisson(lam, (n_veh, EPOCHS))
    key = np.repeat(np.arange(n_veh * EPOCHS), M.ravel())
    veh = key // EPOCHS; ep = key % EPOCHS
    P = len(key)
    si = (rng.random(P) * n_sites[veh]).astype(int)
    site = vs[veh, si]
    vi = rng.choice(len(SPEEDS), P, p=probs)
    if mode == "empirical":
        e = A["ev"][site, vi, rng.integers(0, A["n_eval"], P)]
    else:
        nu = A["nu"][site, vi]
        e = A["dt"][site, vi] + A["swithin"][site, vi] * np.sqrt((nu - 2) / nu) * rng.standard_t(nu)
    if honest["DRIFT"] > 0:                                    # per (vehicle, site) random walk, reset every RECAL epochs
        steps = rng.normal(0, honest["DRIFT"] / np.sqrt(EPOCHS), (n_veh, min(10, nS), EPOCHS))
        seg = np.arange(EPOCHS) // honest["RECAL"]
        walk = np.zeros_like(steps)
        for k in range(EPOCHS):
            walk[:, :, k] = steps[:, :, k] + (walk[:, :, k - 1] if k > 0 and seg[k] == seg[k - 1] else 0.0)
        e = e + walk[veh, si, ep]
    shock = rng.normal(0, honest["SHOCK"], n_veh * EPOCHS)[key]
    v_arr = np.array(SPEEDS)[vi]
    b_vss = (s[veh] + shock - honest["RAD"] * (v_arr / 30.0) ** 2 + rng.normal(0, honest["SLIP"], P)
             + rng.normal(0, honest["LAT"], P)
             + np.where(rng.random(P) < honest["BURST_P"], rng.normal(honest["BURST_MU"], honest["BURST_SD"], P), 0.0))
    if attack is not None:
        kind, a, f = attack
        if kind == "persistent":
            hit = np.ones(P, bool)
        elif kind == "days":
            bad = rng.random((n_veh * EPOCHS, DAYS)) < f
            hit = bad[key, rng.integers(0, DAYS, P)]
        b_vss = np.where(hit, (1 + b_vss) * (1 + a) - 1, b_vss)
    b = b_vss + e + rng.normal(0, honest["SMUDGE"], P) - A["dh"][site, vi]
    se_i = np.sqrt(A["sg"][site, vi] ** 2 + honest["SIGMA0"] ** 2)
    w = 1.0 / se_i ** 2
    nE = n_veh * EPOCHS
    W = np.bincount(key, w, nE)
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.bincount(key, w * b, nE) / W
        bw = m0[key] + np.clip(b - m0[key], -C_WIN * se_i, C_WIN * se_i)
        m = np.bincount(key, w * bw, nE) / W
        SE = 1.0 / np.sqrt(W)
        r = np.clip((b - m[key]) / se_i, -C_WIN, C_WIN)
        Q = np.bincount(key, r * r, nE)
    Mf = M.ravel().astype(float)
    low = Mf < M_MIN
    m[low] = np.nan; SE[low] = np.nan; Q[low] = np.nan
    return dict(M=Mf.reshape(n_veh, EPOCHS), m=m.reshape(n_veh, EPOCHS), SE=SE.reshape(n_veh, EPOCHS),
                Q=Q.reshape(n_veh, EPOCHS), s=s)


def runs(flags, k, n):
    """True per vehicle if any window of n consecutive epochs holds >= k flags."""
    out = np.zeros(flags.shape[0], bool)
    for j in range(flags.shape[1] - n + 1):
        out |= flags[:, j:j + n].sum(1) >= k
    return out


def decide(sim, qtab):
    M, m, SE, Q = sim["M"], sim["m"], sim["SE"], sim["Q"]
    ok = ~np.isnan(m)
    thr = np.where(M >= M_MIN, r2_threshold(np.maximum(M, M_MIN), qtab), np.inf)
    r2 = ok & (Q > thr)
    r1 = ok & (m < S_RULE - stats.norm.ppf(1 - ALPHA_R1) * SE)
    return dict(ok=ok, r2=r2, r1=r1, punitive=runs(r2, 2, 3), notice=runs(r1, 3, 3))


def cp_upper(k, n):
    return float(stats.beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0


def honest_summary(d, n_veh):
    k = int(d["punitive"].sum())
    return dict(r2_epoch=float(d["r2"][d["ok"]].mean()), punitive_per_year=k / n_veh, punitive_ub95=cp_upper(k, n_veh),
                notice_per_year=float(d["notice"].mean()))


def seed(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=20_000)
    ap.add_argument("--qdraws", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra cell tags of the site-accuracy simulation, e.g. extra")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    nv = 500 if a.smoke else a.vehicles
    t0 = time.time()
    qtab = r2_quantiles(200_000 if a.smoke else a.qdraws, a.workers)
    A = load_sites(tuple(a.extra))
    rep = dict(params=dict(HONEST=HONEST, C_WIN=C_WIN, SIGMA_INFL=SIGMA_INFL, N_LEVELS=N_LEVELS, levels=A["levels"].tolist(),
                           S_RULE=S_RULE, ALPHA_R2=ALPHA_R2, qdraws=a.qdraws), honest={}, worst={}, drift={}, sens={}, power={})
    log = lambda *x: print(*x, flush=True)
    # 1. honest, all strata
    for mode in ("empirical", "student-t"):
        for vt in VEHICLE_TYPES:
            for lam in (5, 15, 40):
                h = honest_summary(decide(simulate(seed("h", mode, vt, lam), A, nv, lam, vt, mode), qtab), nv)
                rep["honest"][f"{mode}|{vt}|{lam}"] = h
                log(f"honest {mode:9s} {vt:8s} lam={lam:2d}: R2/epoch {h['r2_epoch']:.1e} punitive/yr {h['punitive_per_year']:.1e} "
                    f"(UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
    # 2. worst-site confinement (ratio of within-site sd to the disclosed level, weighted by the speed profile)
    for vt in ("urban", "highway"):
        score = np.sqrt(((A["swithin"] / A["sg"]) ** 2 * np.array(VEHICLE_TYPES[vt])).sum(1))
        for nworst in (10, 30):
            B = subset(A, np.argsort(-score)[:nworst])
            for lam in (15, 40):
                for mode in ("empirical", "student-t"):
                    h = honest_summary(decide(simulate(seed("w", vt, nworst, lam, mode), B, nv, lam, vt, mode), qtab), nv)
                    rep["worst"][f"{vt}|worst{nworst}|{lam}|{mode}"] = h
                    log(f"worst{nworst:2d} {vt:8s} lam={lam:2d} {mode:9s}: R2/epoch {h['r2_epoch']:.1e} punitive/yr "
                        f"{h['punitive_per_year']:.1e} (UB {h['punitive_ub95']:.1e})")
    # 3. drift of the site bias with recalibration every RECAL epochs
    for drift in (0.003, 0.006, 0.010):
        for recal in (3, 12):
            hon = dict(HONEST, DRIFT=drift, RECAL=recal)
            for vt in ("urban", "highway"):
                h = honest_summary(decide(simulate(seed("d", drift, recal, vt), A, nv, 15, vt, "empirical", honest=hon), qtab), nv)
                rep["drift"][f"{drift}|{recal}|{vt}"] = h
                log(f"drift {drift:.3f}/yr recal every {recal:2d} epochs {vt:8s}: R2/epoch {h['r2_epoch']:.1e} punitive/yr "
                    f"{h['punitive_per_year']:.1e} (UB {h['punitive_ub95']:.1e})")
    # 4. one-at-a-time sensitivity (x0.5, x2) of every honest constant, plus a wider honest band
    changes = [(k, f) for k in ("SHOCK", "RAD", "SLIP", "BURST_P", "BURST_MU", "BURST_SD", "SIGMA0") for f in (0.5, 2.0)]
    for k, f in changes + [("BAND", (-0.05, 0.08))]:
        hon = dict(HONEST, S_LO=f[0], S_HI=f[1]) if k == "BAND" else dict(HONEST, **{k: HONEST[k] * f})
        for vt, lam in (("urban", 15), ("highway", 40)):
            h = honest_summary(decide(simulate(seed("s", k, f, vt), A, nv // 2, lam, vt, "empirical", honest=hon), qtab), nv // 2)
            rep["sens"][f"{k}x{f}|{vt}|{lam}"] = h
            log(f"sens {k:8s} x{f}: {vt:8s} lam={lam}: R2/epoch {h['r2_epoch']:.1e} punitive/yr {h['punitive_per_year']:.1e} "
                f"(UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
    # 5. power and minimal-profitable-attack curves
    attacks = ([("persistent", x, 1.0) for x in (-0.05, -0.10)] + [("days", -0.15, f) for f in (0.05, 0.1, 0.2, 0.3, 0.5)]
               + [("days", -0.30, 0.1)])
    for vt in VEHICLE_TYPES:
        for lam in (5, 15, 40):
            for att in attacks:
                d = decide(simulate(seed("p", att, vt, lam), A, max(nv // 10, 200), lam, vt, "empirical", attack=att), qtab)
                res = dict(punitive_per_year=float(d["punitive"].mean()), r2_epoch=float(d["r2"][d["ok"]].mean()),
                           notice_per_year=float(d["notice"].mean()))
                rep["power"][f"{att[0]}|{att[1]}|{att[2]}|{vt}|{lam}"] = res
                log(f"attack {att[0]:10s} {att[1]:+.2f} {att[2]:<5}: {vt:8s} lam={lam:2d}: punitive/yr {res['punitive_per_year']:.2f} "
                    f"R2/epoch {res['r2_epoch']:.2f}")
    rep["seconds"] = time.time() - t0
    json.dump(rep, open(os.path.join(OUT, f"decisions_base_report_{'smoke' if a.smoke else 'full'}{a.out_tag}.json"), "w"), indent=1, default=float)
    log(f"elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
