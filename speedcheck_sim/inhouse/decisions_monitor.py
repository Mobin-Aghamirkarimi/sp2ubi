"""Decision simulation, monitor variant: insurer decision rules in the final configuration (functions of
decisions_levels.py are reused).

Configuration (fixed choices):
  * sites from the site-accuracy simulation under the surveyed K population (TR 38.901 UMi LOS N(9, 5^2) dB, floor
    mapping); a (site, speed class) produces entries only if its reference-pass sigma_hat <= SIGMA_MAX (site
    qualification);
  * disclosed sigma levels: N_LEVELS (4 or 8, --levels) log-spaced over the qualified cells, rounded up;
  * crossings: non-overlapping bursts of BURST_S; the RSU needs the window [-0.05 D, 1.05 D] (ARC_M = 110 m at
    D = 100 m) inside one burst, so a crossing yields an entry with probability 1 - ARC_M / (v BURST_S);
  * passes per epoch: Poisson(lambda) and, as a heterogeneity case, negative binomial with CV of the per-vehicle rate
    LAM_CV = 1 (gamma-mixed Poisson);
  * R2 with the insurer's per-level population calibration of the residual scale (1.4826 MAD, per epoch);
  * out-of-window results enter R2 as a saturated residual.
Margins: actual noise 1.0 / 1.2 / 1.5 x the disclosed sigma_eff (per-cell multiplier), site-common bias steps,
site-epoch noise multiplier, vehicle x site offset, per-vehicle drift, combined; "bad sites" (a share of sites at 2 x
their disclosed sigma) without and with the RSU monitor (detected sites are stopped).
Threat classes: persistent, day-level, speed cap at 35 m/s, and live-relay framing of an honest vehicle (n relayed
passes per epoch with a -15 % result attributed to the victim).
Resumable: every finished run is saved at once to results/decisions_monitor_L<n>_<tag>.partial.json; rerun to resume.
Usage: python inhouse/decisions_monitor.py --levels 4|8 [--vehicles 20000] [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np

import decisions_base as DB
import decisions_levels as E
from levels_qualification import requalify, scale_to_disclosed
from residual_scale import population_scale
from decisions_base import DAYS, EPOCHS, cp_upper, seed


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SIGMA_MAX = 0.06
BURST_S, ARC_M = 40.0, 110.0
HONEST = dict(E.HONEST, LAM_CV=0.0)


def keep_prob(v):
    return np.clip(1.0 - ARC_M / (np.asarray(v, float) * BURST_S), 0.0, 1.0)


def simulate(rng, A, sw, qual, n_veh, lam, vtype, honest=HONEST, attack=None, rng_att=None, cell_mult=None,
             stop_mask=None):
    """Per-pass arrays (common random numbers for the honest and the manipulated vehicle). cell_mult: per-(site,
    speed) multiplier of the honest error (margin cases); stop_mask: (site, speed) cells stopped by the monitor."""
    probs = E.profile(A, vtype)
    nS, nVmax = A["nS"], min(10, A["nS"])
    n_sites = rng.integers(2, nVmax + 1, n_veh)
    gumbel = np.log(sw)[None, :] - np.log(-np.log(rng.random((n_veh, nS))))
    vs = np.argsort(-gumbel, axis=1)[:, :nVmax]
    s = rng.uniform(honest["S_LO"], honest["S_HI"], n_veh)
    lam_v = (rng.gamma(1 / honest["LAM_CV"] ** 2, lam * honest["LAM_CV"] ** 2, n_veh) if honest["LAM_CV"] > 0
             else np.full(n_veh, float(lam)))
    M0 = rng.poisson(np.repeat(lam_v[:, None], EPOCHS, axis=1))
    key = np.repeat(np.arange(n_veh * EPOCHS), M0.ravel())
    veh = key // EPOCHS; ep = key % EPOCHS
    P = len(key)
    si = (rng.random(P) * n_sites[veh]).astype(int)
    site = vs[veh, si]
    vi = rng.choice(len(A["speeds"]), P, p=probs)
    v_arr = np.array(A["speeds"])[vi]
    e = A["ev"][site, vi, rng.integers(0, A["n_eval"], P)]
    mult = np.ones(P) if cell_mult is None else cell_mult[site, vi]
    dev = (e - A["dt"][site, vi]) * honest["SIG_UNDER"] * mult
    mon_raise = mon_stop = None
    if honest["SMULT"] > 0:
        mm = np.exp(rng.normal(0, honest["SMULT"], (nS, EPOCHS)))
        dev = dev * mm[site, ep]
        if honest.get("MONITOR", False):
            # RSU self-monitor: the running residual level tracks the noise multiplier with a
            # measurement error (lognormal sd 0.1); > 1.25 raises the disclosed level one step, > 1.6 stops the cell
            seen = mm * np.exp(rng.normal(0, 0.1, mm.shape))
            mon_raise, mon_stop = (seen > 1.25)[site, ep], (seen > 1.6)[site, ep]
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
        seg = np.arange(EPOCHS) // honest["RECAL"]; walk = np.zeros_like(steps)
        for k in range(EPOCHS):
            walk[:, :, k] = steps[:, :, k] + (walk[:, :, k - 1] if k > 0 and seg[k] == seg[k - 1] else 0.0)
        e = e + walk[veh, si, ep]
    shock = rng.normal(0, honest["SHOCK"], n_veh * EPOCHS)[key]
    b_vss = (s[veh] + shock - honest["RAD"] * (v_arr / 30.0) ** 2 + rng.normal(0, honest["SLIP"], P)
             + rng.normal(0, honest["LAT"], P)
             + np.where(rng.random(P) < honest["BURST_P"], rng.normal(honest["BURST_MU"], honest["BURST_SD"], P), 0.0))
    noise = e + rng.normal(0, honest["SMUDGE"], P) - A["dh"][site, vi]
    lvl_i = A["lvl"][site, vi]
    if mon_raise is not None:
        lvl_i = np.where(mon_raise, np.minimum(lvl_i + 1, len(A["levels"]) - 1), lvl_i)
    se_i = np.sqrt(A["levels"][lvl_i] ** 2 + honest["SIGMA0"] ** 2)
    sensed = qual[site, vi] & (rng.random(P) < keep_prob(v_arr))
    if mon_stop is not None:
        sensed &= ~mon_stop
    if stop_mask is not None:
        sensed &= ~stop_mask[site, vi]
    out = dict(M=M0, s=s, n_veh=n_veh, b_hon=b_vss + noise)
    if attack is not None:
        ra = rng_att or np.random.default_rng(12345)
        kind = attack[0]
        b_att = out["b_hon"].copy()
        if kind == "persistent":
            b_att = (1 + b_vss) * (1 + attack[1]) - 1 + noise
        elif kind == "days":
            bad = ra.random((n_veh * EPOCHS, DAYS)) < attack[2]
            hit = bad[key, ra.integers(0, DAYS, P)]
            b_att = np.where(hit, (1 + b_vss) * (1 + attack[1]) - 1, b_vss) + noise
        elif kind == "cap":
            ratio = np.minimum(1.0, attack[1] / v_arr)
            b_att = (1 + b_vss) * ratio - 1 + noise
        out["b_att"] = b_att
    # thinning to the passes that produce an entry; append relayed passes for the framing class
    keep = sensed
    out.update(key=key[keep], se_i=se_i[keep], lvl=lvl_i[keep], b_hon=out["b_hon"][keep])
    if "b_att" in out:
        out["b_att"] = out["b_att"][keep]
    if attack is not None and attack[0] == "framing":
        n_inj = attack[1]
        ra = rng_att or np.random.default_rng(12345)
        k_inj = np.repeat(np.arange(n_veh * EPOCHS), n_inj)
        # injected entry: a pass of the victim's OBU measured at another site, result -15 %, at a random level of the
        # victim's own entries in that epoch (or of the population if none)
        lv = A["lvl"][vs[k_inj // EPOCHS, 0], ra.integers(0, len(A["speeds"]), k_inj.size)]
        se_inj = np.sqrt(A["levels"][lv] ** 2 + honest["SIGMA0"] ** 2)
        b_inj = (1 + s[k_inj // EPOCHS]) * (1 + attack[2]) - 1 + ra.normal(0, 1, k_inj.size) * se_inj * 0.6
        out["b_att"] = np.concatenate([out["b_hon"], b_inj])
        out["key_att"] = np.concatenate([out["key"], k_inj])
        out["se_att"] = np.concatenate([out["se_i"], se_inj])
        out["lvl_att"] = np.concatenate([out["lvl"], lv])
    out["M"] = np.bincount(out["key"], minlength=n_veh * EPOCHS).reshape(n_veh, EPOCHS)
    out["M0"] = M0
    return out


def _sim_view(sim, attacked):
    """The insurer's view (keys, sigmas, levels, M) for the honest or the manipulated arm."""
    if attacked and "key_att" in sim:
        v = dict(sim, key=sim["key_att"], se_i=sim["se_att"], lvl=sim["lvl_att"])
        v["M"] = np.bincount(v["key"], minlength=sim["n_veh"] * EPOCHS).reshape(sim["n_veh"], EPOCHS)
        return v
    return sim


def decide_popcal(sim, b, qtab, n_levels, f=None):
    """the decision simulation (level variant) rules with R2 standardised by the per-level population scale f (estimated on the honest population)."""
    if f is None:
        f = population_scale(sim, b, n_levels)
    return E.decide(sim, b, qtab, r2_scale=f[sim["lvl"]]), f


def honest_run(tag, A, sw, qual, nv, lam, vt, qtab, nl, honest=HONEST, cell_mult=None, stop_mask=None):
    sim = simulate(seed("h5", tag, vt, lam), A, sw, qual, nv, lam, vt, honest=honest, cell_mult=cell_mult,
                   stop_mask=stop_mask)
    d, f = decide_popcal(sim, sim["b_hon"], qtab, nl)
    h = E.honest_summary(d)
    h.update(entries_per_epoch_mean=float(sim["M"].mean()), crossings_per_epoch_mean=float(sim["M0"].mean()),
             low_m_share=float((sim["M"] < DB.M_MIN).mean()), popcal_scale=f.round(3).tolist())
    return h


def attack_run(att, A, sw, qual, nv, lam, vt, qtab, nl, honest=HONEST):
    sim = simulate(seed("p5", att, vt, lam), A, sw, qual, nv, lam, vt, honest=honest, attack=att,
                   rng_att=seed("a5", att, vt, lam))
    _, f = decide_popcal(sim, sim["b_hon"], qtab, nl)
    va = _sim_view(sim, True)
    da = E.decide(va, sim["b_att"], qtab, r2_scale=f[va["lvl"]])
    P = float(da["punitive"].mean())
    return dict(punitive_per_year=P, punitive_ub95=cp_upper(int(da["punitive"].sum()), nv),
                notice_per_year=float(da["notice"].mean()))


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
    part = os.path.join(OUT, f"decisions_monitor_L{nl}_{tag}.partial.json")
    rep = json.load(open(part)) if os.path.exists(part) else dict(
        params=dict(levels=A["levels"].tolist(), n_levels=nl, SIGMA_MAX=SIGMA_MAX, BURST_S=BURST_S, ARC_M=ARC_M,
                    keep_prob={str(v): float(keep_prob(v)) for v in A["speeds"]}),
        honest={}, margin={}, badsites={}, power={})
    log(f"resuming: {sum(len(rep[k]) for k in ('honest', 'margin', 'badsites', 'power'))} results already in {part}")

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
                log(f"honest cv={cv} {vt:8s} lam={lam:2d}: entries/epoch {h['entries_per_epoch_mean']:.1f} of crossings "
                    f"{h['crossings_per_epoch_mean']:.1f}, M<5 share {h['low_m_share']:.2f} | R2/ep {h['r2_epoch']:.1e} "
                    f"punitive/yr {h['punitive_per_year']:.1e} (UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
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
            log(f"margin {k:9s} {str(x):6s} {vt:8s} lam={lam:2d}: punitive/yr {h['punitive_per_year']:.1e} "
                f"(UB {h['punitive_ub95']:.1e}) notice/yr {h['notice_per_year']:.1e}")
    # bad sites: a share of sites at 2x their disclosed sigma; without monitor, and with the monitor's stop
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
                log(f"bad sites {share:.2f} x2, {name}: {vt:8s} lam={lam:2d}: punitive/yr {h['punitive_per_year']:.1e} "
                    f"(UB {h['punitive_ub95']:.1e})")
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
                    f"notice/yr {r['notice_per_year']:.2f}")
    rep["seconds_last_run"] = time.time() - t0
    name = f"decisions_monitor_L{nl}_{tag}.json"
    json.dump(rep, open(os.path.join(OUT, name), "w"), indent=1, default=float)
    log(f"elapsed {time.time() - t0:.0f} s -> {name}")


if __name__ == "__main__":
    main()
