"""Stage-3 additions called by pipeline/stage3_criteria.main on the same records. All descriptive (never gating, never
a claim by themselves) except the incomplete-run labels, which follow a rule pre-specified before the data:
  * R8  qualification validity: per cell and variant (gen_on), the per-crossing sigma at qualified vs
        unqualified sites (gated test passes), the share of qualified sites whose test sigma exceeds 1.5 x 6 % with
        Clopper-Pearson, the Spearman rank correlation between the survey sd and the test sigma over all sites;
  * the threshold-sensitivity table, on the governing variant of the stratum, gen_on:
        residual gate tau, qualification limit, decoding window, consistency cut |z|; the SNR offsets come from the
        stage-2 tags of constants evaluation_v5.sensitivity.snr_offsets_db;
  * the V2V-probability blockage reading (not a bound): the urban entry yield if the share
        of crossings with a blocked head equalled the mean V2V NLOSv probability of TR 37.885 Table 6.2-1 (urban
        P_LOS(d) = min(1, 1.05 exp(-0.0114 d))) over the crossing at the site geometry, each such crossing behaving like
        the simulated P3 blocked crossings (partly blocked, head A only); plus the yield of P3 crossings blocked over most
        of their length (blocked-beacon share >= 0.5);
  * qualification probability against the realised in-band K: binned shares with
        Clopper-Pearson and a logistic fit, over the TR 37.885 sites of all cells; descriptive only, never a rule;
  * the incomplete-run labels: per cell, n completed sites of the planned 10;
        n >= 4 -> 'n of 10 planned site draws (compute window)', n <= 3 -> 'not run (compute)'."""
import os
from contextlib import contextmanager

import numpy as np

from pipeline import stage3_criteria as K

C = K.C
EV5 = C.get("evaluation_v5", {})
SENS = EV5.get("sensitivity", {"gate": [1.2, 1.3, 1.5, None], "sigma_q": [0.05, 0.06, 0.07], "window": [0.20, 0.25, 0.30],
                              "z_cut": [1.0, 1.5, 2.0], "snr_offsets_db": [0, -10, -20], "snr_cells": ["P1-15", "P2-30"]})
PLANNED = K.EV4["primary_sites"][1] - K.EV4["primary_sites"][0]


@contextmanager
def knobs(gate=None, sig_q=None, win=None, z_cut=None):
    """Temporarily replace the pre-specified thresholds read by criteria's functions (gate None in the list = no gate)."""
    old = (K.GATE, K.SIG_Q, K.WIN, K.PC["z_threshold"])
    try:
        if gate is not None:
            K.GATE = gate
        if sig_q is not None:
            K.SIG_Q = sig_q
        if win is not None:
            K.WIN = win
        if z_cut is not None:
            K.PC["z_threshold"] = z_cut
        yield
    finally:
        K.GATE, K.SIG_Q, K.WIN, K.PC["z_threshold"] = old


def stratum_of(cell):
    return "highway" if cell == "P2" else "urban"


# ------------------------------------------------------------------------------------------------ R8 qualification validity
def qual_validity(recs, cells, governing):
    from scipy import stats as st
    out = {}
    for cell, v0 in cells:
        rows = []
        for mp, prb in K.COMBOS:
            T, _ = K.site_table(recs, cell, v0, "gen_on", mp, prb)
            if not T:
                continue
            q = [s for s in T if T[s]["sd_ref"] <= K.SIG_Q]
            u = [s for s in T if T[s]["sd_ref"] > K.SIG_Q]
            site_sig = {s: K.sigma(T[s]["e"][T[s]["keep_gate"]]) for s in T}
            over = [s for s in q if np.isfinite(site_sig[s]) and site_sig[s] > K.OC["margin"] * K.SIG_Q]
            ss = [s for s in T if np.isfinite(site_sig[s])]
            rho = (float(st.spearmanr([T[s]["sd_ref"] for s in ss], [site_sig[s] for s in ss])[0])
                   if len(ss) >= 3 else np.nan)
            rows.append(dict(mapping=mp, prb=prb, n_qualified=len(q), n_unqualified=len(u),
                             sigma_qualified=K.sigma(K.pooled(T, q, mask="keep_gate")),
                             sigma_unqualified=K.sigma(K.pooled(T, u, mask="keep_gate")),
                             qualified_over_margin=dict(k=len(over), n=len(q), share=len(over) / len(q) if q else np.nan,
                                                        cp95=K.clopper_pearson(len(over), len(q)),
                                                        threshold=K.OC["margin"] * K.SIG_Q),
                             spearman_sdref_vs_test_sigma=rho, n_sites=len(ss)))
        g = governing.get(stratum_of(cell), {})
        at = [r for r in rows if r["mapping"] == g.get("mapping") and r["prb"] == g.get("prb")]
        out[f"{cell}-{v0:g}"] = dict(variants=rows, at_governing=at[0] if at else None)
    return out


# ------------------------------------------------------------------------------------------------ threshold sensitivities
def _gov_variant(cell, governing):
    g = governing.get(stratum_of(cell), {})
    return g.get("mapping", "e1"), int(g.get("prb", 10))


def _quantities(recs, cell, v0, mp, prb):
    T, cnt = K.site_table(recs, cell, v0, "gen_on", mp, prb)
    q = [s for s in T if T[s]["sd_ref"] <= K.SIG_Q]
    e = K.pooled(T, sorted(T), True)
    disc = (float(np.mean(np.concatenate([~T[s]["keep_gate"] for s in q]))) if q else np.nan)
    return dict(k=len(q), n=len(T), sigma=K.sigma(e), q99=K.q_abs(e, 0.99),
                oow=float(np.mean(np.abs(e) > K.WIN)) if e.size else np.nan, gate_discard=disc, n_kept=int(e.size))


def sensitivity_table(recs, cells, governing, snr_rows=None):
    out = {}
    for cell, v0 in cells:
        mp, prb = _gov_variant(cell, governing)
        key = f"{cell}-{v0:g}"
        row = dict(variant=[mp, prb], gate={}, sigma_q={}, window={}, z_cut={})
        for g in SENS["gate"]:
            with knobs(gate=np.inf if g is None else g):
                row["gate"]["none" if g is None else f"{g:g}"] = _quantities(recs, cell, v0, mp, prb)
        for sq in SENS["sigma_q"]:
            with knobs(sig_q=sq):
                row["sigma_q"][f"{sq:g}"] = _quantities(recs, cell, v0, mp, prb)
        for w in SENS["window"]:
            with knobs(win=w):
                row["window"][f"{w:g}"] = _quantities(recs, cell, v0, mp, prb)
        if key in K.EV4["k3_c4_light"]["cells"]:
            for z in SENS["z_cut"]:
                with knobs(z_cut=z):
                    sp = K.separation(recs, cell, v0, "gen_on", mp, prb)
                row["z_cut"][f"{z:g}"] = dict(Z=sp["Z"], p_zq_minus1_honest=(sp["zq_shares_honest"] or [np.nan])[0],
                                              p_zq_minus1_beta10=sp["p_zq_minus1"].get("-0.10"))
        if snr_rows and key in snr_rows:
            row["snr_offset_db"] = snr_rows[key]
        out[key] = row
    return out


def snr_offset_rows(tag, families, governing):
    """sigma (qualified + gated, gen_on, lb mapping at the governing PRB) at link-budget offsets; offset 0 is the
    primary lb record; the other offsets are the stage-2 tags <tag>_snr<|dB|> (primary: sens_snr<|dB|>)."""
    out = {}
    for off in SENS["snr_offsets_db"]:
        t = tag if off == 0 else (f"sens_snr{abs(off)}" if tag == "primary" else f"{tag}_snr{abs(off)}")
        recs, _ = K.load(families, t)      # a missing tag loads as (reported as missing); a stage2_consistent
        #                                    refusal propagates; the merge requires every tag
        for key in SENS["snr_cells"]:
            cell, v0 = key.split("-")[0], float(key.split("-")[1])
            _, prb = _gov_variant(cell, governing)
            if not recs:
                out.setdefault(key, {})[f"{off:+d}"] = None
                continue
            q = _quantities(recs, cell, v0, "lb", prb)
            out.setdefault(key, {})[f"{off:+d}"] = dict(sigma=q["sigma"], k=q["k"], n=q["n"], oow=q["oow"])
    return out


# ------------------------------------------------------------------------------------------------ V2V-pessimistic row
def p_los_urban(d):
    """TR 37.885 V15.3.0 Table 6.2-1, urban, same street (V2V): P_LOS = min(1, 1.05 exp(-0.0114 d))."""
    return np.minimum(1.0, 1.05 * np.exp(-0.0114 * np.asarray(d, float)))


def v2v_weight(y_lane=-1.75):
    """w = mean over the crossing window (x uniform on [-0.05 L, 1.05 L]) and both heads (pole offset at the midpoint
    of its uniform range) of 1 - P_LOS(horizontal distance)."""
    L = C["geometry"]["L_m"]
    a, b = C["window_rule"]["arc_L"]
    po = float(np.mean(C["geometry"]["pole_offset_m_uniform"]))
    x = np.linspace(a * L, b * L, 1101)
    d = np.concatenate([np.hypot(x - xh, po - y_lane) for xh in (0.0, L)])
    return float(np.mean(1.0 - p_los_urban(d)))


def v2v_reading(recs, governing):
    mp, prb = _gov_variant("P1", governing)
    TU, _ = K.site_table(recs, "P1", 15.0, "gen_on", mp, prb)
    TN, _ = K.site_table(recs, "P3", 15.0, "gen_on", mp, prb)
    if not TU or not TN:
        return None
    qu = [s for s in TU if TU[s]["sd_ref"] <= K.SIG_Q]
    qn = [s for s in TN if TN[s]["sd_ref"] <= K.SIG_Q]
    yu = (sum(int(TU[s]["keep_gate"].sum()) for s in qu) / max(sum(TU[s]["n_test_all"] for s in qu), 1)) if qu else np.nan
    kept_blk = sum(int((TN[s]["keep_gate"] & TN[s]["blocked"]).sum()) for s in qn)
    incl_blk = sum(int(TN[s]["blocked"].sum()) for s in qn)
    kept_maj = sum(int((TN[s]["keep_gate"] & TN[s]["blocked_major"]).sum()) for s in qn)
    incl_maj = sum(int(TN[s]["blocked_major"].sum()) for s in qn)
    # blocked N crossings that were excluded before the gate (window / non-finite) are not in TN: the yield of the
    # blocked included crossings is scaled by the inclusion share of N's test crossings at qualified sites
    incl = sum(TN[s]["e"].size for s in qn) / max(sum(TN[s]["n_test_all"] for s in qn), 1) if qn else np.nan
    yb = (kept_blk / incl_blk) * incl if incl_blk else np.nan
    w = v2v_weight()
    yv = (1 - w) * yu + w * yb if np.isfinite(yu) and np.isfinite(yb) else np.nan
    ym = (kept_maj / incl_maj) * incl if incl_maj else np.nan
    return dict(variant=[mp, prb], w_v2v=w, entry_yield_unblocked_P1_15=yu, entry_yield_blocked_P3_15=yb,
                entry_yield_v2v=yv, entry_yield_blocked_majority=ym, n_blocked_majority=int(incl_maj),
                factor_vs_unblocked=(yv / yu) if (np.isfinite(yv) and yu > 0) else np.nan,
                note=("V2V-probability reading: the share of urban crossings with a blocked head set to the mean "
                      "V2V NLOSv probability of TR 37.885 Table 6.2-1 over the crossing; blocked crossings as simulated (partly "
                      "blocked, head A only); not a bound. entry_yield_blocked_majority: crossings with a blocked-beacon share >= 0.5"))


# ------------------------------------------------------------------------------------------------ qualification vs K
def qual_vs_k(recs, cells, governing, tag="primary", bins=(-np.inf, 2, 4, 6, 8, 10, np.inf)):
    """Per TR 37.885 site: the weaker head's realised in-band K at the governing PRB over its survey passes, where it is qualified
    (realised_k_median, role ref; then the stage-1 fields, then broadband), qualified or not, the survey sd and the gated
    test sigma (Fig. E); the binned shares and the logistic fit leave out P3 (blocked crossings of urban
    sites, not a site type); and the type's realised K over its test passes (k_realised)."""
    from scipy.optimize import minimize
    fam_of = {"P1": "U", "P2": "H", "P3": "N"}
    pts = []
    for cell, v0 in cells:
        mp, prb = _gov_variant(cell, governing)
        T, _ = K.site_table(recs, cell, v0, "gen_on", mp, prb)
        for s_ in T:
            k, src = realised_k_median(recs, cell, prb, v0, s_, role="ref"), "in-band (survey passes)"
            if not np.isfinite(k):
                k, src = site_inband_k(fam_of[cell], s_, tag, prb), "in-band (survey passes, stage-1 fields)"
            if k is None:
                kb = [min(r["K_realised_db"][:2]) for r in recs if r["cell"] == cell and r["v0"] == v0 and r["site"] == s_
                      and r["arm"] == "gen_off" and r["mapping"] == "e1" and r["prb"] == K.PRBS[0] and r["role"] == "ref"
                      and "K_realised_db" in r]
                if not kb:
                    continue
                k, src = float(np.median(kb)), "broadband"
            pts.append((k, bool(T[s_]["sd_ref"] <= K.SIG_Q), f"{cell}-{v0:g}", T[s_]["sd_ref"],
                        K.sigma(T[s_]["e"][T[s_]["keep_gate"]]), src))
    if not pts:
        return None
    fit_pts = [p for p in pts if not p[2].startswith("P3")]    # P3 is a traffic condition of urban sites
    k = np.array([p[0] for p in fit_pts]); y = np.array([p[1] for p in fit_pts], float)
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (k >= lo) & (k < hi)
        n, q = int(m.sum()), int(y[m].sum())
        binned.append(dict(k_lo=lo, k_hi=hi, n=n, qualified=q, share=q / n if n else np.nan, cp95=K.clopper_pearson(q, n)))
    nll = lambda b: np.sum(np.logaddexp(0, b[0] + b[1] * k) - y * (b[0] + b[1] * k))
    fit = minimize(nll, np.array([0.0, 0.0]), method="Nelder-Mead") if 0 < y.sum() < y.size else None
    b = fit.x.tolist() if fit is not None and fit.success else None
    k50 = (-b[0] / b[1]) if b and b[1] > 0 else None
    kr = {}
    for name, cell in (("urban", "P1"), ("highway", "P2")):     # the type's realised K
        if any(c == cell for c, _ in cells):
            v = realised_k_median(recs, cell, _gov_variant(cell, governing)[1])
            kr[name] = v if np.isfinite(v) else None
    return dict(points=[dict(k_db=p[0], qualified=p[1], cell=p[2], sd_ref=p[3], sigma_test=p[4], k_source=p[5]) for p in pts],
                binned=binned, logistic=b, k50_db=k50, k_realised=kr,
                note="descriptive only; never a rule")


# ------------------------------------------------------------------------------------------------ R8 (iv), (f)
def over_governing(recs, cells, governing, head):
    """Share of qualified sites whose gated test sigma exceeds 1.5 x the governing sigma of the cell, with
    Clopper-Pearson (the calibration of the T8 falsification statement)."""
    out = {}
    hd = {f"{h['cell']}-{h['v0']:g}": h for h in head}
    for cell, v0 in cells:
        key = f"{cell}-{v0:g}"
        hf = (hd.get(key) or {}).get("headline_final_qualified_gated") or {}
        if hf.get("value_pct") is None:
            continue
        thr = K.OC["margin"] * hf["value_pct"] / 100
        mp, prb = _gov_variant(cell, governing)
        T, _ = K.site_table(recs, cell, v0, "gen_on", mp, prb)
        q = [s_ for s_ in T if T[s_]["sd_ref"] <= K.SIG_Q]
        over = [s_ for s_ in q if K.sigma(T[s_]["e"][T[s_]["keep_gate"]]) > thr]
        out[key] = dict(threshold=thr, k=len(over), n=len(q), share=len(over) / len(q) if q else np.nan,
                        cp95=K.clopper_pearson(len(over), len(q)))
    return out


# ------------------------------------------------------------------------------------------------ per-beacon, in-band K
def per_beacon_governing(recs, cells, governing, bins=tuple(range(-5, 50, 5))):
    """Fig. C overlay: gen_on, e1 mapping, the stratum's governing PRB; per SNR bin (weaker
    head) the RMS and the q10 / q50 / q90 of |per-beacon FDOA error|, and the SNR histogram."""
    out = {}
    for cell, v0 in cells:
        _, prb = _gov_variant(cell, governing)
        rs = [r for r in recs if r["cell"] == cell and r["v0"] == v0 and r["arm"] == "gen_on" and r["mapping"] == "e1"
              and r["prb"] == prb and "per_beacon" in r and K.included(r)]
        if not rs:
            continue
        err = np.concatenate([np.array(r["per_beacon"]["fdoa_err_hz"], float) for r in rs])
        snr = np.concatenate([np.min(np.array(r["per_beacon"]["snr_db"]), axis=0) for r in rs])
        rows = {}
        for b in bins:
            m = (snr >= b) & (snr < b + 5) & np.isfinite(err)
            if m.sum() >= 20:
                a = np.abs(err[m])
                rows[f"{b}"] = dict(n=int(m.sum()), rms=float(np.sqrt(np.mean(err[m] ** 2))),
                                    q10=float(np.quantile(a, 0.1)), q50=float(np.quantile(a, 0.5)), q90=float(np.quantile(a, 0.9)))
        h, edges = np.histogram(snr[np.isfinite(snr)], bins=np.arange(-10, 61, 2.5))
        dec = decode_threshold_db(prb)
        fin = snr[np.isfinite(snr)]
        out[f"{cell}-{v0:g}"] = dict(prb=prb, mapping="e1", by_snr=rows, snr_hist=dict(counts=h.tolist(), edges=edges.tolist()),
                                     decoding_db=dec,           # the line at this overlay's PRB
                                     below_decoding_share=float(np.mean(fin < dec)) if fin.size else np.nan,
                                     snr_q01_db=float(np.quantile(fin, 0.01)) if fin.size else np.nan)
    return out


def decode_threshold_db(prb):
    """The decoding line at a PRB (constants evaluation_v5.decoding; evaluate.decode_threshold_db)."""
    d = EV5["decoding"]
    return float(d["snr_50rb_db"] + 10 * np.log10(prb / d["refsens_rb"]))


INHOUSE_K_GRID = (6.0, 10.0, 15.0, 20.0)


def realised_k_median(recs, cell, prb, v0=None, site=None, role="test"):
    """The one realised-K function: the equal-K cell (Table B), the bridge label, the
    K-anchor value and Fig. E. Median over the stratum's (or one site's) honest test passes at the PRB of the weaker
    head's in-band K of the TR 37.885 pass (K_inhouse_db: the per-head in-band K at which the paired in-house arm is drawn);
    role="ref": over the survey passes, the K at which a site is qualified."""
    ks = [min(r["K_inhouse_db"]) for r in recs if r["cell"] == cell and r["arm"] == "inhouse" and r["prb"] == prb
          and r["role"] == role and r.get("beta", 0.0) == 0.0 and r.get("K_inhouse_db") and r.get("kin", "sumo") == "sumo"
          and (v0 is None or r["v0"] == v0) and (site is None or r["site"] == site)]
    return float(np.median(ks)) if ks else np.nan


def inhouse_k_cell(k_med):
    """Equal-K rule for the in-house consistency z: the largest in-house K <= the realised median, >= 6 dB."""
    return max([k for k in INHOUSE_K_GRID if np.isfinite(k_med) and k <= k_med] or [INHOUSE_K_GRID[0]])


def site_inband_k(fam, site, tag, prb):
    """Median over the site's reference passes of the weaker head's realised in-band K at the PRB (stage-1 K fields,
    npz or kx extract); None if absent."""
    import json as _j
    d = os.path.join(K.RUNS, f"g1_{fam}")
    js = os.path.join(d, f"site_{site:03d}.json")
    if not os.path.exists(js):
        return None
    S1 = _j.load(open(js))
    if S1.get("provenance", {}).get("tag") != tag:
        return None
    if os.path.exists(js[:-5] + ".npz"):
        Z = np.load(js[:-5] + ".npz")
        prbs = list(Z["prbs"])
    else:
        kx = js[:-5] + ".kx.json"
        if not os.path.exists(kx):
            return None
        Z = {k: np.asarray(v, float) for k, v in _j.load(open(kx)).items()}
        prbs = list(C["beacon"]["prb"]["swept"])
    if prb not in prbs:
        return None
    i = prbs.index(prb)
    vals = [float(np.min(Z[f"t{m['k']}_K_band_db"][:2, i])) for m in S1["meta"]
            if m.get("role") == "ref" and f"t{m['k']}_K_band_db" in Z]
    return float(np.median(vals)) if vals else None


# ------------------------------------------------------------------------------------------------ incomplete run
def incomplete_labels(n_sites):
    """n_sites: completed primary sites per cell."""
    out = {}
    for cell, n in n_sites.items():
        if n >= PLANNED:
            out[cell] = dict(n=n, label=None)
        elif n >= 4:
            out[cell] = dict(n=n, label=f"{n} of {PLANNED} planned site draws (compute window)")
        else:
            out[cell] = dict(n=n, label="not run (compute)")
    return out


def apply_incomplete_scope(sc, labels):
    """: under --incomplete-run, a cell labelled 'not run (compute)' gets that rating and leaves the type rule; a
    type is rated on its run cells, with the suffix '(speed classes [v] not run: compute window)'."""
    out = dict(cells=dict(sc["cells"]), types={})
    for c, lab in labels.items():
        if c in out["cells"] and lab.get("label") == "not run (compute)":
            out["cells"][c] = dict(out["cells"][c], rating="not run (compute)")
    for t, spec in K.TYPES.items():
        run = [c for c in spec["cells"] if c in out["cells"] and out["cells"][c]["rating"] != "not run (compute)"]
        skipped = [c.split("-")[1] for c in spec["cells"] if c in out["cells"] and c not in run]
        if not run:
            if skipped:
                out["types"][t] = "not run (compute)"
            continue
        base = K.type_rating(out["cells"], run)            # the normal type rule on the run cells
        out["types"][t] = base + (f" (speed classes {', '.join(skipped)} m/s not run: compute window)" if skipped else "")
    return out


def window_closed(now_utc=None):
    """True if the pre-specified compute window (constants evaluation_v5.window_end_utc) has closed."""
    import datetime as dt
    end = EV5.get("window_end_utc")
    if not end:
        return False
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    return now >= dt.datetime.fromisoformat(end.replace("Z", "+00:00"))


def add_v5(out, recs, cells, n_sites, tag, families, incomplete=False):
    """All study-1 keys into the criteria output dict."""
    gov = out["governing"]
    out["qualification_validity"] = qual_validity(recs, cells, gov)
    out["sensitivity_thresholds"] = sensitivity_table(recs, cells, gov, snr_offset_rows(tag, families, gov))
    out["v2v_reading"] = v2v_reading(recs, gov)
    out["qualification_vs_k"] = qual_vs_k(recs, cells, gov, tag)
    out["over_governing_r8iv"] = over_governing(recs, cells, gov, out["headline"])
    out["per_beacon_governing"] = per_beacon_governing(recs, cells, gov)
    labels = incomplete_labels(n_sites) if incomplete else None
    out["incomplete_run"] = dict(applied=bool(incomplete), labels=labels)
    if incomplete:
        out["scope"] = apply_incomplete_scope(out["scope"], labels)
    out["design"] = "study-1 evaluation design (constants evaluation_v5)"
    return out


if __name__ == "__main__":
    print(os.path.basename(__file__), "is called by pipeline/stage3_criteria.py")
