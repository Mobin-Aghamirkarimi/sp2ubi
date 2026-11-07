"""Pilot decision (constants evaluation_v4.p0 and lever_d). Reads the stage-2 records
of the P0 families (runs/g1_<fam>_<tag>/eval_<tag>/), computes per cell the metrics Q, S, T, Z and q99 |e| for the plain
LS statistics and for the Huber fit (d), per arm (gen_on governs; in-house for the guards) and per variant (2 mappings x 3 PRB),
takes the worst variant per metric, and applies the pre-specified decision table in its pre-specified order:
  1 pipeline fault   non-finite share of the LS statistic > 5 % in any cell (gen_on or in-house), or a C8 file of the
                     tag failing; a non-finite share > 5 % of (d) alone is no pipeline fault: it makes the adoption rule
                     of (d) fail (reason "(d) non-finite")
  2 lever check      any cell with Q <= Q_max[family], S > 5 % or T > 0.5 % -> the adoption rule of (d)
  3 D-stop (P0H)     0 of 6 sites qualify at P2-20 and at P2-30
  4 D-qual (P0U, P0N) 0 sites qualify at the slowest speed (P1-10, P3-15)
  5 D-run            otherwise
Branches 3-5 use the statistic after that rule (d if the adoption rule passed, else LS). Definitions (as
stage3_criteria.py):
  Q = k / n, k = sites whose select-statistic sd over the included references <= 6 %, n = the family's P0 sites;
  S = sigma (SD of the +-25 %-clipped calibrated residual) of the qualified + gated test passes, pooled;
  T = share of those passes with |e| > 25 %;  q99 = 0.99 quantile of |e| of those passes;
  Z = median over qualified sites of -median(z) at beta = -10 %, z = (d - median_honest) / (1.4826 MAD_honest) over the
      site's honest test passes (d: LS free fit; d_d for the Huber fit);
  worst variant: Q min, S / T / q99 max, Z min.
The adoption rule uses S_all, q99_all and T_all: the same quantities on all included
and gated test passes of all P0 sites of the cell, not conditioned on site qualification, paired (the passes included under
both statistics; matched by site and pass index; the gate is the plain-LS RMS gate for both). It passes only
if, in every trigger cell, S_all and q99_all of (d) are <= 0.9 x LS (gen_on, worst variant) and, in every P0 cell and
both arms, S_all_d <= 1.05 S_all_LS and T_all_d <= T_all_LS + max(0.05 T_all_LS, 1 / n_crossings). The trigger stays on
the qualified-site Q, S, T of LS; Q, S and T on qualified sites are reported for both statistics. Passing the rule is
necessary, not sufficient: adoption also needs the in-house regeneration and its confirmation (lever_d.if_adopted).
Writes results/p0_decision.json (descriptive; never a claim). Usage: python pipeline/pilot_decision.py"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import stage3_criteria as K  # noqa: E402
from pipeline.stage1_channels import C  # noqa: E402
from pipeline.provenance import provenance  # noqa: E402

EV4 = C["evaluation_v4"]
P0 = EV4["p0"]
TAG = P0["tag"]
COMBOS = K.COMBOS
RUNS = os.path.join(ER, "runs")                        # module-level for the synthetic plumbing test only
OUT = os.path.join(ER, "..", "results", "p0_decision.json")


def load_family(fam):
    recs = []
    for f in sorted(glob.glob(os.path.join(RUNS, f"g1_{fam}_{TAG}", f"eval_{TAG}", "site_*.json"))):
        d = json.load(open(f))
        for r in d["recs"]:
            r["family"] = fam
        recs += d["recs"]
    return recs


def as_stat(recs, stat):
    """Records with the chosen statistic in the fields stage3_criteria.py reads (match, prof, finite); mse stays LS."""
    if stat == "ls":
        return recs
    out = []
    for r in recs:
        q = dict(r)
        q["match"], q["prof"] = r.get("match_d", np.nan), r.get("prof_d", np.nan)
        q["finite"] = bool(r.get("window_ok", True) and np.isfinite(q["match"]) and np.isfinite(q["prof"]))
        if "d_d" in r:
            q["d"] = r["d_d"]
        out.append(q)
    return out


def separation(recs, cell, v0, arm, mapping, prb, T):
    """Z: the one k3 definition."""
    return K.separation(recs, cell, v0, arm, mapping, prb, T)["Z"]


def all_gated_paired(T_ls, T_d):
    """The residuals of all included and gated test passes of all sites (qualified or not), for LS and (d) on the same
    passes: the passes present in both site tables (matched by site and pass index j), gated by the plain-LS RMS gate."""
    e_ls, e_d = [], []
    for s in sorted(set(T_ls) & set(T_d)):
        a, b = T_ls[s], T_d[s]
        jb = {int(j): i for i, j in enumerate(b["j"])}
        ia = [i for i, j in enumerate(a["j"]) if int(j) in jb and a["keep_gate"][i]]
        e_ls.append(a["e"][ia])
        e_d.append(b["e"][[jb[int(a["j"][i])] for i in ia]])
    cat = lambda x: np.concatenate(x) if x else np.array([])
    return cat(e_ls), cat(e_d)


def cell_metrics_pair(recs, fam, cell, v0, arm):
    """Per statistic (LS, d): Q, S, T, q99, Z on qualified sites (as before) and S_all, T_all, q99_all on all included
    and gated test passes, paired between the statistics; worst variant per metric."""
    n_sites = EV4["p0"]["families"][fam]["sites"]
    R = {st: as_stat(recs, st) for st in ("ls", "d")}
    rows = {"ls": [], "d": []}
    for mp, prb in COMBOS:
        T = {st: K.site_table(R[st], cell, v0, arm, mp, prb) for st in ("ls", "d")}
        e_all = dict(zip(("ls", "d"), all_gated_paired(T["ls"][0], T["d"][0])))
        for st in ("ls", "d"):
            Tt, cnt = T[st]
            k = sum(Tt[s]["sd_ref"] <= K.SIG_Q for s in Tt)
            e = K.pooled(Tt, sorted(Tt), qualified_gated=True)
            ea = e_all[st]
            rows[st].append(dict(mapping=mp, prb=prb, k=int(k), Q=k / n_sites, S=K.sigma(e), mad=K.scale_estimates(e)[0],
                                 T=float(np.mean(np.abs(e) > K.WIN)) if e.size else np.nan, n=int(e.size), q99=K.q_abs(e, 0.99),
                                 S_all=K.sigma(ea), q99_all=K.q_abs(ea, 0.99),
                                 T_all=float(np.mean(np.abs(ea) > K.WIN)) if ea.size else np.nan, n_all=int(ea.size),
                                 nonfinite=cnt["nonfinite"] / max(cnt["test"], 1),
                                 Z=separation(R[st], cell, v0, arm, mp, prb, Tt) if arm == "gen_on" else np.nan))

    def worst(rr, key, mode):
        v = [r for r in rr if np.isfinite(r[key])]
        if not v:
            return dict(value=np.nan, variant=None, n=0, k=0)
        r = (min if mode == "min" else max)(v, key=lambda x: x[key])
        n = r["n_all"] if key.endswith("_all") else r["n"]
        return dict(value=float(r[key]), variant=[r["mapping"], r["prb"]], n=n, k=r["k"])
    return {st: dict(variants=rr, Q=worst(rr, "Q", "min"), S=worst(rr, "S", "max"), T=worst(rr, "T", "max"),
                     q99=worst(rr, "q99", "max"), Z=worst(rr, "Z", "min"), nonfinite=worst(rr, "nonfinite", "max"),
                     S_all=worst(rr, "S_all", "max"), T_all=worst(rr, "T_all", "max"), q99_all=worst(rr, "q99_all", "max"))
            for st, rr in rows.items()}


def realised_k(recs, cell, v0):
    ks = {}
    for r in recs:
        if r["cell"] == cell and r["v0"] == v0 and r["arm"] == "gen_off" and r["mapping"] == "e1" and r["prb"] == K.PRBS[0]:
            ks.setdefault(r["site"], []).append(r["K_realised_db"])
    return {str(s): [float(np.median(np.array(v)[:, h])) for h in range(2)] for s, v in sorted(ks.items())}


def c8_ok():
    fs = sorted(glob.glob(os.path.join(ER, "..", "results", f"c8_*_{TAG}.json")))
    return bool(fs) and all(json.load(open(f)).get("C8_pass") for f in fs), [os.path.basename(f) for f in fs]


def main():
    fams = list(P0["families"])
    data, out = {}, dict(tag=TAG, provenance=provenance(TAG), note="development data: decides resources and the estimator variant, never a claim",
                         families={})
    for fam in fams:
        recs = load_family(fam)
        cells = sorted({(r["cell"], r["v0"]) for r in recs})
        fd = dict(n_sites_run=len({r["site"] for r in recs}), cells={})
        for cell, v0 in cells:
            key = f"{cell}-{v0:g}"
            m = {arm: cell_metrics_pair(recs, fam, cell, v0, arm) for arm in ("gen_on", "inhouse")}
            fd["cells"][key] = dict(realised_K_db=realised_k(recs, cell, v0),
                                    effective_N=K.per_beacon(recs, [(cell, v0)]).get(key),
                                    **{f"{stat}_{arm}": m[arm][stat] for stat in ("ls", "d") for arm in ("gen_on", "inhouse")})
        data[fam] = recs
        out["families"][fam] = fd
    # 1 pipeline fault
    nf = {f"{fam}/{c}/{sa}": m[sa]["nonfinite"]["value"] for fam, fd in out["families"].items() for c, m in fd["cells"].items()
          for sa in ("ls_gen_on", "ls_inhouse", "d_gen_on", "d_inhouse")}          # both statistics reported
    c8, c8_files = c8_ok()
    bad = lambda v: (not np.isfinite(v)) or v > P0["pipeline_fault"]["nonfinite_share_max"]
    fault = (not c8) or any(bad(v) for k, v in nf.items() if k.split("/")[-1].startswith("ls_"))
    d_nonfinite = sorted(k for k, v in nf.items() if k.split("/")[-1].startswith("d_") and bad(v))
    out["pipeline_fault"] = dict(fault=bool(fault), c8_pass=c8, c8_files=c8_files, nonfinite=nf, d_nonfinite_cells=d_nonfinite,
                                 missing_sites={fam: P0["families"][fam]["sites"] - fd["n_sites_run"] for fam, fd in out["families"].items()})
    if fault or any(out["pipeline_fault"]["missing_sites"].values()):
        out["decision"] = dict(branch="pipeline_fault", action="fix the code; re-run the pilot on new development seeds")
        return write(out)
    # 2 trigger check of (d) (LS, gen_on, worst variant)
    LC = P0["lever_check"]
    trig = [f"{fam}/{c}" for fam, fd in out["families"].items() for c, m in fd["cells"].items()
            if m["ls_gen_on"]["Q"]["k"] <= LC["Q_max"][fam] or m["ls_gen_on"]["S"]["value"] > LC["S_max"]
            or m["ls_gen_on"]["T"]["value"] > LC["T_max"]]
    lever = dict(trigger_cells=trig, rule_passed=False, gains={}, guards={})
    if trig:
        L = EV4["lever_d"]["adoption"]
        ok = True
        for tc in trig:
            fam, c = tc.split("/")
            m = out["families"][fam]["cells"][c]
            ratio = lambda key: (m["d_gen_on"][key]["value"] / m["ls_gen_on"][key]["value"]
                                 if m["ls_gen_on"][key]["value"] > 0 else np.nan)        # NaN > 0 is False: no adoption
            gs, gq = ratio("S_all"), ratio("q99_all")
            good = bool(np.isfinite(gs) and np.isfinite(gq) and gs <= 1 - L["gain_min_rel"] and gq <= 1 - L["gain_min_rel"])
            lever["gains"][tc] = dict(S_ratio=gs, q99_ratio=gq, passes=good)
            ok &= good
        for fam, fd in out["families"].items():
            for c, m in fd["cells"].items():
                for arm in L["guard_arms"]:
                    sl, sd = m[f"ls_{arm}"]["S_all"]["value"], m[f"d_{arm}"]["S_all"]["value"]
                    tl, td = m[f"ls_{arm}"]["T_all"]["value"], m[f"d_{arm}"]["T_all"]["value"]
                    n = max(m[f"ls_{arm}"]["T_all"]["n"], 1)
                    gS = bool(not np.isfinite(sl) or (np.isfinite(sd) and sd <= (1 + L["guard_S_rel"]) * sl))
                    gT = bool(not np.isfinite(tl) or (np.isfinite(td) and td <= tl + max(L["guard_S_rel"] * tl, 1.0 / n)))
                    lever["guards"][f"{fam}/{c}/{arm}"] = dict(S_ok=gS, T_ok=gT)
                    ok &= gS and gT
        if d_nonfinite:                                 # (d) non-finite > 5 % in a cell: the rule fails
            lever["fail_reason"] = "(d) non-finite"
            ok = False
        lever["rule_passed"] = bool(ok)
    out["lever"] = lever
    stat = "d" if lever["rule_passed"] else "ls"
    out["post_lever_statistic"] = stat
    # 3 D-stop, 4 D-qual, 5 D-run (per family; gen_on, worst variant). Both statistics' branches are written: the one
    # of the statistic finally adopted at the evaluation applies
    ds = P0["D_stop"]

    def branches(st):
        k_of = lambda fam, c: out["families"][fam]["cells"][c][f"{st}_gen_on"]["Q"]["k"]
        stop_ = all(k_of(ds["family"], c) == 0 for c in ds["cells"])
        br = {}
        for fam in fams:
            if fam == ds["family"] and stop_:
                br[fam] = "D_stop"
            elif fam in P0["D_qual"]["families"] and k_of(fam, P0["D_qual"]["slowest"][fam]) == 0:
                br[fam] = "D_qual"
            else:
                br[fam] = "D_run"
        return br, stop_
    branch, stop = branches(stat)
    out["per_family_ls"], _ = branches("ls")
    out["per_family_d"], _ = branches("d")
    base = {fam: P0["families"][fam]["base"] for fam in fams}
    out["decision"] = dict(branch="D_stop" if stop else "per_family", per_family=branch,
                           p2_dqual=sorted(base[f] for f, b in branch.items() if b == "D_qual"),
                           lever_rule_passed=lever["rule_passed"],
                           action=("no main run" if stop else
                                   "set constants evaluation_v4.p2_dqual (and lever_d.adopted only after the in-house "
                                   "regeneration is confirmed) before the evaluation"))
    return write(out)


def write(out):
    path = OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps(out["decision"], indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
