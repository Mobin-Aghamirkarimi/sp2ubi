"""Bootstrap coverage check of C1 (constants criteria.C1.coverage_check).
In-house against in-house with independent seeds, so the true R is 1 and the true Δ is 0; nothing from a generator.
  * fresh sites for every replicate: `replicates` x `sites` sites (in-house site-accuracy kinematics at 15 m/s, pole offset U(6, 12) m, site K ~ N(9, 5^2) dB floored at 5 dB as
    the site population of the in-house site-accuracy simulation), per site two independent sets of 30 reference + 60 test passes through the stage-2 in-house
    chain;
  * each replicate runs criteria.c1_cell on its own 20 sites with B = `boot` bootstrap resamples and its own
    bootstrap key: coverage = share of replicates whose 95 % CI of R contains 1 (target 0.95), Clopper-Pearson 95 %;
  * false-"confirmed" rate: the same replicates with the first set's residuals scaled by 1.25 around their site
    reference mean (true R = 1.25): share of replicates with R <= 1.25, |Δ| <= 0.5 % and p <= 0.025 (single cell).
10 sites per replicate; also
  * the coverage of the 95 % site-bootstrap CI of the absolute sigma (qualified + gated, first copy, true R = 1) of the
    pooled sigma of all replicates' sites (the population value of this in-house design); if < 0.93 the stage-3 sigma CIs
    are widened by c(n) (criteria reads widen_sigma_ci);
  * the size of the one-sided optimism check at the margin: the share of single-cell 'not contradicted' verdicts
    (R_hat <= 1.5 and p <= 0.025), over the replicates in which the check runs (>= min_sites sites with test passes),
    when the first copy's reference and test residuals are scaled by 1.5 around their site reference mean, so that site
    qualification responds as under a 1.5x optimistic channel;
    c starts at 1.0 and is raised in steps of 0.05; in each replicate the widening applied is c_eff = max(c(n_rep), c),
    as stage 3 applies it; also P(not contradicted) at true R = 1 with that c.
Populations: two site-K populations, each with its own fresh sites and seeds: 'a' = N(9, 5^2) dB floored at
5 dB and 'b' = N(4.5, 1.5^2) dB floored at 1 dB (the realised in-band K of urban LOS).
c_calibrated is the smallest such c with size <= 0.025 in every population, where the size of a population is taken
over its runnable replicates if the check runs in >= 20 of them, and over all replicates otherwise (a check that rarely
runs can only rarely say 'not contradicted'). If no c <= c_max meets this, c_calibrated is null and calibrated is false:
stage 3 then caps every optimism verdict at 'undetermined' (reason 'K7 not calibrated'). The sigma CIs are widened if
either population's sigma-CI coverage is < 0.93 (stage3_criteria.py reads widen_sigma_ci), the coverage being taken over the
replicates in which a sigma CI exists (>= min_sites qualified sites contributing passes), and 'not estimable' (no
widening from that population) with fewer than min_runnable_replicates such replicates.
Runs before the primary stage 3; the per-site cache is stage 2, the result stage 3.
The statistic is the Huber fit (d) if constants evaluation_v4.lever_d.adopted (stage3_criteria.use_statistic). Every per-site cache
file carries its provenance and is reused only with the same provenance; the whole check is skipped if its result file
exists with the same provenance.
Writes results/coverage_check_<tag>.json. Usage: python pipeline/coverage_check.py --tag sens_coverage"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
from pipeline import stage3_criteria as K  # noqa: E402
from pipeline.coverage_sim import CC, POPS, V0, V5K7, pool  # noqa: E402  (the stage-2 site simulation)
from pipeline.provenance import check_tag, provenance, same_provenance  # noqa: E402


def records(sites, n_ref, scale=1.0, cell="PCOV", whole_site=False):
    """whole_site: the first copy's reference and test residuals are scaled around the site reference mean (the K7 size
    at the margin); otherwise only its test residuals (the C1 false-confirmed rate)."""
    recs = []
    for i, d in enumerate(sites):
        for c, arm in ((0, "gen_off"), (1, "inhouse")):
            ps = d["copies"][c]
            keys = [x for x in ("match", "prof", "match_d", "prof_d") if x in ps[0]]
            ref_mean = {key: np.nanmean([p[key] for p in ps[:n_ref]]) for key in keys}
            for k, p in enumerate(ps):
                q = dict(p)
                if c == 0 and scale != 1.0 and (whole_site or k >= n_ref):   # true R = scale around the ref mean
                    for key in keys:
                        q[key] = ref_mean[key] + scale * (p[key] - ref_mean[key])
                for a_ in ((arm, "gen_on") if c == 0 else (arm,)):   # the first copy is also the gen_on arm (optimism check)
                    recs.append(dict(q, cell=cell, v0=V0, arm=a_, mapping="e1", prb=10, kin="sumo", beta=0.0, site=i,
                                     role="ref" if k < n_ref else "test", track=k, j=k, y_mean=-1.75))
    return K.use_statistic(recs, "d" if K.LEVER_D else "ls")


def replicate_rows(sites, rep, n_ref, tag_key):
    """One replicate: C1 rows, the optimism draws at the margin (whole-site scaling) and at R = 1, the sigma CI."""
    cell = f"PCOV{tag_key}{rep:03d}"                                # own bootstrap key per replicate and population
    rec1 = records(sites, n_ref, cell=cell)
    r1 = K.c1_cell(rec1, cell, V0, "e1", 10)["all"]
    r2 = K.c1_cell(records(sites, n_ref, CC["inflation"], cell=cell), cell, V0, "e1", 10)["all"]
    conf = bool(r2["R"] <= K.R_MAX and abs(r2["delta"]) <= K.D_MAX and max(r2["p_R"], r2["p_delta"]) <= K.ALPHA)
    om = K.optimism_draws(records(sites, n_ref, K.OC["margin"], cell=cell, whole_site=True), cell, V0, "e1", 10)[:3]
    o1 = K.optimism_draws(rec1, cell, V0, "e1", 10)[:3]
    T, _ = K.site_table(rec1, cell, V0, "gen_on", "e1", 10)
    ss = sorted(T)
    sb = [K.sigma(K.pooled(T, [ss[i] for i in idx], True)) for idx in K.draws(("COVSIG", cell), len(ss))]
    n_q = K.n_contributing(T, ss, True)
    sig = dict(sigma=K.sigma(K.pooled(T, ss, True)), ci=K.ci(sb), n=n_q, e=K.pooled(T, ss, True),
               estimable=bool(n_q >= K.OC["min_sites"] and np.all(np.isfinite(K.ci(sb)))))
    row = dict(rep=rep, R=r1["R"], R_ci=r1["R_ci"], R_inflated=r2["R"], confirmed_inflated=conf,
               cover=bool(r1["R_ci"][0] <= 1.0 <= r1["R_ci"][1]), n_sites_test_margin=int(om[2]))
    return row, om, o1, sig


def size_at(c, draws_, min_runs=None):
    """Size of the optimism check at the margin as stage 3 applies it: per
    replicate c_eff = max(c(n_rep), c); a replicate runs if R is finite and >= min_sites sites have test passes. With
    >= min_runs runnable replicates the size is conditional (over the runnable replicates), otherwise unconditional
    (over all replicates: a check that rarely runs can only rarely give 'not contradicted').
    Returns (size, runnable, conditional)."""
    min_runs = V5K7["min_runnable_replicates"] if min_runs is None else min_runs
    nc, runs = 0, 0
    for R, Rb, nn in draws_:
        if not np.isfinite(R) or nn < K.OC["min_sites"]:
            continue
        runs += 1
        c_eff = max(K.c_of_n(nn), c)
        nc += bool(R <= K.OC["margin"] and K.pval(K.widen(R, Rb, c_eff), lambda b: b >= K.OC["margin"]) <= K.OC["alpha"])
    if not draws_:
        return np.nan, 0, False
    cond = runs >= min_runs
    return (nc / runs if cond else nc / len(draws_)), runs, cond


def sigma_ci_coverage(sig, sig_pop, min_runs=None):
    """sigma-CI coverage over the replicates in which a sigma CI exists (>= min_sites qualified sites contributing
    passes, finite CI); with fewer than min_runnable_replicates such replicates it is not estimable.
    Returns (list of covered flags, estimable)."""
    min_runs = V5K7["min_runnable_replicates"] if min_runs is None else min_runs
    cov = [bool(x["ci"][0] <= sig_pop <= x["ci"][1]) for x in sig if x["estimable"]]
    return cov, len(cov) >= min_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="sens_coverage (pre-specified) or dev_<x>")
    ap.add_argument("--replicates", type=int, default=CC["replicates"])
    ap.add_argument("--sites-per-rep", type=int, default=CC["sites"], help="development only")
    ap.add_argument("--n-ref", type=int, default=CC["ref"], help="development only")
    ap.add_argument("--n-test", type=int, default=CC["test"], help="development only")
    a = ap.parse_args()
    check_tag(a.tag, stage=3)
    if not a.tag.startswith("dev_") and (a.replicates != CC["replicates"] or a.sites_per_rep != CC["sites"]
                                         or a.n_ref != CC["ref"] or a.n_test != CC["test"]):
        raise ValueError("pre-specified coverage check: constants criteria.C1.coverage_check")
    prov = provenance(a.tag, stage=3, replicates=a.replicates, sites_per_rep=a.sites_per_rep, n_ref=a.n_ref, n_test=a.n_test,
                      statistic="d" if K.LEVER_D else "ls")
    # the per-site cache is a stage-2 product (in-house chain + estimator; it stores both statistics), so a stage-3 rerun
    # recomputes the result from it without re-simulating
    prov_site = provenance(a.tag, stage=2, n_ref=a.n_ref, n_test=a.n_test)
    path = os.path.join(K.RESULTS, f"coverage_check_{a.tag}.json")
    if os.path.exists(path) and same_provenance(json.load(open(path)).get("provenance", {}), prov):
        print(path, "skipped (exists, same provenance)")
        return
    n_sites = CC["sites"] if not a.tag.startswith("dev_") else min(CC["sites"], a.sites_per_rep)
    K.B = CC["boot"]
    from scipy import stats as st
    cp = lambda k, n: [float(st.beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(st.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]
    margin = K.OC["margin"]
    res = {}
    for pop in POPS:
        out_dir = os.path.join(ER, "runs", f"coverage_{a.tag}" + ("" if pop == "a" else f"_pop{pop}"))
        os.makedirs(out_dir, exist_ok=True)
        rows, opt_m, opt_1, sig = [], [], [], []
        for rep in range(a.replicates):
            sites = pool(range(rep * n_sites, (rep + 1) * n_sites), a.n_ref, a.n_test, out_dir, prov_site, pop)
            row, om, o1, sg = replicate_rows(sites, rep, a.n_ref, pop)
            rows.append(row); opt_m.append(om); opt_1.append(o1); sig.append(sg)
        sig_pop = K.sigma(np.concatenate([x["e"] for x in sig]))           # population value of this design
        cov_sig, sig_est = sigma_ci_coverage(sig, sig_pop)
        n = len(rows)
        res[pop] = dict(population=POPS[pop], rows=rows, opt_m=opt_m, opt_1=opt_1,
                        n_sites_median=int(np.median([x[2] for x in opt_m])),
                        coverage_R1=float(np.mean([r["cover"] for r in rows])),
                        coverage_R1_cp95=cp(int(sum(r["cover"] for r in rows)), n),
                        false_confirmed_rate_R125=float(np.mean([r["confirmed_inflated"] for r in rows])),
                        false_confirmed_cp95=cp(int(sum(r["confirmed_inflated"] for r in rows)), n),
                        sigma_population=float(sig_pop), sigma_ci_runs=len(cov_sig), sigma_ci_estimable=sig_est,
                        sigma_ci_coverage=float(np.mean(cov_sig)) if sig_est else None,
                        sigma_ci_coverage_cp95=cp(int(sum(cov_sig)), len(cov_sig)) if cov_sig else None)
    c, steps, calibrated = 1.0, [], True
    while True:
        sz = {pop: size_at(c, r["opt_m"]) for pop, r in res.items()}
        steps.append(dict(c=float(c), **{f"size_{p}": float(v[0]) for p, v in sz.items()},
                          **{f"runs_{p}": int(v[1]) for p, v in sz.items()},
                          **{f"conditional_{p}": bool(v[2]) for p, v in sz.items()}))
        if all(v[0] <= K.OC["alpha"] for v in sz.values()):
            break
        if c >= V5K7["c_max"]:                          # pre-specified label, not an error
            calibrated = False
            break
        c = round(c + 0.05, 10)
    pops_out = {}
    for pop, r in res.items():
        pops_out[pop] = {k: v for k, v in r.items() if k not in ("opt_m", "opt_1")}
        sz_, runs_, cond_ = size_at(c, r["opt_m"])
        k_ = int(round(sz_ * (runs_ if cond_ else len(r["opt_m"]))))
        pops_out[pop].update(size_at_margin=sz_, runs_at_margin=runs_, size_conditional=cond_,
                             size_cp95=cp(k_, runs_ if cond_ else len(r["opt_m"])),
                             power_not_contradicted_R1=size_at(c, r["opt_1"])[0])
    widen = any(r["sigma_ci_estimable"] and r["sigma_ci_coverage"] < 0.93 for r in res.values())
    out = dict(tag=a.tag, spec=CC, populations=pops_out, widen_sigma_ci=bool(widen),
               optimism=dict(margin=margin, c_start=1.0, calibration_steps=steps, per_replicate="c_eff = max(c(n_rep), c)",
                             scaling="whole site (reference and test)"),
               c_calibrated=float(c) if calibrated else None, calibrated=calibrated,
               replicates=a.replicates, statistic="d" if K.LEVER_D else "ls",
               # single-population keys (population a)
               coverage_R1=res["a"]["coverage_R1"], sigma_ci_coverage=res["a"]["sigma_ci_coverage"],
               provenance=prov)
    os.makedirs(K.RESULTS, exist_ok=True)
    json.dump(out, open(path + ".tmp", "w"), indent=1, default=float)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: v for k, v in out.items() if k not in ("populations", "provenance")}, indent=1, default=float))


if __name__ == "__main__":
    main()
