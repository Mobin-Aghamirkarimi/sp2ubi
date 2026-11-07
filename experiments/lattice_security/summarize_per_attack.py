"""Aggregate results/per_attack/*.json into results/estimate_summary.json: per case the log2 rop of every attack, the
minimum over the completed attacks, and the attacks that did not complete (timeout or job stop), stated explicitly."""
import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cases = ["single_key_ternary"]
attacks = ["usvp", "bdd", "dual", "dual_hybrid", "bkw", "arora-gb", "bdd_hybrid", "bdd_mitm_hybrid"]
out = {"estimator_commit": None, "sigma": 3.19, "cases": {}}
for c in cases:
    row, missing = {}, []
    for a in attacks:
        f = f"results/per_attack/{c}__{a}.json"
        if not os.path.exists(f):
            missing.append(f"{a}: not completed (job stopped by the time limit)"); continue
        d = json.load(open(f))
        out["estimator_commit"] = d.get("estimator_commit") or out["estimator_commit"]
        if d.get("error"):
            missing.append(f"{a}: {d['error']}")
        elif d.get("infeasible"):
            row[a] = "infeasible (rop = +inf)"
        else:
            row[a] = d["log2_rop"]
    nums = [v for v in row.values() if isinstance(v, (int, float))]
    out["cases"][c] = {"log2_rop": row, "min_log2_rop_completed": min(nums), "not_completed": missing}
json.dump(out, open("results/estimate_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
