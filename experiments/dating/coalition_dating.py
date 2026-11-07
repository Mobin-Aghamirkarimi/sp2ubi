"""How precisely can the insurer alone, and an insurer + TSP coalition, date a single daily risk record under the
upload rule of the risk path (independent delay U[0, W) after the day ends)?

  * W sensitivity: W = 7, 14, 30 d, with common random numbers (the same uniform draw u gives the upload time
    d + 1 + u W for every W);
  * exact marginals by a forward-backward pass over a banded matching DP; feasible for W = 7 and 14 d. For W = 30 d
    the band covers the whole epoch and the exact posterior is not computed;
  * for every W, the exact support of each record's date given the matching constraint (the days d such that some
    complete assignment of the epoch's records to its days, consistent with every record's class and upload time,
    gives this record day d), from the Dulmage-Mendelsohn / SCC characterisation of the edges of a bipartite graph
    that lie in some perfect matching. Support size 1 = date determined (exact); log2(support) bounds the entropy
    from above;
  * the insurer alone: per-record dating probability under a uniform-within-class prior, with the classes the insurer
    decides from the decrypted score alone (score = w_0; event-only; weekday mileage), from the calendar of the epoch
    (28-31 days, every start weekday).
Model and assumptions: the coalition decrypts all six features (weekend miles > 0: weekend day; weekday miles > 0:
weekday; otherwise a non-driving day, which can be either); it sees each record's exact upload time; the delay density
is flat in [0, W); prior uniform over the assignments that respect the classes. Real driving habits are not used, so
the dating numbers are lower bounds on what the coalition can learn. 31-day epochs starting on a Monday, days 5 and 6
of each week are the weekend (the generator of sp2ubi/workload.py).

Run (from the repository root):  python experiments/dating/coalition_dating.py [--policies 300] [--seed 7]
Output: experiments/dating/results/coalition_dating.json (deterministic, seeded); reads results/record_classes.json.
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from sp2ubi.workload import daily_records, iter_policies  # noqa: E402

DATA = os.path.join(ROOT, "data", "telematics_syn-032021.csv")
OUT = os.path.join(HERE, "results", "coalition_dating.json")
DISCLOSURE = os.path.join(HERE, "results", "record_classes.json")
DAYS = 31
EXACT_POSTERIOR_MAX_W = 14


def classify(x):
    return "weekend" if x[3] > 0 else ("weekday" if (x[1] + x[2]) > 0 else "nondriving")


def feasible(t, cls, W):
    ds = [d for d in range(DAYS) if 0 <= t - (d + 1) < W]
    if cls == "weekend":
        ds = [d for d in ds if d % 7 >= 5]
    elif cls == "weekday":
        ds = [d for d in ds if d % 7 < 5]
    return ds


# ------------------------------------------------------------------ exact marginals (forward-backward over the matching DP)
def marginals(feas, los):
    """feas[j]: feasible days of record j (records sorted by upload time); los[j]: lowest day record j can reach
    (non-decreasing). State = bitmask of used days; before record j every day < los[j] must be used (later records
    cannot reach it). Returns (number of complete assignments, posterior[j][d])."""
    n = len(feas)
    full = (1 << DAYS) - 1
    F = [None] * (n + 1)
    F[0] = {0: 1}
    for j in range(n):
        low = (1 << los[j]) - 1
        nxt = {}
        for S, c in F[j].items():
            if S & low != low:
                continue
            for d in feas[j]:
                b = 1 << d
                if not S & b:
                    nxt[S | b] = nxt.get(S | b, 0) + c
        F[j + 1] = nxt
    total = F[n].get(full, 0)
    if total == 0:
        return 0, None
    Bn = {full: 1}
    post = [None] * n
    for j in range(n - 1, -1, -1):
        low = (1 << los[j]) - 1
        Bj, acc = {}, {}
        for S, c in F[j].items():
            if S & low != low:
                continue
            s = 0
            for d in feas[j]:
                b = 1 << d
                if not S & b:
                    v = Bn.get(S | b, 0)
                    if v:
                        s += v
                        acc[d] = acc.get(d, 0) + c * v
            if s:
                Bj[S] = s
        post[j] = {d: v / total for d, v in acc.items()}
        Bn = Bj
    return total, post


# ------------------------------------------------------------------ exact support (edges in some perfect matching)
def perfect_matching(adj, n):
    match_day = [-1] * DAYS

    def aug(j, seen):
        for d in adj[j]:
            if d not in seen:
                seen.add(d)
                if match_day[d] < 0 or aug(match_day[d], seen):
                    match_day[d] = j
                    return True
        return False
    for j in range(n):
        if not aug(j, set()):
            return None
    return match_day


def support(feas):
    """support[j] = set of days d such that edge (j, d) lies in some perfect matching (n = DAYS records)."""
    n = len(feas)
    md = perfect_matching(feas, n)
    assert md is not None
    mr = [None] * n
    for d, j in enumerate(md):
        mr[j] = d
    # directed graph: record j -> day d for non-matching edges, day d -> its matched record; nodes 0..n-1 records,
    # n..n+DAYS-1 days. Edge (j, d) not in M is in some perfect matching iff j and d share a strongly connected comp.
    N = n + DAYS
    g = [[] for _ in range(N)]
    for j in range(n):
        for d in feas[j]:
            if d != mr[j]:
                g[j].append(n + d)
    for d in range(DAYS):
        g[n + d].append(md[d])
    index, low, comp, stack, on = [None] * N, [0] * N, [None] * N, [], [False] * N
    counter = [0]
    ncomp = [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on[v] = True
        for w in g[v]:
            if index[w] is None:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on[w]:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            while True:
                w = stack.pop()
                on[w] = False
                comp[w] = ncomp[0]
                if w == v:
                    break
            ncomp[0] += 1
    for v in range(N):
        if index[v] is None:
            strong(v)
    return [{d for d in feas[j] if d == mr[j] or comp[j] == comp[n + d]} for j in range(n)]


def summarise(rows, exact):
    out = {}
    for cls, v in rows.items():
        if not v:
            continue
        n = len(v)
        sup = sorted(x["support"] for x in v)
        feas = sorted(x["feasible"] for x in v)
        rec = {"records": n,
               "share_date_determined": sum(s == 1 for s in sup) / n,
               "median_support_days_given_matching": sup[n // 2],
               "mean_log2_support_bits_upper_bound_on_entropy": sum(math.log2(s) for s in sup) / n,
               "median_feasible_days_own_window": feas[n // 2]}
        if exact:
            mx = sorted(x["max_post"] for x in v)
            H = sorted(x["H"] for x in v)
            rec.update({"median_max_posterior": mx[n // 2],
                        "share_max_posterior_ge_0.5": sum(m >= 0.5 for m in mx) / n,
                        "share_max_posterior_gt_0.999": sum(m > 0.999 for m in mx) / n,
                        "median_entropy_bits": H[n // 2], "mean_entropy_bits": sum(H) / n})
        out[cls] = rec
    return out


def coalition(policies, seed, Ws):
    rng = random.Random(seed)
    rows = list(iter_policies(DATA))
    rng.shuffle(rows)
    res = {W: {"weekday": [], "weekend": [], "nondriving": []} for W in Ws}
    for i, row in enumerate(rows[:policies]):
        recs = daily_records(row, days=DAYS, seed=seed * 100003 + i)
        draws = [rng.random() for _ in range(DAYS)]                 # common draws for every W
        for W in Ws:
            items = [(d + 1 + draws[d] * W, classify(x)) for d, x in enumerate(recs)]
            order = sorted(range(DAYS), key=lambda j: items[j][0])
            feas = [feasible(*items[j], W) for j in order]
            los = [max(0, math.floor(items[j][0] - 1 - W) + 1) for j in order]
            sup = support(feas)
            exact = W <= EXACT_POSTERIOR_MAX_W
            if exact:
                total, post = marginals(feas, los)
                assert total > 0
            for pos, j in enumerate(order):
                r = {"support": len(sup[pos]), "feasible": len(feas[pos])}
                if exact:
                    p = post[pos]
                    assert set(d for d, v in p.items() if v > 0) == sup[pos]        # DP and matching theory agree
                    r["max_post"] = max(p.values())
                    r["H"] = -sum(v * math.log2(v) for v in p.values() if v > 0)
                res[W][items[j][1]].append(r)
    return {f"W_{W:g}_days": dict(summarise(res[W], W <= EXACT_POSTERIOR_MAX_W), W_days=W,
                                  naive_entropy_bits_uniform_W=math.log2(W),
                                  exact_posterior=W <= EXACT_POSTERIOR_MAX_W) for W in Ws}


def insurer_alone():
    """The insurer alone sees the decrypted score of each record, not its time. Classes decided from the score
    alone (record_classes.py): score = w_0 (a non-driving day or an event-free weekend drive: any day of the
    epoch); event-only (hard events without weekday mileage: a weekend drive in the data, since weekday drives carry
    weekday miles); weekday mileage (a weekday). Uniform-within-class prior: P(date guessed exactly) = 1 / (days of
    the class in the epoch)."""
    disc = json.load(open(DISCLOSURE))["classes_from_score_alone_share"]
    cal = {}
    for L in (28, 29, 30, 31):
        we = sorted({sum(((s + d) % 7) >= 5 for d in range(L)) for s in range(7)})
        wd = sorted({L - n for n in we})
        cal[f"epoch_{L}_days"] = {
            "weekend_days_range": [we[0], we[-1]], "weekdays_range": [wd[0], wd[-1]],
            "P_exact_score_eq_w0": 1 / L, "P_exact_event_only": [1 / we[-1], 1 / we[0]],
            "P_exact_weekday_mileage": [1 / wd[-1], 1 / wd[0]],
            "entropy_bits": {"score_eq_w0": math.log2(L), "event_only": [math.log2(we[0]), math.log2(we[-1])],
                             "weekday_mileage": [math.log2(wd[0]), math.log2(wd[-1])]}}
    return {"record_share_per_class": disc, "per_epoch_length": cal,
            "generator_epoch_31_days_from_monday": {"weekend_days": 8, "weekdays": 23,
                                                    "P_exact": {"score_eq_w0": 1 / 31, "event_only": 1 / 8,
                                                                "weekday_mileage": 1 / 23}},
            "note": "uniform within class (no driving-habit prior), so these are lower bounds; the order of the "
                    "records carries no information (global shuffle); compare 1/D for a class-blind guess"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--W", type=float, nargs="+", default=[7.0, 14.0, 30.0])
    a = ap.parse_args()
    out = {"policies": a.policies,
           "seed": a.seed, "epoch_days": DAYS, "log2_30_factorial": math.log2(math.factorial(30)),
           "coalition_insurer_plus_TSP": coalition(a.policies, a.seed, a.W), "insurer_alone": insurer_alone()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
