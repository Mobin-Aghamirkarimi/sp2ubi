"""Study 2: the stage-2 fields of the development-stage candidates and diagnostics, computed beside the study-1 fields
of every record by pipeline/s2_evaluate.py, which runs pipeline/stage2_evaluate.evaluate_site unchanged (stage-2 re-runs
from the stage-1 records: tags dev_s2_cal and dev_s2) and adds these fields afterwards. Every field is RSU-observable unless marked "truth
(reporting only)"; no VSS datum enters a field (the VSS speed enters only the numerator of b̂). Every field is
computed identically for honest and manipulated (beta) passes and for every arm.

Fields (all new):
  rho_fit          ρ̂ of the pass's tied offset LS fit (fit_pass_offset); L-B′ (N18) and E-e read it
  acc_fit          E-c: p[2], the fitted acceleration of that fit (m/s^2)
  stop_abs         E-d, the RSU-only low-speed-run (stop) detector (N12; pre-specified span only): the
                   smallest score in Hz for which the detector fires: min over runs of consecutive received beacons whose
                   air times span >= 1 s, lying wholly inside the inter-head span (FDOA bracket 0.25 max |FDOA - δ̂|
                   ∪ x̂ in [0, D]), of the run's max |FDOA - δ̂|. Stage 3 detects iff stop_abs < k_stop x the
                   site's survey rms_ref_med (s2_criteria.stop_detected); the threshold is set by free-road survey
                   passes, so it is a level of |FDOA - δ̂| that a slow steady crawl can also stay below (stated in the
                   supplement). No stage-2 verdict is stored: the stop decision is made at stage 3 only.
  c_prof, c_match, c_n
                   E-f (N17): the study-1 sums over the core beacons x̂ in [-0.05 D, 1.05 D] of the full-window fit
  n_beacons        the number of beacons of the record's fit (D0.3: E-f core share = c_n / n_beacons)
  w_mse            the ±4 s window fit's mse (reported only)
  blk_drop_max_db  L-B: the largest drop of the head-A received power per beacon (10 log10 mean |y_A|^2 over the beacon's
                   samples) below the pass's own fitted large-scale trend a + b log10 r̂_A (r̂_A from the tied fit's
                   x̂ and ρ̂), dB. No entry under L-B if blk_drop_max_db > Δ_blk (N13) (stage-3 inclusion, pass level).
  blk_drop_med5_db reported only (D0.6): the largest 5-beacon running median of the same drop
  prof_2s, match_2s[prof_2s_T; match_2s_T]
                   E-e: the profile / matched statistics of the two-segment, ρ-fixed fit (fdoa.estimator
                   .fit_pass_two_segment), ρ fixed per lane from the site's survey ρ̂ (median per survey lane over the
                   included survey passes, same variant; the survey vehicle's lane is known by design); a pass takes the
                   survey lane whose ρ̂ is nearest its own ρ̂_fit (RSU-side; the test pass's SUMO lane is never used).
                   Reading F: survey = role "ref" as recorded; reading T (Dev-2 only, surrogate=True): survey = test
                   passes jr < 30.
  v_min_window     truth (reporting only, D0.1): the true minimum speed inside the window [-0.05 L, 1.05 L]
  g_share_window   truth (reporting only, D0.3): share of the total |Δg| of the pass covered by the ±4 s window
  junction_x       truth (reporting only, D0.1): the SUMO junction position of the site's track (info.junction_x)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ER = os.path.dirname(HERE)
sys.path.insert(0, ER)
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
import site_accuracy as E1  # noqa: E402
from fdoa.estimator import _g, fit_pass_offset, fit_pass_two_segment, two_segment_track  # noqa: E402
from fdoa.geometry import LAMBDA  # noqa: E402
from pipeline import s2_constants as S2  # noqa: E402

VERSION = "study2-fields-2"          # version of the study-2 fields, stored in every record


# ---------------------------------------------------------------------------------------------- tied fit
def tied_fit(t_b, fd, site):
    """The study-1 tied offset fit: p = (x_mid, v_mid, acc, rho), δ̂, t_mid, residuals."""
    q, tm, resid, ok = fit_pass_offset(t_b, fd, site)
    return q[:4], float(q[4]), tm, resid, bool(ok)


def fitted_x(p, t_b, tm):
    tau = t_b - tm
    return p[0] + p[1] * tau + 0.5 * p[2] * tau ** 2


# ---------------------------------------------------------------------------------------------- E-d detector
def run_windows(t_b, run_s=S2.N12_STOP_RUN_S):
    """For each beacon i the smallest j >= i with t_j - t_i >= run_s (or -1): the shortest run of consecutive beacons
    starting at i that spans run_s. A longer run starting at i only raises the run's maximum, so these suffice."""
    j = np.searchsorted(t_b, t_b + run_s - 1e-12, side="left")
    return np.where(j < t_b.size, j, -1)


def bracket_fit(xhat, D):
    """'Inside [0, D]' from the tied fit's x̂."""
    return (xhat >= 0.0) & (xhat <= D)


def bracket_fdoa(fd, delta, frac=S2.N12_BRACKET_FRAC):
    """N12: RSU-observable head bracketing from the pass's own FDOA curve (inside the heads |g| >~ 1): the
    inter-head span runs from the first to the last beacon with |FDOA - δ̂| >= frac x its maximum; a
    stop inside the heads lies within it."""
    a = np.abs(fd - delta)
    inside = np.zeros(fd.size, bool)
    if fd.size and np.isfinite(a).any():
        hi = np.flatnonzero(a >= frac * np.nanmax(a))
        inside[hi[0]:hi[-1] + 1] = True
    return inside


def bracket_span(fd, delta, xhat, D):
    """N12: the pre-specified inter-head span: the FDOA bracket ∪ {x̂ in [0, D]}."""
    return bracket_fdoa(fd, delta) | bracket_fit(xhat, D)


def core_statistics(t_b, fd, vx, p, tm, delta, D, arc):
    """N17, E-f: the study-1 profile / matched sums restricted to the core set C = {i : x̂(τ_i) in [arc0 D, arc1 D]}
    of the full-window tied fit (same p, δ̂, tm; no refit): c_prof = Σ_C v̂ v_sen / Σ_C v̂² - 1, c_match =
    Σ_C g² v_sen / (λ Σ_C g f̃) - 1, f̃ = f - δ̂. Fewer than 12 core beacons -> NaN (no entry)."""
    tau = t_b - tm
    xh = p[0] + p[1] * tau + 0.5 * p[2] * tau ** 2
    c = (xh >= arc[0] * D) & (xh <= arc[1] * D)
    if c.sum() < S2.N17_CORE_MIN_BEACONS:
        return np.nan, np.nan, int(c.sum())
    vf = p[1] + p[2] * tau
    g = _g(xh, D, p[3])
    fdc = fd - delta
    with np.errstate(invalid="ignore", divide="ignore"):
        return (float(np.sum(vf[c] * vx[c]) / np.sum(vf[c] ** 2) - 1),
                float(np.sum(g[c] ** 2 * vx[c]) / (LAMBDA * np.sum(g[c] * fdc[c])) - 1), int(c.sum()))


def stop_scores(t_b, fd, delta, rms, xhat, D, run_s=S2.N12_STOP_RUN_S, inside=None):
    """Per start beacon i: the max of |FDOA - δ̂| / RMS over the run [i, j(i)] if the whole run lies inside the head
    bracket (default: x̂ in [0, D]), else +inf. The detector fires at k if some score < k. rms = 1 gives the score in Hz
    (normalised at stage 3 by the site scale rms_ref_med)."""
    n = t_b.size
    sc = np.full(n, np.inf)
    if n == 0 or not np.isfinite(rms) or rms <= 0:
        return sc
    r = np.abs(fd - delta) / rms
    inside = bracket_fit(xhat, D) if inside is None else inside
    jj = run_windows(t_b, run_s)
    for i in range(n):
        j = jj[i]
        if j < 0:
            break
        if inside[i:j + 1].all():
            sc[i] = r[i:j + 1].max()
    return sc


def stop_crit(scores):
    return float(np.min(scores)) if scores.size else float("inf")


# ---------------------------------------------------------------------------------------------- L-B
def head_power_db(yb_head):
    """Received power per beacon at one head: 10 log10 of the mean |y|^2 over the beacon's samples (RSU-observable)."""
    return 10 * np.log10(np.maximum(np.mean(np.abs(yb_head) ** 2, axis=-1), 1e-300))


def blk_drop_max_db(p_db, xhat, rho_hat):
    """The largest drop (dB) of the per-beacon power below the pass's own fitted large-scale trend a + b log10 r̂_A,
    r̂_A = sqrt(x̂^2 + ρ̂^2) (head A at x = 0). NaN with fewer than 3 beacons."""
    if p_db.size < 3:
        return float("nan")
    lr = np.log10(np.sqrt(xhat ** 2 + rho_hat ** 2))
    A = np.column_stack([np.ones_like(lr), lr])
    coef, *_ = np.linalg.lstsq(A, p_db, rcond=None)
    return float(np.max(A @ coef - p_db))


def blk_drop_med5_db(p_db, xhat, rho_hat, k=5):
    """Reported only (D0.6): the largest k-beacon running median of the drop below the same fitted trend
    (one beacon is shorter than the coherence time, so the single-beacon maximum is dominated by small-scale dips)."""
    if p_db.size < max(3, k):
        return float("nan")
    lr = np.log10(np.sqrt(xhat ** 2 + rho_hat ** 2))
    A = np.column_stack([np.ones_like(lr), lr])
    coef, *_ = np.linalg.lstsq(A, p_db, rcond=None)
    d = A @ coef - p_db
    return float(np.max(np.median(np.lib.stride_tricks.sliding_window_view(d, k), axis=1)))


# ---------------------------------------------------------------------------------------------- E-e
def stats_two_segment(t_b, fd, vx, site, rho):
    """Profile and matched statistics of the two-segment, ρ-fixed fit."""
    if not np.isfinite(rho) or t_b.size < 12:
        return np.nan, np.nan
    q, tm, _, ok = fit_pass_two_segment(t_b, fd, site, rho)
    if not ok or not np.all(np.isfinite(q)):
        return np.nan, np.nan
    xh, vf = two_segment_track(q, t_b - tm)
    g = _g(xh, site.D, rho)
    fdc = fd - q[5]
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = float(np.sum(vf * vx) / np.sum(vf * vf) - 1)
        match = float(np.sum(g * g * vx) / (LAMBDA * np.sum(g * fdc)) - 1)
    return prof, match


def lane_of(y_mean, lane_width):
    return int(round(-y_mean / lane_width))


def included_s(r):
    """stage3_criteria.included (the RSU-observable inclusion: window rule, finite statistics) without importing criteria."""
    return bool(r.get("window_ok", True) and r.get("finite", np.isfinite(r.get("match", np.nan))
                                                    and np.isfinite(r.get("prof", np.nan))))


def rho_table(survey_recs, lane_width):
    """Survey ρ̂ per lane: median ρ̂_fit over the included survey passes of the lane (the survey vehicle's lane is known;
    as for L-B′, only included survey passes)."""
    by = {}
    for r in survey_recs:
        if included_s(r) and np.isfinite(r.get("rho_fit", np.nan)):
            by.setdefault(lane_of(r["y_mean"], lane_width), []).append(r["rho_fit"])
    return {ln: float(np.median(v)) for ln, v in by.items()}


def rho_for(rec, table):
    """RSU-side lane assignment: the survey lane whose ρ̂ is nearest the pass's own ρ̂_fit."""
    if not table or not np.isfinite(rec.get("rho_fit", np.nan)):
        return np.nan
    return min(table.values(), key=lambda v: abs(v - rec["rho_fit"]))


def rho_lane_offset(recs, is_survey, lane_width):
    """L-B′ (N18): per record, min over the site's surveyed lanes ℓ of
    |ρ̂ - ρ̂_survey,ℓ|, ρ̂ = rho_fit, ρ̂_survey,ℓ = the median rho_fit of the site's survey passes in lane ℓ (same family,
    cell, speed, arm, mapping, PRB); for a survey pass the median of its own lane EXCLUDES the pass itself
    (leave-one-out). Survey passes: is_survey(r) (the caller's inclusion rule); their lane from y_mean (the survey
    vehicle's lane is known by design); a test pass's lane is never used. lane_width: callable r -> lane width (m).
    Returns a list of floats aligned with recs (NaN: no finite ρ̂, or no surveyed lane)."""
    grp = lambda r: (r.get("family"), r["site"], r["cell"], r["v0"], r["arm"], r["mapping"], r["prb"])
    members = {}
    for i, r in enumerate(recs):
        if is_survey(r) and np.isfinite(r.get("rho_fit", np.nan)):
            members.setdefault(grp(r), {}).setdefault(lane_of(r["y_mean"], lane_width(r)), []).append((i, float(r["rho_fit"])))
    out = []
    for i, r in enumerate(recs):
        rho = r.get("rho_fit", np.nan)
        best = np.nan
        if np.isfinite(rho):
            for lane in (members.get(grp(r)) or {}).values():
                vals = [v for j, v in lane if j != i]
                if vals:
                    d = abs(rho - float(np.median(vals)))
                    best = d if not np.isfinite(best) else min(best, d)
        out.append(float(best))
    return out


# ---------------------------------------------------------------------------------------------- the stage-2 hook
class S2Fields:
    """The study-2 fields of one site's stage-2 run (in record order; with the beacons of the record's fit; `attach` adds the fields to
    the finished records). only_roles: restrict the passes (dev_s2_cal: ("ref")); surrogate: also the reading-T E-e fields (Dev-2 only); lane_width: the
    scenario's lane width (stage3_criteria.lane_width rule); win_arc: the window rule's arc (constants window_rule.arc_L)."""

    def __init__(self, only_roles=None, surrogate=False, lane_width=3.5, win_arc=(-0.05, 1.05)):
        self.only_roles = tuple(only_roles) if only_roles else None
        self.surrogate = bool(surrogate)
        self.lane_width = float(lane_width)
        self.win_arc = tuple(float(a) for a in win_arc)
        self._stash = []

    def knobs(self):
        return dict(version=VERSION, only_roles=list(self.only_roles) if self.only_roles else None,
                    surrogate=self.surrogate, lane_width=self.lane_width, win_arc=list(self.win_arc),
                    e_d_span=S2.OPTIONS["e_d_bracket"], e_d_bracket_frac=S2.N12_BRACKET_FRAC)

    def wants(self, m):
        return self.only_roles is None or m["role"] in self.only_roles

    def core(self, t_b, fd, vx, x, rho_true, survey, yb):
        """The per-record fields from the record's beacons (t_b, FDOA, VSS speed, true position, true ρ, the survey
        geometry, the received samples yb [2, n_b, M] of both heads)."""
        out = dict(rho_fit=np.nan, acc_fit=np.nan, blk_drop_max_db=np.nan, blk_drop_med5_db=np.nan, g_share_window=np.nan,
                   c_prof=np.nan, c_match=np.nan, c_n=0, w_mse=np.nan, stop_abs=np.nan)
        if t_b.size >= 6:
            p, delta, tm, resid, ok = tied_fit(t_b, fd, survey)
            xh = fitted_x(p, t_b, tm)
            out["stop_abs"] = stop_crit(stop_scores(t_b, fd, delta, 1.0, xh, survey.D,
                                                    inside=bracket_span(fd, delta, xh, survey.D)))
            pa = head_power_db(yb[0])
            out.update(rho_fit=float(p[3]), acc_fit=float(p[2]), blk_drop_max_db=blk_drop_max_db(pa, xh, p[3]),
                       blk_drop_med5_db=blk_drop_med5_db(pa, xh, p[3]))
            tc = tm + (survey.D / 2 - p[0]) / max(p[1], 0.5)
            g0 = np.abs(np.diff(_g(x, survey.D, rho_true)))
            inwin = np.abs(t_b - tc) <= E1.HALF_WINDOW_S
            out["g_share_window"] = float(g0[inwin[1:]].sum() / g0.sum()) if g0.sum() > 0 else np.nan
            if inwin.sum() >= 12:                            # the window fit's mse (reported only)
                out["w_mse"] = float(E1.statistics(t_b[inwin], fd[inwin], vx[inwin], x[inwin], rho_true[inwin], survey)[4])
            out["c_prof"], out["c_match"], out["c_n"] = core_statistics(t_b, fd, vx, p, tm, delta, survey.D, self.win_arc)
        inw = (x >= self.win_arc[0] * survey.D) & (x <= self.win_arc[1] * survey.D)
        out["v_min_window"] = float(np.min(vx[inw])) if inw.any() else np.nan
        out["n_beacons"] = int(t_b.size)                    # D0.3: the E-f core share = c_n / n_beacons
        self._stash.append((out, t_b, fd, vx, survey))
        return out

    def attach(self, recs, meta):
        """Add the fields to the finished records (one `core` call per record, same order), the junction distance from
        the stage-1 meta, then E-e on the whole site (ρ per lane from the survey of the same variant)."""
        if len(recs) != len(self._stash):
            raise RuntimeError(f"study-2 fields: {len(self._stash)} fits for {len(recs)} records")
        for r, (f, *_ ) in zip(recs, self._stash):
            clash = sorted(set(f) & set(r))
            if clash:
                raise RuntimeError(f"study-2 field names clash with study-1 fields: {clash}")
            r.update(f)
            head = next((q for q in meta if q["cell"] == r["cell"] and q["v0"] == r["v0"] and q.get("sumo")), None)
            jx = (head["sumo"].get(r["role"]) or {}).get("junction_x") if head else None
            r["junction_x"] = None if jx is None else float(jx)
        arrays = [a[1:] for a in self._stash]
        groups = {}
        for i, r in enumerate(recs):
            groups.setdefault((r["cell"], r["v0"], r["arm"], r["mapping"], r["prb"]), []).append(i)
        for idx in groups.values():
            rs = [recs[i] for i in idx]
            surveys = {"": [r for r in rs if r["role"] == "ref" and r["kin"] == "sumo" and r.get("beta", 0.0) == 0.0]}
            if self.surrogate:
                surveys["_T"] = [r for r in rs if r["role"] == "test" and r["kin"] == "sumo" and r.get("beta", 0.0) == 0.0
                                 and r["jr"] is not None and r["jr"] < S2.SURROGATE_SPLIT_JR]
            tables = {suf: rho_table(sv, self.lane_width) for suf, sv in surveys.items()}
            for i in idx:
                t_b, fd, vx, survey = arrays[i][:4]
                for suf, tab in tables.items():
                    recs[i]["prof_2s" + suf], recs[i]["match_2s" + suf] = stats_two_segment(t_b, fd, vx, survey,
                                                                                            rho_for(recs[i], tab))
        self._stash = []
        return recs


# ---------------------------------------------------------------------------------------------- study-1 field check
def record_key(r):
    return (r["site"], r["track"], r["arm"], r["mapping"], r["prb"], r.get("beta", 0.0))


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def study1_field_check(new_recs, old_recs):
    """In every dev_s2 output the study-1 fields equal those of the existing records of the
    same passes. Returns the list of mismatches [(key, field)]; a missing pass is a mismatch (field None). Only the
    passes present in new_recs are checked (dev_s2_cal holds the reference passes only)."""
    old = {record_key(r): r for r in old_recs}
    bad = []
    for r in new_recs:
        k = record_key(r)
        o = old.get(k)
        if o is None:
            bad.append((k, None))
            continue
        for f, v in o.items():
            if f == "family":
                continue
            if f not in r or not _same(r[f], v):
                bad.append((k, f))
    return bad
