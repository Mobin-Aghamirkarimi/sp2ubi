"""Risk model for SP2UBI on the So, Boucher & Valdez (2021) synthetic telematics dataset.

Protocol constraint: the insurer only learns daily scores R_d = w . x_d and sums them over the epoch, so the expected
claim count must be linear in the epoch totals of per-day features. We use an additive Poisson (identity-link) model
with disjoint mileage categories, fitted by maximum likelihood under non-negativity constraints:

    E[N | x] = c(z) * sum_d ( b_0 * 1 + sum_i b_i * x_{i,d} ),   b >= 0,

where x_0 = 1 counts one insured day, and c(z) = exp(gamma . z) is an optional per-policy factor built from
traditional rating variables z that the insurer already knows and applies in plaintext after decryption (the
encrypted part stays linear). The negative log-likelihood is convex in b (for fixed gamma), so the fit is unique and
every prediction is non-negative by construction.

Outputs (results/): data dictionary, CV metrics, coefficients with standard errors, calibration deciles, fixed-point
quantization analysis and the integer model used by the reference implementation (models/).
Usage (from the repository root, environment requirements/riskmodel.lock):  python experiments/risk_model/fit_risk_model.py
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "telematics_syn-032021.csv")
OUT = os.path.join(HERE, "results")
SEED, FOLDS = 20261002, 5

# --------------------------------------------------------------------------------------------- features
DICTIONARY = {
    "days": ("Duration", "x_0: one per insured day (intercept slot)"),
    "offpeak_miles": ("Pct.drive.wkday * M - rush_miles", "weekday miles outside the am/pm rush hours"),
    "rush_miles": ("min(Pct.drive.rush am + Pct.drive.rush pm, Pct.drive.wkday) * M",
                   "weekday miles in the am/pm rush hours (capped at weekday miles: 141 inconsistent records)"),
    "weekend_miles": ("Pct.drive.wkend * M", "miles on Saturday/Sunday"),
    "hard_brakes": ("Brake.08miles * M / 1000", "decelerations >= 8 mph/s (Brake.08miles = count per 1000 miles)"),
    "hard_accels": ("Accel.08miles * M / 1000", "accelerations >= 8 mph/s (Accel.08miles = count per 1000 miles)"),
}
FEATURES = list(DICTIONARY)            # x_0 .. x_5; M = Total.miles.driven
# Integer units sent by the OBU per day (0.1 mile for distances).
UNITS = {"days": 1, "offpeak_miles": 0.1, "rush_miles": 0.1, "weekend_miles": 0.1, "hard_brakes": 1,
         "hard_accels": 1}
# Numerical conditioning only (coefficients are reported in raw units).
FIT_SCALE = {"days": 365.0, "offpeak_miles": 1000.0, "rush_miles": 1000.0, "weekend_miles": 1000.0,
             "hard_brakes": 100.0, "hard_accels": 100.0}
TRAD_NUM = ["Insured.age", "Credit.score", "Years.noclaims", "Car.age"]
TRAD_CAT = ["Insured.sex", "Marital", "Region", "Car.use"]


def load():
    df = pd.read_csv(DATA)
    M = df["Total.miles.driven"]
    rush_share = np.minimum(df["Pct.drive.rush am"] + df["Pct.drive.rush pm"], df["Pct.drive.wkday"])
    X = pd.DataFrame({
        "days": df["Duration"].astype(float),
        "offpeak_miles": (df["Pct.drive.wkday"] - rush_share) * M,
        "rush_miles": rush_share * M,
        "weekend_miles": df["Pct.drive.wkend"] * M,
        "hard_brakes": df["Brake.08miles"] * M / 1000,
        "hard_accels": df["Accel.08miles"] * M / 1000,
    })
    assert np.allclose(X[["offpeak_miles", "rush_miles", "weekend_miles"]].sum(axis=1), M, rtol=1e-9, atol=1e-6)
    num = df[TRAD_NUM].astype(float)
    Z = pd.concat([(num - num.mean()) / num.std(),
                   pd.get_dummies(df[TRAD_CAT], drop_first=True).astype(float)], axis=1)
    inconsistent = int(((df["Pct.drive.rush am"] + df["Pct.drive.rush pm"]) > df["Pct.drive.wkday"] + 1e-12).sum())
    return df, X, Z, inconsistent


# --------------------------------------------------------------------------------------------- model
class NonNegAdditivePoisson:
    """mu_i = exp(Z_i gamma) * (X_i b), b >= 0. L-BFGS-B on the exact negative log-likelihood."""

    def __init__(self, use_traditional=False):
        self.use_trad = use_traditional

    def fit(self, X, y, Z=None):
        self.cols = list(X.columns)
        s = np.array([FIT_SCALE[c] for c in self.cols])
        Xs = X.to_numpy() / s
        Zs = Z.to_numpy() if (self.use_trad and Z is not None) else np.zeros((len(y), 0))
        k, q = Xs.shape[1], Zs.shape[1]
        y = np.asarray(y, float)

        def nll(theta):
            b, g = theta[:k], theta[k:]
            lin = Xs @ b
            fac = np.exp(Zs @ g) if q else 1.0
            mu = np.maximum(fac * lin, 1e-300)
            val = np.sum(mu - y * np.log(mu))
            r = 1.0 - y / mu                                   # d nll / d mu
            grad_b = Xs.T @ (r * fac)
            grad_g = Zs.T @ (r * mu) if q else np.zeros(0)
            return val, np.concatenate([grad_b, grad_g])

        b0 = np.zeros(k)
        b0[self.cols.index("days")] = y.sum() / Xs[:, self.cols.index("days")].sum()
        b0 += 1e-6
        res = minimize(nll, np.concatenate([b0, np.zeros(q)]), jac=True, method="L-BFGS-B",
                       bounds=[(0.0, None)] * k + [(None, None)] * q,
                       options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-15, "gtol": 1e-10})
        if not res.success:
            raise RuntimeError(f"optimizer did not converge: {res.message}")
        self.b_scaled, self.gamma = res.x[:k], res.x[k:]
        self.scale = s
        self.coef = pd.Series(self.b_scaled / s, index=self.cols)     # raw units
        self.gamma_names = list(Z.columns) if q else []
        # standard errors from the observed information (free parameters only: b > 0 and all gamma)
        mu = self._mu(Xs, Zs)
        fac = np.exp(Zs @ self.gamma) if q else np.ones(len(y))
        J = np.hstack([Xs * fac[:, None], Zs * mu[:, None]])
        info = J.T @ (J * (y / mu ** 2)[:, None])
        free = np.concatenate([self.b_scaled > 1e-12, np.ones(q, bool)])
        se = np.full(k + q, np.nan)
        try:
            se[free] = np.sqrt(np.diag(np.linalg.inv(info[np.ix_(free, free)])))
        except np.linalg.LinAlgError:
            pass
        self.se = pd.Series(se[:k] / s, index=self.cols)
        self.at_bound = [c for c, f in zip(self.cols, free[:k]) if not f]
        self.nll = res.fun
        return self

    def _mu(self, Xs, Zs):
        lin = Xs @ self.b_scaled
        return (np.exp(Zs @ self.gamma) if len(self.gamma) else 1.0) * lin

    def predict(self, X, Z=None):
        Xs = X.to_numpy() / self.scale
        Zs = Z.to_numpy() if (self.use_trad and Z is not None) else np.zeros((len(X), 0))
        return self._mu(Xs, Zs)


# --------------------------------------------------------------------------------------------- metrics
def poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-12, None)
    return 2.0 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu))


def gini(y, mu, expo):
    o = np.argsort(mu / expo, kind="stable")
    cy = np.concatenate([[0], np.cumsum(y[o]) / y.sum()])
    ce = np.concatenate([[0], np.cumsum(expo[o]) / expo.sum()])
    return 1.0 - 2.0 * np.trapezoid(cy, ce)


def cross_validate(df, X, Z):
    y = df["NB_Claim"].to_numpy(float)
    expo = (df["Duration"] / 365).to_numpy(float)
    L = sm.add_constant(pd.DataFrame({c: np.log1p(X[c]) for c in FEATURES if c != "days"}))
    specs = {
        "additive_nonneg_poisson (protocol model)": ("add", False),
        "additive_nonneg_poisson x exp(traditional) (protocol model + insurer-side factor)": ("add", True),
        "ols_on_counts (additive, unconstrained)": ("ols", False),
        "loglink_glm_log1p_totals (reference, NOT additive)": ("log", False),
    }
    rows = []
    for name, (kind, trad) in specs.items():
        dev = null = 0.0
        ginis, nonpos = [], 0
        for tr, te in KFold(FOLDS, shuffle=True, random_state=SEED).split(X):
            if kind == "add":
                m = NonNegAdditivePoisson(trad).fit(X.iloc[tr], y[tr], Z.iloc[tr])
                mu = m.predict(X.iloc[te], Z.iloc[te])
            elif kind == "ols":
                mu = sm.OLS(y[tr], X.iloc[tr]).fit().predict(X.iloc[te]).to_numpy()
            else:
                m = sm.GLM(y[tr], L.iloc[tr], family=sm.families.Poisson(), offset=np.log(expo[tr])).fit()
                mu = m.predict(L.iloc[te], offset=np.log(expo[te])).to_numpy()
            nonpos += int(np.sum(mu <= 0))
            lam0 = y[tr].sum() / expo[tr].sum()
            dev += poisson_deviance(y[te], mu)
            null += poisson_deviance(y[te], lam0 * expo[te])
            ginis.append(gini(y[te], np.clip(mu, 1e-12, None), expo[te]))
        rows.append({"model": name, "D2_out_of_fold": 1 - dev / null, "gini_mean": float(np.mean(ginis)),
                     "gini_std": float(np.std(ginis)), "nonpositive_predictions": nonpos})
    return pd.DataFrame(rows)


def quantize(coef, w_bits):
    """w_i = round(beta_i * unit_i * S), S = power of two with max |w_i| < 2^w_bits."""
    per_unit = {c: coef[c] * UNITS[c] for c in coef.index}
    S = 2.0 ** np.floor(np.log2(((1 << w_bits) - 1) / max(abs(v) for v in per_unit.values())))
    return {c: int(round(v * S)) for c, v in per_unit.items()}, S


def main():
    os.makedirs(OUT, exist_ok=True)
    df, X, Z, inconsistent = load()
    y = df["NB_Claim"].to_numpy(float)
    expo = (df["Duration"] / 365).to_numpy(float)
    pd.DataFrame([{"feature": k, "derivation": v[0], "meaning": v[1], "integer_unit": UNITS[k]}
                  for k, v in DICTIONARY.items()]).to_csv(os.path.join(OUT, "data_dictionary.csv"), index=False)
    X.describe(percentiles=[0.5, 0.99, 0.999]).T.to_csv(os.path.join(OUT, "feature_summary.csv"))

    cv = cross_validate(df, X, Z)
    cv.to_csv(os.path.join(OUT, "cv_metrics.csv"), index=False)
    print(f"records with rush share > weekday share (rush capped): {inconsistent}")
    print(cv.round(4).to_string(index=False))

    final = NonNegAdditivePoisson(False).fit(X, y, Z)
    pd.DataFrame({"coef_raw_units": final.coef, "std_err": final.se}).to_csv(
        os.path.join(OUT, "protocol_model_coefficients.csv"))
    print("\nprotocol model coefficients (expected claims per raw unit), at bound:", final.at_bound)
    print(pd.DataFrame({"coef": final.coef, "se": final.se}).to_string())

    mu = final.predict(X)
    q = pd.qcut(mu / expo, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"decile": q, "obs": y, "pred": mu, "expo": expo}).groupby("decile").sum()
    cal["obs_per_year"], cal["pred_per_year"] = cal.obs / cal.expo, cal.pred / cal.expo
    cal.to_csv(os.path.join(OUT, "calibration_deciles.csv"))

    lam0 = y.sum() / expo.sum()
    d_null = poisson_deviance(y, lam0 * expo)
    Xint = pd.DataFrame({c: np.round(X[c] / UNITS[c]) for c in FEATURES})
    rows = []
    for w_bits in (8, 12, 16, 20, 22):
        w, S = quantize(final.coef, w_bits)
        mu_q = (Xint.to_numpy() @ np.array([w[c] for c in FEATURES])) / S
        # weight-quantization error only: same integer features, float weights vs integer weights
        mu_int_float = Xint.to_numpy() @ np.array([final.coef[c] * UNITS[c] for c in FEATURES])
        rel_w = np.abs(mu_q - mu_int_float) / mu_int_float
        rows.append({"w_bits": w_bits, "scale_S": S, "max_rel_error_weights_only": rel_w.max(),
                     "mean_rel_error_weights_only": rel_w.mean(),
                     "D2_float": 1 - poisson_deviance(y, mu) / d_null,
                     "D2_quantized": 1 - poisson_deviance(y, mu_q) / d_null, "weights": json.dumps(w)})
    qdf = pd.DataFrame(rows)
    qdf.to_csv(os.path.join(OUT, "quantization.csv"), index=False)
    print("\n" + qdf.drop(columns="weights").to_string(index=False))

    w, S = quantize(final.coef, 22)
    json.dump({"model": "additive non-negative Poisson (identity link) for NB_Claim, So et al. (2021)",
               "feature_order": FEATURES, "units": UNITS, "w_bits": 22, "scale_S": S, "weights": w,
               "max_daily_feature_values": {c: float(np.ceil((X[c] / UNITS[c] / df["Duration"]).max())) for c in FEATURES},
               "interpretation": "expected claims over the epoch = sum_d sum_i w_i x_{i,d} / scale_S"},
              open(os.path.join(OUT, "integer_model.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
