# ==========================================================================
# REPLICATION PACKAGE — headline_robustness.py
# Tier: CORE
# Paper items: Table 3 (A. Wild cluster bootstrap row); Table 8 (Main estimate; inference (WCB)); Table 8 (Trends and timing: false post 2012, 2013); Table 8 (Distance: 300 km, 200 km, continuous exposure)
# Reads: output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/headline_robustness.txt
# Imports local modules: annual_firstborn_gender
# Run order: step 14 in run_all.sh
# ==========================================================================

"""Headline robustness battery for the paper's main result.

Headline: annual second-birth hazard DID, firstborn-DAUGHTER one-child
families, T x post (Post=2018) = -2.4pp (cluster p=0.024), from
scripts/annual_firstborn_gender.py. Son-first arm = built-in placebo.

Blocks (each run on daughter arm + son placebo arm):
  0. Baseline reproduction of the headline spec.
  1. Wild cluster bootstrap-t (Rademacher, null imposed, cluster=city,
     B=999) on T:post — implementation copied from youth_ideal_scarring.py
     (itself from gender_norm_robustness.py), coef name parameterized.
  2. Placebo shock years: pre-period data only (2009-2015), fake
     Post=2012 / Post=2013. Expect null.
  3. Continuous dose: hpr_exposure = max(0, 1 - dist/250) (rebuilt from
     hpr_treatment_city.csv exactly as data_process.py:498 defines it;
     panel_individual.csv does not carry the HPR columns), z-scored on the
     estimation sample; exposure x post. Prediction: negative.
  4. Alternative distance cutoffs: binary treatment rebuilt at 200 km and
     300 km from dist_to_hpr_km (regulated cities still dropped).

Sample/risk-set construction is reused VERBATIM by importing
build_hazard_panel from annual_firstborn_gender.py: ever-married women
20-45, at risk in year t iff exactly one child as of t-1, 2009-2022,
baseline-city treatment, regulated cities dropped, city + year FE,
age/duration controls, cluster by city.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annual_firstborn_gender import build_hazard_panel  # noqa: E402

CTRL = "age + I(age**2) + dur + I(dur**2)"
FE = " + C(city) + C(year)"

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def fit_did(d, treat_term):
    """OLS with city+year FE, cluster by city; return model."""
    f = f"y ~ {treat_term} + {CTRL}{FE}"
    return smf.ols(f, data=d).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["city"]})


def report(m, d, coef, label):
    b, se, p = m.params[coef], m.bse[coef], m.pvalues[coef]
    emit(f"  {label:<34} {b:+.4f}  (SE {se:.4f}, cluster p={p:.3f})"
         f"   N={int(m.nobs):,}  women={d['pid'].nunique():,}")
    return b, p


def wild_cluster_boot(d, formula, coef, reps=999, seed=90026):
    """WCB-t, Rademacher, null imposed, cluster=city (copied from
    youth_ideal_scarring.py; coefficient name parameterized)."""
    y_full, X_full = dmatrices(formula, d, return_type="dataframe")
    y = y_full.values.ravel()
    x = X_full[coef].values
    Z = X_full.drop(columns=coef).values
    cl = d["city"].values
    Q, _ = np.linalg.qr(Z)
    resid = lambda v: v - Q @ (Q.T @ v)
    xt = resid(x)
    u_r = resid(y)
    y_hat_r = y - u_r
    cities = np.unique(cl)
    G = len(cities)
    idx = [np.where(cl == c)[0] for c in cities]

    def tstat(yv):
        yt = resid(yv)
        b = xt @ yt / (xt @ xt)
        e = yt - xt * b
        meat = sum((xt[i] @ e[i]) ** 2 for i in idx)
        se = np.sqrt(meat * G / (G - 1)) / (xt @ xt)
        return b, b / se

    b0, t0 = tstat(y)
    rng = np.random.default_rng(seed)
    ts = np.empty(reps)
    for r in range(reps):
        w = rng.choice([-1.0, 1.0], size=G)
        yb = y_hat_r.copy()
        for gi, i in enumerate(idx):
            yb[i] += w[gi] * u_r[i]
        ts[r] = tstat(yb)[1]
    p = (np.sum(np.abs(ts) >= abs(t0)) + 1) / (reps + 1)
    return b0, t0, p, G


def main():
    df = build_hazard_panel()

    # re-attach dist_to_hpr_km (build_hazard_panel keeps only treat/regulated)
    hpr = pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
    df = df.merge(hpr[["city_code", "dist_to_hpr_km"]], on="city_code",
                  how="left")
    if df["dist_to_hpr_km"].isna().any():
        emit("WARNING: dist_to_hpr_km missing for "
             f"{df['dist_to_hpr_km'].isna().sum()} rows")
    # continuous exposure, exactly data_process.py:498
    df["hpr_exposure"] = (1 - df["dist_to_hpr_km"] / 250).clip(lower=0)

    df["post"] = (df["year"] >= 2018).astype(int)

    emit("=" * 78)
    emit("HEADLINE ROBUSTNESS BATTERY")
    emit("Annual second-birth hazard DID (one-child risk set, 2009-2022)")
    emit("Daughter-first arm = headline; son-first arm = placebo")
    emit("Conventions: baseline-city treatment, regulated cities dropped,")
    emit("city + year FE, age/dur quadratics, cluster by city, Post=2018")
    emit("=" * 78)
    emit(f"rows={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}  treated share={df['T'].mean():.3f}")

    arms = [(0, "DAUGHTER (headline)"), (1, "SON (placebo)")]

    # ---------------- block 0: baseline reproduction ----------------
    emit("\n" + "-" * 78)
    emit("[0] BASELINE: T x post, Post=2018 (must reproduce -2.4pp p=.024)")
    emit("-" * 78)
    base = {}
    for s, lab in arms:
        d = df[df["son"] == s]
        m = fit_did(d, "T:post")
        base[s] = report(m, d, "T:post", lab)

    b0, p0 = base[0]
    if not (abs(b0 - (-0.024)) < 0.003 and abs(p0 - 0.024) < 0.01):
        emit("\n*** BASELINE DID NOT REPRODUCE (-2.4pp, p=0.024) — STOPPING. ***")
        _save()
        sys.exit(1)
    emit("  -> baseline reproduces the headline; proceeding.")

    # ---------------- block 1: wild cluster bootstrap ----------------
    emit("\n" + "-" * 78)
    emit("[1] WILD CLUSTER BOOTSTRAP (Rademacher, null imposed, B=999,")
    emit("    cluster=city) on T:post, Post=2018")
    emit("-" * 78)
    for s, lab in arms:
        d = df[df["son"] == s]
        f = f"y ~ T:post + {CTRL}{FE}"
        m = fit_did(d, "T:post")
        b, t0, pw, G = wild_cluster_boot(d, f, "T:post")
        emit(f"  {lab:<34} {b:+.4f}  t={t0:+.2f}  "
             f"cluster p={m.pvalues['T:post']:.3f}  WCB p={pw:.4f}  (G={G})")

    # ---------------- block 2: placebo shock years ----------------
    emit("\n" + "-" * 78)
    emit("[2] PLACEBO SHOCK YEARS: pre-period only (2009-2015),")
    emit("    fake Post=2012 / Post=2013. Expect null.")
    emit("-" * 78)
    pre = df[df["year"] <= 2015].copy()
    for fake in (2012, 2013):
        pre["post"] = (pre["year"] >= fake).astype(int)
        emit(f"  fake Post = {fake}:")
        for s, lab in arms:
            d = pre[pre["son"] == s]
            m = fit_did(d, "T:post")
            report(m, d, "T:post", "  " + lab)

    # ---------------- block 3: continuous dose ----------------
    emit("\n" + "-" * 78)
    emit("[3] CONTINUOUS DOSE: hpr_exposure = max(0, 1 - dist/250),")
    emit("    z-scored on estimation sample; exposure x post, Post=2018.")
    emit("    Prediction: negative (more exposure -> more suppression).")
    emit("-" * 78)
    df["post"] = (df["year"] >= 2018).astype(int)
    mu, sd = df["hpr_exposure"].mean(), df["hpr_exposure"].std()
    df["expz"] = (df["hpr_exposure"] - mu) / sd
    emit(f"  hpr_exposure: mean={mu:.3f} sd={sd:.3f} "
         f">0 share={(df['hpr_exposure'] > 0).mean():.3f}")
    for s, lab in arms:
        d = df[df["son"] == s]
        m = fit_did(d, "expz:post")
        report(m, d, "expz:post", lab)

    # ---------------- block 4: alternative distance cutoffs ----------------
    emit("\n" + "-" * 78)
    emit("[4] ALTERNATIVE DISTANCE CUTOFFS: binary treatment rebuilt at")
    emit("    200 km and 300 km from dist_to_hpr_km (regulated already")
    emit("    dropped in the risk set); T x post, Post=2018.")
    emit("-" * 78)
    for cut in (200, 300):
        df["Talt"] = (df["dist_to_hpr_km"] <= cut).astype(int)
        emit(f"  cutoff = {cut} km  (treated share={df['Talt'].mean():.3f}):")
        for s, lab in arms:
            d = df[df["son"] == s]
            m = fit_did(d, "Talt:post")
            report(m, d, "Talt:post", "  " + lab)

    emit("\n" + "=" * 78)
    emit("Done.")
    _save()


def _save():
    out = os.path.join(OUT, "tables", "headline_robustness.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
