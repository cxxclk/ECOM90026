# ==========================================================================
# REPLICATION PACKAGE — causal_forest_hazard.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Causal forest)
# Reads: output/analysis_data.csv
# Writes: output/tables/causal_forest_hazard.txt
# Imports local modules: annual_firstborn_gender
# Run order: step 37 in run_all.sh
# ==========================================================================

"""Causal forest v2 -- DiD-aware, on the correct risk set.

v1 (causal_forest_het.py) used a post-only wave-level cross-section: T then
compares treated vs control CITIES in levels (confounded by age/income
composition), not the DiD suppression. Age and income dominated and the
daughter cell came out positive -- a confound, not a refutation.

This version fixes both flaws:
  * Risk set = the one-child annual second-birth hazard panel (parity fixed by
    construction -> the margin our story is about).
  * Treatment of interest = exposure = T x post (treated city AND post-2018),
    with city + year dummies in W so the forest estimates heterogeneity in the
    DiD effect, not in cross-city levels.
  * X = firstborn son + woman covariates. Question: does the forest, with no
    prior, pick firstborn gender as a top heterogeneity driver and make the
    daughter CATE the most negative?
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "output")
from annual_firstborn_gender import build_hazard_panel  # noqa: E402

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def main():
    df = build_hazard_panel()
    df["post"] = (df["year"] >= 2018).astype(int)
    df["exposure"] = (df["T"] * df["post"]).astype(int)

    # baseline woman covariates from analysis_data (earliest non-missing)
    ad = pd.read_csv(os.path.join(OUT, "analysis_data.csv"),
                     usecols=["pid", "year", "eduy", "log_income", "urban",
                              "homeowner_d"])
    ad = ad.sort_values(["pid", "year"]).groupby("pid").first().reset_index()
    ad = ad.drop(columns=["year"])
    df = df.merge(ad, on="pid", how="left")

    feats = ["son", "age", "dur", "eduy", "log_income", "urban", "homeowner_d"]
    df = df.dropna(subset=feats + ["y", "exposure"])
    emit(f"DiD-aware causal forest on one-child 2nd-birth hazard panel")
    emit(f"  woman-years N={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}")
    emit(f"  exposure share={df['exposure'].mean():.3f}  "
         f"hazard mean={df['y'].mean():.4f}  son share={df['son'].mean():.3f}")

    Y = df["y"].to_numpy(float)
    Tt = df["exposure"].to_numpy(float)
    X = df[feats].to_numpy(float)
    W = pd.get_dummies(df[["city", "year"]].astype(int),
                       columns=["city", "year"], drop_first=True).to_numpy(float)

    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=30,
                                      random_state=0),
        model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=30,
                                       random_state=0),
        discrete_treatment=True, n_estimators=2000, min_samples_leaf=20,
        random_state=0)
    cf.fit(Y, Tt, X=X, W=W)

    emit("\n" + "=" * 64)
    emit("(1) Variable importance for DiD-effect heterogeneity")
    for f, v in sorted(zip(feats, cf.feature_importances_), key=lambda x: -x[1]):
        emit(f"  {f:<16} {v:.3f}")

    emit("\n" + "=" * 64)
    ate = cf.ate(X); lo, hi = cf.ate_interval(X, alpha=0.10)
    emit(f"(2) overall ATE = {ate:+.4f}  (90% CI {lo:+.4f}, {hi:+.4f})")
    df["cate"] = cf.effect(X)
    df["fb"] = np.where(df["son"] == 0, "daughter", "son")
    emit("    mean CATE by firstborn gender:")
    emit(df.groupby("fb")["cate"].agg(["mean", "count"]).round(4).to_string())

    emit("\n(3) Mean CATE by covariate tercile")
    for f in ["age", "eduy", "log_income"]:
        df["_t"] = pd.qcut(df[f], 3, labels=["lo", "mid", "hi"],
                           duplicates="drop")
        m = df.groupby("_t")["cate"].mean().round(4)
        emit(f"  {f:<12} " + "  ".join(f"{k}={v:+.4f}" for k, v in m.items()))

    out = os.path.join(OUT, "tables", "causal_forest_hazard.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
