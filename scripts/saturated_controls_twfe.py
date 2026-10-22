# ==========================================================================
# REPLICATION PACKAGE — saturated_controls_twfe.py
# Tier: CORE
# Paper items: Table 8 (Controls: age × duration cell FE)
# Reads: (via imported modules)
# Writes: output/tables/saturated_controls_twfe.txt
# Imports local modules: annual_firstborn_gender, headline_robustness
# Run order: step 30 in run_all.sh
# ==========================================================================

"""
Saturated-controls / heterogeneity-robust check of the headline TWFE DID.

Motivation (BBMT "When is TSLS actually LATE?"; Sloczynski 2022;
Goldsmith-Pinkham, Hull & Kolesar 2024): with heterogeneous effects, a
regression with NON-saturated controls need not return a convex average of
cell-level effects, and a regression with several treatment terms can suffer
contamination bias. The headline design has common timing + never-treated
controls, so Goodman-Bacon / dCDH negative weights do not arise; the
remaining exposure is (i) the linear age/dur quadratics and (ii) the pooled
DDD with common controls.

Blocks
  [0] baseline (age, age^2, dur, dur^2)  -- must reproduce -2.38pp p=.024
  [1] age FE + dur FE (each saturated, additively)
  [2] age x dur cell FE (fully saturated interaction, single years)
  [3] [2] + WCB
  [4] effect heterogeneity by age-bin x dur-bin cell: T:post interacted with
      each cell, aggregated with treated-post-count (ATT) weights, delta-method
      SE; reports min/max weight sign (all >= 0 by construction) and the
      implicit OLS weights of the pooled coefficient on those cells
  [5] contamination check: pooled DDD with common controls vs split-sample
      difference vs fully interacted pooled DDD

Conventions as headline: one-child risk set 2009-22, baseline city,
regulated dropped, city + year FE, cluster by city, Post=2018.
dur is top-coded at 25 for the dummy specs (13 women carry impossible
durations from roster child-year errors).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annual_firstborn_gender import build_hazard_panel  # noqa: E402
from headline_robustness import wild_cluster_boot  # noqa: E402

FE = " + C(city) + C(year)"
CTRL_Q = "age + I(age**2) + dur + I(dur**2)"

LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def fit(d, rhs):
    return smf.ols(f"y ~ {rhs}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})


def rep(m, coef, label):
    emit(f"  {label:<44} {m.params[coef]:+.4f}  (SE {m.bse[coef]:.4f}, "
         f"p={m.pvalues[coef]:.3f})  N={int(m.nobs):,}")


def main():
    df = build_hazard_panel()
    df["post"] = (df["year"] >= 2018).astype(int)
    df["Tpost"] = df["T"] * df["post"]
    df["age_i"] = df["age"].astype(int)
    df["dur_c"] = df["dur"].clip(upper=25).astype(int)
    df["cell"] = df["age_i"].astype(str) + "_" + df["dur_c"].astype(str)
    df["abin"] = pd.cut(df["age"], [19, 24, 29, 34, 39, 45],
                        labels=["20-24", "25-29", "30-34", "35-39", "40-45"])
    df["dbin"] = pd.cut(df["dur"], [0, 2, 4, 7, 12, 999],
                        labels=["1-2", "3-4", "5-7", "8-12", "13+"])
    df["hcell"] = df["abin"].astype(str) + "|" + df["dbin"].astype(str)

    emit("=" * 78)
    emit("SATURATED CONTROLS / HETEROGENEITY-ROBUST CHECK OF THE HEADLINE DID")
    emit("one-child annual second-birth hazard, Post=2018, city+year FE, "
         "cluster city")
    emit("=" * 78)

    arms = [(0, "DAUGHTER"), (1, "SON (placebo)")]
    for s, lab in arms:
        d = df[df["son"] == s].copy()
        emit(f"\n### {lab} arm: rows={len(d):,} women={d['pid'].nunique():,} "
             f"cities={d['city'].nunique()}  age x dur cells={d['cell'].nunique()}")
        emit("-" * 78)
        rep(fit(d, f"Tpost + {CTRL_Q}{FE}"), "Tpost", "[0] baseline quadratics")
        rep(fit(d, f"Tpost + C(age_i) + C(dur_c){FE}"), "Tpost",
            "[1] age FE + dur FE")
        m2 = fit(d, f"Tpost + C(cell){FE}")
        rep(m2, "Tpost", "[2] age x dur cell FE (saturated)")
        rep(fit(d, f"Tpost + C(cell) + C(year):C(abin){FE}"), "Tpost",
            "[2b] cell FE + year x age-bin FE")

        # [3] WCB on saturated spec
        b, t0, pw, G = wild_cluster_boot(d, f"y ~ Tpost + C(cell){FE}", "Tpost")
        emit(f"  [3] WCB on [2]: {b:+.4f} t={t0:+.2f} WCB p={pw:.4f} (G={G})")

        # [4] cell-specific effects, ATT-weighted aggregation
        cells = sorted(d.loc[d["Tpost"] == 1, "hcell"].unique())
        names = []
        for c in cells:
            nm = "D_" + c.replace("|", "_").replace("-", "_").replace("+", "p")
            d[nm] = d["Tpost"] * (d["hcell"] == c)
            names.append(nm)
        m4 = fit(d, " + ".join(names) + f" + C(cell){FE}")
        n_tp = d.loc[d["Tpost"] == 1].groupby("hcell").size().reindex(cells)
        w = (n_tp / n_tp.sum()).values
        bvec = m4.params[names].values
        V = m4.cov_params().loc[names, names].values
        att = w @ bvec
        se = np.sqrt(w @ V @ w)
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(att / se)))
        emit(f"  [4] cell-heterogeneous T:post, ATT-weighted "
             f"({len(cells)} age-bin x dur-bin cells):")
        emit(f"      {att:+.4f}  (SE {se:.4f}, p={p:.3f})  "
             f"weights all >= 0 by construction")

        # implicit OLS weights of the pooled Tpost on the same cells:
        # FWL — residualize Tpost on the controls/FE, weight_c ∝ sum over
        # treated-post rows in c of the residual (Sloczynski-style)
        from patsy import dmatrices
        _, X = dmatrices(f"y ~ Tpost + C(cell){FE}", d, return_type="dataframe")
        x = X["Tpost"].values
        Z = X.drop(columns="Tpost").values
        Q, _ = np.linalg.qr(Z)
        xt = x - Q @ (Q.T @ x)
        # contribution of cell c to beta = sum_{i in c} xt_i * y_i / sum xt^2;
        # with cell-constant effect tau_c among Tpost rows, beta = sum_c w_c tau_c
        # where w_c = sum_{i in c, Tpost=1} xt_i / sum_i xt_i x_i
        denom = xt @ x
        wo = np.array([xt[(d["hcell"].values == c) & (x == 1)].sum() / denom
                       for c in cells])
        emit(f"      implicit OLS weights on the same cells: min {wo.min():+.3f} "
             f"max {wo.max():+.3f} sum {wo.sum():.3f} "
             f"(negative cells: {(wo < 0).sum()})")
        emit(f"      corr(OLS weight, ATT weight) = {np.corrcoef(wo, w)[0,1]:.3f}")
        tab = pd.DataFrame({"n_TP": n_tp.values, "w_ATT": w, "w_OLS": wo,
                            "tau_c": bvec,
                            "p": m4.pvalues[names].values}, index=cells)
        emit("      cell detail (tau_c = cell-specific T:post):")
        for c, r in tab.iterrows():
            emit(f"        {c:<13} nTP={int(r.n_TP):4d} wATT={r.w_ATT:.3f} "
                 f"wOLS={r.w_OLS:+.3f}  tau={r.tau_c:+.4f} (p={r.p:.2f})")

    # [5] contamination check
    emit("\n" + "-" * 78)
    emit("[5] CONTAMINATION CHECK (GHK 2024): son differential three ways")
    emit("-" * 78)
    d0 = df[df["son"] == 0]
    d1 = df[df["son"] == 1]
    b0 = fit(d0, f"Tpost + {CTRL_Q}{FE}").params["Tpost"]
    b1 = fit(d1, f"Tpost + {CTRL_Q}{FE}").params["Tpost"]
    emit(f"  split-sample difference (son - daughter)      {b1 - b0:+.4f}")
    mp0 = fit(df, f"Tpost + Tpost:son + T:son + son + {CTRL_Q}{FE}")
    rep(mp0, "Tpost:son", "pooled DDD, no post:son (misspecified)")
    mp = fit(df, f"Tpost + Tpost:son + post:son + T:son + son + {CTRL_Q}{FE}")
    rep(mp, "Tpost:son", "pooled DDD, common controls/FE (headline)")
    mp2 = fit(df, f"Tpost + Tpost:son + T:son + C(year):son + {CTRL_Q}{FE}")
    rep(mp2, "Tpost:son", "pooled DDD, + year x son FE")
    mf = fit(df, f"Tpost + Tpost:son + T:son + son + ({CTRL_Q}):son"
                 f" + C(city):son + C(year):son{FE}")
    rep(mf, "Tpost:son", "pooled DDD, fully interacted (= split)")
    ms = fit(df, f"Tpost + Tpost:son + T:son + C(year):son + C(cell):C(son){FE}")
    rep(ms, "Tpost:son", "pooled DDD, year x son + cell x son")

    with open(os.path.join(OUT, "tables", "saturated_controls_twfe.txt"),
              "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
