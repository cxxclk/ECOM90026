# ==========================================================================
# REPLICATION PACKAGE — child_health_wcb.py
# Tier: CORE
# Paper items: Table 5 (B. Wild cluster bootstrap row)
# Reads: (via imported modules)
# Writes: output/tables/child_health_wcb.txt
# Imports local modules: child_health_investment
# Run order: step 23 in run_all.sh
# ==========================================================================

"""T4 wrap-up: wild cluster bootstrap for the child health-investment result.

The batch-6 ticket (child_health_investment.py) found the Q->Q substitution
extends to child medical spending (daughter ihs_oop +0.529 cluster p=0.005,
son placebo clean, DDD p=0.024, late-deepening event study) but shipped
without WCB. This script closes that gap: WCB-t (Rademacher, null imposed,
cluster=city, 999 reps) on every headline coefficient, reusing the ticket's
panel builder verbatim via import (the module is __main__-guarded).

Coefficients bootstrapped:
  1. daughter ihs_oop  T:post   (headline)
  2. daughter ihs_tot  T:post
  3. DDD ihs_oop       T:post:son
  4. DDD ihs_tot       T:post:son
  5. son ihs_oop       T:post   (placebo symmetry)
  6. daughter ihs_oop  Post=2016 variant (cluster p=0.051 -- marginal)
  7. daughter ihs_oop  excl. hospitalized kids (robustness (a))
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from child_health_investment import build_panel, ROOT, OUT  # noqa: E402

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def wild_cluster_boot(d, formula, coef, reps=999, seed=90026):
    """WCB-t, Rademacher, null imposed, cluster=city (headline_robustness.py)."""
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
        b = yt @ xt / (xt @ xt)
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


def run(d, formula, coef, label):
    yv = formula.split("~")[0].strip()
    dd = d.dropna(subset=[yv]).copy()
    m = smf.ols(formula, data=dd).fit(cov_type="cluster",
                                      cov_kwds={"groups": dd["city"]})
    b, t, pw, G = wild_cluster_boot(dd, formula, coef)
    assert abs(b - m.params[coef]) < 1e-8, "WCB/OLS coefficient mismatch"
    emit(f"  {label:<44} {m.params[coef]:+.4f}  cluster p={m.pvalues[coef]:.3f}"
         f"  WCB p={pw:.4f}  (N={int(m.nobs):,}, G={G})")


def main():
    emit("WCB wrap-up for child_health_investment.py (999 reps, Rademacher,")
    emit("null imposed, cluster=city). Reuses the ticket's panel verbatim.")
    d = build_panel()
    oc = d[d["one_child_2015"]].copy()
    dgt, son = oc[oc["son"] == 0], oc[oc["son"] == 1]
    CTRL = " + C(age_c) + C(city) + C(year)"
    DDD = " ~ T:post + T:post:son + post:son + T:son + son" + CTRL

    emit("\nHeadline (one-child-as-of-2015 frame, Post=2018):")
    run(dgt, "ihs_oop ~ T:post" + CTRL, "T:post", "daughter ihs_oop")
    run(dgt, "ihs_tot ~ T:post" + CTRL, "T:post", "daughter ihs_tot")
    run(oc, "ihs_oop" + DDD, "T:post:son", "DDD ihs_oop  T:post:son")
    run(oc, "ihs_tot" + DDD, "T:post:son", "DDD ihs_tot  T:post:son")
    run(son, "ihs_oop ~ T:post" + CTRL, "T:post", "son ihs_oop (placebo)")

    emit("\nVariants:")
    v16 = dgt.copy()
    v16["post"] = (v16["year"] >= 2016).astype(int)
    run(v16, "ihs_oop ~ T:post" + CTRL, "T:post", "daughter ihs_oop, Post=2016")
    nh = dgt[dgt["hosp"] != 1]
    run(nh, "ihs_oop ~ T:post" + CTRL, "T:post", "daughter ihs_oop, excl. hosp")

    out = os.path.join(OUT, "tables", "child_health_wcb.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
