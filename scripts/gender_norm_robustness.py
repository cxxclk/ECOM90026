# ==========================================================================
# REPLICATION PACKAGE — gender_norm_robustness.py
# Tier: CORE
# Paper items: Table 4 (A. High third wild bootstrap p, drop Henan); Table B2 (Drop one province)
# Reads: CFPS .dta files in data/20xx/
# Writes: output/tables/gender_norm_robustness.txt
# Imports local modules: annual_firstborn_gender, gender_norm_index
# Run order: step 17 in run_all.sh
# ==========================================================================

"""Robustness for the gender-norm moderator (gender_norm_index.py):
  1. Province composition of the high-norm top tercile (is it one province?)
  2. Drop-one-province re-estimation of the top-tercile daughter T:post
  3. Wild cluster bootstrap-t (Rademacher, null imposed, cluster = city)
     for the thin-cell headline coefficients.

Note the user's framing: single-province driven would NOT be an automatic
kill — it would say that province's norm environment is special. We report
composition first, then sensitivity.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from patsy import dmatrices

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gender_norm_index import build_norm_index  # noqa: E402
from annual_firstborn_gender import build_hazard_panel  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)

CTRL = "age + I(age**2) + dur + I(dur**2)"
FE = " + C(city) + C(year)"
FORM = f"y ~ T:post + {CTRL}{FE}"


def cluster_fit(d, label):
    m = smf.ols(FORM, data=d).fit(cov_type="cluster",
                                  cov_kwds={"groups": d["city"]})
    emit(f"  {label:<28} T:post {m.params['T:post']:+.4f} "
         f"(p={m.pvalues['T:post']:.3f})  N={int(m.nobs):,} "
         f"cities={d['city'].nunique()}")
    return m


def wild_cluster_boot(d, reps=4999, seed=90026):
    """Wild cluster bootstrap-t for T:post, null imposed, Rademacher weights.
    FWL: cluster-robust t for one coefficient equals the t from the
    partialled-out regression (up to dof scaling, constant across reps)."""
    y_full, X_full = dmatrices(FORM, d, return_type="dataframe")
    y = y_full.values.ravel()
    x = X_full["T:post"].values
    Z = X_full.drop(columns="T:post").values
    cl = d["city"].values

    Q, _ = np.linalg.qr(Z)
    resid = lambda v: v - Q @ (Q.T @ v)
    xt = resid(x)
    u_r = resid(y)          # restricted residuals (null beta=0 imposed)
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
    hits = 0
    for _ in range(reps):
        w = rng.choice([-1.0, 1.0], size=G)
        wl = np.empty(len(y))
        for g, i in enumerate(idx):
            wl[i] = w[g]
        _, t = tstat(y_hat_r + wl * u_r)
        if abs(t) >= abs(t0):
            hits += 1
    return b0, t0, (hits + 1) / (reps + 1), G


def main():
    city = build_norm_index()
    df = build_hazard_panel()
    df = df.merge(city, on="city_code", how="inner")
    df["post"] = (df["year"] >= 2018).astype(int)

    xw, _ = pyreadstat.read_dta(
        os.path.join(DATA, "CFPS_cleaned", "顺序码匹配.dta"))
    prov = (xw[["city", "provname"]].drop_duplicates("city")
            .rename(columns={"city": "city_code"}))
    df = df.merge(prov, on="city_code", how="left")

    cities_in = df.drop_duplicates("city_code")

    for normv in ["son_norm", "trad_pc1"]:
        hi_cut = cities_in[normv].quantile(2 / 3)
        d = df[(df["son"] == 0) & (df[normv] > hi_cut)].copy()

        emit("\n" + "=" * 72)
        emit(f"TOP TERCILE by {normv} — daughter arm")
        emit("\nProvince composition (woman-years; treated/control cities):")
        comp = d.groupby("provname").agg(
            wy=("y", "size"),
            cities_T=("city_code", lambda s: d.loc[s.index]
                      .query("T==1")["city_code"].nunique()),
            cities_C=("city_code", lambda s: d.loc[s.index]
                      .query("T==0")["city_code"].nunique()),
            treated_share=("T", "mean"),
        ).sort_values("wy", ascending=False)
        emit(comp.to_string())

        emit("\nFull top-tercile estimate:")
        cluster_fit(d, "ALL provinces")

        emit("\nDrop-one-province:")
        for p in comp.index:
            sub = d[d["provname"] != p]
            if sub["T"].nunique() < 2 or sub["post"].nunique() < 2:
                emit(f"  drop {p:<24} (degenerate: no T/C variation left)")
                continue
            cluster_fit(sub, f"drop {p} ({comp.loc[p, 'wy']} wy)")

        emit("\nWild cluster bootstrap (Rademacher, null imposed, 4999 reps):")
        b, t, pw, G = wild_cluster_boot(d)
        emit(f"  T:post = {b:+.4f}  t = {t:+.2f}  "
             f"WCB p = {pw:.3f}  (clusters G={G})")

    out = os.path.join(OUT, "tables", "gender_norm_robustness.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
