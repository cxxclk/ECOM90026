# ==========================================================================
# REPLICATION PACKAGE — income_gradient_did.py
# Tier: HELPER (imported by other scripts, not run directly)
# Paper items: none directly (support module)
# Reads: CFPS .dta files in data/20xx/, output/panel_individual.csv
# Writes: output/tables/income_gradient_did.txt
# Imports local modules: gender_norm_index, semisup_typology
# Run order: not run directly
# ==========================================================================

"""Confirmatory distributional income DID: the traditional-poor gradient.

Motivation (2026-07-21): the distributional family-income effect is the ONE
signal that recurs under EVERY level-space classification method — semisup
scan (only BH-FDR survivor, 4 classifications), and now the response-profile
typology (the only outcome dimension whose stratum variation survives EB
shrinkage: urban daughter/higher-income strata GAIN, traditional-poor strata
LOSE). Agreed 2026-07-16, run now: a CONFIRMATORY continuous-gradient DID.

Design: ever-married women universe (unregulated prefectures, baseline-city
treatment), wave-level outcome log(1+family income p.c.) 2010-2022, deduped
to one obs per household-wave; city+year FE, age controls, cluster by city;
Post=2018.

Gradient tpz ("traditional-poor") = mean of available z-components:
  +z(city trad_pc1)   traditional-culture prefecture
  -z(log faminc p.c., 2014 baseline)   poor household
  +z(rural)   rural residence
(>= 2 components required). Robustness: tpz_nm EXCLUDING the baseline-income
component (kills the mean-reversion objection — interacting a level outcome
with its own baseline invites regression to the mean), plus each component's
triple separately.

Inference battery on T:post:tpz — wave event study (base 2014), tercile
split with pre-trend gates, wild cluster bootstrap-t (Rademacher, null
imposed), drop-one-province. Secondary outcome: log house value (the
gainers' mirror: does the top end gain wealth while the bottom loses
income?). Deflation note: pipeline fam_income is fincome1 (nominal, all
waves) — year FE absorb price levels.
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
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from semisup_typology import build_baseline               # noqa: E402
from gender_norm_index import build_norm_index            # noqa: E402

WAVES = [2010, 2012, 2014, 2016, 2018, 2020, 2022]
LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def zs(x):
    return (x - x.mean()) / x.std()


def wcb(d, formula, coef, reps=4999, seed=90026):
    """Wild cluster bootstrap-t (Rademacher, null imposed) for one coef."""
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


def cfit(formula, d, keys, label):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  households={d['fid'].nunique():,}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    base = build_baseline()
    trad = build_norm_index()[["city_code", "trad_pc1"]].rename(
        columns={"trad_pc1": "city_trad"})
    base = base.merge(trad, on="city_code", how="left")

    z_trad = zs(base["city_trad"])
    z_poor = -zs(base["log_faminc_pc"])
    z_rur = zs(1 - base["urban"])
    comp = pd.concat([z_trad, z_poor, z_rur], axis=1)
    comp.columns = ["z_trad", "z_poor", "z_rur"]
    base["tpz"] = comp.mean(axis=1).where(comp.notna().sum(axis=1) >= 2)
    base["tpz_nm"] = pd.concat([z_trad, z_rur], axis=1).mean(axis=1).where(
        z_trad.notna() & z_rur.notna())
    for c in ["z_trad", "z_poor", "z_rur"]:
        base[c] = comp[c]
    base = base[base["tpz"].notna()]
    emit("CONFIRMATORY DISTRIBUTIONAL INCOME DID — traditional-poor gradient")
    emit(f"universe: ever-married women, unregulated prefectures; "
         f"{len(base):,} women with gradient defined")
    q = base["tpz"].quantile([1 / 3, 2 / 3]).values
    base["ter"] = np.searchsorted(q, base["tpz"]) + 1

    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "fid", "year", "birthy",
                               "fam_income_pc", "house_value"])
    keep = ["pid", "treat_hpr", "city_code", "tpz", "tpz_nm", "ter",
            "z_trad", "z_poor", "z_rur"]
    d = pan.merge(base[keep], on="pid", how="inner")
    d = d[d["year"].isin(WAVES)].drop_duplicates(["fid", "year"])
    d["linc"] = np.log1p(d["fam_income_pc"].clip(lower=0))
    d["log_hv"] = np.log1p(d["house_value"].where(d["house_value"] > 0))
    d["age"] = d["year"] - d["birthy"]
    d["T"] = d["treat_hpr"].astype(int)
    d["city"] = d["city_code"].astype(int)
    d["post"] = (d["year"] >= 2018).astype(int)
    d = d[d["linc"].notna() & d["age"].between(18, 60)]
    emit(f"household-wave obs: {len(d):,}  cities={d['city'].nunique()}  "
         f"treated share={d['T'].mean():.3f}")

    FE = " + C(city) + C(year)"
    CT = "age + I(age**2)"
    TRIPLE = f"linc ~ T:post + T:post:tpz + T:tpz + post:tpz + tpz + {CT}{FE}"

    # ---- headline continuous triple -------------------------------------
    m = cfit(TRIPLE, d, ["T:post", "T:post:tpz"],
             "HEADLINE: continuous traditional-poor gradient x income DID")
    # robustness: gradient without the baseline-income component
    cfit(f"linc ~ T:post + T:post:tpz_nm + T:tpz_nm + post:tpz_nm + tpz_nm "
         f"+ {CT}{FE}", d[d["tpz_nm"].notna()], ["T:post", "T:post:tpz_nm"],
         "no-money gradient (trad+rural only; kills mean-reversion story)")
    # component horse-race
    for zc in ["z_trad", "z_poor", "z_rur"]:
        cfit(f"linc ~ T:post + T:post:{zc} + T:{zc} + post:{zc} + {zc} "
             f"+ {CT}{FE}", d[d[zc].notna()], [f"T:post:{zc}"],
             f"component: {zc}")
    joint = ("linc ~ T:post + "
             + " + ".join(f"T:post:{z} + T:{z} + post:{z} + {z}"
                          for z in ["z_trad", "z_poor", "z_rur"]) + f" + {CT}{FE}")
    cfit(joint, d.dropna(subset=["z_trad", "z_poor", "z_rur"]),
         [f"T:post:{z}" for z in ["z_trad", "z_poor", "z_rur"]],
         "all three components jointly")

    # ---- terciles with pre-trend gates ----------------------------------
    emit("\nTercile DIDs (T3 = most traditional-poor):")
    for t in (1, 2, 3):
        s = d[d["ter"] == t].copy()
        m1 = smf.ols(f"linc ~ T:post + {CT}{FE}", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["city"]})
        for yr in [2010, 2012]:
            s[f"Ty{yr}"] = s["T"] * (s["year"] == yr).astype(int)
        s["Tpost"] = s["T"] * s["post"]
        mp = smf.ols(f"linc ~ Ty2010 + Ty2012 + Tpost + {CT}{FE}",
                     data=s).fit(cov_type="cluster",
                                 cov_kwds={"groups": s["city"]})
        viol = sum(mp.pvalues[f"Ty{yr}"] < 0.10 for yr in [2010, 2012])
        emit(f"  T{t}: DID {m1.params['T:post']:+.4f} "
             f"(p={m1.pvalues['T:post']:.3f})  N={int(m1.nobs):,}  "
             f"pre-viol {viol}/2")

    # ---- event study of the gradient (base 2014) ------------------------
    emit("\nEvent study of T x wave x tpz (base 2014):")
    s = d.copy()
    terms = []
    for yr in [w for w in WAVES if w != 2014]:
        s[f"Ty{yr}"] = s["T"] * (s["year"] == yr).astype(int)
        s[f"yz{yr}"] = (s["year"] == yr).astype(int) * s["tpz"]
        s[f"Tyz{yr}"] = s[f"Ty{yr}"] * s["tpz"]
        terms += [f"Ty{yr}", f"yz{yr}", f"Tyz{yr}"]
    me = smf.ols(f"linc ~ {' + '.join(terms)} + T:tpz + tpz + {CT}{FE}",
                 data=s).fit(cov_type="cluster",
                             cov_kwds={"groups": s["city"]})
    for yr in [w for w in WAVES if w != 2014]:
        tag = " (pre)" if yr < 2016 else ""
        emit(f"  T x {yr} x tpz   {me.params[f'Tyz{yr}']:+.4f}  "
             f"(SE {me.bse[f'Tyz{yr}']:.4f}, "
             f"p={me.pvalues[f'Tyz{yr}']:.3f}){tag}")

    # ---- WCB + drop-one-province on the headline triple -----------------
    emit("\nWild cluster bootstrap (4999 reps, Rademacher, null imposed):")
    b, t, p, G = wcb(d, TRIPLE, "T:post:tpz")
    emit(f"  T:post:tpz = {b:+.4f}  t={t:.2f}  WCB p={p:.4f}  (G={G})")

    xw, _ = pyreadstat.read_dta(
        os.path.join(DATA, "CFPS_cleaned", "顺序码匹配.dta"))
    prov = (xw[["city", "provname"]].drop_duplicates("city")
            .rename(columns={"city": "city_code"}))
    d = d.merge(prov, on="city_code", how="left")
    emit("\nDrop-one-province (headline triple):")
    rows = []
    for pv in sorted(d["provname"].dropna().unique()):
        s = d[d["provname"] != pv]
        if s["T"].nunique() < 2:
            continue
        try:
            m1 = smf.ols(TRIPLE, data=s).fit(
                cov_type="cluster", cov_kwds={"groups": s["city"]})
            rows.append((pv, m1.params["T:post:tpz"],
                         m1.pvalues["T:post:tpz"]))
        except Exception:
            continue
    rows.sort(key=lambda r: r[1])
    for pv, c, p1 in rows:
        emit(f"  drop {pv:<12} {c:+.4f} (p={p1:.3f})")
    sign_keep = sum(1 for _, c, _ in rows if c < 0)
    emit(f"  sign preserved in {sign_keep}/{len(rows)} drops")

    # ---- secondary outcome: log house value (the gainers' mirror) -------
    dh = d[d["log_hv"].notna()]
    cfit(f"log_hv ~ T:post + T:post:tpz + T:tpz + post:tpz + tpz + {CT}{FE}",
         dh, ["T:post", "T:post:tpz"],
         "secondary: log house value on the same gradient")

    out = os.path.join(OUT, "tables", "income_gradient_did.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
