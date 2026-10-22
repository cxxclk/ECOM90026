# ==========================================================================
# REPLICATION PACKAGE — semisup_typology.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Three typologies)
# Reads: CFPS .dta files in data/20xx/, output/panel_individual.csv
# Writes: output/semisup_classes.csv, output/tables/semisup_typology.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard
# Run order: step 7 in run_all.sh
# ==========================================================================

"""Semi-supervised PRE-shock typology -> post-shock outcome scan (HPR).

Idea (2026-07-16): fix HPR as the shock; FIRST classify the population using
only pre-shock information (semi-supervised), THEN inspect the class structure
and scan which outcomes visibly move after the shock within each class.

Stage 1 — behavior-propensity embedding (all information dated <= 2015):
  For each woman in the analytic universe (ever-married, unregulated
  prefecture, baseline-city treatment), predict SIX pre-shock behaviors from
  baseline demographics with gradient boosting (5-fold out-of-fold
  predictions, NaN-native):
    births_1015  # of births 2010-2015 (latest famconf roster)
    ideal_kids   qm501 2014  ideal number of children
    son_pref     qm1004 2014 "must bear a son to continue the line" (1-5)
    housework    qq9010 2014 daily housework hours
    hv_inc       house value / family income, 2014 (housing burden)
    log_inc      own labor income 2014 (log1p)
  The 6-vector of OOF predictions is a smoothed "behavioral profile"
  (prognostic-score clustering: supervised signal defines the metric, the
  grouping itself is unsupervised).  GMM, K chosen by silhouette over 3..8
  (BIC reported).  The DID outcome side never enters: everything is pre-2016.
  Because the profile is E[behavior | X], classes are SYSTEMATIC behavioral
  types, not noise; and firstborn gender is excluded by construction (it is
  quasi-random, so no covariate-based class could recover it anyway).

Stage 2 — class structure: profile table (predicted + actual behavior means,
  demographics, treated share, firstborn-daughter share among mothers),
  assignments saved to output/semisup_classes.csv.

Stage 3 — post-shock scan per class (a SCREEN, not confirmatory):
  annual any/first/second-birth hazard DID (Post=2018, city+year FE, cluster
  city) with a pre-trend gate (# of 2010-2014 T x year coefs with p<0.10),
  plus wave-level log family income pc / homeowner / log house value DID.
  Benjamini-Hochberg FDR across all (class x outcome) cells flags survivors.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import annual_parity_hazard as aph                     # noqa: E402
from annual_firstborn_gender import load_rosters       # noqa: E402

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)

TARGETS = ["births_1015", "ideal_kids", "son_pref", "housework",
           "hv_inc", "log_inc"]
XCOLS = ["age14", "eduy", "urban", "parity14", "age_fb",
         "log_faminc_pc", "city_son_norm", "city_urban_share"]


def build_baseline():
    """One row per woman: X (2014 baseline), pre-shock behavior targets."""
    roster, last_wave = load_rosters()
    aph.LOAD_ROSTER = roster
    aph.LOAD_LAST = (roster, last_wave)

    attrs = aph.woman_attrs()          # ever-married, unregulated, treat_hpr
    attrs["age14"] = 2014 - attrs["birthy"]
    attrs = attrs[attrs["age14"].between(15, 45)]

    # roster-derived: parity at 2014, births 2010-2015, first-birth info
    r = roster.sort_values(["pid", "cby"]).copy()
    par14 = r[r["cby"] <= 2014].groupby("pid").size().rename("parity14")
    b1015 = (r[r["cby"].between(2010, 2015)].groupby("pid").size()
             .rename("births_1015"))
    r["k"] = r.groupby("pid").cumcount()
    fb = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "by1", "cgender": "fb_son"})
    d = (attrs.merge(par14, on="pid", how="left")
              .merge(b1015, on="pid", how="left")
              .merge(fb, on="pid", how="left"))
    d["parity14"] = d["parity14"].fillna(0)
    d["births_1015"] = d["births_1015"].fillna(0)
    d["age_fb"] = d["by1"] - d["birthy"]
    d["urban"] = d["urban"].where(d["urban"].isin([0, 1]))  # drop missing codes

    # 2014 wave covariates from the pipeline panel
    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "year", "eduy", "income", "fam_income",
                               "fam_income_pc", "homeowner", "house_value"])
    p14 = pan[pan["year"] == 2014].drop_duplicates("pid")
    d = d.merge(p14.drop(columns="year"), on="pid", how="left")
    d["log_faminc_pc"] = np.log1p(d["fam_income_pc"].clip(lower=0))
    d["log_inc"] = np.log1p(d["income"].clip(lower=0))
    hv = d["house_value"].where(d["house_value"] >= 0)
    fi = d["fam_income"].where(d["fam_income"] > 0)
    ratio = (hv * 1e4) / fi   # house_value in 万元, income in yuan
    d["hv_inc"] = np.log1p(ratio.clip(lower=0, upper=ratio.quantile(0.99)))

    # 2014 adult file: intentions / attitudes / time use (self + city norm)
    ad, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"),
        usecols=["pid", "countyid14", "qm501", "qm1004", "qq9010"])
    ad["ideal_kids"] = ad["qm501"].where(ad["qm501"].between(0, 10))
    ad["son_pref"] = ad["qm1004"].where(ad["qm1004"].between(1, 5))
    ad["housework"] = ad["qq9010"].where(ad["qq9010"].between(0, 24))
    d = d.merge(ad[["pid", "ideal_kids", "son_pref", "housework"]],
                on="pid", how="left")

    # city traits: son-norm (ALL 2014 adults, gender_norm_index recipe) +
    # urban share
    xw, _ = pyreadstat.read_dta(
        os.path.join(DATA, "CFPS_cleaned", "顺序码匹配.dta"))
    xw = xw[["countyid", "city"]].rename(columns={"city": "city_code"})
    xw["countyid"] = pd.to_numeric(xw["countyid"], errors="coerce")
    xw = xw.dropna().drop_duplicates("countyid")
    ad["countyid"] = pd.to_numeric(ad["countyid14"], errors="coerce")
    adc = ad.merge(xw, on="countyid", how="inner")
    norm = (adc.groupby("city_code")["son_pref"].mean()
            .rename("city_son_norm").reset_index())
    pu = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                     usecols=["year", "city_code", "urban"])
    pu = pu[pu["urban"].isin([0, 1])]
    urb = (pu[pu["year"] == 2014].groupby("city_code")["urban"].mean()
           .rename("city_urban_share").reset_index())
    d = d.merge(norm, on="city_code", how="left").merge(
        urb, on="city_code", how="left")
    return d


def oof_embed(d, seed=0):
    """Cross-fitted E[target | X] per target -> embedding matrix."""
    X = d[XCOLS].astype(float).values
    E = np.full((len(d), len(TARGETS)), np.nan)
    emit("\nStage 1 — supervised embedding (HistGBM, 5-fold OOF):")
    for j, t in enumerate(TARGETS):
        y = d[t].astype(float).values
        obs = ~np.isnan(y)
        pred = np.full(len(d), np.nan)
        kf = KFold(5, shuffle=True, random_state=seed)
        io = np.where(obs)[0]
        for tr, te in kf.split(io):
            m = HistGradientBoostingRegressor(random_state=seed,
                                              max_depth=4, max_iter=200)
            m.fit(X[io[tr]], y[io[tr]])
            pred[io[te]] = m.predict(X[io[te]])
        if (~obs).any():
            m = HistGradientBoostingRegressor(random_state=seed,
                                              max_depth=4, max_iter=200)
            m.fit(X[obs], y[obs])
            pred[~obs] = m.predict(X[~obs])
        E[:, j] = pred
        r2 = 1 - np.nanvar(y[obs] - pred[obs]) / np.nanvar(y[obs])
        emit(f"  {t:<12} observed={obs.mean():.0%}  OOF R2={r2:.3f}")
    return E


def choose_k(E, seed=0):
    Es = StandardScaler().fit_transform(E)
    emit("\nGMM model selection on the 6-dim behavior-propensity embedding:")
    best = {}
    for k in range(2, 9):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            n_init=5, random_state=seed).fit(Es)
        lab = g.predict(Es)
        sil = silhouette_score(Es, lab, sample_size=20000,
                               random_state=seed)
        best[k] = (g.bic(Es), sil, lab)
        emit(f"  K={k}: BIC={g.bic(Es):,.0f}  silhouette={sil:.3f}")
    K = max((k for k in best if k >= 3), key=lambda k: best[k][1])
    emit(f"  chosen K={K} (silhouette-best among K>=3); "
         f"finer exploratory cut K=6 also reported")
    return K, best, Es


def profile(d):
    emit("\nStage 2 — class structure (means; E_ = embedding/predicted, "
         "a_ = actual):")
    g = d.groupby("cls")
    tab = pd.DataFrame({"n": g.size(),
                        "treated": g["treat_hpr"].mean().round(2)})
    for t in TARGETS:
        tab["E_" + t] = g["E_" + t].mean().round(2)
    for t in TARGETS:
        tab["a_" + t] = g[t].mean().round(2)
    for x in XCOLS:
        tab[x] = g[x].mean().round(2)
    mothers = d[d["parity14"] >= 1]
    tab["fb_dghtr"] = (1 - mothers.groupby("cls")["fb_son"].mean()).round(2)
    tab["cities"] = g["city_code"].nunique()
    emit(tab.T.to_string())
    return tab


def bh_fdr(pvals, alpha=0.10):
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = np.zeros(m, dtype=bool)
    below = np.where(p[order] <= alpha * (np.arange(1, m + 1) / m))[0]
    if len(below):
        thresh[order[:below.max() + 1]] = True
    return thresh


_CACHE = {}

def _panels():
    if not _CACHE:
        attrs = aph.woman_attrs()
        risk = {k: aph.build_risk_set(attrs, k) for k in (0, 1, 2)}
        _CACHE["risk"] = risk
        _CACHE["pooled"] = pd.concat([risk[0], risk[1], risk[2]],
                                     ignore_index=True)
        _CACHE["pan"] = pd.read_csv(
            os.path.join(OUT, "panel_individual.csv"),
            usecols=["pid", "year", "birthy", "city_code",
                     "fam_income_pc", "homeowner", "house_value"])
    return _CACHE


def scan(d, verbose=True):
    """Stage 3 — per-class post-shock DID screen."""
    attrs_cls = d[["pid", "cls"]]
    cells = []

    # ---- annual hazards --------------------------------------------------
    cache = _panels()
    risk, pooled = cache["risk"], cache["pooled"]
    hazards = {"haz_any": (pooled, "age + I(age**2)"),
               "haz_1st": (risk[0], "age + I(age**2)"),
               "haz_2nd": (risk[1], "age + I(age**2) + dur + I(dur**2)"),
               "haz_3rd": (risk[2], "age + I(age**2) + dur + I(dur**2)")}
    FE = " + C(city) + C(year)"
    for name, (panel, ctrl) in hazards.items():
        pnl = panel.merge(attrs_cls, on="pid", how="inner")
        pnl["post"] = (pnl["year"] >= 2018).astype(int)
        for c in sorted(pnl["cls"].unique()):
            s = pnl[pnl["cls"] == c].copy()
            if (len(s) < 2000 or s["T"].nunique() < 2
                    or s.loc[s["post"] == 1, "y"].sum() < 5
                    or s.loc[s["post"] == 0, "y"].sum() < 5):
                continue   # degenerate: too few birth events on one side
            try:
                m = smf.ols(f"y ~ T:post + {ctrl}{FE}", data=s).fit(
                    cov_type="cluster", cov_kwds={"groups": s["city"]})
                # pre-trend gate: T x year 2010-2014, omitted 2015
                for yr in range(2010, 2015):
                    s[f"Ty{yr}"] = s["T"] * (s["year"] == yr).astype(int)
                terms = " + ".join(f"Ty{yr}" for yr in range(2010, 2015))
                s["Tpost"] = s["T"] * s["post"]
                mp = smf.ols(f"y ~ {terms} + Tpost + {ctrl}{FE}", data=s).fit(
                    cov_type="cluster", cov_kwds={"groups": s["city"]})
                npre = sum(mp.pvalues[f"Ty{yr}"] < 0.10
                           for yr in range(2010, 2015))
                cells.append(dict(cls=c, outcome=name,
                                  coef=m.params["T:post"],
                                  p=m.pvalues["T:post"],
                                  base=s.loc[s["post"] == 0, "y"].mean(),
                                  N=int(m.nobs), pre_viol=npre))
            except Exception as e:
                emit(f"  [scan {name} cls {c}] failed: {e}")

    # ---- wave-level economic outcomes ------------------------------------
    pan = cache["pan"].merge(attrs_cls, on="pid", how="inner")
    pan = pan.merge(d[["pid", "treat_hpr"]], on="pid", how="left")
    pan = pan[pan["year"].between(2010, 2022) & pan["city_code"].notna()]
    pan["T"] = pan["treat_hpr"].astype(int)
    pan["city"] = pan["city_code"].astype(int)
    pan["age"] = pan["year"] - pan["birthy"]
    pan["post"] = (pan["year"] >= 2018).astype(int)
    pan["log_fipc"] = np.log1p(pan["fam_income_pc"].clip(lower=0))
    pan["own"] = pan["homeowner"]
    pan["log_hv"] = np.log1p(pan["house_value"].where(pan["house_value"] > 0))
    for name, yv in [("log_faminc_pc", "log_fipc"), ("homeowner", "own"),
                     ("log_housevalue", "log_hv")]:
        for c in sorted(pan["cls"].unique()):
            s = pan[(pan["cls"] == c) & pan[yv].notna()
                    & pan["age"].notna()].copy()
            if len(s) < 1000 or s["T"].nunique() < 2:
                continue
            try:
                m = smf.ols(f"{yv} ~ T:post + age + I(age**2){FE}",
                            data=s).fit(cov_type="cluster",
                                        cov_kwds={"groups": s["city"]})
                for yr in [2010, 2012, 2014]:
                    s[f"Ty{yr}"] = s["T"] * (s["year"] == yr).astype(int)
                s["Tpost"] = s["T"] * s["post"]
                mp = smf.ols(f"{yv} ~ Ty2010 + Ty2012 + Ty2014 + Tpost"
                             f" + age + I(age**2){FE}", data=s).fit(
                    cov_type="cluster", cov_kwds={"groups": s["city"]})
                npre = sum(mp.pvalues[f"Ty{yr}"] < 0.10
                           for yr in [2010, 2012])   # 2014 wave = omitted-ish
                cells.append(dict(cls=c, outcome=name,
                                  coef=m.params["T:post"],
                                  p=m.pvalues["T:post"],
                                  base=s.loc[s["post"] == 0, yv].mean(),
                                  N=int(m.nobs), pre_viol=npre))
            except Exception as e:
                emit(f"  [scan {name} cls {c}] failed: {e}")

    tab = pd.DataFrame(cells)
    tab["fdr10"] = bh_fdr(tab["p"].values, alpha=0.10)
    tab["rel"] = tab["coef"] / tab["base"]
    tab = tab.sort_values("p")
    if verbose:
        emit("\nStage 3 — post-shock DID screen by class "
             "(Post=2018, city+year FE, cluster city)")
        emit("  pre_viol = # of pre-period (2010-14) T x year coefs with p<0.10")
        emit("  fdr10    = survives Benjamini-Hochberg at q=0.10 across all cells")
        with pd.option_context("display.width", 120):
            emit(tab.round({"coef": 4, "p": 3, "base": 3, "rel": 2})
                 .to_string(index=False))
    return tab


def main():
    d = build_baseline()
    emit("SEMI-SUPERVISED PRE-SHOCK TYPOLOGY (behavior-propensity clustering)")
    emit(f"  universe: ever-married women, unregulated prefectures, "
         f"age 15-45 in 2014: N={len(d):,}")

    E = oof_embed(d)
    K, best, Es = choose_k(E)
    for j, t in enumerate(TARGETS):
        d["E_" + t] = E[:, j]

    for kk, tag in [(K, "MAIN (silhouette-best)"), (6, "FINE (exploratory)")]:
        if kk != K and kk not in best:
            continue
        emit("\n" + "=" * 72)
        emit(f"TYPOLOGY {tag}: K={kk}")
        d["cls"] = best[kk][2]
        profile(d)
        if kk == K:
            d[["pid", "cls", "treat_hpr", "city_code"] +
              ["E_" + t for t in TARGETS]].to_csv(
                os.path.join(OUT, "semisup_classes.csv"), index=False)
            emit("\nassignments saved -> output/semisup_classes.csv")
        scan(d)
        if kk == 6 and K == 6:
            break

    out = os.path.join(OUT, "tables", "semisup_typology.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
