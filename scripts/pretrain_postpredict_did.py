# ==========================================================================
# REPLICATION PACKAGE — pretrain_postpredict_did.py
# Tier: CORE
# Paper items: Table 7 (Linear logit and ensemble rows, post-shock AUCs); Figure 7 (Pre-train, post-predict); Table 8 (Prediction-based counterfactual); Table E1 (28 model settings); Table E2 (Ten learners); Figure E1 (Calibration of the score)
# Reads: output/prognostic_scores.csv
# Writes: output/pretrain_postpredict_preds.csv, output/tables/pretrain_postpredict_did.txt
# Imports local modules: cddf_spine
# Run order: step 8 in run_all.sh
# ==========================================================================

"""Pre-train / post-predict DID (advisor suggestion, 2026-09-04).

Idea: learn the second-birth hazard on PRE-shock woman-years only, predict
every woman-year (pre AND post) out-of-fold, and treat the post-period
prediction as the counterfactual. Naively, actual − predicted in the post
period measures TIME (secular decline + 2016 two-child policy), which is
why we had dismissed the design. With a control arm it DID-izes cleanly:

    e_it = y_it − f̂_pre(x_it)         (f̂ trained on 2009–2015 rows, city folds)
    e_it = γ·T_i·post_t + α_city + λ_year + u_it

The control arm's post-period prediction error absorbs the model's time
drift; γ is the treated arm's post-period "surprise". This is DID with a
flexible ML covariate adjustment (cf. Burlig et al. 2020 JAERE — pre-period
ML predictions, DID on prediction errors; citation to be verified).

What is trained / predicted / tested (the protocol the advisor asked for):
  TRAIN   2009–2015 woman-years of the one-child risk set, 16 woman-level
          2014-baseline features + age + duration (no T, no city id, no year).
  PREDICT all woman-years 2009–2022, cross-fitted by CITY fold (a woman's own
          city never trains her prediction), SEEDS random fold partitions,
          median consensus.
  SELECT  learners compete on PRE-period OOF log-loss only (no post row, and
          in particular no treated-post row, ever touches model selection).
  TEST    (i) temporal holdout: the pre-trained model's fit on POST-period
          CONTROL rows (never treated, only time-shifted);
          (ii) residual event study: pre-year T−C residual gaps must be ≈0;
          (iii) fake shock: train ≤2012, pseudo-post 2013–15, γ must be ≈0;
          (iv) son-first arm placebo; (v) prognostic-quintile residual GATES
          (does the inverted U reappear under the ML counterfactual).
  VARIANT imputation (BJS-style): train on every non-treated-post row with
          year as a feature, impute the treated-post counterfactual directly.

Learner menu (not limited to HistGBM): linear logit (the DID-control
analogue), spline+interaction logit, HistGBM default & 8/11-tuned, LightGBM,
XGBoost, random forest, extra trees, MLP, and a top-3 average ensemble.

Output: output/tables/pretrain_postpredict_did.txt
        output/pretrain_postpredict_preds.csv (row-level OOF predictions)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import (StandardScaler, SplineTransformer,
                                   PolynomialFeatures, FunctionTransformer)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                  # noqa: E402

SEEDS = int(os.environ.get("PP_SEEDS", 3))
FOLDS = 5
WCB_REPS = int(os.environ.get("PP_WCB", 999))
NJ = 8
FE = cs.FE
CTRL = cs.CTRL
FEATS = cs.XCOLS_A + ["age", "dur"]
CONT = ["age", "dur", "age_fb", "birthy", "eduy", "log_faminc_pc",
        "log_inc", "hv_inc", "city_son_norm", "trad_pc1", "city_linc"]

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def design(df, feats, fit_mask):
    """Median-impute (stats from training rows) + missing indicators."""
    X = df[feats].astype(float).copy()
    cols = []
    for c in feats:
        v = X[c].values
        miss = np.isnan(v)
        if miss.any():
            cols.append(pd.Series(miss.astype(float), name=f"{c}_miss"))
            v = np.where(miss, np.nanmedian(v[fit_mask]), v)
        cols.append(pd.Series(v, name=c))
    M = pd.concat(cols, axis=1)
    return M.values, list(M.columns)


# ----------------------------------------------------------------------------
# learner menu
# ----------------------------------------------------------------------------
class AvgEns:
    """Average-probability ensemble of several learner factories."""
    def __init__(self, makers, seed):
        self.ms = [mk(seed) for mk in makers]

    def fit(self, X, y):
        for m in self.ms:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def make_learners(colnames):
    cont_idx = [colnames.index(c) for c in CONT if c in colnames]
    pick = FunctionTransformer(lambda Z, idx=cont_idx: Z[:, idx])
    L = {}
    L["logit_lin"] = lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    L["logit_spline"] = lambda s: make_pipeline(
        make_union(make_pipeline(pick, SplineTransformer(n_knots=6, degree=3)),
                   PolynomialFeatures(degree=2, interaction_only=True,
                                      include_bias=False)),
        StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
    L["hgb_default"] = lambda s: HistGradientBoostingClassifier(
        learning_rate=0.1, max_depth=3, max_iter=200, min_samples_leaf=50,
        random_state=s)
    L["hgb_tuned"] = lambda s: HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=2, min_samples_leaf=100, max_iter=400,
        early_stopping=True, n_iter_no_change=20, random_state=s)
    L["lightgbm"] = lambda s: lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=s, n_jobs=NJ,
        verbose=-1)
    L["xgboost"] = lambda s: xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=3,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", random_state=s, n_jobs=NJ,
        verbosity=0)
    L["rf"] = lambda s: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=50, max_features="sqrt",
        random_state=s, n_jobs=NJ)
    L["extratrees"] = lambda s: ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=50, max_features="sqrt",
        random_state=s, n_jobs=NJ)
    L["mlp"] = lambda s: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                        alpha=1e-3, early_stopping=True,
                                        max_iter=300, random_state=s))
    return L


# ----------------------------------------------------------------------------
# cross-fitting
# ----------------------------------------------------------------------------
def city_fold_vector(df, seed):
    rng = np.random.RandomState(seed)
    fold = {}
    for _, grp in df[["city", "T"]].drop_duplicates().groupby("T"):
        idx = grp["city"].values.copy()
        rng.shuffle(idx)
        for i, c in enumerate(idx):
            fold[c] = i % FOLDS
    return df["city"].map(fold).values


def crossfit(df, X, make, train_mask, seeds):
    """OOF prediction for EVERY row; model fit only on train_mask rows of
    other city folds. Median over seed partitions."""
    yv = df["y"].values
    preds = []
    for s in seeds:
        fvec = city_fold_vector(df, s)
        p = np.full(len(df), np.nan)
        for f in range(FOLDS):
            te = fvec == f
            tr = (~te) & train_mask
            m = make(s)
            m.fit(X[tr], yv[tr])
            p[te] = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        preds.append(p)
    return np.median(np.vstack(preds), axis=0)


# ----------------------------------------------------------------------------
# metrics / regressions
# ----------------------------------------------------------------------------
def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q[1:-1], p, side="right"), 0, bins - 1)
    out = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return out


def metrics(y, p):
    return dict(ll=log_loss(y, p), auc=roc_auc_score(y, p),
                brier=brier_score_loss(y, p), ece=ece(y, p))


def did(d, yvar, extra=""):
    m = smf.ols(f"{yvar} ~ T:post{extra}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    return m.params["T:post"], m.bse["T:post"], m.pvalues["T:post"]


def fmt(b, p, mult=100):
    star = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
    return f"{b*mult:+.2f}pp (p={p:.3f}){star}"


def wild_cluster_boot(d, formula, coef, reps=999, seed=90026):
    """WCB-t, Rademacher, null imposed, cluster=city (copied from
    headline_robustness.py)."""
    y_full, X_full = dmatrices(formula, d, return_type="dataframe")
    y = y_full.values.ravel()
    x = X_full[coef].values
    Z = X_full.drop(columns=coef).values
    cl = d["city"].values
    Q, _ = np.linalg.qr(Z)
    resid = lambda v: v - Q @ (Q.T @ v)                       # noqa: E731
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


def es_fit(d, yvar, base=2015):
    """Explicit T x year dummies (all years except base). NB: patsy codes
    `T:C(year, Treatment(base))` FULL-rank when T is numeric, which with
    city FE is collinear and shifts every coefficient by an arbitrary
    constant — never use that form."""
    d = d.copy()
    years = [yy for yy in sorted(d["year"].unique()) if yy != base]
    terms = []
    for yy in years:
        d[f"yr{yy}"] = (d["year"] == yy).astype(int)
        terms.append(f"T:yr{yy}")
    m = smf.ols(f"{yvar} ~ {' + '.join(terms)}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    return m, years


def event_study(d, yvar, base=2015):
    m, years = es_fit(d, yvar, base)
    return [(yy, m.params[f"T:yr{yy}"], m.pvalues[f"T:yr{yy}"]) for yy in years]


def es_key(m, yy):
    """patsy names the T x C(year) level as '[2009]' or '[T.2009]'."""
    for k in m.params.index:
        if k.startswith("T:C(year") and (k.endswith(f"[{yy}]")
                                        or k.endswith(f"[T.{yy}]")):
            return k
    raise KeyError(yy)


def event_study_prebase(d, yvar, years=range(2016, 2023)):
    d = d.copy()
    terms = []
    for yy in years:
        d[f"yr{yy}"] = (d["year"] == yy).astype(int)
        terms.append(f"T:yr{yy}")
    m = smf.ols(f"{yvar} ~ {' + '.join(terms)}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    return [(yy, m.params[f"T:yr{yy}"], m.pvalues[f"T:yr{yy}"]) for yy in years]


def gates(d, yvar):
    tri = " + ".join(f"T:post:g{k}" for k in range(1, 6))
    two = " + ".join(f"T:g{k} + post:g{k} + g{k}" for k in range(1, 5))
    m = smf.ols(f"{yvar} ~ {tri} + {two}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    return [(m.params[f"T:post:g{k}"], m.pvalues[f"T:post:g{k}"])
            for k in range(1, 6)]


def three_arms(d, yvar, extra=""):
    return {lab: did(d[msk], yvar, extra) for lab, msk in
            [("pooled", np.ones(len(d), bool)),
             ("daughter", (d["son"] == 0).values),
             ("son", (d["son"] == 1).values)]}


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    df = cs.assemble().reset_index(drop=True)
    df["T"] = df["T"].astype(int)
    df["post"] = df["post"].astype(int)
    pre = (df["year"] <= 2015).values
    postC = ((df["post"] == 1) & (df["T"] == 0)).values
    postT = ((df["post"] == 1) & (df["T"] == 1)).values
    yv = df["y"].values.astype(float)

    emit("PRE-TRAIN / POST-PREDICT DID — one-child second-birth hazard")
    emit(f"rows={len(df):,} women={df['pid'].nunique():,} "
         f"cities={df['city'].nunique()}  pre rows={pre.sum():,} "
         f"post rows={(~pre).sum():,}  (Post=2018; 2016-17 rows are neither "
         f"training nor post)")
    emit(f"features ({len(FEATS)}): {', '.join(FEATS)}  [2014 baseline + "
         f"age/dur; no T, no city, no year]")
    emit(f"cross-fit: {FOLDS} city folds x {SEEDS} seeds, median consensus; "
         f"training rows = 2009-2015 only")

    # ---- baseline y-DID for reference --------------------------------------
    emit("\n[0] Reference: standard DID on y with linear age/dur controls")
    ref = three_arms(df, "y", " + " + CTRL)
    for k, (b, se, p) in ref.items():
        emit(f"  {k:<9} {fmt(b, p)}")

    # ---- learner menu ----------------------------------------------------
    X, cols = design(df, FEATS, pre)
    learners = make_learners(cols)
    seeds = list(range(SEEDS))
    P = {}
    rows = []
    emit("\n[1] Learner menu — trained on pre rows only, OOF by city fold")
    emit("    selection metric = PRE-period OOF log-loss (lower is better);"
         " post-CONTROL columns = temporal holdout (never used for selection)")
    hdr = (f"  {'learner':<13}{'pre ll':>9}{'pre auc':>9}{'pre ece':>9}"
           f"{'postC ll':>10}{'postC auc':>10}{'postC ece':>10}"
           f"{'resid DID pooled':>20}{'daughter':>18}{'son':>18}{'sec':>6}")
    emit(hdr)
    for name, make in learners.items():
        tt = time.time()
        p = crossfit(df, X, make, pre, seeds)
        P[name] = p
        df[f"e_{name}"] = yv - p
        mp = metrics(yv[pre], p[pre])
        mc = metrics(yv[postC], p[postC])
        r = three_arms(df, f"e_{name}")
        rows.append(dict(name=name, **{f"pre_{k}": v for k, v in mp.items()},
                         **{f"postC_{k}": v for k, v in mc.items()},
                         pooled=r["pooled"], daughter=r["daughter"],
                         son=r["son"]))
        emit(f"  {name:<13}{mp['ll']:>9.5f}{mp['auc']:>9.4f}{mp['ece']:>9.4f}"
             f"{mc['ll']:>10.5f}{mc['auc']:>10.4f}{mc['ece']:>10.4f}"
             f"{fmt(r['pooled'][0], r['pooled'][2]):>20}"
             f"{fmt(r['daughter'][0], r['daughter'][2]):>18}"
             f"{fmt(r['son'][0], r['son'][2]):>18}{time.time()-tt:>6.0f}")

    # ensemble of top-3 by pre log-loss
    order = sorted(rows, key=lambda r: r["pre_ll"])
    top3 = [r["name"] for r in order[:3]]
    p = np.mean([P[n] for n in top3], axis=0)
    P["ens_top3"] = p
    df["e_ens_top3"] = yv - p
    mp, mc = metrics(yv[pre], p[pre]), metrics(yv[postC], p[postC])
    r = three_arms(df, "e_ens_top3")
    rows.append(dict(name="ens_top3", **{f"pre_{k}": v for k, v in mp.items()},
                     **{f"postC_{k}": v for k, v in mc.items()},
                     pooled=r["pooled"], daughter=r["daughter"], son=r["son"]))
    emit(f"  {'ens_top3':<13}{mp['ll']:>9.5f}{mp['auc']:>9.4f}{mp['ece']:>9.4f}"
         f"{mc['ll']:>10.5f}{mc['auc']:>10.4f}{mc['ece']:>10.4f}"
         f"{fmt(r['pooled'][0], r['pooled'][2]):>20}"
         f"{fmt(r['daughter'][0], r['daughter'][2]):>18}"
         f"{fmt(r['son'][0], r['son'][2]):>18}")
    emit(f"  ensemble members (top-3 pre log-loss): {', '.join(top3)}")
    learners["ens_top3"] = lambda s, mk=[learners[n] for n in top3]: AvgEns(mk, s)

    order = sorted(rows, key=lambda r: r["pre_ll"])
    win = order[0]["name"]
    emit(f"\n  WINNER by pre-period OOF log-loss: {win}")
    lin = next(r for r in rows if r["name"] == "logit_lin")
    emit(f"  (linear-logit benchmark: ll {lin['pre_ll']:.5f}, resid DID "
         f"daughter {fmt(lin['daughter'][0], lin['daughter'][2])} — should "
         f"track the reference y-DID)")

    # mean residual by cell for the winner
    e = df[f"e_{win}"].values
    emit(f"\n  mean residual (y - yhat) by cell, {win}:")
    for lab, m in [("C pre", pre & (df['T'] == 0).values),
                   ("T pre", pre & (df['T'] == 1).values),
                   ("C post", postC), ("T post", postT)]:
        emit(f"    {lab:<7} {e[m].mean()*100:+.2f}pp  (n={m.sum():,})")
    preT = pre & (df["T"] == 1).values
    preC = pre & (df["T"] == 0).values
    raw_did = ((e[postT].mean() - e[preT].mean())
               - (e[postC].mean() - e[preC].mean())) * 100
    emit(f"  raw DID of cell means: {raw_did:+.2f}pp")
    mt = metrics(yv[postT], P[win][postT])
    emit(f"  fit on treated-post rows (informational; degradation = effect): "
         f"ll {mt['ll']:.5f} auc {mt['auc']:.4f} ece {mt['ece']:.4f}")

    # cache predictions
    keep = ["pid", "year", "city", "T", "post", "son", "y"]
    pd.concat([df[keep], pd.DataFrame({f"p_{k}": v for k, v in P.items()})],
              axis=1).to_csv(os.path.join(OUT, "pretrain_postpredict_preds.csv"),
                             index=False)

    # ---- [2] winner deep dive ------------------------------------------
    ev = f"e_{win}"
    emit(f"\n[2] Residual DID, winner = {win}")
    emit("  (a) with vs without linear age/dur controls (ML should have "
         "absorbed lifecycle):")
    for lab, extra in [("no controls", ""), ("+ age/dur", " + " + CTRL)]:
        r = three_arms(df, ev, extra)
        emit(f"    {lab:<12} pooled {fmt(r['pooled'][0], r['pooled'][2]):<22}"
             f"daughter {fmt(r['daughter'][0], r['daughter'][2]):<22}"
             f"son {fmt(r['son'][0], r['son'][2])}")
    # decomposition: DID(y) = DID(yhat) + DID(e)  (same FE spec, no controls)
    emit("  (a2) decomposition DID(y) = DID(yhat) + DID(e): the yhat part is "
         "the OBSERVABLE-COMPOSITION shift of the post-period risk set")
    emit(f"    {'learner':<12}{'arm':<10}{'DID(y)':>18}{'DID(yhat)':>18}"
         f"{'DID(e)':>18}")
    for name in [win, "logit_lin", "hgb_tuned"]:
        df[f"p_{name}"] = P[name]
        for arm, msk in [("pooled", np.ones(len(df), bool)),
                         ("daughter", (df["son"] == 0).values),
                         ("son", (df["son"] == 1).values)]:
            ry = did(df[msk], "y")
            rp = did(df[msk], f"p_{name}")
            re_ = did(df[msk], f"e_{name}")
            emit(f"    {name:<12}{arm:<10}{fmt(ry[0], ry[2]):>18}"
                 f"{fmt(rp[0], rp[2]):>18}{fmt(re_[0], re_[2]):>18}")
    # stayers (women already at risk before 2016) vs full sample: if the
    # composition shift survives among stayers it is DEPLETION of the
    # at-risk pool (births taken in 2016-17), not new-entrant composition
    stay_pid = set(df.loc[pre, "pid"])
    stay = df["pid"].isin(stay_pid).values
    emit("  (a4) decomposition restricted to STAYERS (women with a pre-2016 "
         f"risk-set row; {stay.mean()*100:.0f}% of rows, "
         f"{(stay & ~pre).mean()/(~pre).mean()*100:.0f}% of post rows)")
    for arm, msk in [("pooled", np.ones(len(df), bool)),
                     ("daughter", (df["son"] == 0).values)]:
        mm_ = msk & stay
        ry = did(df[mm_], "y"); rp = did(df[mm_], f"p_{win}"); re_ = did(df[mm_], ev)
        emit(f"    stayers {arm:<10}{fmt(ry[0], ry[2]):>18}{fmt(rp[0], rp[2]):>18}"
             f"{fmt(re_[0], re_[2]):>18}")
    ent = df.assign(ent=(~stay).astype(int))
    for arm, msk in [("pooled", np.ones(len(df), bool)),
                     ("daughter", (df["son"] == 0).values)]:
        r = did(ent[msk], "ent")
        emit(f"    entrant share T:post, {arm}: {fmt(r[0], r[2])} (composition of "
             "the post pool by entry status)")
    # attribute DID(yhat) to features with a linear probability model fit on
    # pre rows: DID(x'b) = sum_k b_k * DID(x_k)
    emit("  (a3) feature attribution of the composition shift (LPM on pre "
         "rows: sum_k beta_k * DID(x_k)), daughter arm and pooled")
    from sklearn.linear_model import LinearRegression
    lpm = LinearRegression().fit(X[pre], yv[pre])
    for arm, msk in [("daughter", (df["son"] == 0).values),
                     ("pooled", np.ones(len(df), bool))]:
        dd = df[msk].copy()
        contrib = []
        for j, c in enumerate(cols):
            dd["_x"] = X[msk, j]
            if dd["_x"].std() == 0:
                continue
            bx, _, px = did(dd, "_x")
            contrib.append((c, lpm.coef_[j] * bx, bx, px))
        tot = sum(v for _, v, _, _ in contrib)
        contrib.sort(key=lambda t: -abs(t[1]))
        emit(f"    {arm}: LPM-implied DID(yhat) = {tot*100:+.2f}pp; top "
             f"contributors [feature: contribution | DID(x) (p)]")
        for c, v, bx, px in contrib[:8]:
            emit(f"      {c:<20}{v*100:+.3f}pp | DID(x)={bx:+.3f} (p={px:.3f})")
    m = smf.ols(f"{ev} ~ T:post + T:post:son + T:son + post:son + son{FE}",
                data=df).fit(cov_type="cluster", cov_kwds={"groups": df["city"]})
    emit(f"  (b) DDD T:post:son {fmt(m.params['T:post:son'], m.pvalues['T:post:son'])}"
         f"  | T:post (daughter) {fmt(m.params['T:post'], m.pvalues['T:post'])}"
         f"  | T:son pre-gap {fmt(m.params['T:son'], m.pvalues['T:son'])}")
    da = df[df["son"] == 0]
    b, t, pw, G = wild_cluster_boot(da, f"{ev} ~ T:post{FE}", "T:post",
                                    reps=WCB_REPS)
    emit(f"  (c) WCB daughter arm: {b*100:+.2f}pp t={t:.2f} WCB p={pw:.3f} "
         f"(G={G}, {WCB_REPS} reps)")
    b, t, pw, G = wild_cluster_boot(df, f"{ev} ~ T:post{FE}", "T:post",
                                    reps=WCB_REPS)
    emit(f"      WCB pooled:       {b*100:+.2f}pp t={t:.2f} WCB p={pw:.3f}")

    emit("\n  (d0) DAUGHTER arm event studies (base 2015): y = raw hazard "
         "(linear ctrl), yhat = predicted (composition), e = residual")
    da_ = df[df["son"] == 0]
    es_y = event_study(da_, "y")
    es_p = event_study(da_, f"p_{win}")
    es_e = event_study(da_, ev)
    emit(f"    {'year':<6}{'y-ES':>20}{'yhat-ES':>20}{'e-ES':>20}")
    for i, (yy, _, _) in enumerate(es_y):
        emit(f"    {yy:<6}{fmt(es_y[i][1], es_y[i][2]):>20}"
             f"{fmt(es_p[i][1], es_p[i][2]):>20}"
             f"{fmt(es_e[i][1], es_e[i][2]):>20}")

    emit("\n  (d1) residual event study, base = ALL pre-years 2009-15 "
         "(T x post-year dummies only; avoids the 2015 base-year anomaly)")
    esb = {lab: event_study_prebase(df[msk], ev) for lab, msk in
           [("pooled", np.ones(len(df), bool)),
            ("daughter", (df["son"] == 0).values),
            ("son", (df["son"] == 1).values)]}
    emit(f"    {'year':<6}{'pooled':>20}{'daughter':>20}{'son':>20}")
    for i, (yy, _, _) in enumerate(esb["pooled"]):
        emit(f"    {yy:<6}" + "".join(
            f"{fmt(esb[k][i][1], esb[k][i][2]):>20}"
            for k in ["pooled", "daughter", "son"]))

    emit("\n  (d) residual event study, T x year (base 2015); pre-years are "
         "the placebo (model never saw post rows; pre gaps should be ~0)")
    es = {lab: event_study(df[msk], ev) for lab, msk in
          [("pooled", np.ones(len(df), bool)),
           ("daughter", (df["son"] == 0).values),
           ("son", (df["son"] == 1).values)]}
    emit(f"    {'year':<6}{'pooled':>20}{'daughter':>20}{'son':>20}")
    for i, (yy, _, _) in enumerate(es["pooled"]):
        emit(f"    {yy:<6}" + "".join(
            f"{fmt(es[k][i][1], es[k][i][2]):>20}"
            for k in ["pooled", "daughter", "son"]))
    for k in ["pooled", "daughter", "son"]:
        pre_es = [(yy, b, p) for yy, b, p in es[k] if yy < 2016]
        d_ = df if k == "pooled" else df[df["son"] == (k == "son")]
        mm, _ = es_fit(d_, ev)
        names = [f"T:yr{yy}" for yy, _, _ in pre_es]
        R = np.zeros((len(names), len(mm.params)))
        for i, n in enumerate(names):
            R[i, list(mm.params.index).index(n)] = 1
        f = mm.f_test(R)
        emit(f"    joint pre-years (2009-14) = 0, {k}: F={float(f.fvalue):.2f} "
             f"p={float(f.pvalue):.3f}")

    # (e) residual GATES by prognostic quintile
    sc = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))[["pid", "grp"]]
    dg = df.merge(sc, on="pid", how="inner")
    for k in range(1, 6):
        dg[f"g{k}"] = (dg["grp"] == k).astype(int)
    emit("\n  (e) residual GATES by prognostic-propensity quintile "
         f"(groups from output/prognostic_scores.csv; n={len(dg):,} rows)")
    emit("      G5 = highest predicted propensity; inverted U = G4 most negative")
    gy = gates(dg, "y")
    ge = gates(dg, ev)
    emit(f"    {'grp':<5}{'y-DID (linear ctrl)':>24}{'residual DID':>24}")
    for k in range(5):
        emit(f"    G{k+1:<4}{fmt(gy[k][0], gy[k][1]):>24}"
             f"{fmt(ge[k][0], ge[k][1]):>24}")
    dgd = dg[dg["son"] == 0]
    ged = gates(dgd, ev)
    emit("    daughter arm, residual: " + "  ".join(
        f"G{k+1} {fmt(ged[k][0], ged[k][1])}" for k in range(5)))

    # (f) support check
    emit("\n  (f) support / extrapolation check (post rows vs training range)")
    for c in ["birthy", "age", "dur", "age_fb"]:
        lo, hi = df.loc[pre, c].min(), df.loc[pre, c].max()
        out = ((df[c] < lo) | (df[c] > hi)) & ~pre
        emit(f"    {c:<8} train range [{lo:.0f},{hi:.0f}]  post rows outside: "
             f"{out.sum():,} ({out.mean()*100:.1f}% of all rows, "
             f"{out.sum()/(~pre).sum()*100:.1f}% of post rows)")
    by_max = df.loc[pre, "birthy"].max()
    cs_mask = ~((df["birthy"] > by_max) & ~pre)
    r = three_arms(df[cs_mask], ev)
    emit(f"    common-support variant (drop post rows of cohorts unseen in "
         f"training, n={cs_mask.sum():,}): pooled {fmt(r['pooled'][0], r['pooled'][2])}"
         f"  daughter {fmt(r['daughter'][0], r['daughter'][2])}"
         f"  son {fmt(r['son'][0], r['son'][2])}")
    dsub = df[cs_mask].copy()
    Xs, colss = design(dsub, FEATS, (dsub["year"] <= 2015).values)
    ps = crossfit(dsub, Xs, learners[win], (dsub["year"] <= 2015).values, seeds)
    dsub["e_cs"] = dsub["y"].values - ps
    r = three_arms(dsub, "e_cs")
    emit(f"    ... retrained on the same support: pooled "
         f"{fmt(r['pooled'][0], r['pooled'][2])}  daughter "
         f"{fmt(r['daughter'][0], r['daughter'][2])}  son "
         f"{fmt(r['son'][0], r['son'][2])}")

    # (g) fake shock: train <=2012, pseudo-post 2013-15
    emit("\n  (g) FAKE SHOCK placebo: train on 2009-2012 only, pseudo-post = "
         "2013-2015, sample restricted to <=2015")
    dfk = df[df["year"] <= 2015].copy().reset_index(drop=True)
    trk = (dfk["year"] <= 2012).values
    dfk["ppost"] = (dfk["year"] >= 2013).astype(int)
    Xk, _ = design(dfk, FEATS, trk)
    for name in [win, "logit_lin"]:
        pk = crossfit(dfk, Xk, learners[name], trk, seeds)
        dfk["ek"] = dfk["y"].values - pk
        res = {}
        for lab, msk in [("pooled", np.ones(len(dfk), bool)),
                         ("daughter", (dfk["son"] == 0).values),
                         ("son", (dfk["son"] == 1).values)]:
            mm = smf.ols(f"ek ~ T:ppost{FE}", data=dfk[msk]).fit(
                cov_type="cluster", cov_kwds={"groups": dfk.loc[msk, "city"]})
            res[lab] = (mm.params["T:ppost"], mm.pvalues["T:ppost"])
        emit(f"    {name:<12} pooled {fmt(*res['pooled']):<22}daughter "
             f"{fmt(*res['daughter']):<22}son {fmt(*res['son'])}")

    # (h) imputation variant: train on all non-treated-post rows incl. year
    emit("\n  (h) IMPUTATION variant (BJS-style): train on every row except "
         "treated-post, year as a feature; residual DID on the same FE spec")
    Xi, _ = design(df, FEATS + ["year"], ~postT)
    for name in [win, "hgb_tuned", "logit_lin"]:
        if name != win and name not in learners:
            continue
        pi = crossfit(df, Xi, learners[name], ~postT, seeds)
        df[f"ei_{name}"] = yv - pi
        r = three_arms(df, f"ei_{name}")
        mtp = (yv[postT] - pi[postT]).mean()
        emit(f"    {name:<12} pooled {fmt(r['pooled'][0], r['pooled'][2]):<22}"
             f"daughter {fmt(r['daughter'][0], r['daughter'][2]):<22}"
             f"son {fmt(r['son'][0], r['son'][2]):<22}"
             f"| raw mean(y-yhat) on treated-post {mtp*100:+.2f}pp")
        if name == win:
            mp = metrics(yv[postC], pi[postC])
            emit(f"      post-control OOF fit with year feature: ll {mp['ll']:.5f}"
                 f" auc {mp['auc']:.4f} ece {mp['ece']:.4f}")

    # ---- [3] cross-learner summary ------------------------------------
    emit("\n[3] Cross-learner summary (residual DID, daughter arm) sorted by "
         "pre log-loss")
    emit(f"  {'learner':<13}{'pre ll':>9}{'postC ll':>10}{'daughter':>20}"
         f"{'son':>20}{'pooled':>20}")
    for r in sorted(rows, key=lambda r: r["pre_ll"]):
        emit(f"  {r['name']:<13}{r['pre_ll']:>9.5f}{r['postC_ll']:>10.5f}"
             f"{fmt(r['daughter'][0], r['daughter'][2]):>20}"
             f"{fmt(r['son'][0], r['son'][2]):>20}"
             f"{fmt(r['pooled'][0], r['pooled'][2]):>20}")
    emit(f"  reference y-DID:  daughter {fmt(ref['daughter'][0], ref['daughter'][2])}"
         f"  son {fmt(ref['son'][0], ref['son'][2])}  pooled "
         f"{fmt(ref['pooled'][0], ref['pooled'][2])}")
    emit(f"\nruntime {time.time()-t0:.0f}s")

    with open(os.path.join(OUT, "tables", "pretrain_postpredict_did.txt"),
              "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
