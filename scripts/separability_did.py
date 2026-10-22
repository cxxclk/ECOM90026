# ==========================================================================
# REPLICATION PACKAGE — separability_did.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Pre/post classifier)
# Reads: output/analysis_data.csv
# Writes: output/tables/separability_did.txt
# Imports local modules: none
# Run order: step 44 in run_all.sh
# ==========================================================================

"""Separability DID: a classifier two-sample test of the shock's distributional
footprint ("找一个超平面把 before/after 人群分得最开" — DID-ized).

Idea (2026-08-02, entertainment project): train a classifier to distinguish
pre-shock from post-shock person-waves. Raw pre/post separability measures
TIME (macro trends, aging, questionnaire drift), not the shock — so the
statistic is the DIFFERENCE in separability between arms:

  A. dAUC = AUC_treated(pre vs post) - AUC_control(pre vs post)
  B. control-detrended treated AUC: subtract control-arm year means from every
     feature, then classify pre vs post WITHIN the treated arm. If treated
     families merely follow control trends, OOF AUC ~ 0.5; any excess is a
     distributional DID signal (a multivariate, omnibus DID).

Inference: city-level permutation of the treatment label, re-running the FULL
pipeline (detrending + cross-fitted classifier) each draw — same philosophy as
effect_subset_scan.py. Direction readout: per-arm standardized logistic
coefficients (the hyperplane normal) and per-feature city-level DID t-stats.

Design notes:
- Sample: analysis_data.csv (married women 20-45), unregulated prefectures,
  treat_hpr non-missing. Pre = {2012, 2014}; Post = {2018, 2020, 2022};
  2016 dropped (transition year, Post=2018 convention).
- Features: wave-consistent behaviors only. mortgage_loan / monthly_rent are
  EXCLUDED (masked to NaN in 2012 -> their missingness is a perfect year
  detector). Missing values median-imputed (year-specific missingness patterns
  are shared across arms and cancel in the arm difference).
- Features are residualized on age + age^2 (pooled) before classification so
  panel-aging composition can't masquerade as a distributional shift; the
  transformation is common to both arms.
- Classifier: logistic (a literal hyperplane), OOF via city GroupKFold;
  HistGradientBoosting as a nonlinearity check.
- Variant C: balanced panel (women observed in >=1 pre AND >=1 post wave)
  to close the composition-drift interpretation gap.

Caveats: repeated cross-section in the main spec (composition drift possible,
hence C); permutation ignores spatial correlation of assignment (standard for
this project); this is an omnibus/closing exhibit, not a discovery device.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
N_PERM = int(os.environ.get("SEP_PERMS", 200))
SEED = 90026
rng = np.random.default_rng(SEED)

FEATURES = [
    "new_birth", "n_children_fam", "homeowner_d", "log_houseasset",
    "log_house_val", "n_other_prop", "log_faminc_pc", "log_own_inc",
    "eduy", "urban",
]

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def load():
    d = pd.read_csv(os.path.join(OUT, "analysis_data.csv"))
    d = d[(d["is_regulated"] == 0) & d["treat_hpr"].notna()
          & d["city_code"].notna()]
    d = d[d["year"].isin([2012, 2014, 2018, 2020, 2022])].copy()
    d["post"] = (d["year"] >= 2018).astype(int)
    d["T"] = d["treat_hpr"].astype(int)
    d["log_houseasset"] = np.log1p(d["houseasset_gross"].clip(lower=0))
    d["log_house_val"] = np.log1p(d["house_value"].clip(lower=0))
    d["log_faminc_pc"] = np.log1p(d["fam_income_pc"].clip(lower=0))
    d["log_own_inc"] = np.log1p(d["income"].clip(lower=0))
    return d


def prep_features(d, feats=None):
    """Median-impute, z-score, residualize on age+age^2, re-z."""
    X = d[feats or FEATURES].copy()
    X = X.fillna(X.median())
    Z = (X - X.mean()) / X.std().replace(0, 1)
    A = np.column_stack([np.ones(len(d)), d["age"].values,
                         d["age"].values ** 2])
    beta, *_ = np.linalg.lstsq(A, Z.values, rcond=None)
    R = Z.values - A @ beta
    R = (R - R.mean(0)) / np.where(R.std(0) == 0, 1, R.std(0))
    return R


def oof_auc(X, y, groups, model="logit", seed=SEED):
    """Out-of-fold AUC, folds grouped by city."""
    ug = np.unique(groups)
    k = min(5, len(ug))
    if k < 2 or len(np.unique(y)) < 2:
        return np.nan
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        if model == "logit":
            m = LogisticRegression(C=1.0, max_iter=2000)
        else:
            m = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(pred)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], pred[ok])


def detrend_by_control(X, year, is_ctrl):
    """Subtract control-arm year means (per feature) from all rows."""
    Xd = X.copy()
    for yr in np.unique(year):
        mc = is_ctrl & (year == yr)
        if mc.sum() < 30:
            mc = (year == yr)  # fallback: pooled year mean
        Xd[year == yr] -= X[mc].mean(0)
    return Xd


def stats_once(X, post, year, cityT_of_row, city, model="logit",
               compute_b=True):
    """(AUC_T, AUC_C, dAUC, detrended treated AUC) for one T assignment."""
    it, ic = cityT_of_row == 1, cityT_of_row == 0
    auc_t = oof_auc(X[it], post[it], city[it], model)
    auc_c = oof_auc(X[ic], post[ic], city[ic], model)
    auc_bt = np.nan
    if compute_b:
        Xd = detrend_by_control(X, year, ic)
        auc_bt = oof_auc(Xd[it], post[it], city[it], model)
    return auc_t, auc_c, (auc_t - auc_c), auc_bt


def run_block(name, d, X, model="logit", n_perm=N_PERM, compute_b=True):
    """compute_b=False for tree models: median-impute point masses shifted by
    per-year detrending give trees a perfect year fingerprint (B degenerates
    to AUC=1 under the null); dAUC is immune (leak shared across arms)."""
    post, year = d["post"].values, d["year"].values
    city = d["city_code"].values
    cities, cidx = np.unique(city, return_inverse=True)
    cityT = d.groupby("city_code")["T"].first().reindex(cities).values
    Trow = cityT[cidx]

    auc_t, auc_c, dauc, auc_bt = stats_once(X, post, year, Trow, city, model,
                                            compute_b)
    emit(f"[{name}] N={len(d)}  cities T/C = "
         f"{int(np.nansum(cityT))}/{int((cityT == 0).sum())}")
    emit(f"  AUC pre-vs-post: treated {auc_t:.4f} | control {auc_c:.4f} "
         f"| dAUC {dauc:+.4f}")
    if compute_b:
        emit(f"  control-detrended treated AUC (B): {auc_bt:.4f}")

    perm_d, perm_b = [], []
    for i in range(n_perm):
        pT = rng.permutation(cityT)
        pTrow = pT[cidx]
        _, _, pd_, pb_ = stats_once(X, post, year, pTrow, city, model,
                                    compute_b)
        if not np.isnan(pd_):
            perm_d.append(pd_)
        if not np.isnan(pb_):
            perm_b.append(pb_)
    perm_d, perm_b = np.array(perm_d), np.array(perm_b)
    p_d = (1 + np.sum(np.abs(perm_d) >= abs(dauc))) / (1 + len(perm_d))
    emit(f"  permutation ({len(perm_d)} draws, city-level): "
         f"dAUC two-sided p = {p_d:.3f} "
         f"[perm null mean {perm_d.mean():+.4f}, sd {perm_d.std():.4f}]")
    if compute_b and len(perm_b):
        p_b = (1 + np.sum(perm_b >= auc_bt)) / (1 + len(perm_b))
        emit(f"  detrended-AUC one-sided p = {p_b:.3f} "
             f"[perm null mean {perm_b.mean():.4f}, sd {perm_b.std():.4f}]")
    emit()


def direction_readout(d, X):
    """Hyperplane normals per arm + per-feature city-level DID t-stats."""
    post = d["post"].values
    emit("--- Direction readout (standardized logistic coefficients) ---")
    coefs = {}
    for lab, mask in [("T", d["T"].values == 1), ("C", d["T"].values == 0)]:
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[mask], post[mask])
        coefs[lab] = m.coef_[0]
    tab = pd.DataFrame({"coef_T": coefs["T"], "coef_C": coefs["C"]},
                       index=FEATURES)
    tab["diff_T_minus_C"] = tab["coef_T"] - tab["coef_C"]

    # per-feature city-level DID t (WLS of city (post-pre) delta on T, HC0)
    tstats = []
    for j, f in enumerate(FEATURES):
        g = pd.DataFrame({"city": d["city_code"].values, "post": post,
                          "T": d["T"].values, "x": X[:, j]})
        cm = g.groupby(["city", "post"])["x"].agg(["mean", "size"]).unstack()
        ok = (cm[("size", 0)].fillna(0) >= 5) & (cm[("size", 1)].fillna(0) >= 5)
        delta = (cm[("mean", 1)] - cm[("mean", 0)])[ok]
        w = (cm[("size", 0)] + cm[("size", 1)])[ok]
        Tc = g.groupby("city")["T"].first()[ok.index][ok]
        wm = np.average(Tc, weights=w)
        xc, sw = Tc - wm, w.values
        sxx = np.sum(sw * xc ** 2)
        b = np.sum(sw * xc * delta) / sxx
        resid = delta - delta.mean() - b * xc
        se = np.sqrt(np.sum((sw * xc * resid) ** 2)) / sxx
        tstats.append(b / se if se > 0 else np.nan)
    tab["city_DID_t"] = tstats
    tab = tab.reindex(tab["diff_T_minus_C"].abs()
                      .sort_values(ascending=False).index)
    emit(tab.round(3).to_string())
    emit()


def main():
    d = load()
    emit("=" * 72)
    emit("SEPARABILITY DID — classifier two-sample test of the shock's")
    emit("distributional footprint (pre {2012,2014} vs post {2018,2020,2022})")
    emit(f"features: {', '.join(FEATURES)}")
    emit(f"permutations: {N_PERM} (city-level), seed {SEED}")
    emit("=" * 72)
    emit()

    X = prep_features(d)
    run_block("A. main / logit", d, X, "logit")
    direction_readout(d, X)
    run_block("A'. nonlinear check / HistGBM (dAUC only)", d, X, "gbm",
              n_perm=min(N_PERM, 100), compute_b=False)

    # D: behavior-only features — drop composition-drift suspects
    # (eduy, n_children_fam, urban track WHO is sampled more than what
    # families DO; if the footprint survives without them it is behavioral)
    behav = ["new_birth", "homeowner_d", "log_houseasset", "log_house_val",
             "n_other_prop", "log_faminc_pc", "log_own_inc"]
    Xd_ = prep_features(d, behav)
    emit(f"[D features] {', '.join(behav)}")
    run_block("D. behavior-only / logit", d, Xd_, "logit")

    # C: balanced panel — women with >=1 pre and >=1 post observation
    pre_ids = set(d.loc[d["post"] == 0, "pid"])
    post_ids = set(d.loc[d["post"] == 1, "pid"])
    bal = d[d["pid"].isin(pre_ids & post_ids)].copy()
    Xb = prep_features(bal)
    run_block("C. balanced panel (same women pre & post) / logit",
              bal, Xb, "logit")

    emit("Read: dAUC ~ 0 & detrended AUC ~ 0.5 => the joint behavior")
    emit("distribution shifted the same way in treated and control =")
    emit("distribution-level completeness certificate. Any excess: check the")
    emit("direction table for which features carry it.")
    path = os.path.join(OUT, "tables", "separability_did.txt")
    with open(path, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
