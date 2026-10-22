# ==========================================================================
# REPLICATION PACKAGE — gbm_feature_importance.py
# Tier: CORE
# Paper items: Figure 6 (SHAP beeswarm)
# Reads: output/gbm_shap_values.csv
# Writes: output/tables/gbm_feature_importance.txt
# Imports local modules: cddf_spine
# Run order: step 9 in run_all.sh
# ==========================================================================

"""Feature importance + SHAP for the prognostic HistGBM.

Same frame/config as prognostic_gates.py (2009-2015 pre-shock woman-years,
one-child risk set, XCOLS_A + age + dur, HistGBM depth=3/iter=200/msl=50),
city-held-out split (20% of cities, stratified by T).

Outputs:
  output/tables/gbm_feature_importance.txt   permutation + SHAP rankings
  output/figures/gbm_shap_beeswarm.png       beeswarm, deck-ready labels

NOTE: this describes the PREDICTION of baseline second-birth propensity
(who has second births), NOT treatment-effect heterogeneity.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                    # noqa: E402

LAB = {"city_son_norm": "City son-norm (lineage culture)",
       "age": "Mother's age",
       "ideal_kids": "Her stated ideal # of kids",
       "dur": "Years since first birth",
       "son": "Firstborn is a son",
       "log_faminc_pc": "Family income p.c. (log)",
       "eduy": "Education (years)",
       "birthy": "Birth cohort",
       "city_urban_share": "City urban share",
       "city_linc": "City income level",
       "log_inc": "Own income (log)",
       "hv_inc": "House value / income",
       "trad_pc1": "City tradition index (PC1)",
       "housework": "Housework hours",
       "age_fb": "Age at first birth",
       "son_pref": "Own son-preference item",
       "homeowner": "Homeowner",
       "urban": "Urban hukou"}

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def main():
    df = cs.assemble()
    pre = df[df["year"] <= 2015].copy()
    feats = cs.XCOLS_A + ["age", "dur"]
    X = pre[feats].astype(float)
    y = pre["y"].values

    rng = np.random.RandomState(0)
    cities = pre[["city", "T"]].drop_duplicates()
    test_cities = set()
    for _, grp in cities.groupby("T"):
        idx = grp["city"].values.copy()
        rng.shuffle(idx)
        test_cities.update(idx[: max(1, len(idx) // 5)])
    te = pre["city"].isin(test_cities).values

    m = HistGradientBoostingRegressor(random_state=0, max_depth=3,
                                      max_iter=200, min_samples_leaf=50)
    m.fit(X[~te].values, y[~te])
    emit("Prognostic HistGBM feature importance (pre-2016, city-held-out)")
    emit(f"n_train={int((~te).sum())}  n_test={int(te.sum())}  "
         f"base_hazard={y.mean():.4f}  R2_test={m.score(X[te].values, y[te]):.4f}")

    # ---- permutation importance (held-out) ----
    r = permutation_importance(m, X[te].values, y[te], n_repeats=20,
                               random_state=0,
                               scoring="neg_mean_squared_error")
    emit("\nPermutation importance (MSE increase, held-out cities):")
    for i in np.argsort(-r.importances_mean):
        emit(f"  {feats[i]:18s} {r.importances_mean[i]:+.6f} "
             f"+- {r.importances_std[i]:.6f}")

    # ---- SHAP ----
    Xte = X[te].reset_index(drop=True)
    sv = shap.TreeExplainer(m).shap_values(Xte.values)
    mabs = np.abs(sv).mean(axis=0)
    emit("\nSHAP mean |value| (pp of hazard) + direction "
         "(corr of SHAP with feature value):")
    for i in np.argsort(-mabs):
        v = Xte.iloc[:, i].values
        ok = ~np.isnan(v)
        c = (np.corrcoef(v[ok], sv[ok, i])[0, 1]
             if ok.sum() > 50 and np.nanstd(v[ok]) > 0 else np.nan)
        emit(f"  {feats[i]:18s} {mabs[i]*100:6.3f}pp   dir_corr={c:+.2f}")

    emit("\nCaveats: prediction importance, not effect heterogeneity; "
         "trad_pc1 shares credit with city_son_norm (corr .84); city_linc/"
         "city_urban_share are used by the model but have negative held-out "
         "permutation importance (mild overfit).")

    # ---- cache SHAP values + feature values for scripts/paper_figures.py ----
    cache = os.path.join(ROOT, "output", "gbm_shap_values.csv")
    pd.concat([pd.DataFrame(sv * 100, columns=[f"shap__{f}" for f in feats]),
               Xte.add_prefix("x__")], axis=1).to_csv(cache, index=False)
    emit(f"cached SHAP values -> {cache}")

    # ---- beeswarm figure (deck-ready) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "font.family": "serif"})
    names = [LAB.get(f, f) for f in feats]
    shap.summary_plot(sv * 100, Xte, feature_names=names, show=False,
                      max_display=12, plot_size=(9.0, 6.4))
    ax = plt.gca()
    ax.set_xlabel("Push on predicted second-birth hazard (pp)", fontsize=13)
    for t in ax.get_yticklabels():
        t.set_fontsize(13)
    plt.tight_layout()
    fig_out = os.path.join(ROOT, "output", "figures", "gbm_shap_beeswarm.png")
    plt.savefig(fig_out, dpi=160, transparent=True)
    emit(f"\nsaved {fig_out}")

    tab = os.path.join(ROOT, "output", "tables", "gbm_feature_importance.txt")
    with open(tab, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("saved", tab)


if __name__ == "__main__":
    main()
