# ==========================================================================
# REPLICATION PACKAGE — causal_forest_scan_c.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Causal forest)
# Reads: (via imported modules)
# Writes: output/tables/causal_forest_scan_c.txt
# Imports local modules: dml_drdid_c
# Run order: step 38 in run_all.sh
# ==========================================================================

"""
Causal-forest DISCOVERY scan on the one-child second-birth-hazard DiD.

Positioning (see session note 2026-07-01): NOT a confirmation device (the prior
causal_forest_het.py already showed firstborn_son ranks ~5th and the forest does
NOT rediscover the gender split). Used here for what forests are actually good
at — DISCOVERY: (1) surface an unanticipated moderator, (2) characterize the
continuous affordability surface, (3) test whether the gender split survives
conditioning on everything else. Every forest hit is then RE-TESTED in a clean
city-clustered DiD (the discovery -> pre-specified-retest discipline).

DiD-CATE construction (textbook DiD heterogeneity): collapse each at-risk woman
to her pre (year<2018) and post (>=2018) mean second-birth hazard; dY = post-pre
differences out time-invariant confounders. Causal forest with T = treat_hpr
(spillover group, binary) on dY, moderators X. CATE_i = effect of the shock on
the pre->post hazard change for woman i. BLP + importances describe where the
effect concentrates. Overlap/cluster caveats identical to the DR-DiD run.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from econml.dml import CausalForestDML

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "output")
from dml_drdid_c import build_panel     # reuse person-year panel + baseline moderators

LINES = []
def emit(s=""):
    print(s); LINES.append(s)

# candidate moderators (mix of tested + NOT-yet-tested, to allow discovery)
MODS = ["log_income0", "log_house_val0", "log_fam_inc_pc0", "urban0", "homeowner_d0",
        "eduy0", "hp_city_avg0", "age_first", "first_by", "provcd", "son"]
PRETTY = {"log_income0": "mother income", "log_house_val0": "house value",
          "log_fam_inc_pc0": "fam income pc", "urban0": "urban", "homeowner_d0": "homeowner",
          "eduy0": "mother edu-yrs", "hp_city_avg0": "baseline city price",
          "age_first": "age at 1st birth", "first_by": "1st-birth cohort",
          "provcd": "province", "son": "firstborn son"}


def collapse_prepost(pv):
    pv = pv.copy()
    pv["per"] = np.where(pv["year"] >= 2018, "post", "pre")
    g = pv.groupby(["pid", "per"])["y"].mean().unstack()
    both = g.dropna(subset=["pre", "post"])
    dY = (both["post"] - both["pre"]).rename("dY")
    w = pv.groupby("pid").agg(
        T=("T", "first"), city=("city", "first"), son=("son", "first"),
        birthy=("birthy", "first"), first_by=("first_by", "first"),
        log_income0=("log_income0", "first"), log_house_val0=("log_house_val0", "first"),
        log_fam_inc_pc0=("log_fam_inc_pc0", "first"), urban0=("urban0", "first"),
        homeowner_d0=("homeowner_d0", "first"), eduy0=("eduy0", "first"),
        hp_city_avg0=("hp_city_avg0", "first"), provcd=("provcd", "first"))
    w = w.join(dY, how="inner").reset_index()
    w["age_first"] = w["first_by"] - w["birthy"]
    return w


def main():
    np.random.seed(7)
    pv = build_panel()
    emit("Causal-forest DISCOVERY scan — DiD-CATE on pre->post 2nd-birth hazard change")
    W = collapse_prepost(pv)
    emit(f"  women at risk in BOTH pre & post: {len(W):,} "
         f"(daughter={int((W['son']==0).sum()):,}, son={int((W['son']==1).sum()):,}); "
         f"treated share={W['T'].mean():.3f}, cities={W['city'].nunique()}")
    emit(f"  mean dY (post-pre hazard change): treated={W.loc[W['T']==1,'dY'].mean():+.4f} "
         f"control={W.loc[W['T']==0,'dY'].mean():+.4f}")

    X = W[MODS].apply(lambda c: c.fillna(c.median()))
    Y = W["dY"].values
    Tt = W["T"].values

    est = CausalForestDML(
        model_y=HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300),
        model_t=HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300),
        discrete_treatment=True, n_estimators=3000, min_samples_leaf=25,
        max_depth=None, random_state=7, cv=5)
    est.fit(Y, Tt, X=X.values)
    tau = est.effect(X.values)
    W["cate"] = tau
    ate = est.ate(X.values)
    ate_ci = est.ate_interval(X.values)
    emit(f"\n  Forest ATE on dY = {ate:+.4f}  95%CI [{ate_ci[0]:+.4f},{ate_ci[1]:+.4f}]")
    emit(f"  CATE spread: p10={np.percentile(tau,10):+.4f} median={np.median(tau):+.4f} "
         f"p90={np.percentile(tau,90):+.4f}")

    # (1) forest variable importance
    emit("\n" + "=" * 68)
    emit("(1) Forest variable importance (drivers of heterogeneity)")
    imp = pd.Series(est.feature_importances_, index=MODS).sort_values(ascending=False)
    for k, v in imp.items():
        emit(f"    {PRETTY[k]:<22} {v:.3f}")

    # (2) best linear projection of CATE on standardized moderators
    emit("\n" + "=" * 68)
    emit("(2) Best linear projection: CATE ~ z(moderators), cluster-robust by city")
    Z = (X - X.mean()) / X.std(ddof=0)
    Z.columns = [f"z_{c}" for c in Z.columns]
    blp_df = pd.concat([W[["cate", "city"]].reset_index(drop=True),
                        Z.reset_index(drop=True)], axis=1)
    m = smf.ols("cate ~ " + " + ".join(Z.columns), data=blp_df).fit(
        cov_type="cluster", cov_kwds={"groups": blp_df["city"]})
    blp = (pd.DataFrame({"slope": m.params, "p": m.pvalues})
           .drop("Intercept").assign(mod=[PRETTY[c[2:]] for c in Z.columns])
           .sort_values("slope"))
    for _, r in blp.iterrows():
        star = "***" if r["p"] < .01 else "**" if r["p"] < .05 else "*" if r["p"] < .1 else ""
        emit(f"    {r['mod']:<22} slope={r['slope']:+.4f}  (p={r['p']:.3f}) {star}")
    emit("    [slope<0 = MORE suppression (more negative dY) as the moderator rises]")

    # (3) most-suppressed profile (bottom vs top CATE quartile)
    emit("\n" + "=" * 68)
    emit("(3) Most- vs least-suppressed profile (bottom vs top CATE quartile)")
    q = W["cate"].quantile([.25, .75])
    lo = W[W["cate"] <= q[.25]]; hi = W[W["cate"] >= q[.75]]
    emit(f"    {'moderator':<22}{'most-suppr':>12}{'least-suppr':>13}")
    for k in ["log_income0", "log_house_val0", "eduy0", "urban0", "homeowner_d0",
              "age_first", "first_by", "son"]:
        emit(f"    {PRETTY[k]:<22}{lo[k].mean():>12.3f}{hi[k].mean():>13.3f}")

    # (4) gender-absorption check
    emit("\n" + "=" * 68)
    emit("(4) Does the gender split survive? CATE by firstborn gender")
    emit(f"    daughter-first mean CATE = {W.loc[W['son']==0,'cate'].mean():+.4f}")
    emit(f"    son-first      mean CATE = {W.loc[W['son']==1,'cate'].mean():+.4f}")
    emit(f"    son importance rank = {list(imp.index).index('son')+1} of {len(MODS)}; "
         f"BLP z_son p = {m.pvalues['z_son']:.3f}")

    # (5) RE-TEST top forest hits in a clean city-clustered DiD (discipline)
    emit("\n" + "=" * 68)
    emit("(5) Re-test forest hits in a CLEAN person-year DiD (T x post x moderator)")
    emit("    daughter-first sample, city+year FE, cluster city; the honest test.")
    dgt = pv[pv["son"] == 0].copy()
    dgt["post"] = (dgt["year"] >= 2018).astype(int)
    top = list(blp.reindex(blp["slope"].abs().sort_values(ascending=False).index)["mod"])
    top_keys = [k for k in MODS if PRETTY[k] in top[:3]] + (["son"] if "son" not in
               [k for k in MODS if PRETTY[k] in top[:3]] else [])
    for k in dict.fromkeys(top_keys):
        if k == "son":
            continue                          # no variation within daughter-first
        d = dgt.dropna(subset=[k]).copy()
        if d[k].std() == 0 or len(d) < 200:
            continue
        d["zmod"] = (d[k] - d[k].mean()) / d[k].std()
        f = smf.ols("y ~ T:post + T:post:zmod + post:zmod + T:zmod + zmod "
                    "+ age + I(age**2) + dur + I(dur**2) + C(city) + C(year)",
                    data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})
        emit(f"    {PRETTY[k]:<20} T:post:zmod = {f.params['T:post:zmod']:+.4f} "
             f"(SE {f.bse['T:post:zmod']:.4f}, p={f.pvalues['T:post:zmod']:.3f}), "
             f"base T:post={f.params['T:post']:+.4f}")

    emit("\n" + "=" * 68)
    emit("CAVEAT: forest ATE/CATE inherit the DR-DiD overlap+~113-cluster limits; "
         "absolute CIs are wide. Read (1)-(4) as DISCOVERY signals; the clean DiD "
         "re-tests in (5) are the actual evidence.")
    out = os.path.join(OUT, "tables", "causal_forest_scan_c.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
