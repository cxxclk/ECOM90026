# ==========================================================================
# REPLICATION PACKAGE — roster_underreport_did.py
# Tier: CORE
# Paper items: Table 8 (Measurement: roster miss rate)
# Reads: output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/roster_underreport_did.txt
# Imports local modules: annual_firstborn_gender
# Run order: step 35 in run_all.sh
# ==========================================================================

"""ML project #2 — roster underreporting: profile + treatment-orthogonality DID.

Closes the flagged caveat (CLAUDE.md, replication section): contemporaneous
famconf rosters underreport recent births by 20-35% (worse in early waves);
wave-level results lean on year FE absorbing it "unless underreporting
correlates with treatment" — that unless was never tested. Here:

  1. TRUTH label: for each person-wave (pid, w), births with cby in (w-2, w]
     according to the pid's LATEST famconf roster (require last_wave >= w+2 so
     a strictly later roster exists to serve as truth; wave 2022 untestable).
  2. REPORTED: the pipeline's own wave-level new_birth / n_new_births — the
     exact variable the legacy wave-level DIDs used.
  3. miss = 1{true birth in window & new_birth != 1}.
  4. Orthogonality DID: miss ~ T:post + city+year FE, cluster city, among
     true-birth person-waves (Post=2016 legacy convention + 2018 variant),
     + yearly event study.
  5. Bias decomposition: DID on true_birth vs DID on new_birth in the SAME
     sample — the coefficient gap IS the measurement bias in the wave-level DID.
  6. Classifier profile: HistGBM, city-GroupKFold OOF AUC; then add
     {T, post, T*post} and check the AUC gain (expected ~0 if orthogonal).

Sample: married women 20-45 (analytic convention), unregulated prefectures,
baseline-city treatment. All-women variant reported as robustness.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annual_firstborn_gender import load_rosters, OUT  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.RandomState(42)

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def build():
    roster, last_wave = load_rosters()  # latest-wave roster per pid + last_wave

    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "gender", "birthy", "age", "marital", "eduy",
                 "income", "urban", "homeowner", "city_code",
                 "new_birth", "n_new_births", "n_children_fam"])
    women = panel[panel["gender"] == 0].copy()

    # baseline city -> treatment (same convention as hazard designs)
    base = (women.dropna(subset=["city_code"]).sort_values(["pid", "year"])
            .groupby("pid")["city_code"].first().rename("base_city"))
    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    tr = (base.reset_index()
          .merge(hpr[["city_code", "is_regulated", "treat_hpr"]],
                 left_on="base_city", right_on="city_code", how="inner"))
    tr = tr[tr["is_regulated"] == 0][["pid", "base_city", "treat_hpr"]]

    d = women.merge(tr, on="pid").merge(last_wave, on="pid")

    # truth from latest roster: births in (w-2, w], parity before window,
    # firstborn gender
    r = roster.sort_values(["pid", "cby"]).copy()
    fb = r.groupby("pid").first()[["cby", "cgender"]].rename(
        columns={"cby": "first_by", "cgender": "fb_son"})
    d = d.merge(fb, on="pid", how="left")

    counts = []
    for w in sorted(d["year"].unique()):
        in_win = r[(r["cby"] > w - 2) & (r["cby"] <= w)]
        pre = r[r["cby"] <= w - 2]
        c = pd.DataFrame({"pid": d.loc[d["year"] == w, "pid"].unique()})
        c["year"] = w
        c = (c.merge(in_win.groupby("pid").size().rename("n_true"),
                     on="pid", how="left")
             .merge(pre.groupby("pid").size().rename("parity_pre"),
                    on="pid", how="left"))
        counts.append(c)
    d = d.merge(pd.concat(counts), on=["pid", "year"], how="left")
    d["n_true"] = d["n_true"].fillna(0)
    d["parity_pre"] = d["parity_pre"].fillna(0)

    d["true_birth"] = (d["n_true"] >= 1).astype(float)
    d["rep_birth"] = d["new_birth"].fillna(0)
    d["miss"] = ((d["true_birth"] == 1) & (d["rep_birth"] < 1)).astype(float)
    d["false_pos"] = ((d["true_birth"] == 0) & (d["rep_birth"] >= 1)).astype(float)

    d["T"] = d["treat_hpr"].astype(int)
    d["city"] = d["base_city"].astype(int)
    return d


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  mean(y)={m.model.endog.mean():.4f}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    d = build()

    # truth requires a strictly later roster; drop 2010 (new_birth undefined
    # there — no previous wave, so the pipeline codes it NaN and miss would
    # be mechanically 1 for every true birth)
    d = d[(d["last_wave"] >= d["year"] + 2) & (d["year"] >= 2012)].copy()
    # analytic convention
    an = d[(d["marital"] == 2) & d["age"].between(20, 45)].copy()

    emit("Roster underreporting: profile + treatment-orthogonality DID")
    emit("=" * 72)
    emit(f"person-waves with later-roster truth: all women {len(d):,}; "
         f"married 20-45 {len(an):,}")

    emit("\n== 1. Underreporting rates by wave (married women 20-45) ==")
    tab = (an[an["true_birth"] == 1]
           .groupby("year")
           .agg(n_births=("miss", "size"), miss_rate=("miss", "mean")))
    fp = an.groupby("year")["false_pos"].mean().rename("false_pos_rate")
    emit(tab.join(fp).round(4).to_string())
    emit(f"\noverall miss rate: {an.loc[an['true_birth']==1,'miss'].mean():.3f}"
         f"   false-positive rate: {an['false_pos'].mean():.4f}")

    births = an[an["true_birth"] == 1].copy()
    emit("\n== 2. Miss rate by wave x T (conditional on true birth) ==")
    emit(births.pivot_table(index="year", columns="T", values="miss",
                            aggfunc=["mean", "size"]).round(3).to_string())

    FE = " + C(city) + C(year)"
    emit("\n== 3. Orthogonality DID: miss ~ T:post + city+year FE ==")
    for post_yr in (2016, 2018):
        births["post"] = (births["year"] >= post_yr).astype(int)
        fit(f"miss ~ T:post{FE}", births,
            f"conditional miss, Post={post_yr}", ["T:post"])

    # event study, omitted 2014 (last full pre wave)
    ev = [y for y in sorted(births["year"].unique()) if y != 2014]
    for k in ev:
        births[f"Ty{k}"] = births["T"] * (births["year"] == k).astype(int)
    terms = " + ".join(f"Ty{k}" for k in ev)
    m = smf.ols(f"miss ~ {terms}{FE}", data=births).fit(
        cov_type="cluster", cov_kwds={"groups": births["city"]})
    emit("\n[event study on conditional miss, omitted 2014]")
    for k in ev:
        emit(f"  T x {k}   {m.params[f'Ty{k}']:+.4f}  "
             f"(SE {m.bse[f'Ty{k}']:.4f}, p={m.pvalues[f'Ty{k}']:.3f})")

    emit("\n== 4. Bias decomposition: DID(true_birth) vs DID(new_birth), "
         "same sample ==")
    for post_yr in (2016, 2018):
        an["post"] = (an["year"] >= post_yr).astype(int)
        mt = fit(f"true_birth ~ T:post{FE}", an,
                 f"TRUE birth DID, Post={post_yr}", ["T:post"])
        mr = fit(f"rep_birth ~ T:post{FE}", an,
                 f"REPORTED birth DID, Post={post_yr}", ["T:post"])
        gap = mt.params["T:post"] - mr.params["T:post"]
        md = fit(f"I(true_birth - rep_birth) ~ T:post{FE}", an,
                 f"DIFF (true - reported) DID, Post={post_yr}", ["T:post"])
        emit(f"  -> measurement bias in wave-level DID: {gap:+.4f} "
             f"(direct diff-outcome p={md.pvalues['T:post']:.3f})")

    emit("\n== 5. Second-birth margin check (parity_pre==1, the headline "
         "risk set) ==")
    b2 = births[births["parity_pre"] == 1].copy()
    for post_yr in (2016, 2018):
        b2["post"] = (b2["year"] >= post_yr).astype(int)
        fit(f"miss ~ T:post{FE}", b2, f"2nd-birth miss, Post={post_yr}",
            ["T:post"])
    b2d = b2[b2["fb_son"] == 0].copy()
    b2d["post"] = (b2d["year"] >= 2016).astype(int)
    fit(f"miss ~ T:post{FE}", b2d, "2nd-birth miss, daughter-first, Post=2016",
        ["T:post"])

    # ── classifier profile ──────────────────────────────────────────────
    emit("\n== 6. Classifier: who gets missed? (HistGBM, city-GroupKFold "
         "OOF) ==")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.inspection import permutation_importance

    cb = births.copy()
    cb["log_inc"] = np.log1p(cb["income"].clip(lower=0))
    cb["post"] = (cb["year"] >= 2016).astype(int)
    cb["Tpost"] = cb["T"] * cb["post"]
    # NOTE: n_children_fam excluded — it counts the CONTEMPORANEOUS roster's
    # children, so it is mechanically lower when the birth is missed (leakage)
    base_feats = ["age", "eduy", "urban", "log_inc", "homeowner",
                  "parity_pre", "fb_son", "year"]
    trt_feats = ["T", "post", "Tpost"]

    y = cb["miss"].values
    groups = cb["city"].values
    gkf = GroupKFold(n_splits=5)

    def oof_auc(feats):
        X = cb[feats].values.astype(float)
        oof = np.zeros(len(cb))
        for tr_i, te_i in gkf.split(X, y, groups):
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=3,
                random_state=42)
            clf.fit(X[tr_i], y[tr_i])
            oof[te_i] = clf.predict_proba(X[te_i])[:, 1]
        return roc_auc_score(y, oof), oof

    auc_b, oof_b = oof_auc(base_feats)
    auc_t, _ = oof_auc(base_feats + trt_feats)
    emit(f"  OOF AUC, covariates only:        {auc_b:.4f}")
    emit(f"  OOF AUC, + T/post/T*post:        {auc_t:.4f}   "
         f"(gain {auc_t - auc_b:+.4f})")

    # full-fit permutation importance for the profile (descriptive)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=3, random_state=42)
    Xall = cb[base_feats + trt_feats].fillna(-9).values.astype(float)
    clf.fit(Xall, y)
    imp = permutation_importance(clf, Xall, y, n_repeats=10, random_state=42,
                                 scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)
    emit("\n  permutation importance (AUC drop, in-sample fit):")
    for i in order:
        emit(f"    {(base_feats + trt_feats)[i]:<16} "
             f"{imp.importances_mean[i]:+.4f}")

    emit("\n  miss profile (means, missed vs reported births):")
    prof = (cb.groupby("miss")[["age", "eduy", "urban", "log_inc",
                                "parity_pre", "fb_son", "year", "T"]]
            .mean().round(3))
    emit(prof.to_string())

    # ── robustness: all women / strict 2022 truth ───────────────────────
    emit("\n== 7. Robustness ==")
    allb = d[(d["true_birth"] == 1) & d["age"].between(20, 45)].copy()
    for post_yr in (2016, 2018):
        allb["post"] = (allb["year"] >= post_yr).astype(int)
        fit(f"miss ~ T:post{FE}", allb,
            f"ALL women 20-45, Post={post_yr}", ["T:post"])
    strict = births[births["last_wave"] == 2022].copy()
    for post_yr in (2016, 2018):
        strict["post"] = (strict["year"] >= post_yr).astype(int)
        fit(f"miss ~ T:post{FE}", strict,
            f"truth from 2022 roster only, Post={post_yr}", ["T:post"])

    out = os.path.join(OUT, "tables", "roster_underreport_did.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
