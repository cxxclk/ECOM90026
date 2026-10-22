# ==========================================================================
# REPLICATION PACKAGE — prognostic_gates.py
# Tier: CORE
# Paper items: Figure 5 (Effects by fifth of predicted likelihood); Figure 7 (Pre-train, post-predict); Table E4 (Ten fold assignments); Table E5 (Top versus bottom fifth)
# Reads: (via imported modules)
# Writes: output/prognostic_scores.csv, output/tables/prognostic_gates.txt
# Imports local modules: cddf_spine
# Run order: step 5 in run_all.sh
# ==========================================================================

"""Prognostic-score GATES: who was most affected = who HAD the most to lose?

Follow-up to cddf_spine.py (2026-07-21). The effect-proxy CDDF run showed:
  (a) RESIDUAL version (known moderators excluded): no ranked heterogeneity
      at all (BLP p=.67) -> completeness certificate.
  (b) FULL version: the T-learner "effect" score was actually a baseline-
      fertility-propensity index in disguise (shrinkage artifact), its
      predicted sign inverted — but the REALIZED GATES gradient along it was
      monotone: suppression concentrates at the high-propensity end
      (young-at-first-birth, poor, rural-leaning, high son-preference).
So estimate the honest version directly: risk-stratified heterogeneity
(Kent-Hayward style), the standard answer when effect-CATE is unestimable.

Index = OOF-predicted PRE-SHOCK second-birth propensity:
  HistGBM trained ONLY on 2009-2015 woman-years (strictly pre-announcement),
  city-fold cross-fitted (a woman's own city never trains her score),
  repeated over SEEDS partitions, consensus = median score per woman.
  Grouping is a function of baseline covariates only -> subgroup DIDs on
  covariate-defined groups are fully legitimate (no outcome selection).

Then: GATES quintile DIDs (G5 = highest propensity), absolute AND relative
(gamma/base) gradients, per-group pre-trend gate + event studies for the top
groups, concentration decomposition (share of implied lost births by group),
CLAN profile, and unification checks: does the firstborn-gender margin (and
the norm gradient) live INSIDE the high-propensity stratum?
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                  # noqa: E402

SEEDS = int(os.environ.get("PROG_SEEDS", 10))
FOLDS = 5
CTRL = cs.CTRL
FE = cs.FE
XCOLS = cs.XCOLS_A

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def propensity_scores(df, Xmat, seed):
    """OOF pre-shock second-birth propensity, city-fold cross-fitted."""
    rng = np.random.RandomState(seed)
    cities = df[["city", "T"]].drop_duplicates()
    fold = {}
    for _, grp in cities.groupby("T"):
        idx = grp["city"].values.copy()
        rng.shuffle(idx)
        for i, c in enumerate(idx):
            fold[c] = i % FOLDS
    fvec = df["city"].map(fold).values
    pre = (df["year"] <= 2015).values
    yv = df["y"].values

    pred = np.full(len(df), np.nan)
    for f in range(FOLDS):
        te = fvec == f
        tr = (~te) & pre
        m = HistGradientBoostingRegressor(random_state=seed, max_depth=3,
                                          max_iter=200, min_samples_leaf=50)
        m.fit(Xmat[tr], yv[tr])
        pred[te] = m.predict(Xmat[te])
    # OOF fit quality on strictly-pre rows
    r2 = 1 - np.nanvar(yv[pre] - pred[pre]) / np.nanvar(yv[pre])
    return pd.Series(pred, index=df.index).groupby(df["pid"]).mean(), r2


def attach(df, s):
    d = df.copy()
    d["prop"] = d["pid"].map(s)
    wl = d.drop_duplicates("pid")
    qs = wl["prop"].quantile([0.2, 0.4, 0.6, 0.8]).values
    d["grp"] = np.searchsorted(qs, d["prop"].values) + 1   # 5 = most fertile
    for k in range(1, 6):
        d[f"g{k}"] = (d["grp"] == k).astype(int)
    d["propz"] = (d["prop"] - wl["prop"].mean()) / wl["prop"].std()
    return d


def gates(d):
    tri = " + ".join(f"T:post:g{k}" for k in range(1, 6))
    two = " + ".join(f"T:g{k} + post:g{k} + g{k}" for k in range(1, 5))
    m = smf.ols(f"y ~ {tri} + {two} + {CTRL}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    gam = [m.params[f"T:post:g{k}"] for k in range(1, 6)]
    pv = [m.pvalues[f"T:post:g{k}"] for k in range(1, 6)]
    tt = m.t_test("T:post:g5 - T:post:g1")
    return gam, pv, float(np.ravel(tt.effect)[0]), float(np.ravel(tt.pvalue)[0])


def main():
    df = cs.assemble()
    emit("PROGNOSTIC-SCORE GATES — DID gradient along PREDICTED pre-shock "
         "second-birth propensity")
    emit(f"one-child risk set, Post=2018, city+year FE, cluster city; "
         f"propensity: HistGBM on 2009-2015 rows only, {FOLDS} city folds x "
         f"{SEEDS} splits, median consensus")
    emit(f"  rows={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}")

    Xmat = df[XCOLS + ["age", "dur"]].astype(float).values
    scores, r2s = [], []
    for seed in range(SEEDS):
        s, r2 = propensity_scores(df, Xmat, seed)
        scores.append(s.rename(f"s{seed}"))
        r2s.append(r2)
        print(f"  [seed {seed:2d}] OOF R2 (pre rows) = {r2:.4f}", flush=True)
    smed = pd.concat(scores, axis=1).median(axis=1)
    emit(f"OOF R2 on pre rows: median {np.median(r2s):.4f}")

    d = attach(df, smed)
    wl = d.drop_duplicates("pid")
    # validation: does the index track actual pre-period hazard?
    wl_pre = d[d["year"] <= 2015].groupby("pid")["y"].mean().rename("prehaz")
    v = wl.merge(wl_pre, on="pid", how="inner")
    emit(f"validation: corr(consensus propensity, actual 2009-15 hazard) = "
         f"{v[['prop', 'prehaz']].corr().iloc[0, 1]:.3f}  "
         f"(women with pre rows: {len(v):,})")

    # ---- GATES ----------------------------------------------------------
    gam, pv, diff, diff_p = gates(d)
    emit("\nGATES by propensity quintile (G5 = highest predicted propensity):")
    emit(f"  {'grp':<4}{'gamma':>9}{'p':>8}{'base pre-haz':>14}"
         f"{'relative':>10}{'post wy':>9}{'lost births':>12}")
    lost = []
    for k in range(1, 6):
        g = d[d["grp"] == k]
        base = g.loc[g["post"] == 0, "y"].mean()
        npost = int(((g["post"] == 1) & (g["T"] == 1)).sum())
        lb = -gam[k - 1] * npost
        lost.append(lb)
        emit(f"  G{k:<3}{gam[k-1]:>+9.4f}{pv[k-1]:>8.3f}{base:>14.3f}"
             f"{gam[k-1]/base:>+10.1%}{npost:>9,}{lb:>12.1f}")
    tot = sum(lost)
    emit(f"  G5 - G1 difference: {diff:+.4f} (p={diff_p:.3f})")
    if tot > 0:
        shares = [max(x, 0) / tot for x in lost]
        emit("  concentration: share of implied lost births "
             "(treated post woman-years x -gamma, negatives floored): "
             + "  ".join(f"G{k+1}:{shares[k]:.0%}" for k in range(5)))

    # per-seed stability of the GATES gradient
    emit("\nper-split stability (gamma_5, gamma_1, G5-G1 p):")
    for seed in range(SEEDS):
        ds = attach(df, scores[seed])
        gm, _, df_, dp_ = gates(ds)
        emit(f"  seed {seed:2d}: G5 {gm[4]:+.4f}  G1 {gm[0]:+.4f}  "
             f"diff {df_:+.4f} (p={dp_:.3f})")

    # ---- pre-trend gate + event studies ---------------------------------
    emit("\npre-trend gate (# of 2010-14 T x year p<.10):")
    gate = {}
    for k in range(1, 6):
        s = d[d["grp"] == k].copy()
        for yr in range(2010, 2015):
            s[f"Ty{yr}"] = s["T"] * (s["year"] == yr).astype(int)
        s["Tpost"] = s["T"] * s["post"]
        terms = " + ".join(f"Ty{yr}" for yr in range(2010, 2015))
        m = smf.ols(f"y ~ {terms} + Tpost + {CTRL}{FE}", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["city"]})
        gate[k] = sum(m.pvalues[f"Ty{yr}"] < 0.10 for yr in range(2010, 2015))
    emit("  " + "  ".join(f"G{k}:{v}" for k, v in gate.items()))
    for k in (5, 4):
        sub = d[d["grp"] == k]
        ev = [y for y in range(2010, 2023) if y != 2015]
        s = sub.copy()
        for yy in ev:
            s[f"Ty{yy}"] = s["T"] * (s["year"] == yy).astype(int)
        terms = " + ".join(f"Ty{yy}" for yy in ev)
        m = smf.ols(f"y ~ {terms} + {CTRL}{FE}", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["city"]})
        emit(f"\n[event study G{k}]  N={int(m.nobs):,}  "
             f"women={s['pid'].nunique():,}  mean(y)={s['y'].mean():.4f}")
        for yy in ev:
            tag = " (pre)" if yy < 2016 else ""
            emit(f"  T x {yy}   {m.params[f'Ty{yy}']:+.4f}  "
                 f"(SE {m.bse[f'Ty{yy}']:.4f}, p={m.pvalues[f'Ty{yy}']:.3f}){tag}")

    # ---- CLAN: G5 vs G1 --------------------------------------------------
    wl = d.drop_duplicates("pid").copy()
    wl["pre_haz"] = d[d["post"] == 0].groupby("pid")["y"].mean().reindex(
        wl["pid"]).values
    ext = wl[wl["grp"].isin([1, 5])].copy()
    ext["g5"] = (ext["grp"] == 5).astype(int)
    emit(f"\nCLAN — highest-propensity G5 (n={int((wl['grp']==5).sum()):,}) "
         f"vs lowest G1 (n={int((wl['grp']==1).sum()):,}):")
    emit(f"  {'covariate':<18}{'G5 mean':>10}{'G1 mean':>10}{'diff':>9}{'p':>8}")
    for x in XCOLS + ["T", "pre_haz"]:
        sub = ext[ext[x].notna()]
        if sub[x].nunique() < 2:
            continue
        m = smf.ols(f"{x} ~ g5", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["city"]})
        g1m = m.params["Intercept"]
        emit(f"  {x:<18}{g1m + m.params['g5']:>10.3f}{g1m:>10.3f}"
             f"{m.params['g5']:>+9.3f}{m.pvalues['g5']:>8.3f}")

    # ---- unification: known margins INSIDE the active stratum ------------
    emit("\nUnification checks (top-2 propensity quintiles = 'active margin'"
         " vs bottom-3):")
    for name, sub in [("TOP (G4+G5)", d[d["grp"] >= 4]),
                      ("BOTTOM (G1-G3)", d[d["grp"] <= 3])]:
        m = smf.ols(f"y ~ T:post + T:post:son + post:son + T:son + son + "
                    f"{CTRL}{FE}", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["city"]})
        emit(f"  [{name}] daughter DID (T:post) "
             f"{m.params['T:post']:+.4f} (p={m.pvalues['T:post']:.3f}); "
             f"DDD T:post:son {m.params['T:post:son']:+.4f} "
             f"(p={m.pvalues['T:post:son']:.3f})  N={int(m.nobs):,}")
    # continuous version: does propensity absorb or coexist with the
    # firstborn-gender margin? (daughter arm only, propz interaction)
    da = d[d["son"] == 0].copy()
    m = smf.ols(f"y ~ T:post + T:post:propz + T:propz + post:propz + propz + "
                f"{CTRL}{FE}", data=da).fit(
        cov_type="cluster", cov_kwds={"groups": da["city"]})
    emit(f"  [daughter arm] T:post {m.params['T:post']:+.4f} "
         f"(p={m.pvalues['T:post']:.3f});  T:post:propz "
         f"{m.params['T:post:propz']:+.4f} (p={m.pvalues['T:post:propz']:.3f})")
    sa = d[d["son"] == 1].copy()
    m = smf.ols(f"y ~ T:post + T:post:propz + T:propz + post:propz + propz + "
                f"{CTRL}{FE}", data=sa).fit(
        cov_type="cluster", cov_kwds={"groups": sa["city"]})
    emit(f"  [son arm]      T:post {m.params['T:post']:+.4f} "
         f"(p={m.pvalues['T:post']:.3f});  T:post:propz "
         f"{m.params['T:post:propz']:+.4f} (p={m.pvalues['T:post:propz']:.3f})")

    keep = wl[["pid", "prop", "grp"]]
    keep.to_csv(os.path.join(OUT, "prognostic_scores.csv"), index=False)
    emit("\nscores saved -> output/prognostic_scores.csv")
    out = os.path.join(OUT, "tables", "prognostic_gates.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
