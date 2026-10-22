# ==========================================================================
# REPLICATION PACKAGE — cddf_spine.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (T-learner effect proxy)
# Reads: output/panel_individual.csv
# Writes: output/cddf_scores.csv, output/tables/cddf_spine.txt
# Imports local modules: annual_firstborn_gender, gender_norm_index, semisup_typology
# Run order: step 6 in run_all.sh
# ==========================================================================

"""CDDF generic-ML heterogeneity spine on the annual second-birth hazard DID.

Goal (2026-07-21): effect-targeted discovery — let ML rank WHO is most
affected by the HPR spillover shock, with honest inference, instead of
clustering people in covariate space (semisup/trajectory scans) or point-
estimating individual CATEs (causal forest — both failed for known reasons).

Method = Chernozhukov-Demirer-Duflo-Fernandez-Val (2018) adapted to DID:
  1. Effect proxy: four-cell T-learner. Fit E[y | X, cell] separately in the
     {T,C} x {pre,post} cells with HistGBM (NaN-native), CROSS-FITTED BY CITY
     (5 folds, stratified by T; a woman's own city never trains her score).
     delta_it = [m11 - m10](x) - [m01 - m00](x) evaluated at the SAME x
     -> a conditional-DID proxy; woman-level score s_i = mean_t delta_it.
     Only the RANKING of s is used downstream (robust to noisy levels).
  2. BLP: y ~ T:post + T:post:sz + (T:sz + post:sz + sz) + age/dur controls
     + city + year FE, cluster city. T:post:sz != 0 <=> ML found *ranked*
     heterogeneity. (DID version of CDDF's BLP: heterogeneity rides on the
     conditional-parallel-trends-within-s-strata assumption, same as every
     interacted moderator spec in the project.)
  3. GATES: quintiles of s (G1 = most suppressed). Group-specific DIDs
     gamma_g via full interaction (T:post:Gg for all g; T:Gg, post:Gg, Gg
     mains, city+year FE). Test gamma_1 - gamma_5.
  4. Repeated splits: SEEDS random city partitions; report MEDIAN estimates,
     median p, and CDDF adjusted p = min(1, 2 * median p).
  5. CLAN: covariate means of G1 vs G5 (consensus groups = median-s across
     seeds; each seed's s is out-of-fold, so the consensus stays honest),
     clustered-by-city t-tests.
Two covariate menus:
  A FULL:      everything incl. known moderators (firstborn gender, norms,
               preferences, affordability) -> does free search re-discover
               the theory picture, and how big is the top-group effect?
  B RESIDUAL:  known moderators EXCLUDED -> is there any ranked heterogeneity
               we have NOT already found?  (BLP null here = completeness
               certificate for the four-layer arc.)
Diagnostics on consensus groups: per-group pre-trend gate (2010-14 T x year,
count p<.10), full event study for G1/G5, overlap with the theory group
(daughter-first x top-norm-tercile city).
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
from annual_firstborn_gender import build_hazard_panel   # noqa: E402
from semisup_typology import build_baseline              # noqa: E402
from gender_norm_index import build_norm_index           # noqa: E402

SEEDS = int(os.environ.get("CDDF_SEEDS", 20))
FOLDS = 5
CTRL = "age + I(age**2) + dur + I(dur**2)"
FE = " + C(city) + C(year)"

XCOLS_A = ["son", "birthy", "age_fb", "eduy", "urban", "log_faminc_pc",
           "log_inc", "hv_inc", "homeowner", "ideal_kids", "son_pref",
           "housework", "city_son_norm", "trad_pc1", "city_urban_share",
           "city_linc"]
# B: strip every known-moderator family — firstborn gender, own/city son
# preference & tradition, own fertility ideal, affordability (incomes, house
# value ratio, ownership), city income.
XCOLS_B = ["birthy", "age_fb", "eduy", "urban", "housework",
           "city_urban_share"]

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def assemble():
    """One-child risk-set woman-year panel + woman-level baseline features."""
    panel = build_hazard_panel()
    panel["post"] = (panel["year"] >= 2018).astype(int)

    base = build_baseline()
    keep = ["pid", "eduy", "urban", "log_faminc_pc", "log_inc", "hv_inc",
            "homeowner", "ideal_kids", "son_pref", "housework",
            "city_son_norm", "city_urban_share", "age_fb"]
    base = base[keep].drop_duplicates("pid")

    city = build_norm_index()[["city_code", "trad_pc1"]]
    pan14 = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "city_code", "fam_income_pc"])
    p14 = pan14[pan14["year"] == 2014].copy()
    p14["lfipc"] = np.log1p(p14["fam_income_pc"].clip(lower=0))
    clinc = (p14.groupby("city_code")["lfipc"].mean()
             .rename("city_linc").reset_index())

    df = (panel.merge(base, on="pid", how="left")
                .merge(city, on="city_code", how="left")
                .merge(clinc, on="city_code", how="left"))

    # FE-residualize the outcome ONCE for the proxy stage: strip city levels,
    # secular year effects, and lifecycle (age/dur) so the four-cell learner
    # sees only interaction-relevant variation. Raw-y T-learners load on
    # baseline hazard LEVELS via differential shrinkage across cells
    # (regularization bias) — the smoke run showed s ~ pre-period hazard and
    # an inverted realized ranking; residualizing removes the level signal.
    m0 = smf.ols(f"y ~ {CTRL}{FE}", data=df).fit()
    df["yres"] = df["y"] - m0.fittedvalues
    return df


def proxy_scores(df, Xmat, seed):
    """One split: city folds -> OOF four-cell T-learner -> woman-level s."""
    rng = np.random.RandomState(seed)
    cities = df[["city", "T"]].drop_duplicates()
    fold = {}
    for _, grp in cities.groupby("T"):
        idx = grp["city"].values.copy()
        rng.shuffle(idx)
        for i, c in enumerate(idx):
            fold[c] = i % FOLDS
    fvec = df["city"].map(fold).values

    delta = np.full(len(df), np.nan)
    Tv, Pv, yv = df["T"].values, df["post"].values, df["yres"].values
    for f in range(FOLDS):
        te = fvec == f
        tr = ~te
        preds = {}
        for tval in (0, 1):
            for pval in (0, 1):
                cell = tr & (Tv == tval) & (Pv == pval)
                m = HistGradientBoostingRegressor(
                    random_state=seed, max_depth=3, max_iter=150,
                    min_samples_leaf=50)
                m.fit(Xmat[cell], yv[cell])
                preds[(tval, pval)] = m.predict(Xmat[te])
        delta[te] = ((preds[(1, 1)] - preds[(1, 0)])
                     - (preds[(0, 1)] - preds[(0, 0)]))
    return pd.Series(delta, index=df.index).groupby(df["pid"]).mean()


def attach_score(df, s):
    d = df.copy()
    d["s"] = d["pid"].map(s)
    wl = d.drop_duplicates("pid")
    d["sz"] = (d["s"] - wl["s"].mean()) / wl["s"].std()
    qs = wl["s"].quantile([0.2, 0.4, 0.6, 0.8]).values
    d["grp"] = np.searchsorted(qs, d["s"].values) + 1   # 1 = most suppressed
    for k in range(1, 6):
        d[f"g{k}"] = (d["grp"] == k).astype(int)
    return d


def blp_gates(d):
    """BLP + GATES on a scored panel; returns the seed-level stats dict."""
    r = {}
    m = smf.ols(f"y ~ T:post + T:post:sz + T:sz + post:sz + sz + {CTRL}{FE}",
                data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})
    r["blp"] = m.params["T:post:sz"]
    r["blp_p"] = m.pvalues["T:post:sz"]

    tri = " + ".join(f"T:post:g{k}" for k in range(1, 6))
    two = " + ".join(f"T:g{k} + post:g{k} + g{k}" for k in range(1, 5))
    mg = smf.ols(f"y ~ {tri} + {two} + {CTRL}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    r["gam"] = [mg.params[f"T:post:g{k}"] for k in range(1, 6)]
    r["gam_p"] = [mg.pvalues[f"T:post:g{k}"] for k in range(1, 6)]
    tt = mg.t_test("T:post:g1 - T:post:g5")
    r["diff"] = float(np.ravel(tt.effect)[0])
    r["diff_p"] = float(np.ravel(tt.pvalue)[0])
    return r


def pretrend_gate(d):
    """Per consensus group: # of 2010-14 T x year coefs with p<.10."""
    out = {}
    for k in range(1, 6):
        s = d[d["grp"] == k].copy()
        for yr in range(2010, 2015):
            s[f"Ty{yr}"] = s["T"] * (s["year"] == yr).astype(int)
        s["Tpost"] = s["T"] * s["post"]
        terms = " + ".join(f"Ty{yr}" for yr in range(2010, 2015))
        m = smf.ols(f"y ~ {terms} + Tpost + {CTRL}{FE}", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["city"]})
        out[k] = sum(m.pvalues[f"Ty{yr}"] < 0.10 for yr in range(2010, 2015))
    return out


def event_study(d, label):
    ev = [y for y in range(2010, 2023) if y != 2015]
    s = d.copy()
    for k in ev:
        s[f"Ty{k}"] = s["T"] * (s["year"] == k).astype(int)
    terms = " + ".join(f"Ty{k}" for k in ev)
    m = smf.ols(f"y ~ {terms} + {CTRL}{FE}", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s["city"]})
    emit(f"\n[event study: {label}]  N={int(m.nobs):,}  "
         f"women={s['pid'].nunique():,}  mean(y)={s['y'].mean():.4f}")
    for k in ev:
        tag = " (pre)" if k < 2016 else ""
        emit(f"  T x {k}   {m.params[f'Ty{k}']:+.4f}  "
             f"(SE {m.bse[f'Ty{k}']:.4f}, p={m.pvalues[f'Ty{k}']:.3f}){tag}")


def clan(d, xcols):
    """G1 vs G5 covariate means, clustered-by-city t-tests (woman level)."""
    wl = d.drop_duplicates("pid").copy()
    wl["pre_haz"] = d[d["post"] == 0].groupby("pid")["y"].mean().reindex(
        wl["pid"]).values
    ext = wl[wl["grp"].isin([1, 5])].copy()
    ext["g1"] = (ext["grp"] == 1).astype(int)
    emit(f"\nCLAN — most-affected G1 (n={int((wl['grp']==1).sum()):,}) vs "
         f"least-affected G5 (n={int((wl['grp']==5).sum()):,}), "
         f"clustered-by-city t-tests:")
    emit(f"  {'covariate':<18}{'G1 mean':>10}{'G5 mean':>10}"
         f"{'diff':>9}{'p':>8}")
    for x in xcols + ["T", "pre_haz"]:
        sub = ext[ext[x].notna()]
        if sub[x].nunique() < 2:
            continue
        m = smf.ols(f"{x} ~ g1", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["city"]})
        g5m = m.params["Intercept"]
        dif = m.params["g1"]
        emit(f"  {x:<18}{g5m + dif:>10.3f}{g5m:>10.3f}"
             f"{dif:>+9.3f}{m.pvalues['g1']:>8.3f}")


def run_version(df, xcols, tag):
    emit("\n" + "=" * 74)
    emit(f"VERSION {tag}: features = {', '.join(xcols)}")
    Xmat = df[xcols + ["age", "dur"]].astype(float).values

    recs, scores = [], []
    for seed in range(SEEDS):
        s = proxy_scores(df, Xmat, seed)
        d = attach_score(df, s)
        r = blp_gates(d)
        recs.append(r)
        scores.append(s.rename(f"s{seed}"))
        print(f"  [{tag} seed {seed:2d}] BLP {r['blp']:+.4f} (p={r['blp_p']:.3f})"
              f"  G1 {r['gam'][0]:+.4f}  G5 {r['gam'][4]:+.4f}"
              f"  diff p={r['diff_p']:.3f}", flush=True)

    med = lambda key: np.median([r[key] for r in recs])
    medk = lambda key, k: np.median([r[key][k] for r in recs])
    emit(f"\n--- {tag}: median over {SEEDS} city-fold splits "
         f"(CDDF adjusted p = min(1, 2 x median p)) ---")
    emit(f"BLP  T:post:sz      {med('blp'):+.4f}   median p={med('blp_p'):.3f}"
         f"   adj p={min(1, 2 * med('blp_p')):.3f}")
    emit("GATES group-specific DID (G1 = ML-most-suppressed):")
    for k in range(5):
        emit(f"  G{k+1}  gamma={medk('gam', k):+.4f}   "
             f"median p={medk('gam_p', k):.3f}")
    emit(f"G1 - G5 difference  {med('diff'):+.4f}   median p={med('diff_p'):.3f}"
         f"   adj p={min(1, 2 * med('diff_p')):.3f}")

    smed = pd.concat(scores, axis=1).median(axis=1)
    return smed, recs


def consensus_block(df, smed, xcols, tag):
    emit(f"\n--- {tag}: CONSENSUS diagnostics (median-s groups; "
         f"each seed's s is OOF) ---")
    d = attach_score(df, smed)
    wl = d.drop_duplicates("pid")
    emit(f"consensus s: mean={wl['s'].mean():+.4f}  sd={wl['s'].std():.4f}  "
         f"share<0 = {(wl['s'] < 0).mean():.2f}")
    r = blp_gates(d)
    emit(f"consensus BLP {r['blp']:+.4f} (p={r['blp_p']:.3f}); GATES "
         + "  ".join(f"G{k+1} {r['gam'][k]:+.4f}(p={r['gam_p'][k]:.3f})"
                     for k in range(5))
         + f";  G1-G5 {r['diff']:+.4f} (p={r['diff_p']:.3f})")
    gates_pre = pretrend_gate(d)
    emit("pre-trend gate (# of 2010-14 T x year p<.10, by group): "
         + "  ".join(f"G{k}:{v}" for k, v in gates_pre.items()))
    event_study(d[d["grp"] == 1], f"{tag} G1 most-affected")
    event_study(d[d["grp"] == 5], f"{tag} G5 least-affected")
    clan(d, xcols)

    # ranking interpretability: corr of s with each feature (woman level)
    emit("\ncorr(s, feature) at woman level (negative = feature predicts "
         "MORE suppression):")
    for x in xcols:
        c = wl[["s", x]].astype(float).corr().iloc[0, 1]
        emit(f"  {x:<18}{c:+.3f}")

    # overlap with the theory group: daughter-first x top-norm-tercile city
    cutoff = (wl.drop_duplicates("city")["city_son_norm"]
              .quantile(2 / 3))
    wl2 = wl.assign(theory=((wl["son"] == 0)
                            & (wl["city_son_norm"] >= cutoff)).astype(int))
    g1 = wl2[wl2["grp"] == 1]
    th = wl2[wl2["theory"] == 1]
    emit(f"\noverlap with theory group (daughter-first x top-norm-tercile):")
    emit(f"  P(theory | G1) = {g1['theory'].mean():.2f}   "
         f"P(G1 | theory) = {(th['grp'] == 1).mean():.2f}   "
         f"base P(theory) = {wl2['theory'].mean():.2f}")
    return d


def main():
    df = assemble()
    emit("CDDF SPINE — generic-ML heterogeneity on the annual second-birth "
         "hazard DID")
    emit(f"one-child risk set, unregulated prefectures, 2009-2022, Post=2018, "
         f"city+year FE, cluster city; {SEEDS} splits x {FOLDS} city folds")
    emit(f"  rows={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}  treated={df['T'].mean():.3f}  "
         f"hazard={df['y'].mean():.4f}")
    cov = df.drop_duplicates("pid")[XCOLS_A].notna().mean()
    emit("feature coverage (woman level): "
         + "  ".join(f"{k}={v:.0%}" for k, v in cov.items()))

    m = smf.ols(f"y ~ T:post + {CTRL}{FE}", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["city"]})
    emit(f"reference pooled DID (gender-pooled layer 2): "
         f"{m.params['T:post']:+.4f} (p={m.pvalues['T:post']:.3f})")

    sA, _ = run_version(df, XCOLS_A, "A FULL")
    sB, _ = run_version(df, XCOLS_B, "B RESIDUAL (known moderators excluded)")
    dA = consensus_block(df, sA, XCOLS_A, "A FULL")
    consensus_block(df, sB, XCOLS_B, "B RESIDUAL")

    keep = dA.drop_duplicates("pid")[["pid", "s", "grp"]].rename(
        columns={"s": "s_A", "grp": "grp_A"})
    keep["s_B"] = keep["pid"].map(sB)
    keep.to_csv(os.path.join(OUT, "cddf_scores.csv"), index=False)
    emit("\nscores saved -> output/cddf_scores.csv")

    out = os.path.join(OUT, "tables", "cddf_spine.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
