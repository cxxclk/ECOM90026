# ==========================================================================
# REPLICATION PACKAGE — prognostic_no_motive.py
# Tier: SUPPLEMENTARY
# Paper items: Table E6 (Models without motive variables)
# Reads: output/prognostic_scores.csv
# Writes: output/tables/prognostic_no_motive.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard, cddf_spine, income_gradient_did, prognostic_gates
# Run order: step 51 in run_all.sh
# ==========================================================================

"""Design: does the prognostic-score inverted U / threshold-density test survive
when the prediction model never sees firstborn sex or any son-preference /
traditional-norm input?  Same risk set, same folds, same hyperparameters as
prognostic_gates.py / inverted_u_tests.py / calibrated_survival_density.py;
only the feature list changes.

Variants (all + age, dur):
  FULL                 = cddf_spine.XCOLS_A (16 baseline vars; 18 inputs) — reproduction
  NO-MOTIVE            = drop son, son_pref, city_son_norm, trad_pc1
  NO-MOTIVE-NO-IDEAL   = NO-MOTIVE also minus ideal_kids (own fertility ideal; supplementary)
  NO-MOTIVE-NO-CITY    = NO-MOTIVE also minus city_urban_share, city_linc
                         (only individual/household traits + age/dur)

Per variant: (1) OOF fit on pre rows (R2, AUC, log-loss) — regressor score as in
prognostic_gates.py, 5 city folds x 10 seeds, median consensus; (2) corr with FULL
score; (3) annual quintile GATES pooled / daughter / son, Post=2018, city+year FE,
age+dur quadratics, cluster city; (4) annual density test with crude
Pc=1-(1-p)^5 (pooled/daughter/son, cluster p + WCB 4999); (5) cohort K=4 DID by
quintile + cohort density test with crude Pc4 and with the raw-GBM-classifier
survival-curve Pc4 (calibrated_survival_density.py RAW); (6) composition of the
new quintiles (daughter-first share, mean city son-norm, trad_pc1, urban, eduy,
ideal_kids).
"""
import os
import sys
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                   # noqa: E402
import prognostic_gates as pg                             # noqa: E402
from annual_firstborn_gender import load_rosters          # noqa: E402
import annual_parity_hazard as aph                        # noqa: E402
from income_gradient_did import wcb                       # noqa: E402

SEEDS = 10
FOLDS = 5
HMAX = 0.6
CTRL = cs.CTRL
FE = cs.FE
FULL = list(cs.XCOLS_A)
MOTIVE = ["son", "son_pref", "city_son_norm", "trad_pc1"]
CITY = ["city_urban_share", "city_linc"]
VARIANTS = {
    "FULL": FULL,
    "NO-MOTIVE": [x for x in FULL if x not in MOTIVE],
    "NO-MOTIVE-NO-IDEAL": [x for x in FULL if x not in MOTIVE + ["ideal_kids"]],
    "NO-MOTIVE-NO-CITY": [x for x in FULL if x not in MOTIVE + CITY],
}
REPS = int(os.environ.get("WCB_REPS", 4999))


def cohort_frame():
    roster, last_wave = load_rosters()
    r = roster.sort_values(["pid", "cby"]).copy()
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "by1", "cgender": "g1"})
    second = r[r["k"] == 1][["pid", "cby"]].rename(columns={"cby": "by2"})
    attrs = aph.woman_attrs()
    d = (first.merge(second, on="pid", how="left")
              .merge(attrs, on="pid", how="inner")
              .merge(last_wave, on="pid"))
    d = d[(d["by2"].isna()) | (d["by2"] > d["by1"])]
    d["agefb"] = d["by1"] - d["birthy"]
    d = d[d["agefb"].between(18, 40)]
    d["T"] = d["treat_hpr"].astype(int)
    d["city"] = d["city_code"].astype(int)
    s = d[((d["by1"].between(2009, 2013)) | (d["by1"].between(2016, 2018)))
          & (d["last_wave"] >= d["by1"] + 4)].copy()
    s["Y"] = ((s["by2"] - s["by1"]) <= 4).fillna(False).astype(float)
    s["postc"] = (s["by1"] >= 2016).astype(int)
    return s


def fold_map(df, seed):
    rng = np.random.RandomState(seed)
    cities = df[["city", "T"]].drop_duplicates()
    fold = {}
    for _, grp in cities.groupby("T"):
        idx = grp["city"].values.copy()
        rng.shuffle(idx)
        for i, c in enumerate(idx):
            fold[c] = i % FOLDS
    return fold


def raw_curves(df, xcols, seed, kmax=4):
    """RAW HistGBM classifier (calibrated_survival_density.py defaults) ->
    per-woman survival-curve Pc_k; also OOF predicted probability per row."""
    fold = fold_map(df, seed)
    fvec = df["city"].map(fold).values
    pre = (df["year"] <= 2015).values
    yv = df["y"].values
    Xmat = df[xcols + ["age", "dur"]].astype(float).values
    wl = df.drop_duplicates("pid")[["pid", "city"] + xcols].copy()
    grid = wl.merge(pd.DataFrame({"dur": range(1, kmax + 1)}), how="cross")
    grid["age"] = grid["age_fb"] + grid["dur"]
    Gmat = grid[xcols + ["age", "dur"]].astype(float).values
    gfold = grid["city"].map(fold).values
    haz = np.full(len(grid), np.nan)
    oof = np.full(len(df), np.nan)
    for f in range(FOLDS):
        tr = (fvec != f) & pre
        m = HistGradientBoostingClassifier(random_state=seed, max_depth=3,
                                           max_iter=200, min_samples_leaf=50)
        m.fit(Xmat[tr], yv[tr])
        oof[fvec == f] = m.predict_proba(Xmat[fvec == f])[:, 1]
        haz[gfold == f] = m.predict_proba(Gmat[gfold == f])[:, 1]
    g = grid[["pid", "dur"]].copy()
    g["h"] = np.clip(haz, 0, HMAX)
    g = g.sort_values(["pid", "dur"])
    g["Pc"] = 1 - np.exp(np.log1p(-g["h"]).groupby(g["pid"]).cumsum())
    pc4 = g[g["dur"] == kmax].set_index("pid")["Pc"]
    return pc4, oof


def fit(formula, d, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    return {k: (m.params[k], m.pvalues[k]) for k in keys if k in m.params}


def run_variant(name):
    xcols = VARIANTS[name]
    L = []
    emit = L.append
    df = cs.assemble()
    df["post"] = (df["year"] >= 2018).astype(int)
    Xmat = df[xcols + ["age", "dur"]].astype(float).values
    pre = (df["year"] <= 2015).values
    yv = df["y"].values

    emit("=" * 78)
    emit(f"VARIANT {name}: {len(xcols)} baseline vars + age, dur")
    emit(f"  inputs: {', '.join(xcols)}")
    dropped = [x for x in FULL if x not in xcols]
    emit(f"  dropped vs FULL: {', '.join(dropped) if dropped else '(none)'}")

    # ---- 1. regressor score, prognostic_gates protocol -------------------
    scores, r2s, oofs = [], [], []
    for seed in range(SEEDS):
        s, r2 = pg.propensity_scores(df, Xmat, seed)
        scores.append(s)
        r2s.append(r2)
    smed = pd.concat(scores, axis=1).median(axis=1)
    # row-level OOF fit diagnostics for the regressor (seed-average)
    # (propensity_scores returns woman means; recompute row OOF via classifier
    # below for AUC/log-loss)
    pc4s = []
    for seed in range(SEEDS):
        pc4, oof = raw_curves(df, xcols, seed)
        pc4s.append(pc4)
        oofs.append(oof)
    oofm = np.nanmean(np.vstack(oofs), axis=0)
    ok = pre & ~np.isnan(oofm)
    auc = roc_auc_score(yv[ok], oofm[ok])
    ll = log_loss(yv[ok], np.clip(oofm[ok], 1e-6, 1 - 1e-6))
    emit(f"\n[1] OOF fit on pre-2016 rows: regressor R2 median {np.median(r2s):.4f};"
         f" raw-GBM classifier AUC {auc:.4f}, log-loss {ll:.5f}")
    Pc4r = pd.concat(pc4s, axis=1).median(axis=1).rename("Pc4_raw")

    d = pg.attach(df, smed)
    wl = d.drop_duplicates("pid").copy()
    res = {"name": name, "prop": smed.rename("prop"), "Pc4r": Pc4r,
           "auc": auc, "ll": ll, "r2": np.median(r2s)}

    # ---- 2. GATES pooled / arms ------------------------------------------
    emit("\n[2] annual quintile GATES (gamma = T:post within quintile; "
         "quintiles on the pooled woman distribution)")
    emit(f"  {'sample':<10}" + "".join(f"{'G'+str(k):>17}" for k in range(1, 6))
         + f"{'G5-G1':>17}")
    gat = {}
    for lab, sub in [("pooled", d), ("daughter", d[d["son"] == 0]),
                     ("son", d[d["son"] == 1])]:
        gam, pv, df_, dp_ = pg.gates(sub)
        gat[lab] = (gam, pv)
        emit(f"  {lab:<10}" + "".join(f"{g:>+9.4f} ({p:.3f})" for g, p in
                                      zip(gam, pv))
             + f"{df_:>+9.4f} ({dp_:.3f})")
    base = [d.loc[(d["grp"] == k) & (d["post"] == 0), "y"].mean()
            for k in range(1, 6)]
    emit("  pooled base pre-hazard: " + "  ".join(f"G{k+1} {b:.3f}"
                                                 for k, b in enumerate(base)))
    gam = gat["pooled"][0]
    npost = [int(((d["grp"] == k) & (d["post"] == 1) & (d["T"] == 1)).sum())
             for k in range(1, 6)]
    lost = [max(-g * n, 0) for g, n in zip(gam, npost)]
    emit("  share of implied lost births: " + "  ".join(
        f"G{k+1} {l/sum(lost):.0%}" for k, l in enumerate(lost)) if sum(lost)
         else "  no implied lost births")
    res["gates"] = gat

    # ---- 3. composition of the new quintiles ------------------------------
    emit("\n[3] composition by quintile of THIS score (woman level)")
    comp = wl.groupby("grp").agg(n=("pid", "size"),
                                 daughter_first=("son", lambda s: 1 - s.mean()),
                                 city_son_norm=("city_son_norm", "mean"),
                                 trad_pc1=("trad_pc1", "mean"),
                                 son_pref=("son_pref", "mean"),
                                 urban=("urban", "mean"),
                                 eduy=("eduy", "mean"),
                                 ideal_kids=("ideal_kids", "mean"),
                                 T=("T", "mean"))
    emit("  " + comp.round(3).to_string().replace("\n", "\n  "))
    res["comp"] = comp

    # ---- 4. annual density test (crude Pc5) -------------------------------
    d["Pc"] = 1 - (1 - d["prop"].clip(0, HMAX)) ** 5
    d["dens"] = d["Pc"] * (1 - d["Pc"])
    S = (f"y ~ T:post + T:post:Pc + T:post:dens + post:Pc + post:dens"
         f" + T:Pc + T:dens + Pc + dens + {CTRL}{FE}")
    wlp = d.drop_duplicates("pid")
    emit(f"\n[4] annual threshold-density test, crude Pc=1-(1-p)^5 "
         f"(mean Pc {wlp['Pc'].mean():.3f}, share>0.5 "
         f"{(wlp['Pc'] > .5).mean():.2f}, corr(Pc,dens) "
         f"{wlp[['Pc','dens']].corr().iloc[0,1]:.2f})")
    dens_res = {}
    for lab, sub in [("pooled", d), ("daughter", d[d["son"] == 0]),
                     ("son", d[d["son"] == 1])]:
        r = fit(S, sub, ["T:post:Pc", "T:post:dens"])
        b, t, p, G = wcb(sub, S, "T:post:dens", reps=REPS)
        dens_res[lab] = (r["T:post:dens"][0], r["T:post:dens"][1], p)
        emit(f"  {lab:<9} T:post:dens {r['T:post:dens'][0]:+.4f} "
             f"(cluster p={r['T:post:dens'][1]:.3f}, WCB p={p:.4f})   "
             f"T:post:Pc {r['T:post:Pc'][0]:+.4f} (p={r['T:post:Pc'][1]:.3f})")
    r = fit(f"y ~ T:post + T:post:Pc + post:Pc + T:Pc + Pc + {CTRL}{FE}", d,
            ["T:post:Pc"])
    emit(f"  linear-only benchmark T:post:Pc {r['T:post:Pc'][0]:+.4f} "
         f"(p={r['T:post:Pc'][1]:.3f})")
    res["dens"] = dens_res

    # ---- 4b. horse-race: density vs the motive itself ----------------------
    wn = d.drop_duplicates("pid")
    d["normz"] = ((d["city_son_norm"] - wn["city_son_norm"].mean())
                  / wn["city_son_norm"].std())
    dn_ = d.dropna(subset=["normz"])
    SH = (f"y ~ T:post + T:post:Pc + T:post:dens + post:Pc + post:dens"
          f" + T:Pc + T:dens + Pc + dens + T:post:normz + post:normz"
          f" + T:normz + {CTRL}{FE}")
    emit("  [4b] same test + T:post:city_son_norm(z) horse-race "
         "(does density survive controlling the motive gradient directly?)")
    for lab, sub in [("pooled", dn_), ("daughter", dn_[dn_["son"] == 0])]:
        r = fit(SH, sub, ["T:post:dens", "T:post:normz"])
        emit(f"  {lab:<9} T:post:dens {r['T:post:dens'][0]:+.4f} "
             f"(p={r['T:post:dens'][1]:.3f})   T:post:normz "
             f"{r['T:post:normz'][0]:+.4f} (p={r['T:post:normz'][1]:.3f})"
             f"  N={len(sub):,}")

    # ---- 5. cohort K=4 -----------------------------------------------------
    s = cohort_frame()
    s = s.merge(wl[["pid", "prop", "grp"]], on="pid", how="inner")
    s = s.merge(Pc4r.reset_index(), on="pid", how="inner")
    CT = "agefb + I(agefb**2)"
    FEc = " + C(by1) + C(city)"
    emit(f"\n[5] cohort K=4 (2nd birth within 4 yrs; cohorts 2009-13 vs "
         f"2016-18): women={len(s):,}, mean Y={s['Y'].mean():.3f}")
    r = fit(f"Y ~ T:postc + {CT}{FEc}", s, ["T:postc"])
    emit(f"  overall T:postc {r['T:postc'][0]:+.4f} (p={r['T:postc'][1]:.3f})")
    coh_q = []
    for g in range(1, 6):
        ss = s[s["grp"] == g]
        if len(ss) < 150 or ss["T"].nunique() < 2:
            emit(f"  G{g}: n={len(ss)} too thin")
            coh_q.append((np.nan, np.nan))
            continue
        bse = ss.loc[ss["postc"] == 0, "Y"].mean()
        r = fit(f"Y ~ T:postc + {CT}{FEc}", ss, ["T:postc"])
        coh_q.append(r["T:postc"])
        emit(f"  G{g}: n={len(ss):5,}  base={bse:.3f}  T:postc="
             f"{r['T:postc'][0]:+.4f} (p={r['T:postc'][1]:.3f})")
    res["coh_q"] = coh_q
    s["Pc4_old"] = 1 - (1 - s["prop"].clip(0, HMAX)) ** 4
    s["dens4_old"] = s["Pc4_old"] * (1 - s["Pc4_old"])
    s["dens4_raw"] = s["Pc4_raw"] * (1 - s["Pc4_raw"])
    coh_d = {}
    for tag, pc, dn in [("crude Pc4", "Pc4_old", "dens4_old"),
                        ("raw survival-curve Pc4", "Pc4_raw", "dens4_raw")]:
        F = (f"Y ~ T:postc + T:postc:{pc} + T:postc:{dn} + postc:{pc}"
             f" + postc:{dn} + T:{pc} + T:{dn} + {pc} + {dn} + {CT}{FEc}")
        r = fit(F, s, [f"T:postc:{pc}", f"T:postc:{dn}"])
        coh_d[tag] = r[f"T:postc:{dn}"]
        emit(f"  cohort structural ({tag}): T:postc:dens "
             f"{r[f'T:postc:{dn}'][0]:+.4f} (p={r[f'T:postc:{dn}'][1]:.3f});"
             f"  T:postc:Pc {r[f'T:postc:{pc}'][0]:+.4f} "
             f"(p={r[f'T:postc:{pc}'][1]:.3f})")
        for gv, lab in [(0, "daughter"), (1, "son")]:
            ss = s[s["g1"] == gv]
            try:
                r = fit(F, ss, [f"T:postc:{dn}"])
                emit(f"     {lab:<9} T:postc:dens {r[f'T:postc:{dn}'][0]:+.4f}"
                     f" (p={r[f'T:postc:{dn}'][1]:.3f})  n={len(ss):,}")
            except Exception as e:                      # noqa: BLE001
                emit(f"     {lab}: failed {e}")
    res["coh_d"] = coh_d
    res["lines"] = L
    return res


def main():
    with Pool(len(VARIANTS)) as pool:
        results = pool.map(run_variant, list(VARIANTS))
    R = {r["name"]: r for r in results}
    full = R["FULL"]["prop"]
    head = [
        "PROGNOSTIC SCORE WITHOUT THE MOTIVE — does the inverted U / density test",
        "depend on the model seeing firstborn sex or son-preference/norm inputs?",
        "one-child annual 2nd-birth panel, unregulated prefectures, 2009-2022, "
        "Post=2018, city+year FE, age/dur quadratics, cluster city.",
        f"score: HistGBM regressor (depth 3, 200 iter, msl 50) on 2009-15 rows "
        f"only, {FOLDS} city folds (stratified by T) x {SEEDS} seeds, median; "
        f"same seeds/folds across variants. WCB reps={REPS}.",
        "",
        "SUMMARY",
        f"  {'variant':<20}{'R2':>7}{'AUC':>7}{'logloss':>9}{'corr FULL':>10}"
        f"{'GATES G4 (p)':>17}{'G5 (p)':>17}{'dens pooled WCB':>22}"
        f"{'dens dghtr WCB':>22}{'dens son WCB':>20}{'coh dens crude':>18}"
        f"{'coh dens surv':>18}",
    ]
    for n in VARIANTS:
        r = R[n]
        c = pd.concat([full, r["prop"]], axis=1).corr().iloc[0, 1]
        g, p = r["gates"]["pooled"]
        dp, dd, ds = r["dens"]["pooled"], r["dens"]["daughter"], r["dens"]["son"]
        cc, cs_ = r["coh_d"]["crude Pc4"], r["coh_d"]["raw survival-curve Pc4"]
        head.append(
            f"  {n:<20}{r['r2']:>7.4f}{r['auc']:>7.4f}{r['ll']:>9.5f}{c:>10.3f}"
            f"{g[3]:>+9.4f} ({p[3]:.3f}){g[4]:>+9.4f} ({p[4]:.3f})"
            f"{dp[0]:>+10.3f} ({dp[2]:.3f})   "
            f"{dd[0]:>+10.3f} ({dd[2]:.3f})   {ds[0]:>+9.3f} ({ds[2]:.3f})"
            f"{cc[0]:>+10.3f} ({cc[1]:.3f}){cs_[0]:>+10.3f} ({cs_[1]:.3f})")
    head.append("  (dens columns: coefficient (WCB p); cohort columns: coef "
                "(cluster p))")
    cached = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))
    cc = cached.set_index("pid")["prop"]
    j = pd.concat([cc.rename("cached"), full.rename("rerun")], axis=1).dropna()
    qa = pd.qcut(j["cached"], 5, labels=False)
    qb = pd.qcut(j["rerun"], 5, labels=False)
    head.append(f"\n  reproduction: corr(FULL rerun, cached prognostic_scores.csv)"
                f" = {j.corr().iloc[0,1]:.4f}; same quintile for "
                f"{(qa == qb).mean():.1%} of {len(j):,} women")
    head.append("\n  cohort K=4 DID by quintile, T:postc (p):")
    for n in VARIANTS:
        head.append(f"  {n:<20}" + "".join(
            f"  G{k+1} {b:+.3f} ({p:.3f})" if not np.isnan(b) else
            f"  G{k+1} thin" for k, (b, p) in enumerate(R[n]["coh_q"])))
    lines = head + [""]
    for n in VARIANTS:
        lines += R[n]["lines"] + [""]
    out = os.path.join(OUT, "tables", "prognostic_no_motive.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
