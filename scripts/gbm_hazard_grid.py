# ==========================================================================
# REPLICATION PACKAGE — gbm_hazard_grid.py
# Tier: CORE
# Paper items: Table 7 (Boosted trees rows (default, best of 28)); Table B3 (Tuned-model column); Table E1 (28 model settings)
# Reads: output/prognostic_scores.csv
# Writes: output/tables/calibrated_survival_density.txt, output/tables/gbm_hazard_grid.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard, cddf_spine, income_gradient_did
# Run order: step 28 in run_all.sh
# ==========================================================================

"""HistGBM hyperparameter grid search for the hazard model behind the
threshold-density test (calibrated_survival_density.py RAW-GBM curve).

Protocol (forking-paths safe): hyperparameters are selected ONLY on the
predictive criterion — OOF log-loss on pre-2016 woman-years under the same
city-GroupKFold scheme (own city never trains the score), averaged over
3 seeds. AUC and ECE reported alongside. The downstream structural test is
run ONCE, with the selected config, after selection is frozen.

Stage 1: 27-combo grid (lr x depth x min_samples_leaf) + current default.
Stage 2: winner -> 10-seed median-consensus survival-curve Pc (raw, no
isotonic — established anti-pattern) -> annual structural test (full /
daughter / son, cluster + WCB) + cohort-K4 structural test, side by side
with the previous default RAW config and the OLD crude score.
"""
import os
import sys
import warnings
from itertools import product
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                   # noqa: E402
from annual_firstborn_gender import load_rosters          # noqa: E402
import annual_parity_hazard as aph                        # noqa: E402
from income_gradient_did import wcb                       # noqa: E402

GRID_SEEDS = 3
CURVE_SEEDS = int(os.environ.get("CAL_SEEDS", 10))
FOLDS = 5
CTRL = cs.CTRL
FE = cs.FE
XCOLS = cs.XCOLS_A
HMAX = 0.6

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def city_folds(df, seed):
    rng = np.random.RandomState(seed)
    cities = df[["city", "T"]].drop_duplicates()
    fold = {}
    for _, grp in cities.groupby("T"):
        idx = grp["city"].values.copy()
        rng.shuffle(idx)
        for i, c in enumerate(idx):
            fold[c] = i % FOLDS
    return df["city"].map(fold).values, fold


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"pred": p, "obs": y}).groupby(q).mean()
    w = pd.Series(p).groupby(q).size().values / len(p)
    return (np.abs(t["pred"] - t["obs"]) * w).sum()


def main():
    df = cs.assemble()
    df["post"] = (df["year"] >= 2018).astype(int)
    Xmat = df[XCOLS + ["age", "dur"]].astype(float).values
    yv = df["y"].values
    pre = (df["year"] <= 2015).values

    emit("HISTGBM HYPERPARAMETER GRID — hazard model for the threshold-"
         "density test")
    emit(f"one-child risk set rows={len(df):,}; pre-2016 training rows="
         f"{int(pre.sum()):,}; {FOLDS} city folds x {GRID_SEEDS} seeds; "
         f"selection metric = OOF log-loss (pre rows only)")

    # ---- Stage 1: grid on predictive criterion --------------------------
    grid = [dict(learning_rate=lr, max_depth=d, min_samples_leaf=m,
                 max_iter=400, early_stopping=True, n_iter_no_change=20,
                 validation_fraction=0.15)
            for lr, d, m in product([0.03, 0.05, 0.1],
                                    [2, 3, None],
                                    [20, 50, 100])]
    default = dict(learning_rate=0.1, max_depth=3, min_samples_leaf=50,
                   max_iter=200, early_stopping=False)
    configs = [("DEFAULT(current)", default)] + [
        (f"lr={g['learning_rate']} d={g['max_depth']} "
         f"msl={g['min_samples_leaf']}", g) for g in grid]

    rows = []
    for name, cfg in configs:
        lls, aucs, eces = [], [], []
        for seed in range(GRID_SEEDS):
            fvec, _ = city_folds(df, seed)
            oof = np.full(len(df), np.nan)
            for f in range(FOLDS):
                tr = (fvec != f) & pre
                m = HistGradientBoostingClassifier(random_state=seed, **cfg)
                m.fit(Xmat[tr], yv[tr])
                te = (fvec == f) & pre
                oof[te] = m.predict_proba(Xmat[te])[:, 1]
            ok = pre & ~np.isnan(oof)
            lls.append(log_loss(yv[ok], oof[ok]))
            aucs.append(roc_auc_score(yv[ok], oof[ok]))
            eces.append(ece(yv[ok], oof[ok]))
        rows.append(dict(name=name, ll=np.mean(lls), ll_sd=np.std(lls),
                         auc=np.mean(aucs), ece=np.mean(eces), cfg=cfg))
        print(f"  [{name:<24}] ll={np.mean(lls):.5f} auc={np.mean(aucs):.4f}"
              f" ece={np.mean(eces):.4f}", flush=True)

    res = pd.DataFrame(rows).sort_values("ll").reset_index(drop=True)
    emit("\n== 1. Grid ranking (by OOF log-loss, pre-2016 rows) ==")
    emit(f"  {'config':<26}{'logloss':>9}{'(sd)':>9}{'AUC':>8}{'ECE':>8}")
    for _, r in res.iterrows():
        emit(f"  {r['name']:<26}{r['ll']:>9.5f}{r['ll_sd']:>9.5f}"
             f"{r['auc']:>8.4f}{r['ece']:>8.4f}")
    base_ll = res.loc[res["name"] == "DEFAULT(current)", "ll"].iloc[0]
    best = res.iloc[0]
    emit(f"\n  WINNER: {best['name']}  (logloss {best['ll']:.5f} vs default "
         f"{base_ll:.5f}, gain {base_ll - best['ll']:+.5f}; "
         f"default rank {int(res.index[res['name'] == 'DEFAULT(current)'][0]) + 1}"
         f"/{len(res)})")

    # ---- Stage 2: winner -> survival-curve Pc -> structural tests -------
    def curves_for(cfg, kmax=5):
        wl = df.drop_duplicates("pid")[["pid", "city"] + XCOLS].copy()
        grid_ = wl.merge(pd.DataFrame({"dur": range(1, kmax + 1)}),
                         how="cross")
        grid_["age"] = grid_["age_fb"] + grid_["dur"]
        Gmat = grid_[XCOLS + ["age", "dur"]].astype(float).values
        all5, all4 = [], []
        for seed in range(CURVE_SEEDS):
            fvec, fold = city_folds(df, seed)
            gfold = grid_["city"].map(fold).values
            haz = np.full(len(grid_), np.nan)
            for f in range(FOLDS):
                tr = (fvec != f) & pre
                m = HistGradientBoostingClassifier(random_state=seed, **cfg)
                m.fit(Xmat[tr], yv[tr])
                gte = gfold == f
                haz[gte] = m.predict_proba(Gmat[gte])[:, 1]
            g = grid_.copy()
            g["h"] = np.clip(haz, 0, HMAX)
            g["log_s"] = np.log1p(-g["h"])
            gs = g.sort_values(["pid", "dur"]).groupby("pid")["log_s"].cumsum()
            g["Pc_k"] = 1 - np.exp(gs)
            piv = g.pivot_table(index="pid", columns="dur", values="Pc_k")
            all5.append(piv[5].rename(f"s{seed}"))
            all4.append(piv[4].rename(f"s{seed}"))
            print(f"    [curve seed {seed}] done", flush=True)
        return (pd.concat(all5, axis=1).median(axis=1),
                pd.concat(all4, axis=1).median(axis=1))

    emit("\n== 2. Winner config -> per-woman survival-curve Pc "
         f"({CURVE_SEEDS} seeds, median) ==")
    Pc5, Pc4 = curves_for(best["cfg"])
    d2 = df.merge(Pc5.rename("Pc_gs"), left_on="pid", right_index=True,
                  how="left")
    d2["dens_gs"] = d2["Pc_gs"] * (1 - d2["Pc_gs"])
    prog = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))
    d2 = d2.merge(prog[["pid", "prop"]], on="pid", how="left")
    d2["Pc_old"] = 1 - (1 - d2["prop"].clip(0, HMAX)) ** 5
    d2["dens_old"] = d2["Pc_old"] * (1 - d2["Pc_old"])
    est = d2.dropna(subset=["Pc_gs", "Pc_old"]).copy()

    wl = est.drop_duplicates("pid")
    emit(f"  corr(Pc_gs, Pc_old)={wl[['Pc_gs','Pc_old']].corr().iloc[0,1]:.3f}"
         f"  mean={wl['Pc_gs'].mean():.3f}"
         f"  share>0.5={(wl['Pc_gs'] > 0.5).mean():.2f}"
         f"  p10/p50/p90={wl['Pc_gs'].quantile([.1,.5,.9]).round(3).tolist()}")

    def cfit(formula, d, keys, label):
        m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                         cov_kwds={"groups": d["city"]})
        emit(f"\n[{label}]  N={int(m.nobs):,}")
        for k in keys:
            if k in m.params.index:
                emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                     f"p={m.pvalues[k]:.3f})")
        return m

    emit("\n== 3. Structural test, annual hazard (Post=2018), TUNED score ==")
    S = ("y ~ T:post + T:post:Pc_gs + T:post:dens_gs + post:Pc_gs"
         " + post:dens_gs + T:Pc_gs + T:dens_gs + Pc_gs + dens_gs"
         f" + {CTRL}{FE}")
    cfit(S, est, ["T:post", "T:post:Pc_gs", "T:post:dens_gs"],
         "TUNED: main structural (prediction dens<0, Pc~0)")
    for arm, lab in [(0, "daughter"), (1, "son")]:
        cfit(S, est[est["son"] == arm],
             ["T:post:Pc_gs", "T:post:dens_gs"], f"TUNED: {lab} arm")

    emit("\n== 4. WCB on T:post:dens (4999 reps), TUNED ==")
    for tag, d_ in [("TUNED full", est), ("TUNED daughter",
                                          est[est["son"] == 0]),
                    ("TUNED son", est[est["son"] == 1])]:
        b, t, p, G = wcb(d_, S, "T:post:dens_gs")
        emit(f"  {tag:<16} T:post:dens = {b:+.4f}  t={t:.2f}  "
             f"WCB p={p:.4f}  (G={G})")

    emit("\n== 5. Cohort-K4 structural version, TUNED ==")
    roster, last_wave = load_rosters()
    r = roster.sort_values(["pid", "cby"]).copy()
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "by1", "cgender": "g1"})
    second = r[r["k"] == 1][["pid", "cby"]].rename(columns={"cby": "by2"})
    attrs = aph.woman_attrs()
    dch = (first.merge(second, on="pid", how="left")
                .merge(attrs, on="pid", how="inner")
                .merge(last_wave, on="pid"))
    dch = dch[(dch["by2"].isna()) | (dch["by2"] > dch["by1"])]
    dch["agefb"] = dch["by1"] - dch["birthy"]
    dch = dch[dch["agefb"].between(18, 40)]
    dch = dch.merge(Pc4.rename("Pc4_gs").reset_index(), on="pid",
                    how="inner")
    dch["T"] = dch["treat_hpr"].astype(int)
    dch["city"] = dch["city_code"].astype(int)
    s = dch[((dch["by1"].between(2009, 2013))
             | (dch["by1"].between(2016, 2018)))
            & (dch["last_wave"] >= dch["by1"] + 4)].copy()
    s["Y"] = ((s["by2"] - s["by1"]) <= 4).fillna(False).astype(float)
    s["postc"] = (s["by1"] >= 2016).astype(int)
    s["dens4_gs"] = s["Pc4_gs"] * (1 - s["Pc4_gs"])
    cfit("Y ~ T:postc + T:postc:Pc4_gs + T:postc:dens4_gs + postc:Pc4_gs"
         " + postc:dens4_gs + T:Pc4_gs + T:dens4_gs + Pc4_gs + dens4_gs"
         " + agefb + I(agefb**2) + C(by1) + C(city)", s,
         ["T:postc", "T:postc:Pc4_gs", "T:postc:dens4_gs"],
         "TUNED: cohort structural K=4")

    emit("\n== Reference (from calibrated_survival_density.txt) ==")
    emit("  RAW default annual full  T:post:dens −0.171 WCB p=0.222")
    emit("  RAW default daughter     −0.405 WCB p=0.072 | son +0.10 p=0.57")
    emit("  RAW default cohort K4    T:postc:dens4 −2.41 (p=0.029)")
    emit("  OLD crude  annual full   −0.261 WCB p=0.0524; cohort −1.88 "
         "(p=0.035)")

    out = os.path.join(OUT, "tables", "gbm_hazard_grid.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
