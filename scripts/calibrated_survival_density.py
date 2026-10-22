# ==========================================================================
# REPLICATION PACKAGE — calibrated_survival_density.py
# Tier: SUPPLEMENTARY
# Paper items: Table B3 (Survival-curve column)
# Reads: output/prognostic_scores.csv
# Writes: output/tables/calibrated_survival_density.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard, cddf_spine, income_gradient_did
# Run order: step 46 in run_all.sh
# ==========================================================================

"""ML project #4 — calibrated discrete-time survival model to sharpen the
threshold-density test (inverted_u_tests.py TEST 2).

The star exhibit's density dens = Pc(1-Pc) currently uses a crude conversion
Pc = 1-(1-p_hat)^5 where p_hat is a woman's AVERAGE OOF-predicted annual
hazard from a HistGBM *regressor* (prognostic_gates.py). Two measurement
upgrades, everything else identical:

  1. Proper probability model + calibration: HistGBM *classifier* wrapped in
     CalibratedClassifierCV(method='isotonic') — dens is a nonlinear function
     of Pc, so calibration error directly distorts where the vertex (Pc=0.5)
     lands.
  2. Per-woman survival CURVE instead of the flat-hazard formula: evaluate
     the calibrated hazard on her own counterfactual grid dur=1..5 with
     age = age_fb + dur (baseline covariates fixed), and compose
     Pc = 1 - prod_d (1 - h_d).

Honesty protocol unchanged from prognostic_gates.py: trained ONLY on
2009-2015 woman-years, city-fold cross-fitted (own city never trains the
score), SEEDS repeated splits, median consensus. Then re-run the structural
spec verbatim (T:post:dens with linear T:post:Pc control, city+year FE,
cluster city, WCB 4999) + daughter/son arms + cohort-K4 version, and an
apples-to-apples comparison with the OLD Pc on the same estimation sample.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                   # noqa: E402
from annual_firstborn_gender import load_rosters          # noqa: E402
import annual_parity_hazard as aph                        # noqa: E402
from income_gradient_did import wcb                       # noqa: E402

SEEDS = int(os.environ.get("CAL_SEEDS", 10))
FOLDS = 5
CTRL = cs.CTRL
FE = cs.FE
XCOLS = cs.XCOLS_A
HMAX = 0.6   # same clip as the original crude conversion

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def cfit(formula, d, keys, label, note=""):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}{note}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def calibrated_curves(df, seed, kmax=5):
    """One split: city folds -> calibrated hazard model on pre rows ->
    per-woman survival-curve Pc for horizons 1..kmax + OOF diagnostics."""
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
    Xmat = df[XCOLS + ["age", "dur"]].astype(float).values

    # per-woman counterfactual grid: dur = 1..kmax, age = age_fb + dur
    wl = df.drop_duplicates("pid")[["pid", "city"] + XCOLS].copy()
    grid = wl.merge(pd.DataFrame({"dur": range(1, kmax + 1)}), how="cross")
    grid["age"] = grid["age_fb"] + grid["dur"]
    Gmat = grid[XCOLS + ["age", "dur"]].astype(float).values
    gfold = grid["city"].map(fold).values

    haz = np.full(len(grid), np.nan)
    haz_raw = np.full(len(grid), np.nan)
    oof_raw = np.full(len(df), np.nan)
    oof_cal = np.full(len(df), np.nan)
    for f in range(FOLDS):
        tr = (fvec != f) & pre
        base = HistGradientBoostingClassifier(
            random_state=seed, max_depth=3, max_iter=200,
            min_samples_leaf=50)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(Xmat[tr], yv[tr])
        raw = HistGradientBoostingClassifier(
            random_state=seed, max_depth=3, max_iter=200,
            min_samples_leaf=50).fit(Xmat[tr], yv[tr])
        te = fvec == f
        oof_cal[te] = cal.predict_proba(Xmat[te])[:, 1]
        oof_raw[te] = raw.predict_proba(Xmat[te])[:, 1]
        gte = gfold == f
        haz[gte] = cal.predict_proba(Gmat[gte])[:, 1]
        haz_raw[gte] = raw.predict_proba(Gmat[gte])[:, 1]

    def compose(hvec):
        g = grid.copy()
        g["h"] = np.clip(hvec, 0, HMAX)
        g["log_s"] = np.log1p(-g["h"])
        gs = g.sort_values(["pid", "dur"]).groupby("pid")["log_s"].cumsum()
        g["Pc_k"] = 1 - np.exp(gs)
        return g.pivot_table(index="pid", columns="dur", values="Pc_k")

    return compose(haz), compose(haz_raw), oof_raw, oof_cal, pre


def reliability(y, p, label, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"pred": p, "obs": y}).groupby(q).mean()
    ece = (np.abs(t["pred"] - t["obs"])
           * pd.Series(p).groupby(q).size().values / len(p)).sum()
    emit(f"\n  reliability, {label} (pre-2016 OOF rows; ECE={ece:.4f}):")
    emit("    " + " ".join(f"{v:.3f}" for v in t["pred"]) + "   <- predicted")
    emit("    " + " ".join(f"{v:.3f}" for v in t["obs"]) + "   <- observed")
    return ece


def main():
    df = cs.assemble()
    df["post"] = (df["year"] >= 2018).astype(int)
    emit("CALIBRATED SURVIVAL DENSITY — threshold-density test with "
         "isotonic-calibrated per-woman survival curves")
    emit(f"one-child risk set: rows={len(df):,}  women="
         f"{df['pid'].nunique():,}; {FOLDS} city folds x {SEEDS} seeds, "
         f"median consensus; hazard clip {HMAX}")

    all5, all4, all5r, all4r, raws, cals = [], [], [], [], None, None
    for seed in range(SEEDS):
        curves, curves_r, oof_raw, oof_cal, pre = calibrated_curves(df, seed)
        all5.append(curves[5].rename(f"s{seed}"))
        all4.append(curves[4].rename(f"s{seed}"))
        all5r.append(curves_r[5].rename(f"s{seed}"))
        all4r.append(curves_r[4].rename(f"s{seed}"))
        raws = oof_raw if raws is None else raws + oof_raw
        cals = oof_cal if cals is None else cals + oof_cal
        print(f"  [seed {seed}] done", flush=True)
    raws /= SEEDS
    cals /= SEEDS

    emit("\n== 1. Calibration diagnostics ==")
    ok = pre & ~np.isnan(raws)
    reliability(df["y"].values[ok], raws[ok], "UNCALIBRATED GBM")
    reliability(df["y"].values[ok], cals[ok], "isotonic-calibrated")

    Pc5 = pd.concat(all5, axis=1).median(axis=1).rename("Pc_cal")
    Pc4 = pd.concat(all4, axis=1).median(axis=1).rename("Pc4_cal")
    Pc5r = pd.concat(all5r, axis=1).median(axis=1).rename("Pc_raw")
    Pc4r = pd.concat(all4r, axis=1).median(axis=1).rename("Pc4_raw")
    df = df.merge(Pc5, left_on="pid", right_index=True, how="left")
    df = df.merge(Pc5r, left_on="pid", right_index=True, how="left")
    df["dens_cal"] = df["Pc_cal"] * (1 - df["Pc_cal"])
    df["dens_raw"] = df["Pc_raw"] * (1 - df["Pc_raw"])

    # old score, same women
    prog = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))
    df = df.merge(prog[["pid", "prop"]], on="pid", how="left")
    df["Pc_old"] = 1 - (1 - df["prop"].clip(0, HMAX)) ** 5
    df["dens_old"] = df["Pc_old"] * (1 - df["Pc_old"])

    wl = df.drop_duplicates("pid").dropna(subset=["Pc_cal", "Pc_raw",
                                                  "Pc_old"])
    emit("\n== 2. New vs old Pc (woman level) ==")
    emit(f"  corr(Pc_cal, Pc_old)={wl[['Pc_cal','Pc_old']].corr().iloc[0,1]:.3f}"
         f"   corr(Pc_raw, Pc_old)={wl[['Pc_raw','Pc_old']].corr().iloc[0,1]:.3f}"
         f"   corr(dens_cal, dens_old)="
         f"{wl[['dens_cal','dens_old']].corr().iloc[0,1]:.3f}"
         f"   corr(dens_raw, dens_old)="
         f"{wl[['dens_raw','dens_old']].corr().iloc[0,1]:.3f}")
    for c in ("Pc_cal", "Pc_raw", "Pc_old"):
        emit(f"  {c:<7}: mean={wl[c].mean():.3f}  "
             f"share>0.5={(wl[c] > 0.5).mean():.2f}  "
             f"p10/p50/p90={wl[c].quantile([.1,.5,.9]).round(3).tolist()}")

    # estimation sample: women with ALL scores (apples-to-apples)
    est = df.dropna(subset=["Pc_cal", "Pc_raw", "Pc_old"]).copy()

    emit("\n== 3. Structural test, annual hazard (Post=2018) ==")
    for tag, pc, dn in [("RAW-GBM curve", "Pc_raw", "dens_raw"),
                        ("CALIBRATED", "Pc_cal", "dens_cal"),
                        ("OLD (crude)", "Pc_old", "dens_old")]:
        S = (f"y ~ T:post + T:post:{pc} + T:post:{dn} + post:{pc} + post:{dn}"
             f" + T:{pc} + T:{dn} + {pc} + {dn} + {CTRL}{FE}")
        cfit(S, est, ["T:post", f"T:post:{pc}", f"T:post:{dn}"],
             f"{tag}: main structural (prediction dens<0, Pc~0)")
        for arm, lab in [(0, "daughter"), (1, "son")]:
            cfit(S, est[est["son"] == arm],
                 [f"T:post:{pc}", f"T:post:{dn}"], f"{tag}: {lab} arm")

    emit("\n== 4. Wild cluster bootstrap on T:post:dens (4999 reps) ==")
    for tag, pc, dn, d_ in [
            ("RAW full", "Pc_raw", "dens_raw", est),
            ("RAW daughter", "Pc_raw", "dens_raw", est[est["son"] == 0]),
            ("RAW son", "Pc_raw", "dens_raw", est[est["son"] == 1]),
            ("CALIBRATED full", "Pc_cal", "dens_cal", est),
            ("CALIBRATED daughter", "Pc_cal", "dens_cal",
             est[est["son"] == 0]),
            ("CALIBRATED son", "Pc_cal", "dens_cal", est[est["son"] == 1]),
            ("OLD full", "Pc_old", "dens_old", est)]:
        S = (f"y ~ T:post + T:post:{pc} + T:post:{dn} + post:{pc} + post:{dn}"
             f" + T:{pc} + T:{dn} + {pc} + {dn} + {CTRL}{FE}")
        b, t, p, G = wcb(d_, S, f"T:post:{dn}")
        emit(f"  {tag:<22} T:post:dens = {b:+.4f}  t={t:.2f}  "
             f"WCB p={p:.4f}  (G={G})")

    emit("\n== 5. Cohort-K4 structural version (depletion-proof frame) ==")
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
    dch = dch.merge(Pc4.reset_index(), on="pid", how="inner")
    dch = dch.merge(Pc4r.reset_index(), on="pid", how="inner")
    dch = dch.merge(prog[["pid", "prop"]], on="pid", how="inner")
    dch["T"] = dch["treat_hpr"].astype(int)
    dch["city"] = dch["city_code"].astype(int)
    s = dch[((dch["by1"].between(2009, 2013))
             | (dch["by1"].between(2016, 2018)))
            & (dch["last_wave"] >= dch["by1"] + 4)].copy()
    s["Y"] = ((s["by2"] - s["by1"]) <= 4).fillna(False).astype(float)
    s["postc"] = (s["by1"] >= 2016).astype(int)
    s["dens4"] = s["Pc4_cal"] * (1 - s["Pc4_cal"])
    s["dens4_raw"] = s["Pc4_raw"] * (1 - s["Pc4_raw"])
    s["Pc4_old"] = 1 - (1 - s["prop"].clip(0, HMAX)) ** 4
    s["dens4_old"] = s["Pc4_old"] * (1 - s["Pc4_old"])
    for tag, pc, dn in [("RAW-GBM curve", "Pc4_raw", "dens4_raw"),
                        ("CALIBRATED", "Pc4_cal", "dens4"),
                        ("OLD (crude)", "Pc4_old", "dens4_old")]:
        cfit(f"Y ~ T:postc + T:postc:{pc} + T:postc:{dn} + postc:{pc}"
             f" + postc:{dn} + T:{pc} + T:{dn} + {pc} + {dn}"
             f" + agefb + I(agefb**2) + C(by1) + C(city)", s,
             ["T:postc", f"T:postc:{pc}", f"T:postc:{dn}"],
             f"{tag}: cohort structural K=4")

    out = os.path.join(OUT, "tables", "calibrated_survival_density.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
