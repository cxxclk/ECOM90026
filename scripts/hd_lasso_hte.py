# ==========================================================================
# REPLICATION PACKAGE — hd_lasso_hte.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (LASSO)
# Reads: CFPS .dta files in data/20xx/, output/panel_individual.csv
# Writes: output/tables/hd_lasso_hte.txt
# Imports local modules: cddf_spine
# Run order: step 39 in run_all.sh
# ==========================================================================

"""High-dimensional LASSO heterogeneity selection (method 2, the last untried
one) + catalog-wide upgrade of the CDDF-B completeness certificate.

Two gaps this closes (2026-07-21 session 5):
  1. The T x post x X_k interaction-selection approach (Belloni-Chernozhukov-
     Hansen flavor) was never run.
  2. Every previous method used a NARROW feature menu (<= 16 vars). Here the
     menu is built from the full 2014 adult (1,191 cols) + famecon (459 cols)
     baseline files — subjective housing pressure qn6017, relative income /
     status qn8011/8012, financial assets, non-housing debts, consumption
     structure, time use, health, ... — so the completeness certificate
     becomes "wide-menu" rather than "our-16-vars".

Pipeline:
  A. Feature build: all numeric 2014 baseline vars, CFPS negative codes ->
     NaN, blacklist ids/weights/interview meta, coverage >= 30% among
     risk-set women interviewed in 2014, arcsinh for heavy-tailed money
     vars, cap 150 (must-include marquee list first, then by coverage).
  B. LASSO selection: residualize y and d = T x post on age/dur + city +
     year FE; partial the average DID out of y; LassoCV (city-GroupKFold)
     of the residual on the dictionary {d~ * Xk (targets), Xk, post*Xk,
     T*Xk (controls)}, everything standardized. Report selected d~*Xk.
  C. Post-selection OLS (full sample, city+year FE, cluster city) on the
     selected interactions — labeled OPTIMISTIC (selection and inference on
     the same data).
  D. HONEST city-split check: 10 random 50/50 city splits (stratified by T);
     select on half A, estimate on half B; report per-variable selection
     frequency and median held-out coef/p. This is the result that counts.
  E. CDDF-B-WIDE: rerun the cddf_spine version-B GATES/BLP with the wide
     menu MINUS known-moderator families (ideal-kids, norm battery, incomes,
     house value) — if BLP stays null, the certificate upgrades.
"""
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                   # noqa: E402

MAX_FEAT = 150
N_SPLITS = 10
CTRL, FE = cs.CTRL, cs.FE

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)
cs.LINES = LINES   # share the log so run_version output lands in our table

BLACKLIST = re.compile(
    r"(^pid|^fid|^cid|^countyid|^provcd|^psu|^subsample|weight|^rswt|^sw_|"
    r"code$|^cyear|^cmonth|^cday|^interv|^releas|^vers|^obs|^proxy|^resp)",
    re.I)
MUST = ["qn6017", "qn8011", "qn8012", "finance_asset", "nonhousing_debts",
        "house_debts", "total_asset", "durables_asset", "land_asset",
        "fixed_asset", "pce", "food", "dress", "daily", "expense",
        "houseasset_gross", "savings", "ftransfer_1", "ft601", "ft4", "ft6",
        "ft7", "qq9010", "qn406", "qp201", "urban14"]
KNOWN = re.compile(r"(^qm50|^qm10|^qm11|^fincome|income|^fq4|^fq6|"
                   r"^houseasset)", re.I)


def load_2014_features(pids, fid_map):
    """Wide numeric 2014 baseline table, one row per pid."""
    frames, labels = [], {}
    ad, meta_a = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"))
    labels.update(dict(zip(meta_a.column_names, meta_a.column_labels)))
    ad = ad[ad["pid"].isin(pids)]
    frames.append(("adult", ad.set_index("pid")))

    fe_, meta_f = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014famecon_201906.dta"))
    labels.update(dict(zip(meta_f.column_names, meta_f.column_labels)))
    fe_ = fe_.rename(columns={"fid14": "_fid"})
    fe_ = fe_[fe_["_fid"].isin(set(fid_map.values()))].set_index("_fid")
    fam = pd.DataFrame(index=pd.Index(list(fid_map.keys()), name="pid"))
    fam = fam.join(pd.Series(fid_map, name="_fid"))
    fam = fam.join(fe_, on="_fid").drop(columns="_fid")
    frames.append(("famecon", fam))

    feats = {}
    for tag, df in frames:
        num = df.select_dtypes(include=[np.number])
        for c in num.columns:
            if BLACKLIST.search(c) or c in feats:
                continue
            v = num[c].copy()
            v[v < 0] = np.nan            # CFPS negative missing codes
            if v.notna().mean() < 0.01 or v.nunique() < 2:
                continue
            if v.quantile(0.99) > 1000:   # heavy-tailed money vars
                v = np.arcsinh(v)
            feats[c] = v
    X = pd.DataFrame(feats)
    return X, labels


def main():
    df = cs.assemble()
    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "fid", "year"])
    fid14 = (pan[pan["year"] == 2014].dropna().drop_duplicates("pid")
             .set_index("pid")["fid"].to_dict())
    women = df["pid"].unique()
    X, labels = load_2014_features(set(women),
                                   {p: f for p, f in fid14.items()
                                    if p in set(women)})
    emit("HIGH-DIMENSIONAL LASSO HTE SELECTION + WIDE-MENU CERTIFICATE")
    emit(f"raw 2014 feature pool after cleaning: {X.shape[1]} vars")

    # restrict to women actually observed in 2014 (>=30% features present)
    cover = X.notna().mean(axis=1)
    ok_pids = set(cover[cover >= 0.30].index)
    d = df[df["pid"].isin(ok_pids)].copy()
    emit(f"risk-set women with 2014 baseline: {len(ok_pids):,} of "
         f"{df['pid'].nunique():,}  ->  woman-years {len(d):,}")

    Xw = X.loc[X.index.isin(ok_pids)]
    covw = Xw.notna().mean()
    keep = [c for c in MUST if c in Xw.columns and covw[c] >= 0.25]
    rest = [c for c in covw[covw >= 0.30].sort_values(ascending=False).index
            if c not in keep]
    cols = (keep + rest)[:MAX_FEAT]
    Xw = Xw[cols]
    emit(f"feature menu: {len(cols)} vars (must-include hits: "
         f"{[c for c in MUST if c in cols]})")

    # merge raw features (GBM certificate, "_r") and imputed-z (LASSO, "_z");
    # suffixes prevent collisions with panel columns (eduy, urban, ...)
    d = d.merge(Xw.add_suffix("_r"), left_on="pid", right_index=True,
                how="left")
    Z = (Xw - Xw.median()) / Xw.std()
    Z = Z.fillna(0).add_suffix("_z")
    d = d.merge(Z, left_on="pid", right_index=True, how="left")

    # ---- B. residualize & build dictionary ------------------------------
    m0 = smf.ols(f"y ~ {CTRL}{FE}", data=d).fit()
    ytil = (d["y"] - m0.fittedvalues).values
    d["_d"] = d["T"] * d["post"]
    md = smf.ols(f"_d ~ {CTRL}{FE}", data=d).fit()
    dtil = (d["_d"] - md.fittedvalues).values
    b_avg = dtil @ ytil / (dtil @ dtil)
    y2 = ytil - b_avg * dtil
    emit(f"partialled average DID (check vs -0.014): {b_avg:+.4f}")

    zcols = [c + "_z" for c in cols]
    Zm = d[zcols].values.astype(np.float32)
    Tv = d["T"].values[:, None].astype(np.float32)
    Pv = d["post"].values[:, None].astype(np.float32)
    dict_blocks = [Zm * dtil[:, None].astype(np.float32),   # targets
                   Zm, Zm * Pv, Zm * Tv]                    # controls
    D = np.hstack(dict_blocks)
    D = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1)
    names = ([f"TPx_{c}" for c in cols] + [f"m_{c}" for c in cols]
             + [f"px_{c}" for c in cols] + [f"tx_{c}" for c in cols])
    emit(f"dictionary: {D.shape[1]} columns x {D.shape[0]:,} rows")

    gkf = GroupKFold(n_splits=5)
    las = LassoCV(cv=gkf.split(D, y2, groups=d["city"].values),
                  n_alphas=60, max_iter=5000, n_jobs=-1)
    las.fit(D, y2)
    nz = np.flatnonzero(las.coef_)
    sel_all = [names[i] for i in nz]
    sel_tp = [n for n in sel_all if n.startswith("TPx_")]
    emit(f"\nLASSO (city-GroupKFold CV, alpha={las.alpha_:.5f}): "
         f"{len(nz)} of {D.shape[1]} columns selected, "
         f"{len(sel_tp)} are T x post x X targets:")
    coefmap = dict(zip(names, las.coef_))
    for n in sorted(sel_tp, key=lambda n: -abs(coefmap[n])):
        v = n[4:]
        emit(f"  {n:<26} lasso-coef {coefmap[n]:+.5f}   "
             f"[{str(labels.get(v, ''))[:55]}]")

    # ---- C. post-selection OLS (optimistic) -----------------------------
    if sel_tp:
        for n in sel_tp:
            v = n[4:] + "_z"
            d[f"TP_{n[4:]}"] = d["T"] * d["post"] * d[v]
            d[f"P_{n[4:]}"] = d["post"] * d[v]
            d[f"Tx_{n[4:]}"] = d["T"] * d[v]
        terms = " + ".join(f"TP_{n[4:]} + P_{n[4:]} + Tx_{n[4:]} + {n[4:]}_z"
                           for n in sel_tp)
        m = smf.ols(f"y ~ T:post + {terms} + {CTRL}{FE}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        emit("\nPost-selection OLS (full sample — OPTIMISTIC p-values):")
        emit(f"  T:post           {m.params['T:post']:+.4f} "
             f"(p={m.pvalues['T:post']:.3f})")
        for n in sel_tp:
            k = f"TP_{n[4:]}"
            emit(f"  {k:<24} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    else:
        emit("\nLASSO selected NO T x post x X interactions at CV-optimal "
             "alpha — nothing to post-estimate.")

    # ---- D. honest city-split selection stability -----------------------
    emit(f"\nHONEST check: {N_SPLITS} random 50/50 city splits "
         f"(select on A, estimate on B):")
    cities = d[["city", "T"]].drop_duplicates()
    freq, held = {}, {}
    rng = np.random.RandomState(90026)
    for it in range(N_SPLITS):
        grpA = set()
        for _, g in cities.groupby("T"):
            cc = g["city"].values.copy()
            rng.shuffle(cc)
            grpA.update(cc[: len(cc) // 2])
        maskA = d["city"].isin(grpA).values
        lasA = LassoCV(cv=5, n_alphas=40, max_iter=3000, n_jobs=-1)
        lasA.fit(D[maskA], y2[maskA])
        selA = [names[i] for i in np.flatnonzero(lasA.coef_)
                if names[i].startswith("TPx_")]
        for n in selA:
            freq[n] = freq.get(n, 0) + 1
            v = n[4:]
            sB = d[~maskA].copy()
            sB["TPv"] = sB["T"] * sB["post"] * sB[v + "_z"]
            sB["Pv"] = sB["post"] * sB[v + "_z"]
            sB["Tv"] = sB["T"] * sB[v + "_z"]
            try:
                mB = smf.ols(f"y ~ T:post + TPv + Pv + Tv + {v}_z + "
                             f"{CTRL}{FE}", data=sB).fit(
                    cov_type="cluster", cov_kwds={"groups": sB["city"]})
                held.setdefault(n, []).append(
                    (mB.params["TPv"], mB.pvalues["TPv"]))
            except Exception:
                pass
        print(f"  [split {it}] selected {len(selA)} targets", flush=True)
    if freq:
        emit(f"  {'target':<26}{'sel.freq':>9}{'held-out coef (med)':>21}"
             f"{'held-out p (med)':>18}")
        for n, f_ in sorted(freq.items(), key=lambda kv: -kv[1]):
            hs = held.get(n, [])
            cmed = np.median([h[0] for h in hs]) if hs else np.nan
            pmed = np.median([h[1] for h in hs]) if hs else np.nan
            emit(f"  {n:<26}{f_}/{N_SPLITS:<7}{cmed:>+18.4f}{pmed:>18.3f}")
        emit("  (a real moderator should be selected in most splits AND "
             "keep a small held-out p)")
    else:
        emit("  no target selected in any split — clean high-dimensional "
             "null.")

    # ---- E. CDDF-B-WIDE certificate -------------------------------------
    emit("\n" + "=" * 74)
    emit("CDDF-B-WIDE: generic-ML certificate on the wide menu MINUS known "
         "moderator families")
    wide_b = [c + "_r" for c in cols if not KNOWN.search(c)]
    emit(f"  {len(wide_b)} features (excluded {len(cols)-len(wide_b)} "
         f"known-family vars)")
    cs.SEEDS = 10
    cs.run_version(d, wide_b, "B-WIDE")

    out = os.path.join(OUT, "tables", "hd_lasso_hte.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
