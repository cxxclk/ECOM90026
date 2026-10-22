# ==========================================================================
# REPLICATION PACKAGE — couple_wx_scan.py
# Tier: SUPPLEMENTARY
# Paper items: Table D2 (All couples, composite rows, honest half-splits)
# Reads: CFPS .dta files in data/20xx/, output/panel_individual.csv, output/prognostic_scores.csv
# Writes: output/tables/couple_wx_scan.txt
# Imports local modules: couple_intentions_annual
# Run order: step 49 in run_all.sh
# ==========================================================================

"""ML queue #3 — couple decision-weight function w(x) scan.

Target: the causal DID-in-weight T:post:diff (+0.027, p~0.055; 2014 cohort,
couple_avg_vs_diff.py). Question: WHERE does the husband->wife redistribution
of fertility decision weight concentrate? Known univariate moderations are
murky (income share: both halves; caregiving: wrong way; horse-race kills
both). This scan does it systematically with an honest out-of-city check.

Design:
  (a) replicate the baseline T:post:diff on the 2014 cohort;
  (b) univariate 4-way scan: y ~ T*post*diff*xz + T*post*avg + CTRL + FE for
      each standardized candidate moderator xz, BH-FDR across the scan;
  (c) HONEST city-split validation (the hd_lasso lesson: in-sample selection
      is hallucination): 10 random 50/50 city splits; pick the winning
      moderator by |t| on the training cities, refit on held-out cities,
      count same-sign holdout p<0.10;
  (d) LASSO-psi check: influence-style signal psi_i for the T:post:diff
      coefficient, LASSO of couple-mean psi on X (train cities), correlate
      prediction with realized psi on held-out cities.

Expectation management: base effect is only p=0.055 with ~2k couples; a 4-way
interaction has a fraction of that power. A clean negative certificate ("the
channel is not localizable in CFPS-measurable x") is the likely and
acceptable outcome.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
from couple_intentions_annual import build_annual_panel  # noqa: E402

RNG = np.random.RandomState(90026)
LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def city_norm_2014():
    """City-level son-continuation norm (qm1004 mean over ALL 2014 adults)."""
    ad, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"),
        usecols=["pid", "qm1004"])
    ad["qm1004"] = pd.to_numeric(ad["qm1004"], errors="coerce")
    ad.loc[(ad["qm1004"] < 1) | (ad["qm1004"] > 5), "qm1004"] = np.nan
    geo = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "year", "city_code"])
    geo = geo[(geo["year"] == 2014) & geo["city_code"].notna()]
    ad = ad.merge(geo[["pid", "city_code"]], on="pid", how="inner")
    return (ad.groupby("city_code")["qm1004"].mean()
            .rename("son_norm").reset_index())


def housework_2014():
    """Daily housework hours (qq9010), 2014 adults -- for both spouses."""
    ad, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"),
        usecols=["pid", "qq9010"])
    ad["qq9010"] = pd.to_numeric(ad["qq9010"], errors="coerce")
    ad.loc[(ad["qq9010"] < 0) | (ad["qq9010"] > 16), "qq9010"] = np.nan
    return ad.rename(columns={"qq9010": "hw"})


def build():
    cp, d = build_annual_panel()
    d = d[d["wave"] == 2014].copy()               # causal cohort
    d["post"] = (d["yr"] >= 2016).astype(int)
    d["avg"] = (d["ideal_w"] + d["ideal_h"]) / 2
    d["diff"] = d["ideal_w"] - d["ideal_h"]

    # ---- candidate moderators (couple-level, 2014 baseline) ----
    norm = city_norm_2014()
    d = d.merge(norm, on="city_code", how="left")
    hw = housework_2014()
    d = d.merge(hw.rename(columns={"hw": "hw_w"}), on="pid", how="left")
    d = d.merge(hw.rename(columns={"pid": "husband_pid", "hw": "hw_h"}),
                on="husband_pid", how="left")
    d["hw_share_w"] = d["hw_w"] / (d["hw_w"] + d["hw_h"]).replace(0, np.nan)

    prog = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))
    d = d.merge(prog[["pid", "prop"]], on="pid", how="left")

    inc_w = np.expm1(d["log_income"].clip(lower=0)) if False else None
    # wife income share from raw incomes (income, h_income in couples_panel)
    tot = d["income"].clip(lower=0) + d["h_income"].clip(lower=0)
    d["inc_share_w"] = np.where(tot > 0, d["income"].clip(lower=0) / tot, np.nan)
    d["hv_inc"] = d["house_value"] / d["fam_income"].replace(0, np.nan)
    d.loc[d["hv_inc"] < 0, "hv_inc"] = np.nan
    d["hv_inc"] = d["hv_inc"].clip(upper=d["hv_inc"].quantile(0.99))

    CANDS = {
        "age":          d["age"],
        "eduy_w":       d["eduy"],
        "eduy_h":       d["h_eduy"],
        "log_inc_w":    d["log_income"],
        "log_inc_h":    d["log_h_income"],
        "inc_share_w":  d["inc_share_w"],
        "log_faminc_pc": d["log_fam_inc_pc"],
        "urban":        d["urban"],
        "parity14":     d["parity"],
        "homeowner":    d["homeowner"],
        "hv_inc":       d["hv_inc"],
        "age_gap":      d["age_gap"],
        "son_norm":     d["son_norm"],
        "hw_share_w":   d["hw_share_w"],
        "propensity":   d["prop"],
    }
    for k, v in CANDS.items():
        v = pd.to_numeric(v, errors="coerce")
        d[f"xz_{k}"] = (v - v.mean()) / v.std()
    return d, list(CANDS.keys())


CTRL = ("age_t + I(age_t**2) + dur + eduy + log_income + C(parity) + "
        "C(provcd) + C(yr)")
BASE = f"y ~ T*post*avg + T*post*diff + {CTRL}"


def cfit(formula, dd, key, cl="city"):
    m = smf.ols(formula, data=dd).fit(cov_type="cluster",
                                      cov_kwds={"groups": dd[cl]})
    if key not in m.params:
        return np.nan, np.nan, np.nan, int(m.nobs)
    return (m.params[key], m.bse[key], m.pvalues[key], int(m.nobs))


def scan_one(d, name):
    x = f"xz_{name}"
    dd = d.dropna(subset=[x])
    f = f"y ~ T*post*diff*{x} + T*post*avg + {CTRL}"
    key = f"T:post:diff:{x}"
    b, se, p, n = cfit(f, dd, key)
    return b, se, p, n, dd


def main():
    d, cands = build()
    emit("Couple decision-weight w(x) scan -- 2014 cohort DID-in-weight")
    emit(f"  couple-years={len(d):,}  couples={d['cid'].nunique():,}  "
         f"cities={d['city'].nunique()}")

    # (a) baseline
    b, se, p, n = cfit(BASE, d, "T:post:diff")
    emit(f"\n(a) baseline T:post:diff = {b:+.4f} (SE {se:.4f}, p={p:.3f})  N={n:,}")
    ba, sea, pa, _ = cfit(BASE, d, "T:post:avg")
    emit(f"    baseline T:post:avg  = {ba:+.4f} (SE {sea:.4f}, p={pa:.3f})")

    # (b) univariate 4-way scan
    emit("\n(b) univariate scan: T:post:diff:xz  (BH-FDR across the scan)")
    emit(f"  {'moderator':<14} {'coef':>8} {'SE':>7} {'p':>7} {'N':>7}")
    res = []
    for name in cands:
        bb, ss, pp, nn, _ = scan_one(d, name)
        res.append((name, bb, ss, pp, nn))
        emit(f"  {name:<14} {bb:+8.4f} {ss:7.4f} {pp:7.3f} {nn:7,}")
    rdf = pd.DataFrame(res, columns=["x", "b", "se", "p", "n"]).dropna()
    rdf = rdf.sort_values("p").reset_index(drop=True)
    m_ = len(rdf)
    ps = rdf["p"].values  # BH step-up
    bh = ps * m_ / (np.arange(m_) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    rdf["p_bh"] = bh
    emit("\n  ranked by p, with BH-FDR q-values:")
    for r in rdf.itertuples():
        emit(f"    {r.x:<14} p={r.p:.3f}  q={r.p_bh:.3f}")

    # (c) honest city-split: pick winner on train, test on holdout
    emit("\n(c) honest 50/50 city-split (10 draws): winner-on-train -> holdout")
    cities = d["city"].unique()
    wins = []
    for s in range(10):
        RNG.seed(90026 + s)
        half = RNG.choice(cities, size=len(cities) // 2, replace=False)
        tr, te = d[d["city"].isin(half)], d[~d["city"].isin(half)]
        best, bestt = None, 0.0
        for name in cands:
            x = f"xz_{name}"
            dd = tr.dropna(subset=[x])
            if dd["city"].nunique() < 15:
                continue
            bb, ss, pp, nn = cfit(
                f"y ~ T*post*diff*{x} + T*post*avg + {CTRL}", dd,
                f"T:post:diff:{x}")
            if np.isfinite(bb) and ss > 0 and abs(bb / ss) > abs(bestt):
                best, bestt, bsign = name, bb / ss, np.sign(bb)
        if best is None:
            continue
        x = f"xz_{best}"
        dd = te.dropna(subset=[x])
        bb, ss, pp, nn = cfit(
            f"y ~ T*post*diff*{x} + T*post*avg + {CTRL}", dd,
            f"T:post:diff:{x}")
        ok = (np.sign(bb) == bsign) and (pp < 0.10)
        wins.append(ok)
        emit(f"  split {s}: train winner={best:<13} t={bestt:+.2f} | "
             f"holdout b={bb:+.4f} p={pp:.3f} {'HOLD' if ok else 'fail'}")
    emit(f"  holdout success: {sum(wins)}/{len(wins)} (need ~>5/10 to believe)")

    # (d) LASSO-psi honest correlation
    emit("\n(d) LASSO-psi honest check (couple-level influence signal)")
    try:
        from sklearn.linear_model import LassoCV
        from patsy import dmatrix
        # residualize y and r=T*post*diff on the baseline nuisance
        nuis = (f"T*post + T*post*avg + avg + diff + T:diff + post:diff + "
                f"{CTRL}")
        dd = d.dropna(subset=[f"xz_{c}" for c in cands]).copy()
        emit(f"  complete-case couples={dd['cid'].nunique():,} "
             f"(rows={len(dd):,})")
        my = smf.ols(f"y ~ {nuis}", data=dd).fit()
        dd["r"] = dd["T"] * dd["post"] * dd["diff"]
        mr = smf.ols(f"r ~ {nuis}", data=dd).fit()
        dd["yt"], dd["rt"] = my.resid, mr.resid
        v = (dd["rt"] ** 2).mean()
        dd["psi"] = dd["rt"] * dd["yt"] / v
        cpl = dd.groupby("cid").agg(
            psi=("psi", "mean"), city=("city", "first"),
            **{c: (f"xz_{c}", "first") for c in cands}).reset_index()
        cors = []
        for s in range(10):
            RNG.seed(4200 + s)
            half = RNG.choice(cities, size=len(cities) // 2, replace=False)
            tr = cpl[cpl["city"].isin(half)]
            te = cpl[~cpl["city"].isin(half)]
            if len(tr) < 50 or len(te) < 50:
                continue
            X_tr, X_te = tr[cands].values, te[cands].values
            las = LassoCV(cv=5, random_state=0).fit(X_tr, tr["psi"])
            nsel = int((las.coef_ != 0).sum())
            if nsel == 0:
                cors.append((s, 0, np.nan)); continue
            pred = las.predict(X_te)
            c = np.corrcoef(pred, te["psi"])[0, 1]
            cors.append((s, nsel, c))
        for s, nsel, c in cors:
            emit(f"  split {s}: selected={nsel:>2}  holdout corr(pred,psi)="
                 f"{c if not np.isnan(c) else float('nan'):+.3f}"
                 if nsel else f"  split {s}: LASSO selected NOTHING")
        cc = [c for _, n_, c in cors if n_ and np.isfinite(c)]
        if cc:
            emit(f"  mean holdout corr = {np.mean(cc):+.3f} over {len(cc)} splits")
        else:
            emit("  LASSO never selected on any split -> no learnable w(x)")
    except ImportError:
        emit("  sklearn unavailable, skipped")

    # (e) are the winners one composite? horse-race + urban-SES index
    emit("\n(e) horse-race of scan winners (all 4-ways jointly) + composite")
    tops = ["urban", "homeowner", "eduy_h", "eduy_w"]
    dd = d.dropna(subset=[f"xz_{t}" for t in tops]).copy()
    f = ("y ~ T*post*diff*(" + " + ".join(f"xz_{t}" for t in tops) +
         f") + T*post*avg + {CTRL}")
    m = smf.ols(f, data=dd).fit(cov_type="cluster",
                                cov_kwds={"groups": dd["city"]})
    for t in tops:
        k = f"T:post:diff:xz_{t}"
        emit(f"  {t:<11} {m.params[k]:+.4f} (SE {m.bse[k]:.4f}, "
             f"p={m.pvalues[k]:.3f})")
    corr = d[[f"xz_{t}" for t in tops]].corr().round(2)
    emit("  pairwise corr:\n" + "\n".join(
        "    " + r for r in corr.to_string().split("\n")))
    d["xz_ses"] = d[[f"xz_{t}" for t in tops]].mean(axis=1)
    d["xz_ses"] = (d["xz_ses"] - d["xz_ses"].mean()) / d["xz_ses"].std()
    b, se, p, n = cfit(f"y ~ T*post*diff*xz_ses + T*post*avg + {CTRL}",
                       d.dropna(subset=["xz_ses"]), "T:post:diff:xz_ses")
    emit(f"  urban-SES composite: {b:+.4f} (SE {se:.4f}, p={p:.3f})  N={n:,}")
    for lab, sub in [("hi SES (>=med)", d[d["xz_ses"] >= d["xz_ses"].median()]),
                     ("lo SES (<med)", d[d["xz_ses"] < d["xz_ses"].median()])]:
        b, se, p, n = cfit(BASE, sub, "T:post:diff")
        emit(f"  T:post:diff | {lab:<15} {b:+.4f} (SE {se:.4f}, p={p:.3f}) "
             f"N={n:,}")

    # (f) HONEST composite: per split build top-4 composite from TRAIN ranks
    emit("\n(f) honest composite: train-ranked top-4 -> composite on holdout")
    holds = []
    for s in range(10):
        RNG.seed(90026 + s)   # same splits as (c)
        half = RNG.choice(cities, size=len(cities) // 2, replace=False)
        tr, te = d[d["city"].isin(half)], d[~d["city"].isin(half)]
        ts = []
        for name in cands:
            x = f"xz_{name}"
            ddx = tr.dropna(subset=[x])
            if ddx["city"].nunique() < 15:
                continue
            bb, ss, pp, nn = cfit(
                f"y ~ T*post*diff*{x} + T*post*avg + {CTRL}", ddx,
                f"T:post:diff:{x}")
            if np.isfinite(bb) and ss > 0:
                ts.append((name, bb / ss))
        ts.sort(key=lambda z: -abs(z[1]))
        top4 = ts[:4]
        te = te.copy()
        te["xz_cmp"] = np.nanmean(
            [np.sign(t) * te[f"xz_{n}"] for n, t in top4], axis=0)
        sd = te["xz_cmp"].std()
        te["xz_cmp"] = (te["xz_cmp"] - te["xz_cmp"].mean()) / sd
        bb, ss, pp, nn = cfit(
            f"y ~ T*post*diff*xz_cmp + T*post*avg + {CTRL}",
            te.dropna(subset=["xz_cmp"]), "T:post:diff:xz_cmp")
        ok = (bb > 0) and (pp < 0.10)
        holds.append(ok)
        emit(f"  split {s}: top4={[n for n, _ in top4]} | holdout "
             f"b={bb:+.4f} p={pp:.3f} {'HOLD' if ok else 'fail'}")
    emit(f"  honest composite holdout: {sum(holds)}/{len(holds)}")

    out = os.path.join(OUT, "tables", "couple_wx_scan.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
