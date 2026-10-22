# ==========================================================================
# REPLICATION PACKAGE — response_profile_typology.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Response-profile shrinkage)
# Reads: CFPS .dta files in data/20xx/, output/panel_individual.csv
# Writes: output/response_profile_strata.csv, output/tables/response_profile_typology.txt
# Imports local modules: annual_firstborn_gender, cddf_spine
# Run order: step 43 in run_all.sh
# ==========================================================================

"""Response-profile typology: classify strata by HOW the shock hits them.

New classification approach #2 (2026-07-21 session 2). Every earlier typology
clustered families by WHO THEY ARE (covariates / behaviors / trajectories) —
three independent nulls, because the effect is motive-based. Here covariates
only set the RESOLUTION (a coarse interpretable grid of strata); the
classification axis is the multi-outcome RESPONSE VECTOR of each stratum:

  h2      annual second-birth hazard DID          (quantity margin)
  h2_son  second-birth-of-a-SON hazard DID        (sex-selection margin)
  h2_dau  second-birth-of-a-DAUGHTER hazard DID
  linc    wave-level log family income p.c. DID   (distributional margin)
  edu     wave-level ihs(education spending) DID  (quality margin, 2014+)

Strata: firstborn gender (2) x city son-norm tercile (3) x family-income half
(2) x urban (2) = 24 cells (thin cells dropped). Each stratum x outcome DID is
estimated with the project-standard FE spec, then EMPIRICAL-BAYES SHRUNKEN
(column-wise: tau2 = max(0, var(coef) - mean(se2)); shrunk = coef *
tau2/(tau2+se2)) so clustering runs on signal, not noise. Ward clustering on
the standardized shrunk matrix -> "response types" (quantity-suppressed /
sex-selection / quality-substituting / income-hit / unaffected), which may cut
ACROSS the known margins. Per-type pooled confirmation DIDs (post-hoc,
exploratory) with pre-trend gates. Cross-margin substitution structure
(quantity -> quality, quantity -> selection) is visible ONLY in this design.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cddf_spine as cs                                  # noqa: E402
from annual_firstborn_gender import load_rosters         # noqa: E402

CTRL, FE = cs.CTRL, cs.FE
MIN_WY = 800

FAMECON = {
    2014: ("2014/ecfps2014famecon_201906.dta", "fid14"),
    2016: ("2016/ecfps2016famecon_201807.dta", "fid16"),
    2018: ("2018/ecfps2018famecon_202101.dta", "fid18"),
    2020: ("2020/ecfps2020famecon_202306.dta", "fid20"),
    2022: ("2022/ecfps2022famecon_202410.dta", "fid22"),
}

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def second_child_sex(df):
    """Merge the SECOND child's gender onto the hazard panel (per pid)."""
    roster, _ = load_rosters()
    r = roster.sort_values(["pid", "cby"]).copy()
    r["k"] = r.groupby("pid").cumcount()
    sec = r[r["k"] == 1][["pid", "cgender"]].rename(
        columns={"cgender": "c2_son"})
    df = df.merge(sec, on="pid", how="left")
    df["y_son"] = df["y"] * (df["c2_son"] == 1).astype(float)
    df["y_dau"] = df["y"] * (df["c2_son"] == 0).astype(float)
    # births with unknown 2nd-child sex contribute to neither margin
    unk = (df["y"] == 1) & (~df["c2_son"].isin([0, 1]))
    df.loc[unk, ["y_son", "y_dau"]] = np.nan
    return df


def wave_panel(df):
    """Wave-level obs of risk-set women in their parity-1 years, with
    log family income p.c. and education spending."""
    wl = df.drop_duplicates("pid")[
        ["pid", "first_by", "second_by", "T", "city", "birthy", "stratum"]]
    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "fid", "year", "fam_income_pc"])
    w = pan.merge(wl, on="pid", how="inner")
    w = w[(w["year"] >= 2010) & (w["year"] <= 2022)]
    end = w["second_by"].fillna(9999)
    w = w[(w["year"] > w["first_by"]) & (w["year"] <= end)]
    w["age"] = w["year"] - w["birthy"]
    w["dur"] = w["year"] - w["first_by"]
    w["post"] = (w["year"] >= 2018).astype(int)
    w["linc"] = np.log1p(w["fam_income_pc"].clip(lower=0))

    frames = []
    for yr, (rel, fidc) in FAMECON.items():
        d, _ = pyreadstat.read_dta(os.path.join(DATA, rel),
                                   usecols=[fidc, "fp510"])
        d.columns = ["fid", "edu_exp"]
        d["edu_exp"] = pd.to_numeric(d["edu_exp"], errors="coerce")
        d.loc[d["edu_exp"] < 0, "edu_exp"] = np.nan
        d["year"] = yr
        frames.append(d.dropna(subset=["fid"]))
    ee = pd.concat(frames, ignore_index=True).drop_duplicates(["fid", "year"])
    w = w.merge(ee, on=["fid", "year"], how="left")
    w["ihs_edu"] = np.arcsinh(w["edu_exp"])
    match = w.loc[w["year"] >= 2014, "edu_exp"].notna().mean()
    emit(f"wave panel: {len(w):,} parity-1 wave-obs; edu_exp match rate "
         f"2014+ = {match:.0%}")
    return w


def did(d, yvar, ctrl):
    d = d[d[yvar].notna()].copy()
    if (len(d) < 200 or d["T"].nunique() < 2
            or d.loc[d["post"] == 1, yvar].notna().sum() < 30):
        return np.nan, np.nan, len(d)
    try:
        m = smf.ols(f"{yvar} ~ T:post + {ctrl}{FE}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        return m.params["T:post"], m.bse["T:post"], int(m.nobs)
    except Exception:
        return np.nan, np.nan, len(d)


def main():
    df = cs.assemble()
    df = second_child_sex(df)

    wlq = df.drop_duplicates("pid")
    norm_t = wlq["city_son_norm"].quantile([1 / 3, 2 / 3]).values
    inc_med = wlq["log_faminc_pc"].median()
    df["norm3"] = np.where(df["city_son_norm"].isna(), -1,
                           np.searchsorted(norm_t, df["city_son_norm"]))
    df["inc2"] = np.where(df["log_faminc_pc"].isna(), -1,
                          (df["log_faminc_pc"] > inc_med).astype(int))
    df["urb"] = df["urban"].where(df["urban"].isin([0, 1]))
    df = df[(df["norm3"] >= 0) & (df["inc2"] >= 0) & df["urb"].notna()]
    df["stratum"] = (df["son"].astype(int).astype(str) + "|"
                     + df["norm3"].astype(int).astype(str) + "|"
                     + df["inc2"].astype(int).astype(str) + "|"
                     + df["urb"].astype(int).astype(str))
    sizes = df.groupby("stratum").size()
    keep = sizes[sizes >= MIN_WY].index
    emit("RESPONSE-PROFILE TYPOLOGY — strata classified by multi-outcome DID "
         "vectors")
    emit(f"grid: son x norm-tercile x income-half x urban; "
         f"{len(sizes)} cells, {len(keep)} kept (>= {MIN_WY} woman-years), "
         f"{df[df['stratum'].isin(keep)]['pid'].nunique():,} women")
    df = df[df["stratum"].isin(keep)]

    w = wave_panel(df)

    outcomes = []
    for st in sorted(keep):
        d = df[df["stratum"] == st]
        row = {"stratum": st, "wy": len(d),
               "prehaz": d.loc[d["post"] == 0, "y"].mean()}
        for yv, lab in [("y", "h2"), ("y_son", "h2_son"), ("y_dau", "h2_dau")]:
            c, s, n = did(d, yv, CTRL)
            row[lab], row[lab + "_se"] = c, s
        dw = w[w["stratum"] == st]
        for yv, lab in [("linc", "linc"), ("ihs_edu", "edu")]:
            c, s, n = did(dw, yv, "age + I(age**2)")
            row[lab], row[lab + "_se"] = c, s
        outcomes.append(row)
    tab = pd.DataFrame(outcomes).set_index("stratum")

    cols = ["h2", "h2_son", "h2_dau", "linc", "edu"]
    emit("\nraw stratum x outcome DID matrix "
         "(stratum = son|norm3|inchalf|urban; * = |t|>1.64, ** = |t|>1.96):")
    hdr = f"  {'stratum':<12}{'wy':>7}{'prehaz':>8}"
    for c in cols:
        hdr += f"{c:>12}"
    emit(hdr)
    for st, r in tab.iterrows():
        line = f"  {st:<12}{int(r['wy']):>7,}{r['prehaz']:>8.3f}"
        for c in cols:
            v, s = r[c], r[c + "_se"]
            if np.isnan(v):
                line += f"{'--':>12}"
            else:
                star = ("**" if abs(v / s) > 1.96 else
                        "*" if abs(v / s) > 1.64 else "")
                line += f"{v:>+10.3f}{star:<2}"
        emit(line)

    # empirical-Bayes shrinkage per outcome column, then standardize
    Z = pd.DataFrame(index=tab.index)
    emit("\nEB shrinkage (tau2 = max(0, var(coef) - mean(se2))):")
    for c in cols:
        v, s = tab[c], tab[c + "_se"]
        ok = v.notna() & s.notna()
        tau2 = max(0.0, v[ok].var() - (s[ok] ** 2).mean())
        shr = v * tau2 / (tau2 + s ** 2)
        emit(f"  {c:<8} tau={np.sqrt(tau2):.4f}  mean|shrunk|/mean|raw| = "
             f"{(shr[ok].abs().mean() / max(v[ok].abs().mean(), 1e-9)):.2f}")
        sd = shr[ok].std()
        Z[c] = (shr / sd if sd > 0 else shr).fillna(0)

    X = Z.values
    lk = linkage(X, method="ward")
    for K in (3, 4):
        lab = fcluster(lk, K, criterion="maxclust")
        Z[f"k{K}"] = lab
    tab["k3"], tab["k4"] = Z["k3"], Z["k4"]

    for K in (3, 4):
        emit(f"\n=== WARD K={K} response types (on shrunk-standardized "
             f"vectors) ===")
        for cl in range(1, K + 1):
            members = tab[tab[f"k{K}"] == cl]
            if not len(members):
                continue
            emit(f"\n[type {cl}] strata: {', '.join(members.index)}  "
                 f"(wy={int(members['wy'].sum()):,})")
            emit("  mean shrunk-z profile: "
                 + "  ".join(f"{c}={Z.loc[members.index, c].mean():+.2f}"
                             for c in cols))
            # pooled confirmation DIDs (post-hoc, exploratory)
            d = df[df["stratum"].isin(members.index)]
            m = smf.ols(f"y ~ T:post + {CTRL}{FE}", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["city"]})
            for yr in range(2010, 2015):
                d = d.assign(**{f"Ty{yr}":
                                d["T"] * (d["year"] == yr).astype(int)})
            terms = " + ".join(f"Ty{yr}" for yr in range(2010, 2015))
            d = d.assign(Tpost=d["T"] * d["post"])
            mp = smf.ols(f"y ~ {terms} + Tpost + {CTRL}{FE}", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["city"]})
            viol = sum(mp.pvalues[f"Ty{yr}"] < 0.10 for yr in range(2010, 2015))
            dw = w[w["stratum"].isin(members.index)]
            mi = smf.ols(f"linc ~ T:post + age + I(age**2){FE}",
                         data=dw[dw["linc"].notna()]).fit(
                cov_type="cluster", cov_kwds={"groups":
                                              dw[dw["linc"].notna()]["city"]})
            de = dw[dw["ihs_edu"].notna()]
            me_ = smf.ols(f"ihs_edu ~ T:post + age + I(age**2){FE}",
                          data=de).fit(
                cov_type="cluster", cov_kwds={"groups": de["city"]})
            emit(f"  pooled h2 DID {m.params['T:post']:+.4f} "
                 f"(p={m.pvalues['T:post']:.3f}, pre-viol {viol}/5)   "
                 f"linc {mi.params['T:post']:+.3f} "
                 f"(p={mi.pvalues['T:post']:.3f})   "
                 f"edu {me_.params['T:post']:+.3f} "
                 f"(p={me_.pvalues['T:post']:.3f})")

    tab.reset_index().to_csv(
        os.path.join(OUT, "response_profile_strata.csv"), index=False)
    emit("\nstratum table saved -> output/response_profile_strata.csv")
    out = os.path.join(OUT, "tables", "response_profile_typology.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
