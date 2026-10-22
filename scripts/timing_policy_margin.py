# ==========================================================================
# REPLICATION PACKAGE — timing_policy_margin.py
# Tier: CORE
# Paper items: Table 4 (C. 1.5-child eligibility); Table 8 (Trends and timing: post from 2016, from 2017, 2016–17 dropped); Table 8 (Sample: only women first observed by 2014); Table C2 (Provincial fertility rules)
# Reads: CFPS .dta files in data/20xx/, output/panel_individual.csv
# Writes: output/tables/timing_policy_margin.txt
# Imports local modules: annual_firstborn_gender
# Run order: step 19 in run_all.sh
# ==========================================================================

"""Timing / baseline-wave / 1.5-child-policy-margin checks on the headline
daughter-first annual second-birth DID (annual_firstborn_gender.py spec:
one-child risk set 2009-22, ages 20-45, unregulated prefectures, city+year FE,
age/age^2/dur/dur^2, cluster city).
  1. Post = 2016 / 2017 / 2018, and Post=2018 dropping 2016-17 years.
  2. Baseline wave (first wave with a city code) composition; re-estimate on
     women first observed by 2014.
  3. 1.5-child margin: eligible = rural (agricultural) hukou at baseline x
     1.5-child province (standard Gu et al. 2007 classification, as in App. C2);
     daughter/son T:post by eligibility, interactions, 4-way DDD, and the
     pre-shock daughter-minus-son premium by eligibility.
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
import annual_firstborn_gender as afg  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


ADULT = {
    2010: ("2010/ecfps2010adult_201906.dta", "qa2"),
    2012: ("2012/ecfps2012adult_202505.dta", "qa301"),
    2014: ("2014/ecfps2014adult_201906.dta", "qa301"),
    2016: ("2016/ecfps2016adult_201906.dta", "pa301"),
    2018: ("2018/ecfps2018person_202012.dta", "qa301"),
    2020: ("2020/ecfps2020person_202306.dta", "qa301"),
    2022: ("2022/ecfps2022person_202410.dta", "qa301"),
}
# provcd (GB 2-digit) classification
ONE_CHILD = {11, 12, 31, 32, 50, 51}          # BJ TJ SH JS CQ SC
TWO_CHILD = {46, 53, 63, 64, 65}              # HI YN QH NX XJ (rural)
TIBET = {54}

CTRL = "age + I(age**2) + dur + I(dur**2)"
FE = " + C(city) + C(year)"


def load_hukou():
    """Long pid-year hukou: 1=agricultural, 0=non-agricultural (codes 1/3);
    2020+ code 7 ('Residence', unified hukou) and negatives -> NaN."""
    fr = []
    for y, (f, v) in ADULT.items():
        d, _ = pyreadstat.read_dta(os.path.join(DATA, f), usecols=["pid", v])
        d = d.rename(columns={v: "hk"})
        d["rural_hk"] = np.where(d["hk"] == 1, 1.0, np.where(d["hk"] == 3, 0.0, np.nan))
        d["year"] = y
        fr.append(d[["pid", "year", "rural_hk"]])
    return pd.concat(fr, ignore_index=True)


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})
    emit(f"[{label}] N={int(m.nobs):,} women={d['pid'].nunique():,} "
         f"cities={d['city'].nunique()} births={int(d['y'].sum())} mean={d['y'].mean():.4f}")
    for k in keys:
        if k in m.params:
            emit(f"    {k:<26} {m.params[k]*100:+.2f}pp (SE {m.bse[k]*100:.2f}, p={m.pvalues[k]:.3f})")
    return m


def trio(d, label):
    emit(f"--- {label}")
    fit(f"y ~ T:post + {CTRL}{FE}", d[d.son == 0], "daughter", ["T:post"])
    fit(f"y ~ T:post + {CTRL}{FE}", d[d.son == 1], "son", ["T:post"])
    fit(f"y ~ T:post + T:post:son + post:son + T:son + son + {CTRL}{FE}", d, "DDD",
        ["T:post", "T:post:son", "T:son"])


def main():
    df = afg.build_hazard_panel()
    # baseline wave / province (same rule as build_hazard_panel: first wave with city)
    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "year", "gender", "provcd", "city_code"])
    w = pan[pan.gender == 0].dropna(subset=["city_code"]).sort_values(["pid", "year"])
    b = w.groupby("pid").first()[["year", "provcd"]].rename(
        columns={"year": "base_year", "provcd": "base_prov"}).reset_index()
    df = df.merge(b, on="pid", how="left")

    emit("Reproduce baseline (Post=2018)")
    df["post"] = (df.year >= 2018).astype(int)
    trio(df, "BASELINE Post=2018, 2009-2022")

    # ---------------- 1. timing
    emit("\n" + "=" * 72 + "\n1. TIMING")
    for py in (2016, 2017):
        df["post"] = (df.year >= py).astype(int)
        trio(df, f"Post={py}")
    df["post"] = (df.year >= 2018).astype(int)
    trio(df[~df.year.isin([2016, 2017])], "Post=2018, drop 2016-17 years")

    # ---------------- 2. baseline wave
    emit("\n" + "=" * 72 + "\n2. BASELINE WAVE composition (one-child panel)")
    g = df.groupby(["base_year", "T", "son"]).agg(
        women=("pid", "nunique"), wy=("y", "size"), births=("y", "sum"))
    emit(g.to_string())
    late = df[df.base_year >= 2016]
    emit(f"\nBaseline >=2016: women={late.pid.nunique():,} of {df.pid.nunique():,} "
         f"({late.pid.nunique()/df.pid.nunique():.1%}); woman-years={len(late):,} "
         f"({len(late)/len(df):.1%}); births={int(late.y.sum())} ({late.y.sum()/df.y.sum():.1%})")
    emit("Baseline>=2016 by arm x exposure (women / woman-years / births / post-2018 wy):")
    emit(late.assign(p18=(late.year >= 2018).astype(int)).groupby(["son", "T"]).agg(
        women=("pid", "nunique"), wy=("y", "size"), births=("y", "sum"),
        post_wy=("p18", "sum")).to_string())
    emit("Share of daughter-first POST(>=2018) woman-years from baseline>=2016, by T:")
    dp = df[(df.son == 0) & (df.year >= 2018)]
    emit(dp.groupby("T").apply(lambda x: (x.base_year >= 2016).mean()).round(3).to_string())
    trio(df[df.base_year <= 2014], "Post=2018, women first observed by 2014")

    # ---------------- 3. 1.5-child margin
    emit("\n" + "=" * 72 + "\n3. 1.5-CHILD POLICY MARGIN")
    hk = load_hukou()
    # hukou at baseline wave; if missing there, earliest valid reading <=2016
    hb = df[["pid", "base_year"]].drop_duplicates().merge(
        hk.rename(columns={"year": "base_year"}), on=["pid", "base_year"], how="left")
    alt = (hk.dropna(subset=["rural_hk"]).query("year <= 2016").sort_values("year")
           .groupby("pid").first()["rural_hk"].rename("hk_alt"))
    hb = hb.merge(alt, on="pid", how="left")
    hb["src"] = np.where(hb.rural_hk.notna(), "base", np.where(hb.hk_alt.notna(), "alt", "none"))
    hb["rural_hk"] = hb.rural_hk.fillna(hb.hk_alt)
    emit("hukou source (women): " + hb.src.value_counts().to_dict().__repr__())
    df = df.merge(hb[["pid", "rural_hk"]], on="pid", how="left")
    df["pol"] = np.select([df.base_prov.isin(ONE_CHILD), df.base_prov.isin(TWO_CHILD),
                           df.base_prov.isin(TIBET)], ["one", "two", "tibet"], "onehalf")
    emit("Policy class used: one-child {11,12,31,32,50,51}; two-child rural {46,53,63,64,65};"
         " Tibet 54; 1.5-child = all other provcd (Gu et al. 2007 standard = App. C2).")
    emit("Women by baseline province class x rural hukou x T:")
    emit(df.groupby(["pol", "rural_hk", "T"]).pid.nunique().unstack([2]).to_string())
    d3 = df.dropna(subset=["rural_hk"]).copy()
    d3["elig"] = ((d3.rural_hk == 1) & (d3.pol == "onehalf")).astype(int)

    # pre-shock premium
    emit("\nPre-shock (2009-2015) raw second-birth rate, daughter minus son, by eligibility x T:")
    pre = d3[d3.year <= 2015]
    tab = pre.groupby(["elig", "T", "son"]).y.agg(["mean", "size"]).unstack("son")
    tab["premium_pp"] = (tab[("mean", 0)] - tab[("mean", 1)]) * 100
    emit(tab.round(4).to_string())
    emit("Pre-shock premium regression (2009-15): y ~ dghtr + dghtr:elig + elig + ctrl + city+year FE")
    pre = pre.assign(dghtr=1 - pre.son)
    fit(f"y ~ dghtr + dghtr:elig + elig + {CTRL}{FE}", pre, "premium", ["dghtr", "dghtr:elig", "elig"])
    # also by policy cell
    emit("Pre-shock premium by cell (rural x class), pooled T:")
    cell = pre.assign(c=np.where(pre.rural_hk == 1, "rural_", "urban_") + pre.pol)
    t2 = cell.groupby(["c", "son"]).y.agg(["mean", "size"]).unstack("son")
    t2["premium_pp"] = (t2[("mean", 0)] - t2[("mean", 1)]) * 100
    emit(t2.round(4).to_string())

    for py in (2018,):
        d3["post"] = (d3.year >= py).astype(int)
        emit(f"\n--- Post={py}: split by eligibility")
        for e in (1, 0):
            for s, lab in ((0, "daughter"), (1, "son")):
                fit(f"y ~ T:post + {CTRL}{FE}", d3[(d3.elig == e) & (d3.son == s)],
                    f"elig={e} {lab}", ["T:post"])
        emit("--- interaction within arm (lower-order: T:elig, post:elig, elig)")
        for s, lab in ((0, "daughter"), (1, "son")):
            fit(f"y ~ T:post + T:post:elig + post:elig + T:elig + elig + {CTRL}{FE}",
                d3[d3.son == s], f"{lab} x elig", ["T:post", "T:post:elig", "T:elig"])
        emit("--- 4-way: T x post x daughter x elig (all lower-order)")
        d3["dg"] = 1 - d3.son
        fit(f"y ~ T:post*dg*elig + T:dg*elig + post:dg*elig + {CTRL}{FE}", d3, "4-way",
            ["T:post", "T:post:dg", "T:post:elig", "T:post:dg:elig", "T:dg:elig"])
        emit("--- variant: drop rural two-child-province women from the non-eligible group")
        d4 = d3[~((d3.rural_hk == 1) & (d3.pol == "two"))]
        for s, lab in ((0, "daughter"), (1, "son")):
            fit(f"y ~ T:post + T:post:elig + post:elig + T:elig + elig + {CTRL}{FE}",
                d4[d4.son == s], f"{lab} x elig (excl rural 2-child)", ["T:post", "T:post:elig"])

    out = os.path.join(OUT, "tables", "timing_policy_margin.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("Saved ->", out)


if __name__ == "__main__":
    main()
