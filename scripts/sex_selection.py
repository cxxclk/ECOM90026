# ==========================================================================
# REPLICATION PACKAGE — sex_selection.py
# Tier: CORE
# Paper items: Table 5 (C. Second birth by sex of second child)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/sex_selection.txt
# Imports local modules: none
# Run order: step 24 in run_all.sh
# ==========================================================================

"""NEW topic #2 — does the house-price shock reduce SEX SELECTION, not just the
count of births?

Our locked story: HPR suppresses the son-chasing SECOND birth of firstborn-
daughter one-child families. If those suppressed births are the ones that would
have been sex-selected (male, via sex-selective abortion of female fetuses), the
shock should reduce sex selection. Two observable signatures:

  (1) Decompose the annual second-birth hazard (daughter-first risk set) into a
      SON-birth hazard and a DAUGHTER-birth hazard. Son-chasing => the shock
      cuts the SON-birth hazard much more than the daughter-birth hazard.
  (2) Sex ratio among second births (share male) falls in treated areas post.

Uses the full one-child risk set (woman-years) for power, not just birth events.
Mirrors annual_firstborn_gender.py (same risk set, treatment, FE, clustering).
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FAMCONF = {
    2010: "2010/ecfps2010famconf_nat072016.dta", 2012: "2012/ecfps2012famconf_092015.dta",
    2014: "2014/ecfps2014famconf_170630.dta", 2016: "2016/ecfps2016famconf_201804.dta",
    2018: "2018/ecfps2018famconf_202008.dta", 2020: "2020/ecfps2020famconf_202306.dta",
    2022: "2022/ecfps2022famconf_202410.dta"}

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def load_rosters():
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        g_cols = [c.replace("tb1y", "tb2") for c in by_cols if c.replace("tb1y", "tb2") in meta.column_names]
        df, _ = pyreadstat.read_dta(path, usecols=["pid"] + by_cols + g_cols)
        recs = []
        for i, byc in enumerate(by_cols, start=1):
            gc = f"tb2_a_c{i}"
            sub = df[["pid", byc] + ([gc] if gc in df.columns else [])].copy()
            sub.columns = ["pid", "cby"] + (["cg"] if gc in df.columns else [])
            if "cg" not in sub.columns:
                sub["cg"] = np.nan
            recs.append(sub)
        long = pd.concat(recs, ignore_index=True)
        long = long[(long["cby"] > 1900) & (long["cby"] <= year)]
        long["cg"] = long["cg"].where(long["cg"].isin([0, 1, 5])).replace({5: 0})
        long["rw"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["rw"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    return allw[allw["rw"] == allw["last_wave"]][["pid", "cby", "cg"]], last.reset_index()


def build(y0=2009, y1=2022):
    roster, last_wave = load_rosters()
    panel = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "gender", "birthy", "city_code"])
    women = panel[panel["gender"] == 0].sort_values(["pid", "year"])
    base = women.dropna(subset=["city_code"]).groupby("pid").first()
    attrs = women.groupby("pid").agg(birthy=("birthy", "first")).join(base["city_code"]).dropna().reset_index()
    hpr = pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv")).rename(columns={"treat_spillover": "treat_hpr"})
    attrs = attrs.merge(hpr[["city_code", "is_regulated", "treat_hpr"]], on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]

    r = roster.sort_values(["pid", "cby"]); r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cg"]].rename(columns={"cby": "first_by", "cg": "fb_son"})
    second = r[r["k"] == 1][["pid", "cby", "cg"]].rename(columns={"cby": "second_by", "cg": "sb_son"})
    kids = first.merge(second, on="pid", how="left")
    kids = kids[(kids["second_by"].isna()) | (kids["second_by"] > kids["first_by"])]

    df = attrs.merge(kids, on="pid", how="inner").merge(last_wave, on="pid")
    df = df[df["fb_son"].isin([0, 1])]
    df = df.merge(pd.DataFrame({"year": range(y0, y1 + 1)}), how="cross")
    end = df[["second_by", "last_wave"]].min(axis=1)
    df = df[(df["year"] > df["first_by"]) & (df["year"] <= end.fillna(df["last_wave"]))]
    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 20) & (df["age"] <= 45)]
    df["y"] = (df["year"] == df["second_by"]).astype(float)
    df["y_son"] = ((df["y"] == 1) & (df["sb_son"] == 1)).astype(float)
    df["y_dghtr"] = ((df["y"] == 1) & (df["sb_son"] == 0)).astype(float)
    df["dur"] = df["year"] - df["first_by"]
    df["son"] = df["fb_son"].astype(int)
    df["T"] = df["treat_hpr"].astype(int)
    df["city"] = df["city_code"].astype(int)
    df["post"] = (df["year"] >= 2018).astype(int)
    return df


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  mean(y)={d[formula.split('~')[0].strip()].mean():.4f}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<10} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    df = build()
    CTRL = "age + I(age**2) + dur + I(dur**2)"
    FE = " + C(city) + C(year)"
    emit("SEX SELECTION — does the shock cut SON births more than DAUGHTER births?")
    emit(f"  one-child risk set: {len(df):,} woman-years, {df['pid'].nunique():,} women")

    for s, lab in [(0, "FIRSTBORN-DAUGHTER families (son-chasing margin)"),
                   (1, "FIRSTBORN-SON families (placebo)")]:
        d = df[df["son"] == s]
        n2 = int(d["y"].sum()); nson = int(d["y_son"].sum())
        emit("\n" + "=" * 72)
        emit(f"{lab}")
        emit(f"  2nd births={n2} (sons={nson}, share male={nson/max(n2,1):.3f})")
        fit(f"y_son ~ T:post + {CTRL}{FE}", d, "SON-birth hazard", ["T:post"])
        fit(f"y_dghtr ~ T:post + {CTRL}{FE}", d, "DAUGHTER-birth hazard", ["T:post"])
        fit(f"y ~ T:post + {CTRL}{FE}", d, "ANY 2nd-birth hazard", ["T:post"])

    # sex ratio among second births: share-male DID (conditional on a birth)
    emit("\n" + "=" * 72)
    emit("Sex ratio among 2nd births (share male | birth), daughter-first")
    bsamp = df[(df["y"] == 1) & (df["son"] == 0)].copy()
    bsamp["male"] = bsamp["sb_son"]
    emit(f"  2nd-birth events: {len(bsamp)}")
    tab = bsamp.groupby([bsamp["post"], "T"])["male"].agg(["mean", "count"]).round(3)
    emit(tab.to_string())
    m = smf.ols("male ~ T*post + age", data=bsamp).fit(cov_type="cluster", cov_kwds={"groups": bsamp["city"]})
    emit(f"  DID T:post on share-male = {m.params.get('T:post', np.nan):+.4f} "
         f"(p={m.pvalues.get('T:post', np.nan):.3f})  [thin, descriptive]")

    out = os.path.join(OUT, "tables", "sex_selection.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
