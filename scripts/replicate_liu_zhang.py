# ==========================================================================
# REPLICATION PACKAGE — replicate_liu_zhang.py
# Tier: SUPPLEMENTARY
# Paper items: Section 5.1 first paragraph (Liu and Zhang Table 3 replication)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/replication_liu_zhang.txt
# Imports local modules: none
# Run order: step 52 in run_all.sh
# ==========================================================================

"""Replicate Liu & Zhang (2025) Table 3: individual-level DID of HPR spillovers
on the annual number of newborns.

Their design (original paper.pdf, Table 3):
  - Sample: women of childbearing age (15-44) in unregulated prefectures
  - Outcome: annual number of newborns per woman, reconstructed from CFPS
    children's birth years, 2009-2020
  - Treat = prefecture within 250 km of nearest regulated city (our treat_hpr)
  - Post  = year >= 2018 (first full year after shock + pregnancy delay)
  - FE: individual + year; trends: prefecture-specific or group-specific linear
  - Cluster: prefecture
  - Result: Treat x Post = -0.023 to -0.026 (se ~0.010), sample mean 0.061,
    N = 80,177

What differs on our side: HPR city list extracted from their appendix (same),
distances from China shapefile centroids (theirs from city coordinates),
CFPS waves 2010-2022 (theirs 2010-2020), no CityRE price data (not needed:
the fertility DID is distance-based).

Specs reported:
  (1) faithful: women 15-44, 2009-2020, indiv+year FE
  (2) faithful + treatment-group linear trend (their col 2)
  (3) extended years 2009-2022
  (4) our analytic restriction: married women 20-45 (carry-forward marital),
      2009-2020 -- diagnoses how much of the average effect runs through
      the unmarried/marriage-market margin
"""
import os
import re
import numpy as np
import pandas as pd
import pyreadstat
from linearmodels.panel import PanelOLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

FAMCONF = {
    2010: "2010/ecfps2010famconf_nat072016.dta",
    2012: "2012/ecfps2012famconf_092015.dta",
    2014: "2014/ecfps2014famconf_170630.dta",
    2016: "2016/ecfps2016famconf_201804.dta",
    2018: "2018/ecfps2018famconf_202008.dta",
    2020: "2020/ecfps2020famconf_202306.dta",
    2022: "2022/ecfps2022famconf_202410.dta",
}


def load_child_rosters():
    """For each pid, keep the child birth-year roster from the LATEST famconf
    wave they appear in (most complete history)."""
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        df, _ = pyreadstat.read_dta(path, usecols=["pid"] + cols)
        long = df.melt(id_vars="pid", value_vars=cols, value_name="cby")
        long = long[(long["cby"] > 1900) & (long["cby"] <= year)]
        long["roster_wave"] = year
        frames.append(long[["pid", "cby", "roster_wave"]])
        print(f"  famconf {year}: {len(df):,} persons, "
              f"{len(cols)} child cols, {len(long):,} valid child-years")
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["roster_wave"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    roster = allw[allw["roster_wave"] == allw["last_wave"]][["pid", "cby"]]
    return roster, last.reset_index()


def load_women():
    """Per-person attributes from the stacked adult panel: female, birth year,
    baseline (earliest-wave) city, marital history."""
    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "gender", "birthy", "marital", "city_code"])
    panel = panel[panel["gender"] == 0]
    panel = panel.sort_values(["pid", "year"])

    base = panel.dropna(subset=["city_code"]).groupby("pid").first()
    attrs = panel.groupby("pid").agg(birthy=("birthy", "first"))
    attrs = attrs.join(base["city_code"]).dropna(subset=["birthy", "city_code"])

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    attrs = attrs.reset_index().merge(
        hpr[["city_code", "is_regulated", "treat_hpr"]], on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]

    marital = panel[["pid", "year", "marital"]].dropna()
    return attrs, marital


def build_annual_panel(attrs, roster, last_wave, y0=2009, y1=2020):
    births = roster.groupby(["pid", "cby"]).size().rename("newborns").reset_index()
    df = attrs.merge(last_wave, on="pid", how="inner")
    df = df.merge(pd.DataFrame({"year": range(y0, y1 + 1)}), how="cross")
    df = df[df["year"] <= df["last_wave"].clip(upper=y1) + 0]
    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 15) & (df["age"] <= 44)]
    df = df.merge(births, left_on=["pid", "year"], right_on=["pid", "cby"], how="left")
    df["newborns"] = df["newborns"].fillna(0.0)
    df["post"] = (df["year"] >= 2018).astype(float)
    df["did"] = df["treat_hpr"] * df["post"]
    df["grp_trend"] = df["treat_hpr"] * (df["year"] - y0)
    return df.drop(columns=["cby"])


def run_spec(df, label, group_trend=False, pref_trend=False):
    d = df.set_index(["pid", "year"])
    rhs = ["did"] + (["grp_trend"] if group_trend else [])
    if pref_trend:
        trend = (df["year"] - df["year"].min()).to_numpy()
        dum = pd.get_dummies(df["city_code"].astype(int), prefix="ct",
                             dtype=float, drop_first=True)
        dum = dum.mul(trend, axis=0)
        dum.index = d.index
        d = pd.concat([d, dum], axis=1)
        rhs = rhs + list(dum.columns)
    mod = PanelOLS(d["newborns"], d[rhs], entity_effects=True, time_effects=True,
                   drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", clusters=d["city_code"])
    b, se, p = res.params["did"], res.std_errors["did"], res.pvalues["did"]
    line = (f"{label:<52} {b:>8.4f} ({se:.4f}) p={p:.3f}  "
            f"mean={df['newborns'].mean():.3f}  N={res.nobs:,}  "
            f"women={df['pid'].nunique():,}  cities={df['city_code'].nunique()}")
    print(line)
    return line


def main():
    print("Loading famconf child rosters ...")
    roster, last_wave = load_child_rosters()
    print("Loading woman attributes ...")
    attrs, marital = load_women()
    print(f"  women with baseline city & unregulated: {len(attrs):,} "
          f"(treated share {attrs['treat_hpr'].mean():.3f})")

    lines = ["Replication of Liu & Zhang (2025) Table 3",
             "Target: Treat x Post = -0.023 to -0.026 (se ~0.010), mean 0.061, N=80,177",
             ""]

    panel = build_annual_panel(attrs, roster, last_wave, 2009, 2020)
    lines.append(run_spec(panel, "(0) women 15-44, 2009-2020, no trend (not in L&Z)"))
    lines.append(run_spec(panel, "(1)   + prefecture linear trends   [their col 1]",
                          pref_trend=True))
    lines.append(run_spec(panel, "(2)   + treatment-group linear trend [their col 2]",
                          group_trend=True))

    panel22 = build_annual_panel(attrs, roster, last_wave, 2009, 2022)
    lines.append(run_spec(panel22, "(3) extended to 2022, group trend", group_trend=True))

    # (4) our analytic restriction: married (carry-forward) & age 20-45
    m = marital.rename(columns={"year": "wave"}).sort_values(["pid", "wave"])
    pm = panel.sort_values(["pid", "year"]).copy()
    pm = pd.merge_asof(pm.sort_values("year"), m.sort_values("wave"),
                       left_on="year", right_on="wave", by="pid",
                       direction="backward")
    sub = pm[(pm["marital"] == 2) & (pm["age"] >= 20) & (pm["age"] <= 45)]
    lines.append(run_spec(sub, "(4) married 20-45 (our restriction), 2009-2020"))
    lines.append(run_spec(sub, "(5)   + treatment-group linear trend", group_trend=True))
    unm = pm[(pm["marital"] != 2) | (pm["age"] < 20) | (pm["age"] > 45)]
    lines.append(run_spec(unm, "(6) complement (unmarried or outside 20-45)",
                          group_trend=True))

    out = os.path.join(OUT, "tables", "replication_liu_zhang.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
