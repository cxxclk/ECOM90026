# ==========================================================================
# REPLICATION PACKAGE — annual_parity_hazard.py
# Tier: CORE
# Paper items: Table 2 (A. Any birth); Table 2 (B. By number of children)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/annual_parity_hazard.txt
# Imports local modules: none
# Run order: step 11 in run_all.sh
# ==========================================================================

"""Annualize fertility layers 1-2: parity-stratified annual birth hazard.

Migrates the wave-level layer-1 (average effect) and layer-2 (the margin is the
second birth) results onto the same annual discrete-time hazard design already
used for layer 3 (annual_firstborn_gender.py). Motivation (see CLAUDE.md
"Design migration decision"): the wave-level new_birth binary suffers 20-35%
roster-lag underreporting, has no testable annual pre-trend, and pools tempo
against quantum. The annual hazard fixes all three.

Three parity-transition risk sets, built from the latest-wave famconf roster:
  k=0  FIRST-birth hazard   (childless risk set)   -> layer-1 extensive margin
  k=1  SECOND-birth hazard  (one-child risk set)   -> layer-2 headline margin
  k=2  THIRD+-birth hazard  (2+ risk set)          -> layer-2 placebo
Plus a pooled ANY-birth hazard across all parities -> layer-1 average effect.

Risk-set rule (woman i, year t): at risk for the k->k+1 transition iff she has
EXACTLY k children as of end of t-1.
  k>=1:  t in (kth_birth_year, (k+1)th_birth_year] capped by last famconf wave
  k=0 :  t in [max(y0, birthy+20), first_birth_year] capped by last wave
Outcome y_it = 1{(k+1)th birth in t}. Women exit after the transition.
Sample: married women 20-45, unregulated prefectures (regulated dropped),
2009-2022. FE city+year, age (+ duration for k>=1) controls, cluster by city.
Post=2018 headline (pregnancy delay), Post=2016 variant, group-trend robustness,
annual event study (2015 base).
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
    2010: "2010/ecfps2010famconf_nat072016.dta",
    2012: "2012/ecfps2012famconf_092015.dta",
    2014: "2014/ecfps2014famconf_170630.dta",
    2016: "2016/ecfps2016famconf_201804.dta",
    2018: "2018/ecfps2018famconf_202008.dta",
    2020: "2020/ecfps2020famconf_202306.dta",
    2022: "2022/ecfps2022famconf_202410.dta",
}

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def load_rosters():
    """Per pid from the LATEST famconf wave: every (child birth year, gender)."""
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        df, _ = pyreadstat.read_dta(path, usecols=["pid"] + by_cols)
        recs = []
        for byc in by_cols:
            sub = df[["pid", byc]].copy()
            sub.columns = ["pid", "cby"]
            recs.append(sub)
        long = pd.concat(recs, ignore_index=True)
        long = long[(long["cby"] > 1900) & (long["cby"] <= year)]
        long["roster_wave"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["roster_wave"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    roster = allw[allw["roster_wave"] == allw["last_wave"]]
    return roster[["pid", "cby"]], last.reset_index()


def woman_attrs():
    """Baseline city (earliest non-missing wave), birth year, married-ever flag,
    urban; HPR treatment from baseline city; drop regulated cities."""
    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "gender", "birthy", "city_code", "urban",
                 "marital"])
    women = panel[panel["gender"] == 0].sort_values(["pid", "year"])
    base = women.dropna(subset=["city_code"]).groupby("pid").first()
    women["_married"] = (women["marital"] == 2).astype(int)
    attrs = women.groupby("pid").agg(birthy=("birthy", "first"),
                                     urban=("urban", "max"),
                                     ever_married=("_married", "max"))
    attrs = attrs.join(base[["city_code"]]).dropna(subset=["city_code", "birthy"])
    # ever-married only (mirrors the wave-level "married women 20-45" sample;
    # the parity>=1 risk sets already condition on a prior birth, but the
    # childless first-birth set otherwise admits never-married young women).
    attrs = attrs[attrs["ever_married"] == 1].reset_index()

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    attrs = attrs.merge(hpr[["city_code", "is_regulated", "treat_hpr"]],
                        on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]
    return attrs


def parity_table(roster):
    """Per pid: ordered birth years by1<=by2<=by3 (NaN if absent), n_kids."""
    r = roster.sort_values(["pid", "cby"]).copy()
    r["k"] = r.groupby("pid").cumcount()
    wide = r.pivot_table(index="pid", columns="k", values="cby", aggfunc="first")
    wide = wide.rename(columns={i: f"by{i+1}" for i in wide.columns})
    for c in ("by1", "by2", "by3"):
        if c not in wide.columns:
            wide[c] = np.nan
    wide["n_kids"] = r.groupby("pid").size()
    return wide.reset_index()


def build_risk_set(attrs, parity, y0=2009, y1=2022):
    """Discrete-time risk set for the k->k+1 transition (k=parity).

    Returns a woman-year panel with outcome y, age, dur (k>=1), T, city.
    """
    pt = parity_table(LOAD_ROSTER).merge(attrs, on="pid", how="inner")
    _, last_wave = LOAD_LAST
    pt = pt.merge(last_wave, on="pid", how="left")

    if parity == 0:
        prev_by = pd.Series(np.nan, index=pt.index)   # no prior birth
        next_by = pt["by1"]
        # never twins-at-zero issue; enter at age 20
        start = np.maximum(y0, pt["birthy"] + 20)
    else:
        prev_by = pt[f"by{parity}"]
        next_by = pt[f"by{parity+1}"]
        # must actually have reached parity k (prev birth observed)
        pt = pt[prev_by.notna()].copy()
        prev_by = prev_by[pt.index]
        next_by = next_by[pt.index]
        start = np.maximum(y0, prev_by + 1)
        # drop twins / same-year ordering anomalies at the entering parity
        ok = next_by.isna() | (next_by > prev_by)
        pt, prev_by, next_by, start = pt[ok], prev_by[ok], next_by[ok], start[ok]

    end = pd.concat([next_by, pt["last_wave"]], axis=1).min(axis=1)
    pt = pt.assign(_start=start.values, _end=end.values,
                   _next=next_by.values, _prev=prev_by.values)

    grid = pd.DataFrame({"year": range(y0, y1 + 1)})
    df = pt.merge(grid, how="cross")
    df = df[(df["year"] >= df["_start"]) & (df["year"] <= df["_end"])]
    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 20) & (df["age"] <= 45)]

    df["y"] = (df["year"] == df["_next"]).astype(float)
    if parity >= 1:
        df["dur"] = df["year"] - df["_prev"]
    df["T"] = df["treat_hpr"].astype(int)
    df["city"] = df["city_code"].astype(int)
    return df


def fit(formula, d, label, keys=None):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    keys = keys or [k for k in m.params.index if ":" in k]
    emit(f"\n[{label}]  N={int(m.nobs):,}  women={d['pid'].nunique():,}  "
         f"mean(y)={d['y'].mean():.4f}")
    for k in keys:
        emit(f"  {k:<14} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def event_study(d, label, ctrl, fe):
    ev = [y for y in range(2010, 2023) if y != 2015]
    for k in ev:
        d[f"Ty{k}"] = d["T"] * (d["year"] == k).astype(int)
    terms = " + ".join(f"Ty{k}" for k in ev)
    m = smf.ols(f"y ~ {terms} + {ctrl}{fe}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    emit(f"\n[event study: {label}]")
    for k in ev:
        tag = " (pre)" if k < 2016 else ""
        emit(f"  T x {k}   {m.params[f'Ty{k}']:+.4f}  "
             f"(SE {m.bse[f'Ty{k}']:.4f}, p={m.pvalues[f'Ty{k}']:.3f}){tag}")


PARITIES = {
    0: ("FIRST-birth hazard  (childless risk set)        [layer-1 margin]",
        "age + I(age**2)"),
    1: ("SECOND-birth hazard (one-child risk set)        [layer-2 headline]",
        "age + I(age**2) + dur + I(dur**2)"),
    2: ("THIRD+-birth hazard (2+ children risk set)      [layer-2 placebo]",
        "age + I(age**2) + dur + I(dur**2)"),
}


def main():
    global LOAD_ROSTER, LOAD_LAST
    roster, last_wave = load_rosters()
    LOAD_ROSTER, LOAD_LAST = roster, (roster, last_wave)
    attrs = woman_attrs()
    FE = " + C(city) + C(year)"

    emit("ANNUAL PARITY-STRATIFIED BIRTH HAZARD (layers 1-2 annualized)")
    emit("married women 20-45, unregulated prefectures, 2009-2022, "
         "city+year FE, cluster city")

    risk = {}
    for k in (0, 1, 2):
        risk[k] = build_risk_set(attrs, k)

    # ---- Layer 1: pooled any-birth hazard (average effect) -------------------
    emit("\n" + "=" * 72)
    emit("LAYER 1 — pooled ANY-birth hazard (average effect across parities)")
    pooled = pd.concat([risk[0], risk[1], risk[2]], ignore_index=True)
    for post_yr in (2018, 2016):
        pooled["post"] = (pooled["year"] >= post_yr).astype(int)
        fit(f"y ~ T:post + age + I(age**2){FE}", pooled,
            f"any birth, Post={post_yr}", keys=["T:post"])

    # ---- Layers 1-2: parity-stratified --------------------------------------
    for k in (0, 1, 2):
        title, ctrl = PARITIES[k]
        d = risk[k]
        emit("\n" + "=" * 72)
        emit(f"PARITY {k} -> {k+1}:  {title}")
        emit(f"  rows={len(d):,}  women={d['pid'].nunique():,}  "
             f"cities={d['city'].nunique()}  treated share={d['T'].mean():.3f}  "
             f"hazard mean={d['y'].mean():.4f}")
        for post_yr in (2018, 2016):
            d["post"] = (d["year"] >= post_yr).astype(int)
            fit(f"y ~ T:post + {ctrl}{FE}", d, f"Post={post_yr}", keys=["T:post"])
        # group linear trend robustness on the Post=2018 spec
        d["post"] = (d["year"] >= 2018).astype(int)
        d["gtr"] = d["T"] * (d["year"] - 2009)
        fit(f"y ~ T:post + gtr + {ctrl}{FE}", d,
            "Post=2018 + group trend", keys=["T:post", "gtr"])
        event_study(d, f"parity {k}->{k+1}", ctrl, FE)
        emit("\nRaw hazard by year x T:")
        raw = (d.groupby(["year", "T"])["y"].agg(["mean", "count"])
               .round(4).unstack())
        emit(raw.to_string())

    out = os.path.join(OUT, "tables", "annual_parity_hazard.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
