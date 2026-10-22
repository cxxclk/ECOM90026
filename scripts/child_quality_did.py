# ==========================================================================
# REPLICATION PACKAGE — child_quality_did.py
# Tier: CORE
# Paper items: Table 5 (B. Child-level education row)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/child_quality_did.txt
# Imports local modules: none
# Run order: step 21 in run_all.sh
# ==========================================================================

"""Direction #2: upgrade Layer 4 from education EXPENDITURE to child QUALITY outcomes.

Question: did the firstborn daughters who "kept" their only-child status after the
HPR price shock actually end up BETTER OFF in measured quality?

Child-level outcomes (from CFPS child / childproxy / person files, label-aligned):
  (a) cognitive z  : word + math test scores, waves 2010/2014 (pre) & 2018/2022 (post)
  (b) exp. years   : parent's expected final education for the child (wd2),
                     mapped to years VIA VALUE LABELS per wave (codes shift across
                     waves: 2014 6=Bachelor vs 2022 7=Bachelor -- known-style collision)
  (c) grades       : parent/self-rated Chinese+math performance last semester
                     (wf501/wf502, 1=excellent..4=poor, flipped, z within wave)
  (d) edu spending : child-level wd5total (ihs) -- child-level check of Layer 4

Design: firstborn children linked to sample mothers (reverse link via pid_m /
pid_a_m; firstborn = child birth year == mother's roster first birth year).
Mother restrictions follow the main line: baseline-city HPR treatment,
unregulated prefectures only (regulated source cities dropped).
FE: city + year + child-age; cluster by city; Post = 2018.
Frames: ALL firstborns (ITT, selection-free) and one-child-at-wave (matches the
locked Layer 4 convention but conditions on post-treatment parity -- caveat).
Split: firstborn daughter vs firstborn son + DDD.
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


# ---------------------------------------------------------------- rosters
def load_rosters():
    """Per pid from the LATEST famconf wave: (child birth year, gender)."""
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        df, _ = pyreadstat.read_dta(
            path, usecols=["pid"] + by_cols +
            [c.replace("tb1y", "tb2") for c in by_cols
             if c.replace("tb1y", "tb2") in meta.column_names])
        recs = []
        for i, byc in enumerate(by_cols, start=1):
            gc = f"tb2_a_c{i}"
            sub = df[["pid", byc] + ([gc] if gc in df.columns else [])].copy()
            sub.columns = ["pid", "cby"] + (["cgender"] if gc in df.columns else [])
            if "cgender" not in sub.columns:
                sub["cgender"] = np.nan
            recs.append(sub)
        long = pd.concat(recs, ignore_index=True)
        long = long[(long["cby"] > 1900) & (long["cby"] <= year)]
        long["cgender"] = long["cgender"].where(long["cgender"].isin([0, 1, 5]))
        long["cgender"] = long["cgender"].replace({5: 0})
        long["roster_wave"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["roster_wave"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    roster = allw[allw["roster_wave"] == allw["last_wave"]]
    return roster[["pid", "cby", "cgender"]]


def build_mothers():
    panel = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "gender", "birthy", "city_code"])
    women = panel[panel["gender"] == 0].sort_values(["pid", "year"])
    base = women.dropna(subset=["city_code"]).groupby("pid").first()
    attrs = women.groupby("pid").agg(m_birthy=("birthy", "first"))
    attrs = attrs.join(base["city_code"]).dropna().reset_index()

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    attrs = attrs.merge(hpr[["city_code", "is_regulated", "treat_hpr"]],
                        on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]

    roster = load_rosters().sort_values(["pid", "cby"])
    roster["k"] = roster.groupby("pid").cumcount()
    first = roster[roster["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "first_by", "cgender": "fb_son"})
    second = roster[roster["k"] == 1][["pid", "cby"]].rename(
        columns={"cby": "second_by"})
    kids = first.merge(second, on="pid", how="left")
    kids = kids[(kids["second_by"].isna()) | (kids["second_by"] > kids["first_by"])]

    m = attrs.merge(kids, on="pid", how="inner")
    m = m[m["fb_son"].isin([0, 1])]
    return m.rename(columns={"pid": "pid_m"})


# ---------------------------------------------------------------- child files
WD2_MAP = [
    (r"illiter|no need|not necessary", 0.0),
    (r"primary", 6.0),
    (r"junior", 9.0),
    (r"senior", 12.0),
    (r"2- or 3-year|3-year college|2- or 3 year", 15.0),
    (r"4-year|bachelor", 16.0),
    (r"master", 19.0),
    (r"doctor", 22.0),
]

# 2018 childproxy wd2 carries NO value labels; its distribution (mode 7,
# then 5/9/8) matches the labelled 2020/2022 scheme (3=Primary .. 7=Bachelor,
# 8=Master, 9=Doctor) -> numeric fallback in that coding
WD2_FALLBACK = {1: 0.0, 2: 0.0, 3: 6.0, 4: 9.0, 5: 12.0, 6: 15.0,
                7: 16.0, 8: 19.0, 9: 22.0}

def map_wd2(series, value_labels):
    out = pd.Series(np.nan, index=series.index)
    if not value_labels:
        return series.map(WD2_FALLBACK)
    for code, lab in value_labels.items():
        for pat, yrs in WD2_MAP:
            if re.search(pat, str(lab), re.I):
                out[series == code] = yrs
                break
    return out


def clean_neg(s):
    return s.where(s >= 0)


CHILD_FILES = {
    2010: [("child", "2010/ecfps2010child_201906.dta",
            dict(pid_m="pid_m", cby="wa1y", word="wordtest", math="mathtest",
                 wd2="wd2", chn="wf501", mth="wf502", exp="wd5total"))],
    2012: [("child", "2012/ecfps2012child_201906.dta",
            dict(pid_m="pid_m", cby="cfps2012_birthy_best",
                 wd2="wd2", chn="wf501", mth="wf502", exp="wd5total"))],
    2014: [("child", "2014/ecfps2014child_201906.dta",
            dict(pid_m="pid_m", cby="cfps_birthy", word="wordtest14_sc2",
                 math="mathtest14_sc2", wd2="wd2", chn="wf501", mth="wf502",
                 exp="wd5total"))],
    2016: [("child", "2016/ecfps2016child_201906.dta",
            dict(pid_m="pid_m", cby="cfps_birthy", wd2="wd2",
                 chn="wf501", mth="wf502", exp="pd5total"))],
    # post-2016 files: wa001y only filled for NEWLY ADDED kids (17 valid in
    # 2018 childproxy!) -> carry the age variable as birth-year fallback
    2018: [("childproxy", "2018/ecfps2018childproxy_202012.dta",
            dict(pid_m="pid_a_m", cby="wa001y", cage="age", wd2="wd2",
                 chn="wf501", mth="wf502", exp="wd5total")),
           ("person", "2018/ecfps2018person_202012.dta",
            dict(pid_m="pid_a_m", cby="ibirthy", cage="age",
                 word="wordtest18_sc2", math="mathtest18_sc2"))],
    2020: [("child", "2020/ecfps2020child_202306.dta",
            dict(pid_m="pid_a_m", cby="wa001y", cage="age", wd2="wd2",
                 chn="wf501", mth="wf502", exp="wd5total"))],
    2022: [("child", "2022/ecfps2022child_202410.dta",
            dict(pid_m="pid_a_m", cby="wa001y", cage="age", wd2="wd2",
                 chn="wf501", mth="wf502", exp="wd5total")),
           ("person", "2022/ecfps2022person_202410.dta",
            dict(pid_m="pid_a_m", cby="ibirthy", cage="age",
                 word="wordtest22_sc2", math="mathtest22_sc2"))],
}


def load_child_waves():
    frames = []
    for year, files in CHILD_FILES.items():
        for tag, rel, cols in files:
            path = os.path.join(DATA, rel)
            _, meta = pyreadstat.read_dta(path, metadataonly=True)
            have = {k: v for k, v in cols.items() if v in meta.column_names}
            missing = set(cols) - set(have)
            usecols = ["pid"] + list(have.values())
            df, meta2 = pyreadstat.read_dta(path, usecols=list(set(usecols)))
            d = pd.DataFrame({"pid_c": df["pid"]})
            for k, v in have.items():
                d[k] = df[v]
            if "cby" in d.columns:
                d["cby"] = d["cby"].where(d["cby"] > 1900)
            if "wd2" in d.columns:
                d["exp_yrs"] = map_wd2(d["wd2"],
                                       meta2.variable_value_labels.get(have["wd2"]))
            for c in ("word", "math", "exp", "cage"):
                if c in d.columns:
                    d[c] = clean_neg(d[c])
            for c in ("chn", "mth"):
                if c in d.columns:
                    d[c] = 5.0 - d[c].where(d[c].between(1, 4))  # higher=better
            d["year"] = year
            d["src"] = tag
            frames.append(d)
            emit(f"  loaded {year} {tag:<10} rows={len(d):>6}  vars={sorted(have)}"
                 + (f"  MISSING={sorted(missing)}" if missing else ""))
    return pd.concat(frames, ignore_index=True)


def build_panel():
    emit("Child-file reconnaissance / load:")
    ch = load_child_waves()
    mothers = build_mothers()

    d = ch.merge(mothers, on="pid_m", how="inner")
    # firstborn = child birth year matches mother's roster first-birth year;
    # where cby is missing (post-2016 preload gap) use year - age, with a
    # +/-1yr tolerance for the age-derived version (interview-date slippage)
    direct = d["cby"].notna()
    d["cby_est"] = d["cby"].where(direct, d["year"] - d.get("cage"))
    gap = np.abs(d["cby_est"] - d["first_by"])
    d = d[(direct & (gap < 0.5)) | (~direct & (gap <= 1.0))]
    d["cby"] = d["cby_est"]
    d["age_c"] = d["year"] - d["cby"]
    d = d[(d["age_c"] >= 0) & (d["age_c"] <= 16)]
    d["one_child"] = d["second_by"].isna() | (d["second_by"] > d["year"])
    d["T"] = d["treat_hpr"].astype(int)
    d["post"] = (d["year"] >= 2018).astype(int)
    d["son"] = d["fb_son"].astype(int)
    d["city"] = d["city_code"].astype(int)

    # collapse person+childproxy duplicates within 2018/2022 (cog from person,
    # rest from child file): keep one row per child-year with combined columns
    agg = {c: "max" for c in ("word", "math", "exp_yrs", "chn", "mth", "exp")
           if c in d.columns}
    keys = ["pid_c", "year", "pid_m", "T", "post", "son", "city", "age_c",
            "one_child", "first_by"]
    d = d.groupby(keys, as_index=False).agg(agg)

    # outcomes
    for c in ("word", "math"):
        d[f"{c}_z"] = d.groupby("year")[c].transform(
            lambda s: (s - s.mean()) / s.std())
    d["cog_z"] = d[["word_z", "math_z"]].mean(axis=1)
    d["grade_raw"] = d[["chn", "mth"]].mean(axis=1)
    d["grade_z"] = d.groupby("year")["grade_raw"].transform(
        lambda s: (s - s.mean()) / s.std())
    d["ihs_exp"] = np.arcsinh(d["exp"])
    return d


def fit(formula, d, label, keys):
    d = d.dropna(subset=[formula.split("~")[0].strip()])
    if len(d) < 50 or d["city"].nunique() < 10:
        emit(f"\n[{label}]  SKIPPED (N={len(d)})")
        return
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  kids={d['pid_c'].nunique():,}  "
         f"cities={d['city'].nunique()}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<16} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")


def cellsizes(d, yvar, label):
    v = d.dropna(subset=[yvar])
    tab = (v.groupby(["year", "T"])["pid_c"].count().unstack(fill_value=0))
    emit(f"\nCell sizes ({label}, firstborn-daughter frame):")
    emit(tab.to_string())


def main():
    d = build_panel()
    emit(f"\nLinked firstborn panel: rows={len(d):,} kids={d['pid_c'].nunique():,} "
         f"mothers={d['pid_m'].nunique():,} cities={d['city'].nunique()}")
    emit(f"  daughter share={1 - d['son'].mean():.3f}  "
         f"one-child-at-wave share={d['one_child'].mean():.3f}")

    CTRL = " + C(age_c) + C(city) + C(year)"
    OUTCOMES = [("cog_z", "cognitive z (word+math)"),
                ("exp_yrs", "expected years of education (wd2)"),
                ("grade_z", "parent-rated grades z (chn+math)"),
                ("ihs_exp", "ihs child edu expenditure")]

    for frame, fname in [(d, "ALL firstborns (ITT)"),
                         (d[d["one_child"]], "one-child-at-wave")]:
        emit("\n" + "=" * 72)
        emit(f"FRAME: {fname}   (Post=2018, FE city+year+child-age, cl. city)")
        for yv, yl in OUTCOMES:
            emit(f"\n--- outcome: {yl} ---")
            cellsizes(frame[frame["son"] == 0], yv, yl) if yv == "cog_z" else None
            fit(f"{yv} ~ T:post{CTRL}", frame[frame["son"] == 0],
                f"{yv} | firstborn DAUGHTER", ["T:post"])
            fit(f"{yv} ~ T:post{CTRL}", frame[frame["son"] == 1],
                f"{yv} | firstborn SON", ["T:post"])
            fit(f"{yv} ~ T:post + T:post:son + post:son + T:son + son{CTRL}",
                frame, f"{yv} | DDD", ["T:post", "T:post:son"])

    # wave-by-wave event studies (base 2014) for the outcomes that moved
    emit("\n" + "=" * 72)
    emit("Event studies (T x wave, base 2014):")

    def evstudy(frame, yv, label):
        dd = frame.dropna(subset=[yv]).copy()
        yrs = sorted(y for y in dd["year"].unique() if y != 2014)
        for k in yrs:
            dd[f"Tw{k}"] = (dd["T"] * (dd["year"] == k)).astype(float)
        terms = " + ".join(f"Tw{k}" for k in yrs)
        m = smf.ols(f"{yv} ~ {terms} + C(age_c) + C(city) + C(year)", data=dd
                    ).fit(cov_type="cluster", cov_kwds={"groups": dd["city"]})
        emit(f"\n[{label}]  N={int(m.nobs):,}")
        for k in yrs:
            emit(f"  T x {k}   {m.params[f'Tw{k}']:+.4f}  "
                 f"(SE {m.bse[f'Tw{k}']:.4f}, p={m.pvalues[f'Tw{k}']:.3f})")

    evstudy(d[d["son"] == 0], "grade_z", "grade_z | daughter, ITT")
    evstudy(d[d["son"] == 1], "grade_z", "grade_z | son, ITT")
    evstudy(d[d["son"] == 1], "exp_yrs", "exp_yrs | son, ITT")
    evstudy(d[(d["son"] == 0) & d["one_child"]], "ihs_exp",
            "ihs_exp | daughter, one-child (Layer-4 child-level)")

    # wave-by-wave T coefficients for cognition (only 4 waves), daughter arm
    emit("\n" + "=" * 72)
    emit("Cognition by wave (T x wave, base 2014), firstborn daughters, ITT frame:")
    dd = d[(d["son"] == 0)].dropna(subset=["cog_z"])
    for k in (2010, 2018, 2022):
        dd[f"Tw{k}"] = (dd["T"] * (dd["year"] == k)).astype(float)
    m = smf.ols("cog_z ~ Tw2010 + Tw2018 + Tw2022 + C(age_c) + C(city) + C(year)",
                data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["city"]})
    for k in (2010, 2018, 2022):
        emit(f"  T x {k}   {m.params[f'Tw{k}']:+.4f}  "
             f"(SE {m.bse[f'Tw{k}']:.4f}, p={m.pvalues[f'Tw{k}']:.3f})")

    out = os.path.join(OUT, "tables", "child_quality_did.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
