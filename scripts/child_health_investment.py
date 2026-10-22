# ==========================================================================
# REPLICATION PACKAGE — child_health_investment.py
# Tier: CORE
# Paper items: Table 5 (B. Medical, insurance, hospitalisation rows)
# Reads: CFPS .dta files in data/20xx/, output/analysis_data.csv, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/child_health_investment.txt
# Imports local modules: none
# Run order: step 22 in run_all.sh
# ==========================================================================

"""Lottery ticket: does the Q->Q substitution also show up in HEALTH investment?

The locked result: treated daughter-first one-child families forgo the second
birth AND raise education spending on the existing child. Education is the only
quality margin tested so far. This script tests HEALTH investment margins on
the same design.

Child-level outcomes (wc/wp health block, harmonized per wave):
  INVESTMENT (primary, 6 tests -- all reported, no cherry-picking):
    any_comm : any commercial medical insurance bought for the child
               (2010-14: wc802 spend>0; 2016+: wc8015 yes/no -- measurement
                seam at 2016 is arm-neutral, absorbed by year FE)
    ihs_comm : ihs commercial-insurance premium (zeros filled from skip logic)
    any_soc  : child covered by any social medical insurance
               (2010 wc801; 2012/14 wc801m; 2016 qp605_s_*; 2018+ wp605_a_*)
    ihs_tot  : ihs total child medical spending (2010/12 wc7 with -8->0;
               2014+ metotal, CFPS-constructed, zero-filled)
    ihs_oop  : ihs out-of-pocket child medical spending (wc701/pc701, -8->0)
    doctor1st: "see a doctor right away" for a minor illness (wc3 family)
  HEALTH OUTCOMES (secondary, 2 tests): sick_lm (sick last month), hosp
               (hospitalized last 12m) -- outcomes, not investment.

Design mirrors scripts/child_quality_did.py: firstborn children linked to
sample mothers (pid_m / pid_a_m; 2018+ wa001y landmine -> age-imputed birth
year fallback), baseline-city HPR treatment, regulated cities dropped,
FE city + year + child-age, cluster city, Post=2018.
Frames: one-child-as-of-end-2015 (primary, matches locked convention),
ALL firstborns (ITT robustness); benchmark: families with 2+ kids by 2015.

Part B (cheap panel): household famecon medical expenditure ("med",
CFPS-constructed 'Medical and fitness expenditure', label-consistent
2012-2022; sidesteps the 2012 fp-block collision) on the exact
eduexp_qq_substitution.py design (Post=2016, pooled 2012+2014 baseline).
Caveat stated in output: household med includes the PARENTS' own medical
spending, so the one-child interpretability argument is weaker than for edu.
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


def yn(s):
    """CFPS yes/no: 1=yes, 5 or 0=no, negatives=NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.map({1: 1.0, 5: 0.0, 0: 0.0})


def amt(s, skip_zero=True):
    """CFPS amount: -8 (not applicable / skip) -> 0, other negatives -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    out = s.where(s >= 0)
    if skip_zero:
        out = out.mask(s == -8, 0.0)
    return out


# ---------------------------------------------------------------- rosters
def load_rosters():
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
CHILD_FILES = {
    2010: ("2010/ecfps2010child_201906.dta",
           dict(pid_m="pid_m", cby="wa1y", soc="wc801", comm_exp="wc802",
                oop="wc701", tot="wc7", wc3="wc3", sickn="wc1", hosp="wc401")),
    2012: ("2012/ecfps2012child_201906.dta",
           dict(pid_m="pid_m", cby="cfps2012_birthy_best", soc="wc801m",
                comm_exp="wc802", oop="wc701", tot="wc7", wc3="wc3",
                sick="wc0", hosp="wc401")),
    2014: ("2014/ecfps2014child_201906.dta",
           dict(pid_m="pid_m", cby="cfps_birthy", soc="wc801m",
                comm_exp="wc802", oop="wc701", tot="metotal", wc3="wc3",
                sick="wc0", hosp="wc401")),
    2016: ("2016/ecfps2016child_201906.dta",
           dict(pid_m="pid_m", cby="cfps_birthy", soc1="qp605_s_1",
                soc2="qp605_s_2", comm_yes="wc8015", comm_exp="wc802",
                oop="pc701", tot="metotal", wc3="wc3", sick="wc0",
                hosp="pc401")),
    2018: ("2018/ecfps2018childproxy_202012.dta",
           dict(pid_m="pid_a_m", cby="wa001y", cage="age",
                a1="wp605_a_1", a2="wp605_a_2", a3="wp605_a_3",
                a4="wp605_a_4", a5="wp605_a_5", a78="wp605_a_78",
                comm_yes="wc8015", comm_exp="wc802", oop="wc701",
                tot="metotal", wc3="wc3_b_1", sick="wc0", hosp="wc401")),
    2020: ("2020/ecfps2020child_202306.dta",
           dict(pid_m="pid_a_m", cby="wa001y", cage="age",
                a1="wp605_a_1", a2="wp605_a_2", a3="wp605_a_3",
                a4="wp605_a_4", a5="wp605_a_5", a6="wp605_a_6",
                a78="wp605_a_78", comm_yes="wc8015", comm_exp="wc802",
                oop="wc701", tot="metotal", wc3="wc3n", sick="wc0",
                hosp="wc401")),
    2022: ("2022/ecfps2022child_202410.dta",
           dict(pid_m="pid_a_m", cby="wa001y", cage="age",
                a1="wp605_a_1", a2="wp605_a_2", a3="wp605_a_3",
                a4="wp605_a_4", a5="wp605_a_5", a6="wp605_a_6",
                a78="wp605_a_78", comm_yes="wc8015", comm_exp="wc802",
                oop="wc701", tot="metotal", wc3="wc3n", sick="wc0",
                hosp="wc401")),
}


def harmonize(d, year):
    """Build the harmonized outcome set from the raw per-wave columns in d."""
    o = pd.DataFrame(index=d.index)

    # ---- social medical insurance
    if year <= 2014:
        o["any_soc"] = yn(d["soc"])
    elif year == 2016:
        s1, s2 = pd.to_numeric(d["soc1"], errors="coerce"), pd.to_numeric(
            d["soc2"], errors="coerce")
        ins = s1.isin([1, 2, 3, 4, 5]) | s2.isin([1, 2, 3, 4, 5])
        none = s1 == 78
        o["any_soc"] = np.where(ins, 1.0, np.where(none, 0.0, np.nan))
    else:
        dums = [c for c in ("a1", "a2", "a3", "a4", "a5", "a6") if c in d]
        any1 = pd.concat([yn(d[c]) for c in dums], axis=1).max(axis=1)
        none = yn(d["a78"])
        o["any_soc"] = np.where(any1 == 1, 1.0,
                                np.where(none == 1, 0.0, np.nan))

    # ---- commercial medical insurance
    ce = pd.to_numeric(d["comm_exp"], errors="coerce")
    if year <= 2014:
        ce = ce.where(ce >= 0)                       # asked of everyone
        o["any_comm"] = (ce > 0).astype(float).where(ce.notna())
        o["ihs_comm"] = np.arcsinh(ce)
    else:
        buy = yn(d["comm_yes"])
        o["any_comm"] = buy
        spend = ce.where(ce >= 0)
        spend = spend.mask(buy == 0, 0.0)            # skip -8 -> true zero
        o["ihs_comm"] = np.arcsinh(spend)

    # ---- medical spending on the child
    o["ihs_tot"] = np.arcsinh(amt(d["tot"]))
    o["ihs_oop"] = np.arcsinh(amt(d["oop"]))

    # ---- care-seeking style for minor illness
    w = pd.to_numeric(d["wc3"], errors="coerce")
    o["doctor1st"] = np.where(w == 1, 1.0,
                              np.where(w.isin([2, 3, 4, 5, 77]), 0.0, np.nan))

    # ---- secondary health outcomes
    if year == 2010:
        n = pd.to_numeric(d["sickn"], errors="coerce")
        o["sick_lm"] = (n > 0).astype(float).where(n >= 0)
    else:
        o["sick_lm"] = yn(d["sick"])
    h = pd.to_numeric(d["hosp"], errors="coerce")
    o["hosp"] = yn(h).fillna(pd.Series(np.where(h == -8, 0.0, np.nan),
                                       index=d.index))
    return o


OUTCOMES = [
    ("any_comm", "any commercial medical insurance for child", "invest"),
    ("ihs_comm", "ihs commercial insurance premium", "invest"),
    ("any_soc", "any social medical insurance for child", "invest"),
    ("ihs_tot", "ihs total child medical spending", "invest"),
    ("ihs_oop", "ihs out-of-pocket child medical spending", "invest"),
    ("doctor1st", "see doctor right away for minor illness", "invest"),
    ("sick_lm", "child sick last month (HEALTH OUTCOME)", "health"),
    ("hosp", "child hospitalized last 12m (HEALTH OUTCOME)", "health"),
]


def load_child_waves():
    frames = []
    for year, (rel, cols) in CHILD_FILES.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        have = {k: v for k, v in cols.items() if v in meta.column_names}
        missing = set(cols) - set(have)
        df, _ = pyreadstat.read_dta(path,
                                    usecols=list(set(["pid"] + list(have.values()))))
        d = pd.DataFrame({"pid_c": df["pid"]})
        for k, v in have.items():
            d[k] = df[v]
        if "cby" in d.columns:
            d["cby"] = pd.to_numeric(d["cby"], errors="coerce").where(
                lambda s: s > 1900)
        if "cage" in d.columns:
            d["cage"] = pd.to_numeric(d["cage"], errors="coerce").where(
                lambda s: s >= 0)
        out = harmonize(d, year)
        keep = pd.concat([d[[c for c in ("pid_c", "pid_m", "cby", "cage")
                             if c in d.columns]], out], axis=1)
        keep["year"] = year
        frames.append(keep)
        stats = {k: f"{out[k].notna().sum()}" for k, _, _ in OUTCOMES}
        emit(f"  {year}: rows={len(d):>5}  nonmiss " +
             " ".join(f"{k}={v}" for k, v in stats.items()) +
             (f"  MISSING_RAW={sorted(missing)}" if missing else ""))
    return pd.concat(frames, ignore_index=True)


def build_panel():
    emit("Child-file load & harmonization (nonmissing counts per outcome):")
    ch = load_child_waves()
    mothers = build_mothers()

    d = ch.merge(mothers, on="pid_m", how="inner")
    direct = d["cby"].notna()
    d["cby_est"] = d["cby"].where(direct, d["year"] - d.get("cage"))
    gap = np.abs(d["cby_est"] - d["first_by"])
    d = d[(direct & (gap < 0.5)) | (~direct & (gap <= 1.0))]
    d["cby"] = d["cby_est"]
    d["age_c"] = d["year"] - d["cby"]
    d = d[(d["age_c"] >= 0) & (d["age_c"] <= 16)]
    d["one_child_2015"] = d["second_by"].isna() | (d["second_by"] > 2015)
    d["multi_2015"] = d["second_by"].notna() & (d["second_by"] <= 2015)
    d["T"] = d["treat_hpr"].astype(int)
    d["post"] = (d["year"] >= 2018).astype(int)
    d["son"] = d["fb_son"].astype(int)
    d["city"] = d["city_code"].astype(int)
    return d


def fit(formula, d, label, keys, min_n=50):
    yv = formula.split("~")[0].strip()
    d = d.dropna(subset=[yv])
    if len(d) < min_n or d["city"].nunique() < 10 or d[yv].std() == 0:
        emit(f"  [{label}]  SKIPPED (N={len(d)})")
        return None
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    parts = []
    for k in keys:
        if k in m.params.index:
            parts.append(f"{k} {m.params[k]:+.4f} (SE {m.bse[k]:.4f}, "
                         f"p={m.pvalues[k]:.3f})")
    emit(f"  [{label}]  N={int(m.nobs):,}  " + "; ".join(parts))
    return m


def evstudy(frame, yv, label, base=2014):
    dd = frame.dropna(subset=[yv]).copy()
    yrs = sorted(y for y in dd["year"].unique() if y != base)
    if not yrs or dd["city"].nunique() < 10:
        return
    for k in yrs:
        dd[f"Tw{k}"] = (dd["T"] * (dd["year"] == k)).astype(float)
    terms = " + ".join(f"Tw{k}" for k in yrs)
    m = smf.ols(f"{yv} ~ {terms} + C(age_c) + C(city) + C(year)", data=dd
                ).fit(cov_type="cluster", cov_kwds={"groups": dd["city"]})
    emit(f"\n  event study [{label}]  N={int(m.nobs):,}  (base {base})")
    for k in yrs:
        emit(f"    T x {k}   {m.params[f'Tw{k}']:+.4f}  "
             f"(SE {m.bse[f'Tw{k}']:.4f}, p={m.pvalues[f'Tw{k}']:.3f})")


def main():
    d = build_panel()
    emit(f"\nLinked firstborn panel: rows={len(d):,} kids={d['pid_c'].nunique():,} "
         f"mothers={d['pid_m'].nunique():,} cities={d['city'].nunique()}")
    emit(f"  daughter share={1 - d['son'].mean():.3f}  "
         f"one-child-as-of-2015 share={d['one_child_2015'].mean():.3f}")

    emit("\nBaseline means (one-child-2015 frame, pre-2018):")
    pre = d[d["one_child_2015"] & (d["year"] < 2018)]
    for yv, yl, _ in OUTCOMES:
        g = pre.groupby("T")[yv].mean()
        emit(f"  {yv:<10} C={g.get(0, np.nan):.3f}  T={g.get(1, np.nan):.3f}")

    CTRL = " + C(age_c) + C(city) + C(year)"
    emit("\nMULTIPLE-TESTING NOTE: 6 investment outcomes + 2 health outcomes, "
         "each in 2 frames x 3 specs. All coefficients reported; a lone "
         "p<0.05 with no pattern across margins/frames is noise.")

    frames = [("one-child-as-of-end-2015 (primary)", d[d["one_child_2015"]]),
              ("ALL firstborns (ITT robustness)", d)]
    for fname, frame in frames:
        emit("\n" + "=" * 76)
        emit(f"FRAME: {fname}   (Post=2018, FE city+year+child-age, cl. city)")
        for yv, yl, tier in OUTCOMES:
            emit(f"\n--- {yv}: {yl} ---")
            fit(f"{yv} ~ T:post{CTRL}", frame[frame["son"] == 0],
                "daughter", ["T:post"])
            fit(f"{yv} ~ T:post{CTRL}", frame[frame["son"] == 1],
                "son", ["T:post"])
            fit(f"{yv} ~ T:post + T:post:son + post:son + T:son + son{CTRL}",
                frame, "DDD", ["T:post", "T:post:son"])

    # benchmark: no quantity margin
    emit("\n" + "=" * 76)
    emit("BENCHMARK: firstborns in 2+ -children-by-2015 families (no quantity "
         "margin), pooled DID")
    bm = d[d["multi_2015"]]
    for yv, yl, tier in OUTCOMES:
        fit(f"{yv} ~ T:post{CTRL}", bm, f"{yv} | 2+ kids", ["T:post"])

    # cell sizes, daughter one-child frame
    emit("\nCell sizes (one-child-2015 daughters, ihs_tot nonmissing):")
    v = d[d["one_child_2015"] & (d["son"] == 0)].dropna(subset=["ihs_tot"])
    emit(v.groupby(["year", "T"])["pid_c"].count().unstack(fill_value=0)
         .to_string())

    # event studies on the investment outcomes, daughter one-child frame
    emit("\n" + "=" * 76)
    emit("EVENT STUDIES (one-child-2015 frame, daughter arm, base 2014):")
    for yv in ("any_comm", "ihs_comm", "any_soc", "ihs_tot", "ihs_oop",
               "doctor1st"):
        evstudy(d[d["one_child_2015"] & (d["son"] == 0)], yv,
                f"{yv} | daughter")

    # -------------------------------------------------- robustness on the hit
    emit("\n" + "=" * 76)
    emit("ROBUSTNESS on the medical-spending result (one-child-2015 frame):")
    oc = d[d["one_child_2015"]]
    dg, sn = oc[oc["son"] == 0], oc[oc["son"] == 1]
    emit("\n(a) exclude hospitalized children (is it routine/outpatient care?)")
    fit(f"ihs_oop ~ T:post{CTRL}", dg[dg["hosp"] == 0],
        "daughter ihs_oop | hosp==0", ["T:post"])
    fit(f"ihs_tot ~ T:post{CTRL}", dg[dg["hosp"] == 0],
        "daughter ihs_tot | hosp==0", ["T:post"])
    fit(f"ihs_oop ~ T:post{CTRL}", sn[sn["hosp"] == 0],
        "son ihs_oop | hosp==0", ["T:post"])
    emit("\n(b) control illness reporting (mediator check -- does spend rise")
    emit("    beyond what higher reported sickness explains?)")
    fit(f"ihs_oop ~ T:post + sick_lm{CTRL}", dg.dropna(subset=["sick_lm"]),
        "daughter ihs_oop + sick_lm ctrl", ["T:post", "sick_lm"])
    fit(f"ihs_tot ~ T:post + sick_lm{CTRL}", dg.dropna(subset=["sick_lm"]),
        "daughter ihs_tot + sick_lm ctrl", ["T:post", "sick_lm"])
    emit("\n(c) Post=2016 timing variant (headline uses Post=2018)")
    oc16 = oc.copy()
    oc16["post"] = (oc16["year"] >= 2016).astype(int)
    fit(f"ihs_oop ~ T:post{CTRL}", oc16[oc16["son"] == 0],
        "daughter ihs_oop | Post=2016", ["T:post"])
    fit(f"ihs_oop ~ T:post{CTRL}", oc16[oc16["son"] == 1],
        "son ihs_oop | Post=2016", ["T:post"])
    emit("\n(d) doctor1st caveat: baseline T-C gap is large (0.68 vs 0.52);")
    emit("    son-arm baseline for comparison:")
    for arm, dd_ in (("daughter", dg), ("son", sn)):
        g = dd_[dd_["year"] < 2018].groupby("T")["doctor1st"].mean()
        emit(f"    {arm}: pre C={g.get(0, np.nan):.3f} T={g.get(1, np.nan):.3f}")

    # ---------------------------------------------------------------- Part B
    emit("\n" + "=" * 76)
    emit("PART B: HOUSEHOLD medical+fitness expenditure ('med', CFPS-constructed,")
    emit("label-consistent 2012-2022; avoids the 2012 fp-block collision).")
    emit("Design = eduexp_qq_substitution.py (Post=2016, one-child-at-baseline")
    emit("married women, city+year FE, controls age/log_income/urban).")
    emit("CAVEAT: household med includes the PARENTS' own medical spending --")
    emit("weaker one-child interpretability than edu_exp.")

    ECON = {
        2012: ("data/2012/ecfps2012famecon_201906.dta", "fid12"),
        2014: ("data/2014/ecfps2014famecon_201906.dta", "fid14"),
        2016: ("data/2016/ecfps2016famecon_201807.dta", "fid16"),
        2018: ("data/2018/ecfps2018famecon_202101.dta", "fid18"),
        2020: ("data/2020/ecfps2020famecon_202306.dta", "fid20"),
        2022: ("data/2022/ecfps2022famecon_202410.dta", "fid22"),
    }
    rows = []
    for yr, (path, fid) in ECON.items():
        dd, _ = pyreadstat.read_dta(os.path.join(ROOT, path),
                                    usecols=[fid, "med"])
        dd.columns = ["fid", "med_exp"]
        dd["med_exp"] = pd.to_numeric(dd["med_exp"], errors="coerce")
        dd.loc[dd["med_exp"] < 0, "med_exp"] = np.nan
        dd["year"] = yr
        rows.append(dd)
        emit(f"  {yr}: N={dd['med_exp'].notna().sum():,}  "
             f"median={dd['med_exp'].median():.0f}  "
             f">0 share={(dd['med_exp'] > 0).mean():.2f}")
    med = pd.concat(rows)

    def child_info(year, path):
        by_cols = [f"tb1y_a_c{i}" for i in range(1, 11)]
        g_cols = [f"tb2_a_c{i}" for i in range(1, 11)]
        df, _ = pyreadstat.read_dta(os.path.join(DATA, path),
                                    usecols=["pid"] + by_cols + g_cols)
        df.columns = df.columns.str.lower()
        by = df[by_cols].where((df[by_cols] > 1900) & (df[by_cols] <= year))
        g = (df[g_cols].where(df[g_cols].isin([0, 1, 5])).replace({5: 0})
             .where(by.notna().values))
        return pd.DataFrame({
            "pid": df["pid"],
            "n_kids_base": by.notna().sum(axis=1),
            "firstborn_son": g.where(by.eq(by.min(axis=1), axis=0).values
                                     ).max(axis=1)})

    b14 = child_info(2014, FAMCONF[2014])
    b12 = child_info(2012, FAMCONF[2012])
    base = pd.concat([b14, b12[~b12["pid"].isin(b14["pid"])]])

    df = pd.read_csv(os.path.join(OUT, "analysis_data.csv"))
    df = df[df["year"] != 2010].copy()
    df["pid"] = df["pid"].astype(int)
    base["pid"] = base["pid"].astype(int)
    df = df.merge(base, on="pid", how="left")
    df = df[(df["is_regulated"] == 0) & df["treat_hpr"].notna()
            & df["city_code"].notna()]
    df = df.dropna(subset=["n_kids_base", "age", "log_income", "urban"])
    df = df.merge(med, on=["fid", "year"], how="left")
    df["post"] = (df["year"] >= 2016).astype(int)
    df["T"] = df["treat_hpr"].astype(int)
    df["city"] = df["city_code"].astype(int)
    df["ihs_med"] = np.arcsinh(df["med_exp"])
    df["any_med"] = (df["med_exp"] > 0).astype(float).where(
        df["med_exp"].notna())

    one = df[(df["n_kids_base"] == 1) & df["firstborn_son"].isin([0, 1])].copy()
    one = one.dropna(subset=["med_exp"])
    one["son"] = one["firstborn_son"].astype(int)
    emit(f"\nOne-child sample with med_exp: {len(one):,} obs, "
         f"{one['pid'].nunique():,} women; med>0 share={one['any_med'].mean():.2f}")

    HCTRL = "age + log_income + urban"
    HFE = " + C(city) + C(year)"
    for yv in ("ihs_med", "any_med"):
        emit(f"\n--- household {yv} ---")
        fit(f"{yv} ~ T:post + {HCTRL}{HFE}", one[one["son"] == 0],
            "daughter", ["T:post"])
        fit(f"{yv} ~ T:post + {HCTRL}{HFE}", one[one["son"] == 1],
            "son", ["T:post"])
        fit(f"{yv} ~ T:post + T:post:son + post:son + T:son + son + "
            f"{HCTRL}{HFE}", one, "DDD", ["T:post", "T:post:son"])

    dgt = one[one["son"] == 0].copy()
    for k in (2016, 2018, 2020, 2022):
        dgt[f"Ty{k}"] = dgt["T"] * (dgt["year"] == k).astype(int)
    fit(f"ihs_med ~ Ty2016 + Ty2018 + Ty2020 + Ty2022 + {HCTRL}{HFE}", dgt,
        "daughter ES ihs_med (base 2012+2014)",
        ["Ty2016", "Ty2018", "Ty2020", "Ty2022"])
    multi = df[df["n_kids_base"] >= 2].dropna(subset=["med_exp"]).copy()
    fit(f"ihs_med ~ T:post + {HCTRL}{HFE}", multi, "2+ kids benchmark",
        ["T:post"])

    out = os.path.join(OUT, "tables", "child_health_investment.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
