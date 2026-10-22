# ==========================================================================
# REPLICATION PACKAGE — annual_firstborn_gender.py
# Tier: CORE
# Paper items: Figure 2 (Raw second-birth rates); Figure 3 (Event study by first-child sex); Table 3 (A. DID and triple difference, trend rows, sample rows); Table 8 (Trends and timing: exposed-group trend)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/annual_firstborn_gender.txt
# Imports local modules: none
# Run order: step 12 in run_all.sh
# ==========================================================================

"""Improved layer-3 design: annual second-birth hazard by firstborn gender,
adopting Liu & Zhang (2025) ingredients on top of our locked margin.

What we take from their setup (vs our wave-level firstborn_gender_ddd.py):
  1. ANNUAL outcome reconstructed from famconf child birth years (not the
     wave-level new_birth binary) -> real pre-period variation, yearly
     pre-trend tests instead of the pooled-2012+2014-baseline patch.
  2. Post = 2018 (first full year after the shock + pregnancy delay);
     Post = 2016 shown as variant.
  3. Treatment from BASELINE (earliest-wave) city -> immune to endogenous
     post-shock migration.
  4. Treatment-group linear trend as robustness (load-bearing in their
     Table 3; here the design has a cleaner risk-set so we report both).

Design (discrete-time second-birth hazard):
  Risk set: woman i is at risk in year t if she has EXACTLY ONE child as of
  end of t-1 (t in [first_birth_year+1, second_birth_year], capped by her
  last famconf wave). Outcome y_it = 1{second birth in t}. son_i = gender
  of the single (first) child. Women exit after the second birth.
  Sample: ages 20-45, unregulated prefectures, 2009-2022.
  FE: city + year (our locked convention), age + duration controls,
  cluster by city. DDD adds full T/son interaction set incl. T x son.
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
    """Per pid from the LATEST famconf wave: list of (child birth year, gender)."""
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        g_cols = [c.replace("tb1y", "tb2") for c in by_cols]
        g_cols = [c for c in g_cols if c in meta.column_names]
        df, _ = pyreadstat.read_dta(path, usecols=["pid"] + by_cols + g_cols)
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
        long["cgender"] = long["cgender"].replace({5: 0})  # 5 = female
        long["roster_wave"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["roster_wave"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    roster = allw[allw["roster_wave"] == allw["last_wave"]]
    return roster[["pid", "cby", "cgender"]], last.reset_index()


def build_hazard_panel(y0=2009, y1=2022):
    roster, last_wave = load_rosters()

    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "gender", "birthy", "city_code"])
    women = panel[panel["gender"] == 0].sort_values(["pid", "year"])
    base = women.dropna(subset=["city_code"]).groupby("pid").first()
    attrs = women.groupby("pid").agg(birthy=("birthy", "first"))
    attrs = attrs.join(base["city_code"]).dropna().reset_index()

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    attrs = attrs.merge(hpr[["city_code", "is_regulated", "treat_hpr"]],
                        on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]

    # first / second birth year and firstborn gender per woman
    r = roster.sort_values(["pid", "cby"])
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "first_by", "cgender": "fb_son"})
    second = r[r["k"] == 1][["pid", "cby"]].rename(columns={"cby": "second_by"})
    kids = first.merge(second, on="pid", how="left")
    # twins at first birth -> never at parity 1
    kids = kids[(kids["second_by"].isna()) | (kids["second_by"] > kids["first_by"])]

    df = attrs.merge(kids, on="pid", how="inner").merge(last_wave, on="pid")
    df = df[df["fb_son"].isin([0, 1])]

    df = df.merge(pd.DataFrame({"year": range(y0, y1 + 1)}), how="cross")
    end = df[["second_by", "last_wave"]].min(axis=1)
    df = df[(df["year"] > df["first_by"]) & (df["year"] <= end.fillna(df["last_wave"]))]
    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 20) & (df["age"] <= 45)]

    df["y"] = (df["year"] == df["second_by"]).astype(float)
    df["dur"] = df["year"] - df["first_by"]
    df["son"] = df["fb_son"].astype(int)
    df["T"] = df["treat_hpr"].astype(int)
    df["city"] = df["city_code"].astype(int)
    return df


def fit(formula, d, label, keys=None):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    keys = keys or [k for k in m.params.index if ":" in k or k.startswith("Ty")]
    emit(f"\n[{label}]  N={int(m.nobs):,}  women={d['pid'].nunique():,}  "
         f"mean(y)={d['y'].mean():.4f}")
    for k in keys:
        emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    df = build_hazard_panel()
    emit("Annual second-birth hazard panel (one-child risk set), 2009-2022")
    emit(f"  rows={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}  treated share={df['T'].mean():.3f}")
    emit(f"  hazard mean={df['y'].mean():.4f}  son share={df['son'].mean():.3f}")

    CTRL = "age + I(age**2) + dur + I(dur**2)"
    FE = " + C(city) + C(year)"

    for post_yr in (2018, 2016):
        df["post"] = (df["year"] >= post_yr).astype(int)
        emit("\n" + "=" * 72)
        emit(f"POST = {post_yr}+")
        fit(f"y ~ T:post + {CTRL}{FE}", df[df["son"] == 0], "firstborn DAUGHTER")
        fit(f"y ~ T:post + {CTRL}{FE}", df[df["son"] == 1], "firstborn SON")
        fit(f"y ~ T:post + T:post:son + post:son + T:son + son + {CTRL}{FE}",
            df, "DDD full")

    # group linear trend robustness (their load-bearing adjustment)
    df["post"] = (df["year"] >= 2018).astype(int)
    df["gtr"] = df["T"] * (df["year"] - 2009)
    emit("\n" + "=" * 72)
    emit("Robustness: + treatment-group linear trend (post=2018)")
    fit(f"y ~ T:post + gtr + {CTRL}{FE}", df[df["son"] == 0],
        "DAUGHTER + group trend", keys=["T:post", "gtr"])
    fit(f"y ~ T:post + gtr + {CTRL}{FE}", df[df["son"] == 1],
        "SON + group trend", keys=["T:post", "gtr"])

    # annual event study, baseline = 2015 (last full pre year)
    emit("\n" + "=" * 72)
    emit("Annual event study (T x year, omitted 2015)")
    ev_years = [y for y in range(2010, 2023) if y != 2015]
    for k in ev_years:
        df[f"Ty{k}"] = df["T"] * (df["year"] == k).astype(int)
    terms = " + ".join(f"Ty{k}" for k in ev_years)
    for s, lab in [(0, "firstborn DAUGHTER"), (1, "firstborn SON")]:
        d = df[df["son"] == s]
        m = smf.ols(f"y ~ {terms} + {CTRL}{FE}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        emit(f"\n[{lab}]")
        for k in ev_years:
            emit(f"  T x {k}   {m.params[f'Ty{k}']:+.4f}  "
                 f"(SE {m.bse[f'Ty{k}']:.4f}, p={m.pvalues[f'Ty{k}']:.3f})")

    # raw hazard by year for the daughter group
    emit("\nRaw second-birth hazard by year x T, daughter-firstborn risk set:")
    raw = (df[df["son"] == 0].groupby(["year", "T"])["y"]
           .agg(["mean", "count"]).round(4).unstack())
    emit(raw.to_string())

    out = os.path.join(OUT, "tables", "annual_firstborn_gender.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
