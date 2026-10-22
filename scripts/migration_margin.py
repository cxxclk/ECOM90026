# ==========================================================================
# REPLICATION PACKAGE — migration_margin.py
# Tier: CORE
# Paper items: Table 8 (Attrition: present next wave)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/migration_margin.txt
# Imports local modules: none
# Run order: step 36 in run_all.sh
# ==========================================================================

"""Direction #3 — migration/exit margin.

Does the HPR spillover shock push families OUT of the treated corridor
("voting with feet" as an alternative margin to the fertility response)?

Design: moved_it = 1{current city_code != baseline city_code}, both non-missing.
Baseline city = earliest wave with non-missing city_code. Sample: married women
20-45, unregulated baseline prefectures, treatment = baseline-city treat_hpr.
FE: BASELINE city + year (current-city FE would be endogenous to the outcome),
cluster by baseline city. Wave-level convention: Post=2016 headline + 2018 variant.

Caveats stated up front:
  - city_code exists only for the 162 crosswalk-matched sampling counties; a
    move OUT of the frame -> city_code NaN -> excluded from `moved`. Measured
    mobility is a lower bound and conditions on staying in the matched frame.
  - Pre-check showed mover rate ~1-1.5% -> power-limited; results are
    descriptive. The load-bearing deliverable is the ATTRITION check
    (T x post on next-wave presence), which reviewers will ask for.
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
    """Per pid from the LATEST famconf wave: (child birth year, gender)."""
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
        long["cgender"] = long["cgender"].replace({5: 0})
        long["roster_wave"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["roster_wave"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    roster = allw[allw["roster_wave"] == allw["last_wave"]]
    return roster[["pid", "cby", "cgender"]]


def fit(formula, d, label, cluster, keys=None):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d[cluster]})
    keys = keys or [k for k in m.params.index if ":" in k]
    emit(f"\n[{label}]  N={int(m.nobs):,}  mean(y)={d[m.model.endog_names].mean():.4f}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<22} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "city_code", "gender", "birthy", "marital"])

    # baseline city + treatment
    cc = panel.dropna(subset=["city_code"]).sort_values(["pid", "year"])
    base = cc.groupby("pid").first()[["city_code", "year"]]
    base.columns = ["base_city", "base_year"]
    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    base = base.reset_index().merge(
        hpr[["city_code", "is_regulated", "treat_hpr"]],
        left_on="base_city", right_on="city_code", how="inner")
    base = base[base["is_regulated"] == 0].drop(columns=["city_code"])

    df = panel.merge(base, on="pid")
    df["age"] = df["year"] - df["birthy"]
    df = df[(df["gender"] == 0) & (df["marital"] == 2)
            & (df["age"] >= 20) & (df["age"] <= 45)]
    df["T"] = df["treat_hpr"].astype(int)
    df["bcity"] = df["base_city"].astype(int)

    emit("Direction #3 — migration/exit margin (wave-level)")
    emit("=" * 72)
    emit("Sample: married women 20-45, unregulated baseline prefecture")
    emit(f"  obs={len(df):,}  women={df['pid'].nunique():,}  "
         f"baseline cities={df['bcity'].nunique()}  treated share={df['T'].mean():.3f}")

    # ---- moved indicator (requires current city observed) ----
    mv = df.dropna(subset=["city_code"]).copy()
    mv = mv[mv["year"] > mv["base_year"]]
    mv["moved"] = (mv["city_code"] != mv["base_city"]).astype(float)

    emit("\nMover rate (current city != baseline city | both observed), by year x T:")
    tab = mv.groupby(["year", "T"])["moved"].agg(["mean", "count"]).round(4).unstack()
    emit(tab.to_string())
    emit(f"\nOverall mover rate: {mv['moved'].mean():.4f}  "
         f"(pre-registered gate: <2% -> POWER-LIMITED, results descriptive)")
    emit("NOTE: moves out of the 162-county sampling frame have city_code=NaN and")
    emit("drop out of this measure -> lower bound, conditioned on staying in-frame.")

    # destination split
    mv = mv.merge(hpr[["city_code", "is_regulated"]].rename(
        columns={"city_code": "cur_city", "is_regulated": "cur_reg"}),
        left_on="city_code", right_on="cur_city", how="left")
    mv["moved_reg"] = ((mv["moved"] == 1) & (mv["cur_reg"] == 1)).astype(float)
    mv["moved_unreg"] = ((mv["moved"] == 1) & (mv["cur_reg"] == 0)).astype(float)
    n_mv = int(mv["moved"].sum())
    emit(f"\nDestination of the {n_mv} mover-obs: to REGULATED city "
         f"{int(mv['moved_reg'].sum())}, to other unregulated {int(mv['moved_unreg'].sum())}, "
         f"unmatched dest {n_mv - int(mv['moved_reg'].sum()) - int(mv['moved_unreg'].sum())}")

    FE = " + C(bcity) + C(year)"
    for post_yr in (2016, 2018):
        mv["post"] = (mv["year"] >= post_yr).astype(int)
        emit("\n" + "-" * 72)
        emit(f"DID on moved, Post={post_yr}+  (baseline-city + year FE, cluster baseline city)")
        fit(f"moved ~ T:post + age + I(age**2){FE}", mv, "any move", "bcity", ["T:post"])
        fit(f"moved_reg ~ T:post + age + I(age**2){FE}", mv, "move to regulated", "bcity", ["T:post"])
        fit(f"moved_unreg ~ T:post + age + I(age**2){FE}", mv, "move to other unregulated", "bcity", ["T:post"])

    # ---- daughter-first one-child interaction (parity as of 2015) ----
    roster = load_rosters()
    r = roster.sort_values(["pid", "cby"])
    r15 = r[r["cby"] <= 2015]
    r15["k"] = r15.groupby("pid").cumcount()
    n15 = r15.groupby("pid").size().rename("nkids15")
    fb = r15[r15["k"] == 0].set_index("pid")["cgender"].rename("fb_son")
    mv = mv.merge(n15, on="pid", how="left").merge(fb, on="pid", how="left")
    one = mv[(mv["nkids15"] == 1) & (mv["fb_son"].isin([0, 1]))].copy()
    one["dghtr"] = 1 - one["fb_son"]
    one["post"] = (one["year"] >= 2016).astype(int)
    emit("\n" + "-" * 72)
    emit("One-child-as-of-2015 women: does the locked margin migrate instead? (Post=2016)")
    emit(f"  N={len(one):,}  daughter share={one['dghtr'].mean():.3f}  "
         f"mover rate={one['moved'].mean():.4f}")
    fit(f"moved ~ T:post + T:post:dghtr + post:dghtr + T:dghtr + dghtr + age + I(age**2){FE}",
        one, "moved, DDD by firstborn gender", "bcity",
        ["T:post", "T:post:dghtr"])
    fit(f"moved ~ T:post + age + I(age**2){FE}", one[one["dghtr"] == 1],
        "moved | daughter-first", "bcity", ["T:post"])
    fit(f"moved ~ T:post + age + I(age**2){FE}", one[one["dghtr"] == 0],
        "moved | son-first", "bcity", ["T:post"])

    # ---- attrition check: next-wave presence ----
    emit("\n" + "=" * 72)
    emit("ATTRITION CHECK: T x post on next-wave presence (selective attrition?)")
    waves = sorted(panel["year"].unique())
    nxt = {w: waves[i + 1] for i, w in enumerate(waves[:-1])}
    present = set(zip(panel["pid"], panel["year"]))
    att = df[df["year"] < waves[-1]].copy()
    att["in_next"] = [
        (p, nxt[y]) in present for p, y in zip(att["pid"], att["year"])]
    att["in_next"] = att["in_next"].astype(float)
    emit("\nNext-wave presence by year x T:")
    emit(att.groupby(["year", "T"])["in_next"].agg(["mean", "count"])
         .round(4).unstack().to_string())
    for post_yr in (2016, 2018):
        att["post"] = (att["year"] >= post_yr).astype(int)
        fit(f"in_next ~ T:post + age + I(age**2) + C(bcity) + C(year)",
            att, f"in_next, Post={post_yr}", "bcity", ["T:post"])
    # attrition on the locked margin
    att = att.merge(n15, on="pid", how="left").merge(fb, on="pid", how="left")
    aone = att[(att["nkids15"] == 1) & (att["fb_son"].isin([0, 1]))].copy()
    aone["dghtr"] = 1 - aone["fb_son"]
    aone["post"] = (aone["year"] >= 2016).astype(int)
    fit(f"in_next ~ T:post + T:post:dghtr + post:dghtr + T:dghtr + dghtr + age + I(age**2) + C(bcity) + C(year)",
        aone, "in_next, DDD by firstborn gender (one-child 2015)", "bcity",
        ["T:post", "T:post:dghtr"])

    out = os.path.join(OUT, "tables", "migration_margin.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
