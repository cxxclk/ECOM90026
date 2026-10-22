# ==========================================================================
# REPLICATION PACKAGE — gender_norm_index.py
# Tier: CORE
# Paper items: Table 4 (A. Norm thirds, continuous norm, urban/income horse race, son-first); Figure 4 (Norm thirds (both indices))
# Reads: CFPS .dta files in data/20xx/, output/analysis_data.csv, output/panel_individual.csv
# Writes: output/tables/gender_norm_index.txt
# Imports local modules: annual_firstborn_gender
# Run order: step 16 in run_all.sh
# ==========================================================================

"""A4 + Doepke-Kindermann: baseline gender/son-continuation norms as a
moderator of the HPR price shock's fertility effect.

Question: where the local "carry-on-the-family-line" norm is strong, is the
son-chasing second-birth margin MORE or LESS price-elastic?
  H1 (norm-as-motive): suppression requires the son-chasing motive to exist
      -> concentrated where the norm is STRONG (T:post:norm < 0, daughter arm).
  H2 (norm-as-commitment): a strong norm makes the son non-negotiable
      -> price-inelastic where strong, suppression where WEAK (T:post:norm > 0).

Norm measures (2014 adult file, ALL respondents incl. men & older cohorts,
aggregated to the baseline prefecture -> a FIXED city-level moderator; this
sidesteps the intention-data N wall because the outcome side stays the full
roster hazard):
  son_norm    city mean of qm1004 "women should give birth to at least one
              boy to continue family lineage" (1-5 agree)
  trad_pc1    first principal component of the 10-item traditional battery
              qm1001-qm1006 + qm1101-qm1103 + reversed qm1104, city mean
  gap_mf      male minus female city mean of qm1004 (whose norm is it?)

Design: annual second-birth hazard (one-child risk set) from
annual_firstborn_gender.build_hazard_panel(); Post=2018, city+year FE
(absorb norm & T levels), age/duration controls, cluster by city.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annual_firstborn_gender import build_hazard_panel  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


TRAD_ITEMS = ["qm1001", "qm1002", "qm1003", "qm1004", "qm1005", "qm1006",
              "qm1101", "qm1102", "qm1103", "qm1104"]  # qm1104 reversed


def build_norm_index(min_city_n=25):
    df, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"),
        usecols=["pid", "countyid14", "cfps_gender"] + TRAD_ITEMS)
    for c in TRAD_ITEMS:
        df[c] = df[c].where(df[c].between(1, 5))
    df["qm1104r"] = 6 - df["qm1104"]  # "men should do half housework" -> traditional

    xw, _ = pyreadstat.read_dta(
        os.path.join(DATA, "CFPS_cleaned", "顺序码匹配.dta"))
    xw = xw[["countyid", "city"]].rename(columns={"city": "city_code"})
    xw["countyid"] = pd.to_numeric(xw["countyid"], errors="coerce")
    xw = xw.dropna().drop_duplicates("countyid")
    df["countyid"] = pd.to_numeric(df["countyid14"], errors="coerce")
    df = df.merge(xw, on="countyid", how="inner")

    items = [c for c in TRAD_ITEMS if c != "qm1104"] + ["qm1104r"]
    z = df[items].dropna()
    zs = (z - z.mean()) / z.std()
    # first principal component, oriented so higher = more traditional
    u, s, vt = np.linalg.svd(zs.values, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    load = vt[0]
    if load.mean() < 0:
        pc1, load = -pc1, -load
    df.loc[z.index, "trad_pc1"] = pc1
    emit("PC1 loadings (10-item traditional battery, 2014 adults):")
    for it, l in zip(items, load):
        emit(f"  {it:<8} {l:+.3f}")
    emit(f"  PC1 explained variance share: {s[0]**2 / (s**2).sum():.3f}")

    df["male"] = (df["cfps_gender"] == 1).astype(float)
    g = df.groupby("city_code")
    city = pd.DataFrame({
        "son_norm": g["qm1004"].mean(),
        "trad_pc1": g["trad_pc1"].mean(),
        "n_resp": g["qm1004"].count(),
        "son_norm_m": df[df["male"] == 1].groupby("city_code")["qm1004"].mean(),
        "son_norm_f": df[df["male"] == 0].groupby("city_code")["qm1004"].mean(),
    })
    city["gap_mf"] = city["son_norm_m"] - city["son_norm_f"]
    emit(f"\nCity norm index: {len(city)} cities; respondents/city "
         f"median={city['n_resp'].median():.0f}")
    city = city[city["n_resp"] >= min_city_n]
    emit(f"  kept {len(city)} cities with >= {min_city_n} respondents")
    emit(f"  son_norm mean={city['son_norm'].mean():.3f} sd={city['son_norm'].std():.3f}"
         f"  range=[{city['son_norm'].min():.2f}, {city['son_norm'].max():.2f}]")
    emit(f"  corr(son_norm, trad_pc1) across cities: "
         f"{city[['son_norm', 'trad_pc1']].corr().iloc[0, 1]:.3f}")
    return city.reset_index()


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  women={d['pid'].nunique():,}  "
         f"mean(y)={d['y'].mean():.4f}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<22} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    city = build_norm_index()
    df = build_hazard_panel()
    df = df.merge(city, on="city_code", how="inner")
    df["post"] = (df["year"] >= 2018).astype(int)

    # standardize moderators over the ANALYSIS cities (weight = presence in panel)
    cities_in = df.drop_duplicates("city_code")
    for v in ["son_norm", "trad_pc1", "gap_mf"]:
        mu, sd = cities_in[v].mean(), cities_in[v].std()
        df[v + "z"] = (df[v] - mu) / sd

    emit("\n" + "=" * 72)
    emit("Hazard panel x norm index")
    emit(f"  rows={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}")

    # balance: are treated cities less traditional? (they are urban/east corridor)
    bal = df.drop_duplicates("city_code").groupby("T")[
        ["son_norm", "trad_pc1", "gap_mf"]].mean()
    emit("\nNorm balance across treatment (city-level means):")
    emit(bal.round(3).to_string())

    CTRL = "age + I(age**2) + dur + I(dur**2)"
    FE = " + C(city) + C(year)"

    for norm in ["son_normz", "trad_pc1z"]:
        emit("\n" + "=" * 72)
        emit(f"MODERATOR = {norm}  (Post=2018; city FE absorbs norm & T levels)")
        for s, lab in [(0, "firstborn DAUGHTER"), (1, "firstborn SON (placebo)")]:
            d = df[df["son"] == s]
            fit(f"y ~ T:post + T:post:{norm} + post:{norm} + {CTRL}{FE}",
                d, f"{lab} | continuous interaction",
                ["T:post", f"T:post:{norm}", f"post:{norm}"])
        # median split for readability
        med = cities_in[norm[:-1]].median()
        for half, lab in [(df[norm[:-1]] <= med, "LOW-norm cities"),
                          (df[norm[:-1]] > med, "HIGH-norm cities")]:
            d = df[(df["son"] == 0) & half]
            fit(f"y ~ T:post + {CTRL}{FE}", d, f"DAUGHTER | {lab}", ["T:post"])

    # whose norm? male-female gap in qm1004 as moderator, daughter arm
    emit("\n" + "=" * 72)
    emit("MODERATOR = gap_mfz (male minus female son-norm)  daughter arm")
    d = df[df["son"] == 0]
    fit(f"y ~ T:post + T:post:gap_mfz + post:gap_mfz + {CTRL}{FE}",
        d, "DAUGHTER | m-f norm gap", ["T:post", "T:post:gap_mfz", "post:gap_mfz"])

    # gender-pooled second-birth margin (layer 2) for the average-effect story
    emit("\n" + "=" * 72)
    emit("Gender-pooled one-child risk set (layer-2 margin)")
    fit(f"y ~ T:post + T:post:son_normz + post:son_normz + {CTRL}{FE}",
        df, "POOLED | son_normz interaction",
        ["T:post", "T:post:son_normz", "post:son_normz"])

    # descriptive validation: the son-chasing premium should rise with the norm
    emit("\n" + "=" * 72)
    emit("VALIDATION: pre-2018 second-birth hazard by norm tercile x firstborn sex")
    pre = df[df["post"] == 0].copy()
    pre["tercile"] = pd.qcut(pre["son_norm"], 3, labels=["low", "mid", "high"])
    tab = pre.groupby(["tercile", "son"])["y"].agg(["mean", "count"]).unstack()
    tab[("premium", "")] = tab[("mean", 0)] - tab[("mean", 1)]
    emit(tab.round(4).to_string())

    # sharpest test: does the norm moderate the DAUGHTER-SPECIFIC suppression? (4-way)
    emit("\n" + "=" * 72)
    emit("DDD x norm (4-way): is the daughter-specific suppression norm-graded?")
    for norm in ["son_normz", "trad_pc1z"]:
        fit(f"y ~ T:post + T:post:son + T:post:{norm} + T:post:son:{norm}"
            f" + post:son + post:{norm} + post:son:{norm}"
            f" + T:son + T:son:{norm} + son + son:{norm} + {CTRL}{FE}",
            df, f"DDD 4-way | {norm}",
            ["T:post", f"T:post:{norm}", "T:post:son", f"T:post:son:{norm}"])

    # horse-race: norm vs urbanization vs income (is norm just proxying rural?)
    emit("\n" + "=" * 72)
    emit("HORSE-RACE: T:post:norm vs city urban share & city log income (2014 baseline)")
    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "year", "city_code", "urban"])
    urb = (pan[pan["year"] == 2014].groupby("city_code")["urban"].mean()
           .rename("urban_share").reset_index())
    ana = pd.read_csv(os.path.join(OUT, "analysis_data.csv"),
                      usecols=["year", "city_code", "log_income"])
    inc = (ana[ana["year"] == 2014].groupby("city_code")["log_income"].mean()
           .rename("city_linc").reset_index())
    df = df.merge(urb, on="city_code", how="left").merge(inc, on="city_code", how="left")
    ci = df.drop_duplicates("city_code")
    emit(f"  corr across cities: son_norm x urban_share "
         f"{ci[['son_norm', 'urban_share']].corr().iloc[0, 1]:+.3f}; "
         f"son_norm x city_linc {ci[['son_norm', 'city_linc']].corr().iloc[0, 1]:+.3f}")
    for v in ["urban_share", "city_linc"]:
        mu, sd = ci[v].mean(), ci[v].std()
        df[v + "z"] = (df[v] - mu) / sd
    d = df[(df["son"] == 0) & df["urban_sharez"].notna() & df["city_lincz"].notna()]
    fit(f"y ~ T:post + T:post:son_normz + post:son_normz"
        f" + T:post:urban_sharez + post:urban_sharez"
        f" + T:post:city_lincz + post:city_lincz + {CTRL}{FE}",
        d, "DAUGHTER | norm vs urban vs income",
        ["T:post", "T:post:son_normz", "T:post:urban_sharez", "T:post:city_lincz"])

    # tercile gradient, daughter arm (both indices)
    emit("\n" + "=" * 72)
    emit("Tercile gradient, daughter arm T:post (absolute and % of pre hazard)")
    for normv in ["son_norm", "trad_pc1"]:
        cuts = cities_in[normv].quantile([1 / 3, 2 / 3]).values
        for lo, hi, lab in [(-np.inf, cuts[0], "T1 low"),
                            (cuts[0], cuts[1], "T2 mid"),
                            (cuts[1], np.inf, "T3 high")]:
            d = df[(df["son"] == 0) & (df[normv] > lo) & (df[normv] <= hi)]
            base = d.loc[d["post"] == 0, "y"].mean()
            m = smf.ols(f"y ~ T:post + {CTRL}{FE}", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["city"]})
            b = m.params["T:post"]
            emit(f"  {normv:<9} {lab}:  {b:+.4f} (SE {m.bse['T:post']:.4f}, p={m.pvalues['T:post']:.3f})"
                 f"  pre-hazard={base:.3f}  rel={b / base:+.0%}  N={int(m.nobs):,}")

    # event study by norm half (trad_pc1), daughter arm: pre-trends within halves
    emit("\n" + "=" * 72)
    emit("Event study by trad_pc1 half, daughter arm (T x year, omitted 2015)")
    med = cities_in["trad_pc1"].median()
    ev_years = [y for y in range(2010, 2023) if y != 2015]
    for k in ev_years:
        df[f"Ty{k}"] = df["T"] * (df["year"] == k).astype(int)
    terms = " + ".join(f"Ty{k}" for k in ev_years)
    for half, lab in [(df["trad_pc1"] <= med, "LOW-norm"),
                      (df["trad_pc1"] > med, "HIGH-norm")]:
        d = df[(df["son"] == 0) & half]
        m = smf.ols(f"y ~ {terms} + {CTRL}{FE}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        emit(f"\n[DAUGHTER | {lab}]  N={int(m.nobs):,}")
        for k in ev_years:
            emit(f"  T x {k}   {m.params[f'Ty{k}']:+.4f}  "
                 f"(SE {m.bse[f'Ty{k}']:.4f}, p={m.pvalues[f'Ty{k}']:.3f})")

    out = os.path.join(OUT, "tables", "gender_norm_index.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
