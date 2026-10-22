# ==========================================================================
# REPLICATION PACKAGE — norm_erosion_did.py
# Tier: SUPPLEMENTARY
# Paper items: Section 2.2 / Appendix (norm index stability, corr 0.75)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/norm_erosion_did.txt
# Imports local modules: none
# Run order: step 55 in run_all.sh
# ==========================================================================

"""Ticket E: son-preference norm as OUTCOME — does the housing-price shock
erode stated lineage norms? (Fernandez-style endogenous cultural change.)

Data: qm1004 "women should give birth to at least one boy to continue family
lineage" (1-5 agree), clean 2014 vs 2022 pair (label-verified; the one
attitude item safe across waves). Placebos: qm1001 (respect parents),
qm1006 (visit parents) — elder-norm items, same battery, non-fertility.
2022 battery is a ~7.1k-person subsample of the person file.

Designs (identification strength labelled honestly):
  D1  within-person change Delta = norm22 - norm14 on T (treat_hpr, baseline
      city, regulated dropped), province FE, cluster city. T is time-invariant
      city-level -> this is a treated-vs-control single difference on top of a
      within-person difference (same structure as couple-line A1), NOT an
      event-study DID.
  D2  daughter leverage: among parents (>=1 child by 2014, firstborn gender
      from the famconf roster machinery), Delta on daughter, T, T x daughter.
      Firstborn gender is quasi-random (balance verified in the main line).
  D3  placebo outcomes Delta qm1001 / qm1006.
Gates: panel-inclusion attrition vs T; floor share (2014 qm1004 == 1);
city-index 2014-2022 stability (A4 open item).
"""
import os, re, warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

LINES = []
def emit(s=""):
    print(s); LINES.append(s)

ITEMS = ["qm1001", "qm1004", "qm1006"]

def load_wave(year, path):
    df, _ = pyreadstat.read_dta(os.path.join(DATA, path), usecols=["pid"] + ITEMS)
    for c in ITEMS:
        df[c] = df[c].where(df[c].between(1, 5))
    df = df.dropna(subset=["qm1004"])
    return df.rename(columns={c: f"{c}_{year}" for c in ITEMS})

def load_rosters():
    """firstborn birth year + gender per pid from latest famconf (copied from
    annual_firstborn_gender.py, trimmed)."""
    FAMCONF = {
        2010: "2010/ecfps2010famconf_nat072016.dta",
        2012: "2012/ecfps2012famconf_092015.dta",
        2014: "2014/ecfps2014famconf_170630.dta",
        2016: "2016/ecfps2016famconf_201804.dta",
        2018: "2018/ecfps2018famconf_202008.dta",
        2020: "2020/ecfps2020famconf_202306.dta",
        2022: "2022/ecfps2022famconf_202410.dta",
    }
    frames = []
    for year, rel in FAMCONF.items():
        p = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(p, metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        g_cols = [c.replace("tb1y", "tb2") for c in by_cols if c.replace("tb1y", "tb2") in meta.column_names]
        df, _ = pyreadstat.read_dta(p, usecols=["pid"] + by_cols + g_cols)
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
        long["w"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    allw = allw.merge(allw.groupby("pid")["w"].max().rename("lw"), on="pid")
    r = allw[allw["w"] == allw["lw"]].sort_values(["pid", "cby"])
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cg"]].rename(columns={"cby": "fb_year", "cg": "fb_son"})
    nkid14 = r[r["cby"] <= 2014].groupby("pid").size().rename("nkid14")
    return first, nkid14.reset_index()

def cluster_fit(formula, d, label, keys):
    used = [c for c in d.columns if re.search(rf"\b{re.escape(c)}\b", formula)]
    d = d.dropna(subset=set(used + ["city"]))
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  cities={d['city'].nunique()}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<16} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m

def main():
    w14 = load_wave(2014, "2014/ecfps2014adult_201906.dta")
    w22 = load_wave(2022, "2022/ecfps2022person_202410.dta")
    emit(f"qm1004 valid: 2014 N={len(w14):,}  2022 N={len(w22):,} (2022 battery = subsample)")

    panel = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "gender", "birthy", "urban", "eduy", "city_code", "provcd"])
    base = (panel.dropna(subset=["city_code"]).sort_values(["pid", "year"])
            .groupby("pid").first().reset_index()
            [["pid", "gender", "birthy", "urban", "eduy", "city_code", "provcd"]])
    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    base = base.merge(hpr[["city_code", "is_regulated", "treat_hpr"]], on="city_code", how="inner")
    base = base[base["is_regulated"] == 0]
    base["T"] = base["treat_hpr"].astype(int)
    base["city"] = base["city_code"].astype(int)

    # ---- attrition gate: who makes it into the 2022 battery panel ----
    at = base.merge(w14, on="pid", how="inner")
    at["in22"] = at["pid"].isin(w22["pid"]).astype(int)
    emit(f"\n2014 respondents in frame: {len(at):,}; re-observed in 2022 battery: {at['in22'].mean():.3f}")
    m = smf.ols("in22 ~ T + qm1004_2014 + T:qm1004_2014", data=at.assign(city=at["city"])).fit(
        cov_type="cluster", cov_kwds={"groups": at["city"]})
    emit("  attrition gate (in22 ~ T + norm14 + T:norm14):")
    for k in ["T", "qm1004_2014", "T:qm1004_2014"]:
        emit(f"    {k:<16} {m.params[k]:+.4f}  (p={m.pvalues[k]:.3f})")

    d = at[at["in22"] == 1].merge(w22, on="pid", how="inner")
    d["age14"] = 2014 - d["birthy"]
    d = d[(d["age14"] >= 18) & (d["age14"] <= 60)]
    for it in ITEMS:
        d[f"d_{it}"] = d[f"{it}_2022"] - d[f"{it}_2014"]
    emit(f"\nAnalysis panel (age 18-60 in 2014, unregulated): N={len(d):,}  T share={d['T'].mean():.3f}")
    emit(f"  qm1004 2014 mean {d['qm1004_2014'].mean():.3f} -> 2022 {d['qm1004_2022'].mean():.3f}  "
         f"(Delta mean {d['d_qm1004'].mean():+.3f})")
    emit(f"  floor gate: share at qm1004_2014==1 (cannot fall further): {(d['qm1004_2014']==1).mean():.3f}")

    CTRL = "age14 + I(age14**2) + eduy + urban + gender"
    emit("\n" + "=" * 72)
    emit("D1: Delta norm on T  (province FE, cluster city — single-diff x within-person)")
    for it, lab in [("qm1004", "SON NORM qm1004"), ("qm1001", "placebo qm1001 respect-parents"),
                    ("qm1006", "placebo qm1006 visit-parents")]:
        cluster_fit(f"d_{it} ~ T + {CTRL} + C(provcd)", d, lab, ["T"])
    # baseline-level control variant (regression to mean guard)
    cluster_fit(f"d_qm1004 ~ T + qm1004_2014 + {CTRL} + C(provcd)", d,
                "SON NORM + 2014-level control", ["T", "qm1004_2014"])

    emit("\n" + "=" * 72)
    emit("D2: daughter leverage among parents (firstborn by 2014, quasi-random gender)")
    first, nkid14 = load_rosters()
    p = d.merge(first, on="pid", how="inner").merge(nkid14, on="pid", how="left")
    p = p[(p["fb_year"] <= 2014) & p["fb_son"].isin([0, 1])]
    p["dghtr"] = 1 - p["fb_son"]
    emit(f"  parents panel N={len(p):,}  daughter share={p['dghtr'].mean():.3f}")
    cluster_fit(f"d_qm1004 ~ dghtr + T + T:dghtr + {CTRL} + C(provcd)", p,
                "ALL parents", ["dghtr", "T", "T:dghtr"])
    oc = p[p["nkid14"] == 1]
    emit(f"  one-child-at-2014 subset N={len(oc):,}")
    cluster_fit(f"d_qm1004 ~ dghtr + T + T:dghtr + {CTRL} + C(provcd)", oc,
                "ONE-CHILD parents", ["dghtr", "T", "T:dghtr"])
    cluster_fit(f"d_qm1001 ~ dghtr + T + T:dghtr + {CTRL} + C(provcd)", p,
                "placebo qm1001, all parents", ["dghtr", "T", "T:dghtr"])

    emit("\n" + "=" * 72)
    emit("City norm-index stability 2014 vs 2022 (A4 open item; cells n>=15)")
    allw = pd.concat([
        base.merge(w14, on="pid")[["city", "qm1004_2014"]].rename(columns={"qm1004_2014": "v"}).assign(y=2014),
        base.merge(w22, on="pid")[["city", "qm1004_2022"]].rename(columns={"qm1004_2022": "v"}).assign(y=2022),
    ])
    cm = allw.groupby(["city", "y"])["v"].agg(["mean", "size"]).unstack()
    ok = cm[(cm[("size", 2014)] >= 15) & (cm[("size", 2022)] >= 15)]
    r = np.corrcoef(ok[("mean", 2014)], ok[("mean", 2022)])[0, 1]
    emit(f"  cities with n>=15 both waves: {len(ok)}  corr(city mean 2014, 2022) = {r:.3f}")
    emit(f"  mean city-level change: {(ok[('mean',2022)] - ok[('mean',2014)]).mean():+.3f}")

    with open(os.path.join(OUT, "tables", "norm_erosion_did.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("\nSaved -> output/tables/norm_erosion_did.txt")

if __name__ == "__main__":
    main()
