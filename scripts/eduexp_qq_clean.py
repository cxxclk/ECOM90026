# ==========================================================================
# REPLICATION PACKAGE — eduexp_qq_clean.py
# Tier: CORE
# Paper items: Table 5 (A. Household education expenditure); Section 5.3 last sentence of first paragraph (ORIG block: with income control, 0.89)
# Reads: CFPS .dta files in data/20xx/, output/analysis_data.csv
# Writes: output/tables/eduexp_qq_clean.txt
# Imports local modules: none
# Run order: step 20 in run_all.sh
# ==========================================================================

"""
Clean-controls version of eduexp_qq_substitution.py (advisor round 2, 2026-10-20).

The original controls (age, log_income, urban) are measured in the same wave as the
outcome, so log_income is post-treatment (a bad control) and its missingness also
trims the sample. Three specs, same code path:
  ORIG  : age + log_income + urban (contemporaneous), sample requires all three
  CLEAN : age + age^2 (as in the fertility equations), no income/urban, no income filter
  CLEAN+: CLEAN + urban measured at the earliest wave <= 2014 (pre-shock)
Headline for Table 5 Panel A = CLEAN.

--- original docstring ---
Angle B: quantity-quality substitution — education expenditure response.

The firstborn-gender test shows treated daughter-first one-child families
forgo the second ("try for a son") birth. Where do the freed resources go?
If education spending on the existing child RISES in exactly that group,
the shock induced substitution from child quantity to child quality
("give up chasing a son, invest in the daughter"). If it falls, it is a
pure cost squeeze.

Outcome: household education/training expenditure (famecon):
  2012 → fp508  (⚠ fp510 in 2012 = fitness — another 2012 collision)
  2014–2022 → fp510
Transformed: ihs(edu_exp) = log(x + sqrt(x²+1)); also extensive margin (>0).

Sample/design: identical to firstborn_gender_ddd.py — one-child-at-baseline
married women, unregulated prefectures, city+year FE, cluster by city.
"""

import os
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
CONTROLS = ["age", "log_income", "urban"]

# ── 1. Education expenditure from famecon, wave-specific ─────────────────────
ECON = {
    2012: ("data/2012/ecfps2012famecon_201906.dta", "fid12", "fp508"),
    2014: ("data/2014/ecfps2014famecon_201906.dta", "fid14", "fp510"),
    2016: ("data/2016/ecfps2016famecon_201807.dta", "fid16", "fp510"),
    2018: ("data/2018/ecfps2018famecon_202101.dta", "fid18", "fp510"),
    2020: ("data/2020/ecfps2020famecon_202306.dta", "fid20", "fp510"),
    2022: ("data/2022/ecfps2022famecon_202410.dta", "fid22", "fp510"),
}
rows = []
for yr, (path, fid, var) in ECON.items():
    d, _ = pyreadstat.read_dta(path, usecols=[fid, var])
    d.columns = ["fid", "edu_exp"]
    d["edu_exp"] = pd.to_numeric(d["edu_exp"], errors="coerce")
    d.loc[d["edu_exp"] < 0, "edu_exp"] = np.nan        # CFPS negative codes
    d["year"] = yr
    rows.append(d)
    print(f"  {yr}: N={d['edu_exp'].notna().sum():,}  median={d['edu_exp'].median():.0f}  "
          f">0 share={(d['edu_exp'] > 0).mean():.2f}")
edu = pd.concat(rows)

# ── 2. Rebuild the one-child firstborn-gender sample ──────────────────────────
FAMCONF = {2014: "data/2014/ecfps2014famconf_170630.dta",
           2012: "data/2012/ecfps2012famconf_092015.dta"}

def child_info(year):
    by_cols = [f"tb1y_a_c{i}" for i in range(1, 11)]
    g_cols  = [f"tb2_a_c{i}" for i in range(1, 11)]
    df, _ = pyreadstat.read_dta(FAMCONF[year], usecols=["pid"] + by_cols + g_cols)
    df.columns = df.columns.str.lower()
    by = df[by_cols].where((df[by_cols] > 1900) & (df[by_cols] <= year))
    g = df[g_cols].where(df[g_cols].isin([0, 1, 5])).replace({5: 0}).where(by.notna().values)
    return pd.DataFrame({"pid": df["pid"],
                         "n_kids_base": by.notna().sum(axis=1),
                         "firstborn_son": g.where(by.eq(by.min(axis=1), axis=0).values).max(axis=1)})

b14, b12 = child_info(2014), child_info(2012)
base = pd.concat([b14, b12[~b12["pid"].isin(b14["pid"])]])

df = pd.read_csv("output/analysis_data.csv")
df = df[df["year"] != 2010].copy()
df["pid"] = df["pid"].astype(int)
base["pid"] = base["pid"].astype(int)
df = df.merge(base, on="pid", how="left")
df = df[(df["is_regulated"] == 0) & df["treat_hpr"].notna() & df["city_code"].notna()]
df = df.dropna(subset=["n_kids_base", "age"])
df["age2"] = df["age"] ** 2
ub = (df[df["year"] <= 2014].sort_values("year").groupby("pid")["urban"].first().rename("urban_base"))
df = df.merge(ub, on="pid", how="left")
df = df.merge(edu, on=["fid", "year"], how="left")
df["post"] = (df["year"] >= 2016).astype(int)
df["T"] = df["treat_hpr"].astype(int)
df["city"] = df["city_code"].astype(int)
df["ihs_edu"] = np.arcsinh(df["edu_exp"])
df["any_edu"] = (df["edu_exp"] > 0).astype(float).where(df["edu_exp"].notna())

one = df[(df["n_kids_base"] == 1) & df["firstborn_son"].isin([0, 1])].copy()
one = one.dropna(subset=["edu_exp"])
one["son"] = one["firstborn_son"].astype(int)
print(f"\nOne-child sample with edu_exp: {len(one):,} obs, "
      f"{one['pid'].nunique():,} women; edu_exp>0 share={one['any_edu'].mean():.2f}")
print(one.groupby(["son", "T", "post"])["ihs_edu"].mean().round(3))


FE = " + C(city) + C(year)"
SPECS = {"ORIG": ["age", "log_income", "urban"],
         "CLEAN": ["age", "age2"],
         "CLEAN+": ["age", "age2", "urban_base"]}
OUT = []
def emit(x=""): print(x); OUT.append(x)

def fit(formula, d, keys):
    d = d.dropna(subset=[c.strip() for c in formula.split("~")[1].split("+") if c.strip() and "C(" not in c and ":" not in c and not c.strip().startswith("Ty")])
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})
    return {k: (m.params[k], m.bse[k], m.pvalues[k]) for k in keys}, int(m.nobs)

dgt0 = one[one["son"] == 0].copy()
for k in [2016, 2018, 2020, 2022]:
    dgt0[f"Ty{k}"] = dgt0["T"] * (dgt0["year"] == k).astype(int)
multi = df[(df["n_kids_base"] >= 2)].dropna(subset=["edu_exp"]).copy()

emit("Household education expenditure, wave-level, Post = 2016, prefecture + year FE, cluster prefecture")
emit("coef (SE) [p]  N")
for name, ctrl in SPECS.items():
    CT = " + ".join(ctrl)
    emit(f"\n=== {name}: controls = {CT}")
    rows = [
      ("ihs, daughter-first", f"ihs_edu ~ T:post + {CT}{FE}", one[one['son'] == 0], ["T:post"]),
      ("ihs, son-first",      f"ihs_edu ~ T:post + {CT}{FE}", one[one['son'] == 1], ["T:post"]),
      ("any>0, daughter-first", f"any_edu ~ T:post + {CT}{FE}", one[one['son'] == 0], ["T:post"]),
      ("any>0, son-first",      f"any_edu ~ T:post + {CT}{FE}", one[one['son'] == 1], ["T:post"]),
      ("DDD ihs, T:post:son", f"ihs_edu ~ T:post + T:post:son + post:son + T:son + son + {CT}{FE}", one, ["T:post:son"]),
      ("ES daughter ihs",     f"ihs_edu ~ Ty2016 + Ty2018 + Ty2020 + Ty2022 + {CT}{FE}", dgt0, ["Ty2016", "Ty2018", "Ty2020", "Ty2022"]),
      ("ihs, 2+ children",    f"ihs_edu ~ T:post + {CT}{FE}", multi, ["T:post"]),
    ]
    for lab, f, d, keys in rows:
        r, n = fit(f, d, keys)
        for k, (b, se, p) in r.items():
            emit(f"  {lab:<24} {k:<11} {b:+.3f} ({se:.3f}) [{p:.3f}]  N={n:,}")

os.makedirs("output/tables", exist_ok=True)
with open("output/tables/eduexp_qq_clean.txt", "w") as fh:
    fh.write("\n".join(OUT) + "\n")
print("saved output/tables/eduexp_qq_clean.txt")
