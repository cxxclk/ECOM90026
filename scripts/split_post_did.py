# ==========================================================================
# REPLICATION PACKAGE — split_post_did.py
# Tier: CORE
# Paper items: Table 3 (B. Post-period windows)
# Note: prints to screen; run_all.sh redirects it to output/tables/split_post_did.txt
# Reads: (via imported modules)
# Writes: (nothing on disk)
# Imports local modules: annual_firstborn_gender
# Run order: step 15 in run_all.sh
# ==========================================================================

"""Split-post DID: does the daughter-arm suppression load on 2018-19 or 2020-22?
Reuses build_hazard_panel from annual_firstborn_gender.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statsmodels.formula.api as smf
from annual_firstborn_gender import build_hazard_panel

df = build_hazard_panel()
df["boom"] = df["year"].isin([2016, 2017]).astype(int)
df["post_a"] = df["year"].isin([2018, 2019]).astype(int)
df["post_b"] = (df["year"] >= 2020).astype(int)

CTRL = "age + I(age**2) + dur + I(dur**2) + C(city) + C(year)"


def fit(d, label, formula):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    print(f"\n[{label}]  N={int(m.nobs):,}  mean(y)={d['y'].mean():.4f}")
    for k in m.params.index:
        if ":" in k and "city" not in k and "year" not in k:
            print(f"  {k:<16} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                  f"p={m.pvalues[k]:.3f})")
    return m


for son, arm in [(0, "DAUGHTER"), (1, "SON placebo")]:
    d = df[df["son"] == son]
    fit(d, f"{arm} | split post: boom 16-17 / 18-19 / 20-22",
        f"y ~ T:boom + T:post_a + T:post_b + {CTRL}")
    fit(d, f"{arm} | split post only (16-17 in baseline)",
        f"y ~ T:post_a + T:post_b + {CTRL}")

# raw treated-control gap by period, daughter arm
d = df[df["son"] == 0]
d = d.assign(period=d["year"].map(
    lambda y: "pre 09-15" if y <= 2015 else
              "boom 16-17" if y <= 2017 else
              "post 18-19" if y <= 2019 else "post 20-22"))
g = d.groupby(["period", "T"])["y"].mean().unstack()
g["gap(T-C)"] = g[1] - g[0]
print("\nRaw daughter-arm hazard by period:")
print(g.round(4).to_string())
