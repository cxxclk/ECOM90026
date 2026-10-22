# ==========================================================================
# REPLICATION PACKAGE — welfare_margin_triangulation.py
# Tier: CORE
# Paper items: Figure 8 (Confidence in the future)
# Reads: CFPS .dta files in data/20xx/, output/prognostic_scores.csv
# Writes: output/tables/welfare_margin_triangulation.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard
# Run order: step 29 in run_all.sh
# ==========================================================================

"""Welfare x propensity-margin triangulation: do the SAME contested-middle
families who lose the birth also carry the confidence loss?

The graduated inverted-U (inverted_u_tests.py) says the fertility suppression
is proportional to the density of families at the decision margin
(dens = Pc(1-Pc)), entirely in the daughter arm. The welfare layer
(confidence-in-future qn12014/qn12016; memo `confidence-welfare-story`) says
the persistent confidence decline is localized to one-child WANTS-2nd
families (-0.135, p=0.036) with content families at exactly zero. If both
are registers of the same tightened constraint, then:

  P1  conf ~ T:post:dens < 0 on the parity-1 wave panel (margin carries the
      welfare loss), with the linear Pc term ~ 0;
  P2  the dens effect survives/coexists with the wants-2nd moderator
      (they are different projections: wants-2nd = demand exists,
      dens = the birth was in contention);
  P3  persistence: the dens-localized decline does NOT rebound in 2020-22.

Design: women from the one-child risk set, observed at waves 2012-2022 while
in the parity-1 state (by1 < wave <= by2-or-inf); outcome confidence about
one's future (1-5; qn12014 2012-16, qn12016 2018-22 — rename level shift is
common and absorbed by year FE); Post=2018; age controls, city+year FE,
cluster by city; propensity from prognostic_scores.csv (OOF, pre-trained).
First block replicates the welfare layer's wants-2nd localization to
validate this rebuild against the in-session original.
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
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from annual_firstborn_gender import load_rosters          # noqa: E402
import annual_parity_hazard as aph                        # noqa: E402

CONF = {
    2012: ("2012/ecfps2012adult_202505.dta", "qn12014"),
    2014: ("2014/ecfps2014adult_201906.dta", "qn12014"),
    2016: ("2016/ecfps2016adult_201906.dta", "qn12014"),
    2018: ("2018/ecfps2018person_202012.dta", "qn12016"),
    2020: ("2020/ecfps2020person_202306.dta", "qn12016"),
    2022: ("2022/ecfps2022person_202410.dta", "qn12016"),
}

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def cfit(formula, d, keys, label):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  women={d['pid'].nunique():,}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<20} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    frames = []
    for yr, (rel, var) in CONF.items():
        d, _ = pyreadstat.read_dta(os.path.join(DATA, rel),
                                   usecols=["pid", var])
        d.columns = ["pid", "conf"]
        d["conf"] = d["conf"].where(d["conf"].between(1, 5))
        d["year"] = yr
        frames.append(d.dropna())
    conf = pd.concat(frames, ignore_index=True)

    roster, _ = load_rosters()
    r = roster.sort_values(["pid", "cby"]).copy()
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "by1", "cgender": "fb_son"})
    second = r[r["k"] == 1][["pid", "cby"]].rename(columns={"cby": "by2"})

    prog = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))
    attrs = aph.woman_attrs()
    ad14, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"),
        usecols=["pid", "qm501"])
    ad14["wants2"] = (ad14["qm501"].where(ad14["qm501"].between(0, 10)) >= 2)

    d = (conf.merge(prog, on="pid", how="inner")
             .merge(attrs, on="pid", how="inner")
             .merge(first, on="pid", how="left")
             .merge(second, on="pid", how="left")
             .merge(ad14[["pid", "wants2"]], on="pid", how="left"))
    # parity-1 state at the wave
    d = d[(d["by1"] < d["year"])
          & (d["by2"].isna() | (d["by2"] >= d["year"]))]
    d["age"] = d["year"] - d["birthy"]
    d = d[d["age"].between(20, 50)]
    d["T"] = d["treat_hpr"].astype(int)
    d["city"] = d["city_code"].astype(int)
    d["post"] = (d["year"] >= 2018).astype(int)
    d["Pc"] = 1 - (1 - d["prop"].clip(0, 0.6)) ** 5
    d["dens"] = d["Pc"] * (1 - d["Pc"])
    d["son"] = d["fb_son"].astype(float)

    emit("WELFARE x PROPENSITY-MARGIN TRIANGULATION")
    emit(f"parity-1 wave obs with confidence: {len(d):,}  "
         f"women={d['pid'].nunique():,}  waves 2012-2022  "
         f"conf mean={d['conf'].mean():.2f}")

    CT = "age + I(age**2)"
    FE = " + C(city) + C(year)"

    # ---- validation: replicate the wants-2nd localization ---------------
    emit("\n--- validation vs the in-session welfare layer ---")
    cfit(f"conf ~ T:post + {CT}{FE}", d, ["T:post"], "one-child, ALL")
    for w, lab in [(True, "WANTS-2nd (ideal2014>=2) — expect ~ -0.135"),
                   (False, "CONTENT (<2) — expect ~ 0")]:
        cfit(f"conf ~ T:post + {CT}{FE}", d[d["wants2"] == w],
             ["T:post"], lab)

    # ---- P1: margin carries the welfare loss ----------------------------
    STRUCT = (f"conf ~ T:post + T:post:Pc + T:post:dens + post:Pc + post:dens"
              f" + T:Pc + T:dens + Pc + dens + {CT}{FE}")
    cfit(STRUCT, d, ["T:post", "T:post:Pc", "T:post:dens"],
         "P1 MAIN: confidence structural test (prediction: dens < 0)")
    for arm, lab in [(0, "daughter arm"), (1, "son arm")]:
        cfit(STRUCT, d[d["son"] == arm], ["T:post:Pc", "T:post:dens"],
             f"P1 by firstborn: {lab}")

    # ---- P2: dens vs wants-2nd horse-race -------------------------------
    dw = d[d["wants2"].notna()].copy()
    dw["w2"] = dw["wants2"].astype(int)
    cfit(f"conf ~ T:post + T:post:dens + T:post:w2 + post:dens + post:w2"
         f" + T:dens + T:w2 + dens + w2 + {CT}{FE}", dw,
         ["T:post", "T:post:dens", "T:post:w2"],
         "P2 horse-race: margin density vs wants-2nd")

    # ---- P3: persistence (non-rebound) ----------------------------------
    d["p1819"] = d["year"].isin([2018, 2019]).astype(int)   # 2018 wave only
    d["p2022"] = (d["year"] >= 2020).astype(int)
    cfit(f"conf ~ T:p1819 + T:p2022 + T:p1819:dens + T:p2022:dens"
         f" + p1819:dens + p2022:dens + T:dens + dens + {CT}{FE}", d,
         ["T:p1819", "T:p1819:dens", "T:p2022", "T:p2022:dens"],
         "P3 persistence: 2018 vs 2020-22 (non-rebound => late dens < 0)")

    # descriptive: confidence DID by propensity quintile
    emit("\nconfidence DID by propensity quintile (echo of the U?):")
    for g in range(1, 6):
        s = d[d["grp"] == g]
        if len(s) < 400:
            emit(f"  G{g}: n={len(s)} thin")
            continue
        m = smf.ols(f"conf ~ T:post + {CT}{FE}", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["city"]})
        emit(f"  G{g}: n={len(s):5,}  T:post={m.params['T:post']:+.4f} "
             f"(p={m.pvalues['T:post']:.3f})")

    out = os.path.join(OUT, "tables", "welfare_margin_triangulation.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
