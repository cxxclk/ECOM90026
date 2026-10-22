# ==========================================================================
# REPLICATION PACKAGE — inverted_u_tests.py
# Tier: SUPPLEMENTARY
# Paper items: Table B3 (Crude five-year conversion column); Table E3 (K = 3, 4 and 5 columns)
# Reads: output/prognostic_scores.csv
# Writes: output/tables/inverted_u_tests.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard, income_gradient_did
# Run order: step 45 in run_all.sh
# ==========================================================================

"""Two graduation tests for the inverted-U ("the shock kills the contested
marginal birth").

The prognostic-GATES inverted-U (G4 -2.9pp carries 66% of lost births; G5
near-certain unaffected) faces one serious mechanical objection and needs one
structural upgrade:

TEST 1 — cohort cumulative DID (kills the DYNAMIC-SELECTION objection).
  In a discrete-time hazard, high-propensity types deplete fast: post-period
  G5 women still at risk are the left-behind of their type, which could
  MANUFACTURE an inverted-U from a monotone truth. Fix: woman-level cohort
  outcome immune to risk-set depletion — Y = 1{second birth within K years
  of the first} for first-birth cohorts 2009-2013 (window fully pre) vs
  2016-2018 (window post), requiring the roster to cover by1+K. One row per
  woman, cohort + city FE, age-at-first-birth controls, cluster city.
  If the U is real, the quintile pattern REPRODUCES (G4 most negative);
  if it was depletion, the cohort design shows a monotone gradient instead.

TEST 2 — threshold-model structural test (upgrades shape -> prediction).
  A threshold-crossing model says the effect is proportional to the density
  of families at the decision margin, proxied by Pc(1-Pc) where Pc is the
  cumulative second-birth probability (max at Pc = 0.5 — which is G4:
  annual 0.129 -> 5-yr 0.50, vs G5 0.193 -> 0.66). Single-coefficient test
  on the annual hazard: T:post:dens < 0 with T:post:Pc ~ 0, where
  Pc = 1-(1-p_hat)^5 from the OOF pre-trained propensity, dens = Pc(1-Pc).
  + WCB on the density coefficient, daughter/son arms, and the tempo
  corollary: the 2016-17 boom should ALSO sit at the margin
  (T:boom:dens > 0) — marginal families both rush and retreat.

Propensity: output/prognostic_scores.csv (strictly pre-2016-trained,
city-fold OOF, median over 10 splits) — grouping is covariate-based only.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from annual_firstborn_gender import load_rosters, build_hazard_panel  # noqa
import annual_parity_hazard as aph                                    # noqa
from income_gradient_did import wcb                                   # noqa

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def cfit(formula, d, keys, label, note=""):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}{note}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    prog = pd.read_csv(os.path.join(OUT, "prognostic_scores.csv"))

    # ================= TEST 1: cohort cumulative DID =====================
    roster, last_wave = load_rosters()
    r = roster.sort_values(["pid", "cby"]).copy()
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "by1", "cgender": "g1"})
    second = r[r["k"] == 1][["pid", "cby"]].rename(columns={"cby": "by2"})
    attrs = aph.woman_attrs()
    d = (first.merge(second, on="pid", how="left")
              .merge(attrs, on="pid", how="inner")
              .merge(last_wave, on="pid"))
    d = d[(d["by2"].isna()) | (d["by2"] > d["by1"])]      # drop twin-firsts
    d["agefb"] = d["by1"] - d["birthy"]
    d = d[d["agefb"].between(18, 40)]
    d = d.merge(prog, on="pid", how="inner")
    d["T"] = d["treat_hpr"].astype(int)
    d["city"] = d["city_code"].astype(int)

    emit("TEST 1 — COHORT CUMULATIVE DID (dynamic-selection-proof design)")
    for K in (4, 3, 5):
        s = d[((d["by1"].between(2009, 2013)) | (d["by1"].between(2016, 2018)))
              & (d["last_wave"] >= d["by1"] + K)].copy()
        s["Y"] = ((s["by2"] - s["by1"]) <= K).fillna(False).astype(float)
        s["postc"] = (s["by1"] >= 2016).astype(int)
        CT = "agefb + I(agefb**2)"
        FEc = " + C(by1) + C(city)"
        if K == 4:
            emit(f"\nwindow K={K}: cohorts 2009-13 (pre) vs 2016-18 (post); "
                 f"women={len(s):,}  (pre {int((1-s['postc']).sum()):,} / "
                 f"post {int(s['postc'].sum()):,})  mean Y={s['Y'].mean():.3f}")
            cfit(f"Y ~ T:postc + {CT}{FEc}", s, ["T:postc"],
                 f"overall cohort DID, 2nd birth within {K} yrs")
            for gv, lab in [(0, "daughter-first"), (1, "son-first")]:
                ss = s[s["g1"] == gv]
                cfit(f"Y ~ T:postc + {CT}{FEc}", ss, ["T:postc"],
                     f"cohort DID, {lab}")
        # the U reproduction test: per propensity quintile
        emit(f"\nK={K} — cohort DID by propensity quintile "
             f"(U real -> G4 most negative; depletion artifact -> monotone):")
        for g in range(1, 6):
            ss = s[s["grp"] == g]
            if len(ss) < 150 or ss["T"].nunique() < 2:
                emit(f"  G{g}: n={len(ss)} too thin")
                continue
            base = ss.loc[ss["postc"] == 0, "Y"].mean()
            try:
                m = smf.ols(f"Y ~ T:postc + {CT}{FEc}", data=ss).fit(
                    cov_type="cluster", cov_kwds={"groups": ss["city"]})
                emit(f"  G{g}: n={len(ss):5,}  base={base:.3f}  "
                     f"T:postc={m.params['T:postc']:+.4f} "
                     f"(p={m.pvalues['T:postc']:.3f})")
            except Exception as e:
                emit(f"  G{g}: failed {e}")

    # structural version on the cohort outcome (K=4)
    s = d[((d["by1"].between(2009, 2013)) | (d["by1"].between(2016, 2018)))
          & (d["last_wave"] >= d["by1"] + 4)].copy()
    s["Y"] = ((s["by2"] - s["by1"]) <= 4).fillna(False).astype(float)
    s["postc"] = (s["by1"] >= 2016).astype(int)
    s["Pc"] = 1 - (1 - s["prop"].clip(0, 0.6)) ** 4
    s["dens"] = s["Pc"] * (1 - s["Pc"])
    cfit("Y ~ T:postc + T:postc:Pc + T:postc:dens + postc:Pc + postc:dens"
         " + T:Pc + T:dens + Pc + dens + agefb + I(agefb**2) + C(by1)"
         " + C(city)", s, ["T:postc", "T:postc:Pc", "T:postc:dens"],
         "cohort structural: linear vs density (prediction: dens<0, Pc~0)")

    # ================= TEST 2: annual structural test ====================
    emit("\n" + "=" * 74)
    emit("TEST 2 — THRESHOLD-MODEL STRUCTURAL TEST (annual hazard)")
    df = build_hazard_panel()
    df["post"] = (df["year"] >= 2018).astype(int)
    df = df.merge(prog, on="pid", how="inner")
    df["Pc"] = 1 - (1 - df["prop"].clip(0, 0.6)) ** 5
    df["dens"] = df["Pc"] * (1 - df["Pc"])
    wl = df.drop_duplicates("pid")
    emit(f"woman-years={len(df):,}  women={len(wl):,}  "
         f"Pc: mean={wl['Pc'].mean():.3f}  share>0.5={ (wl['Pc']>0.5).mean():.2f}"
         f"  corr(Pc,dens)={wl[['Pc','dens']].corr().iloc[0,1]:.2f}")

    CTRL = "age + I(age**2) + dur + I(dur**2)"
    FE = " + C(city) + C(year)"
    STRUCT = (f"y ~ T:post + T:post:Pc + T:post:dens + post:Pc + post:dens"
              f" + T:Pc + T:dens + Pc + dens + {CTRL}{FE}")
    cfit(STRUCT, df, ["T:post", "T:post:Pc", "T:post:dens"],
         "MAIN structural test (prediction: T:post:dens < 0, T:post:Pc ~ 0)")
    cfit(f"y ~ T:post + T:post:Pc + post:Pc + T:Pc + Pc + {CTRL}{FE}", df,
         ["T:post", "T:post:Pc"], "linear-only benchmark")
    for arm, lab in [(0, "daughter arm"), (1, "son arm")]:
        cfit(STRUCT, df[df["son"] == arm],
             ["T:post:Pc", "T:post:dens"], f"structural, {lab}")

    emit("\nWild cluster bootstrap on T:post:dens (4999 reps):")
    b, t, p, G = wcb(df, STRUCT, "T:post:dens")
    emit(f"  T:post:dens = {b:+.4f}  t={t:.2f}  WCB p={p:.4f}  (G={G})")

    # tempo corollary: boom (2016-17) at the margin too?
    df["boom"] = df["year"].isin([2016, 2017]).astype(int)
    df["post18"] = (df["year"] >= 2018).astype(int)
    cfit(f"y ~ T:boom + T:post18 + T:boom:dens + T:post18:dens + boom:dens"
         f" + post18:dens + T:dens + dens + {CTRL}{FE}", df,
         ["T:boom", "T:boom:dens", "T:post18", "T:post18:dens"],
         "tempo corollary (prediction: T:boom:dens > 0, T:post18:dens < 0)")
    emit("\nboom & bust by propensity quintile (T:boom / T:post18):")
    for g in range(1, 6):
        s2 = df[df["grp"] == g]
        m = smf.ols(f"y ~ T:boom + T:post18 + {CTRL}{FE}", data=s2).fit(
            cov_type="cluster", cov_kwds={"groups": s2["city"]})
        emit(f"  G{g}: boom {m.params['T:boom']:+.4f} "
             f"(p={m.pvalues['T:boom']:.3f})   post18 "
             f"{m.params['T:post18']:+.4f} (p={m.pvalues['T:post18']:.3f})")

    out = os.path.join(OUT, "tables", "inverted_u_tests.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
