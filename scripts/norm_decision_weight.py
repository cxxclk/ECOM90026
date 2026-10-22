# ==========================================================================
# REPLICATION PACKAGE — norm_decision_weight.py
# Tier: SUPPLEMENTARY
# Paper items: Table D2 (Norm halves, norm thirds, high composite × high norm)
# Reads: (via imported modules)
# Writes: output/tables/norm_decision_weight.txt
# Imports local modules: couple_wx_scan, gender_norm_index
# Run order: step 50 in run_all.sh
# ==========================================================================

"""T2 (lottery batch 4): gender-norm index x couple decision-weight — the
Doepke-Kindermann connection, and the GEOGRAPHY-TENSION test of the unification.

What's new vs couple_wx_scan.py (which already scanned son_norm LINEARLY and
found it dead):
  (1) trad_pc1 (the 10-item index that carries the FERTILITY norm gradient) was
      never tried on the decision-weight redistribution; and the fertility norm
      effect is TOP-TERCILE-concentrated, not linear (linear scans were blind
      to the inverted-U too) => test halves/terciles, not just the z-slope.
  (2) Geography tension: the wife-weight redistribution (T:post:diff) lives in
      urban/homeowner/educated-husband couples (couple_wx_scan); the fertility
      suppression lives in HIGH-norm traditional cities (gender_norm_index);
      corr(son_norm, urban_share) = -0.47 across cities. If the redistribution
      is absent exactly where the suppression happens, the unification story
      has an internal-coherence seam. Direct test: T:post:diff by norm half +
      horse-race vs the SES composite.
  (3) D-K prediction on the OUTCOME side: the shock suppresses births where
      the wife wants fewer (Fact 2, T:post:wife_fewer = -0.048 p=0.033, full
      cohort). Bargaining should bite where disagreement is consequential —
      is Fact 2 concentrated in high-norm cities (where the disagreement is
      about SONS and the son-chasing margin exists)?
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "output")
from couple_wx_scan import build, CTRL, BASE  # noqa: E402
from gender_norm_index import build_norm_index  # noqa: E402

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def cfit(formula, dd, keys, label):
    m = smf.ols(formula, data=dd).fit(cov_type="cluster",
                                      cov_kwds={"groups": dd["city"]})
    emit(f"  [{label}] N={int(m.nobs):,} cities={dd['city'].nunique()}")
    for k in keys:
        if k in m.params:
            emit(f"    {k:<24} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    d, _cands = build()          # 2014 causal cohort, post=yr>=2016, avg/diff
    norm = build_norm_index()    # city_code, son_norm, trad_pc1
    d = d.drop(columns=["son_norm"], errors="ignore").merge(
        norm[["city_code", "son_norm", "trad_pc1"]], on="city_code", how="left")
    d = d.dropna(subset=["trad_pc1"]).copy()

    # city-level cuts (median/terciles over unique cities, as in gender_norm_index)
    cities = d[["city_code", "trad_pc1", "son_norm"]].drop_duplicates("city_code")
    med = cities["trad_pc1"].median()
    d["hi_norm"] = (d["trad_pc1"] > med).astype(int)
    ter = pd.qcut(cities["trad_pc1"], 3, labels=[0, 1, 2])
    d = d.merge(cities.assign(norm_ter=ter.astype(int))[["city_code", "norm_ter"]],
                on="city_code", how="left")
    d["tradz"] = (d["trad_pc1"] - d["trad_pc1"].mean()) / d["trad_pc1"].std()

    # SES composite = mean of z(urban, homeowner, eduy_h) — the w(x) winners
    d["sesz"] = d[["xz_urban", "xz_homeowner", "xz_eduy_h"]].mean(axis=1)
    d["hi_ses"] = (d["sesz"] > d["sesz"].median()).astype(int)
    d["wife_fewer"] = (d["diff"] < 0).astype(int)

    emit("=" * 72)
    emit("(0) Baseline reproduce: causal DID-in-weight, 2014 cohort")
    cfit(BASE, d, ["T:post:diff", "T:post:avg"], "full cohort")

    emit("\n" + "=" * 72)
    emit("(1) T:post:diff by NORM split (trad_pc1, city-level cuts)")
    for lab, cond in [("LOW-norm half", d["hi_norm"] == 0),
                      ("HIGH-norm half", d["hi_norm"] == 1)]:
        cfit(BASE, d[cond], ["T:post:diff"], lab)
    for t in [0, 1, 2]:
        cfit(BASE, d[d["norm_ter"] == t], ["T:post:diff"], f"tercile {t}")
    emit("\n  4-way interactions (linear z + half dummy):")
    cfit(f"y ~ T*post*diff*tradz + T*post*avg + {CTRL}", d,
         ["T:post:diff", "T:post:diff:tradz"], "continuous tradz")
    cfit(f"y ~ T*post*diff*hi_norm + T*post*avg + {CTRL}", d,
         ["T:post:diff", "T:post:diff:hi_norm"], "hi_norm dummy")

    emit("\n" + "=" * 72)
    emit("(2) GEOGRAPHY TENSION: norm vs the SES composite (urban/owner/edu_h)")
    cu = d.drop_duplicates("cid")
    emit(f"  couple-level corr(hi_norm, sesz) = "
         f"{cu[['hi_norm', 'sesz']].corr().iloc[0, 1]:+.3f}")
    ct = pd.crosstab(cu["hi_norm"], cu["hi_ses"], normalize="index")
    emit(f"  P(hi_ses | low-norm)={ct.loc[0, 1]:.2f}   "
         f"P(hi_ses | high-norm)={ct.loc[1, 1]:.2f}")
    emit("\n  horse-race (both 4-ways in one model):")
    cfit(f"y ~ T*post*diff*sesz + T*post*diff*hi_norm + T*post*avg + {CTRL}", d,
         ["T:post:diff", "T:post:diff:sesz", "T:post:diff:hi_norm"], "joint")
    emit("\n  redistribution within the 2x2 (norm half x SES half):")
    for hn in [0, 1]:
        for hs in [0, 1]:
            sub = d[(d["hi_norm"] == hn) & (d["hi_ses"] == hs)]
            if sub["cid"].nunique() < 150:
                emit(f"  [norm={hn} ses={hs}] couples="
                     f"{sub['cid'].nunique()} too thin"); continue
            cfit(BASE, sub, ["T:post:diff"], f"norm={hn} ses={hs} "
                 f"(couples={sub['cid'].nunique()})")

    emit("\n" + "=" * 72)
    emit("(3) D-K outcome side: FACT 2 (T:post:wife_fewer) by norm half")
    F2 = f"y ~ T*post*wife_fewer + {CTRL}"
    cfit(F2, d, ["T:post", "T:post:wife_fewer"], "full cohort (Fact 2 reproduce)")
    for lab, cond in [("LOW-norm half", d["hi_norm"] == 0),
                      ("HIGH-norm half", d["hi_norm"] == 1)]:
        cfit(F2, d[cond], ["T:post", "T:post:wife_fewer"], lab)
    emit("\n  5-way check (interaction of Fact 2 with hi_norm):")
    cfit(f"y ~ T*post*wife_fewer*hi_norm + {CTRL}", d,
         ["T:post:wife_fewer", "T:post:wife_fewer:hi_norm"], "hi_norm grading")

    out = os.path.join(OUT, "tables", "norm_decision_weight.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
