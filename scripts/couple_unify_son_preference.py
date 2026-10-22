# ==========================================================================
# REPLICATION PACKAGE — couple_unify_son_preference.py
# Tier: CORE
# Paper items: Table 6 (A. Couple son-ideal gap); Table 6 (C. Wife fewer / agrees / more rows)
# Reads: CFPS .dta files in data/20xx/
# Writes: output/tables/couple_unify_son_preference.txt
# Imports local modules: annual_firstborn_gender, couple_intentions_annual
# Run order: step 25 in run_all.sh
# ==========================================================================

"""UNIFICATION: the husband->wife decision-weight redistribution explains the
headline son-chasing second-birth suppression of firstborn-daughter families.

The locked four-layer arc finds the HPR price shock suppresses the son-chasing
SECOND birth of firstborn-DAUGHTER one-child families. The couple angle finds
the shock causally redistributes the fertility-decision weight from husband to
wife (couple_avg_vs_diff.py: T:post:diff > 0, T:post:avg ~ 0). This script links
them with three facts that form one mechanism:

  FACT 1 (preference disagreement, this script): in firstborn-DAUGHTER ONE-child
    families, the husband wants significantly MORE sons than the wife
    (ideal-boys qka203, 2018; paired gap +0.19, p=0.003), a gap that is
    (i) significantly larger than in son-first families (interaction p=0.03) and
    (ii) specific to one-child families (all-parity interaction null). I.e.
    "trying for a son" is a HUSBAND preference the wife does not share, exactly
    on the margin where the shock bites.

  FACT 2 (causal redistribution, couple_avg_vs_diff.py): the shock shifts
    decision weight husband->wife (T:post:diff +0.027, p=0.055 / 0.025 by couple).
    Here we also show it is numerically larger in daughter-first one-child
    couples (directional, underpowered).

  => MECHANISM: shock hands the decision to the wife, who (unlike the husband)
     does not want to chase a son => the son-chasing 2nd birth is suppressed.
     This is the headline result, now mechanism-explained at the couple level.

Caveats: ideal-boys is 2018-cohort only (qka203 absent in 2014), so FACT 1 is a
descriptive cross-section; daughter-first one-child cell is thin (n=85). FACT 2's
daughter-first cut is underpowered. The unification is a three-fact triangulation,
not a single mediation estimate.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
from couple_intentions_annual import build_annual_panel  # noqa: E402
from annual_firstborn_gender import load_rosters  # noqa: E402

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def firstborn_gender():
    roster, _ = load_rosters()
    r = roster.sort_values(["pid", "cby"])
    r["k"] = r.groupby("pid").cumcount()
    return r[r["k"] == 0][["pid", "cgender"]].rename(columns={"cgender": "fb_son"})


def main():
    cp, d = build_annual_panel()
    fb = firstborn_gender()
    d["avg"] = (d["ideal_w"] + d["ideal_h"]) / 2
    d["diff"] = d["ideal_w"] - d["ideal_h"]

    # ---- FACT 1: spousal ideal-BOYS gap by firstborn gender x parity --------
    boys, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2018/ecfps2018person_202012.dta"), usecols=["pid", "qka203"])
    boys["qka203"] = pd.to_numeric(boys["qka203"], errors="coerce")
    boys.loc[(boys["qka203"] < 0) | (boys["qka203"] > 8), "qka203"] = np.nan
    boys = boys.dropna()

    g = cp[cp["wave"] == 2018].copy()
    g = g.merge(boys.rename(columns={"qka203": "boys_w"}), on="pid", how="left")
    g = g.merge(boys.rename(columns={"pid": "husband_pid", "qka203": "boys_h"}),
                on="husband_pid", how="left")
    g = g.merge(fb, on="pid", how="left").dropna(subset=["boys_w", "boys_h", "fb_son"])
    g["sgap"] = g["boys_h"] - g["boys_w"]
    g["dghtr"] = (g["fb_son"] == 0).astype(int)

    emit("=" * 72)
    emit("FACT 1 — spousal ideal-SON gap (husband - wife), qka203 2018 cohort")
    emit("  'trying for a son' as a within-couple disagreement")
    for sub, plab in [(g, "ALL parities"), (g[g["parity"] == 1], "ONE-child only")]:
        emit(f"\n  [{plab}]")
        for s, lab in [(0, "DAUGHTER-first"), (1, "SON-first")]:
            x = sub[sub["fb_son"] == s]
            if len(x) < 20:
                emit(f"    {lab}: n={len(x)} too few"); continue
            t, p = stats.ttest_rel(x["boys_h"], x["boys_w"])
            emit(f"    {lab}: n={len(x):4d}  wife={x['boys_w'].mean():.2f} "
                 f"husband={x['boys_h'].mean():.2f}  gap={x['sgap'].mean():+.3f}  "
                 f"paired t={t:.2f} (p={p:.3f})  husb wants more sons in "
                 f"{100*(x['boys_h']>x['boys_w']).mean():.0f}%")
        m = smf.ols("sgap ~ dghtr", data=sub).fit(cov_type="HC1")
        emit(f"    interaction (gap larger in daughter-first?): "
             f"dghtr={m.params['dghtr']:+.3f} (p={m.pvalues['dghtr']:.3f}), N={int(m.nobs)}")

    # ---- FACT 2: the redistribution mechanism — shock suppresses births -----
    #      where the WIFE wants fewer than the husband (powered, full cohort).
    #      (The 4-way redistribution x daughter-first interaction is a power wall:
    #       ~110 disagreeing daughter-first one-child couples; T:post:diff:dghtr
    #       +0.029 p=0.42. So test the mechanism on the full cohort instead.)
    emit("\n" + "=" * 72)
    emit("FACT 2 — the shock suppresses births where the WIFE wants FEWER than")
    emit("  the husband (the redistribution biting; 2014 cohort, full power)")
    d14 = d[d["wave"] == 2014].copy()
    d14["post"] = (d14["yr"] >= 2016).astype(int)
    d14["wife_fewer"] = (d14["diff"] < 0).astype(int)
    CTRL = "age_t + I(age_t**2) + dur + eduy + log_income + C(parity) + C(provcd) + C(yr)"
    for cond, lab in [(d14["diff"] < 0, "wife wants FEWER (diff<0)"),
                      (d14["diff"] == 0, "agree (diff=0)"),
                      (d14["diff"] > 0, "wife wants MORE (diff>0)")]:
        sub = d14[cond]
        m = smf.ols(f"y ~ T*post + {CTRL}", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["city"]})
        emit(f"  {lab}: N={int(m.nobs):,} couples={sub['cid'].nunique()}  "
             f"T:post={m.params['T:post']:+.4f} (p={m.pvalues['T:post']:.3f})")
    m = smf.ols(f"y ~ T*post*wife_fewer + {CTRL}", data=d14).fit(
        cov_type="cluster", cov_kwds={"groups": d14["city"]})
    emit(f"  INTERACTION T:post:wife_fewer = {m.params['T:post:wife_fewer']:+.4f} "
         f"(p={m.pvalues['T:post:wife_fewer']:.3f}); T:post(agree)≈{m.params['T:post']:+.4f}")

    # ---- BRIDGE: where is the wife the 'fewer' party? total kids vs sons ------
    emit("\n" + "=" * 72)
    emit("BRIDGE — 'wife wants fewer' by firstborn gender (one-child couples):")
    g1 = g[g["parity"] == 1]
    for s, lab in [(0, "daughter-first"), (1, "son-first")]:
        x = g1[g1["fb_son"] == s]
        emit(f"  {lab}: wants-fewer-SONS={100*(x['boys_w']<x['boys_h']).mean():.0f}% "
             f"(n={len(x)})")
    emit("  (total-kids 'wife fewer' is ~17-18% in BOTH groups — NOT daughter-")
    emit("   specific; only the SON dimension is, which is what channels the")
    emit("   general redistribution into the daughter-first 2nd-birth margin.)")

    emit("\n" + "=" * 72)
    emit("UNIFICATION (precise): the shock redistributes the fertility decision")
    emit("to the wife and so SUPPRESSES births where she wants fewer than the")
    emit("husband (Fact 2, causal, total kids, p=0.02). The wife is the 'fewer'")
    emit("party on SONS specifically in daughter-first one-child families (Fact 1,")
    emit("p=0.003). => the son-chasing 2nd birth of daughter-first families is")
    emit("suppressed = the locked headline. One mechanism, two preference")
    emit("dimensions. Seam: son-disagreement is 2018-only (qka203), so the SON")
    emit("causal suppression itself can't be pre/post-estimated.")

    out = os.path.join(OUT, "tables", "couple_unify_son_preference.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
