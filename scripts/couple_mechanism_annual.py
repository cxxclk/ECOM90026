# ==========================================================================
# REPLICATION PACKAGE — couple_mechanism_annual.py
# Tier: SUPPLEMENTARY
# Paper items: Table D1 (First stage and income-share rows)
# Reads: output/couples_panel.csv
# Writes: output/tables/couple_mechanism_annual.txt
# Imports local modules: couple_intentions_annual
# Run order: step 47 in run_all.sh
# ==========================================================================

"""Mechanism (step 2), ANNUALIZED: is the wife's rising fertility-decision
weight bought with her own EARNINGS (resource/bargaining channel), or does it
fall on the lower-earning / primary-caregiving spouse (cost-constraint channel)?

Re-runs couple_mechanism.py on the annual couple-year hazard panel
(build_annual_panel), replacing the wave-level binary Y. Wife income share is
measured at the intention wave and used as a fixed couple moderator.

  (1) FIRST STAGE -- does the HPR shock raise the wife's income share /
      employment? (panel DiD on couples_panel; income is wave-level, so this
      stays at the couple-wave resolution — nothing to annualize.)
  (2) MODERATION (annual hazard) -- split the annual panel by wife income share
      and compare T:ideal_w; triple T x ideal_w x hi_w. Resource story => gain
      where she OUT-earns; cost story => gain where she earns LESS.
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
from couple_intentions_annual import build_annual_panel  # noqa: E402

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def wshare(df):
    tot = df["income"].fillna(0) + df["h_income"].fillna(0)
    s = np.where(tot > 0, df["income"].fillna(0) / tot, np.nan)
    return pd.Series(s, index=df.index)


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    # ---- (1) FIRST STAGE: shock -> wife's relative economic position --------
    emit("=" * 72)
    emit("(1) FIRST STAGE -- does the HPR shock raise the wife's income share?")
    cp_panel = pd.read_csv(os.path.join(OUT, "couples_panel.csv"))
    cp_panel = cp_panel[cp_panel["husband_pid"].notna() &
                        (cp_panel["is_regulated"] == 0) &
                        cp_panel["treat_hpr"].notna() &
                        cp_panel["city_code"].notna()].copy()
    cp_panel["city"] = cp_panel["city_code"].astype(int)
    cp_panel["wshare"] = wshare(cp_panel)
    cp_panel["wife_works"] = (cp_panel["income"].fillna(0) > 0).astype(float)
    cp_panel["did_hpr"] = (cp_panel["treat_hpr"].astype(int) *
                           (cp_panel["year"] >= 2016).astype(int))
    pan = cp_panel.dropna(subset=["wshare"])
    emit(f"  couple-waves with income share: {len(pan):,}  "
         f"mean wife share={pan['wshare'].mean():.3f}  "
         f"wife-works rate={cp_panel['wife_works'].mean():.3f}")
    fit("wshare ~ did_hpr + C(city) + C(year)", pan,
        "wife income share ~ did_hpr", keys=["did_hpr"])
    fit("wife_works ~ did_hpr + C(city) + C(year)", cp_panel,
        "wife works (income>0) ~ did_hpr", keys=["did_hpr"])

    # ---- (2) MODERATION on the ANNUAL hazard panel -------------------------
    emit("\n" + "=" * 72)
    emit("(2) MODERATION (annual hazard) -- wife's weight gain by income share")
    cp, d = build_annual_panel()
    d["wshare"] = wshare(d)
    d = d.dropna(subset=["wshare"]).copy()
    med = d["wshare"].median()
    d["hi_w"] = (d["wshare"] >= med).astype(int)
    emit(f"  couple-years with income share: {len(d):,}  "
         f"couples={d['cid'].nunique():,}  median wife share={med:.3f}")

    CTRL = ("age_t + I(age_t**2) + dur + eduy + log_income + "
            "C(parity) + C(provcd) + C(yr)")
    for half, lab in [(1, "wife OUT-earns (share >= median)"),
                      (0, "wife earns LESS (share < median)")]:
        sub = d[d["hi_w"] == half]
        fit(f"y ~ T*ideal_w + T*ideal_h + {CTRL}", sub, lab,
            keys=["T:ideal_w", "T:ideal_h"])

    emit("\n(2b) Triple interaction: T x ideal_w x wife-out-earns (annual)")
    fit(f"y ~ T*ideal_w*hi_w + T*ideal_h*hi_w + {CTRL}", d,
        "y ~ T x ideal_w x hi_w",
        keys=["T:ideal_w", "T:ideal_w:hi_w", "T:ideal_h", "T:ideal_h:hi_w"])

    emit("\n(2c) Causal cut: 2014-cohort triple-DID by wife income share")
    emit("    does the shock-induced wife-weight gain (T:post:ideal_w) live")
    emit("    where she earns LESS? (cost channel) or MORE? (resource channel)")
    d14 = d[d["wave"] == 2014].copy()
    d14["post"] = (d14["yr"] >= 2016).astype(int)
    for half, lab in [(0, "wife earns LESS"), (1, "wife OUT-earns")]:
        sub = d14[d14["hi_w"] == half]
        fit(f"y ~ T*post*ideal_w + T*post*ideal_h + {CTRL}", sub,
            f"2014-cohort DID-in-weight, {lab}",
            keys=["T:post:ideal_w", "T:post:ideal_h"])

    out = os.path.join(OUT, "tables", "couple_mechanism_annual.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
