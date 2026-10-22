# ==========================================================================
# REPLICATION PACKAGE — couple_caregiving_channel.py
# Tier: SUPPLEMENTARY
# Paper items: Table D1 (Caregiver rows, both moderators together)
# Reads: CFPS .dta files in data/20xx/
# Writes: output/tables/couple_caregiving_channel.txt
# Imports local modules: couple_intentions_annual
# Run order: step 48 in run_all.sh
# ==========================================================================

"""Mechanism step 2 (direct): the cost/caregiving channel, with a DIRECT
caregiving proxy instead of income share.

Story: under housing pressure the wife's fertility-decision weight rises. The
income-share moderation (couple_mechanism_annual.py) suggested a COST channel
(gain where she earns LESS, not more -> not resource/bargaining), but the
stricter 2014-cohort causal cut did NOT localize cleanly by income. The cost
channel is really about who bears the TIME/CAREGIVING cost of an extra child.
Test it directly with pre-shock housework time.

Proxy: qq9010 = daily housework hours, taken at the 2014 (pre-shock) wave for
BOTH spouses (wife = pid, husband = husband_pid). Define the wife's caregiving
burden share and a "wife is primary caregiver" dummy. Cost channel predicts the
wife-weight gain concentrates where SHE is the primary caregiver, regardless of
earnings — and that caregiving, not income, is the moderator that survives.

Designs (annual couple-year hazard panel, build_annual_panel):
  (1) Moderation: split by caregiver; T:ideal_w each; triple T:ideal_w:wife_cg.
  (2) Causal cut: 2014-cohort DID-in-weight T:post:ideal_w by caregiver split.
  (3) Horse-race: caregiving vs income-share moderators in one model.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "output")
from couple_intentions_annual import build_annual_panel  # noqa: E402

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def load_housework_2014():
    """Daily housework hours (qq9010) at the 2014 pre-shock wave, per pid."""
    d, _ = pyreadstat.read_dta("data/2014/ecfps2014adult_201906.dta",
                               usecols=["pid", "qq9010"])
    d["hw"] = pd.to_numeric(d["qq9010"], errors="coerce")
    d.loc[~d["hw"].between(0, 24), "hw"] = np.nan
    return d.dropna(subset=["hw"])[["pid", "hw"]]


def wshare(df):
    tot = df["income"].fillna(0) + df["h_income"].fillna(0)
    return pd.Series(np.where(tot > 0, df["income"].fillna(0) / tot, np.nan),
                     index=df.index)


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    cp, d = build_annual_panel()
    hw = load_housework_2014()
    d = d.merge(hw.rename(columns={"pid": "pid", "hw": "hw_w"}), on="pid", how="left")
    d = d.merge(hw.rename(columns={"pid": "husband_pid", "hw": "hw_h"}),
                on="husband_pid", how="left")
    d = d.dropna(subset=["hw_w", "hw_h"]).copy()
    tot = d["hw_w"] + d["hw_h"]
    d["cg_share"] = np.where(tot > 0, d["hw_w"] / tot, 0.5)   # wife's housework share
    d["wife_cg"] = (d["hw_w"] > d["hw_h"]).astype(int)        # wife does strictly more
    d["wshare"] = wshare(d)

    emit("=" * 72)
    emit("DIRECT CAREGIVING CHANNEL — pre-shock (2014) housework hours qq9010")
    emit(f"  couple-years with both spouses' housework: {len(d):,}  "
         f"couples={d['cid'].nunique():,}")
    emit(f"  wife daily hw hrs mean={d['hw_w'].mean():.2f}  "
         f"husband={d['hw_h'].mean():.2f}  "
         f"wife caregiving share mean={d['cg_share'].mean():.3f}")
    emit(f"  wife is primary caregiver (hw_w>hw_h): {d['wife_cg'].mean():.1%} of couple-years")

    CTRL = ("age_t + I(age_t**2) + dur + eduy + log_income + "
            "C(parity) + C(provcd) + C(yr)")

    # (1) Moderation by caregiver, cross-sectional T:ideal_w
    emit("\n" + "=" * 72)
    emit("(1) MODERATION (annual hazard) by who is the primary caregiver")
    for cg, lab in [(1, "WIFE is primary caregiver (hw_w > hw_h)"),
                    (0, "wife NOT primary caregiver (hw_w <= hw_h)")]:
        sub = d[d["wife_cg"] == cg]
        fit(f"y ~ T*ideal_w + T*ideal_h + {CTRL}", sub, lab,
            keys=["T:ideal_w", "T:ideal_h"])
    emit("\n(1b) Triple: T x ideal_w x wife-primary-caregiver")
    fit(f"y ~ T*ideal_w*wife_cg + T*ideal_h*wife_cg + {CTRL}", d,
        "T x ideal_w x wife_cg",
        keys=["T:ideal_w", "T:ideal_w:wife_cg", "T:ideal_h", "T:ideal_h:wife_cg"])

    # (2) Causal cut: 2014-cohort DID-in-weight by caregiver
    emit("\n" + "=" * 72)
    emit("(2) CAUSAL cut — 2014-cohort DID-in-weight by caregiver split")
    emit("    cost channel => shock-induced T:post:ideal_w lives where SHE cares")
    d14 = d[d["wave"] == 2014].copy()
    d14["post"] = (d14["yr"] >= 2016).astype(int)
    for cg, lab in [(1, "WIFE primary caregiver"), (0, "wife NOT primary caregiver")]:
        sub = d14[d14["wife_cg"] == cg]
        fit(f"y ~ T*post*ideal_w + T*post*ideal_h + {CTRL}", sub,
            f"2014-cohort DID-in-weight, {lab}",
            keys=["T:post:ideal_w", "T:post:ideal_h"])

    # (3) Horse-race: caregiving vs income share as the moderator
    emit("\n" + "=" * 72)
    emit("(3) HORSE-RACE — caregiving vs income-share moderator (same model)")
    emit("    which interaction survives: T:ideal_w:wife_cg or T:ideal_w:hi_w?")
    dd = d.dropna(subset=["wshare"]).copy()
    dd["hi_w"] = (dd["wshare"] >= dd["wshare"].median()).astype(int)
    fit(f"y ~ T*ideal_w*wife_cg + T*ideal_w*hi_w + T*ideal_h + {CTRL}", dd,
        "both moderators on wife's weight",
        keys=["T:ideal_w", "T:ideal_w:wife_cg", "T:ideal_w:hi_w"])

    # (4) Continuous caregiving share — rule out that the dummy is too coarse
    emit("\n" + "=" * 72)
    emit("(4) CONTINUOUS caregiving share (cgz = std wife housework share)")
    d["cgz"] = (d["cg_share"] - d["cg_share"].mean()) / d["cg_share"].std()
    fit(f"y ~ T*ideal_w*cgz + T*ideal_h*cgz + {CTRL}", d,
        "continuous caregiving moderation",
        keys=["T:ideal_w", "T:ideal_w:cgz", "T:ideal_h", "T:ideal_h:cgz"])
    d14b = d[d["wave"] == 2014].copy()
    d14b["post"] = (d14b["yr"] >= 2016).astype(int)
    fit(f"y ~ T*post*ideal_w*cgz + {CTRL}", d14b,
        "2014-cohort: does shock-shift scale with caregiving?",
        keys=["T:post:ideal_w", "T:post:ideal_w:cgz"])

    emit("\n" + "=" * 72)
    emit("VERDICT: the DIRECT caregiving proxy does NOT support the cost/")
    emit("caregiving-constraint channel — no moderation of the wife-weight gain")
    emit("by who does the housework (dummy p=0.96, continuous p=0.70; causal")
    emit("cut underpowered). Resource channel already refuted (first stage null).")
    emit("=> reduced-form wife-weight shift is robust+causal, but the CHANNEL is")
    emit("   not pinned down by either income share or caregiving time.")

    out = os.path.join(OUT, "tables", "couple_caregiving_channel.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
