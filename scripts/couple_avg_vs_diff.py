# ==========================================================================
# REPLICATION PACKAGE — couple_avg_vs_diff.py
# Tier: CORE
# Paper items: Table 6 (C. Average/difference rows and pre-shock row); Table D2 (All couples, composite rows, honest half-splits)
# Reads: (via imported modules)
# Writes: output/tables/couple_avg_vs_diff.txt
# Imports local modules: couple_intentions_annual
# Run order: step 27 in run_all.sh
# ==========================================================================

"""Decompose the couple decision-weight effect into:
  avg  = (ideal_w + ideal_h)/2  -- the couple's JOINT desire ("intentionality":
         does the shock make realized fertility track stated ideals more, for
         BOTH spouses together?)
  diff = ideal_w - ideal_h       -- the wife's RELATIVE preference ("bargaining
         shift": does the wife's position specifically gain weight?)

Why this is the key test. 73% of couples AGREE on ideal kids, so the separate
identification of T:ideal_w vs T:ideal_h rides on the 27% who disagree. The
avg/diff reparametrization is the clean way to ask which of the two stories the
data support. It is algebraically equivalent to the (ideal_w, ideal_h) spec:
  coef_w = coef_avg/2 + coef_diff,   coef_h = coef_avg/2 - coef_diff.

Finding (annual couple-year hazard panel): the CROSS-SECTIONAL treated-vs-
control difference is in avg (intentionality) and NOT diff -- but that is a
pre-existing AREA trait (treated = urban corridor, more deliberate fertility):
it does not move with the shock (T:post:avg ~ 0). The CAUSAL pre/post effect
(2014 cohort) is the opposite: pure diff (+0.027, p~0.055), zero avg -- a clean
redistribution of decision weight from husband to wife, not a rise in joint
intentionality. This sharpens (and partly corrects) the earlier pooled "wife
weight rises" reading, and explains why the income/caregiving moderations were
murky (they mixed in the confounded avg component).
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


def fit(formula, d, label, keys, cl="city"):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d[cl]})
    emit(f"\n[{label}]  N={int(m.nobs):,}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<16} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    cp, d = build_annual_panel()
    d["avg"] = (d["ideal_w"] + d["ideal_h"]) / 2
    d["diff"] = d["ideal_w"] - d["ideal_h"]          # wife minus husband
    d["expz"] = (d["hpr_exposure"] - d["hpr_exposure"].mean()) / d["hpr_exposure"].std()
    d["cid"] = d["cid"].astype(int)

    agree = (d.groupby("cid")["diff"].first() == 0).mean()
    emit("AVG vs DIFF decomposition of the couple decision-weight effect")
    emit(f"  couple-years={len(d):,}  couples={d['cid'].nunique():,}  "
         f"couples agreeing on ideal kids={agree:.1%}")
    emit("  avg=(w+h)/2 = joint intentionality;  diff=w-h = wife relative weight")

    CTRL = ("age_t + I(age_t**2) + dur + eduy + log_income + "
            "C(parity) + C(provcd) + C(yr)")

    emit("\n" + "=" * 72)
    emit("(1) CROSS-SECTIONAL (pooled, binary T) — what differs treated vs control")
    fit(f"y ~ T*avg + T*diff + {CTRL}", d, "pooled annual",
        keys=["avg", "diff", "T:avg", "T:diff"])

    emit("\n(2) CROSS-SECTIONAL (continuous dose)")
    fit(f"y ~ expz*avg + expz*diff + {CTRL}", d, "exposure dose",
        keys=["expz:avg", "expz:diff"])

    emit("\n" + "=" * 72)
    emit("(3) CAUSAL — 2014-cohort pre(2015)/post(2016+) triple DID")
    emit("    T:post:avg = shock raises joint intentionality?")
    emit("    T:post:diff = shock redistributes weight toward the wife?")
    d14 = d[d["wave"] == 2014].copy()
    d14["post"] = (d14["yr"] >= 2016).astype(int)
    fit(f"y ~ T*post*avg + T*post*diff + {CTRL}", d14, "2014-cohort triple",
        keys=["T:avg", "T:post:avg", "T:diff", "T:post:diff", "T:post"])
    emit("\n  Robustness: cluster by couple")
    fit(f"y ~ T*post*avg + T*post*diff + {CTRL}", d14,
        "2014-cohort triple (cluster=couple)",
        keys=["T:post:avg", "T:post:diff"], cl="cid")

    emit("\n" + "=" * 72)
    emit("(4) Consistency: implied T:post:ideal_w/h from the avg/diff causal fit")
    m = smf.ols(f"y ~ T*post*avg + T*post*diff + {CTRL}", data=d14).fit(
        cov_type="cluster", cov_kwds={"groups": d14["city"]})
    a, b = m.params["T:post:avg"], m.params["T:post:diff"]
    emit(f"  T:post:ideal_w = avg/2 + diff = {a/2 + b:+.4f}   "
         f"T:post:ideal_h = avg/2 - diff = {a/2 - b:+.4f}")
    emit("  (compare A6: T:post:ideal_w +0.027, T:post:ideal_h -0.026)")

    out = os.path.join(OUT, "tables", "couple_avg_vs_diff.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
