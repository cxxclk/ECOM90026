# ==========================================================================
# REPLICATION PACKAGE — renter_mover_robustness.py
# Tier: CORE
# Paper items: Table 8 (Sample: owners only; women ever in another prefecture dropped)
# Reads: output/panel_individual.csv
# Writes: output/tables/renter_mover_robustness.txt
# Imports local modules: annual_firstborn_gender, headline_robustness
# Run order: step 33 in run_all.sh
# ==========================================================================

"""Renter and mover robustness for the headline (advisor comment #6, 2026-10-06).

Headline: annual second-birth hazard DID, one-child sample, daughter-first arm,
T x post (Post=2018) = -2.4pp (p=0.024). The headline panel
(annual_firstborn_gender.build_hazard_panel) puts NO tenure restriction on the
sample -- renters are already included -- and keeps movers, with exposure fixed
at the baseline-wave prefecture (intention-to-treat). This script makes both
facts explicit and reports:

  [A] Tenure composition of the headline sample (baseline tenure).
  [B] Excluding renters (owners only) / renters only / T x post x renter.
  [C] Movers: share who ever appear in a different prefecture than baseline,
      and the headline re-estimated on stayers only (two definitions):
        strict  = every later observed prefecture equals baseline AND never
                  observed with a missing prefecture after baseline (a missing
                  code = left the 162-county frame, a possible move);
        observed = drop only women observed in a different prefecture.
  [D] Wild cluster bootstrap for the owners-only and strict-stayers rows.

Baseline tenure = first non-missing `homeowner` in a pre-shock wave (<=2014),
as in tenure_wealth_cost.py; women first observed after 2014 fall back to
their earliest non-missing wave (flagged; a pre-shock-only variant is also run).
Conventions unchanged: regulated cities dropped, city + year FE, age/dur
quadratics, cluster by city.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annual_firstborn_gender import build_hazard_panel  # noqa: E402
from headline_robustness import (CTRL, FE, fit_did, wild_cluster_boot,  # noqa: E402
                                 LINES as _HL)

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def report(m, d, coef, label):
    b, se, p = m.params[coef], m.bse[coef], m.pvalues[coef]
    emit(f"  {label:<40} {100*b:+.2f}pp (SE {100*se:.2f}, p={p:.3f})"
         f"  N={int(m.nobs):,} women={d['pid'].nunique():,} mean={100*d['y'].mean():.2f}%")
    return b, p


def woman_traits():
    panel = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "gender", "city_code", "homeowner"])
    w = panel[panel["gender"] == 0].sort_values(["pid", "year"])

    # tenure
    pre = w[(w["year"] <= 2014) & w["homeowner"].notna()]
    own_pre = pre.groupby("pid")["homeowner"].first().rename("owner_pre")
    own_any = (w[w["homeowner"].notna()].groupby("pid")["homeowner"]
               .first().rename("owner_any"))

    # movers
    base = (w.dropna(subset=["city_code"]).groupby("pid")
            .agg(base_city=("city_code", "first"), base_year=("year", "first")))
    w = w.merge(base, on="pid", how="inner")
    later = w[w["year"] > w["base_year"]]
    moved_obs = (later["city_code"].notna()
                 & (later["city_code"] != later["base_city"])).groupby(later["pid"]).any()
    left_frame = later["city_code"].isna().groupby(later["pid"]).any()

    t = base.join(own_pre).join(own_any)
    t["moved_obs"] = moved_obs.reindex(t.index).fillna(False).astype(bool)
    t["left_frame"] = left_frame.reindex(t.index).fillna(False).astype(bool)
    t["owner"] = t["owner_pre"].fillna(t["owner_any"])
    t["owner_from_pre"] = t["owner_pre"].notna()
    return t.reset_index()[["pid", "owner", "owner_from_pre",
                            "moved_obs", "left_frame"]]


def main():
    df = build_hazard_panel()
    df["post"] = (df["year"] >= 2018).astype(int)
    df = df.merge(woman_traits(), on="pid", how="left")
    df["renter"] = 1 - df["owner"]

    arms = [(0, "DAUGHTER (headline)"), (1, "SON (placebo)")]

    emit("=" * 84)
    emit("RENTER AND MOVER ROBUSTNESS — annual second-birth rate, one-child sample")
    emit("city + year FE, age/dur quadratics, cluster by city, Post=2018")
    emit("=" * 84)

    # ---------------- [0] baseline ----------------
    emit("\n[0] BASELINE (headline sample: all tenures, movers kept, ITT exposure)")
    for s, lab in arms:
        d = df[df["son"] == s]
        report(fit_did(d, "T:post"), d, "T:post", lab)

    # ---------------- [A] composition ----------------
    wl = df.drop_duplicates("pid")
    emit("\n[A] COMPOSITION (women in the headline sample)")
    emit(f"  women={len(wl):,}  tenure observed={wl['owner'].notna().mean():.3f}"
         f"  (from pre-2015 wave: {wl['owner_from_pre'].mean():.3f})")
    for s, lab in arms:
        x = wl[wl["son"] == s]
        emit(f"  {lab:<22} renter share={x['renter'].mean():.3f}"
             f"  (exposed {x[x['T']==1]['renter'].mean():.3f} /"
             f" unexposed {x[x['T']==0]['renter'].mean():.3f})")
    emit(f"  ever observed in another prefecture: {wl['moved_obs'].mean():.3f}"
         f"  (exposed {wl[wl['T']==1]['moved_obs'].mean():.3f} /"
         f" unexposed {wl[wl['T']==0]['moved_obs'].mean():.3f})")
    emit(f"  ever observed outside the sampling frame (prefecture missing): "
         f"{wl['left_frame'].mean():.3f}")
    emit(f"  woman-years from women ever observed elsewhere: {df['moved_obs'].mean():.3f}")

    # ---------------- [B] renters ----------------
    emit("\n[B] TENURE")
    dt = df[df["owner"].notna()]
    for s, lab in arms:
        emit(f" {lab}")
        d = dt[dt["son"] == s]
        report(fit_did(d, "T:post"), d, "T:post", "tenure observed (same as [0] sample)")
        d1 = d[d["owner"] == 1]
        report(fit_did(d1, "T:post"), d1, "T:post", "OWNERS ONLY (renters excluded)")
        d0 = d[d["owner"] == 0]
        report(fit_did(d0, "T:post"), d0, "T:post", "RENTERS ONLY")
        m = fit_did(d, "T:post + T:post:renter + post:renter + renter")
        report(m, d, "T:post", "interaction: T x post (owners)")
        report(m, d, "T:post:renter", "interaction: T x post x renter")
        dp = d[d["owner_from_pre"]]
        dp1 = dp[dp["owner"] == 1]
        report(fit_did(dp1, "T:post"), dp1, "T:post", "owners only, pre-2015 tenure")
        dp0 = dp[dp["owner"] == 0]
        report(fit_did(dp0, "T:post"), dp0, "T:post", "renters only, pre-2015 tenure")

    # ---------------- [C] movers ----------------
    emit("\n[C] MOVERS (exposure is always the baseline prefecture)")
    for s, lab in arms:
        emit(f" {lab}")
        d = df[df["son"] == s]
        d_obs = d[~d["moved_obs"]]
        report(fit_did(d_obs, "T:post"), d_obs, "T:post",
               "drop women seen in another prefecture")
        d_str = d[~d["moved_obs"] & ~d["left_frame"]]
        report(fit_did(d_str, "T:post"), d_str, "T:post",
               "strict stayers (also never left frame)")
        dd = d[~d["moved_obs"] & (d["owner"] == 1)]
        report(fit_did(dd, "T:post"), dd, "T:post", "owners AND observed stayers")

    # ---------------- [D] WCB ----------------
    emit("\n[D] WILD CLUSTER BOOTSTRAP (Rademacher, null imposed, B=999), daughter arm")
    d = df[df["son"] == 0]
    f = f"y ~ T:post + {CTRL}{FE}"
    for lab, sub in [("owners only", d[d["owner"] == 1]),
                     ("observed stayers", d[~d["moved_obs"]]),
                     ("strict stayers", d[~d["moved_obs"] & ~d["left_frame"]])]:
        sub = sub.reset_index(drop=True)
        b, t0, pw, G = wild_cluster_boot(sub, f, "T:post")
        emit(f"  {lab:<22} {100*b:+.2f}pp  t={t0:+.2f}  WCB p={pw:.3f}  (G={G})")

    out = os.path.join(OUT, "tables", "renter_mover_robustness.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
