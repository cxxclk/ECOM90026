# ==========================================================================
# REPLICATION PACKAGE — city_price_response.py
# Tier: HELPER (imported by other scripts, not run directly)
# Paper items: none directly (support module)
# Reads: output/housing_price_city.csv
# Writes: output/tables/city_price_response.txt
# Imports local modules: cddf_spine
# Run order: not run directly
# ==========================================================================

"""City-level classification: which treated cities actually got the price
shock, and does the fertility suppression follow it?

Last untried classification unit (2026-07-21): classify CITIES (not
families) by their REALIZED post-2016 second-hand price trajectory — the
first-stage intensity — then test whether the second-birth suppression
concentrates in treated cities where prices actually surged. This validates
the causal chain (HPR spillover -> price -> fertility) beyond the binary
reduced form, and is the classification analogue of a dose-response.

Framing caveat (carry into any write-up): realized price growth is
ENDOGENOUS (local demand shocks can move both prices and fertility) and the
price panel's coverage is selected (larger/more urban cities are covered) —
this is mechanism validation / effect-location, NOT a new identification.

Design:
  g_c = log hp(city, t1) - log hp(city, t0), primary window 2016->2019
        (best joint coverage), alt 2015->2019 reported.
  First stage: g_c ~ T among sample cities (HC1), + covered-subsample
        reduced form to show the fertility DID survives conditioning on
        coverage.
  Split: treated cities median-split into HI/LO price response ->
        y ~ T_hi:post + T_lo:post + ... vs all control cities (city+year FE),
        difference test, daughter/son arms, event studies.
  Continuous: T:post:g_z among covered cities (T x post x price-growth).
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
import cddf_spine as cs                                  # noqa: E402

CTRL, FE = cs.CTRL, cs.FE
LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def growth(hp, y0, y1):
    a = hp[hp["year"] == y0][["city_code", "log_hp_city"]].rename(
        columns={"log_hp_city": "l0"})
    b = hp[hp["year"] == y1][["city_code", "log_hp_city"]].rename(
        columns={"log_hp_city": "l1"})
    g = a.merge(b, on="city_code")
    g["g"] = g["l1"] - g["l0"]
    return g[["city_code", "g"]]


def fit(formula, d, keys, label):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  cities={d['city'].nunique()}  "
         f"mean(y)={d['y'].mean():.4f}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<16} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    df = cs.assemble()
    hp = pd.read_csv(os.path.join(OUT, "housing_price_city.csv"))
    emit("CITY-LEVEL PRICE-RESPONSE CLASSIFICATION x SECOND-BIRTH HAZARD")

    sample_cities = df.drop_duplicates("city_code")[["city_code", "T"]]
    for (y0, y1) in [(2016, 2019), (2015, 2019), (2016, 2020)]:
        g = growth(hp, y0, y1).merge(sample_cities, on="city_code")
        emit(f"  window {y0}->{y1}: covered sample cities = {len(g)} "
             f"(T={int(g['T'].sum())}, C={int((1-g['T']).sum())})")
    g = growth(hp, 2016, 2019).merge(sample_cities, on="city_code")
    emit(f"\nprimary window 2016->2019; growth mean={g['g'].mean():.3f} "
         f"sd={g['g'].std():.3f}")

    # ---- first stage at city level --------------------------------------
    m = smf.ols("g ~ T", data=g).fit(cov_type="HC1")
    emit(f"first stage (city level): treated minus control 3-yr log-price "
         f"growth = {m.params['T']:+.4f} (SE {m.bse['T']:.4f}, "
         f"p={m.pvalues['T']:.3f})")
    emit(f"  treated mean {g.loc[g['T']==1,'g'].mean():+.3f}   "
         f"control mean {g.loc[g['T']==0,'g'].mean():+.3f}")

    # ---- classify treated cities by realized response -------------------
    gt = g[g["T"] == 1]
    med = gt["g"].median()
    hi_cities = set(gt.loc[gt["g"] > med, "city_code"])
    lo_cities = set(gt.loc[gt["g"] <= med, "city_code"])
    emit(f"\ntreated split at median growth ({med:+.3f}): "
         f"HI={len(hi_cities)} cities, LO={len(lo_cities)} cities; "
         f"HI mean {gt.loc[gt['g']>med,'g'].mean():+.3f} vs "
         f"LO {gt.loc[gt['g']<=med,'g'].mean():+.3f}")

    d = df.copy()
    d["covered"] = d["city_code"].isin(set(g["city_code"]))
    d["T_hi"] = d["city_code"].isin(hi_cities).astype(int)
    d["T_lo"] = d["city_code"].isin(lo_cities).astype(int)
    gz = dict(zip(g["city_code"], (g["g"] - g["g"].mean()) / g["g"].std()))
    d["gz"] = d["city_code"].map(gz)

    # coverage-conditioned reduced form (uncovered treated cities dropped,
    # controls kept covered-only too for symmetry)
    dc = d[d["covered"]]
    emit(f"\ncovered subsample: {len(dc):,} woman-years "
         f"({dc['pid'].nunique():,} women)")
    fit(f"y ~ T:post + {CTRL}{FE}", dc, ["T:post"],
        "reduced form, covered cities only (benchmark)")

    # ---- HI vs LO response split ----------------------------------------
    m = fit(f"y ~ T_hi:post + T_lo:post + {CTRL}{FE}", dc,
            ["T_hi:post", "T_lo:post"],
            "split DID: HI- vs LO-price-response treated vs covered controls")
    tt = m.t_test("T_hi:post - T_lo:post")
    emit(f"  HI - LO difference: {float(np.ravel(tt.effect)[0]):+.4f} "
         f"(p={float(np.ravel(tt.pvalue)[0]):.3f})")

    for arm, lab in [(0, "daughter"), (1, "son")]:
        s = dc[dc["son"] == arm]
        m = fit(f"y ~ T_hi:post + T_lo:post + {CTRL}{FE}", s,
                ["T_hi:post", "T_lo:post"], f"split DID, {lab} arm")
        tt = m.t_test("T_hi:post - T_lo:post")
        emit(f"  HI - LO difference: {float(np.ravel(tt.effect)[0]):+.4f} "
             f"(p={float(np.ravel(tt.pvalue)[0]):.3f})")

    # ---- continuous dose among covered cities ---------------------------
    fit(f"y ~ T:post + T:post:gz + post:gz + {CTRL}{FE}", dc,
        ["T:post", "T:post:gz", "post:gz"],
        "continuous: T x post x realized price growth (gz)")
    da = dc[dc["son"] == 0]
    fit(f"y ~ T:post + T:post:gz + post:gz + {CTRL}{FE}", da,
        ["T:post", "T:post:gz", "post:gz"], "continuous dose, daughter arm")

    # ---- event studies for HI/LO treated (vs covered controls) ----------
    for grp, cset in [("HI", hi_cities), ("LO", lo_cities)]:
        s = dc[(dc["T"] == 0) | dc["city_code"].isin(cset)].copy()
        ev = [y for y in range(2010, 2023) if y != 2015]
        for k in ev:
            s[f"Ty{k}"] = s["T"] * (s["year"] == k).astype(int)
        terms = " + ".join(f"Ty{k}" for k in ev)
        m = smf.ols(f"y ~ {terms} + {CTRL}{FE}", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s["city"]})
        emit(f"\n[event study: {grp}-response treated vs covered controls]  "
             f"N={int(m.nobs):,}")
        for k in ev:
            tag = " (pre)" if k < 2016 else ""
            emit(f"  T x {k}   {m.params[f'Ty{k}']:+.4f}  "
                 f"(SE {m.bse[f'Ty{k}']:.4f}, p={m.pvalues[f'Ty{k}']:.3f}){tag}")

    # ---- who are the HI cities? -----------------------------------------
    prof = dc.drop_duplicates("pid")
    emit("\nHI vs LO treated-city composition (woman-level means):")
    for c in ["city_son_norm", "trad_pc1", "city_urban_share", "city_linc",
              "log_faminc_pc", "urban"]:
        hi = prof.loc[prof["T_hi"] == 1, c].mean()
        lo = prof.loc[prof["T_lo"] == 1, c].mean()
        emit(f"  {c:<18} HI {hi:8.3f}   LO {lo:8.3f}")

    out = os.path.join(OUT, "tables", "city_price_response.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
