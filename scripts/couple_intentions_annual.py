# ==========================================================================
# REPLICATION PACKAGE — couple_intentions_annual.py
# Tier: CORE
# Paper items: Table 6 (B. Whose ideal governs the next birth)
# Reads: CFPS .dta files in data/20xx/, output/couples_panel.csv
# Writes: output/tables/couple_intentions_annual.txt
# Imports local modules: none
# Run order: step 26 in run_all.sh
# ==========================================================================

"""Couple decision-weight, ANNUALIZED (powering up couple_intentions_pooled.py).

Yesterday's winner (wave-level binary outcome): under HPR housing pressure the
realized birth stops tracking only the HUSBAND's stated ideal and the WIFE's
ideal becomes equally decisive (T:ideal_w = +0.056, p=0.003). The outcome there
was one binary row per couple (any birth after the intention wave through 2022).

This script replaces that with a discrete-time hazard of the NEXT birth at the
couple-year level, the same migration we just did for the fertility layers:
  * Each couple observed at its intention wave w in {2014, 2018} (both spouses
    state ideal kids: qm501 2014 / qka202 2018) is expanded to couple-years
    t in (w, min(next_birth, 2022)].
  * y_it = 1{the woman's next birth is in year t} (from the famconf roster, so
    immune to the wave-level roster-lag underreporting). Couple exits at first
    subsequent birth.
  * Adds YEAR FE (absorbs the aggregate post-2016 fertility decline that the
    wave-level pooled spec could only proxy with a wave dummy) + time-varying
    age + duration-since-intention, on top of province FE + city clustering.

Decision-weight test is unchanged in spirit: T x ideal_w vs T x ideal_h. The
gain is power (couple-years, not one row/couple) and clean aggregate-trend
control. Province FE (city FE would absorb the city-level HPR treatment).
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
LINES = []
def emit(s=""):
    print(s); LINES.append(s)

FAMCONF = {
    2010: "2010/ecfps2010famconf_nat072016.dta",
    2012: "2012/ecfps2012famconf_092015.dta",
    2014: "2014/ecfps2014famconf_170630.dta",
    2016: "2016/ecfps2016famconf_201804.dta",
    2018: "2018/ecfps2018famconf_202008.dta",
    2020: "2020/ecfps2020famconf_202306.dta",
    2022: "2022/ecfps2022famconf_202410.dta",
}
IDEAL = {2014: ("data/2014/ecfps2014adult_201906.dta", "qm501"),
         2018: ("data/2018/ecfps2018person_202012.dta", "qka202")}
Y_END = 2022


def birth_years_by_pid():
    frames = []
    for year, rel in FAMCONF.items():
        _, meta = pyreadstat.read_dta(os.path.join(DATA, rel), metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        df, _ = pyreadstat.read_dta(os.path.join(DATA, rel), usecols=["pid"] + by_cols)
        long = df.melt(id_vars="pid", value_name="cby")[["pid", "cby"]]
        long = long[(long["cby"] > 1900) & (long["cby"] <= year)]
        long["rw"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["rw"].transform("max")
    return allw[allw["rw"] == last][["pid", "cby"]]


def load_ideal(path, var):
    d, _ = pyreadstat.read_dta(path, usecols=["pid", var])
    d[var] = pd.to_numeric(d[var], errors="coerce")
    d.loc[(d[var] < 0) | (d[var] > 8), var] = np.nan
    return d.rename(columns={var: "ideal"})[["pid", "ideal"]]


def build_wave(wave, by):
    """One row per couple at the intention wave, with both ideals + parity +
    next-birth year. Mirrors couple_intentions_pooled.build_wave, minus the
    pre-collapsed Y (we keep next_birth for the annual expansion)."""
    cp = pd.read_csv(os.path.join(OUT, "couples_panel.csv"))
    cp = cp[(cp["year"] == wave) & cp["husband_pid"].notna() &
            (cp["is_regulated"] == 0) & cp["treat_hpr"].notna() &
            cp["city_code"].notna() & cp["hpr_exposure"].notna()].copy()
    cp["pid"] = cp["pid"].astype(int)
    cp["husband_pid"] = cp["husband_pid"].astype(int)

    path, var = IDEAL[wave]
    ideal = load_ideal(path, var)
    cp = cp.merge(ideal.rename(columns={"ideal": "ideal_w"}), on="pid", how="left")
    cp = cp.merge(ideal.rename(columns={"pid": "husband_pid", "ideal": "ideal_h"}),
                  on="husband_pid", how="left")
    cp = cp.dropna(subset=["ideal_w", "ideal_h"])

    par = by[by["cby"] <= wave].groupby("pid").size().rename("parity")
    nxt = (by[(by["cby"] > wave) & (by["cby"] <= Y_END)]
           .groupby("pid")["cby"].min().rename("next_by"))
    cp = cp.merge(par, left_on="pid", right_index=True, how="left")
    cp = cp.merge(nxt, left_on="pid", right_index=True, how="left")
    cp["parity"] = cp["parity"].fillna(0)
    cp["wave"] = wave
    return cp


def build_annual_panel(waves=(2014, 2018)):
    """Couple-year discrete-time next-birth hazard panel. Returns (cp, d):
    cp = one row per couple at its intention wave; d = expanded couple-years
    with outcome y, treatment, time-varying age/duration. Reused by the
    mechanism script so both share the exact same panel construction."""
    by = birth_years_by_pid()
    cp = pd.concat([build_wave(w, by) for w in waves], ignore_index=True)
    cp = cp.dropna(subset=["age", "eduy", "log_income"]).reset_index(drop=True)
    cp["cid"] = cp.index

    # annual expansion: couple-year rows from wave+1 to min(next_birth, 2022)
    rows = []
    grid = np.arange(2010, Y_END + 1)
    for r in cp.itertuples():
        end = int(r.next_by) if not np.isnan(r.next_by) else Y_END
        yrs = [t for t in grid if (t > r.wave) and (t <= end)]
        for t in yrs:
            rows.append((r.cid, t, int(t == end and not np.isnan(r.next_by))))
    py = pd.DataFrame(rows, columns=["cid", "yr", "y"])
    d = py.merge(cp, on="cid", how="left")

    d["T"] = d["treat_hpr"].astype(int)
    d["city"] = d["city_code"].astype(int)
    d["exp"] = d["hpr_exposure"].astype(float)
    d["expz"] = (d["exp"] - d["exp"].mean()) / d["exp"].std()
    d["age_t"] = d["age"] + (d["yr"] - d["wave"])
    d["dur"] = d["yr"] - d["wave"]
    d["gap"] = d["ideal_h"] - d["ideal_w"]
    return cp, d


def main():
    cp, d = build_annual_panel()

    emit(f"Annual couple-year hazard panel (intention waves 2014+2018)")
    emit(f"  couples={cp['cid'].nunique():,}  couple-years={len(d):,}  "
         f"cities={d['city'].nunique()}  treated={d['T'].mean():.2f}")
    emit(f"  next-birth hazard mean={d['y'].mean():.4f}  "
         f"(wave-level any-birth rate was 0.140)")
    emit(f"  mean ideal_w={d['ideal_w'].mean():.2f}  ideal_h={d['ideal_h'].mean():.2f}")

    CTRL = "age_t + I(age_t**2) + dur + eduy + log_income + C(parity)"
    FE = " + C(provcd) + C(yr)"

    emit("\n" + "=" * 72)
    emit("(A1) Decision-weight test, ANNUAL hazard (binary T) + year FE")
    fit(f"y ~ T*ideal_w + T*ideal_h + {CTRL}{FE}", d,
        "y ~ T x (ideal_w, ideal_h)",
        keys=["ideal_w", "ideal_h", "T:ideal_w", "T:ideal_h", "T"])

    emit("\n(A2) Same, CONTINUOUS dose (hpr_exposure, std)")
    fit(f"y ~ expz*ideal_w + expz*ideal_h + {CTRL}{FE}", d,
        "y ~ exposure x (ideal_w, ideal_h)",
        keys=["ideal_w", "ideal_h", "expz:ideal_w", "expz:ideal_h", "expz"])

    emit("\n(A3) Implied marginal effect of each spouse's ideal on annual hazard")
    m = smf.ols(f"y ~ T*ideal_w + T*ideal_h + {CTRL}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    for who, base, inter in [("w", "ideal_w", "T:ideal_w"),
                             ("h", "ideal_h", "T:ideal_h")]:
        c0 = m.params[base]; c1 = m.params.get(inter, 0)
        emit(f"   d hazard / d ideal_{who}:  control={c0:+.4f}   treated={c0+c1:+.4f}")

    emit("\n(A4) Robustness: cluster by COUPLE instead of city (within-couple "
         "serial correlation from the annual expansion)")
    m2 = smf.ols(f"y ~ T*ideal_w + T*ideal_h + {CTRL}{FE}", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["cid"]})
    for k in ["T:ideal_w", "T:ideal_h"]:
        emit(f"  {k:<14} {m2.params[k]:+.4f}  (SE {m2.bse[k]:.4f}, p={m2.pvalues[k]:.3f})")

    emit("\n(A5) Timing: does the wife-weight shift switch on AFTER the shock?")
    emit("    post = year>=2018; T:ideal_w:post is the diff-in-diff-in-ideal")
    d["post"] = (d["yr"] >= 2018).astype(int)
    fit(f"y ~ T*ideal_w*post + T*ideal_h*post + {CTRL}{FE}", d,
        "triple: T x ideal x post",
        keys=["T:ideal_w", "T:ideal_w:post", "T:ideal_h", "T:ideal_h:post"])

    # ---- (A6) Causal DID-in-weight: 2014 cohort, pre(2015) vs post(2016+) ----
    # A5's triple is confounded because the 2018 cohort is entirely post-shock
    # (post == cohort). Restricting to the 2014 cohort gives a genuine pre/post
    # contrast: ideal stated in 2014 (pre-shock), births observed 2015 (pre) vs
    # 2016-2022 (post). T:post:ideal_w is then the difference-in-differences in
    # the wife's decision weight, with the pre-shock level diff (T:ideal_w) as a
    # within-design placebo. This is the identification the angle rests on.
    emit("\n" + "=" * 72)
    emit("(A6) CAUSAL DID-in-weight — 2014 cohort only, post=year>=2016")
    emit("    T:ideal_w      = pre-shock treated-vs-control level diff (placebo)")
    emit("    T:post:ideal_w = shock-induced shift in the wife's decision weight")
    d14 = d[d["wave"] == 2014].copy()
    d14["post"] = (d14["yr"] >= 2016).astype(int)
    fit(f"y ~ T*post*ideal_w + T*post*ideal_h + {CTRL}{FE}", d14,
        "2014-cohort triple DID", keys=[
            "T:ideal_w", "T:post:ideal_w", "T:ideal_h", "T:post:ideal_h", "T:post"])
    emit("\n  Split-sample read of the same contrast:")
    for lab, sub in [("PRE  (yr==2015)", d14[d14["yr"] == 2015]),
                     ("POST (yr>=2016)", d14[d14["yr"] >= 2016])]:
        fe6 = " + C(provcd)" + (" + C(yr)" if sub["yr"].nunique() > 1 else "")
        mm = smf.ols(f"y ~ T*ideal_w + T*ideal_h + {CTRL}{fe6}", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["city"]})
        emit(f"  [{lab}] N={int(mm.nobs):,} births={int(sub['y'].sum())}  "
             f"T:ideal_w={mm.params['T:ideal_w']:+.4f} (p={mm.pvalues['T:ideal_w']:.3f})  "
             f"T:ideal_h={mm.params['T:ideal_h']:+.4f} (p={mm.pvalues['T:ideal_h']:.3f})")

    out = os.path.join(OUT, "tables", "couple_intentions_annual.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<18} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


if __name__ == "__main__":
    main()
