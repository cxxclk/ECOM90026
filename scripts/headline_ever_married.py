# ==========================================================================
# REPLICATION PACKAGE — headline_ever_married.py
# Tier: CORE
# Paper items: Table 8 (Sample: only women reported married in some wave)
# Reads: output/panel_individual.csv
# Writes: output/tables/headline_ever_married.txt
# Imports local modules: annual_firstborn_gender, annual_parity_hazard, gender_norm_index, headline_robustness
# Run order: step 34 in run_all.sh
# ==========================================================================

"""Headline re-run with the ever-married restriction (audit of the §3.3 claim
"all samples consist of women who are or have been married").

annual_firstborn_gender.build_hazard_panel (the headline one-child panel) filters
only on age 20-45; annual_parity_hazard.woman_attrs additionally keeps women with
marital==2 in ANY wave. This script (1) decomposes the 40,108 vs 38,614 gap and
describes the excluded women, (2) re-runs the headline battery on the
ever-married sample (otherwise identical: city+year FE, age/dur quadratics,
cluster city, Post=2018, WCB from headline_robustness.py), (3) reconciles the
gender-pooled one-child estimate with Table 2B (-1.18pp, p=0.104).
"""
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")

from annual_firstborn_gender import build_hazard_panel, FAMCONF  # noqa: E402
from headline_robustness import wild_cluster_boot  # noqa: E402
import annual_parity_hazard as aph  # noqa: E402
from gender_norm_index import build_norm_index  # noqa: E402

CTRL = "age + I(age**2) + dur + I(dur**2)"
FE = " + C(city) + C(year)"
LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def ols(f, d):
    return smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["city"]})


# ----------------------------------------------------------------- flags
def marital_flags():
    p = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                    usecols=["pid", "year", "gender", "marital"])
    w = p[p["gender"] == 0]
    g = w.groupby("pid")["marital"]
    fl = pd.DataFrame({
        "em_strict": g.apply(lambda s: (s == 2).any()).astype(int),
        "em_broad": g.apply(lambda s: s.isin([2, 3, 4, 5]).any()).astype(int),
        "ever_div_wid": g.apply(lambda s: s.isin([4, 5]).any()).astype(int),
        "ever_cohab": g.apply(lambda s: (s == 3).any()).astype(int),
        "ever_never": g.apply(lambda s: (s == 1).any()).astype(int),
        "all_missing": g.apply(lambda s: (~s.isin([1, 2, 3, 4, 5])).all()).astype(int),
        "n_waves": g.size(),
    })
    return fl.reset_index()


def spouse_flag():
    """Any famconf wave lists a spouse pid (>0) for this person."""
    out = []
    for y, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        sc = [c for c in meta.column_names if re.fullmatch(r"pid_(a_)?s", c)][0]
        d, _ = pyreadstat.read_dta(path, usecols=["pid", sc])
        out.append(d.assign(has_sp=(d[sc] > 0).astype(int))[["pid", "has_sp"]])
    a = pd.concat(out)
    return a.groupby("pid")["has_sp"].max().rename("famconf_spouse").reset_index()


# ----------------------------------------------------------------- battery
def battery(df, tag, norm_city):
    R = {}
    df = df.copy()
    df["post"] = (df["year"] >= 2018).astype(int)
    dd, ds = df[df["son"] == 0], df[df["son"] == 1]
    R["N"] = (len(df), df["pid"].nunique(), df["y"].mean())
    R["Nd"] = (len(dd), dd["pid"].nunique(), dd["y"].mean())
    R["Ns"] = (len(ds), ds["pid"].nunique(), ds["y"].mean())
    f = f"y ~ T:post + {CTRL}{FE}"
    for k, d in [("daughter", dd), ("son", ds), ("pooled", df)]:
        m = ols(f, d)
        _, _, pw, G = wild_cluster_boot(d, f, "T:post")
        R[k] = (m.params["T:post"], m.pvalues["T:post"], pw, G)
    fddd = f"y ~ T:post + T:post:son + post:son + T:son + son + {CTRL}{FE}"
    m = ols(fddd, df)
    _, _, pw, _ = wild_cluster_boot(df, fddd, "T:post:son")
    R["ddd"] = (m.params["T:post:son"], m.pvalues["T:post:son"], pw)
    R["ddd_Tson"] = (m.params["T:son"], m.pvalues["T:son"])
    # split post
    df["boom"] = df["year"].isin([2016, 2017]).astype(int)
    df["pa"] = df["year"].isin([2018, 2019]).astype(int)
    df["pb"] = (df["year"] >= 2020).astype(int)
    for k, s in [("split_d", 0), ("split_s", 1)]:
        d = df[df["son"] == s]
        m = ols(f"y ~ T:boom + T:pa + T:pb + {CTRL}{FE}", d)
        R[k] = [(m.params[t], m.pvalues[t]) for t in ("T:boom", "T:pa", "T:pb")]
    # event study + joint pre test (2010-2014 vs 2015 base)
    ev = [y for y in range(2010, 2023) if y != 2015]
    for y in ev:
        df[f"Ty{y}"] = df["T"] * (df["year"] == y).astype(int)
    terms = " + ".join(f"Ty{y}" for y in ev)
    for k, s in [("es_d", 0), ("es_s", 1)]:
        d = df[df["son"] == s]
        m = ols(f"y ~ {terms} + {CTRL}{FE}", d)
        pre = [f"Ty{y}" for y in range(2010, 2015)]
        wt = m.wald_test(", ".join(f"{c} = 0" for c in pre), scalar=True)
        R[k] = ({y: (m.params[f"Ty{y}"], m.pvalues[f"Ty{y}"]) for y in ev},
                float(wt.pvalue))
    # top son_norm tercile, daughter arm (gender_norm_index spec)
    dn = df.merge(norm_city[["city_code", "son_norm"]], on="city_code", how="inner")
    cities_in = dn.drop_duplicates("city_code")
    c2 = cities_in["son_norm"].quantile(2 / 3)
    d = dn[(dn["son"] == 0) & (dn["son_norm"] > c2)]
    m = ols(f, d)
    _, _, pw, G = wild_cluster_boot(d, f, "T:post")
    base = d.loc[d["post"] == 0, "y"].mean()
    R["norm3"] = (m.params["T:post"], m.pvalues["T:post"], pw, G, len(d),
                  base, c2)
    return R


def main():
    emit("HEADLINE x EVER-MARRIED RESTRICTION  (scripts/headline_ever_married.py)")
    emit("=" * 76)
    base = build_hazard_panel()
    fl = marital_flags()
    sp = spouse_flag()
    base = base.merge(fl, on="pid", how="left").merge(sp, on="pid", how="left")
    for c in ["em_strict", "em_broad"]:
        base[c] = base[c].fillna(0).astype(int)

    # ---- (1) reconcile with annual_parity_hazard parity-1 set
    roster, last = aph.load_rosters()
    aph.LOAD_ROSTER, aph.LOAD_LAST = roster, (roster, last)
    attrs = aph.woman_attrs()
    r1 = aph.build_risk_set(attrs, 1)
    emit("\n[1] GAP DECOMPOSITION")
    emit(f"  headline panel (annual_firstborn_gender): {len(base):,} wy, "
         f"{base.pid.nunique():,} women")
    emit(f"  parity-1 set  (annual_parity_hazard):    {len(r1):,} wy, "
         f"{r1.pid.nunique():,} women")
    em = base[base["em_strict"] == 1]
    emit(f"  headline panel + marital==2-ever filter:  {len(em):,} wy, "
         f"{em.pid.nunique():,} women")
    k_b = set(zip(em.pid, em.year))
    k_r = set(zip(r1.pid, r1.year))
    emit(f"  rows in filtered headline not in parity-1: {len(k_b - k_r):,}; "
         f"rows in parity-1 not in filtered headline: {len(k_r - k_b):,}")
    extra = r1[[k not in k_b for k in zip(r1.pid, r1.year)]]
    if len(extra):
        # why: firstborn gender missing in roster?
        emit(f"    -> parity-1-only women: {extra.pid.nunique():,} "
             f"(expected: firstborn sex missing, dropped by fb_son filter)")
    broad = base[base["em_broad"] == 1]
    emit(f"  alt BROAD flag (marital in 2/3/4/5 any wave): {len(broad):,} wy, "
         f"{broad.pid.nunique():,} women")

    # ---- describe excluded women
    ex = base[base["em_strict"] == 0]
    w = ex.drop_duplicates("pid")
    wa = base.drop_duplicates("pid")
    emit("\n  EXCLUDED WOMEN (no marital==2 in any wave):")
    emit(f"    women {len(w):,} ({len(w)/len(wa):.1%} of headline women), "
         f"woman-years {len(ex):,} ({len(ex)/len(base):.1%})")
    emit(f"    daughter-first share {1 - w['son'].mean():.3f} "
         f"(retained {1 - wa.loc[wa.em_strict == 1, 'son'].mean():.3f})")
    emit(f"    second births in risk set: {int(ex['y'].sum())} "
         f"(of {int(base['y'].sum())}); hazard {ex['y'].mean():.4f} vs "
         f"retained {em['y'].mean():.4f}")
    emit(f"    daughter-arm woman-years {int((ex.son == 0).sum()):,}, "
         f"births {int(ex.loc[ex.son == 0, 'y'].sum())}; "
         f"treated share {w['T'].mean():.3f} (retained "
         f"{wa.loc[wa.em_strict == 1, 'T'].mean():.3f})")
    w = w.assign(age_fb=w["first_by"] - w["birthy"])
    emit(f"    birth year median {w['birthy'].median():.0f} "
         f"(retained {wa.loc[wa.em_strict == 1, 'birthy'].median():.0f}); "
         f"age at 1st birth median {w['age_fb'].median():.0f}, "
         f"share <20: {(w['age_fb'] < 20).mean():.3f}")
    emit(f"    mean age over woman-years {ex['age'].mean():.1f} "
         f"(retained {em['age'].mean():.1f})")
    emit("    marital status pattern (share of excluded women):")
    emit(f"      ever divorced/widowed (4/5): {w['ever_div_wid'].mean():.3f}")
    emit(f"      ever cohabiting (3):          {w['ever_cohab'].mean():.3f}")
    emit(f"      ever never-married (1):       {w['ever_never'].mean():.3f}")
    emit(f"      all waves missing/neg codes:  {w['all_missing'].mean():.3f}")
    emit(f"      not in adult panel at all:    {w['n_waves'].isna().mean():.3f}")
    emit(f"      famconf lists a spouse pid:   {w['famconf_spouse'].fillna(0).mean():.3f}")
    emit(f"    waves observed (adult panel) median {w['n_waves'].median()}")
    emit("    year of first birth distribution (excluded):")
    emit("      " + w["first_by"].describe()[["min", "25%", "50%", "75%", "max"]]
         .round(0).to_string().replace("\n", "; "))

    base["famconf_spouse"] = base["famconf_spouse"].fillna(0)
    sos = base[(base["em_strict"] == 1) | (base["famconf_spouse"] == 1)]
    emit(f"  alt STRICT-OR-SPOUSE flag: {len(sos):,} wy, {sos.pid.nunique():,} women")
    emit("  raw daughter-arm hazard of EXCLUDED women (births/wy): " + ", ".join(
        f"T{t} {'post' if q else 'pre'} {int(g.y.sum())}/{len(g)}={g.y.mean():.3f}"
        for (t, q), g in ex[ex.son == 0].assign(q=(ex.year >= 2018).astype(int))
        .groupby(["T", "q"])))
    emit(f"  excluded women: first birth >= 2016 share {(w['first_by'] >= 2016).mean():.3f}; "
         f"no marital 2/3/4/5 AND no famconf spouse: "
         f"{((w['em_broad'] == 0) & (w['famconf_spouse'].fillna(0) == 0)).sum()} women")

    # ---- (2)(3) battery
    norm_city = build_norm_index()
    samples = [("BASELINE (headline, no marital filter)", base),
               ("EVER-MARRIED (marital==2 any wave)", em),
               ("BROAD (marital 2/3/4/5 any wave)", broad),
               ("STRICT-OR-SPOUSE (marital==2 any wave OR famconf spouse pid)", sos)]
    RES = {}
    for tag, d in samples:
        print(f"... running {tag}")
        RES[tag] = battery(d.drop(columns=[c for c in d.columns
                                           if c.startswith("Ty")]), tag, norm_city)

    emit("\n[2] HEADLINE BATTERY, side by side (Post=2018, city+year FE, "
         "cluster city; WCB Rademacher B=999 null imposed)")
    for tag, R in RES.items():
        emit("\n" + "-" * 76)
        emit(tag)
        emit(f"  all one-child: N={R['N'][0]:,} women={R['N'][1]:,} mean={R['N'][2]:.4f}")
        emit(f"  daughter arm : N={R['Nd'][0]:,} women={R['Nd'][1]:,} mean={R['Nd'][2]:.4f}")
        emit(f"  son arm      : N={R['Ns'][0]:,} women={R['Ns'][1]:,} mean={R['Ns'][2]:.4f}")
        for k, lab in [("daughter", "DAUGHTER T:post"), ("son", "SON T:post"),
                       ("pooled", "POOLED one-child T:post")]:
            b, p, pw, G = R[k]
            emit(f"  {lab:<26} {b*100:+.2f}pp  cluster p={p:.3f}  WCB p={pw:.3f}  G={G}")
        b, p, pw = R["ddd"]
        emit(f"  {'DDD T:post:son':<26} {b*100:+.2f}pp  cluster p={p:.3f}  WCB p={pw:.3f}"
             f"   (T:son {R['ddd_Tson'][0]*100:+.2f}pp p={R['ddd_Tson'][1]:.3f})")
        for k, lab in [("split_d", "split DAUGHTER"), ("split_s", "split SON")]:
            s = R[k]
            emit(f"  {lab:<26} 16-17 {s[0][0]*100:+.2f} (p={s[0][1]:.3f}) | "
                 f"18-19 {s[1][0]*100:+.2f} (p={s[1][1]:.3f}) | "
                 f"20-22 {s[2][0]*100:+.2f} (p={s[2][1]:.3f})")
        for k, lab in [("es_d", "event DAUGHTER"), ("es_s", "event SON")]:
            es, jp = R[k]
            emit(f"  {lab:<26} joint pre 2010-14 p={jp:.3f}; pre: " +
                 ", ".join(f"{y} {es[y][0]*100:+.1f}" for y in range(2010, 2015)))
            emit(f"  {'':<26} post: " +
                 ", ".join(f"{y} {es[y][0]*100:+.1f}(p{es[y][1]:.2f})"
                           for y in range(2016, 2023)))
        b, p, pw, G, n, bm, c2 = R["norm3"]
        emit(f"  {'son_norm T3 DAUGHTER':<26} {b*100:+.2f}pp  cluster p={p:.3f}  "
             f"WCB p={pw:.3f}  G={G}  N={n:,}  pre-mean={bm:.3f} rel={b/bm:+.0%}"
             f"  (cut {c2:.3f})")

    emit("\n[3] RECONCILE WITH TABLE 2B (annual_parity_hazard parity-1, Post=2018)")
    r1["post"] = (r1["year"] >= 2018).astype(int)
    m = ols(f"y ~ T:post + {CTRL}{FE}", r1)
    emit(f"  parity-1 set as built there: {m.params['T:post']*100:+.2f}pp "
         f"p={m.pvalues['T:post']:.3f}  N={len(r1):,} women={r1.pid.nunique():,}")
    for tag, R in RES.items():
        b, p, pw, _ = R["pooled"]
        emit(f"  {tag:<42} pooled {b*100:+.2f}pp p={p:.3f} WCB {pw:.3f}")

    out = os.path.join(OUT, "tables", "headline_ever_married.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
