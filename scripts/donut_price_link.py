# ==========================================================================
# REPLICATION PACKAGE — donut_price_link.py
# Tier: CORE
# Paper items: Table 8 (Distance: comparison group ≥ 450 km only)
# Reads: output/housing_price_city.csv, output/hpr_treatment_city.csv
# Writes: output/tables/donut_price_link.txt
# Imports local modules: distance_ring_profile, headline_robustness
# Run order: step 32 in run_all.sh
# ==========================================================================

"""Donut / price-link test: is the headline second-birth suppression a house-PRICE
effect, or an effect of exposure to the HPR spillover independent of price?

Referee point: the 250-450 km rings (CONTROL in the headline 250 km design) had
2016-19 log price growth (0.31-0.37) like the exposed rings, yet zero fertility
effect; only >=450 km grew less. Design = headline one-child annual second-birth
hazard (distance_ring_profile.build_hazard_panel: unregulated prefectures, city+
year FE, age/dur quadratics, Post=2018, cluster city), with (1) a ring x year
price event study, (2) donut controls (>=450 only / 250-450 only), (3) price-
growth treatment (continuous, terciles, 2SLS with T instrumenting growth),
(4) horse race T x post vs growth x post, (5) ring-group composition.
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
import distance_ring_profile as drp                       # noqa: E402
from headline_robustness import wild_cluster_boot          # noqa: E402

CTRL = "age + I(age**2) + dur + I(dur**2)"
FE = " + C(city) + C(year)"
LINES = []


def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def fitc(f, d, grp="city"):
    return smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[grp]})


def line(m, k, lab, d=None):
    extra = f"  N={int(m.nobs):,} cities={d['city'].nunique()}" if d is not None else ""
    emit(f"  {lab:<34}{m.params[k]:+.4f} (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f}){extra}")


def grp3(dist):
    return np.where(dist <= 250, "E<=250", np.where(dist < 450, "C250-450", "C>=450"))


# ---------------------------------------------------------------- 1. prices
def price_event_study(sample_cities):
    emit("=" * 76)
    emit("1. FIRST STAGE BY RING: city x year log second-hand price, 2012-2022")
    emit("   log p_ct = city FE + year FE + sum_g sum_y G_g x 1[year=y] (base 2015,")
    emit("   ref group >=450 km), cluster city, unbalanced. Groups: E<=250 (exposed),")
    emit("   C250-450 (donut control), C>=450 (far control).")
    hp = pd.read_csv(os.path.join(OUT, "housing_price_city.csv"))
    hpr = pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
    hp = hp.merge(hpr, on="city_code").query("is_regulated == 0")
    hp = hp[hp["year"].between(2012, 2022)].dropna(subset=["log_hp_city"])
    hp["g"] = grp3(hp["dist_to_hpr_km"])
    for univ, lab in [("all", "ALL unregulated prefectures with price data"),
                      ("cfps", "CFPS analytic prefectures only")]:
        d = hp if univ == "all" else hp[hp["city_code"].isin(sample_cities)]
        d = d.copy()
        emit(f"\n[{lab}]")
        cov = d.pivot_table(index="g", columns="year", values="city_code",
                            aggfunc="nunique").fillna(0).astype(int)
        emit("  coverage (cities with a price) by group x year:")
        emit("  " + f"{'':<10}" + "".join(f"{y:>6}" for y in cov.columns))
        for g in ["E<=250", "C250-450", "C>=450"]:
            if g in cov.index:
                emit("  " + f"{g:<10}" + "".join(f"{v:>6}" for v in cov.loc[g]))
        yrs = [y for y in range(2012, 2023) if y != 2015]
        terms = []
        for g, gk in [("E<=250", "E"), ("C250-450", "D")]:
            for y in yrs:
                c = f"{gk}y{y}"
                d[c] = ((d["g"] == g) & (d["year"] == y)).astype(int)
                terms.append(c)
        d["city"] = d["city_code"]
        m = fitc(f"log_hp_city ~ {' + '.join(terms)} + C(city) + C(year)", d)
        for gk, gl in [("E", "E<=250 vs >=450"), ("D", "C250-450 vs >=450")]:
            emit(f"  {gl}: " + "  ".join(
                f"{y}:{m.params[f'{gk}y{y}']:+.3f}({m.pvalues[f'{gk}y{y}']:.2f})"
                for y in yrs))
        # difference E - D
        emit("  E<=250 vs C250-450 (diff): " + "  ".join(
            f"{y}:{m.params[f'Ey{y}'] - m.params[f'Dy{y}']:+.3f}"
            f"({float(m.t_test(f'Ey{y} - Dy{y} = 0').pvalue):.2f})" for y in yrs))
        # pre-trend relative to own 2012-14 and growth 2016->19 net of 2012->15 trend
        for gk, gl in [("E", "E<=250"), ("D", "C250-450")]:
            pre = [f"{gk}y{y}" for y in (2012, 2013, 2014)]
            ft = m.f_test(" = ".join(pre) + " = 0")
            emit(f"  joint pre(2012-14)=0 {gl}: F={float(ft.fvalue):.2f} p={float(ft.pvalue):.3f}")
        # simple growth table, balanced on 2016 & 2019 (and 2012 & 2015 where present)
        w = d.pivot_table(index="city_code", columns="year", values="log_hp_city")
        gtab = d.drop_duplicates("city_code").set_index("city_code")["g"]
        emit("  raw mean log growth by group (cities with both endpoints):")
        for (a, b) in [(2012, 2015), (2015, 2016), (2016, 2019), (2015, 2019), (2019, 2022)]:
            if a in w.columns and b in w.columns:
                gr = (w[b] - w[a]).dropna()
                s = gr.groupby(gtab.reindex(gr.index)).agg(["mean", "count"])
                emit(f"    {a}->{b}: " + "  ".join(
                    f"{g} {s.loc[g,'mean']:+.3f} (n={int(s.loc[g,'count'])})"
                    for g in ["E<=250", "C250-450", "C>=450"] if g in s.index))


# ---------------------------------------------------------------- 2. donut
def es_pretest(d, label):
    d = d.copy()
    yrs = [y for y in range(2010, 2023) if y != 2015]
    for y in yrs:
        d[f"Ty{y}"] = d["T"] * (d["year"] == y).astype(int)
    m = fitc(f"y ~ {' + '.join(f'Ty{y}' for y in yrs)} + {CTRL}{FE}", d)
    pre = [f"Ty{y}" for y in range(2010, 2015)]
    ft = m.f_test(" = ".join(pre) + " = 0")
    emit(f"    ES {label}: pre " + " ".join(
        f"{y}:{m.params[f'Ty{y}']:+.3f}({m.pvalues[f'Ty{y}']:.2f})" for y in range(2010, 2015))
        + f" | joint F={float(ft.fvalue):.2f} p={float(ft.pvalue):.3f}")
    emit(f"    {'':<{len(label)+4}}post " + " ".join(
        f"{y}:{m.params[f'Ty{y}']:+.3f}({m.pvalues[f'Ty{y}']:.2f})" for y in range(2016, 2023)))


def donut(df):
    emit("\n" + "=" * 76)
    emit("2. DONUT DIDs (exposed <=250 km always the treated group)")
    specs = [("FULL control (headline)", df),
             ("control = >=450 only (drop 250-450)", df[df["g"] != "C250-450"]),
             ("control = 250-450 only (drop >=450)", df[df["g"] != "C>=450"]),
             ("PLACEBO: 250-450 'treated' vs >=450", None)]
    for lab, d in specs:
        if d is None:
            d = df[df["g"] != "E<=250"].copy()
            d["T"] = (d["g"] == "C250-450").astype(int)
        emit(f"\n[{lab}]  cities T={d.loc[d['T']==1,'city'].nunique()} "
             f"C={d.loc[d['T']==0,'city'].nunique()}")
        da, so = d[d["son"] == 0], d[d["son"] == 1]
        md = fitc(f"y ~ T:post + {CTRL}{FE}", da)
        line(md, "T:post", "daughter T:post", da)
        f = f"y ~ T:post + {CTRL}{FE}"
        b0, t0, pw, G = wild_cluster_boot(da, f, "T:post", reps=999)
        emit(f"  {'':<34}WCB p={pw:.3f} (G={G})")
        ms = fitc(f"y ~ T:post + {CTRL}{FE}", so)
        line(ms, "T:post", "son T:post (placebo)", so)
        mddd = fitc(f"y ~ T:post + T:post:son + post:son + T:son + son + {CTRL}{FE}", d)
        line(mddd, "T:post:son", "DDD T:post:son")
        es_pretest(da, "daughter")
        es_pretest(so, "son")


# ---------------------------------------------------------------- 3. price treatment
def price_treatment(df):
    emit("\n" + "=" * 76)
    emit("3. PRICE-GROWTH TREATMENT (city 2016->2019 log growth), covered prefectures")
    c = df.dropna(subset=["pgrowth"]).copy()
    emit(f"   covered: cities={c['city'].nunique()} of {df['city'].nunique()}, "
         f"woman-years {len(c):,} of {len(df):,}")
    cc = c.drop_duplicates("city").set_index("city")
    emit("   covered cities by group: " + ", ".join(
        f"{g} {int((cc['g']==g).sum())} (mean g {cc.loc[cc['g']==g,'pgrowth'].mean():.3f})"
        for g in ["E<=250", "C250-450", "C>=450"]))
    sd = cc["pgrowth"].std()
    c["pgz"] = (c["pgrowth"] - cc["pgrowth"].mean()) / sd
    c["pgp"] = c["pgz"] * c["post"]
    q = cc["pgrowth"].quantile([1 / 3, 2 / 3]).values
    c["tq"] = np.digitize(c["pgrowth"], q)  # 0,1,2
    for k in (1, 2):
        c[f"q{k}p"] = (c["tq"] == k).astype(int) * c["post"]
    emit(f"   pgrowth sd across cities={sd:.3f}; tercile cuts {q[0]:.3f}/{q[1]:.3f}")
    tcomp = cc.assign(tq=np.digitize(cc["pgrowth"], q)).groupby("tq").agg(
        n=("pgrowth", "size"), g=("pgrowth", "mean"), shareT=("T", "mean"))
    emit("   terciles: " + "; ".join(
        f"Q{k+1}: n={int(r.n)} mean g={r.g:.3f} share exposed={r.shareT:.2f}"
        for k, r in tcomp.iterrows()))
    from linearmodels.iv import IV2SLS
    for arm, lab in [(0, "DAUGHTER"), (1, "SON")]:
        d = c[c["son"] == arm].copy()
        emit(f"\n  [{lab} arm]")
        m0 = fitc(f"y ~ T:post + {CTRL}{FE}", d)
        line(m0, "T:post", "binary T:post (covered subset)", d)
        m1 = fitc(f"y ~ pgp + {CTRL}{FE}", d)
        line(m1, "pgp", "pgrowth(z) x post (continuous)")
        m2 = fitc(f"y ~ q1p + q2p + {CTRL}{FE}", d)
        line(m2, "q1p", "tercile Q2 x post (ref Q1 low)")
        line(m2, "q2p", "tercile Q3 x post (ref Q1 low)")
        # 2SLS: pgp instrumented by T x post
        d["Tp"] = d["T"] * d["post"]
        fs = fitc(f"pgp ~ Tp + {CTRL}{FE}", d)
        F = (fs.params["Tp"] / fs.bse["Tp"]) ** 2
        emit(f"  first stage pgp ~ T x post: {fs.params['Tp']:+.3f} sd-units "
             f"(p={fs.pvalues['Tp']:.4f}), cluster-robust F={F:.1f}")
        iv = IV2SLS.from_formula(
            f"y ~ 1 + {CTRL} + C(city) + C(year) + [pgp ~ Tp]", d).fit(
            cov_type="clustered", clusters=d["city"])
        emit(f"  2SLS pgrowth(z) x post: {iv.params['pgp']:+.4f} "
             f"(SE {iv.std_errors['pgp']:.4f}, p={iv.pvalues['pgp']:.3f})  "
             f"[= RF/FS; per +1 sd price growth]")

    # 3b. over-identified 2SLS: E<=250 x post and C250-450 x post (vs >=450) both
    # instrument price growth. Price-only exclusion => both reduced forms scale
    # with their first stages => Hansen J ~ 0. Rejection = exposure acts beyond price.
    from linearmodels.iv import IVGMM
    emit("\n  3b. OVER-IDENTIFIED test (instruments: E<=250 x post, C250-450 x post; ref >=450)")
    for arm, lab in [(0, "DAUGHTER"), (1, "SON")]:
        d = c[c["son"] == arm].copy()
        d["Ep"] = (d["g"] == "E<=250").astype(int) * d["post"]
        d["Dp"] = (d["g"] == "C250-450").astype(int) * d["post"]
        fs = fitc(f"pgp ~ Ep + Dp + {CTRL}{FE}", d)
        rf = fitc(f"y ~ Ep + Dp + {CTRL}{FE}", d)
        Fj = fs.f_test("Ep = 0, Dp = 0")
        emit(f"  [{lab}] FS: Ep {fs.params['Ep']:+.3f}(p={fs.pvalues['Ep']:.3f})  "
             f"Dp {fs.params['Dp']:+.3f}(p={fs.pvalues['Dp']:.3f})  joint F={float(Fj.fvalue):.1f}")
        emit(f"  [{lab}] RF: Ep {rf.params['Ep']:+.4f}(p={rf.pvalues['Ep']:.3f})  "
             f"Dp {rf.params['Dp']:+.4f}(p={rf.pvalues['Dp']:.3f})")
        emit(f"  [{lab}] implied Wald ratios RF/FS: E {rf.params['Ep']/fs.params['Ep']:+.4f}  "
             f"D {rf.params['Dp']/fs.params['Dp']:+.4f}  (per +1 sd price growth)")
        g = IVGMM.from_formula(f"y ~ 1 + {CTRL} + C(city) + C(year) + [pgp ~ Ep + Dp]", d).fit(
            cov_type="clustered", clusters=d["city"])
        js = g.j_stat
        emit(f"  [{lab}] GMM pgp {g.params['pgp']:+.4f} (p={g.pvalues['pgp']:.3f}); "
             f"Hansen J={float(js.stat):.2f} p={float(js.pval):.3f}")
        # price-only prediction for the donut ring
        pred = rf.params['Ep'] / fs.params['Ep'] * fs.params['Dp']
        emit(f"  [{lab}] price-only predicted donut RF = (RF_E/FS_E) x FS_D = {pred:+.4f}  "
             f"vs observed {rf.params['Dp']:+.4f} (SE {rf.bse['Dp']:.4f})")
    # city-level first stage F
    fs_c = smf.ols("pgrowth ~ T", data=cc.reset_index()).fit(cov_type="HC1")
    emit(f"\n  city-level first stage pgrowth ~ T (covered CFPS cities): "
         f"{fs_c.params['T']:+.3f} (p={fs_c.pvalues['T']:.4f}), F={fs_c.tvalues['T']**2:.1f}")
    return c


# ---------------------------------------------------------------- 4. horse race
def horse_race(df, c):
    emit("\n" + "=" * 76)
    emit("4. HORSE RACE T x post vs pgrowth x post")
    for arm, lab in [(0, "DAUGHTER"), (1, "SON")]:
        d = c[c["son"] == arm]
        m = fitc(f"y ~ T:post + pgp + {CTRL}{FE}", d)
        emit(f"\n  [{lab}, covered subset]  N={int(m.nobs):,}")
        line(m, "T:post", "T x post")
        line(m, "pgp", "pgrowth(z) x post")
    # full sample with missing-indicator
    f = df.copy()
    cc = c.drop_duplicates("city")
    mu, sd = cc["pgrowth"].mean(), cc["pgrowth"].std()
    f["miss"] = f["pgrowth"].isna().astype(int)
    f["pgp"] = ((f["pgrowth"] - mu) / sd).fillna(0) * f["post"]
    f["missp"] = f["miss"] * f["post"]
    d = f[f["son"] == 0]
    m = fitc(f"y ~ T:post + pgp + missp + {CTRL}{FE}", d)
    emit(f"\n  [DAUGHTER, FULL sample, missing growth=0 + miss x post]  N={int(m.nobs):,}")
    line(m, "T:post", "T x post")
    line(m, "pgp", "pgrowth(z) x post")
    line(m, "missp", "missing-price x post")
    # within-exposed: does price growth grade the effect?
    for g in ["E<=250", "C250-450"]:
        dd = c[(c["son"] == 0)]
        dd = dd[(dd["g"] == g) | (dd["g"] == "C>=450")].copy()
        dd["G"] = (dd["g"] == g).astype(int)
        dd["Gp"] = dd["G"] * dd["post"]
        dd["Gpg"] = dd["Gp"] * dd["pgz"]
        m = fitc(f"y ~ Gp + Gpg + pgp + {CTRL}{FE}", dd)
        emit(f"\n  [DAUGHTER, {g} vs >=450, covered]  N={int(m.nobs):,}")
        line(m, "Gp", f"{g} x post (at mean growth)")
        line(m, "Gpg", f"{g} x post x pgrowth(z)")
        line(m, "pgp", "pgrowth(z) x post (in >=450)")


# ---------------------------------------------------------------- 5. composition
def composition(df):
    emit("\n" + "=" * 76)
    emit("5. COMPOSITION by distance group (one-child risk set)")
    emit(f"  {'group':<10}{'cities':>7}{'women':>7}{'wyrs':>8}{'rural':>7}{'norm':>7}"
         f"{'pg16-19':>8}{'pre hzD':>8}{'pre hzS':>8}{'D-S pre':>8}{'D-S post':>9}")
    for g in ["E<=250", "C250-450", "C>=450"]:
        s = df[df["g"] == g]
        pre, post = s[s["year"] < 2018], s[s["year"] >= 2018]
        hd = pre.loc[pre.son == 0, "y"].mean(); hs = pre.loc[pre.son == 1, "y"].mean()
        pd_ = post.loc[post.son == 0, "y"].mean() - post.loc[post.son == 1, "y"].mean()
        cityl = s.drop_duplicates("city")
        emit(f"  {g:<10}{s['city'].nunique():>7}{s['pid'].nunique():>7}{len(s):>8,}"
             f"{1 - s['urban0'].mean():>7.3f}{cityl['son_norm'].mean():>7.3f}"
             f"{cityl['pgrowth'].mean():>8.3f}{hd:>8.4f}{hs:>8.4f}{hd - hs:>+8.4f}{pd_:>+9.4f}")
    emit("  (norm/pgrowth = city means, unweighted; hazards woman-year means; pre = 2009-17)")


def main():
    df = drp.build_hazard_panel()
    df = df.merge(drp.build_norm_index(), left_on="city_code", right_index=True, how="left")
    df = df.merge(drp.city_price_growth(), left_on="city_code", right_index=True, how="left")
    df["g"] = grp3(df["dist_to_hpr_km"])
    emit("DONUT / PRICE-LINK TEST of the headline second-birth suppression")
    emit(f"Panel rows={len(df):,} women={df['pid'].nunique():,} cities={df['city'].nunique()}")
    price_event_study(df["city_code"].unique())
    composition(df)
    donut(df)
    c = price_treatment(df)
    horse_race(df, c)
    out = os.path.join(OUT, "tables", "donut_price_link.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print("Saved ->", out)


if __name__ == "__main__":
    main()
