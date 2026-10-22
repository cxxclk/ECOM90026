# ==========================================================================
# REPLICATION PACKAGE — intentions_did.py
# Tier: SUPPLEMENTARY
# Paper items: Section 6.1 (stated ideals do not change)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/intentions_did.txt
# Imports local modules: none
# Run order: step 53 in run_all.sh
# ==========================================================================

"""Analysis C: intentions vs realizations (tier-1 essay gap).

Claim tested: the HPR price shock suppressed REALIZED second births of
firstborn-daughter one-child families (annual hazard -2.4pp, p=0.024,
annual_firstborn_gender.py) but did NOT move their STATED fertility ideals
=> binding constraints / unmet fertility demand, not preference change.

Data landmine handled: "ideal number of kids" is qm501 in the 2014 adult
file ONLY (in 2010/16/18 the same name = "importance of money"); 2018/2022
person files use qka202. Loading machinery follows youth_ideal_scarring.py
(label-asserted). Supplementary: qka205 (2022 only) = "want to have a child
in the following two years" -- 2022 T-vs-C cross-section only.

Design: wave-level DID on stated ideals. Married women 20-45, unregulated
prefectures (regulated source cities dropped), baseline-city treatment
treat_hpr, city + year FE, age controls, cluster by city. Waves: 2014 pre,
2018 + 2022 post. Groups: (a) all married women; (b) one-child mothers
(parity=1 as of the wave, latest-roster construction as in
annual_firstborn_gender.py); (c) firstborn-DAUGHTER one-child mothers
(the headline margin); (d) firstborn-SON one-child mothers.

Outcomes: ideal #kids (values kept 0-6, >6 dropped and reported) and
P(ideal>=2) (top-coding-insensitive). WCB for the headline group (c).
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from patsy import dmatrices

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

IDEAL_SRC = {
    2014: ("2014/ecfps2014adult_201906.dta", "qm501"),
    2018: ("2018/ecfps2018person_202012.dta", "qka202"),
    2022: ("2022/ecfps2022person_202410.dta", "qka202"),
}

FAMCONF = {
    2010: "2010/ecfps2010famconf_nat072016.dta",
    2012: "2012/ecfps2012famconf_092015.dta",
    2014: "2014/ecfps2014famconf_170630.dta",
    2016: "2016/ecfps2016famconf_201804.dta",
    2018: "2018/ecfps2018famconf_202008.dta",
    2020: "2020/ecfps2020famconf_202306.dta",
    2022: "2022/ecfps2022famconf_202410.dta",
}

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


# ------------------------------------------------------------------ loaders
def load_ideal():
    """Ideal number of kids, label-verified per wave (qm501 collision)."""
    frames = []
    emit("Ideal-kids sources (label-verified):")
    for year, (rel, var) in IDEAL_SRC.items():
        df, meta = pyreadstat.read_dta(os.path.join(DATA, rel),
                                       usecols=["pid", var])
        lab = str(meta.column_names_to_labels.get(var, ""))
        assert ("ideal" in lab.lower()) and ("kid" in lab.lower()
               or "child" in lab.lower()), f"label check failed {year} {var}"
        df = df.rename(columns={var: "ideal_raw"})
        df["ideal_raw"] = pd.to_numeric(df["ideal_raw"], errors="coerce")
        df.loc[df["ideal_raw"] < 0, "ideal_raw"] = np.nan  # -1/-2/-8 codes
        n_big = int((df["ideal_raw"] > 6).sum())
        df["ideal"] = df["ideal_raw"].where(df["ideal_raw"] <= 6)
        emit(f"  {year}: {var} = '{lab}'  valid N={df['ideal'].notna().sum():,}"
             f"  (dropped {n_big} answers >6)")
        df["year"] = year
        frames.append(df[["pid", "year", "ideal"]])
    return pd.concat(frames, ignore_index=True)


def load_want2yr():
    """qka205 (2022 only): want to have a child in the following two years."""
    df, meta = pyreadstat.read_dta(
        os.path.join(DATA, "2022/ecfps2022person_202410.dta"),
        usecols=["pid", "qka205"])
    lab = str(meta.column_names_to_labels.get("qka205", ""))
    assert "child" in lab.lower(), "qka205 label check failed"
    df["want2yr"] = pd.to_numeric(df["qka205"], errors="coerce")
    df["want2yr"] = df["want2yr"].where(df["want2yr"].isin([0, 1]))
    emit(f"  2022: qka205 = '{lab}'  valid N={df['want2yr'].notna().sum():,}"
         f"  (skip-pattern -8 excluded)")
    return df[["pid", "want2yr"]]


def load_rosters():
    """Per pid from the LATEST famconf wave: (child birth year, gender).
    Same construction as annual_firstborn_gender.py."""
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        by_cols = [c for c in meta.column_names
                   if re.fullmatch(r"tb1y_a_c\d+", c)]
        g_cols = [c.replace("tb1y", "tb2") for c in by_cols
                  if c.replace("tb1y", "tb2") in meta.column_names]
        df, _ = pyreadstat.read_dta(path, usecols=["pid"] + by_cols + g_cols)
        recs = []
        for i, byc in enumerate(by_cols, start=1):
            gc = f"tb2_a_c{i}"
            sub = df[["pid", byc] + ([gc] if gc in df.columns else [])].copy()
            sub.columns = ["pid", "cby"] + (["cgender"] if gc in df.columns
                                            else [])
            if "cgender" not in sub.columns:
                sub["cgender"] = np.nan
            recs.append(sub)
        long = pd.concat(recs, ignore_index=True)
        long = long[(long["cby"] > 1900) & (long["cby"] <= year)]
        long["cgender"] = long["cgender"].where(long["cgender"].isin([0, 1, 5]))
        long["cgender"] = long["cgender"].replace({5: 0})  # 5 = female
        long["roster_wave"] = year
        frames.append(long)
    allw = pd.concat(frames, ignore_index=True)
    last = allw.groupby("pid")["roster_wave"].max().rename("last_wave")
    allw = allw.merge(last, on="pid")
    roster = allw[allw["roster_wave"] == allw["last_wave"]]
    return roster[["pid", "cby", "cgender"]]


# ------------------------------------------------------------------ build
def build():
    panel = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "gender", "birthy", "marital",
                                 "urban", "city_code", "provcd"])
    women = panel[panel["gender"] == 0]
    base = (women.dropna(subset=["city_code"]).sort_values(["pid", "year"])
            .groupby("pid")["city_code"].first().rename("base_city"))
    women = women.merge(base, on="pid", how="left")

    ideal = load_ideal()
    df = women.merge(ideal, on=["pid", "year"], how="inner")
    df = df.merge(load_want2yr(), on="pid", how="left")
    df.loc[df["year"] != 2022, "want2yr"] = np.nan

    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 20) & (df["age"] <= 45)]
    df = df[df["marital"] == 2]                      # currently married

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    df = df.merge(hpr[["city_code", "is_regulated", "treat_hpr"]],
                  left_on="base_city", right_on="city_code",
                  how="inner", suffixes=("", "_h"))
    df = df[(df["is_regulated"] == 0) & df["treat_hpr"].notna()]
    df["T"] = df["treat_hpr"].astype(int)
    df["post"] = (df["year"] >= 2018).astype(int)
    df["city"] = df["base_city"].astype(int)

    # parity + firstborn gender AS OF each wave (latest-roster children)
    roster = load_rosters().sort_values(["pid", "cby"])
    kids = df[["pid", "year"]].drop_duplicates().merge(roster, on="pid",
                                                       how="left")
    kids = kids[kids["cby"] <= kids["year"]]
    par = kids.groupby(["pid", "year"]).agg(
        parity=("cby", "size"), first_by=("cby", "min")).reset_index()
    fb = kids.sort_values("cby").groupby(["pid", "year"]).first().reset_index()
    par = par.merge(fb[["pid", "year", "cgender"]].rename(
        columns={"cgender": "fb_son"}), on=["pid", "year"])
    df = df.merge(par, on=["pid", "year"], how="left")
    df["parity"] = df["parity"].fillna(0).astype(int)

    df = df.dropna(subset=["ideal"])
    df["ideal2"] = (df["ideal"] >= 2).astype(float)
    return df


# ------------------------------------------------------------------ fitting
CTRL = "age + I(age**2)"
FE = " + C(city) + C(year)"


def fit(formula, d, keys=("T:post",)):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    return m


def wild_cluster_boot(d, formula, key="T:post", reps=999, seed=90026):
    y_full, X_full = dmatrices(formula, d, return_type="dataframe")
    y = y_full.values.ravel()
    x = X_full[key].values
    Z = X_full.drop(columns=key).values
    cl = d["city"].values
    Q, _ = np.linalg.qr(Z)
    resid = lambda v: v - Q @ (Q.T @ v)
    xt = resid(x)
    u_r = resid(y)
    y_hat_r = y - u_r
    cities = np.unique(cl)
    G = len(cities)
    idx = [np.where(cl == c)[0] for c in cities]

    def tstat(yv):
        yt = resid(yv)
        b = xt @ yt / (xt @ xt)
        e = yt - xt * b
        meat = sum((xt[i] @ e[i]) ** 2 for i in idx)
        se = np.sqrt(meat * G / (G - 1)) / (xt @ xt)
        return b, b / se

    b0, t0 = tstat(y)
    rng = np.random.default_rng(seed)
    ts = np.empty(reps)
    for r in range(reps):
        w = rng.choice([-1.0, 1.0], size=G)
        yb = y_hat_r.copy()
        for gi, i in enumerate(idx):
            yb[i] += w[gi] * u_r[i]
        ts[r] = tstat(yb)[1]
    p = (np.sum(np.abs(ts) >= abs(t0)) + 1) / (reps + 1)
    return b0, t0, p, G


def report_group(d, name, wcb=False):
    emit("\n" + "=" * 74)
    emit(f"GROUP: {name}")
    wave_n = d.groupby("year").size()
    wave_t = d.groupby(["year", "T"]).size().unstack(fill_value=0)
    emit(f"  N by wave: " + ", ".join(f"{y}: {n:,}"
                                      for y, n in wave_n.items())
         + f"   (total {len(d):,}, women {d['pid'].nunique():,},"
         f" cities {d['city'].nunique()})")
    emit("  N by wave x arm (C/T):")
    emit("    " + wave_t.to_string().replace("\n", "\n    "))
    emit("  mean ideal by year x T:")
    emit("    " + d.groupby(["year", "T"])["ideal"].mean().round(3)
         .unstack().to_string().replace("\n", "\n    "))

    rows = []
    for yvar, lab in (("ideal", "ideal #kids"), ("ideal2", "P(ideal>=2)")):
        f_ = f"{yvar} ~ T:post + {CTRL}{FE}"
        m = fit(f_, d)
        rows.append((lab + " (pooled post)", m.params["T:post"],
                     m.bse["T:post"], m.pvalues["T:post"], int(m.nobs)))
        if wcb:
            b, t, p, G = wild_cluster_boot(d, f_)
            emit(f"  [WCB] {lab}: b={b:+.4f}  t={t:+.2f}  "
                 f"WCB p={p:.4f}  (G={G})")
        for py in (2018, 2022):
            dd = d[d["year"].isin([2014, py])].copy()
            dd["post"] = (dd["year"] == py).astype(int)
            m = fit(f"{yvar} ~ T:post + {CTRL}{FE}", dd)
            rows.append((f"{lab} (2014 vs {py})", m.params["T:post"],
                         m.bse["T:post"], m.pvalues["T:post"], int(m.nobs)))
    emit(f"  {'outcome':<28} {'T:post':>9} {'SE':>8} {'p':>7} {'N':>8}")
    for lab, b, se, p, n in rows:
        emit(f"  {lab:<28} {b:>+9.4f} {se:>8.4f} {p:>7.3f} {n:>8,}")
    return rows


def main():
    emit("Analysis C: intentions vs realizations -- stated ideal #kids DID")
    emit("Married women 20-45, unregulated prefectures, baseline-city T,")
    emit("city+year FE, age+age^2, cluster by city. 2014 pre | 2018+2022 post.")
    emit("Ideal kept in [0,6]; >6 dropped (reported above); negatives = NA.\n")

    df = build()
    emit(f"\nAnalytic sample: {len(df):,} obs, {df['pid'].nunique():,} women, "
         f"{df['city'].nunique()} cities, treated share {df['T'].mean():.3f}")
    emit(f"Parity distribution: "
         + str(df["parity"].clip(upper=3).value_counts().sort_index()
               .to_dict()) + "  (3 = 3+)")

    one = df[df["parity"] == 1]
    groups = [
        ("(a) ALL married women 20-45", df, False),
        ("(b) ONE-CHILD mothers (parity=1 as of wave)", one, False),
        ("(c) firstborn-DAUGHTER one-child mothers  [headline margin]",
         one[one["fb_son"] == 0], True),
        ("(d) firstborn-SON one-child mothers  [comparison]",
         one[one["fb_son"] == 1], False),
    ]
    all_rows = {}
    for name, d, wcb in groups:
        all_rows[name] = report_group(d, name, wcb=wcb)

    # DDD on the one-child sample (symmetry with layer 3)
    emit("\n" + "=" * 74)
    emit("DDD (one-child sample): T:post + T:post:son + full interactions")
    dd = one[one["fb_son"].isin([0, 1])].copy()
    dd["son"] = dd["fb_son"].astype(int)
    for yvar, lab in (("ideal", "ideal #kids"), ("ideal2", "P(ideal>=2)")):
        m = fit(f"{yvar} ~ T:post + T:post:son + post:son + T:son + son"
                f" + {CTRL}{FE}", dd)
        emit(f"  {lab}: T:post {m.params['T:post']:+.4f} "
             f"(p={m.pvalues['T:post']:.3f})   "
             f"T:post:son {m.params['T:post:son']:+.4f} "
             f"(p={m.pvalues['T:post:son']:.3f})   N={int(m.nobs):,}")

    # ------------------------------------------------------------ qka205
    emit("\n" + "=" * 74)
    emit("SUPPLEMENTARY: qka205 (2022 only) 'want a child in next 2 years'")
    emit("2022 T-vs-C cross-section (no pre-period): age+age^2 + C(provcd),")
    emit("cluster by city. Treatment coef = T level gap in 2022.")
    d22 = df[(df["year"] == 2022) & df["want2yr"].notna()].copy()
    emit(f"  qka205 valid in analytic 2022 sample: {len(d22):,} of "
         f"{(df['year'] == 2022).sum():,} "
         f"(skip pattern: asked conditionally)")
    subs = [("(a) all married women", d22),
            ("(b) one-child mothers", d22[d22["parity"] == 1]),
            ("(c) daughter-first one-child",
             d22[(d22["parity"] == 1) & (d22["fb_son"] == 0)]),
            ("(d) son-first one-child",
             d22[(d22["parity"] == 1) & (d22["fb_son"] == 1)])]
    emit(f"  {'group':<32} {'T coef':>9} {'SE':>8} {'p':>7} {'N':>7} "
         f"{'mean(y)':>8}")
    for name, d in subs:
        if len(d) < 60 or d["city"].nunique() < 20:
            emit(f"  {name:<32} SKIPPED (N={len(d)})")
            continue
        m = smf.ols(f"want2yr ~ T + {CTRL} + C(provcd)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        emit(f"  {name:<32} {m.params['T']:>+9.4f} {m.bse['T']:>8.4f} "
             f"{m.pvalues['T']:>7.3f} {int(m.nobs):>7,} "
             f"{d['want2yr'].mean():>8.3f}")

    # ------------------------------------------------------------ context
    emit("\n" + "=" * 74)
    emit("CONTEXT: established realized-birth effect on the same margin:")
    emit("  daughter-first one-child annual 2nd-birth hazard T:post = -2.4pp")
    emit("  on 8.4% base (p=0.024), son placebo null "
         "(annual_firstborn_gender.py).")
    emit("Prediction was T:post ~ 0 on stated ideals in all groups, esp (c).")

    out = os.path.join(OUT, "tables", "intentions_did.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
