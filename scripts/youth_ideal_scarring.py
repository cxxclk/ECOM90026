# ==========================================================================
# REPLICATION PACKAGE — youth_ideal_scarring.py
# Tier: SUPPLEMENTARY
# Paper items: Section 6.1 (never-married 18-30, -8.2 points)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/youth_ideal_scarring.txt
# Imports local modules: none
# Run order: step 54 in run_all.sh
# ==========================================================================

"""Ticket F1: youth fertility-ideal "scarring".

Question: did the HPR price shock lower the STATED fertility ideal of
UNMARRIED young adults (18-30) before family formation -- i.e. preference
formation / demand destruction, the mirror of Analysis C (unmet demand)?

Data: ideal number of kids = qm501 (2014 adult file ONLY -- in 2010/16/18
the same var name is "importance of money", never touch) + qka202
(2018/2022 person files). Repeated cross-section: 2014 pre, 2018/2022 post.

Design: unmarried (never-married) adults 18-30, unregulated prefectures,
HPR spillover treatment; two anchoring variants (baseline-wave city vs
current-wave city -- young people move); city + year FE, age controls,
cluster by city. Honest limits: single pre wave (no pre-trend test),
repeated cross-section (cohort composition shifts -- cohort-bin variant).
"""
import os
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

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def load_ideal():
    frames = []
    for year, (rel, var) in IDEAL_SRC.items():
        df, meta = pyreadstat.read_dta(os.path.join(DATA, rel),
                                       usecols=["pid", var])
        lab = meta.column_names_to_labels.get(var, "")
        emit(f"  {year}: {var} = '{lab}'  raw N={df[var].notna().sum():,}")
        assert ("ideal" in str(lab).lower()) or ("kid" in str(lab).lower()) \
            or ("child" in str(lab).lower()), f"label check failed {year}"
        df = df.rename(columns={var: "ideal"})
        df["ideal"] = pd.to_numeric(df["ideal"], errors="coerce")
        df.loc[(df["ideal"] < 0) | (df["ideal"] > 10), "ideal"] = np.nan
        df["year"] = year
        frames.append(df[["pid", "year", "ideal"]])
    return pd.concat(frames, ignore_index=True)


def build(marital_code=1, keep_regulated=False):
    panel = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                        usecols=["pid", "year", "gender", "birthy", "marital",
                                 "urban", "city_code", "homeowner", "provcd"])
    # baseline city = earliest wave with non-missing city, any wave
    base = (panel.dropna(subset=["city_code"]).sort_values(["pid", "year"])
            .groupby("pid")["city_code"].first().rename("base_city"))
    panel = panel.merge(base, on="pid", how="left")

    ideal = load_ideal()
    df = panel.merge(ideal, on=["pid", "year"], how="inner")

    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 18) & (df["age"] <= 30)]
    df = df[df["marital"] == marital_code]           # 1=never married
    df = df.dropna(subset=["ideal"])

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    hmap = hpr.set_index("city_code")[["is_regulated", "treat_hpr",
                                       "dist_to_hpr_km"]]

    out = {}
    for anchor, col in (("baseline-city", "base_city"),
                        ("current-city", "city_code")):
        d = df.dropna(subset=[col]).copy()
        d = d.join(hmap, on=col)
        if not keep_regulated:
            d = d[(d["is_regulated"] == 0) & d["treat_hpr"].notna()]
        else:
            d = d[d["treat_hpr"].notna() | (d["is_regulated"] == 1)]
        d["T"] = d["treat_hpr"].fillna(0).astype(int)
        d["post"] = (d["year"] >= 2018).astype(int)
        d["city"] = d[col].astype(int)
        d["ideal2"] = (d["ideal"] >= 2).astype(float)
        d["cohort5"] = (d["birthy"] // 5 * 5).astype(int)
        d["female"] = (d["gender"] == 0).astype(int)
        out[anchor] = d
    return out


def fit(formula, d, label, keys=("T:post",)):
    if len(d) < 200 or d["city"].nunique() < 30:
        emit(f"\n[{label}]  SKIPPED (N={len(d)}, cities={d['city'].nunique()})")
        return None
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  people={d['pid'].nunique():,}  "
         f"cities={d['city'].nunique()}  mean(y)={m.model.endog.mean():.3f}")
    for k in keys:
        if k in m.params.index:
            emit(f"  {k:<12} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, "
                 f"p={m.pvalues[k]:.3f})")
    return m


def main():
    emit("Ticket F1: youth (18-30, never-married) ideal number of kids, DID")
    emit("Ideal sources (label-verified):")
    frames = build()

    CTRL = "C(age)"
    FE = " + C(city) + C(year)"

    for anchor, d in frames.items():
        emit("\n" + "=" * 72)
        emit(f"ANCHOR = {anchor}   (waves 2014 pre | 2018+2022 post)")
        emit(f"  T share={d['T'].mean():.3f}  wave N: "
             + str(d.groupby('year').size().to_dict()))
        emit(f"  mean ideal by year x T:")
        emit(d.groupby(["year", "T"])["ideal"].mean().round(3)
             .unstack().to_string())

        fit(f"ideal ~ T:post + {CTRL}{FE}", d, "ALL youth, ideal #kids")
        fit(f"ideal2 ~ T:post + {CTRL}{FE}", d, "ALL youth, P(ideal>=2)")

        # cohort-bin variant (composition robustness)
        fit(f"ideal ~ T:post + C(cohort5) + age{FE}", d,
            "ALL youth, cohort-bin ctrl")

        # wave-by-wave (2018 vs 2022 separately, vs 2014)
        for py in (2018, 2022):
            dd = d[d["year"].isin([2014, py])].copy()
            dd["post"] = (dd["year"] == py).astype(int)
            fit(f"ideal ~ T:post + {CTRL}{FE}", dd, f"ALL youth, 2014 vs {py}")

        # cuts
        fit(f"ideal ~ T:post + {CTRL}{FE}", d[d["female"] == 1], "FEMALE")
        fit(f"ideal ~ T:post + {CTRL}{FE}", d[d["female"] == 0], "MALE")
        fit(f"ideal ~ T:post + {CTRL}{FE}", d[d["urban"] == 1], "URBAN")
        fit(f"ideal ~ T:post + {CTRL}{FE}", d[d["urban"] == 0], "RURAL")
        fit(f"ideal ~ T:post + {CTRL}{FE}", d[d["homeowner"] == 1],
            "HH OWNS HOME (parents' home proxy)")
        fit(f"ideal ~ T:post + {CTRL}{FE}", d[d["homeowner"] == 0],
            "HH DOES NOT OWN")

        # DDD gender within one model
        fit(f"ideal ~ T:post + T:post:female + post:female + T:female + female"
            f" + {CTRL}{FE}", d, "DDD gender", keys=("T:post", "T:post:female"))

    return frames


# ---------------------------------------------------------------- PART 2
FORM2 = "ideal2 ~ T:post + C(age) + C(city) + C(year)"

TRUST_SRC = {
    2014: "2014/ecfps2014adult_201906.dta",
    2018: "2018/ecfps2018person_202012.dta",
    2022: "2022/ecfps2022person_202410.dta",
}


def wild_cluster_boot(d, formula, reps=999, seed=90026):
    """WCB-t, Rademacher, null imposed, cluster=city (pattern from
    gender_norm_robustness.py)."""
    y_full, X_full = dmatrices(formula, d, return_type="dataframe")
    y = y_full.values.ravel()
    x = X_full["T:post"].values
    Z = X_full.drop(columns="T:post").values
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


def load_trust():
    frames = []
    for year, rel in TRUST_SRC.items():
        df, meta = pyreadstat.read_dta(os.path.join(DATA, rel),
                                       usecols=["pid", "qn1001", "qn10024"])
        for v in ("qn1001", "qn10024"):
            lab = str(meta.column_names_to_labels.get(v, "")).lower()
            assert "trust" in lab, f"trust label check failed {year} {v}"
        df["trust_gen"] = pd.to_numeric(df["qn1001"], errors="coerce")
        df["trust_gen"] = df["trust_gen"].map({1: 1.0, 5: 0.0, 0: 0.0})
        df["trust_str"] = pd.to_numeric(df["qn10024"], errors="coerce")
        df.loc[(df["trust_str"] < 0) | (df["trust_str"] > 10),
               "trust_str"] = np.nan
        df["year"] = year
        frames.append(df[["pid", "year", "trust_gen", "trust_str"]])
    return pd.concat(frames, ignore_index=True)


def part2(frames):
    emit("\n" + "#" * 72)
    emit("PART 2: CONFIRMATORY TESTS for P(ideal>=2)  (coordinator request)")

    # ---- 1. wild cluster bootstrap, both anchors
    emit("\n--- 1. Wild cluster bootstrap (Rademacher, B=999, cluster=city) ---")
    for anchor, d in frames.items():
        b, t, p, G = wild_cluster_boot(d, FORM2)
        emit(f"  [{anchor:>13}] T:post = {b:+.4f}  t = {t:+.2f}  "
             f"WCB p = {p:.4f}  (G={G})")

    d0 = frames["baseline-city"]

    # ---- 2a. placebo outcomes (trust battery, fertility/housing-unrelated)
    emit("\n--- 2a. Placebo outcomes: trust (same sample, same design) ---")
    trust = load_trust()
    dt = d0.merge(trust, on=["pid", "year"], how="left")
    for var, lab in (("trust_gen", "P(most people trustworthy) [qn1001]"),
                     ("trust_str", "trust in strangers 0-10 [qn10024]")):
        dd = dt.dropna(subset=[var]).copy()
        m = smf.ols(f"{var} ~ T:post + C(age) + C(city) + C(year)",
                    data=dd).fit(cov_type="cluster",
                                 cov_kwds={"groups": dd["city"]})
        bw, tw, pw, G = wild_cluster_boot(
            dd, f"{var} ~ T:post + C(age) + C(city) + C(year)")
        emit(f"  {lab}: N={int(m.nobs):,}  T:post {m.params['T:post']:+.4f} "
             f"(cl.p={m.pvalues['T:post']:.3f}, WCB p={pw:.3f})")

    # ---- 2b. coherence: regulated source cities (direct HPR) vs far controls
    emit("\n--- 2b. Coherence: REGULATED source cities vs unregulated controls ---")
    fr_all = build(marital_code=1, keep_regulated=True)
    dr = fr_all["baseline-city"]
    dr = dr[(dr["is_regulated"] == 1) |
            ((dr["is_regulated"] == 0) & (dr["T"] == 0))].copy()
    dr["T"] = dr["is_regulated"].astype(int)
    m = smf.ols(FORM2, data=dr).fit(cov_type="cluster",
                                    cov_kwds={"groups": dr["city"]})
    emit(f"  regulated-as-treated: N={int(m.nobs):,} "
         f"(reg youth={int(dr['T'].sum()):,})  "
         f"T:post {m.params['T:post']:+.4f} (p={m.pvalues['T:post']:.3f})"
         f"   [prediction: negative too — coherence, not placebo]")

    # distance bands within unregulated: treated<=250 / near 250-400 / far>400
    emit("\n--- 2b'. Distance-band gradient (unregulated only, ref = >400km) ---")
    db = d0.copy()
    db["Ttr"] = db["T"]
    db["Tnear"] = ((db["T"] == 0) & (db["dist_to_hpr_km"] <= 400)).astype(int)
    m = smf.ols("ideal2 ~ Ttr:post + Tnear:post + C(age) + C(city) + C(year)",
                data=db).fit(cov_type="cluster",
                             cov_kwds={"groups": db["city"]})
    n_near = db.loc[db["Tnear"] == 1, "pid"].nunique()
    emit(f"  treated(<=250) x post: {m.params['Ttr:post']:+.4f} "
         f"(p={m.pvalues['Ttr:post']:.3f})")
    emit(f"  near(250-400) x post:  {m.params['Tnear:post']:+.4f} "
         f"(p={m.pvalues['Tnear:post']:.3f})  [near youths={n_near:,}; "
         f"prediction: weak/zero]")

    # ---- 3. robustness
    emit("\n--- 3a. Drop-one-province (baseline anchor, P(ideal>=2)) ---")
    coefs = []
    for pv in sorted(d0["provcd"].dropna().unique()):
        dd = d0[d0["provcd"] != pv]
        if dd["city"].nunique() < 30:
            continue
        m = smf.ols(FORM2, data=dd).fit(cov_type="cluster",
                                        cov_kwds={"groups": dd["city"]})
        coefs.append((pv, m.params["T:post"], m.pvalues["T:post"]))
    neg = sum(1 for _, b, _ in coefs if b < 0)
    worst = max(coefs, key=lambda z: z[2])
    strong = min(coefs, key=lambda z: z[2])
    emit(f"  {len(coefs)} drops: {neg}/{len(coefs)} negative;  "
         f"range [{min(b for _, b, _ in coefs):+.4f}, "
         f"{max(b for _, b, _ in coefs):+.4f}]")
    emit(f"  weakest: drop prov {int(worst[0])} -> {worst[1]:+.4f} "
         f"(p={worst[2]:.3f});  strongest: drop prov {int(strong[0])} -> "
         f"{strong[1]:+.4f} (p={strong[2]:.3f})")

    emit("\n--- 3b. Timing: P(ideal>=2) wave-by-wave (baseline anchor) ---")
    for py in (2018, 2022):
        dd = d0[d0["year"].isin([2014, py])].copy()
        dd["post"] = (dd["year"] == py).astype(int)
        m = smf.ols(FORM2, data=dd).fit(cov_type="cluster",
                                        cov_kwds={"groups": dd["city"]})
        emit(f"  2014 vs {py}: T:post {m.params['T:post']:+.4f} "
             f"(p={m.pvalues['T:post']:.3f})")

    emit("\n--- 3c. MARRIED 18-30 comparison profile (baseline anchor) ---")
    frm = build(marital_code=2)
    dm = frm["baseline-city"]
    for f_, lab in ((FORM2, "P(ideal>=2)"),
                    ("ideal ~ T:post + C(age) + C(city) + C(year)",
                     "ideal #kids")):
        m = smf.ols(f_, data=dm).fit(cov_type="cluster",
                                     cov_kwds={"groups": dm["city"]})
        emit(f"  married youth, {lab}: N={int(m.nobs):,}  "
             f"T:post {m.params['T:post']:+.4f} "
             f"(p={m.pvalues['T:post']:.3f})")


if __name__ == "__main__":
    frames = main()
    part2(frames)
    out = os.path.join(OUT, "tables", "youth_ideal_scarring.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")
