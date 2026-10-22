# ==========================================================================
# REPLICATION PACKAGE — distance_ring_profile.py
# Tier: CORE
# Paper items: Table 8 (Distance: ring rows); Table B4 (Distance rings)
# Reads: CFPS .dta files in data/20xx/, output/housing_price_city.csv, output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/distance_ring_profile.txt
# Imports local modules: none
# Run order: step 31 in run_all.sh
# ==========================================================================

"""Distance-RING profile of the HPR spillover effect on the second-birth hazard.

Replaces the binary <=250 km cutoff with distance rings and traces the spatial
decay of the daughter-first suppression. Answers:
  - is the decay monotone or lumpy?
  - is the 250-350 km ring a diluted effect (which would explain why the
    300 km cutoff works and 200 km doesn't) or a clean zero (donut placebo)?
  - why is the continuous LINEAR dose flat? (if the profile is non-linear /
    flat-then-drop, a linear term is structurally blind - cf. inverted-U lesson)

Design: identical machinery to scripts/annual_firstborn_gender.py (annual
second-birth hazard, one-child mothers 20-45, 2009-2022, baseline-city
assignment, 22 regulated cities dropped, city+year FE, age+duration controls,
cluster by city, Post=2018, event-study base 2015).

RING CHOICE (pre-registered before any estimate was run):
  The prompt's candidate breaks [0,100,175,250,325,400,inf) leave only 9 CFPS
  cities in each of [250,325) and [325,400) - below the >=10-city floor. The
  breaks were therefore adjusted ON COMPOSITION ONLY (city/woman counts, no
  outcomes looked at) to [0,100,175,250,350,450), reference >= 450 km, which
  keeps >= 11 cities per ring (14/26/20/12/11, reference 30). No estimate was
  seen before locking these breaks.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

FAMCONF = {
    2010: "2010/ecfps2010famconf_nat072016.dta",
    2012: "2012/ecfps2012famconf_092015.dta",
    2014: "2014/ecfps2014famconf_170630.dta",
    2016: "2016/ecfps2016famconf_201804.dta",
    2018: "2018/ecfps2018famconf_202008.dta",
    2020: "2020/ecfps2020famconf_202306.dta",
    2022: "2022/ecfps2022famconf_202410.dta",
}

BREAKS = [0, 100, 175, 250, 350, 450]          # ring lower edges; ref >= 450
RING_LABELS = ["[0,100)", "[100,175)", "[175,250)", "[250,350)", "[350,450)"]
RING_MID = [50, 137.5, 212.5, 300, 400]
NR = len(RING_LABELS)

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


def load_rosters():
    """Per pid from the LATEST famconf wave: list of (child birth year, gender)."""
    frames = []
    for year, rel in FAMCONF.items():
        path = os.path.join(DATA, rel)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        by_cols = [c for c in meta.column_names if re.fullmatch(r"tb1y_a_c\d+", c)]
        g_cols = [c.replace("tb1y", "tb2") for c in by_cols]
        g_cols = [c for c in g_cols if c in meta.column_names]
        df, _ = pyreadstat.read_dta(path, usecols=["pid"] + by_cols + g_cols)
        recs = []
        for i, byc in enumerate(by_cols, start=1):
            gc = f"tb2_a_c{i}"
            sub = df[["pid", byc] + ([gc] if gc in df.columns else [])].copy()
            sub.columns = ["pid", "cby"] + (["cgender"] if gc in df.columns else [])
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
    return roster[["pid", "cby", "cgender"]], last.reset_index()


def build_norm_index():
    """City mean of qm1004 (2014 adults) - reuse of gender_norm_index.py logic."""
    df, _ = pyreadstat.read_dta(
        os.path.join(DATA, "2014/ecfps2014adult_201906.dta"),
        usecols=["pid", "countyid14", "qm1004"])
    df["qm1004"] = df["qm1004"].where(df["qm1004"].between(1, 5))
    xw, _ = pyreadstat.read_dta(os.path.join(DATA, "CFPS_cleaned", "顺序码匹配.dta"))
    xw = xw[["countyid", "city"]].rename(columns={"city": "city_code"})
    xw["countyid"] = pd.to_numeric(xw["countyid"], errors="coerce")
    xw = xw.dropna().drop_duplicates("countyid")
    df["countyid"] = pd.to_numeric(df["countyid14"], errors="coerce")
    df = df.merge(xw, on="countyid", how="inner")
    g = df.groupby("city_code")["qm1004"].agg(["mean", "count"])
    g = g[g["count"] >= 25]
    return g["mean"].rename("son_norm")


def city_price_growth(y0=2016, y1=2019):
    hp = pd.read_csv(os.path.join(OUT, "housing_price_city.csv"))
    w = hp.pivot_table(index="city_code", columns="year", values="hp_city_avg")
    g = (np.log(w[y1]) - np.log(w[y0])).rename("pgrowth")
    return g.dropna()


def build_hazard_panel(y0=2009, y1=2022):
    roster, last_wave = load_rosters()

    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "gender", "birthy", "city_code", "urban"])
    women = panel[panel["gender"] == 0].sort_values(["pid", "year"])
    base = women.dropna(subset=["city_code"]).groupby("pid").first()
    attrs = women.groupby("pid").agg(birthy=("birthy", "first"),
                                     urban0=("urban", "first"))
    attrs = attrs.join(base["city_code"]).dropna(subset=["birthy", "city_code"])
    attrs = attrs.reset_index()

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    attrs = attrs.merge(
        hpr[["city_code", "is_regulated", "treat_hpr", "dist_to_hpr_km"]],
        on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]

    r = roster.sort_values(["pid", "cby"])
    r["k"] = r.groupby("pid").cumcount()
    first = r[r["k"] == 0][["pid", "cby", "cgender"]].rename(
        columns={"cby": "first_by", "cgender": "fb_son"})
    second = r[r["k"] == 1][["pid", "cby"]].rename(columns={"cby": "second_by"})
    kids = first.merge(second, on="pid", how="left")
    kids = kids[(kids["second_by"].isna()) | (kids["second_by"] > kids["first_by"])]

    df = attrs.merge(kids, on="pid", how="inner").merge(last_wave, on="pid")
    df = df[df["fb_son"].isin([0, 1])]

    df = df.merge(pd.DataFrame({"year": range(y0, y1 + 1)}), how="cross")
    end = df[["second_by", "last_wave"]].min(axis=1)
    df = df[(df["year"] > df["first_by"]) & (df["year"] <= end.fillna(df["last_wave"]))]
    df["age"] = df["year"] - df["birthy"]
    df = df[(df["age"] >= 20) & (df["age"] <= 45)]

    df["y"] = (df["year"] == df["second_by"]).astype(float)
    df["dur"] = df["year"] - df["first_by"]
    df["son"] = df["fb_son"].astype(int)
    df["T"] = df["treat_hpr"].astype(int)
    df["city"] = df["city_code"].astype(int)
    df["post"] = (df["year"] >= 2018).astype(int)

    # distance rings (pre-registered breaks, see module docstring)
    dist = df["dist_to_hpr_km"]
    edges = BREAKS + [np.inf]
    df["ring"] = np.digitize(dist, edges[1:], right=False)  # 0..NR-1 rings, NR=ref
    for k in range(NR):
        df[f"r{k+1}"] = (df["ring"] == k).astype(int)
        df[f"r{k+1}p"] = df[f"r{k+1}"] * df["post"]
    return df


def fit_cluster(formula, d):
    return smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})


def ring_spec(d, label):
    terms = " + ".join(f"r{k+1}p" for k in range(NR))
    m = fit_cluster(f"y ~ {terms} + age + I(age**2) + dur + I(dur**2)"
                    f" + C(city) + C(year)", d)
    emit(f"\n[{label}]  N={int(m.nobs):,}  women={d['pid'].nunique():,}  "
         f"mean(y)={d['y'].mean():.4f}")
    res = []
    for k in range(NR):
        key = f"r{k+1}p"
        b, se, p = m.params[key], m.bse[key], m.pvalues[key]
        emit(f"  {RING_LABELS[k]:<11} x post  {b:+.4f}  (SE {se:.4f}, p={p:.3f})")
        res.append((RING_MID[k], b, se, p))
    return m, res


def main():
    df = build_hazard_panel()
    emit("DISTANCE-RING PROFILE of the HPR spillover on the second-birth hazard")
    emit("=" * 72)
    emit("Rings on dist_to_hpr_km, reference = >=450 km. Breaks pre-registered")
    emit("on composition only (>=10 CFPS cities/ring floor forced a widening of")
    emit("the outer rings from the [250,325)/[325,400) candidates, 9 cities each).")
    emit(f"Panel: rows={len(df):,}  women={df['pid'].nunique():,}  "
         f"cities={df['city'].nunique()}  hazard mean={df['y'].mean():.4f}")

    # ---------------- composition table (reported BEFORE any estimates) ------
    norm = build_norm_index()
    pg = city_price_growth()
    df = df.merge(norm, left_on="city_code", right_index=True, how="left")
    df = df.merge(pg, left_on="city_code", right_index=True, how="left")

    emit("\n--- Ring composition (one-child risk set, woman-years) ---")
    emit(f"{'ring':<11}{'cities':>7}{'women':>8}{'wyears':>9}{'births':>8}"
         f"{'hazard':>8}{'rural':>7}{'norm':>7}{'pgrow':>7}{'pg_cov':>7}")
    for k in range(NR + 1):
        sub = df[df["ring"] == k]
        lab = RING_LABELS[k] if k < NR else ">=450"
        emit(f"{lab:<11}{sub['city'].nunique():>7}{sub['pid'].nunique():>8,}"
             f"{len(sub):>9,}{int(sub['y'].sum()):>8,}{sub['y'].mean():>8.4f}"
             f"{1 - sub['urban0'].mean():>7.3f}{sub['son_norm'].mean():>7.3f}"
             f"{sub['pgrowth'].mean():>7.3f}{sub['pgrowth'].notna().mean():>7.3f}")
    emit("(rural = 1-urban at baseline wave; norm = city mean qm1004 2014;")
    emit(" pgrow = city log price growth 2016->2019, pg_cov = coverage share)")

    # ---------------- sanity: reproduce the binary headline -----------------
    da, so = df[df["son"] == 0], df[df["son"] == 1]
    m = fit_cluster("y ~ T:post + age + I(age**2) + dur + I(dur**2)"
                    " + C(city) + C(year)", da)
    emit(f"\nSanity (binary <=250 headline, daughter arm): T:post = "
         f"{m.params['T:post']:+.4f} (p={m.pvalues['T:post']:.3f})"
         f"  [expect ~ -0.024, p~0.024]")

    # ---------------- (a)-(c) ring x post specs ------------------------------
    emit("\n" + "=" * 72)
    emit("(a)-(c) RING x POST (Post=2018), reference ring >= 450 km")
    _, res_pool = ring_spec(df, "POOLED one-child")
    _, res_d = ring_spec(da, "firstborn DAUGHTER")
    _, res_s = ring_spec(so, "firstborn SON (placebo)")

    # ---------------- (d) linear spline in distance x post, daughter arm -----
    emit("\n" + "=" * 72)
    emit("(d) Linear spline in distance x post, DAUGHTER arm")
    emit("    knots at 100 / 250 / 450 km, distance capped at 800 km;")
    emit("    profile below = effect at d relative to d = 800 km (far ref).")
    dsp = da.copy()
    dcap = dsp["dist_to_hpr_km"].clip(upper=800)
    dsp["s1"] = np.minimum(dcap, 100)
    dsp["s2"] = (dcap - 100).clip(0, 150)
    dsp["s3"] = (dcap - 250).clip(0, 200)
    dsp["s4"] = (dcap - 450).clip(0, 350)
    for j in range(1, 5):
        dsp[f"ps{j}"] = dsp[f"s{j}"] * dsp["post"]
    msp = fit_cluster("y ~ ps1 + ps2 + ps3 + ps4 + age + I(age**2)"
                      " + dur + I(dur**2) + C(city) + C(year)", dsp)
    for j in range(1, 5):
        emit(f"  slope seg{j} (per km): {msp.params[f'ps{j}']:+.6f} "
             f"(p={msp.pvalues[f'ps{j}']:.3f})")
    names = [f"ps{j}" for j in range(1, 5)]
    V = msp.cov_params().loc[names, names].values
    beta = msp.params[names].values

    def svec(d):
        return np.array([min(d, 100), np.clip(d - 100, 0, 150),
                         np.clip(d - 250, 0, 200), np.clip(d - 450, 0, 350)])

    s800 = svec(800)
    emit(f"\n  {'d(km)':>6}  {'effect':>8}  {'SE':>7}  {'p':>6}")
    spline_prof = []
    for dgrid in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 550, 650]:
        c = svec(dgrid) - s800
        eff = float(c @ beta)
        se = float(np.sqrt(c @ V @ c))
        from scipy.stats import norm as _n
        p = 2 * (1 - _n.cdf(abs(eff / se))) if se > 0 else np.nan
        emit(f"  {dgrid:>6}  {eff:>+8.4f}  {se:>7.4f}  {p:>6.3f}")
        spline_prof.append((dgrid, eff, se))
    fj = msp.f_test(" = ".join(names) + " = 0")
    emit(f"  joint F (all spline x post slopes = 0): "
         f"F={float(fj.fvalue):.2f}, p={float(fj.pvalue):.4f}")

    # ---------------- gates: ring-level pre-trends ---------------------------
    emit("\n" + "=" * 72)
    emit("GATE: ring x year event study (base 2015) - pre coefficients 2010-14")
    ev_years = [yy for yy in range(2010, 2023) if yy != 2015]
    for k in range(NR):
        for yy in ev_years:
            df[f"R{k+1}y{yy}"] = df[f"r{k+1}"] * (df["year"] == yy).astype(int)
    terms = " + ".join(f"R{k+1}y{yy}" for k in range(NR) for yy in ev_years)
    for arm, lab in [(0, "DAUGHTER"), (1, "SON")]:
        d = df[df["son"] == arm]
        mev = fit_cluster(f"y ~ {terms} + age + I(age**2) + dur + I(dur**2)"
                          f" + C(city) + C(year)", d)
        emit(f"\n[{lab} arm]")
        for k in range(NR):
            pre = [f"R{k+1}y{yy}" for yy in range(2010, 2015)]
            f = mev.f_test(" = ".join(pre) + " = 0")
            cs = "  ".join(
                f"{yy}:{mev.params[f'R{k+1}y{yy}']:+.3f}"
                f"({mev.pvalues[f'R{k+1}y{yy}']:.2f})" for yy in range(2010, 2015))
            emit(f"  {RING_LABELS[k]:<11} pre: {cs}")
            emit(f"  {'':<11} joint pre F={float(f.fvalue):.2f} "
                 f"p={float(f.pvalue):.3f}")
        if arm == 0:
            post_line = {}
            for k in range(NR):
                ps = [(yy, mev.params[f'R{k+1}y{yy}'], mev.pvalues[f'R{k+1}y{yy}'])
                      for yy in range(2018, 2023)]
                post_line[k] = ps
            emit("\n  post path (2018-22), daughter arm:")
            for k in range(NR):
                cs = "  ".join(f"{yy}:{b:+.3f}({p:.2f})" for yy, b, p in post_line[k])
                emit(f"  {RING_LABELS[k]:<11} {cs}")

    # ---------------- figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for res, lab, col, off in [(res_d, "Firstborn daughter", "#0072B2", -6),
                               (res_s, "Firstborn son (placebo)", "#E69F00", 6)]:
        x = np.array([r[0] for r in res]) + off
        b = np.array([r[1] for r in res])
        se = np.array([r[2] for r in res])
        ax.errorbar(x, b, yerr=1.96 * se, fmt="o", ms=6, lw=1.8, capsize=3,
                    color=col, label=lab)
    xs = [p[0] for p in spline_prof if p[0] <= 460]
    ys = [p[1] for p in spline_prof if p[0] <= 460]
    ax.plot(xs, ys, "-", color="#0072B2", alpha=0.35, lw=1.6,
            label="Daughter arm, distance spline")
    ax.axhline(0, color="0.55", lw=1)
    ax.axvline(250, color="0.75", lw=1, ls="--")
    ax.text(253, ax.get_ylim()[0] * 0.0 + 0.035, "250 km cutoff",
            fontsize=8, color="0.45")
    ax.set_xlabel("Distance to nearest regulated HPR city (km, ring midpoint)")
    ax.set_ylabel("Ring x Post effect on second-birth hazard")
    ax.set_title("Spatial decay of the HPR spillover (Post=2018, ref >= 450 km)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    figpath = os.path.join(OUT, "figures", "ring_profile.png")
    fig.savefig(figpath, dpi=200)
    emit(f"\nFigure saved -> {figpath}")

    out = os.path.join(OUT, "tables", "distance_ring_profile.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
