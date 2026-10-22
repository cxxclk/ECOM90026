# ==========================================================================
# REPLICATION PACKAGE — table1_descriptives.py
# Tier: CORE
# Paper items: Table 1 (Panels A, B, C)
# Reads: output/housing_price_city.csv, output/panel_individual.csv
# Writes: output/tables/table1_descriptives.txt
# Imports local modules: annual_firstborn_gender, city_price_response, gender_norm_index
# Run order: step 10 in run_all.sh
# ==========================================================================

"""Table 1 (three panels) + Figure 2 for essay §3.5.

Panel A  one-child (second-birth) risk set, PRE-shock woman-years (2009-15), by arm:
         age, years since first birth, baseline education / urban / log family income
         p.c. / homeownership (woman-level, earliest wave), firstborn-daughter share;
         N woman-years, women, prefectures.  Diff = exposed - unexposed with p from a
         city-clustered regression of the variable on T.
Panel B  prefecture level, by arm: son-norm index (2014), traditional PC1, second-hand
         price growth 2012->15 (pre) and 2016->19 (post), coverage.
Panel C  raw annual second-birth hazard by arm x period x firstborn sex; premium
         (daughter - son) and gap (exposed - unexposed).
Figure 2 raw annual hazard 2009-2022, exposed vs unexposed, daughter-first and
         son-first panels.

Reuses build_hazard_panel() from annual_firstborn_gender.py (headline risk set),
build_norm_index() from gender_norm_index.py, growth() from city_price_response.py.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "output")
from annual_firstborn_gender import build_hazard_panel          # noqa: E402
from gender_norm_index import build_norm_index                  # noqa: E402
from city_price_response import growth                          # noqa: E402

TXT, MD = [], []
def emit(s=""):
    print(s); TXT.append(s)


def cl_p(d, var):
    """p-value of the exposed-unexposed difference, clustered by prefecture."""
    dd = d.dropna(subset=[var])
    m = smf.ols(f"{var} ~ T", data=dd).fit(cov_type="cluster",
                                          cov_kwds={"groups": dd["city"]})
    return m.params["T"], m.pvalues["T"]


def main():
    df = build_hazard_panel()
    # ---- baseline woman-level covariates: earliest wave with the variable ----
    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "year", "eduy", "urban", "fam_income_pc",
                               "homeowner"])
    pan = pan.sort_values(["pid", "year"])
    base = pan.groupby("pid").agg(eduy=("eduy", "first"), urban=("urban", "first"),
                                  faminc=("fam_income_pc", "first"),
                                  owner=("homeowner", "first"))
    base["lfaminc"] = np.log(base["faminc"].where(base["faminc"] > 0))
    df = df.merge(base, left_on="pid", right_index=True, how="left")
    df["daughter"] = 1 - df["son"]

    # ================= Panel A =================
    pre = df[df["year"] <= 2015]
    rows = [("Age (years)", "age"), ("Years since first birth", "dur"),
            ("Years of education", "eduy"), ("Urban residence", "urban"),
            ("Log family income p.c.", "lfaminc"), ("Homeowner", "owner"),
            ("First child is a daughter", "daughter")]
    emit("TABLE 1 — Descriptive statistics")
    emit("=" * 78)
    emit("Panel A. One-child sample, pre-shock woman-years 2009-2015")
    emit(f"{'':34}{'Unexposed':>12}{'Exposed':>12}{'Diff':>10}{'p':>8}")
    MD.append("**Panel A. One-child sample, pre-shock woman-years 2009–2015**\n")
    MD.append("| | Unexposed | Exposed | Diff. | p |")
    MD.append("|---|---:|---:|---:|---:|")
    for lab, v in rows:
        c = pre.loc[pre["T"] == 0, v].mean(); t = pre.loc[pre["T"] == 1, v].mean()
        dlt, p = cl_p(pre, v)
        emit(f"{lab:34}{c:12.3f}{t:12.3f}{dlt:+10.3f}{p:8.3f}")
        MD.append(f"| {lab} | {c:.2f} | {t:.2f} | {dlt:.2f} | {p:.2f} |")
    for lab, fn in [("Woman-years", lambda d: len(d)),
                    ("Women", lambda d: d["pid"].nunique()),
                    ("Prefectures", lambda d: d["city"].nunique())]:
        src = df if fn is None else pre
        f = (lambda d: len(d)) if fn is None else fn
        c, t = f(src[src["T"] == 0]), f(src[src["T"] == 1])
        emit(f"{lab:34}{c:12,}{t:12,}")
        MD.append(f"| {lab} | {c:,} | {t:,} | | |")

    # ================= Panel B =================
    cities = (df.groupby("city")["T"].first().rename("T").reset_index()
                .rename(columns={"city": "city_code"}))
    norm = build_norm_index()[["city_code", "son_norm", "trad_pc1", "n_resp"]]
    hp = pd.read_csv(os.path.join(OUT, "housing_price_city.csv"))
    g_pre = growth(hp, 2012, 2015).rename(columns={"g": "g_pre"})
    g_post = growth(hp, 2016, 2019).rename(columns={"g": "g_post"})
    cb = (cities.merge(norm, on="city_code", how="left")
                .merge(g_pre, on="city_code", how="left")
                .merge(g_post, on="city_code", how="left"))
    cb["city"] = cb["city_code"]
    for gcol in ("g_pre", "g_post"):
        cb[gcol] = 100 * cb[gcol]   # 100 x change in log price ~ percent
    emit("\nPanel B. Prefecture level")
    emit(f"{'':34}{'Unexposed':>12}{'Exposed':>12}{'Diff':>10}{'p':>8}{'n cities':>12}")
    MD.append("\n**Panel B. Prefecture level**\n")
    MD.append("| | Unexposed | Exposed | Diff. | p |")
    MD.append("|---|---:|---:|---:|---:|")
    for lab, v, fmt in [("Lineage-norm index, 2014 (1–5)", "son_norm", "{:.2f}"),
                        ("House-price growth 2012–15 (%)", "g_pre", "{:.1f}"),
                        ("House-price growth 2016–19 (%)", "g_post", "{:.1f}")]:
        sub = cb.dropna(subset=[v])
        c = sub.loc[sub["T"] == 0, v].mean(); t = sub.loc[sub["T"] == 1, v].mean()
        m = smf.ols(f"{v} ~ T", data=sub).fit(cov_type="HC1")
        n0, n1 = (sub["T"] == 0).sum(), (sub["T"] == 1).sum()
        emit(f"{lab:34}{fmt.format(c):>12}{fmt.format(t):>12}{m.params['T']:+10.3f}"
             f"{m.pvalues['T']:8.3f}{n0:>6}/{n1:<5}")
        dfmt = "{:.1f}" if v.startswith("g_") else "{:.2f}"
        MD.append(f"| {lab} | {fmt.format(c)} | {fmt.format(t)} | {dfmt.format(m.params['T'])} "
                  f"| {m.pvalues['T']:.2f} |")
    sd0, sd1 = cb.loc[cb["T"] == 0, "son_norm"].std(), cb.loc[cb["T"] == 1, "son_norm"].std()
    emit(f"{'  s.d. of son-norm index':34}{sd0:12.3f}{sd1:12.3f}")

    # ================= Panel C =================
    per = pd.cut(df["year"], [2008, 2015, 2017, 2022],
                 labels=["2009–15", "2016–17", "2018–22"])
    df["period"] = per
    emit("\nPanel C. Raw annual second-birth rate (%), one-child families")
    cats = list(per.cat.categories)
    H, N = {}, {}
    for p in cats:
        s = df[df["period"] == p]
        for a in (0, 1):
            for k in (0, 1):
                H[(p, a, k)] = 100 * s[(s["T"] == a) & (s["son"] == k)]["y"].mean()
                N[(p, a, k)] = len(s[(s["T"] == a) & (s["son"] == k)])
    pre, post = cats[0], cats[-1]
    def row(label, vals, fmt="{:.2f}", chg=True):
        v = [fmt.format(x) for x in vals]
        c = fmt.format(vals[-1] - vals[0]) if chg else ""
        emit(f"{label:36}" + "".join(f"{x:>12}" for x in v) + f"{c:>14}")
        MD.append(f"| {label} | " + " | ".join(v) + f" | {c} |")
    hdr = f"{'':36}" + "".join(f"{c:>12}" for c in cats) + f"{'Change':>14}"
    emit(hdr)
    MD.append("\n**Panel C. Raw annual second-birth rate (%), one-child families**\n")
    MD.append("| | " + " | ".join(cats) + " | Change |")
    MD.append("|---|" + "---:|" * (len(cats) + 1))
    for k, name in ((0, "First child a daughter"), (1, "First child a son")):
        emit(name); MD.append(f"| *{name}* | " + " | " * len(cats) + " |")
        row("Unexposed", [H[(p, 0, k)] for p in cats], fmt="{:.1f}")
        row("Exposed", [H[(p, 1, k)] for p in cats], fmt="{:.1f}")
        row("Exposed − unexposed", [H[(p, 1, k)] - H[(p, 0, k)] for p in cats], fmt="{:.1f}")
    for k, name in ((0, "Daughter-first"), (1, "Son-first")):   # txt only
        emit(f"{name:36}" + "".join(f"{N[(p, 0, k)]:>7,}/{N[(p, 1, k)]:<7,}" for p in cats))
    MD.append("\n" + '*Notes.* Panel A: means over pre-shock woman-years of the one-child sample; education, urban residence, income and homeownership from the woman\'s earliest wave. Panel B: the norm index is the 2014 prefecture mean answer (1–5) to "Women should give birth to at least one boy to continue the family line"; price growth is 100 × the change in log second-hand prices, for prefectures with data in both years (18 for 2012–15, 95 for 2016–19). p-values from regressions on the exposure indicator, clustered by prefecture (Panel A) or robust (Panel B). Panel C: raw annual second-birth rates in percent; the Change column of the "Exposed − unexposed" rows is a raw difference-in-differences.')

    # ================= Figure 2 =================
    yr = (df.groupby(["son", "T", "year"])["y"].mean() * 100).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    col = {0: "#1F5F7A", 1: "#C8642B"}   # unexposed teal, exposed orange
    name = {0: "Unexposed", 1: "Exposed"}
    for ax, k, ttl in zip(axes, (0, 1), ("First child a daughter", "First child a son")):
        ax.axvspan(2017.5, 2022.5, color="#000000", alpha=0.05, lw=0)
        ax.axvline(2015.5, color="#7C8290", lw=1, ls=":")
        for a in (0, 1):
            s = yr[(yr["son"] == k) & (yr["T"] == a)]
            ax.plot(s["year"], s["y"], color=col[a], lw=2, marker="o", ms=4, label=name[a])
            ax.annotate(name[a], (s["year"].iloc[-1] + 0.15, s["y"].iloc[-1]),
                        color=col[a], fontsize=9, va="center")
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.set_xlim(2008.5, 2023.8); ax.set_xticks(range(2009, 2023, 2))
        ax.grid(axis="y", color="#D9DBE0", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(2015.4, 0.6, "restrictions\n(late 2016)", fontsize=8, color="#7C8290",
                ha="right", va="bottom")
        ax.set_ylim(0, 16)
    axes[0].set_ylabel("Second-birth hazard (%)")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fp = os.path.join(OUT, "figures", "fig2_raw_hazards.png")
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    emit(f"\nFigure 2 -> {fp}")

    with open(os.path.join(OUT, "tables", "table1_descriptives.txt"), "w") as f:
        f.write("\n".join(TXT))
    with open(os.path.join(OUT, "tables", "table1_descriptives.md"), "w") as f:
        f.write("\n".join(MD))
    print("saved table1_descriptives.txt / .md")


if __name__ == "__main__":
    main()
