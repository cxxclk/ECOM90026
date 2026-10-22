# ==========================================================================
# REPLICATION PACKAGE — paper_figures.py
# Tier: CORE
# Paper items: Figure 1 (First stage in second-hand prices); Figure 2 (Raw second-birth rates); Figure 3 (Event study by first-child sex); Figure 4 (Norm thirds (both indices)); Figure 5 (Effects by fifth of predicted likelihood); Figure 6 (SHAP beeswarm); Figure 7 (Pre-train, post-predict); Figure 8 (Confidence in the future)
# Reads: output/analysis_data.csv, output/gbm_shap_values.csv, output/housing_price_city.csv, output/pretrain_postpredict_preds.csv, output/prognostic_scores.csv
# Writes: output/tables/annual_firstborn_gender.txt, output/tables/gender_norm_index.txt, output/tables/prognostic_gates.txt, output/tables/welfare_margin_triangulation.txt
# Imports local modules: annual_firstborn_gender, gbm_feature_importance, paper_style
# Run order: step 56 in run_all.sh
# ==========================================================================

"""Paper figures, one function per figure, all on `paper_style`.

Every number is parsed from the result table the caption cites; nothing is
re-estimated and nothing is typed in.  Output: output/figures/paper/figN_*.{png,pdf}.

    .venv/bin/python scripts/paper_figures.py            # all
    .venv/bin/python scripts/paper_figures.py fig3       # one
"""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (EXPOSED, GREY, INK, LIGHT, POST, ROOT, TABLES, UNEXPOSED, Z, band,  # noqa: E402
                         ci_points, new_fig, panel, parse_es, plt, post_shade, save,
                         shock_line, year_axis, zero_line)

OUT = ROOT / "output"


def se_from_p(b, p):
    """Back out a normal-approximation SE from a coefficient and two-sided p."""
    return np.abs(np.asarray(b)) / norm.ppf(1 - np.asarray(p) / 2)


# ── Figure 3: event study, daughter-first vs son-first ──────────────────────
def fig3():
    """annual_firstborn_gender.txt — 'Annual event study (T x year, omitted 2015)'."""
    fig, axes = new_fig("double", ncols=2, sharey=True)
    arms = (("[firstborn DAUGHTER]", "Firstborn daughter"), ("[firstborn SON]", "Firstborn son"))
    for ax, (hdr, lab), letter in zip(axes, arms, "AB"):
        yrs, b, se, _ = parse_es("annual_firstborn_gender.txt", hdr, after="Annual event study")
        post_shade(ax)
        zero_line(ax)
        shock_line(ax)
        ci_points(ax, yrs, b, se, color=EXPOSED, ref=2015)
        year_axis(ax)
        panel(ax, letter, lab)
    axes[0].set_ylim(-9, 9)
    axes[0].set_yticks(range(-8, 9, 4))
    axes[0].set_ylabel("Exposure × year, percentage points")
    axes[0].text(2017.7, 8.2, "post period", color=GREY, fontsize=7.5, va="top")
    save(fig, "fig3_event_study")


# ── Figure 1: first stage in second-hand prices ─────────────────────────────
def fig1():
    """Mean within-prefecture log price change vs 2015, exposed vs unexposed; balanced
    2015–22 set solid, balanced 2012–22 set dotted for 2012–15 (same data code as
    fig1_first_stage.py)."""
    ad = pd.read_csv(OUT / "analysis_data.csv", usecols=["city_code", "is_regulated", "treat_hpr"])
    cities = (ad.dropna(subset=["city_code", "treat_hpr"]).query("is_regulated == 0")
                .drop_duplicates("city_code")[["city_code", "treat_hpr"]].rename(columns={"treat_hpr": "T"}))
    cities["city_code"] = cities["city_code"].astype(int)
    hp = pd.read_csv(OUT / "housing_price_city.csv").merge(cities, on="city_code")

    def balanced(y0, y1):
        yrs = list(range(y0, y1 + 1))
        sub = hp[hp["year"].isin(yrs)]
        ok = sub.groupby("city_code")["year"].nunique()
        sub = sub[sub["city_code"].isin(ok[ok == len(yrs)].index)].copy()
        base = sub[sub["year"] == 2015].set_index("city_code")["log_hp_city"]
        sub["rel"] = sub["log_hp_city"] - sub["city_code"].map(base)
        g = sub.groupby(["T", "year"])["rel"].agg(["mean", "sem", "count"]).reset_index()
        n = sub.drop_duplicates("city_code")["T"].value_counts()
        return g, int(n.get(1, 0)), int(n.get(0, 0))

    gm, nT, nC = balanced(2015, 2022)
    gp, _, _ = balanced(2012, 2022)
    fig, ax = new_fig("single")
    post_shade(ax)
    zero_line(ax)
    shock_line(ax)
    for T, col, lab in ((1, EXPOSED, f"Exposed prefectures (n = {nT})"),
                        (0, UNEXPOSED, f"Unexposed prefectures (n = {nC})")):
        s = gm[gm["T"] == T]
        band(ax, s["year"], s["mean"], s["sem"], col, label=lab)
        sp = gp[(gp["T"] == T) & (gp["year"] <= 2015)]
        ax.plot(sp["year"], sp["mean"], ls=":", color=col, lw=1.0, zorder=2)
    ax.set_xlim(2011.6, 2022.6)
    ax.set_xticks(range(2012, 2023, 2))
    ax.set_ylabel("Log second-hand price, change from 2015")
    ax.legend(loc="upper left")
    save(fig, "fig1_first_stage")


# ── Figure 2: raw second-birth hazards ──────────────────────────────────────
def fig2():
    """Unadjusted annual hazard of the one-child risk set by arm × firstborn sex."""
    from annual_firstborn_gender import build_hazard_panel
    df = build_hazard_panel()
    yr = (df.groupby(["son", "T", "year"])["y"].mean() * 100).reset_index()
    fig, axes = new_fig("double", ncols=2, sharey=True)
    for ax, k, lab, letter in zip(axes, (0, 1), ("Firstborn daughter", "Firstborn son"), "AB"):
        post_shade(ax)
        shock_line(ax)
        for a, col, name in ((1, EXPOSED, "Exposed"), (0, UNEXPOSED, "Unexposed")):
            s = yr[(yr["son"] == k) & (yr["T"] == a)]
            ax.plot(s["year"], s["y"], color=col, marker="o", label=name, zorder=3)
        year_axis(ax, lo=2008.5, hi=2022.6, ticks=(2009, 2012, 2015, 2018, 2021))
        panel(ax, letter, lab)
    axes[0].set_ylim(0, 16)
    axes[0].set_ylabel("Second-birth rate, percent")
    axes[0].legend(loc="upper left")
    save(fig, "fig2_raw_hazards")


# ── Figure 4: norm terciles ─────────────────────────────────────────────────
def fig4():
    """gender_norm_index.txt — 'Tercile gradient, daughter arm T:post'; effect as % of
    the tercile's pre-shock hazard, 95 % interval from the printed p (normal approx.)."""
    text = (TABLES / "gender_norm_index.txt").read_text()
    pat = re.compile(r"(son_norm|trad_pc1)\s+T(\d) \w+:\s+([+-]\d+\.\d+) \((?:SE \d+\.\d+, )?p=(\d+\.\d+)\)\s+pre-hazard=(\d+\.\d+)")
    rows = pat.findall(text[text.index("Tercile gradient"):])
    fig, axes = new_fig("double", ncols=2, sharey=True)
    for ax, key, lab, letter in zip(axes, ("son_norm", "trad_pc1"),
                                    ("Lineage-norm index", "Traditional-attitudes index"), "AB"):
        r = [x for x in rows if x[0] == key]
        b = np.array([float(x[2]) for x in r]); p = np.array([float(x[3]) for x in r])
        pre = np.array([float(x[4]) for x in r])
        se = se_from_p(b, p)
        x = np.arange(3)
        zero_line(ax)
        ci_points(ax, x, 100 * b / pre, 100 * se / pre, color=EXPOSED, connect=False)
        for xi, pi in zip(x, p):
            ax.text(xi, -78, f"p = {pi:.2f}" if pi >= 0.01 else "p < 0.01", ha="center", va="top",
                    color=GREY, fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(["Low", "Middle", "High"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_xlabel(f"Third of prefectures by {lab.lower()}")
        panel(ax, letter, lab)
    axes[0].set_ylim(-85, 75)
    axes[0].set_ylabel("Exposure × Post, % of pre-shock birth rate")
    save(fig, "fig4_norm_terciles")


# ── Figure 5: GATES by propensity quintile ──────────────────────────────────
def fig5():
    """prognostic_gates.txt — 'GATES by propensity quintile' rows G1..G5."""
    text = (TABLES / "prognostic_gates.txt").read_text()
    pat = re.compile(r"^\s+G(\d)\s+([+-]\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", re.M)
    rows = pat.findall(text[text.index("GATES by propensity quintile"):text.index("G5 - G1")])
    g = np.array([float(r[1]) for r in rows]) * 100
    p = np.array([float(r[2]) for r in rows])
    base = np.array([float(r[3]) for r in rows]) * 100
    se = se_from_p(g, p)
    fig, ax = new_fig("single")
    x = np.arange(1, 6)
    zero_line(ax)
    ci_points(ax, x, g, se, color=EXPOSED, connect=False)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{k}\n({b:.1f})" for k, b in zip(x, base)])
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(-7.5, 5)
    ax.set_xlabel("Fifth of predicted likelihood of a second birth (pre-shock birth rate, %)")
    ax.set_ylabel("Exposure × Post, percentage points")
    save(fig, "fig5_gates")


# ── Figure 6: SHAP beeswarm ─────────────────────────────────────────────────
def fig6():
    """output/gbm_shap_values.csv (cached by gbm_feature_importance.py): SHAP values in
    pp of hazard + feature values on held-out prefectures.  Own beeswarm so the
    typeface, size and palette match the other figures."""
    from matplotlib.colors import LinearSegmentedColormap
    from gbm_feature_importance import LAB
    d = pd.read_csv(OUT / "gbm_shap_values.csv")
    feats = [c[6:] for c in d.columns if c.startswith("shap__")]
    sv = d[[f"shap__{f}" for f in feats]].values
    X = d[[f"x__{f}" for f in feats]].values
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:12]
    cmap = LinearSegmentedColormap.from_list("paper_seq", ["#E3DDEB", "#8E79B5", "#3B1F5E"])
    rng = np.random.RandomState(0)
    fig, ax = new_fig("tall")
    for row, j in enumerate(order[::-1]):
        v, x = sv[:, j], X[:, j]
        ok = ~np.isnan(x)
        pct = np.full(len(x), 0.5)
        pct[ok] = pd.Series(x[ok]).rank(pct=True).values
        # density-based vertical spread (shap-style): rank within fine x-bins
        bins = np.digitize(v, np.quantile(v, np.linspace(0, 1, 60)))
        off = np.zeros(len(v))
        for bnum in np.unique(bins):
            idx = np.where(bins == bnum)[0]
            n = len(idx)
            spread = min(0.38, 0.012 * n)
            off[idx] = np.linspace(-spread, spread, n) if n > 1 else 0.0
            off[idx] = off[idx][rng.permutation(n)]
        ax.scatter(v, row + off, c=pct, cmap=cmap, s=3, alpha=0.6, lw=0, zorder=3, rasterized=True)
    ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(order)))
    lab = {**LAB, "city_son_norm": "Prefecture lineage-norm index", "ideal_kids": "Stated ideal number of children",
           "city_urban_share": "Prefecture urban share", "city_linc": "Prefecture mean income"}
    ax.set_yticklabels([lab.get(feats[j], feats[j]) for j in order[::-1]])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Contribution to predicted second-birth rate, percentage points")
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=8)
    cb.outline.set_visible(False)
    save(fig, "fig6_shap_beeswarm")


# ── Figure 7: pre-train / post-predict ──────────────────────────────────────
def fig7():
    """output/pretrain_postpredict_preds.csv + prognostic_scores.csv; same estimation
    code as fig_pretrain_postpredict.py (explicit T×year dummies, city+year FE)."""
    import statsmodels.formula.api as smf
    FE = " + C(city) + C(year)"

    def es(d, yvar, base=2015):
        d = d.copy(); years = sorted(d["year"].unique()); terms = []
        for yy in years:
            if yy == base:
                continue
            d[f"yr{yy}"] = (d["year"] == yy).astype(int); terms.append(f"T:yr{yy}")
        m = smf.ols(f"{yvar} ~ {' + '.join(terms)}{FE}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        return pd.DataFrame([(yy, 0.0, 0.0) if yy == base else (yy, m.params[f"T:yr{yy}"], m.bse[f"T:yr{yy}"])
                             for yy in years], columns=["year", "b", "se"])

    def gates(d, yvar):
        tri = " + ".join(f"T:post:g{k}" for k in range(1, 6))
        two = " + ".join(f"T:g{k} + post:g{k} + g{k}" for k in range(1, 5))
        m = smf.ols(f"{yvar} ~ {tri} + {two}{FE}", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["city"]})
        return (np.array([m.params[f"T:post:g{k}"] for k in range(1, 6)]),
                np.array([m.bse[f"T:post:g{k}"] for k in range(1, 6)]))

    d = pd.read_csv(OUT / "pretrain_postpredict_preds.csv")
    d["yhat"] = d["p_ens_top3"]; d["e"] = d["y"] - d["yhat"]
    da = d[d["son"] == 0]
    ey, ep, ee = es(da, "y"), es(da, "yhat"), es(da, "e")
    dg = d.merge(pd.read_csv(OUT / "prognostic_scores.csv")[["pid", "grp"]], on="pid", how="inner")
    for k in range(1, 6):
        dg[f"g{k}"] = (dg["grp"] == k).astype(int)
    gy, gys = gates(dg, "y"); ge, ges = gates(dg, "e")

    fig, (a, b) = new_fig("double", ncols=2, gridspec_kw={"width_ratios": [1.5, 1]})
    post_shade(a); zero_line(a); shock_line(a)
    a.plot(ep["year"], ep["b"] * 100, color=GREY, lw=1.6, zorder=2, label="Predicted (composition)")
    for df_, col, fill, dx, lab in ((ey, INK, "white", -0.14, "Second-birth rate"),
                                    (ee, EXPOSED, EXPOSED, +0.14, "Residual after prediction")):
        x = df_["year"].values + dx; bb = df_["b"].values * 100; ss = df_["se"].values * 100
        a.plot([x, x], [bb - Z * ss, bb + Z * ss], color=col, lw=0.8, zorder=3)
        a.plot(x, bb, ls="none", marker="o", mfc=fill, mec=col, mew=0.9, zorder=4, label=lab)
    year_axis(a, lo=2008.5, hi=2022.6, ticks=(2009, 2012, 2015, 2018, 2021))
    a.set_ylim(-8, 7.5)
    a.set_ylabel("Exposure × year, percentage points")
    a.legend(loc="lower left")
    panel(a, "A", "Daughter-first families, year by year")

    x = np.arange(1, 6)
    zero_line(b)
    for g, s, col, fill, dx, lab in ((gy, gys, INK, "white", -0.16, "Raw"),
                                     (ge, ges, EXPOSED, EXPOSED, +0.16, "Residual")):
        b.plot([x + dx, x + dx], [(g - Z * s) * 100, (g + Z * s) * 100], color=col, lw=0.8, zorder=3)
        b.plot(x + dx, g * 100, ls="none", marker="o", mfc=fill, mec=col, mew=0.9, zorder=4, label=lab)
    b.set_xticks(x); b.set_xticklabels([f"G{k}" for k in x]); b.set_xlim(0.4, 5.6)
    b.set_xlabel("Fifth of predicted likelihood")
    b.set_ylabel("Exposure × Post, percentage points")
    b.legend(loc="lower left")
    panel(b, "B", "By fifth of predicted likelihood")
    save(fig, "fig7_pretrain_postpredict")


# ── Figure 8: welfare ───────────────────────────────────────────────────────
def fig8():
    """welfare_margin_triangulation.txt — WANTS-2nd / CONTENT T:post rows (panel A) and
    'confidence DID by propensity quintile' rows (panel B; SE backed out from p)."""
    text = (TABLES / "welfare_margin_triangulation.txt").read_text()
    def row(header):
        m = re.search(re.escape(header) + r".*?\n\s+T:post\s+([+-]\d+\.\d+)\s+\(SE (\d+\.\d+)", text, re.S)
        return float(m.group(1)), float(m.group(2))
    bw, sw = row("[WANTS-2nd"); bc, sc = row("[CONTENT")
    q = re.findall(r"G(\d): n=[\d,]+\s+T:post=([+-]\d+\.\d+) \(p=(\d+\.\d+)\)",
                   text[text.index("confidence DID by propensity quintile"):])
    bq = np.array([float(r[1]) for r in q]); pq = np.array([float(r[2]) for r in q])
    sq = se_from_p(bq, pq)

    fig, (a, b) = new_fig("double", ncols=2, sharey=True, gridspec_kw={"width_ratios": [1, 1.6]})
    zero_line(a)
    ci_points(a, [0, 1], [bw, bc], [sw, sc], color=EXPOSED, connect=False)
    a.set_xticks([0, 1]); a.set_xticklabels(["Wants a\nsecond child", "Content\nwith one"])
    a.set_xlim(-0.6, 1.6)
    a.set_ylabel("Exposure × Post on confidence in the future")
    panel(a, "A", "By stated demand")
    zero_line(b)
    x = np.arange(1, 6)
    ci_points(b, x, bq, sq, color=EXPOSED, connect=False)
    b.set_xticks(x); b.set_xticklabels([f"G{k}" for k in x]); b.set_xlim(0.4, 5.6)
    b.set_xlabel("Fifth of predicted likelihood of a second birth")
    panel(b, "B", "By fifth of predicted likelihood")
    a.set_ylim(-0.6, 0.25)
    save(fig, "fig8_welfare")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
        "fig6": fig6, "fig7": fig7, "fig8": fig8}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
