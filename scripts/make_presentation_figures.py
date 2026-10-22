# ==========================================================================
# REPLICATION PACKAGE — make_presentation_figures.py
# Tier: HELPER (imported by other scripts, not run directly)
# Paper items: none directly (support module)
# Reads: (via imported modules)
# Writes: output/tables/annual_firstborn_gender.txt, output/tables/annual_parity_hazard.txt, output/tables/couple_intentions_pooled.txt, output/tables/gender_norm_index.txt, output/tables/prognostic_gates.txt
# Imports local modules: none
# Run order: not run directly
# ==========================================================================

"""Presentation figures for the 2026-09-01 progress deck, journal style.

Conventions (AER/QJE-type figures): serif type, point estimates with 95 %
confidence intervals rather than bars for coefficient plots, left/bottom spines
only, no gridlines, panel letters, one accent colour on a black/grey base,
transparent background. Numbers are transcribed from the result tables named
in each block (output/tables/*.txt); nothing is re-estimated here.

Output: output/figures/pres_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, GREY, LIGHT = "#1C1F26", "#7C8290", "#C4C8CF"
ACC, RUST = "#1F5F7A", "#A8452B"          # deck accent (treated / daughter), post marker

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
})
Z = 1.96


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", pad=8)


def zero(ax):
    ax.axhline(0, color=INK, lw=0.7, ls=(0, (3, 3)))


def save(fig, name):
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", pad_inches=0.08, transparent=True)
    plt.close(fig)
    print("wrote", OUT / name)


def dot_whisker(ax, x, b, se, accent_idx=(), labels=None, label_dy=0.0):
    """Point estimate + 95 % CI. Accent points filled in the deck accent, others hollow black."""
    for i, (xi, bi, si) in enumerate(zip(x, b, se)):
        acc = i in accent_idx
        col = ACC if acc else INK
        ax.plot([xi, xi], [bi - Z * si, bi + Z * si], color=col, lw=1.1, solid_capstyle="butt", zorder=3)
        ax.plot([xi - 0.08, xi + 0.08], [bi - Z * si] * 2, color=col, lw=1.1, zorder=3)
        ax.plot([xi - 0.08, xi + 0.08], [bi + Z * si] * 2, color=col, lw=1.1, zorder=3)
        ax.plot(xi, bi, marker="o", ms=6.5, mfc=col if acc else "white", mec=col, mew=1.2, zorder=4)
        if labels and labels[i]:
            side = -1 if acc else 1
            ax.text(xi + side * 0.14, bi + label_dy, labels[i], color=col, fontsize=9.5, va="center",
                    ha="right" if acc else "left", fontweight="bold" if acc else "normal")


# 1. Parity — annual_parity_hazard.txt (Post = 2018)
def fig_parity():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(3); b = [4.0, -1.4, 0.0]; se = [2.6, 0.72, 0.6]
    dot_whisker(ax, x, b, se, accent_idx=(1,), labels=["+4.0", "−1.4", "0.0"])
    zero(ax)
    ax.set_xticks(x); ax.set_xticklabels(["Childless\n(first birth)", "One child\n(second birth)", "Two or more\n(third birth)"])
    ax.set_ylim(-5, 10); ax.set_xlim(-0.5, 2.6)
    ax.set_ylabel("Treat × Post, percentage points")
    ax.text(0, -4.6, "pre-trends fail", ha="center", color=GREY, fontsize=8.5)
    ax.text(1, -4.6, "p = 0.052", ha="center", color=GREY, fontsize=8.5)
    ax.text(2, -4.6, "p = 0.82", ha="center", color=GREY, fontsize=8.5)
    save(fig, "pres_parity.png")


# 2. Event study — annual_firstborn_gender.txt (T × year, 2015 omitted)
def fig_event_study():
    yrs = np.arange(2010, 2023)
    d = np.array([-0.03, 1.18, -0.59, 0.09, 1.52, 0, 2.46, 3.10, -0.70, 0.70, -3.07, -3.62, -1.97])
    dse = np.array([1.72, 1.74, 1.70, 1.73, 1.74, 0, 1.94, 1.89, 1.87, 2.26, 1.93, 2.10, 2.02])
    s = np.array([-1.06, -1.14, -0.93, -2.45, -0.18, 0, -1.12, -0.11, -0.62, -1.07, -0.56, -2.28, -3.07])
    sse = np.array([1.32, 1.11, 1.22, 1.12, 1.10, 0, 1.60, 1.70, 1.73, 1.62, 1.38, 1.65, 1.51])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, y, se, col, lab in ((axes[0], d, dse, ACC, "A. Firstborn daughter"), (axes[1], s, sse, INK, "B. Firstborn son")):
        for yr, yi, si in zip(yrs, y, se):
            if yr == 2015:
                ax.plot(yr, 0, marker="o", ms=5, mfc="white", mec=col, mew=1.1, zorder=4)
                continue
            ax.plot([yr, yr], [yi - Z * si, yi + Z * si], color=col, lw=0.9, zorder=3)
            ax.plot(yr, yi, marker="o", ms=4.5, color=col, zorder=4)
        ax.plot(yrs, y, color=col, lw=0.8, alpha=0.6, zorder=2)
        zero(ax)
        ax.axvline(2017.5, color=RUST, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks([2010, 2013, 2016, 2019, 2022]); ax.set_xlim(2009.3, 2022.7)
        panel_label(ax, lab)
    axes[0].set_ylim(-9, 9)
    axes[0].set_ylabel("Coefficient on Treat × year, pp")
    axes[0].text(2017.8, 8.3, "Post", color=RUST, fontsize=8.5)
    save(fig, "pres_event_study.png")


# 3. Raw second-birth hazard — annual_firstborn_gender.txt (daughter arm, mean by T × year)
def fig_raw_hazard():
    yrs = np.arange(2009, 2023)
    c = [6.99, 6.56, 6.86, 6.86, 6.98, 5.84, 6.71, 9.10, 8.37, 7.66, 6.00, 5.30, 5.57, 1.93]
    t = [11.29, 9.82, 11.26, 9.85, 10.74, 11.41, 9.72, 14.91, 14.58, 10.02, 10.04, 5.49, 5.54, 3.42]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(yrs, t, color=ACC, lw=1.6, marker="o", ms=3.2, label="Treated (≤ 250 km)")
    ax.plot(yrs, c, color=INK, lw=1.3, ls=(0, (4, 2)), marker="s", ms=2.8, label="Control")
    ax.axvline(2017.5, color=RUST, lw=0.8, ls=(0, (2, 2)))
    ax.text(2017.7, 15.6, "Post", color=RUST, fontsize=8.5, va="top")
    ax.set_ylim(0, 16.5); ax.set_ylabel("Second-birth hazard, percent")
    ax.set_xticks([2009, 2012, 2015, 2018, 2021]); ax.set_xlim(2008.5, 2022.5)
    ax.legend(loc="lower left")
    save(fig, "pres_raw_hazard.png")


# 4. Norm terciles — gender_norm_index.txt (daughter arm T:post as % of pre hazard)
def fig_norm():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, vals, ps, lab in ((axes[0], [-2, -6, -28], ["p = 0.97", "p = 0.77", "p = 0.026"], "A. Son-norm index"),
                              (axes[1], [11, -17, -41], ["p = 0.71", "p = 0.28", "p = 0.018"], "B. Traditional-attitudes PC1")):
        x = np.arange(3)
        ax.bar(x, vals, width=0.5, color=["white", "white", ACC], edgecolor=[INK, INK, ACC], linewidth=0.9, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v - 2 if v < 0 else v + 1.5, f"{v:+d}%", ha="center", va="top" if v < 0 else "bottom",
                    color=ACC if i == 2 else INK, fontsize=9.5, fontweight="bold" if i == 2 else "normal")
            ax.text(i, -60, ps[i], ha="center", va="top", color=GREY, fontsize=8.5)
        zero(ax)
        ax.set_xticks(x); ax.set_xticklabels(["Low", "Middle", "High"])
        ax.set_xlabel("City tercile"); panel_label(ax, lab)
    axes[0].set_ylim(-68, 22); axes[0].set_ylabel("Treat × Post, % of pre-shock hazard")
    save(fig, "pres_norm_terciles.png")


# 5. Couple weights — couple_intentions_pooled.txt (∂Y/∂ideal × 100)
def fig_couple():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(2); w = 0.3
    # B3 implied marginal effects: husband control +0.0506, treated +0.0616
    # (T:ideal_h = +0.011 ns); wife control -0.0091, treated +0.0472
    h = [5.1, 6.2]; wf = [-0.9, 4.7]
    ax.bar(x - w / 2, h, w, color="white", edgecolor=INK, linewidth=0.9, label="Husband's ideal", zorder=3)
    ax.bar(x + w / 2, wf, w, color=[LIGHT, ACC], edgecolor=[INK, ACC], linewidth=0.9, label="Wife's ideal", zorder=3)
    for xi, v in zip(x - w / 2, h):
        ax.text(xi, v + 0.15, f"{v/100:+.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    for xi, v, col in zip(x + w / 2, wf, [INK, ACC]):
        ax.text(xi, v + 0.15 if v > 0 else v - 0.15, f"{v/100:+.3f}", ha="center", va="bottom" if v > 0 else "top",
                color=col, fontsize=9, fontweight="bold" if col == ACC else "normal")
    zero(ax)
    ax.set_xticks(x); ax.set_xticklabels(["Control areas", "Treated areas"])
    ax.set_ylim(-2.2, 7.5); ax.set_ylabel("∂P(birth) / ∂ stated ideal, × 100")
    ax.legend(loc="upper left")
    ax.text(1.16, 6.9, "T × ideal$_w$ = +0.056\n(p = 0.003)", ha="left", va="top", fontsize=9, color=ACC)
    save(fig, "pres_couple_weights.png")


# 6. GATES — prognostic_gates.txt
def fig_gates():
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    x = np.arange(5)
    b = [0.30, -0.47, -0.36, -2.90, -0.75]; se = [1.15, 0.70, 1.04, 1.72, 2.09]
    base = [0.006, 0.026, 0.071, 0.129, 0.193]
    dot_whisker(ax, x, b, se, accent_idx=(3,), labels=["+0.3", "−0.5", "−0.4", "−2.9 (p = 0.09)", "−0.8"])
    zero(ax)
    ax.set_xticks(x); ax.set_xticklabels([f"G{i+1}\nbase {bs:.3f}" for i, bs in enumerate(base)])
    ax.set_xlim(-0.5, 4.9); ax.set_ylim(-7.5, 4.8)
    ax.set_ylabel("Treat × Post, percentage points")
    ax.set_xlabel("Quintile of pre-2016 predicted second-birth propensity")
    save(fig, "pres_gates.png")


if __name__ == "__main__":
    fig_parity(); fig_event_study(); fig_raw_hazard(); fig_norm(); fig_couple(); fig_gates()
