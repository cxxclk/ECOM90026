# ==========================================================================
# REPLICATION PACKAGE — fig_pretrain_calibration.py
# Tier: SUPPLEMENTARY
# Paper items: Figure E1 (Calibration of the score)
# Reads: output/pretrain_postpredict_preds.csv
# Writes: (nothing on disk)
# Imports local modules: make_presentation_figures
# Run order: step 57 in run_all.sh
# ==========================================================================

"""Reliability (calibration) figure for the pre-trained hazard model.

Decile bins of OOF yhat (cut points from PRE rows), realized second-birth
rate vs mean predicted, for three row sets: pre 2009-15 (training period,
OOF), post 2018-22 CONTROL (temporal holdout: never treated, time-shifted)
and post 2018-22 TREATED. Left: all one-child rows; right: daughter-first.
Writes output/figures/pretrain_calibration.png
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_presentation_figures import ACC, GREY, INK, RUST, plt, panel_label  # noqa
OUT = os.path.join(ROOT, "output")
WIN = os.environ.get("PP_WIN", "ens_top3")


def reliab(d, p, bins):
    b = np.clip(np.searchsorted(bins, d[p].values, side="right"), 1, len(bins) - 1)
    g = d.assign(b=b).groupby("b").agg(pred=(p, "mean"), real=("y", "mean"), n=("y", "size"))
    g["se"] = np.sqrt(g["real"] * (1 - g["real"]) / g["n"])
    return g


def panel(ax, d, p, title):
    pre = d[d["year"] <= 2015]
    bins = np.quantile(pre[p], np.linspace(0, 1, 11)); bins[0] = -1; bins[-1] = 2
    sets = [("2009–15, training period (held-out prefectures)", pre, INK, "o"),
            ("2018–22, unexposed prefectures", d[(d["post"] == 1) & (d["T"] == 0)], GREY, "s"),
            ("2018–22, exposed prefectures", d[(d["post"] == 1) & (d["T"] == 1)], ACC, "D")]
    lim = 0
    for lab, dd, col, mk in sets:
        g = reliab(dd, p, bins)
        ax.errorbar(g["pred"] * 100, g["real"] * 100, yerr=1.96 * g["se"] * 100, fmt=mk,
                    color=col, ms=5, mfc=col if col != INK else "white", mec=col, lw=1, capsize=0, label=lab)
        ax.plot(g["pred"] * 100, g["real"] * 100, color=col, lw=1.2, alpha=0.8)
        lim = max(lim, (g["pred"].max()) * 100, (g["real"] + 1.96 * g["se"]).max() * 100)
    lim = np.ceil(lim / 5) * 5
    ax.plot([0, lim], [0, lim], color=INK, lw=0.7, ls=(0, (3, 3)))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted second-birth rate in tenth, %"); ax.set_ylabel("Observed second-birth rate, %")
    ax.set_aspect("equal")
    panel_label(ax, title)


def main():
    d = pd.read_csv(os.path.join(OUT, "pretrain_postpredict_preds.csv"))
    p = f"p_{WIN}"
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    panel(a, d, p, "A  All one-child woman-years")
    panel(b, d[d["son"] == 0], p, "B  Daughter-first")
    a.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(w_pad=2.5)
    path = os.path.join(OUT, "figures", "pretrain_calibration.png")
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.08, facecolor="white", transparent=False)
    print("wrote", path)
    # numbers for the record
    for lab, dd in [("pre", d[d.year <= 2015]), ("postC", d[(d.post == 1) & (d["T"] == 0)]),
                    ("postT", d[(d.post == 1) & (d["T"] == 1)])]:
        print(f"{lab:<6} mean pred {dd[p].mean()*100:.2f}%  realized {dd['y'].mean()*100:.2f}%  "
              f"ratio {dd['y'].mean()/dd[p].mean():.2f}")


if __name__ == "__main__":
    main()
