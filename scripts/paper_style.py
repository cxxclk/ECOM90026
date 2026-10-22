# ==========================================================================
# REPLICATION PACKAGE — paper_style.py
# Tier: HELPER (imported by other scripts, not run directly)
# Paper items: none directly (support module)
# Reads: (via imported modules)
# Writes: (nothing on disk)
# Imports local modules: none
# Run order: not run directly
# ==========================================================================

"""Shared figure style for the paper (import before any plotting).

One place for typeface, sizes, palette, panel geometry and the small helpers
every paper figure uses.  Rules:

  * typeface = Latin Modern Roman (the PDF body font), maths in Computer Modern;
  * base 9 pt, ticks 8 pt, panel letters bold 9 pt; no in-figure titles
    (the caption carries the description);
  * two group colours only: EXPOSED (blue) and UNEXPOSED (orange); within a
    group, series differ by line style / marker fill, never by a third colour;
  * left and bottom spines only, no grid, 95 % bands at 12 % alpha;
  * post period = light grey span from 2017.5; the shock itself = dotted
    vertical at 2016.75, both drawn the same way in every time-series panel;
  * geometry: SINGLE 6.3 x 3.4 in, DOUBLE 6.3 x 3.0 in, TALL 6.3 x 4.5 in
    (6.3 in = text width of the A4 build);
  * saved as PNG at 300 dpi (what the markdown/pandoc build embeds) and as PDF.

Numbers are never typed into a figure script: every figure parses the result
table it cites (see `parse_es`, `parse_coef`).
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "figures" / "paper"
TABLES = ROOT / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── typeface ────────────────────────────────────────────────────────────────
_LM = Path("/usr/share/texmf/fonts/opentype/public/lm")
for _f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic", "lmroman10-bolditalic"):
    _p = _LM / f"{_f}.otf"
    if _p.exists():
        fm.fontManager.addfont(str(_p))
_FAMILY = "Latin Modern Roman" if any(f.name == "Latin Modern Roman" for f in fm.fontManager.ttflist) else "STIXGeneral"

# ── palette ─────────────────────────────────────────────────────────────────
INK = "#1F1F1F"          # text, axes, neutral series
GREY = "#8A8A8A"         # secondary text, reference lines
LIGHT = "#D4D4D4"        # faint rules
POST = "#F0F0F0"         # post-period span
EXPOSED = "#1F4E79"      # treated / exposed prefectures
UNEXPOSED = "#B5651D"    # control / unexposed prefectures
SHOCK = "#8A8A8A"        # the 2016 marker (neutral, not a third accent)
Z = 1.96

plt.rcParams.update({
    "font.family": _FAMILY, "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": INK, "ytick.color": INK,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 100, "savefig.dpi": 300,
})

# ── geometry ────────────────────────────────────────────────────────────────
W = 6.3
SIZES = {"single": (W, 3.4), "double": (W, 3.0), "tall": (W, 4.5), "wide": (W, 2.6)}


def new_fig(kind="single", ncols=1, sharey=False, **kw):
    fig, axes = plt.subplots(1, ncols, figsize=SIZES[kind], sharey=sharey, **kw)
    return fig, axes


def panel(ax, letter, text=""):
    """Bold panel letter at top-left, optional short descriptor after it."""
    ax.set_title(f"{letter}. {text}" if text else letter, loc="left", fontweight="bold", pad=6)


def zero_line(ax):
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)


def post_shade(ax, start=2017.5, end=2022.6):
    ax.axvspan(start, end, color=POST, lw=0, zorder=0)


def shock_line(ax, x=2016.75, label=None, y=None):
    ax.axvline(x, color=SHOCK, lw=0.7, ls=":", zorder=1)
    if label:
        yy = y if y is not None else ax.get_ylim()[1]
        ax.text(x - 0.15, yy, label, color=GREY, fontsize=7.5, ha="right", va="top")


def year_axis(ax, lo=2009.4, hi=2022.6, ticks=(2010, 2013, 2016, 2019, 2022)):
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(ticks))


def ci_points(ax, x, b, se, color=EXPOSED, connect=True, hollow=None, ref=None):
    """Point estimates with 95 % whiskers.  `hollow` = x values drawn as open
    markers; `ref` = x value to draw as the omitted reference (open marker at 0)."""
    x, b, se = map(np.asarray, (x, b, se))
    hollow = set(hollow or [])
    if connect:
        ax.plot(x, b, color=color, lw=0.7, alpha=0.55, zorder=2)
    for xi, bi, si in zip(x, b, se):
        ax.plot([xi, xi], [bi - Z * si, bi + Z * si], color=color, lw=0.8, zorder=3,
                solid_capstyle="butt")
        ax.plot(xi, bi, "o", mfc="white" if xi in hollow else color, mec=color, mew=0.9, zorder=4)
    if ref is not None:
        ax.plot(ref, 0, "o", mfc="white", mec=color, mew=0.9, zorder=4)


def band(ax, x, m, se, color, ls="-", label=None, marker="o"):
    """Mean series with a 95 % band."""
    x, m, se = map(np.asarray, (x, m, se))
    ax.fill_between(x, m - Z * se, m + Z * se, color=color, alpha=0.12, lw=0, zorder=1)
    ax.plot(x, m, ls=ls, marker=marker, color=color, label=label, zorder=3)


def save(fig, stem):
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", FIG_DIR / f"{stem}.png")


# ── table parsers ───────────────────────────────────────────────────────────
_ES = re.compile(r"T x (\d{4})\s+([+-]?\d+\.\d+)\s+\(SE (\d+\.\d+), p=(\d+\.\d+)\)")
_CO = re.compile(r"^\s*(\S+)\s+([+-]?\d+\.\d+)\s+\(SE (\d+\.\d+), p=(\d+\.\d+)\)")


def _block(text, header, after=None):
    """Text from `header` (searched after the optional `after` anchor) to the
    next blank line that is followed by a non-indented line."""
    start = 0
    if after:
        start = text.find(after)
        if start < 0:
            raise KeyError(after)
    i = text.find(header, start)
    if i < 0:
        raise KeyError(header)
    rest = text[i + len(header):]
    m = re.search(r"\n\s*\n(?=\S)", rest.lstrip("\n"))
    return rest.lstrip("\n")[: m.start()] if m else rest


def parse_es(table, header, after=None, scale=100.0):
    """Event-study rows 'T x YYYY  coef (SE se, p=p)' under `header` -> (years, b, se, p)."""
    text = (TABLES / table).read_text()
    rows = _ES.findall(_block(text, header, after))
    if not rows:
        raise ValueError(f"no event-study rows under {header!r} in {table}")
    yrs = np.array([int(r[0]) for r in rows])
    b = np.array([float(r[1]) for r in rows]) * scale
    se = np.array([float(r[2]) for r in rows]) * scale
    p = np.array([float(r[3]) for r in rows])
    return yrs, b, se, p


def parse_coef(table, header, key, after=None, scale=100.0):
    """Single row '<key>  coef (SE se, p=p)' under `header` -> (b, se, p)."""
    text = (TABLES / table).read_text()
    for line in _block(text, header, after).splitlines():
        m = _CO.match(line)
        if m and m.group(1) == key:
            return float(m.group(2)) * scale, float(m.group(3)) * scale, float(m.group(4))
    raise KeyError(f"{key} under {header!r} in {table}")
