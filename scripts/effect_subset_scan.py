# ==========================================================================
# REPLICATION PACKAGE — effect_subset_scan.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Beam search)
# Reads: (via imported modules)
# Writes: output/tables/effect_subset_scan.txt
# Imports local modules: cddf_spine
# Run order: step 40 in run_all.sh
# ==========================================================================

"""Treatment-effect subset scan: the most-suppressed DEFINABLE subgroup.

New classification approach #1 (2026-07-21 session 2). Instead of clustering
people (covariate space — 3x null) or predicting individual effects (CATE —
2x failed), search the INTERPRETABLE RULE SPACE directly: conjunctions of up
to 3 binary conditions (e.g. daughter-first & high-norm-city & below-median
income), scoring each rule by its DID t-statistic on the annual second-birth
hazard. The search maximizes suppression; multiple-testing is handled by a
CITY-LEVEL PERMUTATION test of the ENTIRE SEARCH (re-run the beam search on
each permuted treatment assignment, compare the observed extreme t to the
permutation distribution of extreme t's) — the scan-statistic answer to "what
is the most-affected definable group, and is it significant AFTER accounting
for having searched?".

Fast rule scoring: y is FE-residualized once (age/dur + city + year FE, no
treatment terms); a rule's DID = WLS of city-level (post minus pre) mean
residuals on T, weighted by city woman-years, HC0 t-stat. The winning rule is
then re-estimated with the full FE regression + event study. Caveats: the
permutation ignores spatial correlation of assignment (standard); conditions
with missing covariates exclude NaN rows on both sides of the cut.
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
import cddf_spine as cs                                  # noqa: E402

N_PERM = int(os.environ.get("SCAN_PERMS", 200))
BEAM = 8
MIN_ROWS = 3000
MIN_CITIES = 15
CTRL, FE = cs.CTRL, cs.FE

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def build_conditions(df):
    """(name, variable-group, row mask). Complementary cuts share a group so
    the beam never stacks two cuts of the same variable."""
    conds = []
    def add(name, var, mask):
        conds.append((name, var, mask.values if hasattr(mask, "values")
                      else mask))
    med = lambda c: df.drop_duplicates("pid")[c].median()

    add("daughter", "son", df["son"] == 0)
    add("son_first", "son", df["son"] == 1)
    for c, lab in [("city_son_norm", "norm"), ("trad_pc1", "trad")]:
        hi = df[c] >= df.drop_duplicates("pid")[c].quantile(2 / 3)
        lo = df[c] <= df.drop_duplicates("pid")[c].quantile(1 / 3)
        add(f"{lab}_hi", lab, hi & df[c].notna())
        add(f"{lab}_lo", lab, lo & df[c].notna())
    for c, lab in [("log_faminc_pc", "faminc"), ("log_inc", "owninc"),
                   ("hv_inc", "hvburden"), ("city_linc", "cityinc"),
                   ("housework", "housew"), ("birthy", "cohort"),
                   ("age_fb", "agefb"), ("eduy", "edu")]:
        m = med(c)
        add(f"{lab}_lo", lab, (df[c] <= m) & df[c].notna())
        add(f"{lab}_hi", lab, (df[c] > m) & df[c].notna())
    add("rural", "urban", df["urban"] == 0)
    add("urban", "urban", df["urban"] == 1)
    add("sonpref_hi", "spref", (df["son_pref"] >= 4) & df["son_pref"].notna())
    add("sonpref_lo", "spref", (df["son_pref"] <= 2) & df["son_pref"].notna())
    add("ideal2plus", "ideal", (df["ideal_kids"] >= 2) & df["ideal_kids"].notna())
    add("ideal_lt2", "ideal", (df["ideal_kids"] < 2) & df["ideal_kids"].notna())
    add("renter", "own", df["homeowner"] == 0)
    add("owner", "own", df["homeowner"] == 1)
    return conds


class Scanner:
    def __init__(self, df, conds):
        self.conds = conds
        self.cidx, self.cities = pd.factorize(df["city"])
        self.nC = len(self.cities)
        self.post = df["post"].values.astype(bool)
        self.yres = df["yres"].values
        self.cityT = df.groupby("city")["T"].first().reindex(self.cities).values
        self.cache = {}

    def city_stats(self, key, mask):
        """Per-city (delta, weight) for a rule; cached across permutations."""
        if key in self.cache:
            return self.cache[key]
        mp, mq = mask & self.post, mask & ~self.post
        s1 = np.bincount(self.cidx[mp], weights=self.yres[mp], minlength=self.nC)
        n1 = np.bincount(self.cidx[mp], minlength=self.nC)
        s0 = np.bincount(self.cidx[mq], weights=self.yres[mq], minlength=self.nC)
        n0 = np.bincount(self.cidx[mq], minlength=self.nC)
        ok = (n1 >= 5) & (n0 >= 5)
        d = np.where(ok, s1 / np.maximum(n1, 1) - s0 / np.maximum(n0, 1), np.nan)
        w = np.where(ok, n1 + n0, 0.0)
        res = (d, w, ok, int(mask.sum()))
        self.cache[key] = res
        return res

    def score(self, key, mask, Tlab):
        d, w, ok, nrows = self.city_stats(key, mask)
        if nrows < MIN_ROWS:
            return None
        use = ok
        t1, t0 = use & (Tlab == 1), use & (Tlab == 0)
        if t1.sum() < MIN_CITIES or t0.sum() < MIN_CITIES:
            return None
        dd, ww, TT = d[use], w[use], Tlab[use].astype(float)
        wm = np.average(TT, weights=ww)
        xc = TT - wm
        sxx = np.sum(ww * xc * xc)
        if sxx <= 0:
            return None
        b1 = np.sum(ww * xc * (dd - np.average(dd, weights=ww))) / sxx
        b0 = np.average(dd, weights=ww) - b1 * wm
        e = dd - b0 - b1 * TT
        var = np.sum((ww * xc * e) ** 2) / sxx ** 2
        if var <= 0:
            return None
        return b1 / np.sqrt(var), b1, nrows, int(t1.sum()), int(t0.sum())

    def beam_search(self, Tlab):
        """Return (best result, all evaluated depth-results) minimizing t."""
        results = {}
        def ev(combo):
            key = frozenset(n for n, _, _ in combo)
            if key in results:
                return
            mask = np.logical_and.reduce([m for _, _, m in combo])
            sc = self.score(key, mask, Tlab)
            if sc is not None:
                results[key] = sc
        for c in self.conds:
            ev((c,))
        frontier = sorted(results.items(), key=lambda kv: kv[1][0])[:BEAM]
        for depth in (2, 3):
            base = [set(k) for k, _ in frontier]
            for keyset in base:
                vars_used = {v for n, v, _ in self.conds if n in keyset}
                cur = [c for c in self.conds if c[0] in keyset]
                for c in self.conds:
                    if c[1] in vars_used or c[0] in keyset:
                        continue
                    ev(tuple(cur) + (c,))
            frontier = sorted(results.items(), key=lambda kv: kv[1][0])[:BEAM]
        best = min(results.items(), key=lambda kv: kv[1][0])
        return best, results


def main():
    df = cs.assemble()
    conds = build_conditions(df)
    emit("TREATMENT-EFFECT SUBSET SCAN — annual second-birth hazard DID")
    emit(f"rule space: conjunctions of <=3 of {len(conds)} conditions; "
         f"beam={BEAM}; constraints: >= {MIN_ROWS} woman-years, "
         f">= {MIN_CITIES} cities per arm")
    emit(f"inference: {N_PERM} city-level treatment permutations of the "
         f"FULL search (scan-statistic p)")

    sc = Scanner(df, conds)
    Tobs = sc.cityT
    (bkey, bval), results = sc.beam_search(Tobs)
    emit(f"\nOBSERVED search: {len(results)} admissible rules evaluated")
    top = sorted(results.items(), key=lambda kv: kv[1][0])[:10]
    emit("top-10 most-suppressed rules (screen t on residualized y):")
    emit(f"  {'rule':<44}{'t':>7}{'DID':>9}{'rows':>8}{'T/C cities':>11}")
    for k, (t, b, n, nt, nc) in top:
        emit(f"  {' & '.join(sorted(k)):<44}{t:>7.2f}{b:>+9.4f}{n:>8,}"
             f"{nt:>6}/{nc}")

    # permutation distribution of the search minimum t
    rng = np.random.RandomState(0)
    obs_min = bval[0]
    ext = 0
    perm_mins = []
    for p in range(N_PERM):
        Tp = Tobs.copy()
        rng.shuffle(Tp)
        (pk, pv), _ = sc.beam_search(Tp)
        perm_mins.append(pv[0])
        ext += pv[0] <= obs_min
        if (p + 1) % 25 == 0:
            print(f"  [perm {p+1}/{N_PERM}] running p = "
                  f"{(1 + ext) / (p + 2):.3f}", flush=True)
    pval = (1 + ext) / (N_PERM + 1)
    emit(f"\nscan-statistic inference: observed min t = {obs_min:.2f}; "
         f"permutation p = {pval:.3f}")
    emit(f"permutation min-t distribution: 5th pct {np.percentile(perm_mins, 5):.2f}, "
         f"median {np.median(perm_mins):.2f}, mean {np.mean(perm_mins):.2f}")

    # re-estimate the winning rule with the honest FE regression
    combo = [c for c in conds if c[0] in bkey]
    mask = np.logical_and.reduce([m for _, _, m in combo])
    sub = df[mask].copy()
    emit(f"\nWINNING RULE: {' & '.join(sorted(bkey))}")
    emit(f"  woman-years={len(sub):,}  women={sub['pid'].nunique():,}  "
         f"treated share={sub['T'].mean():.2f}  hazard={sub['y'].mean():.4f}")
    m = smf.ols(f"y ~ T:post + {CTRL}{FE}", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["city"]})
    emit(f"  FE-regression DID: {m.params['T:post']:+.4f} "
         f"(SE {m.bse['T:post']:.4f}, p={m.pvalues['T:post']:.3f})")
    ev = [y for y in range(2010, 2023) if y != 2015]
    for k in ev:
        sub[f"Ty{k}"] = sub["T"] * (sub["year"] == k).astype(int)
    terms = " + ".join(f"Ty{k}" for k in ev)
    me = smf.ols(f"y ~ {terms} + {CTRL}{FE}", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["city"]})
    emit("  event study (base 2015):")
    for k in ev:
        tag = " (pre)" if k < 2016 else ""
        emit(f"    T x {k}   {me.params[f'Ty{k}']:+.4f}  "
             f"(SE {me.bse[f'Ty{k}']:.4f}, p={me.pvalues[f'Ty{k}']:.3f}){tag}")

    # symmetric check: most POSITIVE rule (screen only)
    worst = max(results.items(), key=lambda kv: kv[1][0])
    emit(f"\nmost POSITIVE rule (screen only): "
         f"{' & '.join(sorted(worst[0]))}: t={worst[1][0]:.2f}, "
         f"DID={worst[1][1]:+.4f}")

    out = os.path.join(OUT, "tables", "effect_subset_scan.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
