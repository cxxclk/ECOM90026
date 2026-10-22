# ==========================================================================
# REPLICATION PACKAGE — trajectory_typology.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Three typologies)
# Reads: output/panel_individual.csv, output/semisup_classes.csv
# Writes: output/tables/trajectory_typology.txt
# Imports local modules: annual_parity_hazard, semisup_typology
# Run order: step 42 in run_all.sh
# ==========================================================================

"""Trajectory clustering (Bonhomme-Manresa-style) of PRE-shock life paths,
then the same post-shock DID screen (2026-07-16).

Unlike the level-based typologies (semisup_typology / semisup_methods_compare,
which cluster on WHERE a woman is in behavior space at baseline), this
clusters on the SHAPE of her pre-shock path — birth timing and economic
direction — the one method family the level sweep cannot replicate.

Trajectories (all dated <= 2015):
  parity path   parity at each year 2009..2015 (famconf roster, no missing)
  income path   log family income p.c. at waves 2010/2012/2014
  owner path    homeowner at waves 2010/2012/2014
  value path    log house value at waves 2010/2012/2014
Wave paths are row-interpolated (require >= 1 observed wave; the count of
observed waves is reported).

Feature variants:
  level  concatenated raw paths (level + shape together)      [BM-GFE proper]
  shape  within-woman information only: yearly birth increments 2010..2015 +
         demeaned econ paths (income direction, tenure change, value drift)
Algorithms: KMeans (BM canonical), Ward (kNN-structured), GMM; K by
silhouette over 3..8. Silhouette-best cut per variant goes through the
st.scan() DID screen; ARI vs the level baseline (output/semisup_classes.csv)
shows whether trajectories carry genuinely new grouping information.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import annual_parity_hazard as aph                  # noqa: E402
import semisup_typology as st                       # noqa: E402

from sklearn.preprocessing import StandardScaler    # noqa: E402
from sklearn.cluster import KMeans, AgglomerativeClustering  # noqa: E402
from sklearn.mixture import GaussianMixture         # noqa: E402
from sklearn.neighbors import kneighbors_graph      # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_rand_score  # noqa: E402

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)
st.emit = emit

SEED = 0
YEARS = list(range(2009, 2016))
WAVES = [2010, 2012, 2014]


def build_paths(d):
    """Per woman: parity path 2009-2015 + econ paths over waves 2010/12/14."""
    r = aph.LOAD_ROSTER[["pid", "cby"]]
    r = r[r["pid"].isin(d["pid"])]
    P = pd.DataFrame(index=d["pid"].values)
    for t in YEARS:
        s = r[r["cby"] <= t].groupby("pid").size()
        P[f"par_{t}"] = s.reindex(P.index).fillna(0)

    pan = pd.read_csv(os.path.join(OUT, "panel_individual.csv"),
                      usecols=["pid", "year", "fam_income_pc", "homeowner",
                               "house_value"])
    pan = pan[pan["year"].isin(WAVES) & pan["pid"].isin(d["pid"])]
    pan["linc"] = np.log1p(pan["fam_income_pc"].clip(lower=0))
    pan["own"] = pan["homeowner"].where(pan["homeowner"].isin([0, 1]))
    pan["lhv"] = np.log1p(pan["house_value"].where(pan["house_value"] >= 0))
    for var in ("linc", "own", "lhv"):
        w = pan.pivot_table(index="pid", columns="year", values=var,
                            aggfunc="first").reindex(P.index)
        w = w.reindex(columns=WAVES)
        n_obs = w.notna().sum(axis=1)
        w = w.interpolate(axis=1, limit_direction="both")
        for y in WAVES:
            P[f"{var}_{y}"] = w[y]
        P[f"{var}_nobs"] = n_obs
    emit("Wave-path coverage (# of 2010/12/14 waves observed):")
    for var in ("linc", "own", "lhv"):
        vc = P[f"{var}_nobs"].value_counts().sort_index()
        emit(f"  {var}: " + "  ".join(f"{int(k)}w={v:,}" for k, v in vc.items()))
    # require at least one observed income wave (economic path exists)
    keep = P["linc_nobs"] >= 1
    emit(f"  kept {keep.sum():,} / {len(P):,} women with >=1 income wave")
    P = P[keep]
    # residual NaN (own/lhv never observed) -> column median
    for c in P.columns:
        if P[c].isna().any():
            P[c] = P[c].fillna(P[c].median())
    return P


def feature_variants(P):
    lvl_cols = ([f"par_{t}" for t in YEARS] +
                [f"{v}_{y}" for v in ("linc", "own", "lhv") for y in WAVES])
    level = P[lvl_cols].values

    shp = pd.DataFrame(index=P.index)
    for t in YEARS[1:]:
        shp[f"dpar_{t}"] = P[f"par_{t}"] - P[f"par_{t-1}"]
    for v in ("linc", "own", "lhv"):
        block = P[[f"{v}_{y}" for y in WAVES]]
        dev = block.sub(block.mean(axis=1), axis=0)
        for y in WAVES:
            shp[f"{v}dev_{y}"] = dev[f"{v}_{y}"]
    shape = shp.values
    return {"level": StandardScaler().fit_transform(level),
            "shape": StandardScaler().fit_transform(shape)}


def cluster_grid(Fs):
    res = {}
    emit("\nCluster grid (K by silhouette over 3..8):")
    for name, Z in Fs.items():
        conn = kneighbors_graph(Z, 15, include_self=False)
        for algo in ("kmeans", "ward", "gmm"):
            best = (None, -2, None)
            for k in range(3, 9):
                if algo == "kmeans":
                    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
                elif algo == "ward":
                    lab = AgglomerativeClustering(
                        n_clusters=k, linkage="ward",
                        connectivity=conn).fit_predict(Z)
                else:
                    lab = GaussianMixture(k, covariance_type="full", n_init=3,
                                          random_state=SEED).fit(Z).predict(Z)
                sil = silhouette_score(Z, lab)
                if sil > best[1]:
                    best = (lab, sil, k)
            res[(name, algo)] = best
            emit(f"  {name:<6} {algo:<7} K={best[2]}  sil={best[1]:.3f}")
    return res


def traj_profile(d):
    """Class means: trajectory summaries + baseline descriptors."""
    g = d.groupby("cls")
    tab = pd.DataFrame({"n": g.size(), "treated": g["treat_hpr"].mean()})
    tab["par09"] = g["par_2009"].mean()
    tab["par15"] = g["par_2015"].mean()
    tab["births0915"] = (g["par_2015"].mean() - g["par_2009"].mean())
    tab["linc10"] = g["linc_2010"].mean()
    tab["linc14"] = g["linc_2014"].mean()
    tab["dlinc"] = tab["linc14"] - tab["linc10"]
    tab["own10"] = g["own_2010"].mean()
    tab["own14"] = g["own_2014"].mean()
    tab["lhv14"] = g["lhv_2014"].mean()
    for c in ("age14", "eduy", "urban", "ideal_kids", "son_pref"):
        tab[c] = g[c].mean()
    mothers = d[d["parity14"] >= 1]
    tab["fb_dghtr"] = 1 - mothers.groupby("cls")["fb_son"].mean()
    return tab.round(2)


def descriptor(prof, c):
    r = prof.loc[c]
    return (f"n={int(r['n'])},age={r['age14']:.0f},par09={r['par09']:.1f},"
            f"b0915={r['births0915']:.1f},dlinc={r['dlinc']:+.1f},"
            f"own14={r['own14']:.1f}")


def main():
    d = st.build_baseline()          # also primes aph.LOAD_ROSTER
    emit("TRAJECTORY TYPOLOGY (pre-shock life paths, BM-GFE style)")
    emit(f"  universe N={len(d):,}")

    P = build_paths(d)
    d = d[d["pid"].isin(P.index)].copy()
    d = d.merge(P, left_on="pid", right_index=True, how="left")
    Fs = feature_variants(P)
    res = cluster_grid(Fs)

    base = pd.read_csv(os.path.join(OUT, "semisup_classes.csv"),
                       usecols=["pid", "cls"]).rename(columns={"cls": "cls0"})
    d = d.merge(base, on="pid", how="left")
    emit("\nARI vs level-based baseline typology (oof+gmm, K=6):")
    ok = d["cls0"].notna().values
    for (name, algo), (lab, sil, k) in res.items():
        ari = adjusted_rand_score(d.loc[ok, "cls0"], lab[ok])
        emit(f"  {name:<6} {algo:<7} K={k}  sil={sil:.3f}  ARI={ari:.3f}")

    all_cells = []
    for name in ("level", "shape"):
        algo = max(("kmeans", "ward", "gmm"), key=lambda a: res[(name, a)][1])
        lab, sil, k = res[(name, algo)]
        tag = f"{name}+{algo}"
        dd = d.copy()
        dd["cls"] = lab
        emit("\n" + "=" * 72)
        emit(f"VARIANT {tag}  (K={k}, sil={sil:.3f})")
        prof = traj_profile(dd)
        emit(prof.to_string())
        tab = st.scan(dd, verbose=False)
        tab["method"] = tag
        all_cells.append(tab)
        emit(f"\n  top cells (of {len(tab)}), Post=2018 DID:")
        for r in tab.head(8).itertuples():
            emit(f"    cls {r.cls} [{descriptor(prof, r.cls)}] {r.outcome:<15}"
                 f" {r.coef:+.4f} (p={r.p:.3f}, pre_viol={r.pre_viol},"
                 f" fdr10={r.fdr10}, N={r.N})")
        # income-lead recurrence: poorest / declining-income classes
        emit("\n  income-lead check (log_faminc_pc DID by class):")
        for r in tab[tab["outcome"] == "log_faminc_pc"].itertuples():
            emit(f"    cls {r.cls} [{descriptor(prof, r.cls)}]"
                 f" {r.coef:+.4f} (p={r.p:.3f}, pre_viol={r.pre_viol})")

    pool = pd.concat(all_cells, ignore_index=True)
    hits = pool[pool["p"] < 0.05].sort_values(["outcome", "p"])
    emit("\n" + "=" * 72)
    emit("ALL CELLS p<0.05 (screen — recurrence across variants is the bar)")
    for r in hits.itertuples():
        emit(f"  {r.method:<14} cls {r.cls} {r.outcome:<15}"
             f" {r.coef:+.4f} (p={r.p:.3f}, pre_viol={r.pre_viol}, N={r.N})")

    out = os.path.join(OUT, "tables", "trajectory_typology.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
