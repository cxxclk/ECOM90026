# ==========================================================================
# REPLICATION PACKAGE — semisup_methods_compare.py
# Tier: SUPPLEMENTARY
# Paper items: Table A1 (Three typologies)
# Reads: (via imported modules)
# Writes: output/tables/semisup_methods_compare.txt
# Imports local modules: semisup_typology
# Run order: step 41 in run_all.sh
# ==========================================================================

"""Method sweep for the semi-supervised pre-shock typology (2026-07-16).

Grid: 3 embeddings x 5 cluster algorithms on the SAME pre-shock behavior
signal, then the same post-shock DID screen for the silhouette-best cut per
embedding (+ each embedding's HDBSCAN). Question: is the typology structure
— and the "traditional high-fertility enclave income hit" from the baseline
run — a method artifact or a recurring feature?

Embeddings (all supervision dated <= 2015; see semisup_typology.py):
  oof    6-dim cross-fitted E[behavior | X]     (baseline)
  behav  actual behaviors, OOF-imputed          (keeps idiosyncratic variation)
  leaf   RF leaf co-occurrence -> SVD(8)        (supervised nonlinear metric)
Algorithms: GMM, KMeans, Ward (kNN-structured), Spectral (kNN), HDBSCAN.
K for the first four picked by silhouette over 3..8; HDBSCAN finds its own
cluster count and flags noise (-1, excluded from the scan).

Cross-method credibility criterion: a finding counts only if it RECURS across
embeddings/algorithms — single-method hits are treated as artifacts. Anchor
types are matched across methods by profile (most-traditional = max mean
ideal_kids; housing-squeezed = max hv_inc; young-active = max births_1015).
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
import semisup_typology as st                       # noqa: E402

from sklearn.preprocessing import StandardScaler, OneHotEncoder  # noqa: E402
from sklearn.impute import SimpleImputer            # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.decomposition import TruncatedSVD      # noqa: E402
from sklearn.mixture import GaussianMixture         # noqa: E402
from sklearn.cluster import (KMeans, AgglomerativeClustering,  # noqa: E402
                             SpectralClustering, HDBSCAN)
from sklearn.neighbors import kneighbors_graph      # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_rand_score  # noqa: E402

LINES = []
def emit(s=""):
    print(s, flush=True)
    LINES.append(s)
st.emit = emit          # route semisup_typology output through this file

SEED = 0
PROF_COLS = ["age14", "eduy", "urban", "parity14", "log_faminc_pc"]


def build_embeddings(d):
    E = st.oof_embed(d)                       # 6-dim OOF predictions
    A = d[st.TARGETS].astype(float).values    # actual behaviors
    B = np.where(np.isnan(A), E, A)           # OOF-imputed actuals

    X = d[st.XCOLS].astype(float).values
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    leafs = []
    for j, t in enumerate(st.TARGETS):
        y = A[:, j]
        obs = ~np.isnan(y)
        rf = RandomForestRegressor(n_estimators=40, min_samples_leaf=50,
                                   random_state=SEED, n_jobs=-1)
        rf.fit(Xi[obs], y[obs])
        leafs.append(rf.apply(Xi))
    L = np.hstack(leafs)
    oh = OneHotEncoder(handle_unknown="ignore").fit_transform(L)
    Z = TruncatedSVD(8, random_state=SEED).fit_transform(oh)
    emit(f"\nleaf embedding: {L.shape[1]} trees -> one-hot "
         f"{oh.shape[1]:,} leaves -> SVD(8)")

    return {"oof": StandardScaler().fit_transform(E),
            "behav": StandardScaler().fit_transform(B),
            "leaf": StandardScaler().fit_transform(Z)}


def fit_algo(algo, Z, k, conn=None):
    if algo == "gmm":
        return GaussianMixture(k, covariance_type="full", n_init=3,
                               random_state=SEED).fit(Z).predict(Z)
    if algo == "kmeans":
        return KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    if algo == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward",
                                       connectivity=conn).fit_predict(Z)
    if algo == "spectral":
        return SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                  n_neighbors=15, assign_labels="kmeans",
                                  random_state=SEED).fit_predict(Z)


def cluster_grid(Es):
    """(embedding, algo) -> (labels, silhouette, K).  K by silhouette 3..8."""
    res = {}
    for name, Z in Es.items():
        conn = kneighbors_graph(Z, 15, include_self=False)
        for algo in ("gmm", "kmeans", "ward", "spectral"):
            best = (None, -2, None)
            for k in range(3, 9):
                lab = fit_algo(algo, Z, k, conn)
                sil = silhouette_score(Z, lab)
                if sil > best[1]:
                    best = (lab, sil, k)
            res[(name, algo)] = best
            emit(f"  {name:<6} {algo:<9} K={best[2]}  sil={best[1]:.3f}")
        hl = HDBSCAN(min_cluster_size=250).fit_predict(Z)
        kk = len(set(hl[hl >= 0]))
        mask = hl >= 0
        sil = (silhouette_score(Z[mask], hl[mask])
               if kk > 1 else float("nan"))
        res[(name, "hdbscan")] = (hl, sil, kk)
        emit(f"  {name:<6} hdbscan   K={kk}  sil={sil:.3f}  "
             f"noise={np.mean(hl < 0):.0%}")
    return res


def compact_profile(d):
    g = d.groupby("cls")
    tab = pd.DataFrame({"n": g.size(), "treated": g["treat_hpr"].mean()})
    for c in ["births_1015", "ideal_kids", "son_pref", "hv_inc", "log_inc"]:
        tab[c] = g[c].mean()
    for c in PROF_COLS:
        tab[c] = g[c].mean()
    mothers = d[d["parity14"] >= 1]
    tab["fb_dghtr"] = 1 - mothers.groupby("cls")["fb_son"].mean()
    return tab.round(2)


def descriptor(prof, c):
    r = prof.loc[c]
    return (f"n={int(r['n'])},age={r['age14']:.0f},edu={r['eduy']:.0f},"
            f"par={r['parity14']:.1f},ideal={r['ideal_kids']:.1f},"
            f"son={r['son_pref']:.1f},hv={r['hv_inc']:.1f}")


def main():
    d = st.build_baseline()
    emit("SEMI-SUPERVISED TYPOLOGY — METHOD SWEEP")
    emit(f"  N={len(d):,} women; embeddings: oof / behav / leaf; "
         "algos: gmm kmeans ward spectral hdbscan")

    Es = build_embeddings(d)
    emit("\nCluster grid (K by silhouette over 3..8):")
    res = cluster_grid(Es)

    base_lab = res[("oof", "gmm")][0]
    emit("\nAgreement with baseline typology (oof+gmm), adjusted Rand index:")
    for (name, algo), (lab, sil, k) in res.items():
        ari = adjusted_rand_score(base_lab, lab)
        emit(f"  {name:<6} {algo:<9} K={k}  sil={sil:.3f}  ARI={ari:.3f}")

    # scanned methods: silhouette-best algo per embedding + hdbscan per emb
    scanned = []
    for name in Es:
        algo = max(("gmm", "kmeans", "ward", "spectral"),
                   key=lambda a: res[(name, a)][1])
        scanned.append((name, algo))
        lab, sil, k = res[(name, "hdbscan")]
        if k >= 2 and np.mean(lab < 0) < 0.7:
            scanned.append((name, "hdbscan"))

    all_cells = []
    anchors = []
    for name, algo in scanned:
        lab, sil, k = res[(name, algo)]
        tag = f"{name}+{algo}"
        dd = d.copy()
        dd["cls"] = lab
        if algo == "hdbscan":
            dd = dd[dd["cls"] >= 0]
        emit("\n" + "=" * 72)
        emit(f"METHOD {tag}  (K={k}, sil={sil:.3f})")
        prof = compact_profile(dd)
        emit(prof.to_string())
        tab = st.scan(dd, verbose=False)
        tab["method"] = tag
        all_cells.append(tab)
        emit(f"\n  top cells (of {len(tab)}), Post=2018 DID:")
        top = tab.head(6)
        for r in top.itertuples():
            emit(f"    cls {r.cls} [{descriptor(prof, r.cls)}] {r.outcome:<15}"
                 f" {r.coef:+.4f} (p={r.p:.3f}, pre_viol={r.pre_viol},"
                 f" fdr10={r.fdr10})")

        # anchor types matched by profile
        big = prof[prof["n"] >= 150]
        for aname, col, outs in [
                ("traditional", "ideal_kids", ["log_faminc_pc", "haz_any"]),
                ("squeezed", "hv_inc", ["haz_2nd", "haz_any"]),
                ("young-active", "births_1015", ["haz_2nd", "haz_any"])]:
            c = big[col].idxmax()
            for o in outs:
                cell = tab[(tab["cls"] == c) & (tab["outcome"] == o)]
                if len(cell):
                    r = cell.iloc[0]
                    anchors.append(dict(method=tag, anchor=aname, cls=c,
                                        outcome=o, coef=r["coef"], p=r["p"],
                                        pre_viol=r["pre_viol"],
                                        desc=descriptor(prof, c)))

    emit("\n" + "=" * 72)
    emit("ANCHOR TYPES ACROSS METHODS (matched by profile, not label)")
    at = pd.DataFrame(anchors)
    for aname in ("traditional", "squeezed", "young-active"):
        emit(f"\n  [{aname}]")
        sub = at[at["anchor"] == aname]
        for r in sub.itertuples():
            emit(f"    {r.method:<16} cls {r.cls} {r.outcome:<15}"
                 f" {r.coef:+.4f} (p={r.p:.3f}, pre_viol={r.pre_viol})"
                 f"  [{r.desc}]")

    emit("\n" + "=" * 72)
    emit("RECURRENCE POOL — all cells with p<0.05 across methods")
    pool = pd.concat(all_cells, ignore_index=True)
    hits = pool[pool["p"] < 0.05].sort_values(["outcome", "p"])
    if len(hits):
        for r in hits.itertuples():
            emit(f"  {r.method:<16} cls {r.cls} {r.outcome:<15}"
                 f" {r.coef:+.4f} (p={r.p:.3f}, pre_viol={r.pre_viol},"
                 f" N={r.N})")
    else:
        emit("  none")

    out = os.path.join(OUT, "tables", "semisup_methods_compare.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
