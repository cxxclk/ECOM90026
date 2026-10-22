# ==========================================================================
# REPLICATION PACKAGE — sdid_mc.py
# Tier: CORE
# Paper items: Table 2 (C. Alternative estimators); Table B1 (All rows)
# Reads: (via imported modules)
# Writes: output/tables/sdid_mc.txt
# Imports local modules: annual_firstborn_gender
# Run order: step 13 in run_all.sh
# ==========================================================================

"""ML project #1 — Synthetic DID + matrix completion on the city x year
second-birth hazard panel.

Fourth estimator family (after FE-DiD, DR-DiD/DML, causal forest) applied to
the locked margin. City x year panel aggregated from the annual one-child
second-birth hazard risk set (annual_firstborn_gender.build_hazard_panel):

  - Balanced panel 2009-2022, min 3 woman-years per city-year cell
    (retention reported — no silent caps).
  - Post = 2018 (headline convention). Arms: pooled one-child (layer 2),
    daughter-first (layer 3 headline), son-first (placebo).
  - Estimators: DID (aggregate), SC (unit weights only, no intercept),
    SDID (Arkhangelsky et al. AER 2021: ridge simplex unit weights with
    intercept + simplex time weights), and matrix completion (Athey et al.
    JASA 2021: soft-impute with two-way FE, nuclear-norm regularization
    chosen on masked control-post cells).
  - Inference: 200 permutations of the treated label across cities (treated
    share ~50%, so placebo-on-controls-only is infeasible); MC permutations
    reuse the lambda chosen on the real assignment (cost).
  - Reference row: micro FE-DiD re-run on the SAME retained cities.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annual_firstborn_gender import build_hazard_panel, OUT  # noqa: E402

warnings.filterwarnings("ignore")

Y0, Y1, POST = 2009, 2022, 2018
YEARS = list(range(Y0, Y1 + 1))
PRE = [y for y in YEARS if y < POST]
PST = [y for y in YEARS if y >= POST]
FLOOR = 3
NPERM = 200

LINES = []
def emit(s=""):
    print(s)
    LINES.append(s)


# ── simplex least squares (projected gradient) ─────────────────────────────
def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def simplex_ls(A, b, ridge=0.0, intercept=True, iters=2000):
    """min_{w in simplex} ||A w - b||^2 + ridge*||w||^2 (optionally with a
    free intercept, absorbed by row-centering A and b)."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if intercept:
        A = A - A.mean(axis=0, keepdims=True)
        b = b - b.mean()
    n = A.shape[1]
    w = np.full(n, 1.0 / n)
    L = np.linalg.norm(A, 2) ** 2 + ridge
    if L <= 0:
        return w
    for _ in range(iters):
        g = A.T @ (A @ w - b) + ridge * w
        w_new = proj_simplex(w - g / L)
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w = w_new
    return w


# ── estimators on an N x T outcome matrix ──────────────────────────────────
def did_est(Y, tr):
    yt, yc = Y[tr], Y[~tr]
    ip = np.isin(YEARS, PST)
    return ((yt[:, ip].mean() - yt[:, ~ip].mean())
            - (yc[:, ip].mean() - yc[:, ~ip].mean()))


def sc_est(Y, tr):
    ip = np.isin(YEARS, PST)
    Yc, Yt = Y[~tr], Y[tr]
    w = simplex_ls(Yc[:, ~ip].T, Yt[:, ~ip].mean(axis=0),
                   ridge=1e-8, intercept=False)
    pre_rmse = np.sqrt(((Yt[:, ~ip].mean(axis=0) - w @ Yc[:, ~ip]) ** 2)
                       .mean())
    return (Yt[:, ip].mean(axis=0) - w @ Yc[:, ip]).mean(), w, pre_rmse


def sdid_est(Y, tr):
    ip = np.isin(YEARS, PST)
    Yc, Yt = Y[~tr], Y[tr]
    n_tr, t_post = tr.sum(), ip.sum()
    # zeta: sd of first-differenced control pre outcomes (AER 2021 eq. 5)
    dif = np.diff(Yc[:, ~ip], axis=1)
    zeta2 = np.sqrt(n_tr * t_post) * dif.var()
    om = simplex_ls(Yc[:, ~ip].T, Yt[:, ~ip].mean(axis=0),
                    ridge=zeta2 * (~ip).sum(), intercept=True)
    lam = simplex_ls(Yc[:, ~ip], Yc[:, ip].mean(axis=1),
                     ridge=1e-8 * (~ip).sum(), intercept=True)
    tr_gap = Yt[:, ip].mean() - (Yt[:, ~ip].mean(axis=0) @ lam)
    co_gap = om @ (Yc[:, ip].mean(axis=1) - Yc[:, ~ip] @ lam)
    return tr_gap - co_gap, om, lam


def mc_est(Y, tr, lam_mc=None, W=None, seed=0):
    """Soft-impute with two-way FE; returns (tau, lam_mc, rank)."""
    ip = np.isin(YEARS, PST)
    obs = np.ones_like(Y, bool)
    obs[np.ix_(tr, ip)] = False

    def fit(lmbda, obs_mask, iters=200):
        L = np.zeros_like(Y)
        a = np.zeros(Y.shape[0]); b = np.zeros(Y.shape[1]); mu = 0.0
        for _ in range(iters):
            R = Y - L
            mu = R[obs_mask].mean()
            for _ in range(3):
                a = np.array([(R[i, obs_mask[i]] - mu - b[obs_mask[i]]).mean()
                              if obs_mask[i].any() else 0.0
                              for i in range(Y.shape[0])])
                b = np.array([(R[obs_mask[:, t], t] - mu
                               - a[obs_mask[:, t]]).mean()
                              if obs_mask[:, t].any() else 0.0
                              for t in range(Y.shape[1])])
            Z = Y - mu - a[:, None] - b[None, :]
            Zf = np.where(obs_mask, Z, L)
            U, s, Vt = np.linalg.svd(Zf, full_matrices=False)
            s2 = np.maximum(s - lmbda, 0)
            L_new = (U * s2) @ Vt
            if np.max(np.abs(L_new - L)) < 1e-9:
                L = L_new
                break
            L = L_new
        return mu + a[:, None] + b[None, :] + L, (s2 > 0).sum()

    if lam_mc is None:
        rng = np.random.RandomState(seed)
        cand = obs & ~tr[:, None] & ip[None, :]  # control-post cells
        idx = np.argwhere(cand)
        val = idx[rng.choice(len(idx), max(5, len(idx) // 10), replace=False)]
        obs_cv = obs.copy()
        obs_cv[val[:, 0], val[:, 1]] = False
        s_max = np.linalg.svd(np.where(obs, Y - Y[obs].mean(), 0),
                              compute_uv=False)[0]
        best = (np.inf, s_max)
        for lm in s_max * np.array([.5, .25, .1, .05, .02, .01, .005]):
            Yh, _ = fit(lm, obs_cv)
            mse = ((Y - Yh)[val[:, 0], val[:, 1]] ** 2).mean()
            if mse < best[0]:
                best = (mse, lm)
        lam_mc = best[1]

    Yh, rank = fit(lam_mc, obs)
    gap = (Y - Yh)[np.ix_(tr, ip)]
    tau_u = gap.mean()
    if W is not None:
        ww = W[np.ix_(tr, ip)]
        tau = (gap * ww).sum() / ww.sum()
    else:
        tau = tau_u
    return tau, lam_mc, rank, tau_u


# ── one arm ────────────────────────────────────────────────────────────────
def run_arm(df, label, micro_formula, rng):
    g = df.groupby(["city", "year"]).agg(y=("y", "mean"), n=("y", "size"))
    piv_y = g["y"].unstack().reindex(columns=YEARS)
    piv_n = g["n"].unstack().reindex(columns=YEARS)
    ok = (piv_n >= FLOOR).all(axis=1)
    cities = piv_y.index[ok]
    Y = piv_y.loc[cities].values
    W = piv_n.loc[cities].values.astype(float)
    tvec = df.groupby("city")["T"].first().reindex(cities).astype(bool).values
    kept_wy = df[df["city"].isin(cities)].shape[0]

    emit("\n" + "=" * 72)
    emit(f"ARM: {label}")
    emit(f"  cities kept {len(cities)}/{df['city'].nunique()} "
         f"(balanced {Y0}-{Y1}, cell floor >= {FLOOR} woman-years); "
         f"treated {int(tvec.sum())}; woman-years retained "
         f"{kept_wy:,}/{len(df):,} ({kept_wy/len(df):.0%})")
    emit(f"  mean hazard pre: T {Y[tvec][:, :len(PRE)].mean():.4f}  "
         f"C {Y[~tvec][:, :len(PRE)].mean():.4f}")

    t_did = did_est(Y, tvec)
    t_sc, w_sc, sc_rmse = sc_est(Y, tvec)
    t_sdid, om, lam = sdid_est(Y, tvec)
    t_mc, lam_mc, rank, t_mc_u = mc_est(Y, tvec, W=W)

    # permutation inference (relabel cities, same treated count)
    perms = {"DID": [], "SC": [], "SDID": [], "MC": [], "MCu": []}
    for _ in range(NPERM):
        p = rng.permutation(tvec)
        perms["DID"].append(did_est(Y, p))
        perms["SC"].append(sc_est(Y, p)[0])
        perms["SDID"].append(sdid_est(Y, p)[0])
        pm = mc_est(Y, p, lam_mc=lam_mc, W=W)
        perms["MC"].append(pm[0])
        perms["MCu"].append(pm[3])

    def pv(tau, key):
        arr = np.abs(np.array(perms[key]))
        return (1 + (arr >= abs(tau)).sum()) / (1 + NPERM)

    emit(f"\n  {'estimator':<10} {'tau':>9}  perm p   (perm sd)")
    for key, tau in [("DID", t_did), ("SC", t_sc), ("SDID", t_sdid),
                     ("MC", t_mc), ("MCu", t_mc_u)]:
        emit(f"  {key:<10} {tau:+.4f}   {pv(tau, key):.3f}    "
             f"({np.std(perms[key]):.4f})")
    emit(f"  MC: lambda={lam_mc:.4f}, residual rank={rank} "
         f"(MC=woman-year-weighted, MCu=unweighted city cells)")
    emit(f"  SC pre-fit RMSE {sc_rmse:.4f} vs treated-control pre level gap "
         f"{Y[tvec][:, :len(PRE)].mean() - Y[~tvec][:, :len(PRE)].mean():+.4f}"
         f"  (SC has no intercept -> judges level fit)")

    # micro FE-DiD on the same cities (reference row)
    d = df[df["city"].isin(cities)].copy()
    d["post"] = (d["year"] >= POST).astype(int)
    m = smf.ols(micro_formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"]})
    emit(f"  micro FE-DiD same cities: {m.params['T:post']:+.4f} "
         f"(cluster p={m.pvalues['T:post']:.3f})")

    # diagnostics: weights + yearly SDID gap path
    eff_n = 1.0 / (om ** 2).sum()
    top = np.argsort(-om)[:5]
    ctrl_cities = np.array(cities)[~tvec]
    emit(f"\n  SDID unit weights: effective N controls {eff_n:.1f} "
         f"of {int((~tvec).sum())}; top-5 "
         + ", ".join(f"{int(ctrl_cities[i])}:{om[i]:.3f}" for i in top))
    emit("  SDID time weights (pre): "
         + ", ".join(f"{y}:{l:.2f}" for y, l in zip(PRE, lam) if l > 0.01))
    gap = Y[tvec].mean(axis=0) - om @ Y[~tvec]
    gap = gap - gap[: len(PRE)] @ lam
    emit("  yearly gap (treated - synthetic, pre aligned by lambda):")
    for y, gp in zip(YEARS, gap):
        mark = " <- post" if y == POST else ""
        emit(f"    {y}  {gp:+.4f}{mark}")
    return {"DID": t_did, "SC": t_sc, "SDID": t_sdid, "MC": t_mc}


def main():
    rng = np.random.RandomState(42)
    df = build_hazard_panel()
    emit("Synthetic DID + matrix completion, city x year second-birth hazard")
    emit(f"Post={POST}; {NPERM} city-label permutations; floor={FLOOR}")

    CTRL = "age + I(age**2) + dur + I(dur**2)"
    f_micro = f"y ~ T:post + {CTRL} + C(city) + C(year)"

    res = {}
    res["pooled one-child"] = run_arm(df, "POOLED one-child (layer 2)",
                                      f_micro, rng)
    res["daughter-first"] = run_arm(df[df["son"] == 0],
                                    "DAUGHTER-first (layer 3)", f_micro, rng)
    res["son-first"] = run_arm(df[df["son"] == 1],
                               "SON-first (placebo)", f_micro, rng)

    emit("\n" + "=" * 72)
    emit("SUMMARY (tau, percentage points of annual hazard)")
    emit(f"{'arm':<20}" + "".join(f"{k:>9}" for k in ["DID", "SC", "SDID",
                                                      "MC"]))
    for arm, r in res.items():
        emit(f"{arm:<20}" + "".join(f"{100*r[k]:>+8.2f}p" for k in r))

    out = os.path.join(OUT, "tables", "sdid_mc.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
