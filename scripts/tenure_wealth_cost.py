# ==========================================================================
# REPLICATION PACKAGE — tenure_wealth_cost.py
# Tier: CORE
# Paper items: Table 4 (B. Affordability within owners)
# Reads: output/hpr_treatment_city.csv, output/panel_individual.csv
# Writes: output/tables/tenure_wealth_cost.txt
# Imports local modules: annual_parity_hazard
# Run order: step 18 in run_all.sh
# ==========================================================================

"""NEW topic #1 — does the ~0 average fertility effect hide a WEALTH effect
(owners) netting against a COST effect (renters)?

House price up under HPR spillover:
  * OWNERS  -> housing wealth gain -> can afford a(nother) child  -> fertility UP
  * RENTERS -> higher rent / future purchase cost (incl. marriage house)
              -> affordability binds                              -> fertility DOWN
If both are present, the SAMPLE AVERAGE can be ~0 while the tenure-split effects
are large and opposite. This directly attacks the central "average ~0" puzzle,
and uses the FULL sample (tenure observed for ~everyone) so power is good —
unlike the thin couple cells.

Design: reuse the annual discrete-time hazard machinery (annual_parity_hazard).
Baseline (pre-shock, first wave <=2014) homeownership, fixed -> immune to
post-shock buy/sell. Pooled any-birth hazard (where the ~0 average lives) and
the second-birth hazard (the headline margin), each split / interacted by tenure.
Married women 20-45, unregulated prefectures, city+year FE, cluster city.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "output")
import annual_parity_hazard as aph  # noqa: E402

LINES = []
def emit(s=""):
    print(s); LINES.append(s)


def attrs_with_tenure():
    """woman_attrs (baseline city, birthy, urban, ever-married, HPR) + baseline
    pre-shock homeownership (first non-missing wave <= 2014)."""
    panel = pd.read_csv(
        os.path.join(OUT, "panel_individual.csv"),
        usecols=["pid", "year", "gender", "birthy", "city_code", "urban",
                 "marital", "homeowner", "house_value", "income"])
    women = panel[panel["gender"] == 0].sort_values(["pid", "year"])
    base = women.dropna(subset=["city_code"]).groupby("pid").first()
    women["_married"] = (women["marital"] == 2).astype(int)
    attrs = women.groupby("pid").agg(birthy=("birthy", "first"),
                                     urban=("urban", "max"),
                                     ever_married=("_married", "max"))
    attrs = attrs.join(base[["city_code"]]).dropna(subset=["city_code", "birthy"])

    # baseline tenure: first non-missing homeowner in a pre-shock wave (<=2014)
    pre = women[(women["year"] <= 2014) & women["homeowner"].notna()]
    own = pre.groupby("pid")["homeowner"].first().rename("owner")
    attrs = attrs.join(own)
    # baseline housing wealth and income (pre-shock) for the within-owner gradient
    hv = women[(women["year"] <= 2014) & (women["house_value"] > 0)]
    attrs = attrs.join(np.log(hv.groupby("pid")["house_value"].first()).rename("lhv"))
    inc = women[women["year"] <= 2014].groupby("pid")["income"].first()
    attrs = attrs.join(np.log1p(inc).rename("linc"))
    attrs = attrs[attrs["ever_married"] == 1].reset_index()

    hpr = (pd.read_csv(os.path.join(OUT, "hpr_treatment_city.csv"))
           .rename(columns={"treat_spillover": "treat_hpr"}))
    attrs = attrs.merge(hpr[["city_code", "is_regulated", "treat_hpr"]],
                        on="city_code", how="inner")
    attrs = attrs[attrs["is_regulated"] == 0]
    return attrs.dropna(subset=["owner"])


def fit(formula, d, label, keys):
    m = smf.ols(formula, data=d).fit(cov_type="cluster",
                                     cov_kwds={"groups": d["city"]})
    emit(f"\n[{label}]  N={int(m.nobs):,}  women={d['pid'].nunique():,}  "
         f"mean(y)={d['y'].mean():.4f}")
    for k in keys:
        if k in m.params:
            emit(f"  {k:<16} {m.params[k]:+.4f}  (SE {m.bse[k]:.4f}, p={m.pvalues[k]:.3f})")
    return m


def main():
    roster, last_wave = aph.load_rosters()
    aph.LOAD_ROSTER, aph.LOAD_LAST = roster, (roster, last_wave)
    attrs = attrs_with_tenure()
    emit("WEALTH (owners) vs COST (renters): tenure-split fertility response")
    emit(f"  women with baseline tenure: {attrs['pid'].nunique():,}  "
         f"owner share={attrs['owner'].mean():.3f}")

    FE = " + C(city) + C(year)"
    risk = {0: aph.build_risk_set(attrs, 0),
            1: aph.build_risk_set(attrs, 1),
            2: aph.build_risk_set(attrs, 2)}
    for k in risk:
        risk[k]["owner"] = risk[k]["owner"].astype(int)

    pooled = pd.concat([risk[0], risk[1], risk[2]], ignore_index=True)
    pooled["post"] = (pooled["year"] >= 2018).astype(int)

    specs = [("ANY birth (pooled) — where the ~0 average lives",
              pooled, "age + I(age**2)"),
             ("SECOND birth (one-child, headline margin)",
              risk[1].assign(post=(risk[1]["year"] >= 2018).astype(int)),
              "age + I(age**2) + dur + I(dur**2)")]

    for title, d, ctrl in specs:
        emit("\n" + "=" * 72)
        emit(title)
        emit(f"  owner share={d['owner'].mean():.3f}  "
             f"hazard owners={d[d.owner==1]['y'].mean():.4f}  "
             f"renters={d[d.owner==0]['y'].mean():.4f}")
        # split
        fit(f"y ~ T:post + {ctrl}{FE}", d[d["owner"] == 1], "OWNERS (wealth)", ["T:post"])
        fit(f"y ~ T:post + {ctrl}{FE}", d[d["owner"] == 0], "RENTERS (cost)", ["T:post"])
        # interaction (formal test of the difference)
        fit(f"y ~ T:post + T:post:owner + owner + {ctrl}{FE}", d,
            "interaction T:post x owner", ["T:post", "T:post:owner"])

    # ---- The viable finding: WITHIN-OWNER wealth/income gradient -------------
    # Binary owner/renter has no power (renters 9%, multi-home owners 1% — ~90%
    # are single-home owners, so there is no portfolio variation). But the
    # suppression is graded by how CONSTRAINED the owner is: it concentrates on
    # LOW house-value / LOW income owners and vanishes for the wealthy. Housing
    # wealth and income each insulate independently (low mutual corr ~0.17).
    emit("\n" + "=" * 72)
    emit("WITHIN-OWNER affordability gradient (second-birth margin, owners only)")
    d = risk[1][risk[1]["owner"] == 1].copy()
    d["post"] = (d["year"] >= 2018).astype(int)
    d = d.dropna(subset=["lhv", "linc"])
    for c in ("lhv", "linc"):
        d[c + "z"] = (d[c] - d[c].mean()) / d[c].std()
    ctrl = "age + I(age**2) + dur + I(dur**2)"
    emit(f"  corr(log house value, log income) = {d[['lhv','linc']].corr().iloc[0,1]:.2f}")
    fit(f"y ~ T:post + T:post:lhvz + lhvz + {ctrl}{FE}", d,
        "house-wealth gradient (alone)", ["T:post", "T:post:lhvz"])
    fit(f"y ~ T:post + T:post:lhvz + T:post:lincz + lhvz + lincz + {ctrl}{FE}", d,
        "HORSE-RACE: housing wealth vs income", ["T:post", "T:post:lhvz", "T:post:lincz"])
    emit("\n  => suppression concentrates on the financially CONSTRAINED owner;")
    emit("     both housing wealth and income independently insulate. Partly")
    emit("     answers the ~0 average (constrained suppressed, wealthy unaffected/")
    emit("     benefited -> nets out). Binary owner/renter & multi-home: no power.")

    out = os.path.join(OUT, "tables", "tenure_wealth_cost.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    emit(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
