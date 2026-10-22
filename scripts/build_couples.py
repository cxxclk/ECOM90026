# ==========================================================================
# REPLICATION PACKAGE — build_couples.py
# Tier: CORE
# Paper items: none directly (support module)
# Reads: output/analysis_data.csv
# Writes: output/couples_panel.csv
# Imports local modules: data_process
# Run order: step 4 in run_all.sh
# ==========================================================================

"""
Build a couple-linked panel (angle ①).

For each married woman in the analytic sample, attach her husband's attributes by
linking famconf spouse id (pid_s / pid_a_s) → that wave's adult/person file.
Enables intra-household analysis: husband-side vs wife-side housing burden,
relative income / education (bargaining), gendered response to the HPR shock.

Output: output/couples_panel.csv  (analytic women × wave + h_* husband columns).
"""
import os
import sys
import numpy as np
import pandas as pd
import pyreadstat
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_process as dp     # reuse WAVES config + DATA_DIR

DATA_DIR = dp.DATA_DIR
OUT = os.path.join(dp.BASE_DIR, "output")

# spouse-id column differs by wave
SPOUSE_COL = {2014: "pid_s", 2016: "pid_s", 2018: "pid_a_s",
              2020: "pid_a_s", 2022: "pid_s", 2012: "pid_s", 2010: "pid_s"}

# hukou type column: qa301 (1=Agricultural/rural, 3=Non-agricultural); 2016 uses pa301
HUKOU_COL = {2014: "qa301", 2016: "pa301", 2018: "qa301",
             2020: "qa301", 2022: "qa301", 2012: "qa301", 2010: "qa301"}

def num(s):
    return pd.to_numeric(s, errors="coerce")

def build_wave(year, cfg):
    # 1. spouse link from famconf
    fam, _ = pyreadstat.read_dta(os.path.join(DATA_DIR, cfg["famconf"]))
    fam.columns = fam.columns.str.lower()
    scol = SPOUSE_COL.get(year)
    if scol not in fam.columns:
        scol = next((c for c in ["pid_s", "pid_a_s"] if c in fam.columns), None)
    if scol is None:
        return None
    link = fam[["pid", scol]].copy()
    link.columns = ["pid", "husband_pid"]
    link["pid"] = num(link["pid"]); link["husband_pid"] = num(link["husband_pid"])
    link = link[link["husband_pid"] > 0].drop_duplicates("pid")

    # 2. husband attributes from adult/person file
    ad, _ = pyreadstat.read_dta(os.path.join(DATA_DIR, cfg["adult"]))
    ad.columns = ad.columns.str.lower()
    hukou_src = HUKOU_COL.get(year)
    keep = {cfg["birthy"].lower(): "h_birthy", cfg["eduy"].lower(): "h_eduy",
            cfg["income"].lower(): "h_income", cfg["gender"].lower(): "h_gender"}
    if hukou_src and hukou_src in ad.columns:
        keep[hukou_src] = "h_hukou"
    keep = {k: v for k, v in keep.items() if k in ad.columns}
    hus = ad[["pid"] + list(keep)].rename(columns={"pid": "husband_pid", **keep})
    hus["husband_pid"] = num(hus["husband_pid"])
    for c in ["h_birthy", "h_eduy", "h_income"]:
        if c in hus.columns:
            hus[c] = num(hus[c]); hus.loc[hus[c] < 0, c] = np.nan
    if "h_hukou" in hus.columns:
        hus["h_hukou"] = num(hus["h_hukou"])
        hus["h_rural"] = (hus["h_hukou"] == 1).astype(float)   # 1=Agricultural
        hus.loc[~hus["h_hukou"].isin([1, 3]), "h_rural"] = np.nan
        hus = hus.drop(columns="h_hukou")
    hus = hus.drop_duplicates("husband_pid")

    link = link.merge(hus, on="husband_pid", how="left")
    link["year"] = year
    return link


def main():
    a = pd.read_csv(os.path.join(OUT, "analysis_data.csv"))
    a["pid"] = a["pid"].astype(int)

    links = []
    for yr, cfg in dp.WAVES.items():
        if yr == 2010:
            continue
        try:
            lk = build_wave(yr, cfg)
            if lk is not None:
                links.append(lk)
                print(f"  {yr}: {len(lk):,} spouse links, "
                      f"{lk['h_income'].notna().mean():.0%} w/ husband income")
        except Exception as e:
            print(f"  {yr}: FAILED {e}")
    link_all = pd.concat(links, ignore_index=True)

    df = a.merge(link_all, on=["pid", "year"], how="left")

    # derived couple variables
    df["h_age"]      = df["year"] - df["h_birthy"]
    df["matched"]    = df["husband_pid"].notna().astype(int)
    df["h_observed"] = df["h_income"].notna().astype(int)     # husband in adult file
    df["log_h_income"] = np.log1p(df["h_income"].clip(lower=0))
    # relative (wife − husband); needs both
    df["edu_gap_wife"]    = df["eduy"] - df["h_eduy"]          # +ve: wife more educated
    df["inc_gap_wife"]    = df["log_income"] - df["log_h_income"]
    df["wife_higher_edu"] = (df["edu_gap_wife"] > 0).astype(float)
    df.loc[df["h_eduy"].isna() | df["eduy"].isna(), "wife_higher_edu"] = np.nan
    df["age_gap"]         = df["h_age"] - df["age"]            # husband older by

    df.to_csv(os.path.join(OUT, "couples_panel.csv"), index=False)
    print(f"\nSaved → output/couples_panel.csv ({len(df):,} rows)")

    # coverage report
    print("\n=== couple-match coverage by wave ===")
    rep = df.groupby("year").agg(
        n=("pid", "size"),
        matched=("matched", "mean"),
        husband_observed=("h_observed", "mean"),
        both_income=("inc_gap_wife", lambda s: s.notna().mean()))
    print((rep.assign(matched=lambda d: (d.matched*100).round(1),
                      husband_observed=lambda d: (d.husband_observed*100).round(1),
                      both_income=lambda d: (d.both_income*100).round(1))).to_string())
    print("\nwife more educated than husband: "
          f"{df['wife_higher_edu'].mean():.1%} | "
          f"husband older by (mean yrs): {df['age_gap'].mean():.1f}")


if __name__ == "__main__":
    main()
