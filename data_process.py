# ==========================================================================
# REPLICATION PACKAGE — data_process.py
# Tier: CORE
# Paper items: Figure 1 (First stage in second-hand prices); Table C3 (Questionnaire codes)
# Reads: CFPS .dta files in data/20xx/, output/hpr_treatment_city.csv
# Writes: output/analysis_data.csv, output/housing_price_city.csv, output/housing_price_prov.csv, output/panel_individual.csv, output/tables/data_summary.txt
# Imports local modules: none
# Run order: step 1, step 3 in run_all.sh
# ==========================================================================

"""
CFPS Data Processing Pipeline
==============================
Dataset: China Family Panel Studies (CFPS), 2010–2022
Purpose: Clean and merge CFPS waves with housing prices for a DID analysis
         of housing price effects on fertility (using 2016 HPR as external shock).

Outputs (saved to /output/):
  - panel_individual.csv   : cleaned individual-level panel (all waves)
  - housing_price_prov.csv : annual average housing price by province
  - analysis_data.csv      : merged, sample-restricted data ready for regression
  - data_summary.txt       : obs counts and basic stats at each stage
"""

import os
import warnings
import pyreadstat
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Wave configuration ────────────────────────────────────────────────────────
# Waves 2010–2016 use 'adult' files; 2018–2022 use 'person' files.

WAVES = {
    2010: {
        "adult":   "2010/ecfps2010adult_201906.dta",
        "famconf": "2010/ecfps2010famconf_nat072016.dta",
        "famecon": "2010/ecfps2010famecon_201906.dta",
        "fid":      "fid",
        "provcd":   "provcd",
        "countyid": "countyid",
        "urban":    "urban",
        "birthy":   "qa1y_best",
        "gender":   "gender",
        "marital":  "qe1_best",
        "edu":      "cfps2010edu_best",
        "eduy":     "cfps2010eduy_best",
        "income":   "income",
        "childn":   "cfps_childn",
    },
    2012: {
        "adult":   "2012/ecfps2012adult_202505.dta",
        "famconf": "2012/ecfps2012famconf_092015.dta",
        "famecon": "2012/ecfps2012famecon_201906.dta",
        "fid":      "fid12",
        "provcd":   "provcd",
        "countyid": "countyid",
        "urban":    "urban12",
        "birthy":   "cfps2012_birthy_best",
        "gender":   "cfps2012_gender_best",
        "marital":  "qe104",
        "edu":      "edu2012",
        "eduy":     "eduy2012",
        "income":   "income_adj",
        "childn":   "cfps2010_num_child",
    },
    2014: {
        "adult":   "2014/ecfps2014adult_201906.dta",
        "famconf": "2014/ecfps2014famconf_170630.dta",
        "famecon": "2014/ecfps2014famecon_201906.dta",
        "fid":      "fid14",
        "provcd":   "provcd14",
        "countyid": "countyid14",
        "urban":    "urban14",
        "birthy":   "cfps_birthy",
        "gender":   "cfps_gender",
        "marital":  "qea0",
        "edu":      "cfps2014edu",
        "eduy":     "cfps2014eduy",
        "income":   "income",
        "childn":   "cfps_childn",
    },
    2016: {
        "adult":   "2016/ecfps2016adult_201906.dta",
        "famconf": "2016/ecfps2016famconf_201804.dta",
        "famecon": "2016/ecfps2016famecon_201807.dta",
        "fid":      "fid16",
        "provcd":   "provcd16",
        "countyid": "countyid16",
        "urban":    "urban16",
        "birthy":   "cfps_birthy",
        "gender":   "cfps_gender",
        "marital":  "qea0",
        "edu":      "cfps2016edu",
        "eduy":     "cfps2016eduy",
        "income":   "income",
        "childn":   "cfps_childn",
    },
    2018: {
        "adult":   "2018/ecfps2018person_202012.dta",
        "famconf": "2018/ecfps2018famconf_202008.dta",
        "famecon": "2018/ecfps2018famecon_202101.dta",
        "fid":      "fid18",
        "provcd":   "provcd18",
        "countyid": "countyid18",
        "urban":    "urban18",
        "birthy":   "ibirthy",
        "gender":   "gender",
        "marital":  "qea0",
        "edu":      "cfps2018edu",
        "eduy":     "cfps2018eduy",
        "income":   "income",
        "childn":   "cfps_childn",
    },
    2020: {
        "adult":   "2020/ecfps2020person_202306.dta",
        "famconf": "2020/ecfps2020famconf_202306.dta",
        "famecon": "2020/ecfps2020famecon_202306.dta",
        "fid":      "fid20",
        "provcd":   "provcd20",
        "countyid": "countyid20",
        "urban":    "urban20",
        "birthy":   "ibirthy",
        "gender":   "gender",
        "marital":  "qea0",
        "edu":      "cfps2020edu",
        "eduy":     "cfps2020eduy",
        "income":   "emp_income",
        "childn":   "cfps_childn",
    },
    2022: {
        "adult":   "2022/ecfps2022person_202410.dta",
        "famconf": "2022/ecfps2022famconf_202410.dta",
        "famecon": "2022/ecfps2022famecon_202410.dta",
        "fid":      "fid22",
        "provcd":   "provcd22",
        "countyid": "countyid22",
        "urban":    "urban22",
        "birthy":   "ibirthy",
        "gender":   "gender",
        "marital":  "qea0",
        "edu":      "cfps2022edu",
        "eduy":     "cfps2022eduy",
        "income":   "emp_income",
        "childn":   "cfps_childn",
    },
}

# Columns to extract from famecon (consistent across 2014–2022; subset for 2010–2012)
# NOTE: homeowner and house_value are handled wave-specifically (see OWNER_SRC /
# HOUSEVAL_SRC) because the 2012 questionnaire used different variables:
#   - ownership:  2012 = fq1 (7-category tenure); 2014+ = fq2 (binary owner y/n)
#   - house value:2012 = fq4a_best;               2014+ = fq6
# (2012 fq2 = "rented OUT?", 2012 fq6 = "year property right secured" — both
#  collide with the 2014+ meanings, which is why a flat mapping corrupted 2012.)
FAMECON_VARS = {
    "houseasset_net":  "houseasset_net",  # net house asset (yuan)
    "houseasset_gross":"houseasset_gross",
    "ft301":           "mortgage_loan",   # outstanding mortgage (ten-thousand yuan)
                                          # ⚠ 2012 ft301 = govt-bond face value (collision); unused there
    "fp407":           "monthly_rent",    # monthly rent paid (yuan)
                                          # ⚠ 2012 fp407 = entertainment expenditure (collision); masked to NaN for 2012
    "fincome1":        "fam_income",      # net family income (yuan)
    "fincome1_per":    "fam_income_pc",   # per-capita net family income (yuan)
    "fr101":           "n_other_prop",    # number of other properties owned
}

# Wave-specific source variables for ownership and current house value.
OWNER_SRC    = {2012: "fq1", 2014: "fq2", 2016: "fq2",
                2018: "fq2", 2020: "fq2", 2022: "fq2"}
HOUSEVAL_SRC = {2012: "fq4a_best", 2014: "fq6", 2016: "fq6",
                2018: "fq6", 2020: "fq6", 2022: "fq6"}

def build_homeowner(series, year):
    """Standardise ownership to 1=owner, 0=non-owner, NaN=missing.
    2012 fq1: codes 1 (solely owned) & 2 (partly owned) = owner; 3–7,77 = rental/other.
    2014+ fq2: 1 = owner, 0/5 = no."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s >= 0)  # negative CFPS codes = missing
    if year == 2012:
        return s.isin([1, 2]).astype(float).where(s.notna())
    return (s == 1).astype(float).where(s.notna())

# ── Helper: safe column getter ────────────────────────────────────────────────
def safe_get(df, col):
    """Return series if col exists, else None-filled series of same length."""
    if col and col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


# ── County → city crosswalk ───────────────────────────────────────────────────
# CFPS county sequence codes (`countyid`) are privacy-masked but STABLE across
# waves. 顺序码匹配.dta (from the CFPS_cleaned release) maps each countyid to the
# real GB administrative codes and names. Validated against the cleaned panel's
# already-resolved (pid, year) → city: 99.9% agreement on 77k overlapping rows.
# Coverage ≈ 88% of panel rows (original 162-county PSU frame; later-wave sample-
# expansion counties are unmatched).
def load_crosswalk():
    xw, _ = pyreadstat.read_dta(
        os.path.join(DATA_DIR, "CFPS_cleaned", "顺序码匹配.dta"))
    xw = xw[["countyid", "code", "provname", "cityname", "countyname", "city"]].copy()
    xw = xw.rename(columns={"code": "county_code", "city": "city_code"})
    xw["countyid"] = pd.to_numeric(xw["countyid"], errors="coerce")
    xw = xw.dropna(subset=["countyid"]).drop_duplicates(subset="countyid")
    print(f"\nCrosswalk loaded: {len(xw):,} counties → "
          f"{xw['city_code'].nunique()} cities")
    return xw


# ── Step 1: Load each wave and standardise ────────────────────────────────────
def load_wave(year, cfg):
    print(f"\n{'='*55}\nLoading wave {year}...")

    # ── 1a. Adult / Person file ──────────────────────────────────────────────
    adult_path = os.path.join(DATA_DIR, cfg["adult"])
    df_adult, _ = pyreadstat.read_dta(adult_path)
    df_adult.columns = df_adult.columns.str.lower()

    adult_cols_wanted = ["pid", cfg["fid"],
                         cfg["provcd"], cfg["countyid"], cfg["urban"],
                         cfg["birthy"], cfg["gender"], cfg["marital"],
                         cfg["edu"], cfg["eduy"], cfg["income"], cfg["childn"]]
    adult_cols_wanted = [c.lower() for c in adult_cols_wanted
                         if c and c.lower() in df_adult.columns]
    df_adult = df_adult[adult_cols_wanted].copy()

    # Standardise column names
    rename_map = {}
    for key, std_name in [
        ("fid",      "fid"),
        ("provcd",   "provcd"),
        ("countyid", "countyid"),
        ("urban",    "urban"),
        ("birthy",   "birthy"),
        ("gender",   "gender"),
        ("marital",  "marital"),
        ("edu",      "edu"),
        ("eduy",     "eduy"),
        ("income",   "income"),
        ("childn",   "childn"),
    ]:
        src = cfg.get(key)
        if src and src.lower() in df_adult.columns:
            rename_map[src.lower()] = std_name

    df_adult.rename(columns=rename_map, inplace=True)
    df_adult["year"] = year
    print(f"  Adult/person: {len(df_adult):,} rows, cols: {list(df_adult.columns)}")

    # ── 1b. Famconf – extract child birth years ──────────────────────────────
    famconf_path = os.path.join(DATA_DIR, cfg["famconf"])
    df_fam, _ = pyreadstat.read_dta(famconf_path)
    df_fam.columns = df_fam.columns.str.lower()

    child_birthy_cols = [f"tb1y_a_c{i}" for i in range(1, 11)
                         if f"tb1y_a_c{i}" in df_fam.columns]

    fam_keep = ["pid"] + child_birthy_cols
    if "tb2_a_p" in df_fam.columns:
        fam_keep.append("tb2_a_p")
    df_fam = df_fam[[c for c in fam_keep if c in df_fam.columns]].copy()

    if child_birthy_cols:
        # CFPS codes -8 (N/A) etc. in the birth-year columns; count only
        # plausible birth years (≤ survey year). notna() alone counts -8 → saturates.
        valid_by = df_fam[child_birthy_cols].where(
            (df_fam[child_birthy_cols] > 1900) & (df_fam[child_birthy_cols] <= year))
        df_fam["n_children_fam"]       = valid_by.notna().sum(axis=1)
        df_fam["youngest_child_birthy"] = valid_by.max(axis=1)
        c1 = df_fam["tb1y_a_c1"].copy() if "tb1y_a_c1" in df_fam.columns else np.nan
        df_fam["first_child_birthy"] = c1.where(c1 > 0, np.nan) if hasattr(c1, "where") else np.nan
    else:
        df_fam["n_children_fam"]       = np.nan
        df_fam["youngest_child_birthy"] = np.nan
        df_fam["first_child_birthy"]    = np.nan

    if "tb2_a_p" in df_fam.columns:
        df_fam.rename(columns={"tb2_a_p": "gender_fam"}, inplace=True)

    # Keep child birth year columns for birth-window computation
    df_fam["_child_cols_json"] = df_fam[child_birthy_cols].to_json(orient="records") \
        if child_birthy_cols else None
    # Actually, keep the raw columns for the merge; we'll drop them afterwards
    print(f"  Famconf: {len(df_fam):,} rows, {len(child_birthy_cols)} child-year cols")

    # ── 1c. Famecon – housing and income ────────────────────────────────────
    famecon_path = os.path.join(DATA_DIR, cfg["famecon"])
    df_econ, _ = pyreadstat.read_dta(famecon_path)
    df_econ.columns = df_econ.columns.str.lower()

    econ_rename = {}
    econ_keep = [cfg["fid"].lower()]
    for src, dst in FAMECON_VARS.items():
        if src in df_econ.columns:
            econ_rename[src] = dst
            econ_keep.append(src)

    # Wave-specific ownership + house value (built before subsetting)
    owner_src = OWNER_SRC.get(year)
    if owner_src and owner_src in df_econ.columns:
        df_econ["homeowner"] = build_homeowner(df_econ[owner_src], year)
    else:
        df_econ["homeowner"] = np.nan
    hv_src = HOUSEVAL_SRC.get(year)
    if hv_src and hv_src in df_econ.columns:
        hv = pd.to_numeric(df_econ[hv_src], errors="coerce")
        df_econ["house_value"] = hv.where(hv >= 0)  # 10K-yuan units, drop neg codes
    else:
        df_econ["house_value"] = np.nan
    econ_keep += ["homeowner", "house_value"]

    df_econ = (df_econ[[c for c in econ_keep if c in df_econ.columns]]
               .copy()
               .rename(columns={cfg["fid"].lower(): "fid"})
               .rename(columns=econ_rename)
               .drop_duplicates(subset="fid"))
    print(f"  Famecon: {len(df_econ):,} rows | owner_src={owner_src} "
          f"hv_src={hv_src} | cols: {list(df_econ.columns)}")

    # ── 1d. Merge ────────────────────────────────────────────────────────────
    df = df_adult.merge(df_fam, on="pid", how="left")
    df = df.merge(df_econ, on="fid", how="left")

    # Fill gender from famconf where missing in adult
    if "gender_fam" in df.columns:
        if "gender" not in df.columns:
            df["gender"] = df["gender_fam"]
        else:
            df["gender"] = df["gender"].fillna(df["gender_fam"])
        df.drop(columns=["gender_fam"], inplace=True)

    # Recode CFPS special codes (negative values = missing/not applicable)
    for col in ["edu", "eduy", "childn"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Ensure standard columns exist
    for col in ["gender", "marital", "edu", "eduy", "income", "childn",
                "homeowner", "house_value", "houseasset_net", "houseasset_gross",
                "mortgage_loan", "monthly_rent", "fam_income", "fam_income_pc",
                "n_other_prop", "n_children_fam", "youngest_child_birthy",
                "first_child_birthy"]:
        if col not in df.columns:
            df[col] = np.nan

    # 2012 questionnaire collision: the ft3–ft7 / fp4xx codes were reused for a
    # different concept before the 2014 restructure. Two reach our pipeline via
    # the flat FAMECON_VARS map and must be masked for 2012:
    #   ft301 -> mortgage_loan : 2012 = govt-bond face value (NOT mortgage)
    #   fp407 -> monthly_rent  : 2012 = entertainment expenditure (NOT rent)
    # (owner/house_value already handled wave-specifically via OWNER_SRC/HOUSEVAL_SRC)
    if year == 2012:
        for col in ["mortgage_loan", "monthly_rent"]:
            if col in df.columns:
                df[col] = np.nan

    # Drop helper column
    if "_child_cols_json" in df.columns:
        df.drop(columns=["_child_cols_json"], inplace=True)

    df["age"] = year - df["birthy"]
    print(f"  Merged: {len(df):,} rows")
    return df, child_birthy_cols


# ── Step 2: Construct birth outcome ──────────────────────────────────────────
#
# For wave T, "new_birth = 1" if the respondent has any child born in
# (prev_wave_year, T].  All wave gaps are 2 years.

PREV_WAVE_YEAR = {2010: None, 2012: 2010, 2014: 2012,
                  2016: 2014, 2018: 2016, 2020: 2018, 2022: 2020}

def add_birth_outcome(df, year, child_cols):
    prev = PREV_WAVE_YEAR[year]

    available = [c for c in child_cols if c in df.columns]
    if not available or prev is None:
        df["new_birth"]    = np.nan
        df["n_new_births"] = np.nan
        return df

    born_matrix = df[available]
    in_window = ((born_matrix > prev) & (born_matrix <= year))
    df["n_new_births"] = in_window.sum(axis=1).astype(float)
    df["new_birth"]    = (df["n_new_births"] >= 1).astype(float)
    # Set NaN where child birth-year info is entirely missing
    all_missing = born_matrix.isna().all(axis=1)
    df.loc[all_missing, "new_birth"]    = np.nan
    df.loc[all_missing, "n_new_births"] = np.nan
    return df


# ── Step 3: Process housing price data ───────────────────────────────────────
def process_housing_prices():
    print("\n" + "="*55)
    print("Processing housing price data...")
    hp = pd.read_excel(os.path.join(DATA_DIR, "Housing Price.xlsx"),
                       usecols=[0, 1, 2, 3, 4, 5])
    hp.columns = ["admin_code", "city", "province", "year", "month", "price_sqm"]

    # Province code: first 2 digits of the 6-digit admin code
    hp["provcd"] = (hp["admin_code"] // 10000).astype(int)

    # Annual average price per province
    hp_prov = (hp.groupby(["provcd", "year"])["price_sqm"]
                 .mean()
                 .reset_index()
                 .rename(columns={"price_sqm": "hp_prov_avg"}))

    hp_prov = hp_prov.sort_values(["provcd", "year"])
    hp_prov["hp_prov_growth"] = hp_prov.groupby("provcd")["hp_prov_avg"].pct_change()
    hp_prov["log_hp"]         = np.log(hp_prov["hp_prov_avg"])

    print(f"  Province-year obs: {len(hp_prov):,}")
    print(f"  Year range: {hp_prov['year'].min()} – {hp_prov['year'].max()}")
    print(f"  Provinces covered: {hp_prov['provcd'].nunique()}")
    hp_prov.to_csv(os.path.join(OUT_DIR, "housing_price_prov.csv"), index=False)
    print("  Saved → output/housing_price_prov.csv")
    return hp_prov


def process_city_prices():
    """City × year average second-hand price, keyed on GB admin code (city_code)."""
    print("\n" + "="*55)
    print("Processing CITY-level housing prices...")
    hp = pd.read_excel(os.path.join(DATA_DIR, "Housing Price.xlsx"),
                       usecols=[0, 1, 2, 3, 4, 5])
    hp.columns = ["city_code", "city_name", "province", "year", "month", "price_sqm"]
    hp["city_code"] = hp["city_code"].astype(int)

    hp_city = (hp.groupby(["city_code", "year"])["price_sqm"]
                 .mean().reset_index()
                 .rename(columns={"price_sqm": "hp_city_avg"})
                 .sort_values(["city_code", "year"]))
    hp_city["hp_city_growth"] = hp_city.groupby("city_code")["hp_city_avg"].pct_change()
    hp_city["log_hp_city"]    = np.log(hp_city["hp_city_avg"])

    print(f"  City-year obs: {len(hp_city):,}  |  cities: {hp_city['city_code'].nunique()}")
    hp_city.to_csv(os.path.join(OUT_DIR, "housing_price_city.csv"), index=False)
    print("  Saved → output/housing_price_city.csv")
    return hp_city


# ── Step 4: DID treatment variable ───────────────────────────────────────────
#
# Binary treatment proxy: provinces with above-median average housing price in
# 2015 (year before the HPR shock). These "high-price" provinces faced stronger
# demand pressure and were more likely to implement / be affected by the 2016
# Housing Purchase Restriction policy.
#
# LIMITATION: CFPS county IDs are obfuscated (privacy-masked) and cannot be
# directly matched to GB administrative codes.  A precise city-level HPR
# assignment would require an external crosswalk.  This province-level proxy is
# a feasible first pass; refine with a crosswalk if one becomes available.

def build_treatment(df_panel, hp_prov):
    hp_2015 = hp_prov[hp_prov["year"] == 2015][["provcd", "hp_prov_avg"]].copy()
    if hp_2015.empty:
        # Fall back to earliest available year
        earliest = hp_prov["year"].min()
        hp_2015  = hp_prov[hp_prov["year"] == earliest][["provcd", "hp_prov_avg"]].copy()
        print(f"  WARNING: no 2015 price data; using {earliest} as pre-shock baseline")

    median_price = hp_2015["hp_prov_avg"].median()
    hp_2015["treated"] = (hp_2015["hp_prov_avg"] >= median_price).astype(int)
    print(f"  Treatment threshold (median 2015 price): {median_price:,.0f} CNY/sqm")
    print(f"  Treated provinces (high-price): {hp_2015['treated'].sum()}")

    df_panel = df_panel.merge(hp_2015[["provcd", "treated"]], on="provcd", how="left")
    df_panel["post"] = (df_panel["year"] >= 2016).astype(int)
    df_panel["did"]  = df_panel["treated"] * df_panel["post"]
    return df_panel


def merge_hpr_treatment(df_panel):
    """Attach the HPR spillover treatment (output/hpr_treatment_city.csv).

    Adds:
      dist_to_hpr_km  – distance to nearest regulated city (continuous intensity)
      is_regulated    – 1 if the prefecture is itself a regulated source city
      treat_hpr       – 1 if unregulated & ≤250 km of a regulated city
      hpr_exposure    – max(0, 1 − dist/250): continuous decay exposure (0 at ≥250 km)
      did_hpr         – treat_hpr × post (main spillover DID interaction)
    """
    path = os.path.join(OUT_DIR, "hpr_treatment_city.csv")
    if not os.path.exists(path):
        print("  WARNING: output/hpr_treatment_city.csv not found — run "
              "scripts/build_hpr_spillover.py first. Skipping HPR treatment.")
        for c in ["dist_to_hpr_km", "is_regulated", "treat_hpr",
                  "hpr_exposure", "did_hpr"]:
            df_panel[c] = np.nan
        return df_panel

    hpr = (pd.read_csv(path)
             .rename(columns={"treat_spillover": "treat_hpr"})
             [["city_code", "dist_to_hpr_km", "is_regulated", "treat_hpr"]])
    df_panel = df_panel.merge(hpr, on="city_code", how="left")
    df_panel["hpr_exposure"] = (1 - df_panel["dist_to_hpr_km"] / 250).clip(lower=0)
    df_panel["did_hpr"] = df_panel["treat_hpr"] * df_panel["post"]
    cov = df_panel["treat_hpr"].notna().mean()
    print(f"  HPR spillover treatment merged: {cov:.1%} of rows")
    return df_panel


# ── Step 5: Sample restrictions ──────────────────────────────────────────────
def apply_sample_restrictions(df):
    n_raw = len(df)
    log   = []

    df = df[df["year"] > 2010].copy()
    log.append(f"Drop 2010 baseline:          {n_raw:>8,} → {len(df):>8,}")

    # Keep women only (CFPS gender coding: 0 = female, 1 = male)
    n = len(df); df = df[df["gender"] == 0].copy()
    log.append(f"Keep women (gender=0):       {n:>8,} → {len(df):>8,}")

    # Keep currently married (CFPS marital coding: 2 = married)
    n = len(df); df = df[df["marital"] == 2].copy()
    log.append(f"Keep married (marital=2):    {n:>8,} → {len(df):>8,}")

    # Reproductive age window
    n = len(df); df = df[(df["age"] >= 20) & (df["age"] <= 45)].copy()
    log.append(f"Age 20–45:                   {n:>8,} → {len(df):>8,}")

    # Non-missing treatment
    n = len(df); df = df[df["treated"].notna()].copy()
    log.append(f"Non-missing treatment:       {n:>8,} → {len(df):>8,}")

    # Non-missing birth outcome
    n = len(df); df = df[df["new_birth"].notna()].copy()
    log.append(f"Non-missing outcome:         {n:>8,} → {len(df):>8,}")

    # Drop implausible income
    n = len(df)
    df = df[(df["income"].isna()) | (df["income"] >= 0)].copy()
    log.append(f"Drop negative income:        {n:>8,} → {len(df):>8,}")

    log.append(f"\nFinal analytic sample: {len(df):,} obs across "
               f"{df['pid'].nunique():,} individuals, {df['year'].nunique()} waves")
    return df, "\n".join(log)


# ── Step 6: Final variable construction ──────────────────────────────────────
def build_analysis_vars(df):
    df["log_income"]      = np.log1p(df["income"].clip(lower=0))
    df["log_fam_inc_pc"]  = np.log1p(df["fam_income_pc"].clip(lower=0))
    df["log_house_val"]   = np.log1p(df["house_value"].clip(lower=0) * 1e4)

    # Homeowner dummy (fq2 == 1 in 2016/2020/2022)
    df["homeowner_d"] = (df["homeowner"] == 1).astype(float)
    df.loc[df["homeowner"].isna(), "homeowner_d"] = np.nan

    df["birth_cohort"]  = (df["birthy"] // 5) * 5
    df["provcd_year"]   = df["provcd"].astype(str) + "_" + df["year"].astype(str)

    # Age at first birth (years); only valid for women with at least one child
    fc = df["first_child_birthy"].where(df["first_child_birthy"] > 0, np.nan)
    df["age_at_first_birth"] = fc - df["birthy"]
    df.loc[df["age_at_first_birth"] < 10, "age_at_first_birth"] = np.nan  # sanity filter

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    summary_lines = ["CFPS Data Processing Summary", "="*55]

    KEEP_COLS = [
        "pid", "fid", "year", "provcd", "countyid", "urban",
        "birthy", "age", "gender", "marital", "edu", "eduy", "income",
        "childn", "n_children_fam", "youngest_child_birthy", "first_child_birthy",
        "new_birth", "n_new_births",
        "homeowner", "house_value", "houseasset_net", "houseasset_gross",
        "mortgage_loan", "monthly_rent", "fam_income", "fam_income_pc",
        "n_other_prop",
    ]

    # Load waves
    panel_frames = []
    for yr, cfg in WAVES.items():
        try:
            df_wave, child_cols = load_wave(yr, cfg)
            df_wave = add_birth_outcome(df_wave, yr, child_cols)
            cols    = [c for c in KEEP_COLS if c in df_wave.columns]
            panel_frames.append(df_wave[cols])
            summary_lines.append(f"Wave {yr}: {len(df_wave):,} obs loaded")
        except Exception as e:
            print(f"  ERROR loading wave {yr}: {e}")
            summary_lines.append(f"Wave {yr}: FAILED ({e})")

    panel = pd.concat(panel_frames, ignore_index=True)
    summary_lines.append(f"\nRaw stacked panel: {len(panel):,} obs, "
                         f"{panel['pid'].nunique():,} unique individuals")

    # ── Attach city geography via county crosswalk ───────────────────────────
    xw = load_crosswalk()
    panel["countyid"] = pd.to_numeric(panel["countyid"], errors="coerce")
    panel = panel.merge(xw, on="countyid", how="left")
    cov = panel["city_code"].notna().mean()
    summary_lines.append(f"City geography matched: {cov:.1%} of rows "
                         f"({panel['city_code'].nunique()} cities)")
    print(f"  City geography matched: {cov:.1%} of rows")

    panel.to_csv(os.path.join(OUT_DIR, "panel_individual.csv"), index=False)
    print(f"\nSaved → output/panel_individual.csv ({len(panel):,} rows)")

    # Housing prices (province + city)
    hp_prov = process_housing_prices()
    hp_city = process_city_prices()

    # Merge prices into panel
    panel = panel.merge(hp_prov, on=["provcd", "year"], how="left")
    panel = panel.merge(hp_city, on=["city_code", "year"], how="left")

    # DID treatment — province proxy (legacy fallback): treated / post / did
    panel = build_treatment(panel, hp_prov)

    # DID treatment — HPR spillover (headline; Liu & Zhang 2025 design)
    # Precomputed by scripts/build_hpr_spillover.py (needs the China shapefile).
    # treat_hpr=1 if unregulated prefecture ≤250 km of a regulated city;
    # is_regulated=1 flags the 22 source cities (drop these in analysis).
    panel = merge_hpr_treatment(panel)

    # Sample restrictions
    analysis, restriction_log = apply_sample_restrictions(panel)
    summary_lines += ["\n--- Sample Restriction Log ---", restriction_log]

    # Final vars
    analysis = build_analysis_vars(analysis)
    analysis.to_csv(os.path.join(OUT_DIR, "analysis_data.csv"), index=False)
    print(f"\nSaved → output/analysis_data.csv ({len(analysis):,} rows)")

    # Summary stats
    summary_lines.append("\n--- Key Variable Summary (analytic sample) ---")
    key_vars = ["new_birth", "n_new_births", "age", "eduy", "log_income",
                "homeowner_d", "log_house_val", "log_fam_inc_pc",
                "treated", "post", "did"]
    for v in key_vars:
        if v in analysis.columns:
            s = analysis[v].dropna()
            summary_lines.append(
                f"  {v:<22s}  N={len(s):>6,}  mean={s.mean():.3f}  "
                f"sd={s.std():.3f}  min={s.min():.2f}  max={s.max():.2f}"
            )

    summary_lines.append("\n--- Wave × Treatment cell counts ---")
    if "treated" in analysis.columns:
        tab = analysis.groupby(["year", "treated"])["pid"].count().unstack(fill_value=0)
        summary_lines.append(tab.to_string())

    summary_lines.append("\n--- Birth rate by wave and treatment ---")
    if "treated" in analysis.columns:
        tab2 = analysis.groupby(["year", "treated"])["new_birth"].mean().unstack()
        summary_lines.append(tab2.round(4).to_string())

    summary_lines.append("\n--- City geography coverage (analytic sample) ---")
    has_city = analysis["city_code"].notna()
    summary_lines.append(f"  Rows with city_code:   {has_city.sum():,} of "
                         f"{len(analysis):,} ({has_city.mean():.1%})")
    summary_lines.append(f"  Distinct cities:       {analysis['city_code'].nunique()}")
    has_cityprice = analysis["hp_city_avg"].notna()
    summary_lines.append(f"  Rows with city price:  {has_cityprice.sum():,} "
                         f"({has_cityprice.mean():.1%})")

    if "treat_hpr" in analysis.columns and analysis["treat_hpr"].notna().any():
        summary_lines.append("\n--- HPR spillover treatment (analytic sample) ---")
        reg  = (analysis["is_regulated"] == 1).sum()
        unreg = analysis[analysis["is_regulated"] != 1]
        trt  = (unreg["treat_hpr"] == 1).sum()
        ctl  = (unreg["treat_hpr"] == 0).sum()
        summary_lines.append(f"  Regulated (source, EXCLUDE): {reg:,}")
        summary_lines.append(f"  Unregulated treated (≤250km): {trt:,} "
                             f"({unreg.loc[unreg['treat_hpr']==1,'city_code'].nunique()} cities)")
        summary_lines.append(f"  Unregulated control (>250km): {ctl:,} "
                             f"({unreg.loc[unreg['treat_hpr']==0,'city_code'].nunique()} cities)")
        summary_lines.append("  Birth rate, unregulated × treat_hpr × post:")
        u = unreg.assign(post=(unreg["year"] >= 2016).astype(int))
        summary_lines.append(u.pivot_table("new_birth", "treat_hpr", "post",
                                           aggfunc="mean").round(4).to_string())

    text = "\n".join(summary_lines)
    with open(os.path.join(OUT_DIR, "data_summary.txt"), "w") as f:
        f.write(text)
    print("\nSaved → output/data_summary.txt")
    print("\n" + text)


if __name__ == "__main__":
    main()
