# ==========================================================================
# REPLICATION PACKAGE — build_hpr_spillover.py
# Tier: CORE
# Paper items: none directly (support module)
# Reads: output/analysis_data.csv
# Writes: output/hpr_treatment_city.csv
# Imports local modules: none
# Run order: step 2 in run_all.sh
# ==========================================================================

"""
Build the HPR spillover treatment (Liu & Zhang 2025 design).

Treat = 1 if an UNREGULATED prefecture lies within 250 km of the nearest
regulated (HPR) city; regulated cities themselves are the shock source and are
flagged for exclusion.

Steps:
  1. Load prefecture polygons (data/ChinaAdminDivisonSHP-master/3. City/city.shp),
     project to China Albers Equal Area, take centroids (metres).
  2. Build centroids for the 22 regulated HPR cities (data/hpr_cities.csv),
     handling direct-controlled municipalities (codes ending 0000) by dissolving
     their districts.
  3. For every prefecture, compute Euclidean distance (Albers ≈ true km) to the
     nearest HPR city → dist_to_hpr_km.
  4. treat_spillover = 1 if 0 < dist ≤ 250 and not itself regulated.
Output: output/hpr_treatment_city.csv  (city_code, dist_to_hpr_km, is_regulated,
        treat_spillover) + a coverage report against the CFPS analytic sample.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHP  = os.path.join(BASE, "data/ChinaAdminDivisonSHP-master/3. City/city.shp")
ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs")
CUTOFF_KM = 250.0

# ── 1. Prefecture centroids ───────────────────────────────────────────────────
g = gpd.read_file(SHP).to_crs(ALBERS)
g["cx"] = g.geometry.centroid.x
g["cy"] = g.geometry.centroid.y
g["ct_adcode"] = g["ct_adcode"].astype(int)
g["pr_adcode"] = g["pr_adcode"].astype(int)

# Municipality (4 direct-controlled) centroids = dissolve districts by province code
muni = (g[g["pr_adcode"].isin([110000, 120000, 310000, 500000])]
        .dissolve(by="pr_adcode").reset_index())
muni["cx"] = muni.geometry.centroid.x
muni["cy"] = muni.geometry.centroid.y

def centroid_for(code):
    """Return (cx, cy) for a GB city/municipality code, else None."""
    code = int(code)
    hit = g[g["ct_adcode"] == code]
    if len(hit):
        return hit.iloc[0]["cx"], hit.iloc[0]["cy"]
    hit = muni[muni["pr_adcode"] == code]      # municipality fallback
    if len(hit):
        return hit.iloc[0]["cx"], hit.iloc[0]["cy"]
    # last resort: any prefecture whose pr_adcode matches (e.g. code given at prov level)
    hit = g[g["pr_adcode"] == code]
    if len(hit):
        return hit["cx"].mean(), hit["cy"].mean()
    return None

# ── 2. HPR city centroids ─────────────────────────────────────────────────────
hpr = pd.read_csv(os.path.join(BASE, "data/hpr_cities.csv"))
hpr_pts = []
missing = []
for _, r in hpr.iterrows():
    c = centroid_for(r["city_code"])
    if c is None:
        missing.append((r["city_en"], r["city_code"]))
    else:
        hpr_pts.append((r["city_code"], c[0], c[1]))
print(f"HPR centroids resolved: {len(hpr_pts)}/22",
      f"| missing: {missing}" if missing else "")
hx = np.array([p[1] for p in hpr_pts]); hy = np.array([p[2] for p in hpr_pts])
hpr_codes = set(int(p[0]) for p in hpr_pts)

# ── 3. Distance from every prefecture to nearest HPR city ─────────────────────
def nearest_km(cx, cy):
    return np.min(np.hypot(hx - cx, hy - cy)) / 1000.0

rows = []
# build a master code list: all shapefile prefectures + the 4 municipalities
codes = list(g["ct_adcode"].unique()) + [110000, 120000, 310000, 500000]
seen = set()
for code in codes:
    if code in seen:
        continue
    seen.add(code)
    c = centroid_for(code)
    if c is None:
        continue
    d = nearest_km(*c)
    is_reg = int(code in hpr_codes)
    rows.append({"city_code": int(code), "dist_to_hpr_km": round(d, 1),
                 "is_regulated": is_reg,
                 "treat_spillover": int((0 < d <= CUTOFF_KM) and not is_reg)})
tab = pd.DataFrame(rows).drop_duplicates("city_code").sort_values("city_code")
tab.to_csv(os.path.join(BASE, "output/hpr_treatment_city.csv"), index=False)
print(f"Saved output/hpr_treatment_city.csv — {len(tab)} prefectures "
      f"({tab['treat_spillover'].sum()} treated, {tab['is_regulated'].sum()} regulated)")

# ── 4. Coverage in the CFPS analytic sample ───────────────────────────────────
a = pd.read_csv(os.path.join(BASE, "output/analysis_data.csv"))
a = a[a["city_code"].notna()].copy(); a["city_code"] = a["city_code"].astype(int)
# analysis_data may already carry (empty or old) treatment columns; use the fresh ones
a = a.drop(columns=[c for c in tab.columns if c != "city_code" and c in a.columns])
m = a.merge(tab, on="city_code", how="left")
print("\n=== CFPS analytic sample × HPR spillover treatment ===")
print(f"rows with city_code:        {len(a):,}")
print(f"matched to treatment table: {m['treat_spillover'].notna().sum():,}")
print(f"  regulated (EXCLUDE/source): {(m['is_regulated']==1).sum():,}  "
      f"({m.loc[m['is_regulated']==1,'cityname'].nunique()} cities)")
unreg = m[m["is_regulated"] != 1]
print(f"  unregulated treated (≤250km): {(unreg['treat_spillover']==1).sum():,}  "
      f"({unreg.loc[unreg['treat_spillover']==1,'cityname'].nunique()} cities)")
print(f"  unregulated control (>250km): {(unreg['treat_spillover']==0).sum():,}  "
      f"({unreg.loc[unreg['treat_spillover']==0,'cityname'].nunique()} cities)")
print("\ntreated/control birth rate (unregulated only, pre/post 2016):")
unreg = unreg.assign(post=(unreg["year"]>=2016).astype(int))
print(unreg.pivot_table("new_birth", "treat_spillover", "post",
                        aggfunc=["mean","size"]).round(4))
