# -*- coding: utf-8 -*-
"""
Builds the interactive map with ACS-powered age, income, and poverty layers,
plus food pantry buffers & markers geocoded via Google Places (Text Search + Details).

Output:
  TranspoFoodiePovMap5__python3_reproduce_scrape.html
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import MarkerCluster, GroupedLayerControl
import branca
from typing import Dict, Any, Optional
from urllib.parse import quote_plus

# ============================================================
# Google Places helpers (Text Search -> place_id -> Details)
# ============================================================

# Prefer env var; fall back to literal if you really must.
API_KEY = "SIKE!"

if not API_KEY:
    # As a last resort, set directly (discouraged — consider env var)
    API_KEY = "REPLACE_ME"

def _check_api_key():
    if API_KEY in ("REPLACE_ME", "NOPE", "", None):
        raise RuntimeError("Google Maps API key missing. Set GOOGLE_MAPS_API_KEY or edit API_KEY.")

def get_place_id(query: str, *, session: Optional[requests.Session] = None) -> str:
    """Fetch Place ID from the Places Text Search API using a free-form query (e.g., location name)."""
    _check_api_key()
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": API_KEY}
    s = session or requests
    resp = s.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    if status != "OK":
        raise RuntimeError(f"Text Search failed: {status} - {data.get('error_message','')}")
    results = data.get("results", [])
    if not results:
        raise RuntimeError("No results returned for that query.")
    return results[0]["place_id"]

def get_place_details(place_id: str, *, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch name, address, and coordinates from Place Details."""
    _check_api_key()
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = "name,formatted_address,geometry"
    params = {"place_id": place_id, "fields": fields, "key": API_KEY}
    s = session or requests
    resp = s.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status")
    if status != "OK":
        raise RuntimeError(f"Place Details failed: {status} - {data.get('error_message','')}")
    result = data.get("result", {})
    loc = (result.get("geometry") or {}).get("location") or {}
    return {
        "place_id": place_id,
        "name": result.get("name"),
        "address": result.get("formatted_address"),
        "lat": loc.get("lat"),
        "long": loc.get("lng"),
    }

# ============================================================
# Helper: continuous colormap
# ============================================================

def make_colormap(colors, values, caption):
    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 1.0
    cmap = branca.colormap.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    cmap.caption = caption
    return cmap

# ============================================================
# 1) Read & prep census tracts (Indiana)
# ============================================================

tracts = gpd.read_file("tl_2021_18_tract.shp")
if tracts.crs is None:
    tracts = tracts.set_crs(4269)
tracts = tracts.to_crs(4326)

tracts["COUNTYFP"] = tracts["COUNTYFP"].astype(str)
county_fips = ["099", "141", "039"]  # Marshall, St Joseph, Elkhart
filtered = tracts[tracts["COUNTYFP"].isin(county_fips)].copy()

# ============================================================
# 2) Pull ACS (no CSVs) and join by GEOID
# ============================================================

YEAR = "2023"
STATE = "18"
COUNTIES = ["039", "099", "141"]

def fetch_acs(vars, dataset="acs/acs5"):
    frames = []
    for c in COUNTIES:
        url = f"https://api.census.gov/data/{YEAR}/{dataset}"
        params = {"get": ",".join(["NAME"] + vars), "for": "tract:*", "in": f"state:{STATE}+county:{c}"}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        cols, *rows = r.json()
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["geoid"] = out["state"] + out["county"] + out["tract"]
    return out

# Median household income
inc = fetch_acs(["B19013_001E"])
inc.rename(columns={"B19013_001E": "MedianIncomeNum"}, inplace=True)
inc["MedianIncomeNum"] = pd.to_numeric(inc["MedianIncomeNum"], errors="coerce")

# Poverty percent
pov = fetch_acs(["S1701_C03_001E"], dataset="acs/acs5/subject")
pov.rename(columns={"S1701_C03_001E": "PovertyPct"}, inplace=True)
pov["PovertyPct"] = pd.to_numeric(pov["PovertyPct"], errors="coerce")

# Age bins
age_vars = ["B01001_001E",
            "B01001_003E","B01001_004E","B01001_005E","B01001_006E",
            "B01001_027E","B01001_028E","B01001_029E","B01001_030E",
            "B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
            "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E"]
age = fetch_acs(age_vars)
age[age_vars] = age[age_vars].apply(pd.to_numeric, errors="coerce")

age["Total"] = age["B01001_001E"]
age["Under_18"] = age[["B01001_003E","B01001_004E","B01001_005E","B01001_006E",
                       "B01001_027E","B01001_028E","B01001_029E","B01001_030E"]].sum(axis=1)
age["Over_65"]  = age[["B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
                       "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E"]].sum(axis=1)
age["Under_18Per"] = 100 * age["Under_18"] / age["Total"]
age["Over_65Per"]  = 100 * age["Over_65"]  / age["Total"]

# Combine ACS tables
acs = (inc[["geoid","MedianIncomeNum"]]
       .merge(pov[["geoid","PovertyPct"]], on="geoid", how="left")
       .merge(age[["geoid","Total","Under_18Per","Over_65Per"]], on="geoid", how="left"))

# Join ACS to tract geometries
merged_gdf = filtered.merge(acs, left_on="GEOID", right_on="geoid", how="left")

# Labels for tooltips/popups
merged_gdf["IncomeLabel"] = merged_gdf["MedianIncomeNum"].map(lambda v: f"${float(v):,.0f}" if pd.notna(v) else "NA")
merged_gdf["PovertyLabel"] = merged_gdf["PovertyPct"].map(lambda v: f"{float(v):.1f}%" if pd.notna(v) else "NA")
merged_gdf["Population"] = pd.to_numeric(merged_gdf["Total"], errors="coerce")
merged_gdf["PopulationLabel"] = merged_gdf["Population"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "NA")
merged_gdf["CensusReporter_Link"] = "https://censusreporter.org/profiles/14000US" + merged_gdf["geoid"].astype(str)

# ============================================================
# 3) Pantry geocoding via Google + buffers (1 mile)
# ============================================================

pantries = pd.read_csv("location_2025.csv")
name_col = "Location Names"     # <-- expects this column with location names
if name_col not in pantries.columns:
    raise RuntimeError(f"Expected a '{name_col}' column in location_2025.csv")

def _norm(s: str) -> str:
    return (s or "").strip().lower()

# Ensure Missing Addresses column exists
if "Missing Addresses" not in pantries.columns:
    pantries["Missing Addresses"] = np.nan

# Normalize name once; needed for fallback name-based geocoding
pantries["_norm_name"] = pantries[name_col].astype(str).map(_norm)

# Ensure output columns exist
if "place_id" not in pantries.columns:
    pantries["place_id"] = None
if "lat" not in pantries.columns:
    pantries["lat"] = np.nan
if "long" not in pantries.columns:
    pantries["long"] = np.nan
if "Address" not in pantries.columns:
    pantries["Address"] = np.nan

# ------------------------------------------------------------
# 1) Geocode rows that HAVE a "Missing Addresses" value
#    → use that address to get exact lat/long
# ------------------------------------------------------------
mask_manual = pantries["Missing Addresses"].notna()

if mask_manual.any():
    manual_addrs = (
        pantries.loc[mask_manual, "Missing Addresses"]
        .astype(str)
        .str.strip()
    )

    unique_manual = (
        manual_addrs
        .replace({"": np.nan})
        .dropna()
        .unique()
    )

    manual_cache: Dict[str, Dict[str, Any]] = {}

    with requests.Session() as sess:
        for addr in unique_manual:
            if addr in manual_cache:
                continue
            try:
                pid = get_place_id(addr, session=sess)      # query by address string
                det = get_place_details(pid, session=sess)
                manual_cache[addr] = det
            except Exception as e:
                manual_cache[addr] = {
                    "place_id": None,
                    "name": None,
                    "address": addr,
                    "lat": np.nan,
                    "long": np.nan,
                }
                print(f"[warn] geocode failed for manual address '{addr}': {e}", file=sys.stderr)

    # Apply geocoded results back to rows with Missing Addresses
    pantries.loc[mask_manual, "place_id"] = manual_addrs.map(
        lambda a: (manual_cache.get(a) or {}).get("place_id")
    )
    pantries.loc[mask_manual, "lat"] = manual_addrs.map(
        lambda a: (manual_cache.get(a) or {}).get("lat")
    )
    pantries.loc[mask_manual, "long"] = manual_addrs.map(
        lambda a: (manual_cache.get(a) or {}).get("long")
    )
    # For popup: keep the manual address text as the display
    pantries.loc[mask_manual, "Address"] = manual_addrs

# ------------------------------------------------------------
# 2) Only use Google API by LOCATION NAME for rows where
#    Missing Addresses is NULL (fallback behavior)
# ------------------------------------------------------------
mask_need_api = pantries["Missing Addresses"].isna()

# Determine if ANY of those rows still need coords or address
need_lat_long_for_mask = pantries.loc[mask_need_api, ["lat", "long"]].isna().any().any()
need_address_for_mask = pantries.loc[mask_need_api, "Address"].isna().any()

if mask_need_api.any() and (need_lat_long_for_mask or need_address_for_mask):
    # Only names for rows that actually need API-based geocoding
    names_to_geocode = pantries.loc[mask_need_api, name_col]

    unique_names = (
        names_to_geocode.astype(str)
        .map(_norm)
        .replace({"": np.nan})
        .dropna()
        .unique()
    )

    cache: Dict[str, Dict[str, Any]] = {}
    with requests.Session() as sess:
        for nm in unique_names:
            if nm in cache:
                continue
            try:
                pid = get_place_id(nm, session=sess)        # query by location name
                det = get_place_details(pid, session=sess)
                cache[nm] = det
            except Exception as e:
                cache[nm] = {
                    "place_id": None,
                    "name": None,
                    "address": None,
                    "lat": np.nan,
                    "long": np.nan,
                }
                print(f"[warn] geocode failed for '{nm}': {e}", file=sys.stderr)

    # Apply results ONLY to rows where Missing Addresses is null
    norm_need_api = pantries.loc[mask_need_api, "_norm_name"]

    # place_id
    pantries.loc[mask_need_api, "place_id"] = pantries.loc[mask_need_api, "place_id"].fillna(
        norm_need_api.map(lambda k: (cache.get(k) or {}).get("place_id"))
    )

    # lat
    pantries.loc[mask_need_api, "lat"] = pantries.loc[mask_need_api, "lat"].fillna(
        norm_need_api.map(lambda k: (cache.get(k) or {}).get("lat"))
    )

    # long
    pantries.loc[mask_need_api, "long"] = pantries.loc[mask_need_api, "long"].fillna(
        norm_need_api.map(lambda k: (cache.get(k) or {}).get("long"))
    )

    # Address (only for rows without manual addresses)
    pantries.loc[mask_need_api, "Address"] = pantries.loc[mask_need_api, "Address"].fillna(
        norm_need_api.map(lambda k: (cache.get(k) or {}).get("address"))
    )

# ------------------------------------------------------------
# 3) Clean + buffers
# ------------------------------------------------------------
# Drop rows without coordinates
pantries = pantries.dropna(subset=["lat", "long"]).copy()

# Deduplicate by place_id; fallback to rounded coordinates
if "place_id" in pantries.columns:
    pantries = pantries.drop_duplicates(subset=["place_id"]).copy()
pantries["_lat_r"] = pantries["lat"].round(6)
pantries["_lon_r"] = pantries["long"].round(6)
pantries = pantries.drop_duplicates(subset=["_lat_r", "_lon_r"]).copy()

# Geo buffer 1 mile
pantries_gdf = gpd.GeoDataFrame(
    pantries, geometry=gpd.points_from_xy(pantries["long"], pantries["lat"]), crs=4326
)
# UTM 16N for northern Indiana; adjust if needed
pantries_buf = pantries_gdf.to_crs(26916).buffer(1609.34)   # 1 mile
pantries_buf_gdf = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(pantries_buf, crs=26916).to_crs(4326),
    crs=4326
)


# ============================================================
# 4) Bus routes & counties
# ============================================================

routes = gpd.read_file("TranspoRoutes.shp").to_crs(4326)
line_attr = "line_name" if "line_name" in routes.columns else ("clean_name" if "clean_name" in routes.columns else None)
if not line_attr:
    raise ValueError("TranspoRoutes.shp must include 'line_name' or 'clean_name'.")

route_colors = {
    "1 Madison / Mishawaka": "navy",
    "10 Western Avenue": "turquoise",
    "11 Southside Mishawaka": "maroon",
    "12 Rum Village": "midnightblue",
    "12/14 Rum Village / Sample": "thistle",
    "13 Corby / Town & Country": "gold",
    "14 Sample / Mayflower": "mediumpurple",
    "15A University Park Mall / Mishawaka (via Main Stree": "saddlebrown",
    "15B University Park Mall / Mishawaka (via Grape Road": "burlywood",
    "16 Blackthorn Express": "hotpink",
    "17 The Sweep": "olivedrab",
    "3A Portage": "firebrick",
    "3B Portage": "crimson",
    "4 Lincolnway West / Excel Center / Airport": "darkorange",
    "5 North Michigan / Laurel Woods": "navy",
    "6 South Michigan / Erskine Village": "red",
    "7 Notre Dame / University Park Mall": "forestgreen",
    "7A Notre Dame Midnight Express": "seagreen",
    "8 Miami / Scottsdale": "turquoise",
    "8/6 Miami / Scottsdale / South Michigan / Erskine Vi": "red",
    "9 Northside Mishawaka": "magenta"
}

counties = gpd.read_file("County_Boundaries_of_Indiana_Current.shp").to_crs(4326)
target_counties = counties[counties["name"].isin(["Elkhart", "Marshall", "St Joseph"])].copy()

# ============================================================
# 5) Map & layers (build ONCE)
# ============================================================

# Create the map ONCE (don’t overwrite later)
m = folium.Map(location=[41.68, -86.25], zoom_start=9, control_scale=True, tiles="OpenStreetMap")
# Add some alternative base tiles to get radio buttons
folium.TileLayer("CartoDB positron", name="Light", overlay = True, control=True).add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark",overlay= True, control=True).add_to(m)

# Poverty choropleth (overlay checkbox)
pov_vals = merged_gdf["PovertyPct"].dropna()
pov_cmap = make_colormap(["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"], pov_vals, "Poverty Level (%)")

def style_poverty(feat):
    v = feat["properties"].get("PovertyPct")
    v = pov_cmap.vmin if v is None else float(v)
    return {"fillColor": pov_cmap(v), "color": "white", "weight": 0.3, "fillOpacity": 0.55}

tooltip_pov = folium.GeoJsonTooltip(
    fields=["NAME", "PovertyLabel", "PopulationLabel"],
    aliases=["Tract", "Poverty", "Population"], localize=True, sticky=True,
)
popup_pov = folium.GeoJsonPopup(
    fields=["NAME", "PovertyPct", "CensusReporter_Link"],
    aliases=["Tract", "Poverty (%)", "CensusReporter"], localize=True, labels=True,
)

pov_fg = FeatureGroup(name="Poverty Level", overlay = False, show=True)
folium.GeoJson(
    merged_gdf.to_json(),
    style_function=style_poverty,
    tooltip=tooltip_pov,
    popup=popup_pov,
    highlight_function=lambda x: {"weight": 2, "color": "#666", "fillOpacity": 0.9}
).add_to(pov_fg)
pov_fg.add_to(m)
pov_cmap.add_to(m)

# Age choropleths (overlays)
merged_gdf["Over_65Per"]  = pd.to_numeric(merged_gdf["Over_65Per"], errors="coerce")
merged_gdf["Under_18Per"] = pd.to_numeric(merged_gdf["Under_18Per"], errors="coerce")

age65_cmap = make_colormap(["#edf8fb","#b2e2e2","#66c2a4","#238b45"],
                           merged_gdf["Over_65Per"].dropna(), "Over 65 (%)")
u18_cmap   = make_colormap(["#eff3ff","#bdd7e7","#6baed6","#08519c"],
                           merged_gdf["Under_18Per"].dropna(), "Under 18 (%)")

def style_age65(f):
    v = f["properties"].get("Over_65Per")
    v = age65_cmap.vmin if v is None else float(v)
    return {"fillColor": age65_cmap(v), "color": "white", "weight": 0.3, "fillOpacity": 0.55}
def style_u18(f):
    v = f["properties"].get("Under_18Per")
    v = u18_cmap.vmin if v is None else float(v)
    return {"fillColor": u18_cmap(v), "color": "white", "weight": 0.3, "fillOpacity": 0.55}

tooltip_age65 = folium.GeoJsonTooltip(
    fields=["NAME", "Over_65Per", "Under_18Per", "PopulationLabel"],
    aliases=["Tract", "% 65+", "% <18", "Population"], localize=True, sticky=True,
)
tooltip_u18 = folium.GeoJsonTooltip(
    fields=["NAME", "Under_18Per", "Over_65Per", "PopulationLabel"],
    aliases=["Tract", "% <18", "% 65+", "Population"], localize=True, sticky=True,
)

age65_fg = FeatureGroup(name="Over 65 (%)", overlay = False, show=False)
u18_fg   = FeatureGroup(name="Under 18 (%)", overlay = False, show=False)
folium.GeoJson(merged_gdf.to_json(), style_function=style_age65, tooltip=tooltip_age65).add_to(age65_fg)
folium.GeoJson(merged_gdf.to_json(), style_function=style_u18,  tooltip=tooltip_u18 ).add_to(u18_fg)
age65_fg.add_to(m); u18_fg.add_to(m)
age65_cmap.add_to(m); u18_cmap.add_to(m)

# Median income choropleth (overlay)
inc_vals = merged_gdf["MedianIncomeNum"].dropna()
inc_cmap = make_colormap(["#f7fbff", "#deebf7", "#9ecae1", "#3182bd"], inc_vals, "Median Income ($)")

def style_income(feat):
    v = feat["properties"].get("MedianIncomeNum")
    v = inc_cmap.vmin if v is None else float(v)
    return {"fillColor": inc_cmap(v), "color": "white", "weight": 0.3, "fillOpacity": 0.55}

tooltip_income = folium.GeoJsonTooltip(
    fields=["NAME", "IncomeLabel", "PopulationLabel"],
    aliases=["Tract", "Median Income", "Population"], localize=True, sticky=True,
)
popup_income = folium.GeoJsonPopup(
    fields=["NAME", "MedianIncomeNum", "CensusReporter_Link"],
    aliases=["Tract", "Median Income ($)", "CensusReporter"], localize=True, labels=True,
)

inc_fg = FeatureGroup(name="Median Income", overlay = False, show=False)
folium.GeoJson(
    merged_gdf.to_json(),
    style_function=style_income,
    tooltip=tooltip_income,
    popup=popup_income,
    highlight_function=lambda x: {"weight": 2, "color": "#666", "fillOpacity": 0.9}
).add_to(inc_fg)
inc_fg.add_to(m)
inc_cmap.add_to(m)

# FoodOutgoing county impact (overlay)
food = pd.read_csv("FoodOutgoing2025_1.csv", encoding="latin-1")
food["County"] = food["County"].astype(str).str.strip()
food["Total Pounds"] = pd.to_numeric(food["Total Pounds"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
food = food[food["County"].isin(["ELK", "SJ", "MAR"])].copy()

sum_by_code = (
    food.groupby("County", as_index=False)["Total Pounds"]
        .sum()
        .rename(columns={"Total Pounds": "TotalPounds"})
)
code_to_name = {"ELK": "Elkhart", "SJ": "St Joseph", "MAR": "Marshall"}
sum_by_code["name"] = sum_by_code["County"].map(code_to_name)

impact_gdf = target_counties.merge(sum_by_code[["name", "TotalPounds"]], on="name", how="left")
impact_gdf["TotalPounds"] = pd.to_numeric(impact_gdf["TotalPounds"], errors="coerce").fillna(0)
impact_gdf["TotalPoundsLabel"] = impact_gdf["TotalPounds"].map(lambda v: f"{int(round(v)):,}")

impact_vals = impact_gdf["TotalPounds"]
impact_cmap = make_colormap(
    colors=["#f7fcf5", "#c7e9c0", "#74c476", "#238b45"],
    values=impact_vals,
    caption="Total Pounds Distributed (by County)"
)

def style_impact(feat):
    v = feat["properties"].get("TotalPounds", 0.0)
    return {
        "fillColor": impact_cmap(float(v)),
        "color": "black",
        "weight": 2,
        "fillOpacity": 0.55
    }

tooltip_impact = folium.GeoJsonTooltip(
    fields=["name", "TotalPoundsLabel"],
    aliases=["County", "Total Pounds"],
    localize=True,
    sticky=True,
)

impact_fg = FeatureGroup(name="County Impact: Total Pounds", overlay = False, show=False)
folium.GeoJson(
    impact_gdf.to_json(),
    style_function=style_impact,
    tooltip=tooltip_impact,
    highlight_function=lambda x: {"weight": 3, "color": "#333", "fillOpacity": 0.75},
).add_to(impact_fg)
impact_fg.add_to(m)
impact_cmap.add_to(m)

# County boundaries (overlay)
county_fg = FeatureGroup(name="County Boundaries", overlay=True, show=True)
folium.GeoJson(
    target_counties.to_json(),
    style_function=lambda f: {"color": "black", "weight": 3, "opacity": 0.8},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["County"], localize=True, sticky=True),
).add_to(county_fg)
county_fg.add_to(m)


# Bus routes (overlay)
routes_fg = FeatureGroup(name="Bus Routes", overlay = False, show=True)
def route_style(feat):
    nm = feat["properties"].get(line_attr, "")
    col = route_colors.get(nm, "#808080")
    return {"color": col, "weight": 3, "opacity": 0.9}
folium.GeoJson(
    routes.to_json(),
    name="Bus Routes",
    style_function=route_style,
    tooltip=folium.GeoJsonTooltip(fields=[line_attr], aliases=["Route"]),
).add_to(routes_fg)
routes_fg.add_to(m)

# Pantry buffers (overlay)
buffers_fg = FeatureGroup(name="Pantry Coverage (1 mi)", overlay = False, show=True)
folium.GeoJson(
    pantries_buf_gdf.to_json(),
    name="Pantry Coverage",
    style_function=lambda f: {"fillColor": "#6A5ACD", "color": "#6A5ACD", "weight": 1, "fillOpacity": 0.1},
).add_to(buffers_fg)
buffers_fg.add_to(m)

# Pantry markers (overlay)
pantries["_program_norm"] = pantries.get("Program", "").astype(str).str.strip().str.lower()

all_df       = pantries
ccfn_df      = pantries[pantries["_program_norm"] == "ccfn"]
backpack_df  = pantries[pantries["_program_norm"] == "backpack"]

def _add_marker_group(group_df: pd.DataFrame, group_name: str, show: bool = False):
    fg = FeatureGroup(name=group_name, overlay=False, show=show)
    mc = MarkerCluster().add_to(fg)

    for _, r in group_df.iterrows():
        if pd.isna(r.get("lat")) or pd.isna(r.get("long")):
            continue

        pantry_name = str(r.get(name_col, "Unnamed")).strip()

        # Prefer Missing Addresses value if present, then Address
        missing_raw = r.get("Missing Addresses")
        if pd.isna(missing_raw):
            missing_addr = ""
        else:
            missing_addr = str(missing_raw).strip()

        base_addr = str(r.get("Address", "") or "").strip()

        display_addr = missing_addr or base_addr or "N/A"

        # Build Google Maps query:
        # 1) address (manual or from API)
        # 2) pantry name
        # 3) lat,long as last resort
        if display_addr != "N/A":
            query_str = display_addr
        elif pantry_name and pantry_name != "Unnamed":
            query_str = pantry_name
        else:
            query_str = f"{r['lat']},{r['long']}"

        maps_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(query_str)}"

        html = f"""
        <b>{pantry_name}</b><br>
        Address: {display_addr}<br>
        <a href="{maps_url}" target="_blank">View on Google Maps</a>
        """

        folium.Marker(
            [r["lat"], r["long"]],
            popup=folium.Popup(html, max_width=350)
        ).add_to(mc)

    fg.add_to(m)
    return fg

# Create three base layers for markers
pantries_all_fg      = _add_marker_group(all_df,      "Pantries: All",      show=True)
pantries_ccfn_fg     = _add_marker_group(ccfn_df,     "Pantries: CCFN",     show=False)
pantries_backpack_fg = _add_marker_group(backpack_df, "Pantries: Backpack", show=False)


# Fit to pantry extent if available
if not pantries.empty:
    minx, miny, maxx, maxy = pantries_gdf.total_bounds
    if np.isfinite([minx, miny, maxx, maxy]).all():
        m.fit_bounds([[miny, minx], [maxy, maxx]])

# Layer control at the end (for base tiles etc.)
LayerControl(collapsed=False).add_to(m)

# Grouped layer control with collapsible headings and radio-circle buttons
GroupedLayerControl(
    groups={
        "Demographics": [pov_fg, age65_fg, u18_fg, inc_fg],
        "Pantries": [buffers_fg, pantries_all_fg, pantries_ccfn_fg, pantries_backpack_fg],
        "Transportation": [routes_fg],
        "Counties & Impact": [county_fg, impact_fg],
    },
    # exclusive_groups=True  # default; keeps the radio-circle behavior within each group
    collapsed=False,
).add_to(m)


# Save
out_file = "TranspoFoodiePovMap5__python3_reproduce_scrape.html"
m.save(out_file)
print(f"Wrote {out_file}")
