# -*- coding: utf-8 -*-
"""
Builds the interactive map with ACS-powered age, income, and poverty layers.

Output:
  TranspoFoodiePovMap5__python3_reproduce_scrape.html
"""

import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import geopandas as gpd
import numpy as np
import requests
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import MarkerCluster
import branca

# -------------------------------
# Helpers
# -------------------------------

def make_colormap(colors, values, caption):
    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 1.0
    cmap = branca.colormap.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    cmap.caption = caption
    return cmap

# -------------------------------
# 1) Read & prep tracts
# -------------------------------

tracts = gpd.read_file("tl_2021_18_tract.shp")
if tracts.crs is None:
    tracts = tracts.set_crs(4269)
tracts = tracts.to_crs(4326)

tracts["COUNTYFP"] = tracts["COUNTYFP"].astype(str)
county_fips = ["099", "141", "039"]  # Marshall, St Joseph, Elkhart
filtered = tracts[tracts["COUNTYFP"].isin(county_fips)].copy()  # keep NAME, GEOID, geometry, etc.

# -------------------------------
# 2) Pull ACS (no CSVs) and join by GEOID
# -------------------------------

YEAR = "2023"
STATE = "18"
COUNTIES = ["039", "099", "141"]

def fetch_acs(vars, dataset="acs/acs5"):
    frames = []
    for c in COUNTIES:
        url = f"https://api.census.gov/data/{YEAR}/{dataset}"
        params = {"get": ",".join(["NAME"] + vars), "for": "tract:*", "in": f"state:{STATE}+county:{c}"}
        r = requests.get(url, params=params); r.raise_for_status()
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

# Poverty percent (share of people below poverty)
pov = fetch_acs(["S1701_C03_001E"], dataset="acs/acs5/subject")
pov.rename(columns={"S1701_C03_001E": "PovertyPct"}, inplace=True)
pov["PovertyPct"] = pd.to_numeric(pov["PovertyPct"], errors="coerce")

# Age bins: total pop, <18 (8 bins M/F), 65+ (12 bins M/F)
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
merged_gdf["IncomeLabel"]      = merged_gdf["MedianIncomeNum"].map(lambda v: f"${float(v):,.0f}" if pd.notna(v) else "NA")
merged_gdf["PovertyLabel"]     = merged_gdf["PovertyPct"].map(lambda v: f"{float(v):.1f}%" if pd.notna(v) else "NA")
merged_gdf["Population"]       = pd.to_numeric(merged_gdf["Total"], errors="coerce")
merged_gdf["PopulationLabel"]  = merged_gdf["Population"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "NA")
merged_gdf["CensusReporter_Link"] = "https://censusreporter.org/profiles/14000US" + merged_gdf["geoid"].astype(str)

# -------------------------------
# 3) Pantries (geocode if needed) + buffers (1 mile)
# -------------------------------

pantries = pd.read_csv("UPDATE5_Final_Cleaned_Pantry_Locations.csv - Sheet1.csv")
if "lat" not in pantries.columns or "long" not in pantries.columns:
    geolocator = ArcGIS(timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.2)
    lat, lon = [], []
    for addr in pantries["Address"].astype(str):
        loc = geocode(addr)
        if loc:
            lat.append(loc.latitude); lon.append(loc.longitude)
        else:
            lat.append(np.nan); lon.append(np.nan)
    pantries["lat"] = lat; pantries["long"] = lon

pantries = pantries.dropna(subset=["lat", "long"]).copy()
pantries_gdf = gpd.GeoDataFrame(
    pantries, geometry=gpd.points_from_xy(pantries["long"], pantries["lat"]), crs=4326
)
pantries_buf = pantries_gdf.to_crs(26916).buffer(1609.34)   # 1 mile in meters (UTM 16N)
pantries_buf_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(pantries_buf, crs=26916).to_crs(4326), crs=4326)

# -------------------------------
# 4) Bus routes & counties
# -------------------------------

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

# -------------------------------
# 5) Map & layers
# -------------------------------

m = folium.Map(location=[41.68, -86.25], zoom_start=9, tiles="CartoDB positron")

# Poverty choropleth
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

pov_fg = FeatureGroup(name="Poverty Level", show=True)
folium.GeoJson(merged_gdf.to_json(), style_function=style_poverty,
               tooltip=tooltip_pov, popup=popup_pov,
               highlight_function=lambda x: {"weight": 2, "color": "#666", "fillOpacity": 0.9}
).add_to(pov_fg)
pov_fg.add_to(m)
pov_cmap.add_to(m)

# Age choropleths
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

age65_fg = FeatureGroup(name="Over 65 (%)", show=False)
u18_fg   = FeatureGroup(name="Under 18 (%)", show=False)
folium.GeoJson(merged_gdf.to_json(), style_function=style_age65, tooltip=tooltip_age65).add_to(age65_fg)
folium.GeoJson(merged_gdf.to_json(), style_function=style_u18,  tooltip=tooltip_u18 ).add_to(u18_fg)
age65_fg.add_to(m); u18_fg.add_to(m)
age65_cmap.add_to(m); u18_cmap.add_to(m)

# Median income choropleth
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

inc_fg = FeatureGroup(name="Median Income", show=False)
folium.GeoJson(merged_gdf.to_json(), style_function=style_income,
               tooltip=tooltip_income, popup=popup_income,
               highlight_function=lambda x: {"weight": 2, "color": "#666", "fillOpacity": 0.9}
).add_to(inc_fg)
inc_fg.add_to(m)
inc_cmap.add_to(m)

# 1) Read and aggregate the FoodOutgoing file
food = pd.read_csv("FoodOutgoing2025_1.csv", encoding="latin-1")
food["County"] = food["County"].astype(str).str.strip()

food["Total Pounds"] = pd.to_numeric(
    food["Total Pounds"].astype(str).str.replace(",", ""),
    errors="coerce"
).fillna(0)
food = food[food["County"].isin(["ELK", "SJ", "MAR"])].copy()

# Sum by county code
sum_by_code = (
    food.groupby("County", as_index=False)["Total Pounds"]
        .sum()
        .rename(columns={"Total Pounds": "TotalPounds"})
)

# 2) Map codes -> county names that match your county shapefile
code_to_name = {"ELK": "Elkhart", "SJ": "St Joseph", "MAR": "Marshall"}
sum_by_code["name"] = sum_by_code["County"].map(code_to_name)

# 3) Join totals onto the county polygons you already loaded
impact_gdf = target_counties.merge(sum_by_code[["name", "TotalPounds"]],
                                   on="name", how="left")
impact_gdf["TotalPounds"] = pd.to_numeric(impact_gdf["TotalPounds"], errors="coerce").fillna(0)
impact_gdf["TotalPoundsLabel"] = impact_gdf["TotalPounds"].map(lambda v: f"{int(round(v)):,}")

# 4) Build a color scale and styles
impact_vals = impact_gdf["TotalPounds"]
impact_cmap = make_colormap(
    colors=["#f7fcf5", "#c7e9c0", "#74c476", "#238b45"],  # light -> dark
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

# 5) Add the county impact layer
impact_fg = FeatureGroup(name="County Impact: Total Pounds", show=False)
folium.GeoJson(
    impact_gdf.to_json(),
    style_function=style_impact,
    tooltip=tooltip_impact,
    highlight_function=lambda x: {"weight": 3, "color": "#333", "fillOpacity": 0.75},
).add_to(impact_fg)
impact_fg.add_to(m)
impact_cmap.add_to(m)

# County boundaries (always on)
folium.GeoJson(
    target_counties.to_json(),
    name="County Boundaries",
    style_function=lambda f: {"color": "black", "weight": 3, "opacity": 0.8},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["County"], localize=True, sticky=True),
).add_to(m)

# Bus routes
routes_fg = FeatureGroup(name="Bus Routes", show=True)
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

# Pantry buffers & markers
buffers_fg = FeatureGroup(name="Pantry Coverage (1 mi)", show=False)
folium.GeoJson(
    pantries_buf_gdf.to_json(),
    name="Pantry Coverage",
    style_function=lambda f: {"fillColor": "#6A5ACD", "color": "#6A5ACD", "weight": 1, "fillOpacity": 0.1},
).add_to(buffers_fg)
buffers_fg.add_to(m)

markers_fg = FeatureGroup(name="Food Pantries", show=False)
mc = MarkerCluster().add_to(markers_fg)
for _, r in pantries.iterrows():
    if pd.isna(r["lat"]) or pd.isna(r["long"]): 
        continue
    html = f"""
    <b>{r.get('Pantry.Name','Pantry')}</b><br>
    Address: {r.get('Address','N/A')}<br>
    Hours: {r.get('Recurring.Hours','N/A')}<br>
    Requirements: {r.get('What.to.Bring','N/A')}<br>
    <a href="{r.get('Link','')}" target="_blank">View on Google Maps</a>
    """
    folium.Marker([r["lat"], r["long"]], popup=folium.Popup(html, max_width=350)).add_to(mc)
markers_fg.add_to(m)

LayerControl(collapsed=False).add_to(m)

# Save
m.save("TranspoFoodiePovMap5__python3_reproduce_scrape.html")
print("Wrote TranspoFoodiePovMap5__python3_reproduce_scrape.html")
