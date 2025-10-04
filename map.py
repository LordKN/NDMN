# -*- coding: utf-8 -*-
"""
Python translation of the final R script.

Outputs:
  - TranspoFoodiePovMap5__python3_reproduce.html
"""

import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import MarkerCluster
import branca

# -------------------------------
# Helpers
# -------------------------------

def parse_number(x):
    """Float from strings like '$56,123' or '12.3%'; None on failure."""
    if pd.isna(x):
        return None
    s = re.sub(r"[^\d\.\-]", "", str(x))
    try:
        return float(s)
    except ValueError:
        return None

def norm_tract_name(x):
    """
    Normalize tract NAME strings so '113.10' -> '113.1' and strip whitespace.
    Keeps names like '101.02' as-is.
    """
    s = str(x).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        # Remove trailing zero(s) but keep necessary decimals
        s = s.rstrip('0').rstrip('.') if '.' in s else s
        if s == "":
            s = "0"
    return s

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
# TIGER usually ships as EPSG:4269; move to 4326 once for web maps
if tracts.crs is None:
    tracts = tracts.set_crs(4269)
tracts = tracts.to_crs(4326)

# Normalize joining key to mirror R's NAME-based merges safely
tracts["NAME"] = tracts["NAME"].astype(str).map(norm_tract_name)
tracts["COUNTYFP"] = tracts["COUNTYFP"].astype(str)

county_fips = ["099", "141", "039"]
filtered = tracts[tracts["COUNTYFP"].isin(county_fips)].copy()

# Per-county subsets like R (optional)
c_elkhart  = filtered[filtered["COUNTYFP"] == "039"].copy()
c_stjoseph = filtered[filtered["COUNTYFP"] == "141"].copy()
c_marshall = filtered[filtered["COUNTYFP"] == "099"].copy()

# -------------------------------
# 2) Read CSVs exactly like R (keep names)
# -------------------------------

# Age
df1 = pd.read_csv("Elkhart_Age_Data_Transposed.csv")
df2 = pd.read_csv("St_Joseph_Age_Data_Transposed.csv")
df3 = pd.read_csv("Marshall_Age_Data_Transposed.csv")
print(df1)


# Median income
df_1 = pd.read_csv("Elkhart_MedianIncome_Transposed.csv")
df_2 = pd.read_csv("St_Joseph_MedianIncome_Transposed.csv")
df_3 = pd.read_csv("Marshall_MedianIncome_Tranposed.csv")  # (typo kept to match your file)

# Poverty
DF1 = pd.read_csv("Elkhart_PovPercent_Transposed.csv")
DF2 = pd.read_csv("St_Joseph_PovPercent_Transposed.csv")
DF3 = pd.read_csv("Marhsall_PovPercent_Transposed.csv")    # (typo kept to match your file)

# Sex (first col becomes NAME in R)
DF_1 = pd.read_csv("elkhart_sex.csv")
DF_2 = pd.read_csv("joseph_sex.csv")
DF_3 = pd.read_csv("marshall_sex.csv")
for d in (DF_1, DF_2, DF_3):
    d.rename(columns={d.columns[0]: "NAME"}, inplace=True)

# Tract metadata / links
DataF1 = pd.read_csv("Elkhart_Tracts_Cleaned.csv")
DataF2 = pd.read_csv("StJOE_Renamed_Tract_Column.csv")
DataF3 = pd.read_csv("Marshall_Tracts_Cleaned.csv")

# Force NAME to str + normalize like the shapefile
for d in (df1, df2, df3, df_1, df_2, df_3, DF1, DF2, DF3, DF_1, DF_2, DF_3, DataF1, DataF2, DataF3):
    if "NAME" in d.columns:
        d["NAME"] = d["NAME"].astype(str).map(norm_tract_name)

# -------------------------------
# 3) Merge per county (by NAME) like R
# -------------------------------

def merge_chain(sf_gdf, *dfs):
    out = sf_gdf.copy()
    for d in dfs:
        if "NAME" in d.columns:
            out = out.merge(d, on="NAME", how="left")
        else:
            raise ValueError("All attribute frames must include NAME for this merge path.")
    return out

# Elkhart
Counties_1 = merge_chain(c_elkhart,  df_1, df1, DF1, DataF1, DF_1)
# St. Joseph
Counties_2 = merge_chain(c_stjoseph, df_2, df2, DF2, DataF2, DF_2)
# Marshall
Counties_3 = merge_chain(c_marshall, df_3, df3, DF3, DataF3, DF_3)

# -------------------------------
# 4) Derived metrics (Under 18, Over 65, percents, poverty+income clean)
# -------------------------------

def safe_sum(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[use].apply(pd.to_numeric, errors="coerce").sum(axis=1)

def add_age_metrics(df):
    def cols_for(bin_rx):
        # candidate columns that match the bin but are NOT percent or MOE
        cand = [c for c in df.columns if re.search(bin_rx, c, re.I)]
        cand = [c for c in cand if not re.search(r'percent|%|margin|moe', c, re.I)]
        # prefer "Total" bin (already both sexes) if present; else use Male+Female
        tot = [c for c in cand if re.search(r'\btotal\b', c, re.I)]
        if tot:
            return tot
        return [c for c in cand if re.search(r'\bmale\b|\bfemale\b', c, re.I)] or cand

    u18_bins = [r'under\s*5\s*years', r'5\s*to\s*9\s*years', r'10\s*to\s*14\s*years', r'15\s*to\s*17\s*years']
    o65_bins = [r'65\s*and\s*66\s*years', r'67\s*to\s*69\s*years', r'70\s*to\s*74\s*years',
                r'75\s*to\s*79\s*years', r'80\s*to\s*84\s*years', r'85\s*years\s*and\s*(over|older)']

    u18_cols = sorted(set(sum((cols_for(rx) for rx in u18_bins), [])))
    o65_cols = sorted(set(sum((cols_for(rx) for rx in o65_bins), [])))

    df["Under_18"] = df[u18_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    df["Over_65"]  = df[o65_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    total_col = next(c for c in df.columns if re.search(r'\btotal(\s|$)|total.*population|estimate.*total', c, re.I))
    total = pd.to_numeric(df[total_col], errors="coerce")

    df["Under_18Per"] = 100 * df["Under_18"] / total
    df["Over_65Per"]  = 100 * df["Over_65"]  / total
    return df



# drop-in replacement: parse MedianIncomeNum robustly (and keep poverty)
def add_income_poverty(df):
    # POVERTY (pick the most-populated "poverty" column, exclude %/MOE)
    pov_cols = [c for c in df.columns if re.search(r'poverty', c, re.I)]
    pov_cols = [c for c in pov_cols if not re.search(r'percent|%|moe|margin', c, re.I)]
    src_pov = max(pov_cols, key=lambda c: df[c].notna().sum()) if pov_cols else None
    df["POVERTY"] = df[src_pov] if src_pov else pd.NA
    df["PovertyNum"] = pd.to_numeric(
        df["POVERTY"].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
    )

    # MEDIAN INCOME (tokens: median + income; prefer dollars/household/family; exclude %/MOE)
    inc_cols = [c for c in df.columns if re.search(r'\bmedian\b.*\bincome\b', c, re.I)]
    inc_cols = [c for c in inc_cols if not re.search(r'percent|%|moe|margin', c, re.I)]
    inc_cols = sorted(
        inc_cols,
        key=lambda c: (
            0 if re.search(r'dollar|household|family', c, re.I) else 1,  # prefer dollars/household/family
            -df[c].notna().sum()                                          # then most non-nulls
        )
    )
    inc_col = inc_cols[0] if inc_cols else None
    df["MedianIncomeNum"] = (
        pd.to_numeric(df[inc_col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
        if inc_col else pd.NA
    )
    return df

for g in (Counties_1, Counties_2, Counties_3):
    add_age_metrics(g)
    add_income_poverty(g)

for nm, g in [("Elkhart", Counties_1), ("St. Joseph", Counties_2), ("Marshall", Counties_3)]:
    print(nm, "non-null Under_18Per:", g["Under_18Per"].notna().sum(),
               "non-null Over_65Per:", g["Over_65Per"].notna().sum())
    print(nm, "non-null MedianIncomeNum:", g["MedianIncomeNum"].notna().sum(),
               "non-null PovertyNum:", g["PovertyNum"].notna().sum())

# Select columns (keep geometry!)
sel_cols = [
    "NAME", "Median.Income.", "Over_65Per", "Under_18Per", "POVERTY",
    "CensusReporter_Link", "MedianIncomeNum", "PovertyNum", "Total", "geometry"
]
Selected_1 = Counties_1[[c for c in sel_cols if c in Counties_1.columns]].copy()
Selected_2 = Counties_2[[c for c in sel_cols if c in Counties_2.columns]].copy()
Selected_3 = Counties_3[[c for c in sel_cols if c in Counties_3.columns]].copy()

merged_gdf = gpd.GeoDataFrame(
    pd.concat([Selected_1, Selected_2, Selected_3], ignore_index=True),
    geometry="geometry", crs=Counties_1.crs
)

# Convert poverty to percent if it came in as fraction (0–1)
pn = pd.to_numeric(merged_gdf["PovertyNum"], errors="coerce")
merged_gdf["PovertyPct"] = np.where(pn <= 1.5, pn * 100, pn)

# Labels for hover
merged_gdf["Population"]     = pd.to_numeric(merged_gdf.get("Total"), errors="coerce")
merged_gdf["PovertyLabel"]   = merged_gdf["PovertyPct"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "NA")
merged_gdf["IncomeLabel"]    = merged_gdf["MedianIncomeNum"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "NA")
merged_gdf["PopulationLabel"]= merged_gdf["Population"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "NA")

# -------------------------------
# 5) Pantries (geocode if needed) + buffers (1 mile)
# -------------------------------

pantries = pd.read_csv("UPDATE5_Final_Cleaned_Pantry_Locations.csv - Sheet1.csv")

# Geocode with ArcGIS only if lat/long missing
if "lat" not in pantries.columns or "long" not in pantries.columns:
    geolocator = ArcGIS(timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.2)
    lat, lon = [], []
    for addr in pantries["Address"].astype(str).tolist():
        loc = geocode(addr)
        if loc:
            lat.append(loc.latitude)
            lon.append(loc.longitude)
        else:
            lat.append(np.nan)
            lon.append(np.nan)
    pantries["lat"]  = lat
    pantries["long"] = lon

pantries = pantries.dropna(subset=["lat", "long"]).copy()
pantries_gdf = gpd.GeoDataFrame(
    pantries, geometry=gpd.points_from_xy(pantries["long"], pantries["lat"]), crs=4326
)

# Buffer in a projected CRS: UTM 16N (EPSG:26916) then back to 4326
pantries_buf = pantries_gdf.to_crs(26916).buffer(1609.34)  # 1 mile in meters
pantries_buf = gpd.GeoSeries(pantries_buf, crs=26916).to_crs(4326)
pantries_buf_gdf = gpd.GeoDataFrame(geometry=pantries_buf, crs=4326)

# -------------------------------
# 6) Bus routes & counties
# -------------------------------

routes = gpd.read_file("TranspoRoutes.shp").to_crs(4326)
line_attr = "line_name" if "line_name" in routes.columns else ("clean_name" if "clean_name" in routes.columns else None)
if not line_attr:
    raise ValueError("TranspoRoutes.shp must include a 'line_name' or 'clean_name' column.")

# Map R colors (only valid CSS color names)
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
# 7) Map: poverty overlay + counties + routes + pantry buffers + markers
# -------------------------------

m = folium.Map(location=[41.68, -86.25], zoom_start=9, tiles="CartoDB positron")

# Poverty choropleth (overlay, initially hidden to match your R hideGroup)
poverty_vals = merged_gdf["PovertyPct"].dropna()
pov_cmap = make_colormap(
    ["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"],
    poverty_vals, "Poverty Level (%)"
)



poverty_fg = FeatureGroup(name="Poverty Level", show=True)

def style_poverty(feat):
    v = feat["properties"].get("PovertyPct", None)
    color = "#bdbdbd" if v is None else pov_cmap(v)
    return {"fillColor": color, "color": "white", "weight": 0.3, "fillOpacity": 0.55}


tooltip_pov = folium.GeoJsonTooltip(
    fields=["NAME", "PovertyLabel", "PopulationLabel"],
    aliases=["Tract", "Poverty", "Population"],
    localize=True, sticky=True
)

popup_pov = folium.GeoJsonPopup(
    fields=["NAME", "PovertyPct", "CensusReporter_Link"],
    aliases=["Tract", "Poverty (%)", "CensusReporter"],
    localize=True, labels=True
)

folium.GeoJson(
    data=merged_gdf.to_json(),
    style_function=style_poverty,
    tooltip=tooltip_pov,
    popup=popup_pov,
    highlight_function=lambda x: {"weight": 2, "color": "#666", "fillOpacity": 0.9}
).add_to(poverty_fg)
poverty_fg.add_to(m)


# ---- Age choropleths ----
# ONLY change: convert to numeric and coerce errors to NaN
# before colormaps (optional but helps with real features):
merged_gdf["Over_65Per"]  = pd.to_numeric(merged_gdf["Over_65Per"], errors="coerce")
merged_gdf["Under_18Per"] = pd.to_numeric(merged_gdf["Under_18Per"], errors="coerce")

# colormaps (as you had)
age65_cmap = make_colormap(["#edf8fb","#b2e2e2","#66c2a4","#238b45"],
                           merged_gdf["Over_65Per"].dropna(), "Over 65 (%)")
u18_cmap   = make_colormap(["#eff3ff","#bdd7e7","#6baed6","#08519c"],
                           merged_gdf["Under_18Per"].dropna(), "Under 18 (%)")

# ONLY change: guard None INSIDE the style funcs so Folium's test feature passes
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
    aliases=["Tract", "% 65+", "% <18", "Population"], localize=True, sticky=True
)
tooltip_u18 = folium.GeoJsonTooltip(
    fields=["NAME", "Under_18Per", "Over_65Per", "PopulationLabel"],
    aliases=["Tract", "% <18", "% 65+", "Population"], localize=True, sticky=True
)

age65_fg = FeatureGroup(name="Over 65 (%)", show=True)
u18_fg = FeatureGroup(name="Under 18 (%)", show=True)

folium.GeoJson(merged_gdf.to_json(), style_function=style_age65, tooltip=tooltip_age65).add_to(age65_fg)
folium.GeoJson(merged_gdf.to_json(), style_function=style_u18,  tooltip=tooltip_u18 ).add_to(u18_fg)

age65_fg.add_to(m)
u18_fg.add_to(m)
age65_cmap.add_to(m)
u18_cmap.add_to(m)

# ---- Median income choropleths ----
income_vals = merged_gdf["MedianIncomeNum"].dropna()
income_cmap = make_colormap(
    ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd"],
    income_vals, "Median Income ($)"
)
def style_income(feat):
    v = feat["properties"].get("MedianIncomeNum", None)
    color = "#bdbdbd" if v is None else income_cmap(v)
    return {"fillColor": color, "color": "white", "weight": 0.3, "fillOpacity": 0.55}
tooltip_income = folium.GeoJsonTooltip(
    fields=["NAME", "IncomeLabel", "PopulationLabel"],
    aliases=["Tract", "Median Income", "Population"],
    localize=True, sticky=True
)
popup_income = folium.GeoJsonPopup(
    fields=["NAME", "MedianIncomeNum", "CensusReporter_Link"],
    aliases=["Tract", "Median Income ($)", "CensusReporter"],
    localize=True, labels=True
)  
income_fg = FeatureGroup(name="Median Income", show=True)
folium.GeoJson(
    data=merged_gdf.to_json(),
    style_function=style_income,
    tooltip=tooltip_income,
    popup=popup_income,
    highlight_function=lambda x: {"weight": 2, "color": "#666", "fillOpacity": 0.9}
).add_to(income_fg)

income_fg.add_to(m)
income_cmap.add_to(m)




# Legend
pov_cmap.add_to(m)


# County boundaries (always on; not toggled)
folium.GeoJson(
    target_counties.to_json(),
    name="County Boundaries",
    style_function=lambda f: {"color": "black", "weight": 3, "opacity": 0.8},
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["County"], localize=True, sticky=True)
).add_to(m)

# Bus routes (overlay, initially hidden like R)
routes_fg = FeatureGroup(name="Bus Routes", show=True)

def route_style(feat):
    nm = feat["properties"].get(line_attr, "")
    col = route_colors.get(nm, "#808080")
    return {"color": col, "weight": 3, "opacity": 0.9}

folium.GeoJson(
    routes.to_json(),
    name="Bus Routes",
    style_function=route_style,
    tooltip=folium.GeoJsonTooltip(fields=[line_attr], aliases=["Route"])
).add_to(routes_fg)
routes_fg.add_to(m)

# Pantry coverage buffers (overlay, initially hidden)
buffers_fg = FeatureGroup(name="Pantry Coverage", show=False)
folium.GeoJson(
    pantries_buf_gdf.to_json(),
    name="Pantry Coverage",
    style_function=lambda f: {"fillColor": "#6A5ACD", "color": "#6A5ACD", "weight": 1, "fillOpacity": 0.1}
).add_to(buffers_fg)
buffers_fg.add_to(m)

# Pantry markers (overlay, initially hidden) with clustering
markers_fg = FeatureGroup(name="Food Pantries", show=False)
mc = MarkerCluster().add_to(markers_fg)
for _, r in pantries.iterrows():
    lat, lon = r["lat"], r["long"]
    if pd.isna(lat) or pd.isna(lon):
        continue
    name = r.get("Pantry.Name", "Pantry")
    addr = r.get("Address", "N/A")
    hours = r.get("Recurring.Hours", "N/A")
    reqs  = r.get("What.to.Bring", "N/A")
    link  = r.get("Link", "")
    html = f"""
    <b>{name}</b><br>
    Address: {addr}<br>
    Hours: {hours}<br>
    Requirements: {reqs}<br>
    <a href="{link}" target="_blank">View on Google Maps</a>
    """
    folium.Marker([lat, lon], popup=folium.Popup(html, max_width=350)).add_to(mc)
markers_fg.add_to(m)

# Layers control (like your R addLayersControl + hideGroup defaults via show=False above)
LayerControl(collapsed=False).add_to(m)

# Save
m.save("TranspoFoodiePovMap5__python3_reproduce.html")
print(" Wrote TranspoFoodiePovMap5__python3_reproduce.html")