import glob
import io
import os
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\felix\OneDrive - Erasmus University Rotterdam\New folder\New folder\Courses\Big Data Managment and Analytics\Assignments\Group_Assignment\New"
RAW_WEATHER_DIR = os.path.join(BASE_DIR, "raw_weather")


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine great-circle distance in kilometers."""
    r = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def parse_knmi_txt_file(path: str) -> pd.DataFrame:
    """Parse a raw KNMI hourly text file if CSVs are not yet combined."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if l.lstrip("#").strip().startswith(("STN,", "STN ,"))),
        None,
    )
    if header_idx is None:
        return pd.DataFrame()
    header = lines[header_idx].lstrip("#").strip()
    n_commas = header.count(",")
    rows = [header] + [
        l.strip()
        for l in lines[header_idx + 1 :]
        if l.strip() and not l.lstrip().startswith("#") and l.strip().count(",") == n_commas
    ]
    df = pd.read_csv(io.StringIO("\n".join(rows)), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_weather_data() -> pd.DataFrame:
    combined_candidates = [
        os.path.join(BASE_DIR, "knmi_hourly_weather_2017_2025.csv"),
        os.path.join(RAW_WEATHER_DIR, "knmi_hourly_weather_2017_2025.csv"),
    ]
    for p in combined_candidates:
        if os.path.exists(p):
            print(f"Loading combined weather file: {p}")
            return pd.read_csv(p)

    year_csvs = sorted(glob.glob(os.path.join(RAW_WEATHER_DIR, "knmi_hourly_*.csv")))
    if year_csvs:
        print(f"Loading {len(year_csvs)} yearly KNMI CSVs from {RAW_WEATHER_DIR}...")
        return pd.concat([pd.read_csv(p) for p in year_csvs], ignore_index=True)

    year_txts = sorted(glob.glob(os.path.join(RAW_WEATHER_DIR, "knmi_hourly_*.txt")))
    parsed = [parse_knmi_txt_file(p) for p in year_txts]
    parsed = [d for d in parsed if not d.empty]
    if parsed:
        print(f"Parsed {len(parsed)} KNMI .txt files from {RAW_WEATHER_DIR}...")
        return pd.concat(parsed, ignore_index=True)

    raise FileNotFoundError(f"No KNMI hourly weather files found in {RAW_WEATHER_DIR} or {BASE_DIR}.")


# 1. Load disruptions & stations
trains_path = os.path.join(BASE_DIR, "disruptions_2017_2025.csv")
stations_path = os.path.join(BASE_DIR, "stations-2023-09.csv")

trains = pd.read_csv(trains_path)
stations = pd.read_csv(stations_path)
print(f"Loaded disruptions: {len(trains):,} rows | stations: {len(stations):,} rows")

# 2. Extract primary station code & station metadata
code_col = next((c for c in ["rdt_station_codes", "station_codes", "stations"] if c in trains.columns), None)
stn_code_col = next((c for c in ["code", "station_code", "id"] if c in stations.columns), None)
lat_col = next((c for c in ["geo_lat", "lat", "latitude"] if c in stations.columns), None)
lon_col = next((c for c in ["geo_lng", "lon", "longitude"] if c in stations.columns), None)

if code_col and stn_code_col and lat_col and lon_col:
    trains["primary_station"] = (
        trains[code_col].astype(str).str.split(",").str[0].str.strip().str.upper()
    )
    stn_lookup = stations.copy()
    stn_lookup["_code"] = stn_lookup[stn_code_col].astype(str).str.strip().str.upper()
    stn_lookup = stn_lookup.drop_duplicates(subset=["_code"])

    keep_stn_cols = ["_code", lat_col, lon_col] + [
        c for c in ["country", "type", "name_long", "name"] if c in stn_lookup.columns and c not in trains.columns
    ]
    trains = trains.merge(stn_lookup[keep_stn_cols], left_on="primary_station", right_on="_code", how="left")
    trains = trains.rename(columns={lat_col: "geo_lat", lon_col: "geo_lng"}).drop(columns=["_code"])
else:
    print("Warning: Could not auto-match station code/lat/lon columns; check stations-2023-09.csv headers.")

# Count affected stations/lines if present
if "num_stations" not in trains.columns and code_col:
    trains["num_stations"] = trains[code_col].fillna("").apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )
line_col = next((c for c in ["rdt_lines_id", "rdt_lines", "ns_lines"] if c in trains.columns), None)
if "num_lines" not in trains.columns and line_col:
    trains["num_lines"] = trains[line_col].fillna("").apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )

# 3. Load and clean KNMI hourly weather
weather = load_weather_data()
weather.columns = [c.strip() for c in weather.columns]

before_w = len(weather)
weather = weather.drop_duplicates(subset=["STN", "YYYYMMDD", "HH"]).reset_index(drop=True)
print(f"Weather rows: {before_w:,} -> {len(weather):,} after deduplicating (STN, YYYYMMDD, HH)")

# Convert numeric KNMI columns (strip spaces, coerce non-numeric)
num_w_cols = [c for c in ["STN", "YYYYMMDD", "HH", "FH", "FX", "T", "RH", "DR", "VV", "S", "Y", "O", "M"] if c in weather.columns]
for c in num_w_cols:
    weather[c] = pd.to_numeric(weather[c], errors="coerce")

# KNMI HH=1..24 represents hour ending at HH UTC -> start of hour is (HH - 1) UTC
hh_int = weather["HH"].fillna(1).astype(int).clip(1, 24)
date_str = weather["YYYYMMDD"].astype(int).astype(str).str.zfill(8)
utc_dt = pd.to_datetime(date_str, format="%Y%m%d", utc=True) + pd.to_timedelta(hh_int - 1, unit="h")
ams_dt = utc_dt.dt.tz_convert("Europe/Amsterdam")

weather["ams_hour_key"] = ams_dt.dt.tz_localize(None).dt.floor("h")
weather["local_hour"] = ams_dt.dt.strftime("%Y-%m-%d %H:00")

# Convert KNMI 0.1-unit scales where applicable (check ranges carefully)
# In KNMI hourly tables: FH, FX, T are in 0.1 m/s or 0.1 C; RH in 0.1 mm; DR in 0.1 hour (0..10)
for col, factor, thresh in [("FH", 0.1, 60), ("FX", 0.1, 60), ("T", 0.1, 60), ("RH", 0.1, 60), ("DR", 0.1, 1.0)]:
    if col in weather.columns and weather[col].dropna().abs().max() > thresh:
        weather[col] = (weather[col] * factor).round(1)
        # Only scale if values look like tenths (e.g., max > 45 for T or max > 60 for FH)
        if weather[col].dropna().abs().max() > 60:
            weather[col] = weather[col] * factor

# Clean negative precipitation flags (-1 in KNMI means <0.05 mm)
if "RH" in weather.columns:
    weather.loc[weather["RH"] < 0, "RH"] = 0.0
if "DR" in weather.columns:
    weather.loc[weather["DR"] < 0, "DR"] = 0.0

# 4. Find nearest KNMI station for each disruption with coordinates
# Reference KNMI station coordinates (STN -> lat, lon) if knmi_stations.csv doesn't have lat/lon
knmi_ref_coords = {
    215: (52.141, 4.437),  # Voorschoten
    235: (52.928, 4.781),  # De Kooy
    240: (52.318, 4.790),  # Schiphol
    260: (52.100, 5.180),  # De Bilt
    275: (52.056, 5.873),  # Deelen
    278: (52.435, 6.259),  # Heino
    280: (53.125, 6.585),  # Eelde
    290: (52.274, 6.891),  # Twenthe
    310: (51.442, 3.596),  # Vlissingen
    344: (51.962, 4.447),  # Rotterdam
    350: (51.566, 4.936),  # Gilze-Rijen
    370: (51.451, 5.377),  # Eindhoven
    380: (50.906, 5.762),  # Maastricht
}

knmi_stn_candidates = [
    os.path.join(RAW_WEATHER_DIR, "knmi_stations.csv"),
    os.path.join(BASE_DIR, "knmi_stations.csv"),
]
for knmi_stn_path in knmi_stn_candidates:
    if os.path.exists(knmi_stn_path):
        ks = pd.read_csv(knmi_stn_path)
        ks_stn_col = next((c for c in ["STN", "stn", "id"] if c in ks.columns), None)
        ks_lat_col = next((c for c in ["LAT", "lat", "geo_lat"] if c in ks.columns), None)
        ks_lon_col = next((c for c in ["LON", "lon", "geo_lng"] if c in ks.columns), None)
        if ks_stn_col and ks_lat_col and ks_lon_col:
            for _, r in ks.dropna(subset=[ks_stn_col, ks_lat_col, ks_lon_col]).iterrows():
                knmi_ref_coords[int(r[ks_stn_col])] = (float(r[ks_lat_col]), float(r[ks_lon_col]))
            print(f"Loaded exact KNMI station coordinates from: {knmi_stn_path}")
        break

knmi_stn_path = os.path.join(BASE_DIR, "knmi_stations.csv")
if os.path.exists(knmi_stn_path):
    ks = pd.read_csv(knmi_stn_path)
    ks_stn_col = next((c for c in ["STN", "stn", "id"] if c in ks.columns), None)
    ks_lat_col = next((c for c in ["LAT", "lat", "geo_lat"] if c in ks.columns), None)
    ks_lon_col = next((c for c in ["LON", "lon", "geo_lng"] if c in ks.columns), None)
    if ks_stn_col and ks_lat_col and ks_lon_col:
        for _, r in ks.dropna(subset=[ks_stn_col, ks_lat_col, ks_lon_col]).iterrows():
            knmi_ref_coords[int(r[ks_stn_col])] = (float(r[ks_lat_col]), float(r[ks_lon_col]))

available_stns = sorted(set(weather["STN"].dropna().astype(int)).intersection(knmi_ref_coords.keys()))
stn_ids = np.array(available_stns)
stn_lats = np.array([knmi_ref_coords[s][0] for s in available_stns])
stn_lons = np.array([knmi_ref_coords[s][1] for s in available_stns])

trains["nearest_stn"] = np.nan
trains["dist_to_stn_km"] = np.nan

valid_coords = trains["geo_lat"].notna() & trains["geo_lng"].notna()
for idx in trains.index[valid_coords]:
    d_km = haversine_km(trains.at[idx, "geo_lat"], trains.at[idx, "geo_lng"], stn_lats, stn_lons)
    min_i = int(np.argmin(d_km))
    trains.at[idx, "nearest_stn"] = stn_ids[min_i]
    trains.at[idx, "dist_to_stn_km"] = float(d_km[min_i])

# 5. Parse start_time, compute target, and merge weather on (nearest_stn, ams_hour_key)
trains["start_dt"] = pd.to_datetime(trains["start_time"], errors="coerce")
if "duration_minutes" not in trains.columns and "end_time" in trains.columns:
    end_dt = pd.to_datetime(trains["end_time"], errors="coerce")
    trains["duration_minutes"] = (end_dt - trains["start_dt"]).dt.total_seconds() / 60.0
else:
    trains["duration_minutes"] = pd.to_numeric(trains["duration_minutes"], errors="coerce")

trains = trains[trains["duration_minutes"].notna() & (trains["duration_minutes"] >= 0)].copy()
trains["is_long_disruption"] = (trains["duration_minutes"] > 60).astype(int)

trains["ams_hour_key"] = trains["start_dt"].dt.floor("h")
trains["year"] = trains["start_dt"].dt.year
trains["month"] = trains["start_dt"].dt.month
trains["day_of_week"] = trains["start_dt"].dt.dayofweek
trains["hour"] = trains["start_dt"].dt.hour
trains["is_weekend"] = trains["day_of_week"].isin([5, 6]).astype(int)
trains["is_rush_hour"] = (
    (trains["is_weekend"] == 0) & (trains["hour"].isin([7, 8, 9, 16, 17, 18]))
).astype(int)

weather_merge_cols = [
    c for c in ["STN", "ams_hour_key", "local_hour", "FH", "FX", "T", "RH", "DR", "VV", "S", "Y", "O", "M"]
    if c in weather.columns
]
merged = trains.merge(
    weather[weather_merge_cols],
    left_on=["nearest_stn", "ams_hour_key"],
    right_on=["STN", "ams_hour_key"],
    how="left",
)

# Drop end_time so post-event resolution time cannot leak into modeling
if "end_time" in merged.columns:
    merged = merged.drop(columns=["end_time"])

out_path = os.path.join(BASE_DIR, "modeling_dataset_2017_2025.csv")
merged.to_csv(out_path, index=False)

# Audit printouts
print(f"\nSaved final modeling dataset: {out_path}")
print(f"Final shape: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
print("Rows per year:")
print(merged["year"].value_counts().sort_index().to_dict())
print(f"Share > 60 min (is_long_disruption=1): {merged['is_long_disruption'].mean():.4f}")
print(f"Unlocalized rows (nearest_stn missing): {merged['nearest_stn'].isna().sum()}")
if "dist_to_stn_km" in merged.columns and merged["dist_to_stn_km"].notna().any():
    print(
        f"dist_to_stn_km -> min: {merged['dist_to_stn_km'].min():.2f}, "
        f"median: {merged['dist_to_stn_km'].median():.2f}, "
        f"max: {merged['dist_to_stn_km'].max():.2f}, "
        f">50km count: {(merged['dist_to_stn_km'] > 50).sum()}"
    )