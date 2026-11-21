import numpy as np
import pandas as pd

# 1. Load datasets
disruptions = pd.read_csv("data/disruptions_2017_2025.csv")
stations = pd.read_csv("data/stations-2023-09.csv")
knmi_stations = pd.read_csv("data/knmi_stations.csv")
weather = pd.read_csv("data/knmi_hourly_weather_2017_2025.csv")

print("Weather columns:", list(weather.columns))
print("Stations columns:", list(stations.columns))

# 2. Match each railway station to its nearest KNMI weather station (Haversine distance)
# Standardize station code & coordinate column names in stations-2023-09.csv
stations["code_clean"] = stations["code"].astype(str).str.strip().str.upper()
stations_valid = stations.dropna(subset=["code_clean", "geo_lat", "geo_lng"]).copy()

def nearest_knmi_station(lat, lng, knmi_df):
    R = 6371.0  # Earth radius in km
    lat1, lon1 = np.radians(lat), np.radians(lng)
    lat2, lon2 = np.radians(knmi_df["LAT"].values), np.radians(knmi_df["LON"].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(a))
    min_idx = np.argmin(dist_km)
    return pd.Series([int(knmi_df.iloc[min_idx]["STN"]), dist_km[min_idx]])

# Filter knmi_stations to only those actually present in your downloaded weather CSV
available_stns = set(weather["STN"].unique())
knmi_available = knmi_stations[knmi_stations["STN"].isin(available_stns)].reset_index(drop=True)

stations_valid[["nearest_stn", "dist_to_stn_km"]] = stations_valid.apply(
    lambda row: nearest_knmi_station(row["geo_lat"], row["geo_lng"], knmi_available),
    axis=1
)
stations_valid["nearest_stn"] = stations_valid["nearest_stn"].astype(int)
station_to_knmi = stations_valid.set_index("code_clean")[["nearest_stn", "dist_to_stn_km"]].to_dict("index")

# 3. Extract primary station code & network scope features from disruptions
def parse_primary_station(codes_str):
    if pd.isna(codes_str) or not str(codes_str).strip():
        return None
    first_code = str(codes_str).split(",")[0].strip().upper()
    return first_code

disruptions["primary_station"] = disruptions["rdt_station_codes"].apply(parse_primary_station)
disruptions["num_stations"] = disruptions["rdt_station_codes"].fillna("").apply(
    lambda x: len([c for c in str(x).split(",") if c.strip()])
)
disruptions["num_lines"] = disruptions["rdt_lines"].fillna("").apply(
    lambda x: len([l for l in str(x).split(",") if l.strip()])
)

disruptions["nearest_stn"] = disruptions["primary_station"].map(
    lambda c: station_to_knmi[c]["nearest_stn"] if c in station_to_knmi else np.nan
)
disruptions["dist_to_stn_km"] = disruptions["primary_station"].map(
    lambda c: station_to_knmi[c]["dist_to_stn_km"] if c in station_to_knmi else np.nan
)

# 4. Align timestamps (KNMI UTC hour-ending -> Dutch local time Europe/Amsterdam)
# KNMI HH is 1..24 (hour ending at HH UTC). Subtract 1 hour to get start of hour in UTC, then convert to Amsterdam time.
weather["utc_dt"] = pd.to_datetime(weather["YYYYMMDD"].astype(str), format="%Y%m%d", utc=True) + pd.to_timedelta(
    weather["HH"] - 1, unit="h"
)
weather["local_hour"] = weather["utc_dt"].dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)

# Parse disruption start_time in Dutch local time and floor to the start hour
disruptions["start_dt"] = pd.to_datetime(disruptions["start_time"], errors="coerce")
disruptions["local_hour"] = disruptions["start_dt"].dt.floor("h")

# 5. Clean & rescale KNMI weather variables
# Divide tenth-unit columns by 10 if present in weather CSV
tenth_cols = [c for c in ["FH", "FX", "T", "RH", "DR", "FF", "TD", "SQ", "P"] if c in weather.columns]
if "RH" in weather.columns:
    weather["RH"] = weather["RH"].replace(-1, 0.25)  # -1 means <0.05 mm (0.5 tenths); 0.25 tenths = 0.025 mm
for col in tenth_cols:
    weather[col] = pd.to_numeric(weather[col], errors="coerce") / 10.0

# Keep relevant weather columns
keep_weather_cols = ["STN", "local_hour"] + [
    c for c in ["FH", "FX", "T", "RH", "DR", "VV", "S", "Y", "O", "M"] if c in weather.columns
]
weather_clean = weather[keep_weather_cols].drop_duplicates(subset=["STN", "local_hour"])

# 6. Merge disruptions with hourly weather
merged = pd.merge(
    disruptions,
    weather_clean,
    left_on=["nearest_stn", "local_hour"],
    right_on=["STN", "local_hour"],
    how="left"
)

# 7. Create target variable and temporal features
merged = merged.dropna(subset=["duration_minutes", "start_dt"]).copy()
merged["is_long_disruption"] = (merged["duration_minutes"] > 60).astype(int)
merged["year"] = merged["start_dt"].dt.year
merged["month"] = merged["start_dt"].dt.month
merged["day_of_week"] = merged["start_dt"].dt.dayofweek  # 0=Monday, 6=Sunday
merged["hour"] = merged["start_dt"].dt.hour
merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)
merged["is_rush_hour"] = (
    ((merged["day_of_week"] < 5) & (merged["hour"].between(7, 9) | merged["hour"].between(16, 18)))
).astype(int)

# Save final modeling dataset
output_file = "data/modeling_dataset_2017_2025.csv"
merged.to_csv(output_file, index=False)

print(f"\nSaved {output_file}: {merged.shape[0]:,} rows, {merged.shape[1]} columns")
print("Class balance (is_long_disruption):")
print(merged["is_long_disruption"].value_counts(normalize=True).round(4))
print("\nMissing values per column:")
print(merged.isna().sum()[merged.isna().sum() > 0])