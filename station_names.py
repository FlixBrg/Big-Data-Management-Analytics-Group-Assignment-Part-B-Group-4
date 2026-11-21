import os
import glob
import io
import requests
import pandas as pd

# 1. Check all CSV files currently in your project and data/ folders
print("CSV files found:")
for f in glob.glob("**/*.csv", recursive=True):
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f" - {f} ({size_mb:.1f} MB)")

# 2. Fetch KNMI station metadata (station ID, name, latitude, longitude)
KNMI_HOURLY_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
resp = requests.post(
    KNMI_HOURLY_URL,
    data={"stns": "ALL", "vars": "T", "start": "2025010101", "end": "2025010101"},
    timeout=60,
)
resp.raise_for_status()

# Extract the station metadata lines (# STN  LON(east)  LAT(north)  ALT(m)  NAME)
meta_lines = [line.lstrip("# ").strip() for line in resp.text.splitlines() if line.startswith("# ") and any(c.isdigit() for c in line[:10])]
knmi_stations = pd.DataFrame(
    [line.split(maxsplit=4) for line in meta_lines],
    columns=["STN", "LON", "LAT", "ALT", "NAME"]
)
knmi_stations["STN"] = knmi_stations["STN"].astype(int)
knmi_stations["LON"] = knmi_stations["LON"].astype(float)
knmi_stations["LAT"] = knmi_stations["LAT"].astype(float)

os.makedirs("data", exist_ok=True)
knmi_stations.to_csv("data/knmi_stations.csv", index=False)
print(f"\nSaved {len(knmi_stations)} KNMI weather stations to data/knmi_stations.csv:")
print(knmi_stations.head())