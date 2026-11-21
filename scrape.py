# Download hourly KNMI weather data in chunks (one station, one year per request)
# to stay under KNMI's per-request row limit, then combine into one CSV.

import io
import time
import requests
import pandas as pd

KNMI_HOURLY_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"

# Main rail-region stations (verify numbers on the KNMI station list)
STATION_IDS = [240, 260, 344, 215, 350, 370, 380, 275, 278, 290, 280, 235, 310]

# Wind, gusts, temperature, precipitation, precipitation duration,
# visibility, snow, ice, thunder, fog
WEATHER_VARIABLES = ["FH", "FX", "T", "RH", "DR", "VV", "S", "Y", "O", "M"]

YEARS = range(2017, 2026)  # 2017 up to and including 2025


def download_station_year(station_id, year):
    """Request one station's hourly data for one year and return a DataFrame."""
    request_params = {
        "stns": str(station_id),
        "vars": ":".join(WEATHER_VARIABLES),
        "start": f"{year}010101",  # YYYYMMDDHH
        "end": f"{year}123124",
    }
    response = requests.post(KNMI_HOURLY_URL, data=request_params, timeout=120)
    response.raise_for_status()

    lines = response.text.splitlines()
    # The column names are on the last comment line that starts with '# STN'
    header_line = [line for line in lines if line.startswith("# STN")][-1]
    column_names = [name.strip() for name in header_line.lstrip("# ").split(",")]
    data_lines = [line for line in lines if line.strip() and not line.startswith("#")]

    return pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        names=column_names,
        skipinitialspace=True,
    )


all_chunks = []
for station_id in STATION_IDS:
    for year in YEARS:
        print(f"Downloading station {station_id}, year {year}...")
        all_chunks.append(download_station_year(station_id, year))
        time.sleep(1)  # be polite to KNMI's server

knmi_hourly_weather = pd.concat(all_chunks, ignore_index=True)
knmi_hourly_weather.to_csv("knmi_hourly_weather_2017_2025.csv", index=False)
print(f"Saved {len(knmi_hourly_weather):,} rows")