import calendar
import io
import os
import time
import pandas as pd
import requests

BASE_DIR = r"C:\Users\felix\OneDrive - Erasmus University Rotterdam\New folder\New folder\Courses\Big Data Managment and Analytics\Assignments\Group_Assignment\New"
RAW_WEATHER_DIR = os.path.join(BASE_DIR, "raw_weather")
os.makedirs(RAW_WEATHER_DIR, exist_ok=True)

KNMI_HOURLY_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
YEARS = range(2017, 2026)  # 2017 up to and including 2025
VARS = "WIND:TEMP:PRCP:VICL:WEATHER"

headers = {"User-Agent": "Mozilla/5.0 (Academic Research Project BM04BIM)"}


def fetch_knmi_with_retry(payload: dict, label: str, max_retries: int = 3) -> str:
    """POST to KNMI with retries on transient 5xx/network errors."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(KNMI_HOURLY_URL, data=payload, headers=headers, timeout=120)
            if resp.status_code >= 500:
                print(f"  [{label}] HTTP {resp.status_code} on attempt {attempt}/{max_retries}...")
                if attempt < max_retries:
                    time.sleep(3 * attempt)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt < max_retries:
                print(f"  [{label}] Request error ({exc}) on attempt {attempt}/{max_retries}, retrying...")
                time.sleep(3 * attempt)
            else:
                raise
    raise RuntimeError(f"[{label}] Failed after {max_retries} attempts.")


def parse_knmi_hourly_response(raw_text: str, label: str) -> tuple[pd.DataFrame, list[str]]:
    """Extract KNMI header comments and parse the tabular hourly block cleanly."""
    stripped_start = raw_text.lstrip()
    if stripped_start.startswith("<!DOCTYPE html") or stripped_start.startswith("<html"):
        if "te veel resultaten" in raw_text or "Query Error" in raw_text:
            raise RuntimeError(f"[{label}] KNMI returned 'too many results' HTML error.")
        raise RuntimeError(f"[{label}] KNMI returned an HTML page instead of data.")

    lines = raw_text.splitlines()
    comment_lines = [line.lstrip("#").strip() for line in lines if line.lstrip().startswith("#")]

    header_idx = None
    for i, line in enumerate(lines):
        s = line.lstrip("#").strip()
        if s.startswith("STN,") or s.startswith("STN ,"):
            header_idx = i
            break

    if header_idx is None:
        preview = "\n".join(lines[:12])
        raise ValueError(f"[{label}] Could not find 'STN,' header. Preview:\n{preview}")

    header_line = lines[header_idx].lstrip("#").strip()
    expected_commas = header_line.count(",")

    clean_lines = [header_line]
    skipped = 0
    for line in lines[header_idx + 1 :]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.count(",") == expected_commas:
            clean_lines.append(s)
        else:
            skipped += 1

    if skipped > 0:
        print(f"  [{label}] Skipped {skipped} non-matching line(s).")

    df = pd.read_csv(io.StringIO("\n".join(clean_lines)), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df, comment_lines


def fetch_month_in_half_chunks(year: int, month: int) -> tuple[pd.DataFrame, list[str]]:
    """Fetch a month as two half-month windows (days 1-15 and 16-end) to reduce load."""
    last_day = calendar.monthrange(year, month)[1]
    windows = [
        (f"{year}{month:02d}0101", f"{year}{month:02d}1524", f"{year}-{month:02d}a"),
        (f"{year}{month:02d}1601", f"{year}{month:02d}{last_day:02d}24", f"{year}-{month:02d}b"),
    ]
    parts = []
    comments = []
    for start_str, end_str, w_label in windows:
        raw_txt = fetch_knmi_with_retry({"start": start_str, "end": end_str, "vars": VARS}, w_label)
        df_w, c_w = parse_knmi_hourly_response(raw_txt, w_label)
        comments.extend(c_w)
        if not df_w.empty:
            parts.append(df_w)
        time.sleep(0.5)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), comments


all_frames = []
all_comments = []

for year in YEARS:
    year_csv = os.path.join(RAW_WEATHER_DIR, f"knmi_hourly_{year}.csv")
    if os.path.exists(year_csv):
        df_year = pd.read_csv(year_csv)
        print(f"{year}: Loaded cached {year_csv} ({len(df_year):,} rows)")
        all_frames.append(df_year)
        continue

    year_frames = []
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        label = f"{year}-{month:02d}"
        payload = {
            "start": f"{year}{month:02d}0101",
            "end": f"{year}{month:02d}{last_day:02d}24",
            "vars": VARS,
        }
        try:
            raw_txt = fetch_knmi_with_retry(payload, label)
            df_m, comments = parse_knmi_hourly_response(raw_txt, label)
        except requests.HTTPError:
            print(f"  [{label}] Monthly request hit 500; trying two half-month chunks...")
            df_m, comments = fetch_month_in_half_chunks(year, month)

        all_comments.extend(comments)
        if not df_m.empty:
            year_frames.append(df_m)
        time.sleep(0.5)

    if year_frames:
        df_year = pd.concat(year_frames, ignore_index=True)
        df_year.to_csv(year_csv, index=False)
        n_stn = df_year["STN"].nunique() if "STN" in df_year.columns else "N/A"
        print(f"{year}: {len(df_year):,} hourly rows | stations: {n_stn} | cols: {list(df_year.columns)}")
        all_frames.append(df_year)
    else:
        print(f"Warning: {year} produced no rows.")

if not all_frames:
    raise RuntimeError("No KNMI hourly weather data was retrieved across 2017-2025.")

weather_all = pd.concat(all_frames, ignore_index=True)
weather_out = os.path.join(BASE_DIR, "knmi_hourly_weather_2017_2025.csv")
weather_all.to_csv(weather_out, index=False)
print(f"\nSaved combined KNMI hourly weather: {weather_out} ({len(weather_all):,} total rows)")

station_lines = [c for c in all_comments if ":" in c and any(ch.isdigit() for ch in c[:12])]
if station_lines:
    stations_out = os.path.join(BASE_DIR, "knmi_stations.csv")
    pd.DataFrame({"raw_station_info": sorted(set(station_lines))}).to_csv(stations_out, index=False)
    print(f"Saved station metadata notes to: {stations_out}")