import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "raw").exists() or (SCRIPT_DIR.parent / "raw_weather").exists() else SCRIPT_DIR
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DISRUPTIONS_PAGE = "https://www.rijdendetreinen.nl/en/open-data/disruptions"
STATIONS_PAGE = "https://www.rijdendetreinen.nl/en/open-data/stations"
YEARS = range(2017, 2026)  # 2017 up to and including 2025

headers = {"User-Agent": "Mozilla/5.0 (Academic Research Project BM04BIM)"}


def download_file(url: str, dest_path: str) -> bool:
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code == 200 and len(resp.content) > 500:
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        print(f"Downloaded: {os.path.basename(dest_path)} ({len(resp.content):,} bytes)")
        return True
    print(f"Failed ({resp.status_code}): {url}")
    return False


# 1. Scrape disruption CSV links from Rijden de Treinen
page = requests.get(DISRUPTIONS_PAGE, headers=headers, timeout=30)
page.raise_for_status()
soup = BeautifulSoup(page.text, "html.parser")

found_links = {}
for a in soup.find_all("a", href=True):
    href = a["href"]
    for y in YEARS:
        if f"disruptions-{y}.csv" in href:
            full_url = href if href.startswith("http") else f"https://www.rijdendetreinen.nl{href}"
            found_links[y] = full_url

for y in YEARS:
    url = found_links.get(y, f"https://opendata.rijdendetreinen.nl/public/disruptions/disruptions-{y}.csv")
    dest = os.path.join(RAW_DIR, f"disruptions-{y}.csv")
    download_file(url, dest)

# 2. Download stations-2023-09.csv (saved directly in BASE_DIR)
stations_dest = os.path.join(BASE_DIR, "stations-2023-09.csv")
stations_url = "https://opendata.rijdendetreinen.nl/public/stations/stations-2023-09.csv"
if not download_file(stations_url, stations_dest):
    spage = requests.get(STATIONS_PAGE, headers=headers, timeout=30)
    ssoup = BeautifulSoup(spage.text, "html.parser")
    for a in ssoup.find_all("a", href=True):
        if "stations-2023-09.csv" in a["href"]:
            href = a["href"]
            full_url = href if href.startswith("http") else f"https://www.rijdendetreinen.nl{href}"
            download_file(full_url, stations_dest)
            break