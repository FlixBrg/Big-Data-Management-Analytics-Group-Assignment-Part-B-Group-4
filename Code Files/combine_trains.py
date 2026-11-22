import os
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "raw").exists() else SCRIPT_DIR
RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "Datasets" if (BASE_DIR / "Datasets").exists() else BASE_DIR
YEARS = range(2017, 2026)

dfs = []
for y in YEARS:
    path = os.path.join(RAW_DIR, f"disruptions-{y}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df_y = pd.read_csv(path)
    print(f"{y}: {len(df_y):,} rows | cols: {list(df_y.columns)} | start_time: {df_y['start_time'].min()} -> {df_y['start_time'].max()}")
    dfs.append(df_y)

combined = pd.concat(dfs, ignore_index=True)

# Deduplicate by rdt_id just in case year boundaries overlap
before_dedup = len(combined)
if "rdt_id" in combined.columns:
    combined = combined.drop_duplicates(subset=["rdt_id"]).reset_index(drop=True)
print(f"\nTotal combined rows: {len(combined):,} (removed {before_dedup - len(combined)} duplicate rdt_id rows)")

DATA_DIR.mkdir(parents=True, exist_ok=True)
out_path = DATA_DIR / "disruptions_2017_2025.csv"
combined.to_csv(out_path, index=False)
print(f"Saved: {out_path}")