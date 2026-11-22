import os
import pandas as pd

BASE_DIR = r"C:\Users\felix\OneDrive - Erasmus University Rotterdam\New folder\New folder\Courses\Big Data Managment and Analytics\Assignments\Group_Assignment\New"
RAW_DIR = os.path.join(BASE_DIR, "raw")
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

out_path = os.path.join(BASE_DIR, "disruptions_2017_2025.csv")
combined.to_csv(out_path, index=False)
print(f"Saved: {out_path}")