import os
import glob
import pandas as pd

# Use a raw string (r"...") for Windows paths and include *.csv at the end
folder_path = r"C:\Users\felix\OneDrive - Erasmus University Rotterdam\New folder\New folder\Courses\Big Data Managment and Analytics\Assignments\Group_Assignment\Train_csvs\*.csv"

# 1. Find all disruption CSV files from 2017 to 2025
file_paths = sorted(glob.glob(folder_path))
file_paths = [f for f in file_paths if any(str(year) in os.path.basename(f) for year in range(2017, 2026))]
print(f"Found {len(file_paths)} files:")
for f in file_paths:
    print(" -", os.path.basename(f))

# 2. Read and stack them vertically
df_list = [pd.read_csv(file) for file in file_paths]
disruptions_combined = pd.concat(df_list, ignore_index=True)

# 3. Quick sanity checks and deduplication
print(f"\nCombined shape: {disruptions_combined.shape[0]:,} rows, {disruptions_combined.shape[1]} columns")
print("Columns:", list(disruptions_combined.columns))
disruptions_combined = disruptions_combined.drop_duplicates()

# 4. Save inside your Group_Assignment folder (creates a 'data' subfolder automatically)
output_dir = r"C:\Users\felix\OneDrive - Erasmus University Rotterdam\New folder\New folder\Courses\Big Data Managment and Analytics\Assignments\Group_Assignment\data"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "disruptions_2017_2025.csv")
disruptions_combined.to_csv(output_path, index=False)
print(f"\nSaved {len(disruptions_combined):,} rows to:\n{output_path}")