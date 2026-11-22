from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = (
    SCRIPT_DIR.parent
    if (SCRIPT_DIR.parent / "modeling_dataset_2017_2025.csv").exists()
    or (SCRIPT_DIR.parent / "Datasets").exists()
    or (SCRIPT_DIR.parent / "outputs").exists()
    else SCRIPT_DIR
)
DATA_DIR = BASE_DIR / "Datasets" if (BASE_DIR / "Datasets").exists() else BASE_DIR
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load localized dataset
df = pd.read_csv(DATA_DIR / "modeling_dataset_2017_2025.csv")
df_loc = df.dropna(subset=["nearest_stn"]).copy()

print(f"Total rows: {len(df):,} | Localized rows: {len(df_loc):,} | Dropped unlocalized: {len(df) - len(df_loc)}")
print(f"Target balance (>60 min): {df_loc['is_long_disruption'].mean()*100:.2f}% long vs {(1-df_loc['is_long_disruption'].mean())*100:.2f}% short\n")

# Descriptive summary table
num_vars = ["duration_minutes", "FH", "FX", "T", "RH", "DR", "VV", "num_stations", "num_lines", "dist_to_stn_km"]
summary = df_loc[num_vars].agg(
    ["count", "mean", "std", "min", lambda s: s.quantile(0.25), "median", lambda s: s.quantile(0.75), "max"]
).T
summary.columns = ["Count", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max"]
print(summary.round(2).to_string())

# Generate 3-panel EDA figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
plt.rcParams.update({"font.size": 10})

# Figure 1A: Duration distribution (capped at 360+ min)
dur_capped = df_loc["duration_minutes"].clip(upper=360)
axes[0].hist(dur_capped, bins=36, color="#3b7ea1", edgecolor="white", alpha=0.88)
axes[0].axvline(60, color="#d62728", linestyle="--", linewidth=2, label="60-min cutoff")
axes[0].axvline(
    df_loc["duration_minutes"].median(),
    color="#111111",
    linestyle=":",
    linewidth=1.8,
    label=f"Median ({df_loc['duration_minutes'].median():.0f} min)",
)
axes[0].set_title("Figure 1A. Disruption Duration (Capped at 360+ min)")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Number of Disruptions")
axes[0].legend(frameon=False, fontsize=9)

# Figure 1B: Share >60 min by local start hour
hr_rate = df_loc.groupby("hour")["is_long_disruption"].mean() * 100
axes[1].plot(hr_rate.index, hr_rate.values, marker="o", linewidth=2, color="#1f5a7a")
axes[1].axhline(
    df_loc["is_long_disruption"].mean() * 100,
    color="#888888",
    linestyle="--",
    linewidth=1.2,
    label=f"Overall avg ({df_loc['is_long_disruption'].mean()*100:.1f}%)",
)
axes[1].set_title("Figure 1B. Share >60 Min by Start Hour")
axes[1].set_xlabel("Start Hour (Local Time)")
axes[1].set_ylabel("% Lasting >60 Minutes")
axes[1].set_xticks(range(0, 24, 3))
axes[1].set_ylim(25, 95)
axes[1].legend(frameon=False, fontsize=9)

# Figure 1C: Share >60 min by wind gust bracket (FX)
fx_bins = [-0.1, 5, 10, 15, 20, 100]
fx_labels = ["0–5", "5–10", "10–15", "15–20", ">20\n(Gale)"]
df_loc["fx_bin"] = pd.cut(df_loc["FX"], bins=fx_bins, labels=fx_labels)
fx_rate = df_loc.groupby("fx_bin", observed=False)["is_long_disruption"].mean() * 100
fx_n = df_loc.groupby("fx_bin", observed=False)["is_long_disruption"].count()

bars = axes[2].bar(
    fx_rate.index,
    fx_rate.values,
    color=["#6baed6", "#6baed6", "#4292c6", "#2171b5", "#d62728"],
    alpha=0.9,
)
axes[2].axhline(df_loc["is_long_disruption"].mean() * 100, color="#888888", linestyle="--", linewidth=1.2)
axes[2].set_title("Figure 1C. Share >60 Min by Wind Gust (FX)")
axes[2].set_xlabel("Max Hourly Wind Gust (m/s)")
axes[2].set_ylabel("% Lasting >60 Minutes")
axes[2].set_ylim(0, 75)
for bar, pct, n in zip(bars, fx_rate.values, fx_n.values):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        pct + 1.5,
        f"{pct:.1f}%\n(n={n:,})",
        ha="center",
        va="bottom",
        fontsize=8,
    )

fig.tight_layout()
out_path = OUT_DIR / "eda_charts.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved EDA chart to: {out_path}")