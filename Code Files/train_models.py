from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, RocCurveDisplay
)

# Resolve paths cleanly whether run from script folder, repo root, or interactive session
start_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
candidates = [start_dir, start_dir.parent]

data_file = None
base_dir = start_dir
for cand in candidates:
    for rel in [
        Path("data/modeling_dataset_2017_2025.csv"),
        Path("Datasets/modeling_dataset_2017_2025.csv"),
        Path("modeling_dataset_2017_2025.csv"),
    ]:
        p = cand / rel
        if p.exists():
            data_file = p
            base_dir = cand
            break
    if data_file is not None:
        break

if data_file is None:
    raise FileNotFoundError("Could not find modeling_dataset_2017_2025.csv in expected locations.")

out_dir = base_dir / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load merged modeling dataset
df = pd.read_csv(data_file)
print(f"Loaded rows: {len(df):,} from {data_file}")

# Drop rows missing nearest_stn / station coordinates
df_loc = df.dropna(subset=["nearest_stn"]).copy()
print(f"After dropping unlocalized disruptions: {len(df_loc):,} (dropped {len(df) - len(df_loc)})")

# Ensure RH is in mm (1 decimal) in case an earlier double-scaling happened
if "RH" in df_loc.columns and df_loc["RH"].dropna().max() < 10:
    df_loc["RH"] = (df_loc["RH"] * 10).round(1)

# Fill missing binary weather flags (S, Y, O, M) with 0 (stations without present-weather sensors)
flag_cols = [c for c in ["S", "Y", "O", "M"] if c in df_loc.columns]
for col in flag_cols:
    df_loc[col] = df_loc[col].fillna(0)

# Engineer onset features
df_loc["is_gale"] = (df_loc["FX"] > 20).astype(int)
df_loc["is_freezing"] = (df_loc["T"] <= 0).astype(int)
df_loc["hour_sin"] = np.sin(2 * np.pi * df_loc["hour"] / 24.0)
df_loc["hour_cos"] = np.cos(2 * np.pi * df_loc["hour"] / 24.0)

target = "is_long_disruption"

# 2. Descriptive Statistics & Charts for Section 4
print("\n--- Summary Statistics (Selected Features) ---")
desc_cols = [
    c for c in [
        "duration_minutes", "is_long_disruption", "FH", "FX", "T",
        "RH", "DR", "VV", "num_stations", "num_lines", "dist_to_stn_km"
    ]
    if c in df_loc.columns
]
print(df_loc[desc_cols].describe().round(2).to_string())

fig_eda, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot A: Distribution of duration_minutes (capped at 360 min)
axes[0].hist(df_loc["duration_minutes"].clip(upper=360), bins=36, color="steelblue", edgecolor="black", alpha=0.85)
axes[0].axvline(60, color="crimson", linestyle="--", linewidth=2, label="60-min threshold")
axes[0].set_title("Disruption Duration Distribution (Capped at 360m)")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Plot B: Long disruption rate by Wind Gust (FX) category
df_loc["gust_cat"] = pd.cut(
    df_loc["FX"],
    bins=[-0.1, 5, 10, 15, 20, 100],
    labels=["0-5", "5-10", "10-15", "15-20", ">20 (Gale)"]
)
gust_rates = df_loc.groupby("gust_cat", observed=False)[target].agg(["mean", "count"])
axes[1].bar(gust_rates.index.astype(str), gust_rates["mean"] * 100, color="teal", edgecolor="black")
axes[1].set_title("Long Disruption (>60m) Rate by Wind Gust")
axes[1].set_xlabel("Max Hourly Wind Gust (m/s)")
axes[1].set_ylabel("% Lasting > 60 Minutes")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_ylim(0, 75)

# Plot C: Long disruption rate by Hour of Day
hourly_rates = df_loc.groupby("hour")[target].mean() * 100
axes[2].plot(hourly_rates.index, hourly_rates.values, marker="o", color="darkorange", linewidth=2)
axes[2].set_title("Long Disruption Rate by Start Hour")
axes[2].set_xlabel("Hour of Day (Local Time)")
axes[2].set_ylabel("% Lasting > 60 Minutes")
axes[2].set_xticks(range(0, 24, 3))
axes[2].set_ylim(20, 95)

fig_eda.tight_layout()
eda_path = out_dir / "eda_charts.png"
fig_eda.savefig(eda_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved exploratory charts to {eda_path}")

# 3. Chronological Train / Test Split (Train: 2017-2023, Test: 2024-2025)
train_df = df_loc[df_loc["year"] <= 2023].copy().reset_index(drop=True)
test_df = df_loc[df_loc["year"] >= 2024].copy().reset_index(drop=True)

y_train = train_df[target].values
y_test = test_df[target].values

# 4. Leak-free 5-fold Out-of-Fold (OOF) smoothed target encoding for station_hist_rate
global_mean = y_train.mean()
m_smooth = 20

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rate = np.zeros(len(train_df))
for tr_idx, val_idx in kf.split(train_df):
    tr_fold = train_df.iloc[tr_idx]
    g_m = tr_fold[target].mean()
    s_st = tr_fold.groupby("primary_station")[target].agg(["count", "mean"])
    s_sm = (s_st["count"] * s_st["mean"] + m_smooth * g_m) / (s_st["count"] + m_smooth)
    oof_rate[val_idx] = train_df.iloc[val_idx]["primary_station"].map(s_sm).fillna(g_m).values

train_df["station_hist_rate"] = oof_rate

stn_full = train_df.groupby("primary_station")[target].agg(["count", "mean"])
stn_sm_full = (stn_full["count"] * stn_full["mean"] + m_smooth * global_mean) / (stn_full["count"] + m_smooth)
test_df["station_hist_rate"] = test_df["primary_station"].map(stn_sm_full).fillna(global_mean)

# Strictly non-leaky onset features (Scenario A)
numeric_features = [
    c for c in [
        "FH", "FX", "T", "RH", "DR", "VV",
        "is_gale", "is_freezing",
        "num_stations", "num_lines", "geo_lat", "geo_lng", "station_hist_rate",
        "hour", "hour_sin", "hour_cos", "day_of_week", "month", "is_weekend", "is_rush_hour"
    ]
    if c in train_df.columns
]
cat_features = [c for c in ["country", "type"] if c in train_df.columns]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("flags", SimpleImputer(strategy="constant", fill_value=0), flag_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_features),
])

print(f"\nTrain set (2017-2023): {len(train_df):,} rows | Positive rate: {y_train.mean():.3f}")
print(f"Test set  (2024-2025): {len(test_df):,} rows | Positive rate: {y_test.mean():.3f}")

# 5. Build and Compare Three Predictive Models (Scenario A, 5-fold OOF)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=42
    ),
}

feature_cols = numeric_features + flag_cols + cat_features
X_train, X_test = train_df[feature_cols], test_df[feature_cols]

results = []
fitted_pipes = {}
fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))

for name, clf in models.items():
    pipe = Pipeline([
        ("prep", clone(preprocessor)),
        ("clf", clone(clf))
    ])
    pipe.fit(X_train, y_train)
    fitted_pipes[name] = pipe

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
    })

    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_roc)

ax_roc.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax_roc.set_title("ROC Curves on 2024–2025 Test Set (Scenario A, OOF)")
ax_roc.legend()
fig_roc.tight_layout()
roc_path = out_dir / "roc_curves.png"
fig_roc.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved ROC curves to {roc_path}")

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n--- Model Comparison on 2024–2025 Test Set ---")
print(results_df.to_string())

# 6. Feature Importances from Random Forest
rf_pipe = fitted_pipes["Random Forest"]
rf_model = rf_pipe.named_steps["clf"]
encoded_names = rf_pipe.named_steps["prep"].get_feature_names_out()
clean_names = [n.split("__", 1)[-1] for n in encoded_names]

importances = pd.Series(rf_model.feature_importances_, index=clean_names).sort_values(ascending=False)
print("\n--- Top 10 Most Important Features (Random Forest, Impurity) ---")
print(importances.head(10).round(4).to_string())