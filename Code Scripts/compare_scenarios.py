import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, RocCurveDisplay
)

os.makedirs("outputs", exist_ok=True)

# 1. Load modeling dataset and station metadata
df = pd.read_csv("data/modeling_dataset_2017_2025.csv")
stations = pd.read_csv("data/stations-2023-09.csv")

# Drop the 7 unlocalized disruptions
df = df.dropna(subset=["nearest_stn"]).copy()

# Merge station coordinates, country, and station type onto primary_station
stations["code_clean"] = stations["code"].astype(str).str.strip().str.upper()
station_meta = stations[["code_clean", "geo_lat", "geo_lng", "country", "type"]].drop_duplicates("code_clean")
df = df.merge(station_meta, left_on="primary_station", right_on="code_clean", how="left")

# Fill missing binary weather flags with 0
for col in ["S", "Y", "O", "M"]:
    df[col] = df[col].fillna(0)

# Clean categoricals
df["country"] = df["country"].fillna("UNKNOWN")
df["type"] = df["type"].fillna("UNKNOWN")
df["cause_group"] = df["cause_group"].fillna("unknown")

# 2. Engineer domain features (zero leakage)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
df["is_gale"] = (df["FX"] >= 17.2).astype(int)       # Beaufort 8+ wind gusts
df["is_freezing"] = (df["T"] <= 0.0).astype(int)     # Freezing temperatures

# 3. Chronological Train / Test Split (Train: 2017–2023, Test: 2024–2025)
train_df = df[df["year"] <= 2023].copy()
test_df = df[df["year"] >= 2024].copy()

# Compute smoothed historical station long-disruption rate STRICTLY on train_df
global_mean = train_df["is_long_disruption"].mean()
stn_stats = train_df.groupby("primary_station")["is_long_disruption"].agg(["count", "mean"])
smoothing_m = 20  # shrinks rare stations smoothly toward the global mean
stn_stats["smoothed_rate"] = (
    (stn_stats["count"] * stn_stats["mean"] + smoothing_m * global_mean)
    / (stn_stats["count"] + smoothing_m)
)
rate_map = stn_stats["smoothed_rate"].to_dict()

train_df["station_hist_rate"] = train_df["primary_station"].map(rate_map).fillna(global_mean)
test_df["station_hist_rate"] = test_df["primary_station"].map(rate_map).fillna(global_mean)

# 4. Define Scenario A (Strict Onset) vs. Scenario B (+ cause_group)
num_cols = [
    "FH", "FX", "T", "RH", "DR", "VV", "dist_to_stn_km",
    "num_stations", "num_lines", "hour", "hour_sin", "hour_cos",
    "day_of_week", "month", "geo_lat", "geo_lng", "station_hist_rate",
    "S", "Y", "O", "M", "is_weekend", "is_rush_hour", "is_gale", "is_freezing"
]
cat_cols_A = ["country", "type"]
cat_cols_B = ["country", "type", "cause_group"]

target = "is_long_disruption"
y_train, y_test = train_df[target], test_df[target]

def run_scenario(cat_cols, scenario_label, ax):
    X_tr = train_df[num_cols + cat_cols]
    X_te = test_df[num_cols + cat_cols]

    preprocessor_scaled = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    preprocessor_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor_scaled),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor_tree),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=15, random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting (HistGB)": Pipeline([
            ("prep", preprocessor_tree),
            ("clf", HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=0.05, random_state=42))
        ])
    }

    rows = []
    best_probs = None
    for name, pipe in models.items():
        pipe.fit(X_tr, y_train)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
        })
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
        if name == "Gradient Boosting (HistGB)":
            best_probs = y_prob

    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_title(scenario_label)
    ax.legend(loc="lower right", fontsize=9)
    return pd.DataFrame(rows).set_index("Model").round(4), best_probs

# 5. Run both scenarios & plot side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

res_A, probs_A = run_scenario(cat_cols_A, "Scenario A: Weather + Time + Station (Zero Leakage)", axes[0])
res_B, probs_B = run_scenario(cat_cols_B, "Scenario B: + Broad Cause Group (Sensitivity Check)", axes[1])

plt.tight_layout()
plt.savefig("outputs/roc_comparison_A_vs_B.png", dpi=300)

print("\n=== Scenario A: Weather + Time + Station (Zero Leakage) ===")
print(res_A)
print("\n=== Scenario B: + cause_group (Sensitivity / Leakage Benchmark) ===")
print(res_B)

# 6. Threshold tuning preview on Scenario A's best model (HistGB)
print("\n=== Threshold Tuning on Scenario A (HistGB) ===")
for thr in [0.50, 0.45, 0.40, 0.35]:
    preds = (probs_A >= thr).astype(int)
    print(f"Threshold {thr:.2f} -> Precision: {precision_score(y_test, preds):.3f} | Recall: {recall_score(y_test, preds):.3f} | F1: {f1_score(y_test, preds):.3f}")

print("\nSaved side-by-side ROC comparison to outputs/roc_comparison_A_vs_B.png")