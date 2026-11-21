import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, RocCurveDisplay
)

os.makedirs("outputs", exist_ok=True)

# 1. Load merged modeling dataset
df = pd.read_csv("data/modeling_dataset_2017_2025.csv")
print(f"Loaded rows: {len(df):,}")

# Drop the 7 rows missing primary_station / nearest_stn
df = df.dropna(subset=["nearest_stn"]).copy()
print(f"After dropping unlocalized disruptions: {len(df):,}")

# 2. Define feature set (strictly non-leaky features known at disruption onset)
numeric_features = ["FH", "FX", "T", "RH", "DR", "VV", "dist_to_stn_km", "num_stations", "num_lines", "hour", "day_of_week", "month"]
binary_features = ["S", "Y", "O", "M", "is_weekend", "is_rush_hour"]

# Fill missing binary weather flags (S, Y, O, M) with 0 (stations without present-weather sensors)
for col in ["S", "Y", "O", "M"]:
    df[col] = df[col].fillna(0)

features = numeric_features + binary_features
target = "is_long_disruption"

# 3. Descriptive Statistics & Charts for Section 4
print("\n--- Summary Statistics (Selected Features) ---")
print(df[["duration_minutes", "is_long_disruption", "FH", "FX", "T", "RH", "num_stations", "num_lines"]].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot A: Distribution of duration_minutes (capped at 300 min for readability)
axes[0].hist(df["duration_minutes"].clip(upper=300), bins=40, color="steelblue", edgecolor="black", alpha=0.85)
axes[0].axvline(60, color="crimson", linestyle="--", linewidth=2, label="60-min threshold")
axes[0].set_title("Disruption Duration Distribution (Capped at 300m)")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Plot B: Long disruption rate by Wind Gust (FX) category
df["gust_cat"] = pd.cut(df["FX"], bins=[-1, 10, 15, 20, 100], labels=["Calm (<10 m/s)", "Moderate (10-15)", "Strong (15-20)", "Storm (>20 m/s)"])
gust_rates = df.groupby("gust_cat", observed=False)[target].agg(["mean", "count"])
axes[1].bar(gust_rates.index.astype(str), gust_rates["mean"] * 100, color="teal", edgecolor="black")
axes[1].set_title("Long Disruption (>60m) Rate by Wind Gust")
axes[1].set_ylabel("% Lasting > 60 Minutes")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_ylim(0, 70)

# Plot C: Long disruption rate by Hour of Day
hourly_rates = df.groupby("hour")[target].mean() * 100
axes[2].plot(hourly_rates.index, hourly_rates.values, marker="o", color="darkorange", linewidth=2)
axes[2].set_title("Long Disruption Rate by Start Hour")
axes[2].set_xlabel("Hour of Day (Local Time)")
axes[2].set_ylabel("% Lasting > 60 Minutes")
axes[2].set_xticks(range(0, 24, 3))
axes[2].set_ylim(20, 75)

plt.tight_layout()
plt.savefig("outputs/eda_charts.png", dpi=300)
print("\nSaved exploratory charts to outputs/eda_charts.png")

# 4. Chronological Train / Test Split (Train: 2017-2023, Test: 2024-2025)
train_df = df[df["year"] <= 2023].copy()
test_df = df[df["year"] >= 2024].copy()

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"\nTrain set (2017-2023): {len(X_train):,} rows | Positive rate: {y_train.mean():.3f}")
print(f"Test set  (2024-2025): {len(X_test):,} rows | Positive rate: {y_test.mean():.3f}")

# 5. Build and Compare Three Predictive Models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20, random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting (HistGB)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42))
    ])
}

results = []
fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

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
ax_roc.set_title("ROC Curves on 2024–2025 Test Set")
ax_roc.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=300)

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n--- Model Comparison on 2024–2025 Test Set ---")
print(results_df)

# 6. Feature Importances from Random Forest
rf_model = models["Random Forest"].named_steps["clf"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Top 10 Most Important Features (Random Forest) ---")
print(importances.head(10).round(4))