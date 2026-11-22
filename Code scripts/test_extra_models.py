import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
)

BASE_DIR = r"C:\Users\felix\OneDrive - Erasmus University Rotterdam\New folder\New folder\Courses\Big Data Managment and Analytics\Assignments\Group_Assignment\New"
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load localized 2017-2025 dataset
df = pd.read_csv(os.path.join(BASE_DIR, "modeling_dataset_2017_2025.csv")).dropna(subset=["nearest_stn"]).copy()

# 2. Engineer onset features
df["is_gale"] = (df["FX"] > 20).astype(int)
df["is_freezing"] = (df["T"] <= 0).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# 3. Chronological split: 2017-2023 Train, 2024-2025 Test
train = df[df["year"] <= 2023].copy()
test = df[df["year"] >= 2024].copy()

# Smoothed historical station severity rate (fit strictly on train)
global_mean = train["is_long_disruption"].mean()
m_smooth = 20
stn_stats = train.groupby("primary_station")["is_long_disruption"].agg(["count", "mean"])
stn_smooth = (stn_stats["count"] * stn_stats["mean"] + m_smooth * global_mean) / (stn_stats["count"] + m_smooth)

train["station_hist_rate"] = train["primary_station"].map(stn_smooth).fillna(global_mean)
test["station_hist_rate"] = test["primary_station"].map(stn_smooth).fillna(global_mean)

num_cols = [
    "FH", "FX", "T", "RH", "DR", "VV",
    "is_gale", "is_freezing",
    "num_stations", "num_lines", "geo_lat", "geo_lng", "station_hist_rate",
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "is_weekend", "is_rush_hour",
]
flag_cols = ["S", "Y", "O", "M"]
cat_cols_A = ["country", "type"]
cat_cols_B = ["country", "type", "cause_group"]

y_train = train["is_long_disruption"].values
y_test = test["is_long_disruption"].values


def build_preprocessor(cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("flags", SimpleImputer(strategy="constant", fill_value=0), flag_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ]
    )


models_spec = {
    "Random Forest (Current Best)": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=8, random_state=42, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=30,
        l2_regularization=1.0, random_state=42,
    ),
}

rows = []
probs = {}  # (scenario, model) -> predicted probabilities on test
for scenario, c_cols in [("Scenario A (Onset Only)", cat_cols_A), ("Scenario B (+ cause_group)", cat_cols_B)]:
    X_tr = train[num_cols + flag_cols + c_cols]
    X_te = test[num_cols + flag_cols + c_cols]
    for name, clf in models_spec.items():        
        pipe = Pipeline([
            ("prep", build_preprocessor(c_cols)),
            ("clf", clone(clf)),
        ])
        pipe.fit(X_tr, y_train)
        prob = pipe.predict_proba(X_te)[:, 1]
        probs[(scenario, name)] = prob
        pred = (prob >= 0.50).astype(int)
        rows.append({
            "Scenario": scenario,
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, prob),
            "PR-AUC": average_precision_score(y_test, prob),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })

res = pd.DataFrame(rows)
print("\n=== Extra Models Benchmark (2024-2025 Test Set, n=12,627) ===")
print(res.round(4).to_string(index=False))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

scenarios = ["Scenario A (Onset Only)", "Scenario B (+ cause_group)"]
model_names = list(models_spec.keys())
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, scenario in zip(axes, scenarios):
    for name, color in zip(model_names, colors):
        prob = probs[(scenario, name)]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1, label="Random (AUC = 0.500)")
    ax.set_title(scenario, fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

axes[0].set_ylabel("True Positive Rate")
fig.suptitle(
    f"ROC Curves: Long Disruption Prediction (2024-2025 Test Set, n={len(test):,})",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "roc_curves_extra_models.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved ROC plot to: {out_path}")
plt.show()