import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
train = df[df["year"] <= 2023].copy().reset_index(drop=True)
test = df[df["year"] >= 2024].copy().reset_index(drop=True)
y_train = train["is_long_disruption"].values
y_test = test["is_long_disruption"].values

# 4. Leak-free 5-fold Out-of-Fold (OOF) target encoding for station_hist_rate
global_mean = y_train.mean()
m_smooth = 20

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rate = np.zeros(len(train))
for tr_idx, val_idx in kf.split(train):
    tr_fold = train.iloc[tr_idx]
    g_m = tr_fold["is_long_disruption"].mean()
    s_st = tr_fold.groupby("primary_station")["is_long_disruption"].agg(["count", "mean"])
    s_sm = (s_st["count"] * s_st["mean"] + m_smooth * g_m) / (s_st["count"] + m_smooth)
    oof_rate[val_idx] = train.iloc[val_idx]["primary_station"].map(s_sm).fillna(g_m).values

train["station_hist_rate"] = oof_rate

stn_full = train.groupby("primary_station")["is_long_disruption"].agg(["count", "mean"])
stn_sm_full = (stn_full["count"] * stn_full["mean"] + m_smooth * global_mean) / (stn_full["count"] + m_smooth)
test["station_hist_rate"] = test["primary_station"].map(stn_sm_full).fillna(global_mean)

# 5. Feature definitions
num_cols = [
    "FH", "FX", "T", "RH", "DR", "VV",
    "is_gale", "is_freezing",
    "num_stations", "num_lines", "geo_lat", "geo_lng", "station_hist_rate",
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "is_weekend", "is_rush_hour",
]
flag_cols = ["S", "Y", "O", "M"]
cat_cols_A = ["country", "type"]
cat_cols_B = ["country", "type", "cause_group"]


def build_preprocessor(cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("flags", SimpleImputer(strategy="constant", fill_value=0), flag_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ]
    )


models_spec = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=30, l2_regularization=1.0, random_state=42
    ),
}

probs = {}
for scen_key, c_cols in [("A", cat_cols_A), ("B", cat_cols_B)]:
    for name, clf in models_spec.items():
        pipe = Pipeline([("prep", build_preprocessor(c_cols)), ("clf", clone(clf))])
        pipe.fit(train[num_cols + flag_cols + c_cols], y_train)
        probs[(scen_key, name)] = pipe.predict_proba(test[num_cols + flag_cols + c_cols])[:, 1]

# 6. Two-panel ROC comparison matching reference layout
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
plt.rcParams.update({"font.size": 10})

panels = [
    ("A", "Scenario A: Weather + Time + Station (Zero Leakage)"),
    ("B", "Scenario B: + Broad Cause Group (Sensitivity Check)"),
]

model_styles = [
    ("Logistic Regression", "#1f77b4", "-"),
    ("Random Forest", "#ff7f0e", "-"),
    ("Gradient Boosting (HistGB)", "#2ca02c", "-"),
]

for ax, (scen_key, title) in zip(axes, panels):
    for m_name, col, ls in model_styles:
        p = probs[(scen_key, m_name)]
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=1.8, label=f"{m_name} (AUC = {auc_val:.2f})")
    ax.plot([0, 1], [0, 1], color="#111111", linestyle="--", linewidth=1.3, label="Random Baseline")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("False Positive Rate (Positive label: 1)")
    ax.set_ylabel("True Positive Rate (Positive label: 1)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "roc_comparison_A_vs_B.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved ROC comparison plot to: {out_path}")