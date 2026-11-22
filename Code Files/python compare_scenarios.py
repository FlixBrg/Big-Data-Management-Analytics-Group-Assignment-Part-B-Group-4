import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "modeling_dataset_2017_2025.csv").exists() or (SCRIPT_DIR.parent / "Datasets").exists() or (SCRIPT_DIR.parent / "outputs").exists() else SCRIPT_DIR
DATA_DIR = BASE_DIR / "Datasets" if (BASE_DIR / "Datasets").exists() else BASE_DIR
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load localized 2017-2025 dataset
df = pd.read_csv(DATA_DIR / "modeling_dataset_2017_2025.csv").dropna(subset=["nearest_stn"]).copy()

# 2. Engineer onset features
df["is_gale"] = (df["FX"] > 20).astype(int)
df["is_freezing"] = (df["T"] <= 0).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# 3. Chronological split: 2017-2023 Train, 2024-2025 Test
train = df[df["year"] <= 2023].copy().reset_index(drop=True)
test = df[df["year"] >= 2024].copy().reset_index(drop=True)

print(f"Train (2017-2023): {len(train):,} rows | Share >60m: {train['is_long_disruption'].mean()*100:.2f}%")
print(f"Test  (2024-2025): {len(test):,} rows | Share >60m: {test['is_long_disruption'].mean()*100:.2f}%")

# Leak-free 5-fold out-of-fold (OOF) smoothed target encoding for station_hist_rate:
# each training row is encoded using only the other 4 folds, so it never sees its own label.
# The test set is encoded with rates from the full training set.
global_mean = train["is_long_disruption"].mean()
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

stn_stats = train.groupby("primary_station")["is_long_disruption"].agg(["count", "mean"])
stn_smooth = (stn_stats["count"] * stn_stats["mean"] + m_smooth * global_mean) / (stn_stats["count"] + m_smooth)
test["station_hist_rate"] = test["primary_station"].map(stn_smooth).fillna(global_mean)

num_cols = [
    "FH", "FX", "T", "RH", "DR", "VV",
    "is_gale", "is_freezing",
    "num_stations", "num_lines", "geo_lat", "geo_lng", "station_hist_rate",
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "is_weekend", "is_rush_hour"
]
flag_cols = ["S", "Y", "O", "M"]
cat_cols_A = ["country", "type"]
cat_cols_B = ["country", "type", "cause_group"]

y_train = train["is_long_disruption"].values
y_test = test["is_long_disruption"].values


def build_preprocessor(cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("flags", SimpleImputer(strategy="constant", fill_value=0), flag_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols)
        ]
    )


models_spec = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
}

results = []
fitted_pipes = {}
test_probs = {}

for scenario, c_cols in [("Scenario A (Onset Only)", cat_cols_A), ("Scenario B (+ cause_group)", cat_cols_B)]:
    X_tr = train[num_cols + flag_cols + c_cols]
    X_te = test[num_cols + flag_cols + c_cols]
    for name, clf in models_spec.items():
        pipe = Pipeline([
            ("prep", build_preprocessor(c_cols)),
            ("clf", clone(clf))
        ])
        pipe.fit(X_tr, y_train)
        prob = pipe.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.50).astype(int)

        results.append({
            "Scenario": scenario,
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, prob),
            "PR-AUC": average_precision_score(y_test, prob),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred)
        })
        fitted_pipes[(scenario, name)] = pipe
        test_probs[(scenario, name)] = prob

res_df = pd.DataFrame(results)
print("\n=== Model Comparison (2024-2025 Test Set) ===")
print(res_df.round(4).to_string(index=False))

# Threshold tuning on best Scenario A model (Random Forest)
best_A_prob = test_probs[("Scenario A (Onset Only)", "Random Forest")]
print("\n=== Threshold Tuning: Scenario A - Random Forest ===")
th_rows = []
for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    p = (best_A_prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    th_rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p),
        "Recall": recall_score(y_test, p),
        "F1": f1_score(y_test, p),
        "FP": fp,
        "FN": fn
    })
print(pd.DataFrame(th_rows).round(4).to_string(index=False))

# Top 10 Feature Importances (Scenario A - Random Forest)
rf_pipe = fitted_pipes[("Scenario A (Onset Only)", "Random Forest")]
rf_feat_names = [f.split("__", 1)[-1] for f in rf_pipe.named_steps["prep"].get_feature_names_out()]
rf_fi = pd.DataFrame({
    "feature": rf_feat_names,
    "importance": rf_pipe.named_steps["clf"].feature_importances_
}).sort_values("importance", ascending=False)
print("\n=== Top 10 Feature Importances (Scenario A - Random Forest) ===")
print(rf_fi.head(10).round(4).to_string(index=False))

# Generate 2-panel Model Performance Chart (Figure 2)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
plt.rcParams.update({"font.size": 10})

plot_specs = {
    ("Scenario A (Onset Only)", "Logistic Regression"): ("#7f7f7f", "--", "Scen. A: Logistic Reg."),
    ("Scenario A (Onset Only)", "Gradient Boosting"): ("#3b7ea1", "-", "Scen. A: Gradient Boosting"),
    ("Scenario A (Onset Only)", "Random Forest"): ("#d62728", "-", "Scen. A: Random Forest"),
    ("Scenario B (+ cause_group)", "Random Forest"): ("#2ca02c", ":", "Scen. B: Random Forest (+cause)")
}
for key, (col, ls, label) in plot_specs.items():
    fpr, tpr, _ = roc_curve(y_test, test_probs[key])
    auc_val = roc_auc_score(y_test, test_probs[key])
    axes[0].plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.2, label=f"{label} (AUC = {auc_val:.3f})")

axes[0].plot([0, 1], [0, 1], color="#cccccc", linestyle="--", linewidth=1)
axes[0].set_title("Figure 2A. Test-Set ROC Curves (2024–2025)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].legend(loc="lower right", frameon=False, fontsize=9)

top10 = rf_fi.head(10).iloc[::-1]
axes[1].barh(top10["feature"], top10["importance"] * 100, color="#1f5a7a", alpha=0.9)
axes[1].set_title("Figure 2B. Top 10 Feature Importances (Scenario A: RF)")
axes[1].set_xlabel("Relative Importance (%)")
for idx, val in enumerate(top10["importance"] * 100):
    axes[1].text(val + 0.4, idx, f"{val:.1f}%", va="center", fontsize=9)
axes[1].set_xlim(0, top10["importance"].max() * 100 + 5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "model_performance.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved model performance chart to: {out_path}")