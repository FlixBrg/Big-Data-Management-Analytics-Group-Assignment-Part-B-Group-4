from pathlib import Path
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

# 1. Resolve dataset path automatically
start_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
candidates = [start_dir, start_dir.parent]

data_file = None
base_dir = start_dir
for cand in candidates:
    for rel in [
        Path("modeling_dataset_2017_2025.csv"),
        Path("Datasets/modeling_dataset_2017_2025.csv"),
        Path("data/modeling_dataset_2017_2025.csv"),
    ]:
        p = cand / rel
        if p.exists():
            data_file = p
            base_dir = cand
            break
    if data_file is not None:
        break

if data_file is None:
    raise FileNotFoundError("Could not find modeling_dataset_2017_2025.csv.")

out_dir = base_dir / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Load localized 2017-2025 dataset
df = pd.read_csv(data_file).dropna(subset=["nearest_stn"]).copy()
if df["RH"].dropna().max() < 10:
    df["RH"] = (df["RH"] * 10).round(1)

# 3. Engineer onset features
df["is_gale"] = (df["FX"] > 20).astype(int)
df["is_freezing"] = (df["T"] <= 0).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# 4. Chronological split: 2017-2023 Train, 2024-2025 Test
train = df[df["year"] <= 2023].copy().reset_index(drop=True)
test = df[df["year"] >= 2024].copy().reset_index(drop=True)
y_train = train["is_long_disruption"].values
y_test = test["is_long_disruption"].values

# 5. Leak-free 5-fold Out-of-Fold (OOF) target encoding for station_hist_rate
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

# 6. Preprocessing & Model Pipelines
num_cols = [
    "FH", "FX", "T", "RH", "DR", "VV",
    "is_gale", "is_freezing",
    "num_stations", "num_lines", "geo_lat", "geo_lng", "station_hist_rate",
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "is_weekend", "is_rush_hour"
]
flag_cols = ["S", "Y", "O", "M"]
cat_cols_A = ["country", "type"]
cat_cols_B = ["country", "type", "cause_group"]

def build_preprocessor(cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("flags", SimpleImputer(strategy="constant", fill_value=0), flag_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

models_spec = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1),
}

rows = []
pipes = {}
probs = {}
for scenario, c_cols in [("Scenario A (Onset Only)", cat_cols_A), ("Scenario B (+ cause_group)", cat_cols_B)]:
    X_tr = train[num_cols + flag_cols + c_cols]
    X_te = test[num_cols + flag_cols + c_cols]
    for name, clf in models_spec.items():
        pipe = Pipeline([("prep", build_preprocessor(c_cols)), ("clf", clone(clf))])
        pipe.fit(X_tr, y_train)
        prob = pipe.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.50).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        rows.append({
            "Scenario": scenario,
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, prob),
            "PR-AUC": average_precision_score(y_test, prob),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "TPR@20%FPR": np.interp(0.20, fpr, tpr),
        })
        pipes[(scenario, name)] = pipe
        probs[(scenario, name)] = prob

res_df = pd.DataFrame(rows)
print("\n=== Leak-Free 5-Fold OOF Model Comparison (2024-2025 Test Set) ===")
print(res_df.round(4).to_string(index=False))

# Threshold tuning on Scenario A RF
p_rf_A = probs[("Scenario A (Onset Only)", "Random Forest")]
print("\n=== Threshold Tuning: Scenario A - Random Forest (OOF) ===")
th_rows = []
for th in [0.40, 0.45, 0.50, 0.55, 0.60]:
    pred = (p_rf_A >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    th_rows.append({
        "Threshold": th,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "FP": fp,
        "FN": fn,
    })
print(pd.DataFrame(th_rows).round(4).to_string(index=False))

# Feature importances (Scenario A RF, Impurity)
rf_pipe_A = pipes[("Scenario A (Onset Only)", "Random Forest")]
rf_clf_A = rf_pipe_A.named_steps["clf"]
feat_names = [n.split("__", 1)[-1] for n in rf_pipe_A.named_steps["prep"].get_feature_names_out()]
imp_series = pd.Series(rf_clf_A.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\n=== Top 10 Feature Importances (Scenario A: RF, OOF Impurity %) ===")
print((imp_series.head(10) * 100).round(2).to_string())

# 7. Plot Single Combined Figure 2 (2A: ROC Curves | 2B: Top 10 Impurity Importances)
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
plt.rcParams.update({"font.size": 10})

roc_curves_to_plot = [
    (("Scenario A (Onset Only)", "Logistic Regression"), "#7f7f7f", "--", "Scen. A: Logistic Reg."),
    (("Scenario A (Onset Only)", "Gradient Boosting"), "#1f77b4", "-", "Scen. A: Gradient Boosting"),
    (("Scenario A (Onset Only)", "Random Forest"), "#d62728", "-", "Scen. A: Random Forest"),
    (("Scenario B (+ cause_group)", "Random Forest"), "#2ca02c", ":", "Scen. B: Random Forest (+cause)"),
]

for key, col, ls, label_prefix in roc_curves_to_plot:
    p = probs[key]
    fpr, tpr, _ = roc_curve(y_test, p)
    auc_v = roc_auc_score(y_test, p)
    axes[0].plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.1, label=f"{label_prefix} (AUC = {auc_v:.3f})")

axes[0].plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=1.1, label="Random Baseline (AUC = 0.500)")
axes[0].set_title("Figure 2A. Test-Set ROC Curves (2024–2025, 5-Fold OOF)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].grid(True, linestyle=":", alpha=0.5)
axes[0].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

top10 = (imp_series.head(10) * 100).iloc[::-1]
axes[1].barh(top10.index, top10.values, color="#2b5c7a")
axes[1].set_title("Figure 2B. Top 10 Feature Importances (Scenario A: RF)")
axes[1].set_xlabel("Training-Set Impurity Importance (%)")
for idx, val in enumerate(top10.values):
    axes[1].text(val + 0.3, idx, f"{val:.1f}%", va="center", fontsize=8.5)
axes[1].set_xlim(0, top10.max() + 4.5)

plt.tight_layout()
out_fig = out_dir / "figure2_section5.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved combined Figure 2 to: {out_fig}")