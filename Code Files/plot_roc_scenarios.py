import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# 5. Define feature sets
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
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1),
}

probs = {}
for scenario, c_cols in [("Scenario A", cat_cols_A), ("Scenario B", cat_cols_B)]:
    for name, clf in models_spec.items():
        pipe = Pipeline([("prep", build_preprocessor(c_cols)), ("clf", clone(clf))])
        pipe.fit(train[num_cols + flag_cols + c_cols], y_train)
        probs[(scenario, name)] = pipe.predict_proba(test[num_cols + flag_cols + c_cols])[:, 1]

# 6. Plot clean two-scenario ROC curve (best model per scenario: Random Forest)
fig, ax = plt.subplots(figsize=(6.8, 5.2))
plt.rcParams.update({"font.size": 10})

pA = probs[("Scenario A", "Random Forest")]
pB = probs[("Scenario B", "Random Forest")]
fprA, tprA, _ = roc_curve(y_test, pA)
fprB, tprB, _ = roc_curve(y_test, pB)

aucA, prA = roc_auc_score(y_test, pA), average_precision_score(y_test, pA)
aucB, prB = roc_auc_score(y_test, pB), average_precision_score(y_test, pB)
tprA_20 = np.interp(0.20, fprA, tprA)
tprB_20 = np.interp(0.20, fprB, tprB)

ax.plot(fprA, tprA, color="#d62728", linewidth=2.4,
        label=f"Scenario A: Onset Only RF (AUC = {aucA:.3f}, PR-AUC = {prA:.3f})")
ax.plot(fprB, tprB, color="#1f77b4", linewidth=2.4,
        label=f"Scenario B: + cause_group RF (AUC = {aucB:.3f}, PR-AUC = {prB:.3f})")
ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", linewidth=1.2, label="Random Baseline (AUC = 0.500)")

ax.axvline(0.20, color="#666666", linestyle=":", linewidth=1.2)
ax.scatter([0.20, 0.20], [tprA_20, tprB_20], color=["#d62728", "#1f77b4"], zorder=5, s=45)
ax.annotate(f"TPR = {tprB_20*100:.1f}% @ 20% FPR", xy=(0.20, tprB_20), xytext=(0.30, 0.65),
            fontsize=8.5, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.1))
ax.annotate(f"TPR = {tprA_20*100:.1f}% @ 20% FPR", xy=(0.20, tprA_20), xytext=(0.30, 0.30),
            fontsize=8.5, color="#d62728", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))

ax.set_title(f"Test-Set ROC Curves (2024–2025, Base Rate = {y_test.mean()*100:.1f}%)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "roc_both_scenarios.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved ROC plot to: {out_path}")