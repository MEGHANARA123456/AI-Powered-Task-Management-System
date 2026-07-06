"""
Week 3: Priority prediction (Random Forest, GridSearchCV-tuned) +
heuristic workload balancing logic.

Note on XGBoost: this sandbox has no internet access to install it, so
Gradient Boosting (sklearn's built-in equivalent) is used as the second
model here. If xgboost is available in your environment, swap in
`from xgboost import XGBClassifier` -- everything else (feature prep,
GridSearchCV, evaluation) stays the same.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

CLEAN_PATH = "./data/tasks_clean.csv"
OUT_DIR = "./outputs"
MODEL_DIR = "./models"

df = pd.read_csv(CLEAN_PATH)

# --- feature engineering for priority prediction ---
cat_encoder = LabelEncoder()
df["category_enc"] = cat_encoder.fit_transform(df["category"])

feature_cols = ["category_enc", "deadline_days", "workload_hours", "desc_length"]
X = df[feature_cols]
y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Random Forest + GridSearchCV hyperparameter tuning ---
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 8, 12],
    "min_samples_leaf": [1, 3],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=4, scoring="f1_weighted", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("Best Random Forest params:", rf_grid.best_params_)

# --- Gradient Boosting (XGBoost stand-in) ---
gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

results = {}
trained = {"Random Forest (tuned)": best_rf, "Gradient Boosting": gb}
preds_map = {}
for name, model in trained.items():
    preds = model.predict(X_test)
    preds_map[name] = preds
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
    results[name] = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, preds))

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = trained[best_name]
print(f"\nBest priority model: {best_name}")

# --- confusion matrix ---
labels = ["Low", "Medium", "High"]
cm = confusion_matrix(y_test, preds_map[best_name], labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {best_name} (Priority Prediction)")
plt.ylabel("True Priority")
plt.xlabel("Predicted Priority")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_priority.png", dpi=140)
plt.close()

# --- feature importance (Random Forest) ---
importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="teal")
plt.title("Feature Importance - Priority Prediction (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance_priority.png", dpi=140)
plt.close()

joblib.dump(best_model, f"{MODEL_DIR}/priority_model_{best_name.split()[0]}.joblib")
joblib.dump(cat_encoder, f"{MODEL_DIR}/category_label_encoder.joblib")
with open(f"{OUT_DIR}/priority_results.json", "w") as f:
    json.dump({"results": results, "best_model": best_name,
                "best_rf_params": rf_grid.best_params_}, f, indent=2)

# =====================================================================
# Week 3 (continued): Heuristic workload-balancing / auto-assignment
# =====================================================================
# Logic: for each new/unassigned task, assign it to the assignee with the
# lowest *current total predicted workload*, among assignees, breaking ties
# by picking whoever has fewer High-priority tasks already on their plate.
# This is a common, explainable heuristic used before investing in a full
# optimization solver (e.g. linear programming assignment).

def balance_workload(tasks_df):
    tasks_df = tasks_df.copy()
    assignee_load = {a: 0.0 for a in tasks_df["current_assignee"].unique()}
    assignee_high_count = {a: 0 for a in tasks_df["current_assignee"].unique()}
    new_assignments = []

    # Process high-priority tasks first (they should be placed with the
    # least-loaded person while capacity is still available)
    priority_rank = {"High": 0, "Medium": 1, "Low": 2}
    ordered = tasks_df.sort_values(
        by="priority", key=lambda s: s.map(priority_rank)
    )

    for _, row in ordered.iterrows():
        # pick least-loaded assignee; tie-break by fewer High tasks assigned
        chosen = min(
            assignee_load,
            key=lambda a: (assignee_load[a], assignee_high_count[a])
        )
        assignee_load[chosen] += row["workload_hours"]
        if row["priority"] == "High":
            assignee_high_count[chosen] += 1
        new_assignments.append(chosen)

    ordered = ordered.copy()
    ordered["balanced_assignee"] = new_assignments
    return ordered, assignee_load


sample_batch = df.sample(40, random_state=7)[
    ["task_id", "category", "priority", "workload_hours", "current_assignee"]
]
balanced, final_loads = balance_workload(sample_batch)
balanced.to_csv(f"{OUT_DIR}/workload_balancing_demo.csv", index=False)

print("\n=== Workload Balancing Demo (40 sample tasks) ===")
print("Final workload per assignee (hours):")
for a, load in sorted(final_loads.items(), key=lambda x: -x[1]):
    print(f"  {a:10s}: {load:6.1f} hrs")
print(f"\nSaved workload balancing demo -> {OUT_DIR}/workload_balancing_demo.csv")
