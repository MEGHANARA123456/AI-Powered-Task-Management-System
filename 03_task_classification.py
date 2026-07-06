"""
Week 2: Feature extraction (TF-IDF) + task category classification
(Naive Bayes and SVM), evaluated with accuracy / precision / recall / F1.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

CLEAN_PATH = "./data/tasks_clean.csv"
OUT_DIR = "./outputs"
MODEL_DIR = "./models"

df = pd.read_csv(CLEAN_PATH)

X_text = df["clean_description"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

results = {}
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM (LinearSVC)": LinearSVC(random_state=42, max_iter=5000),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }
    trained_models[name] = model

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, preds))

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = trained_models[best_name]
print(f"\nBest category classifier: {best_name}")

preds_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, preds_best, labels=sorted(y.unique()))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title(f"Confusion Matrix - {best_name} (Category Classification)")
plt.ylabel("True Category")
plt.xlabel("Predicted Category")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_category.png", dpi=140)
plt.close()

metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind="bar", figsize=(8, 5), colormap="viridis")
plt.title("Task Category Classification: Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison_classification.png", dpi=140)
plt.close()

joblib.dump(tfidf, f"{MODEL_DIR}/tfidf_vectorizer.joblib")
safe_name = best_name.replace(" ", "_").replace("(", "").replace(")", "")
joblib.dump(best_model, f"{MODEL_DIR}/category_classifier_{safe_name}.joblib")
with open(f"{OUT_DIR}/classification_results.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

print(f"\nSaved TF-IDF vectorizer, best model, plots, and metrics JSON to {MODEL_DIR} / {OUT_DIR}")
