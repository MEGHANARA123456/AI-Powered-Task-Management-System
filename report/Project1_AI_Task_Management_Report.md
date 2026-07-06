# AI-Powered Task Management System
### Final Project Report

---

## 1. Problem Statement

Design an intelligent task management system that uses NLP and ML to
automatically **classify** tasks by type, **predict priority**, and
**balance workload** across a team — reducing manual triage time and making
assignment fairer and more consistent.

---

## 2. Dataset

Real Jira/Trello exports require paid API access and OAuth setup, so a
**synthetic dataset of 1,500 tasks** was generated to mirror realistic
ticket data:

- `description` — free-text task description (5 categories × 5 phrasing
  templates × randomized filler terms, with some phrasing shared across
  categories to avoid unrealistically clean separation)
- `category` — bug / feature / documentation / testing / deployment
- `deadline_days`, `workload_hours` — numeric fields drawn from skewed
  distributions matching how these values look in practice
- `current_assignee` — one of 8 simulated team members
- `priority` — Low / Medium / High, derived from a weighted mix of category
  urgency, deadline pressure, and workload, plus random noise

To make the dataset realistic rather than trivially clean, **6% label
noise** was injected into `category` (simulating human mis-tagging), plus
duplicate rows and missing `workload_hours` values, so that cleaning and
evaluation reflect real-world messiness.

---

## 3. Week 1 — Data Collection, Cleaning & NLP Preprocessing

**What was done:**
- Loaded 1,515 raw rows; found 14 duplicate rows and 20 missing
  `workload_hours` values
- Dropped duplicates, imputed missing workload with the column median →
  clean dataset of 1,500 rows
- EDA covered category distribution, priority distribution, workload and
  deadline distributions, and description length by category (see
  `outputs/eda_overview.png`, `outputs/text_length_dist.png`)
- NLP preprocessing pipeline: lowercasing → punctuation removal →
  tokenization → stopword removal → lightweight suffix-stripping
  (stemming), implemented with plain Python/regex so the project has
  **zero external NLP dependencies** (no nltk download required — it runs
  identically in Colab, Jupyter, or a bare Python environment)

**Mid-project checkpoint deliverables — done:**
- ✅ Cleaned, preprocessed dataset (`data/tasks_clean.csv`)
- ✅ EDA visualizations
- ✅ (see Week 2 below) Task classifier trained and evaluated

---

## 4. Week 2 — Feature Extraction & Task Classification

**What was done:**
- TF-IDF vectorization on cleaned text (unigrams + bigrams, top 2,000
  features)
- Trained **Naive Bayes** and **SVM (LinearSVC)** to predict task
  `category` from the description text
- Evaluated with accuracy, weighted precision/recall/F1, and a confusion
  matrix

**Results:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Naive Bayes** | **93.7%** | 93.9% | 93.7% | **93.6%** |
| SVM (LinearSVC) | 93.4% | 93.5% | 93.4% | 93.3% |

Naive Bayes edges out SVM slightly and is the faster model to train, so it
was selected as the production classifier. The confusion matrix
(`outputs/confusion_matrix_category.png`) shows most errors happen between
**deployment** and **testing** tasks — sensible, since both categories
share vocabulary like "pipeline," "environment," and "staging."

Accuracy sits in the low-90s rather than near-100% because of the 6% label
noise injected into the data — this is a **more trustworthy result** than
a suspiciously perfect score, and mirrors what you'd see on a real,
imperfectly-tagged ticket backlog.

---

## 5. Week 3 — Priority Prediction & Workload Balancing

**Priority prediction:**
- Features used: encoded category, deadline (days remaining), workload
  hours, and description length
- Models: **Random Forest** (tuned via `GridSearchCV` over
  `n_estimators`, `max_depth`, `min_samples_leaf` on 4-fold CV) and
  **Gradient Boosting** (sklearn's built-in stand-in for XGBoost, since
  this sandbox has no internet access to install the `xgboost` package —
  swapping in `XGBClassifier` is a one-line change if you have it
  installed locally)

**Best Random Forest hyperparameters found:** `n_estimators=200,
max_depth=8, min_samples_leaf=1`

**Results:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest (tuned)** | **75.4%** | 75.6% | 75.4% | 75.3% |
| Gradient Boosting | 75.4% | 75.6% | 75.4% | 75.3% |

75% on a genuinely noisy 3-class problem (priority was generated from a
weighted score *plus* random noise) is a solid, realistic result —
`outputs/feature_importance_priority.png` shows **category** and
**deadline pressure** are the two strongest predictors of priority, which
matches how the labels were constructed and gives a sanity-check that the
model learned real signal rather than noise.

**Workload balancing heuristic:**
A greedy heuristic assigns each incoming task to whichever team member
currently has the **lowest total workload**, processing High-priority
tasks first so urgent work lands with the least-loaded person while
capacity is still available. A 40-task demo run
(`outputs/workload_balancing_demo.csv`) shows the heuristic keeps the
spread between the busiest and least-busy team member to about 20 hours,
versus a much wider spread under random assignment.

---

## 6. Week 4 — Final Models & Summary

**Final deliverables:**

| Artifact | Location |
|---|---|
| Cleaned dataset | `data/tasks_clean.csv` |
| Category classifier (Naive Bayes) | `models/category_classifier_Naive_Bayes.joblib` |
| Priority model (Random Forest, tuned) | `models/priority_model_Random.joblib` |
| TF-IDF vectorizer | `models/tfidf_vectorizer.joblib` |
| EDA visualizations | `outputs/eda_overview.png`, `outputs/text_length_dist.png` |
| Classification confusion matrix & comparison chart | `outputs/confusion_matrix_category.png`, `outputs/model_comparison_classification.png` |
| Priority confusion matrix & feature importance | `outputs/confusion_matrix_priority.png`, `outputs/feature_importance_priority.png` |
| Workload balancing demo | `outputs/workload_balancing_demo.csv` |
| Metrics (machine-readable) | `outputs/classification_results.json`, `outputs/priority_results.json` |

**Summary:** The system reliably classifies incoming task descriptions
into the right category (~94% F1), predicts a reasonable priority level
using deadline/workload/category signals (~75% F1 on a 3-class problem
with intentional label noise), and includes a transparent, explainable
workload-balancing heuristic ready to sit behind a dashboard or Slack/Jira
bot in a real deployment.

**Next steps for a production version:**
1. Replace the synthetic dataset with a real Jira/Trello export via their
   APIs (the pipeline's cleaning/preprocessing steps don't need to change)
2. Swap Gradient Boosting for `XGBClassifier` if `xgboost` is available in
   your environment — same code, just a different import
3. Replace the workload-balancing heuristic with a proper optimization
   solver (e.g. linear assignment via `scipy.optimize.linear_sum_assignment`)
   once there are hard constraints like skills-matching or PTO calendars
4. Wrap the trained models in a small Flask/Streamlit app for a live demo
