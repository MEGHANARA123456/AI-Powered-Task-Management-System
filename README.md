# AI-Powered Task Management System

An end-to-end machine learning pipeline that classifies software tasks by category, predicts their priority, and simulates workload balancing across a team — built as a 4-week ML project using classic NLP and supervised learning techniques.

## Overview

Given a raw task description (e.g. bug reports, feature requests, deployment tickets), the pipeline:

1. Generates a realistic synthetic dataset of task tickets
2. Cleans and preprocesses text (tokenization, stopword removal, stemming)
3. Classifies each task into a category (`bug`, `feature`, `documentation`, `deployment`, `testing`) using TF-IDF + Naive Bayes/SVM
4. Predicts task priority (`Low`, `Medium`, `High`) using tuned Random Forest / Gradient Boosting
5. Simulates workload balancing across team members based on predicted priority and effort

## Results

| Task | Model | Accuracy | F1 Score |
|---|---|---|---|
| Category Classification | Naive Bayes (best) | 93.7% | 0.936 |
| Category Classification | SVM (LinearSVC) | 93.4% | 0.933 |
| Priority Prediction | Random Forest (tuned) | 75.4% | 0.753 |
| Priority Prediction | Gradient Boosting | 75.4% | 0.753 |

Random Forest hyperparameters were tuned via `GridSearchCV` (best params: `max_depth=8, min_samples_leaf=1, n_estimators=200`).

## Project Structure

```
.
├── 01_generate_data.py           # Synthetic task dataset generator
├── 02_eda_preprocessing.py       # EDA + text cleaning/preprocessing
├── 03_task_classification.py     # TF-IDF + Naive Bayes/SVM category classifier
├── 04_priority_and_workload.py   # Priority prediction + workload balancing demo
├── data/
│   ├── tasks_raw.csv
│   └── tasks_clean.csv
├── models/
│   ├── tfidf_vectorizer.joblib
│   └── best_category_model.joblib
├── outputs/
│   ├── eda_overview.png
│   ├── text_length_dist.png
│   ├── confusion_matrix_category.png
│   ├── confusion_matrix_priority.png
│   ├── workload_balancing_demo.csv
│   └── metrics.json
└── report/
    └── project1_ai_task_management_report.md
```

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

Run the scripts in order — each depends on the previous step's output:

```bash
python 01_generate_data.py
python 02_eda_preprocessing.py
python 03_task_classification.py
python 04_priority_and_workload.py
```

## Notes

- The dataset is synthetically generated (documented in `report/`) since real Jira/Trello API access requires paid accounts — a standard substitute for portfolio/academic projects.
- Gradient Boosting (scikit-learn) is used in place of XGBoost for zero external dependencies; swapping in XGBoost is a one-line change if desired.
- Label noise and category overlap were deliberately introduced into the synthetic data generator to avoid unrealistic 100% accuracy and produce defensible, real-world-like results.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

## Author

Meghana Kamatam
