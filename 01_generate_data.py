"""
Week 1 - Step 1: Generate a synthetic task-management dataset.

Why synthetic data? Trello/Jira APIs need paid accounts + OAuth setup, which
is a hassle for a portfolio project. A well-designed synthetic dataset with
realistic noise demonstrates the same skills (EDA, NLP, classification,
imbalance handling) without that friction. Swap this file out later for a
real Jira/Trello export -- the rest of the pipeline doesn't change.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_TASKS = 1500
ASSIGNEES = ["alice", "bharath", "chen", "divya", "esha", "farhan", "gita", "hassan"]

CATEGORY_TEMPLATES = {
    "bug": [
        "Fix crash when user uploads {x} file on {y} page",
        "Null pointer exception occurring in {y} module during {x} operation",
        "Application freezes after clicking {x} button on {y}",
        "Login fails intermittently with {x} error on {y} browser",
        "Data not saving correctly after {x} update in {y} form",
    ],
    "feature": [
        "Add {x} filter option to the {y} dashboard",
        "Implement dark mode support for {y} screen",
        "Build export to {x} functionality for {y} reports",
        "Add notification system for {x} events in {y}",
        "Create new {x} widget for the {y} homepage",
    ],
    "documentation": [
        "Update API docs for the {y} endpoint covering {x} usage",
        "Write onboarding guide for new {y} users about {x}",
        "Document deployment steps for {y} service on {x}",
        "Add code comments explaining {x} logic in {y} module",
        "Create changelog entry for {x} release of {y}",
    ],
    "testing": [
        "Write unit tests for {x} function in {y} module",
        "Add integration tests covering {x} flow in {y}",
        "Increase test coverage for {y} edge cases in {x}",
        "Set up regression test suite for {y} after {x} change",
        "Perform load testing on {y} service under {x} traffic",
    ],
    "deployment": [
        "Configure CI/CD pipeline for {y} service on {x}",
        "Roll out {x} update to {y} production environment",
        "Set up monitoring alerts for {y} after {x} deploy",
        "Migrate {y} database to {x} during maintenance window",
        "Provision new {x} servers for {y} scaling",
    ],
}

FILLERS_X = ["large", "recurring", "critical", "minor", "urgent", "scheduled",
             "quarterly", "nightly", "user-reported", "automated"]
FILLERS_Y = ["billing", "checkout", "auth", "profile", "search", "inventory",
             "reporting", "admin", "mobile app", "API gateway"]

# Rough real-world signal: bugs tend to be more urgent, docs less urgent
CATEGORY_BASE_URGENCY = {
    "bug": 0.75, "feature": 0.45, "documentation": 0.20,
    "testing": 0.40, "deployment": 0.60,
}


# Generic phrasing that can plausibly show up in any category's ticket --
# this creates real vocabulary overlap so classifiers aren't trivially perfect.
GENERIC_SUFFIXES = [
    "Please review before the next sprint.",
    "Related to the {y} incident from last week.",
    "Customer escalation, please prioritize.",
    "Blocked by another team, follow up needed.",
    "",  # sometimes no suffix at all
    "",
]


def make_description(category):
    template = np.random.choice(CATEGORY_TEMPLATES[category])
    desc = template.format(x=np.random.choice(FILLERS_X), y=np.random.choice(FILLERS_Y))
    suffix = np.random.choice(GENERIC_SUFFIXES)
    if suffix:
        desc = f"{desc}. {suffix.format(y=np.random.choice(FILLERS_Y))}"
    return desc


def make_priority(category, deadline_days, workload_hours):
    """Priority is driven by category urgency + tight deadline + noise,
    so the label is *learnable* from features but not trivially deterministic
    -- this mimics how real task-priority signals behave."""
    urgency = CATEGORY_BASE_URGENCY[category]
    deadline_pressure = max(0, (7 - deadline_days) / 7)
    workload_pressure = min(workload_hours / 40, 1.0)
    score = 0.5 * urgency + 0.35 * deadline_pressure + 0.15 * workload_pressure
    score += np.random.normal(0, 0.08)
    if score >= 0.62:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"


def generate_dataset(n=N_TASKS):
    rows = []
    for i in range(n):
        category = np.random.choice(list(CATEGORY_TEMPLATES.keys()),
                                     p=[0.30, 0.25, 0.15, 0.15, 0.15])
        description = make_description(category)
        deadline_days = int(np.clip(np.random.exponential(scale=5), 0, 30))
        workload_hours = round(float(np.clip(np.random.gamma(shape=2, scale=6), 1, 60)), 1)
        assignee = np.random.choice(ASSIGNEES)
        priority = make_priority(category, deadline_days, workload_hours)

        # 6% label noise: mislabeled category, as happens with real ticket
        # tagging (person picks the wrong dropdown value).
        recorded_category = category
        if np.random.rand() < 0.06:
            other_categories = [c for c in CATEGORY_TEMPLATES if c != category]
            recorded_category = np.random.choice(other_categories)

        rows.append({
            "task_id": f"T{i+1:05d}",
            "description": description,
            "category": recorded_category,
            "deadline_days": deadline_days,
            "workload_hours": workload_hours,
            "current_assignee": assignee,
            "priority": priority,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    # Inject messiness on purpose (duplicates + missing values) since the
    # project brief explicitly requires an EDA + cleaning step.
    dupe_idx = np.random.choice(df.index, size=15, replace=False)
    df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)
    missing_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_idx, "workload_hours"] = np.nan

    out_path = "./data/tasks_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
