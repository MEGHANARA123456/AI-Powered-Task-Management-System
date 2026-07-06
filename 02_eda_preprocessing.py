"""
Week 1 - Step 2: EDA + cleaning + NLP preprocessing on task descriptions.

Note on tooling: instead of importing nltk (which needs a one-time corpus
download and isn't always available offline), this script implements
tokenization / stopword removal / light stemming manually with plain Python
+ regex. This keeps the pipeline dependency-free and reproducible anywhere
(Colab, local Jupyter, CI). If you have nltk installed, feel free to swap
`simple_tokenize` / `STOPWORDS` for `nltk.word_tokenize` / `nltk.corpus.stopwords`
-- the rest of the pipeline (TF-IDF, models) doesn't care which you use.
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

RAW_PATH = "./data/tasks_raw.csv"
CLEAN_PATH = "./data/tasks_clean.csv"
OUT_DIR = "./outputs"

# --- a compact English stopword list (covers the common function words) ---
STOPWORDS = set("""
a an the and or but if while of to in on at for with by from as is are was
were be been being this that these those it its it's their his her they
he she we you your our i not no nor do does did doing have has had having
will would shall should can could may might must into over under again
further then once here there when where why how all any both each few more
most other some such only own same so than too very s t can will just don
should now
""".split())


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return tokens


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


SUFFIXES = ["ing", "edly", "ed", "ly", "es", "s"]


def simple_stem(word):
    """Lightweight Porter-style suffix stripper. Not linguistically perfect,
    but sufficient to collapse 'uploading'/'uploads'/'uploaded' -> 'upload'
    for TF-IDF purposes."""
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def preprocess_text(text):
    tokens = simple_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)


def run_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    sns.countplot(data=df, x="category", order=df["category"].value_counts().index,
                  ax=axes[0, 0], hue="category", palette="viridis", legend=False)
    axes[0, 0].set_title("Task Count by Category")
    axes[0, 0].tick_params(axis="x", rotation=20)

    sns.countplot(data=df, x="priority", order=["Low", "Medium", "High"],
                  ax=axes[0, 1], hue="priority", palette="rocket", legend=False)
    axes[0, 1].set_title("Task Count by Priority")

    sns.histplot(df["workload_hours"].dropna(), bins=30, kde=True,
                 ax=axes[1, 0], color="steelblue")
    axes[1, 0].set_title("Workload Hours Distribution")

    sns.histplot(df["deadline_days"], bins=30, kde=True, ax=axes[1, 1], color="indianred")
    axes[1, 1].set_title("Deadline (Days Remaining) Distribution")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=140)
    plt.close()

    # description length distribution, split by category
    df["desc_length"] = df["description"].str.split().apply(len)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="category", y="desc_length", hue="category",
                palette="viridis", legend=False)
    plt.title("Description Word Count by Category")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/text_length_dist.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)

    print("=== RAW DATA SHAPE ===", df.shape)
    print("\n=== MISSING VALUES ===\n", df.isna().sum())
    print("\n=== DUPLICATE ROWS ===", df.duplicated().sum())

    # --- cleaning ---
    df = df.drop_duplicates()
    median_workload = df["workload_hours"].median()
    df["workload_hours"] = df["workload_hours"].fillna(median_workload)
    df = df.reset_index(drop=True)

    print("\n=== AFTER CLEANING ===", df.shape)

    # --- EDA plots ---
    run_eda(df)

    # --- NLP preprocessing ---
    df["clean_description"] = df["description"].apply(preprocess_text)

    print("\n=== SAMPLE PREPROCESSING ===")
    for orig, clean in df[["description", "clean_description"]].head(5).values:
        print(f"  RAW : {orig}\n  CLEAN: {clean}\n")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned dataset -> {CLEAN_PATH}")
    print(f"Saved EDA plots -> {OUT_DIR}/eda_overview.png, {OUT_DIR}/text_length_dist.png")
