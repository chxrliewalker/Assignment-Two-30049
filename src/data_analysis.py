"""
data_analysis.py
----------------
Exploratory analysis of the processed dataset (report Section 5).

Outputs (in models/plots/):
    class_balance.png        URLs per original type and binary label
    scheme_by_type.png       share of raw URLs that start with a scheme, per type
    feature_boxplots.png     key features, benign vs malicious
    feature_correlation.png  correlation heatmap of all features
and prints the numbers behind each chart.

Usage (from the project root):
    python src/data_analysis.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "models", "plots")
TYPE_ORDER = ["benign", "defacement", "phishing", "malware"]
BOX_FEATURES = ["url_len", "path_len", "count_digits", "domain_entropy",
                "count_equal", "count_slashes"]


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f))
                    for f in ("train_features.csv", "test_features.csv")], ignore_index=True)
    df["class"] = df["label"].map({0: "Benign", 1: "Malicious"})

    # 1. Class balance
    counts = df["type"].value_counts().reindex(TYPE_ORDER)
    print("URLs per type:\n" + counts.to_string())
    plt.figure(figsize=(6, 4))
    counts.plot.bar(color=["#4C72B0", "#DD8452", "#C44E52", "#8172B3"])
    plt.title("URLs per type after cleaning")
    plt.ylabel("Number of URLs")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_balance.png"), dpi=150)
    plt.close()

    # 2. Scheme artefact: how often the RAW url starts with http:// or https://
    scheme = df["url"].str.lower().str.match(r"^https?://").groupby(df["type"]).mean()
    scheme = scheme.reindex(TYPE_ORDER) * 100
    print("\n% of raw URLs with a scheme:\n" + scheme.round(1).to_string())
    plt.figure(figsize=(6, 4))
    scheme.plot.bar(color="#4C72B0")
    plt.title("Raw URLs that start with http:// or https://")
    plt.ylabel("% of URLs")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scheme_by_type.png"), dpi=150)
    plt.close()

    # 3. Distributions of key features by class (log scale: lengths are skewed)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, feat in zip(axes.ravel(), BOX_FEATURES):
        sns.boxplot(data=df, x="class", y=np.log1p(df[feat]), ax=ax,
                    showfliers=False, hue="class", legend=False,
                    palette={"Benign": "#4C72B0", "Malicious": "#C44E52"})
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.set_ylabel("log(1 + value)")
    fig.suptitle("Key features, benign vs malicious (outliers hidden)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_boxplots.png"), dpi=150)
    plt.close(fig)
    print("\nMedians by class:\n" + df.groupby("class")[BOX_FEATURES].median().to_string())

    # 4. Correlation between features
    corr = df[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.7})
    plt.title("Correlation between features")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_correlation.png"), dpi=150)
    plt.close()
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .sort_values(key=abs, ascending=False))
    print("\nMost correlated feature pairs:\n" + pairs.head(6).round(3).to_string())


if __name__ == "__main__":
    main()
