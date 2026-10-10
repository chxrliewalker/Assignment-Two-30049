"""
train_clustering.py
-------------------
Unsupervised k-means clustering WITHIN the malicious class only.

The label and the original `type` column are NOT used to build the clusters.
`type` is attached afterwards only to check that the clusters are real.

Steps:
    1. Keep malicious training URLs (label == 1), feature columns only
    2. log1p on count/length features (they are heavily skewed, so a few
       extreme URLs would otherwise dominate), then standardise
    3. Choose k by silhouette score for k = 2..8 (on a 10,000-row sample):
       the smallest k within 5% of the best score
    4. Fit k-means with the chosen k
    5. Describe each cluster:
         - z-score of each feature's cluster mean vs. the overall malicious
           mean (in standard deviations), and the 3 most different features
         - label composition by original `type` (check only)
         - example URLs closest to the cluster centre

Outputs (in models/):
    cluster_summary.csv      z-score profile, size, top features, composition
    cluster_examples.csv     5 example URLs per cluster
    plots/silhouette_k.png   silhouette score for each k
    plots/kmeans_clusters.png  2-D PCA view of the clusters

Usage (from the project root):
    python src/train_clustering.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import FEATURE_COLUMNS  # noqa: E402

DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "processed", "train_features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
RANDOM_STATE = 42
K_RANGE = range(2, 9)

# Binary flags and entropy are already on a small scale; everything else is a
# count or a length with a long tail.
SKEWED = [c for c in FEATURE_COLUMNS if c.startswith(("count_", "url_len", "domain_len",
                                                      "path_len", "domain_count"))]


def load_malicious(path=DATA_PATH):
    """Malicious training URLs only. Returns (features, url+type for reporting)."""
    df = pd.read_csv(path)
    mal = df[df["label"] == 1].reset_index(drop=True)
    print(f"Clustering {len(mal):,} malicious URLs (labels not used)")
    return mal[FEATURE_COLUMNS], mal[["url", "type"]]


def prepare(X):
    """log1p on skewed columns, then standardise every column."""
    Xt = X.astype(float).copy()
    Xt[SKEWED] = np.log1p(Xt[SKEWED])
    return StandardScaler().fit_transform(Xt)


def choose_k(Xs):
    """Silhouette score for each k on a fixed sample; returns best k and scores."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample = Xs[rng.choice(len(Xs), size=min(10000, len(Xs)), replace=False)]
    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(sample)
        scores[k] = silhouette_score(sample, labels)
        print(f"  k={k}: silhouette {scores[k]:.4f}")
    plt.figure(figsize=(6, 4))
    plt.plot(list(scores), list(scores.values()), marker="o")
    plt.title("Silhouette score by number of clusters")
    plt.xlabel("k")
    plt.ylabel("Silhouette score (10,000-URL sample)")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "silhouette_k.png"), dpi=150)
    plt.close()
    # Silhouette keeps creeping up as k grows, with smaller and smaller gains.
    # Take the smallest k within 5% of the best score: almost the same cluster
    # quality, but few enough clusters to describe and act on.
    top = max(scores.values())
    best = min(k for k, s in scores.items() if s >= 0.95 * top)
    print(f"Chosen k = {best} (smallest k within 5% of the best score {top:.4f})")
    return best, scores


def profile(X, info, labels, Xs, centres):
    """Describe each cluster against the overall malicious average."""
    overall_mean, overall_std = X.mean(), X.std().replace(0, 1)
    rows, examples = [], []
    for c in sorted(set(labels)):
        mask = labels == c
        z = ((X[mask].mean() - overall_mean) / overall_std).round(2)
        top = z.abs().sort_values(ascending=False).index[:3]
        comp = info.loc[mask, "type"].value_counts(normalize=True).round(3)
        row = {"cluster": c, "size": int(mask.sum()),
               "top_features": "; ".join(f"{f} {z[f]:+.2f}" for f in top),
               "composition": "; ".join(f"{t} {p:.1%}" for t, p in comp.items())}
        row.update({f"z_{f}": z[f] for f in FEATURE_COLUMNS})
        row.update({f"mean_{f}": round(X[mask][f].mean(), 3) for f in FEATURE_COLUMNS})
        rows.append(row)
        # Example URLs: the 5 members closest to the cluster centre
        idx = np.where(mask)[0]
        dist = np.linalg.norm(Xs[idx] - centres[c], axis=1)
        for i in idx[np.argsort(dist)[:5]]:
            examples.append({"cluster": c, "url": info.at[i, "url"], "type": info.at[i, "type"]})
    return pd.DataFrame(rows), pd.DataFrame(examples)


def plot_pca(Xs, labels):
    """2-D PCA scatter of a sample, coloured by cluster."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xs), size=min(20000, len(Xs)), replace=False)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
    pts = pca.transform(Xs[idx])
    plt.figure(figsize=(7, 5))
    colours = plt.get_cmap("tab10")
    for c in sorted(set(labels)):
        sel = labels[idx] == c
        plt.scatter(pts[sel, 0], pts[sel, 1], s=4, alpha=0.6, color=colours(c),
                    label=f"Cluster {c}")
    plt.legend(markerscale=4, frameon=False)
    plt.title("k-means clusters of malicious URLs (PCA, 20,000-URL sample)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "kmeans_clusters.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    X, info = load_malicious()
    Xs = prepare(X)
    k, _ = choose_k(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    summary, examples = profile(X, info, km.labels_, Xs, km.cluster_centers_)
    summary.to_csv(os.path.join(MODELS_DIR, "cluster_summary.csv"), index=False)
    examples.to_csv(os.path.join(MODELS_DIR, "cluster_examples.csv"), index=False)
    plot_pca(Xs, km.labels_)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)
    print(summary[["cluster", "size", "top_features", "composition"]].to_string(index=False))
    print(examples.to_string(index=False))


if __name__ == "__main__":
    main()
