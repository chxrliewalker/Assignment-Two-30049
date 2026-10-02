import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def get_project_root():
    """
    Determines project root directory whether script is run from root or src/.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) in ['src', 'clustering', 'classifier']:
        return os.path.abspath(os.path.join(script_dir, '..'))
    return os.getcwd()

def load_phishing_features(data_dir=None):
    """
    Loads train_features.csv and filters strictly for phishing instances (label == 1).
    """
    if data_dir is None:
        project_root = get_project_root()
        data_dir = os.path.join(project_root, 'datasets', 'processed')

    train_path = os.path.join(data_dir, 'train_features.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Processed features not found at '{train_path}'. "
            "Please run data_prep_and_split-v3.py first to generate processed features."
        )

    df = pd.read_csv(train_path)
    
    # Filter strictly for Phishing class (label == 1)
    if 'label' in df.columns:
        phishing_df = df[df['label'] == 1].drop(columns=['label'])
    else:
        phishing_df = df.copy()
        
    print(f"Loaded {len(phishing_df)} phishing instances for unsupervised clustering.")
    return phishing_df

def run_phishing_clustering(phishing_df, n_clusters=3, output_dir=None):
    """
    Applies StandardScaler and KMeans clustering on phishing URLs.
    Generates PCA visual plots and cluster feature comparison table.
    """
    if output_dir is None:
        project_root = get_project_root()
        output_dir = os.path.join(project_root, 'models')

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n--- Starting K-Means Unsupervised Clustering (k={n_clusters}) ---")

    # 1. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(phishing_df)

    # 2. KMeans Fitting
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Evaluate Silhouette Score if sample size permits
    if len(phishing_df) > n_clusters:
        sil_score = silhouette_score(X_scaled, cluster_labels)
        print(f"Silhouette Score (k={n_clusters}): {sil_score:.4f}")

    # 3. Cluster Profile & Mean Feature Analysis
    df_analyzed = phishing_df.copy()
    df_analyzed['cluster'] = cluster_labels

    cluster_means = df_analyzed.groupby('cluster').mean().round(3)

    print("\nCluster Feature Averages:")
    print(cluster_means)

    # Save summary table
    summary_path = os.path.join(output_dir, 'cluster_summary.csv')
    cluster_means.to_csv(summary_path)
    print(f"\nSaved cluster profile summary table to: {summary_path}")

    # 4. PCA Visualization (2D Projection)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8, edgecolors='k')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Unsupervised K-Means Clustering on Phishing URLs (2D PCA)')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, 'kmeans_clusters.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved cluster scatter plot to: {plot_path}\n")

    return cluster_means

if __name__ == "__main__":
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'datasets', 'processed')
    output_dir = os.path.join(project_root, 'models')

    try:
        phishing_df = load_phishing_features(data_dir)
        run_phishing_clustering(phishing_df, n_clusters=3, output_dir=output_dir)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Creating mock phishing feature dataset for test demonstration...")
        
        mock_phishing = pd.DataFrame({
            'url_len': [35, 61, 28, 40, 55, 30],
            'domain_len': [26, 47, 11, 11, 20, 15],
            'path_len': [9, 14, 17, 29, 35, 15],
            'count_dots': [2, 5, 2, 3, 4, 2],
            'count_hyphens': [2, 4, 0, 1, 3, 1],
            'count_slashes': [2, 2, 3, 3, 4, 2],
            'count_questions': [0, 0, 0, 1, 0, 0],
            'count_equal': [0, 0, 0, 1, 2, 0],
            'count_at': [0, 1, 0, 0, 0, 0],
            'count_digits': [0, 0, 4, 3, 1, 0],
            'domain_count_dots': [2, 4, 3, 1, 2, 1],
            'has_http': [1, 1, 1, 1, 1, 1],
            'has_https': [0, 0, 0, 0, 0, 0],
            'is_ip_address': [0, 0, 1, 0, 0, 0],
            'domain_has_prefix_suffix_hyphen': [0, 0, 0, 0, 0, 0],
            'is_shortened': [0, 0, 0, 0, 0, 0],
            'count_suspicious_keywords': [1, 5, 2, 2, 3, 1],
            'domain_entropy': [3.84, 4.11, 2.30, 3.10, 3.90, 3.20]
        })
        
        os.makedirs(data_dir, exist_ok=True)
        mock_phishing['label'] = 1
        mock_phishing.to_csv(os.path.join(data_dir, 'train_features.csv'), index=False)
        
        print("Mock file created. Retrying clustering pipeline...\n")
        phishing_df = load_phishing_features(data_dir)
        run_phishing_clustering(phishing_df, n_clusters=3, output_dir=output_dir)
