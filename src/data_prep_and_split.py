import os
import sys #bububhyub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dynamic project root calculation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If script is inside 'src', project root is parent directory; otherwise current directory
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from feature_extraction import extract_url_features
except ImportError:
    try:
        from src.feature_extraction import extract_url_features
    except ImportError:
        raise ImportError(
            "Could not import 'feature_extraction'. Ensure 'feature_extraction.py' "
            "is located in the same directory or in 'src/'."
        )

def load_and_preprocess_data(csv_path, url_column='url', label_column='label'):
    """
    Loads raw CSV dataset, cleans raw inputs, and validates target labels.
    """
    print(f"Loading dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}. Please check the file path.")

    df = pd.read_csv(csv_path)
    
    cols_lower = {c.lower(): c for c in df.columns}
    if url_column not in df.columns and 'raw_url' in cols_lower:
        url_column = cols_lower['raw_url']
    elif url_column not in df.columns and 'urls' in cols_lower:
        url_column = cols_lower['urls']

    if label_column not in df.columns and 'target' in cols_lower:
        label_column = cols_lower['target']
    elif label_column not in df.columns and 'class' in cols_lower:
        label_column = cols_lower['class']

    df = df.dropna(subset=[url_column, label_column]).copy()
    df[url_column] = df[url_column].astype(str).str.strip()
    df[label_column] = df[label_column].astype(int)
    
    print(f"Loaded {len(df)} valid records.")
    print(f"Class Distribution:\n{df[label_column].value_counts(normalize=True).round(4) * 100}%")
    return df, url_column, label_column

def prepare_and_split(df, url_column='url', label_column='label', test_size=0.3, random_state=42):
    """
    Extracts features and performs a stratified train/test split.
    """
    print("\n--- Phase 1: Feature Extraction ---")
    features_df = extract_url_features(df, url_column=url_column)
    
    non_feature_cols = [url_column, label_column]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    
    X = features_df[feature_cols].copy()
    y = features_df[label_column].copy()
    
    print(f"Extracted {X.shape[1]} numerical features across {X.shape[0]} samples.")
    print(f"Feature List: {list(X.columns)}")
    
    print(f"\n--- Phase 2: Stratified Train/Test Split (Test Size: {int(test_size*100)}%) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Testing Set:  {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, list(X.columns)

def save_split_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Saves the train/test splits as CSV files inside datasets/processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_features.csv')
    test_path = os.path.join(output_dir, 'test_features.csv')

    train_df = X_train.copy()
    train_df['label'] = y_train
    train_df.to_csv(train_path, index=False)
    
    test_df = X_test.copy()
    test_df['label'] = y_test
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved processed split datasets to: {output_dir}")
    print(f" - {train_path}")
    print(f" - {test_path}")

if __name__ == "__main__":
    datasets_dir = os.path.join(PROJECT_ROOT, 'datasets')
    raw_dir = os.path.join(datasets_dir, 'raw')
    processed_dir = os.path.join(datasets_dir, 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        raw_csv_path = sys.argv[1]
    else:
        raw_csv_path = os.path.join(raw_dir, 'sample_phishing_data.csv')
        sample_data = pd.DataFrame({
            'url': [
                'https://google.com',
                'http://login-verify-secure-account.com/update',
                'https://github.com/scikit-learn/scikit-learn',
                'http://192.168.1.1/admin/login.php',
                'https://paypal.com.account-update.secure.xyz@bad.com/login',
                'https://wikipedia.org/wiki/Main_Page',
                'http://bit.ly/3xYz89',
                'http://malicious-phishing-link-123.net/confirm'
            ],
            'label': [0, 1, 0, 1, 1, 0, 1, 1]  # 0 = Legitimate, 1 = Phishing
        })
        sample_data.to_csv(raw_csv_path, index=False)
    
    print(f"Running Data Prep and Split Pipeline on: {raw_csv_path}\n")
    df, url_col, label_col = load_and_preprocess_data(raw_csv_path)
    X_train, X_test, y_train, y_test, feature_names = prepare_and_split(df, url_column=url_col, label_column=label_col)
    
    save_split_data(X_train, X_test, y_train, y_test, output_dir=processed_dir)
