import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Dynamic project root calculation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR

def load_split_data(data_dir):
    """
    Loads train and test feature matrices from CSV files inside datasets/processed/.
    """
    train_path = os.path.join(data_dir, 'train_features.csv')
    test_path = os.path.join(data_dir, 'test_features.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed features not found in '{data_dir}'. "
            "Please run data_prep_and_split-v3.py first to generate 'train_features.csv' and 'test_features.csv'."
        )
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir):
    """
    Trains multiple classification models, evaluates metrics, saves plots, and exports the top pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42))
        ]),
        'DecisionTree': Pipeline([
            ('model', DecisionTreeClassifier(random_state=42, max_depth=5))
        ]),
        'RandomForest': Pipeline([
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    results = []
    best_f1 = -1.0
    best_model_name = ""
    best_pipeline = None
    
    print("--- Starting Supervised Classification Training ---\n")
    
    for name, pipeline in models.items():
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results.append({
                'Model': name,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1 Score': round(f1, 4)
            })
            
            print(f"=== {name} ===")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            # Save Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Legitimate', 'Phishing'],
                        yticklabels=['Legitimate', 'Phishing'])
            plt.title(f'Confusion Matrix: {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, f'cm_{name.lower()}.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved Confusion Matrix image to: {plot_path}\n")
            
            if f1 >= best_f1:
                best_f1 = f1
                best_model_name = name
                best_pipeline = pipeline
                
        except Exception as e:
            print(f"Could not evaluate {name}: {e}\n")
            
    if best_pipeline is not None:
        model_save_path = os.path.join(output_dir, 'phishing_pipeline.joblib')
        joblib.dump(best_pipeline, model_save_path)
        print(f"--> Saved best performing model ({best_model_name} with F1={best_f1:.4f}) to: {model_save_path}")
        
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved model comparison table to: {results_path}\n")
    
    return results_df

if __name__ == "__main__":
    default_data_dir = os.path.join(PROJECT_ROOT, 'datasets', 'processed')
    default_models_dir = os.path.join(PROJECT_ROOT, 'models')

    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_models_dir
    
    print(f"Loading split feature datasets from: {data_dir}")
    try:
        X_train, X_test, y_train, y_test = load_split_data(data_dir)
        train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
