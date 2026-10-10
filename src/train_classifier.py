"""
train_classifier.py
-------------------
Trains and compares every classifier on the same train/test split.

Outputs (in models/):
    model_comparison.csv          accuracy, precision, recall, F1, FPR, time, size
    phishing_pipeline.joblib      the best model by F1, used by predict.py
    test_predictions.csv          the best model's prediction for every test URL
                                  (used for error analysis)
    plots/cm_<model>.png          one confusion matrix per model
    plots/feature_importance.png  random forest feature importance

Usage (from the project root):
    python src/train_classifier.py
"""

import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to file without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import FEATURE_COLUMNS  # noqa: E402
from model_builders import BEYOND_UNIT_MODELS, build_models  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
INPUT_COLUMNS = ["url"] + FEATURE_COLUMNS  # what every pipeline receives


def load_split(data_dir=DATA_DIR):
    """Load the processed train and test CSVs."""
    train = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_features.csv"))
    print(f"Train: {len(train):,} rows   Test: {len(test):,} rows")
    return train, test


def evaluate(y_true, y_pred):
    """Standard metrics for the malicious class, plus the false-positive rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        # FPR = share of ALL benign URLs wrongly flagged (not 1 - precision)
        "FPR": fp / (fp + tn),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def plot_confusion(cm, name):
    """Save a labelled confusion-matrix heatmap."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Legitimate", "Malicious"],
                yticklabels=["Legitimate", "Malicious"])
    plt.title(f"Confusion matrix: {name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{name.lower()}.png"), dpi=150)
    plt.close()


def plot_feature_importance(pipeline):
    """Bar chart of random forest feature importance (mean decrease in impurity)."""
    imp = pd.Series(pipeline.named_steps["model"].feature_importances_,
                    index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(7, 6))
    imp.plot.barh(color="#4C72B0")
    plt.title("Random forest feature importance")
    plt.xlabel("Mean decrease in impurity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    return imp.sort_values(ascending=False)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    train, test = load_split()
    X_train, y_train = train[INPUT_COLUMNS], train["label"]
    X_test, y_test = test[INPUT_COLUMNS], test["label"]

    rows, fitted = [], {}
    for name, pipeline in build_models().items():
        print(f"\n=== {name} ===")
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_secs = time.time() - start
        y_pred = pipeline.predict(X_test)

        # Size of the saved model file, in MB
        tmp = os.path.join(MODELS_DIR, f"_{name}.joblib")
        joblib.dump(pipeline, tmp, compress=3)
        size_mb = os.path.getsize(tmp) / 1e6
        os.remove(tmp)

        m = evaluate(y_test, y_pred)
        m.update({"Model": name, "Beyond unit": name in BEYOND_UNIT_MODELS,
                  "Train seconds": round(train_secs, 1), "Size MB": round(size_mb, 1)})
        rows.append(m)
        fitted[name] = pipeline
        plot_confusion(confusion_matrix(y_test, y_pred), name)
        print(f"Accuracy {m['Accuracy']:.4f}  Precision {m['Precision']:.4f}  "
              f"Recall {m['Recall']:.4f}  F1 {m['F1']:.4f}  FPR {m['FPR']:.4f}  "
              f"({train_secs:.0f}s, {size_mb:.1f} MB)")

    results = pd.DataFrame(rows)[["Model", "Beyond unit", "Accuracy", "Precision", "Recall",
                                  "F1", "FPR", "TN", "FP", "FN", "TP",
                                  "Train seconds", "Size MB"]]
    results.round(4).to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print("\n" + results.round(4).to_string(index=False))

    # Deploy the best model by F1 on the malicious class
    best = results.loc[results["F1"].idxmax(), "Model"]
    best_pipe = fitted[best]
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "phishing_pipeline.joblib"), compress=3)
    print(f"\nSaved best model ({best}) to models/phishing_pipeline.joblib")

    # Save every test prediction of the best model for error analysis
    preds = test[["url", "type", "label"]].copy()
    preds["predicted"] = best_pipe.predict(X_test)
    preds["malicious_probability"] = best_pipe.predict_proba(X_test)[:, 1].round(4)
    preds.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)

    imp = plot_feature_importance(fitted["RandomForest"])
    print("\nRandom forest feature importance (top 8):")
    print(imp.head(8).round(4).to_string())


if __name__ == "__main__":
    main()
