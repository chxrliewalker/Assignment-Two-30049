"""
evaluate_model.py
-----------------
Stress tests that try to break the models, beyond the random split.

Test 1  Unseen domains: split by registered domain, so no website appears in
        both training and test data. A random split lets the model recognise
        a site it has already seen; this test removes that shortcut.
Test 2  Unseen attack type: train with every `phishing` URL removed, then
        measure how many phishing URLs are still caught.
Test 3  Scheme ablation: add the old has_http / has_https flags back, and test
        on benign URLs rewritten with "https://www." in front. Shows how much
        the scheme artefact would have fooled a model.

Test 4  Popular websites never seen in training: Alexa ranks 20,001-30,000
        (datasets/processed/popular_sites_holdout.csv). Every one is
        legitimate, so any "malicious" verdict is a false alarm.

Run on random forest (best feature-based model) and the character n-gram model
(best overall). Writes models/stress_tests.csv.

Usage (from the project root):
    python src/evaluate_model.py
"""

import os
import sys

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import (FEATURE_COLUMNS, canonicalise, safe_url_parse,  # noqa: E402
                                split_host, url_features)
from model_builders import RANDOM_STATE, build_models  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")
OUT_PATH = os.path.join(PROJECT_ROOT, "models", "stress_tests.csv")
INPUT_COLUMNS = ["url"] + FEATURE_COLUMNS
MODELS = ["RandomForest", "CharNgram"]


def registered_domain(url):
    """The site a URL belongs to, e.g. 'paypal.com.evil.xyz/x' -> 'evil.xyz'."""
    return split_host(safe_url_parse(canonicalise(url)).netloc)[1]


def scores(y_true, y_pred):
    return {"Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)}


def test_unseen_domains(data, models):
    """Test 1: no registered domain is shared between train and test."""
    groups = data["url"].map(registered_domain)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(data, data["label"], groups))
    train, test = data.iloc[tr], data.iloc[te]
    print(f"  {groups.nunique():,} domains; train {len(train):,} rows, test {len(test):,} rows")
    rows = []
    for name in MODELS:
        model = clone(models[name]).fit(train[INPUT_COLUMNS], train["label"])
        r = scores(test["label"], model.predict(test[INPUT_COLUMNS]))
        rows.append({"Test": "Unseen domains", "Model": name, **r})
        print(f"  {name}: {r}")
    return rows


def test_unseen_attack_type(train, test, models, held_out="phishing"):
    """Test 2: the model never sees one malicious type during training."""
    reduced = train[train["type"] != held_out]
    target = test[test["type"] == held_out]
    rows = []
    for name in MODELS:
        full = clone(models[name]).fit(train[INPUT_COLUMNS], train["label"])
        part = clone(models[name]).fit(reduced[INPUT_COLUMNS], reduced["label"])
        seen = recall_score(target["label"], full.predict(target[INPUT_COLUMNS]))
        unseen = recall_score(target["label"], part.predict(target[INPUT_COLUMNS]))
        rows.append({"Test": f"Unseen type ({held_out}) - seen in training",
                     "Model": name, "Recall": seen})
        rows.append({"Test": f"Unseen type ({held_out}) - held out",
                     "Model": name, "Recall": unseen})
        print(f"  {name}: {held_out} recall {seen:.4f} when seen -> {unseen:.4f} when held out")
    return rows


def test_scheme_ablation(train, test):
    """Test 3: random forest with vs. without the raw-scheme flags."""
    def add_scheme(df, urls):
        out = df[FEATURE_COLUMNS].copy()
        low = urls.str.lower()
        out["has_http"] = low.str.startswith("http://").astype(int)
        out["has_https"] = low.str.startswith("https://").astype(int)
        return out

    # Benign test URLs as a user might paste them from a browser
    benign = test[test["label"] == 0]
    rewritten = "https://www." + benign["url"].map(canonicalise)
    rewritten_feats = pd.DataFrame([url_features(u) for u in rewritten], index=benign.index)

    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                random_state=RANDOM_STATE)
    with_scheme = clone(rf).fit(add_scheme(train, train["url"]), train["label"])
    without = clone(rf).fit(train[FEATURE_COLUMNS], train["label"])

    fpr_with = with_scheme.predict(add_scheme(rewritten_feats, rewritten)).mean()
    fpr_without = without.predict(rewritten_feats[FEATURE_COLUMNS]).mean()
    f1_with = f1_score(test["label"], with_scheme.predict(add_scheme(test, test["url"])))
    print(f"  with scheme flags: random-split F1 {f1_with:.4f}, FPR on https:// benign {fpr_with:.4f}")
    print(f"  without (final):   FPR on https:// benign {fpr_without:.4f}")
    return [{"Test": "Scheme ablation - with scheme flags", "Model": "RandomForest",
             "F1": f1_with, "FPR on https benign": fpr_with},
            {"Test": "Scheme ablation - without (final)", "Model": "RandomForest",
             "FPR on https benign": fpr_without}]


def test_popular_sites(train, holdout, models):
    """Test 4: false-alarm rate on popular websites the model never saw."""
    rows = []
    for name in MODELS:
        model = clone(models[name]).fit(train[INPUT_COLUMNS], train["label"])
        fpr = model.predict(holdout[INPUT_COLUMNS]).mean()
        rows.append({"Test": "Unseen popular websites", "Model": name, "FPR": fpr})
        print(f"  {name}: {fpr:.4f} of {len(holdout):,} popular sites flagged malicious")
    return rows


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    models = build_models()
    rows = []
    print("Test 1: unseen domains")
    rows += test_unseen_domains(pd.concat([train, test], ignore_index=True), models)
    print("Test 2: unseen attack type")
    rows += test_unseen_attack_type(train, test, models)
    print("Test 3: scheme ablation")
    rows += test_scheme_ablation(train, test)
    print("Test 4: unseen popular websites")
    holdout = pd.read_csv(os.path.join(DATA_DIR, "popular_sites_holdout.csv"))
    rows += test_popular_sites(train, holdout, models)
    pd.DataFrame(rows).round(4).to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")
