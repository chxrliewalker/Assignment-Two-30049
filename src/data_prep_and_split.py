"""
data_prep_and_split.py
----------------------
Builds the processed datasets from two raw sources.

Sources (datasets/raw/):
    malicious_phish.csv      base dataset (Kaggle, Siddhartha 2021): url, type
    alexa_top30k_2016.csv    additional dataset: the 30,000 most-visited
                             websites, Alexa ranking 2016: rank, domain

Why the additional source: in the base dataset only 277 benign URLs are a bare
domain (e.g. "google.com"), against 21,677 phishing ones, so a model learns
"no path = phishing" and flags google.com. Popular websites from the same year
add the missing legitimate bare domains.

Steps:
    1. Load the base dataset; drop empty rows and malformed strings
    2. Map the four text labels to a binary target
           benign -> 0 ; phishing, defacement, malware -> 1
       `type` is kept for checks (cluster composition, held-out tests) but is
       never a model feature
    3. Transform the additional source into the base format:
           domain -> url, type = "benign", label = 0, source = "alexa"
       ranks 1-20,000 join the dataset; ranks 20,001-30,000 are kept aside
       as an unseen "popular sites" test set
    4. Remove duplicates across both sources, exact and after
       canonicalisation ("http://www.x.com" and "x.com" are the same URL);
       a canonical URL labelled both benign and malicious is dropped
    5. Extract features (feature_extraction.py)
    6. Stratified 70/30 train/test split, random_state=42
    7. Save train_features.csv, test_features.csv and popular_sites_holdout.csv

Usage (from the project root):
    python src/data_prep_and_split.py
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from feature_extraction import (FEATURE_COLUMNS, canonicalise,  # noqa: E402
                                extract_url_features, safe_url_parse)

RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")
BASE_PATH = os.path.join(RAW_DIR, "malicious_phish.csv")
ALEXA_PATH = os.path.join(RAW_DIR, "alexa_top30k_2016.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")

LABEL_MAP = {"benign": 0, "phishing": 1, "defacement": 1, "malware": 1}
ALEXA_TRAIN_RANKS = 20000      # ranks 1..20,000 are added to the dataset
TEST_SIZE = 0.30
RANDOM_STATE = 42
OUT_COLUMNS = ["url", "type", "label", "source"] + FEATURE_COLUMNS


def has_valid_host(url):
    """A usable URL needs a host containing a dot (a domain or an IP address)
    and no control characters. A few raw rows are corrupted binary text."""
    host = safe_url_parse(canonicalise(url)).netloc
    return "." in host and not any(ord(ch) < 32 or ord(ch) == 127 for ch in host)


def load_base(path=BASE_PATH):
    """Load and clean the Kaggle base dataset."""
    print(f"Base dataset: {path}")
    df = pd.read_csv(path)
    print(f"  raw rows:                          {len(df):,}")
    df = df.dropna(subset=["url", "type"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df = df[df["url"] != ""]
    valid = df["url"].map(has_valid_host)
    print(f"  malformed (no valid host) removed: {int((~valid).sum()):,}")
    df = df[valid]

    unknown = set(df["type"]) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected label values: {unknown}")
    df["label"] = df["type"].map(LABEL_MAP).astype(int)
    df["source"] = "kaggle"
    return df[["url", "type", "label", "source"]]


def load_alexa(path=ALEXA_PATH):
    """Bring the Alexa list into the base dataset's shape.
    Returns (rows added to the dataset, rows kept aside for testing)."""
    alexa = pd.read_csv(path)
    alexa = alexa.rename(columns={"domain": "url"})
    alexa["url"] = alexa["url"].astype(str).str.strip().str.lower()
    alexa["type"], alexa["label"], alexa["source"] = "benign", 0, "alexa"
    cols = ["url", "type", "label", "source"]
    add = alexa[alexa["rank"] <= ALEXA_TRAIN_RANKS][cols]
    holdout = alexa[alexa["rank"] > ALEXA_TRAIN_RANKS][cols]
    print(f"Additional dataset: {path}")
    print(f"  added to dataset: {len(add):,}   kept aside for testing: {len(holdout):,}")
    return add, holdout


def deduplicate(df):
    """Remove exact duplicates, then duplicates after canonicalisation.
    A canonical URL that carries both labels is ambiguous: all copies go."""
    before = len(df)
    df = df.drop_duplicates(subset="url", keep="first")
    print(f"  exact duplicates removed:          {before - len(df):,}")

    df = df.assign(canonical=df["url"].map(canonicalise))
    labels = df.groupby("canonical")["label"].nunique()
    conflicting = labels[labels > 1].index
    in_conflict = df["canonical"].isin(conflicting)
    print(f"  rows with conflicting labels:      {int(in_conflict.sum()):,} "
          f"({len(conflicting):,} URLs), all dropped")
    df = df[~in_conflict]
    before = len(df)
    df = df.drop_duplicates(subset="canonical", keep="first")
    print(f"  canonical duplicates removed:      {before - len(df):,}")
    return df.drop(columns="canonical").reset_index(drop=True)


def main():
    base = load_base()
    alexa_add, alexa_holdout = load_alexa()

    print("Combining and de-duplicating:")
    data = deduplicate(pd.concat([base, alexa_add], ignore_index=True))
    print(f"  final rows:                        {len(data):,}")
    print("\nRows per source and type:")
    print(data.groupby(["source", "type"]).size().to_string())

    # The held-out popular sites must not overlap the main dataset
    seen = set(data["url"].map(canonicalise))
    alexa_holdout = alexa_holdout[~alexa_holdout["url"].map(canonicalise).isin(seen)]

    print(f"\nExtracting {len(FEATURE_COLUMNS)} features...")
    full = extract_url_features(data)[OUT_COLUMNS]
    holdout = extract_url_features(alexa_holdout)[OUT_COLUMNS]

    train, test = train_test_split(full, test_size=TEST_SIZE,
                                   stratify=full["label"], random_state=RANDOM_STATE)
    print(f"Train: {len(train):,}   Test: {len(test):,}   Popular-sites holdout: {len(holdout):,}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, part in (("train_features.csv", train), ("test_features.csv", test),
                       ("popular_sites_holdout.csv", holdout)):
        part.to_csv(os.path.join(OUT_DIR, name), index=False)
        print(f"Saved datasets/processed/{name}")


if __name__ == "__main__":
    main()
