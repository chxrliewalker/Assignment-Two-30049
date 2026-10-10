# Phishing URL Detection (COS30049 Assignment 2, Session 13 Group 6)

Paste a web address and get a trust score plus plain-language warnings, without the
page ever being visited. This repository holds the data processing, feature
extraction, model training, clustering and evaluation code.

## 1. Project structure

```
datasets/
  raw/
    malicious_phish.csv          base dataset (Kaggle, 651,191 URLs)
    alexa_top30k_2016.csv        additional dataset (30,000 most-visited websites, 2016)
  processed/
    train_features.csv           70% training split: url, type, label, source + 19 features
    test_features.csv            30% test split, same columns
    popular_sites_holdout.csv    10,000 unseen popular websites (Alexa ranks 20,001-30,000)
models/
  phishing_pipeline.joblib       final model (character n-gram classifier)
  model_comparison.csv           all classifiers on the same test set
  test_predictions.csv           final model's prediction for every test URL
  stress_tests.csv               harder evaluations (unseen domains, unseen attack type, ...)
  cluster_summary.csv            k-means cluster profiles (malicious URLs only)
  cluster_examples.csv           example URLs per cluster
  plots/                         confusion matrices, feature charts, cluster plots
src/
  feature_extraction.py          URL canonicalisation + 19 features (shared by every script)
  data_prep_and_split.py         cleaning, label mapping, merging sources, de-duplication, split
  data_analysis.py               exploratory charts for the report
  model_builders.py              the five classifiers as scikit-learn pipelines
  train_classifier.py            trains and compares the classifiers, saves the best
  evaluate_model.py              stress tests
  train_clustering.py            k-means within the malicious class
  predict.py                     verdict for a single URL
environment.yml                  conda environment
```

## 2. Set up the environment

With conda (recommended):

```
conda env create -f environment.yml
conda activate phishing-url
```

Without conda (Python 3.10 or newer):

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

On Windows, use `py` instead of `python` in the commands below if `python` is not recognised.
Run every command from the project root folder.

## 3. Process the data (optional: the processed files are already included)

```
python src/data_prep_and_split.py
```

This reads both raw files and:
1. drops empty rows and 86 corrupted strings with no valid host
2. maps labels: `benign` -> 0; `phishing`, `defacement`, `malware` -> 1 (the original `type` is kept for checks only)
3. adds Alexa ranks 1-20,000 as legitimate URLs; ranks 20,001-30,000 become a separate test set
4. removes duplicates, both exact and after canonicalisation (`http://www.x.com` = `x.com`), and drops URLs labelled both benign and malicious
5. extracts the 19 features and makes a stratified 70/30 split (`random_state=42`)

Takes about 1 minute.

## 4. Explore the data

```
python src/data_analysis.py
```

Saves the class balance, scheme artefact, feature box plots and correlation heatmap to `models/plots/`.

## 5. Train the classifiers

```
python src/train_classifier.py
```

Trains five models on the same split and prints accuracy, precision, recall, F1 and
false-positive rate for each:

| Model | In unit? | Input |
|---|---|---|
| Logistic regression | yes | 19 features (standardised) |
| Decision tree (max depth 5) | yes | 19 features |
| Random forest (100 trees) | yes | 19 features |
| Gradient boosting (`HistGradientBoostingClassifier`, 300 iterations) | no | 19 features |
| Character n-gram (3-5 character fragments, TF-IDF, linear classifier) | no | URL text |

The model with the best F1 is saved as `models/phishing_pipeline.joblib`. Takes about 3 minutes.

## 6. Run the stress tests

```
python src/evaluate_model.py
```

Four harder tests on random forest and the character n-gram model: unseen websites
(split by domain), an unseen attack type (phishing removed from training), the URL
scheme artefact, and 10,000 unseen popular websites. Results go to
`models/stress_tests.csv`. Takes about 10 minutes.

## 7. Cluster the malicious URLs

```
python src/train_clustering.py
```

Runs k-means on malicious URLs only, without labels. Chooses k from silhouette
scores (k = 2 to 8), then saves each cluster's profile, its composition by original
type, and example URLs. Takes under a minute.

## 8. Predict a single URL

```
python src/predict.py --url "http://paypal.com.secure-login.xyz/verify"
```

Example output:

```
Verdict:         PHISHING / MALICIOUS
Risk Score:      100.0% (Phishing Likelihood)
Trust Score:     0.0% (Safety Rating)

Diagnostic Risk Flags:
 - [!] Suspicious Security Keywords Detected (3)
 - [!] High Domain Randomness / Entropy (4.06)
 - [!] Frequently Abused Top-Level Domain
 - [!] Brand Name Used Outside the Real Domain
```

A different model file can be passed with `--model path/to/model.joblib`.
No network request is ever made: the URL is only parsed as text.

## Data sources

- Siddhartha, M. (2021) *Malicious URLs dataset*, Kaggle. https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
- Alexa top 1 million websites (2016), via https://github.com/zer0h/top-1000000-domains
