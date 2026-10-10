"""
model_builders.py
-----------------
Defines every classifier compared in the project as a scikit-learn Pipeline.

Every pipeline takes the SAME input: a DataFrame with a `url` column plus all
FEATURE_COLUMNS. A ColumnTransformer at the start of each pipeline picks what
that model needs:
    - feature-based models use the 19 engineered features
    - the character n-gram model reads the canonical URL text itself
This lets train_classifier.py, evaluate_model.py and predict.py treat every
model the same way, and guarantees a fair comparison on one test set.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from feature_extraction import FEATURE_COLUMNS, canonicalise

RANDOM_STATE = 42


def canonical_urls(urls):
    """Turn the `url` column (a pandas Series) into a list of canonical URLs."""
    return [canonicalise(u) for u in urls]


def _features(scale=False):
    """Select the engineered features, optionally standardised."""
    step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("features", step, FEATURE_COLUMNS)], remainder="drop")


def build_models():
    """Return {name: unfitted Pipeline} for every model in the comparison."""
    return {
        # --- Unit algorithms -------------------------------------------------
        # Linear baseline; needs scaling because features have very different ranges
        "LogisticRegression": Pipeline([
            ("select", _features(scale=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        # One shallow, fully readable tree
        "DecisionTree": Pipeline([
            ("select", _features()),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        # Average of 100 trees; min_samples_leaf keeps the saved file small
        "RandomForest": Pipeline([
            ("select", _features()),
            ("model", RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                             n_jobs=-1, random_state=RANDOM_STATE)),
        ]),

        # --- Beyond the unit -------------------------------------------------
        # Gradient boosting with histogram binning (the LightGBM design,
        # Ke et al. 2017): trees are built in sequence, each correcting the
        # previous ones' errors, so it keeps focusing on the hard URLs.
        "GradientBoosting": Pipeline([
            ("select", _features()),
            ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                     random_state=RANDOM_STATE)),
        ]),
        # Reads the URL text itself instead of our features: overlapping 3- to
        # 5-character fragments, TF-IDF weighted, fed to a linear classifier
        # (Ma et al. 2009; Le et al. 2018). Hashing keeps memory bounded.
        "CharNgram": Pipeline([
            ("select", ColumnTransformer([
                ("text", Pipeline([
                    ("canonical", FunctionTransformer(canonical_urls)),
                    ("ngrams", HashingVectorizer(analyzer="char", ngram_range=(3, 5),
                                                 n_features=2 ** 20, alternate_sign=False)),
                ]), "url")], remainder="drop")),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("model", SGDClassifier(loss="modified_huber", alpha=1e-6,
                                    max_iter=20, tol=None, random_state=RANDOM_STATE)),
        ]),
    }


# Models taught in the unit vs. models added beyond it (used in output tables)
UNIT_MODELS = ["LogisticRegression", "DecisionTree", "RandomForest"]
BEYOND_UNIT_MODELS = ["GradientBoosting", "CharNgram"]
