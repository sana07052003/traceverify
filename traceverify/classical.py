"""Classical ML model zoo on TF-IDF text features (non-deep-learning algorithms)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from . import baseline, config

NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM (calibrated)",
    "naive_bayes": "Complement Naive Bayes",
    "random_forest": "Random Forest",
    "deep_multimodal": "Deep multimodal network",
    "stacking_ensemble": "Stacking ensemble",
}
MODELS_PATH = config.ARTIFACT_DIR / "classical" / "models.joblib"


def _tfidf(max_features: int = 50000) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                           min_df=2, max_features=max_features)


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": baseline.build_pipeline(),
        "linear_svm": Pipeline([("tfidf", _tfidf()),
                                ("clf", CalibratedClassifierCV(LinearSVC(), cv=3))]),   # balanced via sample_weight in fit_all
        "naive_bayes": Pipeline([("tfidf", _tfidf()), ("clf", ComplementNB(alpha=0.5))]),
        "random_forest": Pipeline([("tfidf", _tfidf(20000)),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                  class_weight="balanced_subsample",
                                                                  n_jobs=-1, random_state=config.SEED))]),
    }


def fit_all(models: dict[str, Pipeline], texts, labels) -> dict[str, Pipeline]:
    texts, labels = list(texts), list(labels)
    weights = compute_sample_weight("balanced", labels)
    for name, pipe in models.items():
        print(f"  fitting {name} ...")
        if name == "linear_svm":
            # Platt calibration ignores class_weight and drifts toward the majority class on imbalanced data
            # (all-'refuted' predictions); balanced sample weights keep base fit AND calibration balanced.
            pipe.fit(texts, labels, clf__sample_weight=weights)
        else:
            pipe.fit(texts, labels)
    return models


def proba_matrix(pipe: Pipeline, texts) -> np.ndarray:
    """Probabilities in config.LABELS column order."""
    p = pipe.predict_proba(list(texts))
    classes = list(pipe.classes_)
    return p[:, [classes.index(l) for l in config.LABELS]]


def select_by_validation(train, val, make_models=None):
    """Fit every model on (a) all training claims and (b) only claims that have evidence, then keep the
    better variant per model by VALIDATION macro-F1 (never the test set).

    Motivation: in MOCHEG's training split many claims have no evidence and those are never 'refuted',
    while validation/test claims all have evidence -> a train/test population mismatch worth testing."""
    make_models = make_models or build_models
    subsets = {"all": train}
    with_ev = train[train["text_evidence"].apply(len) > 0]
    if 0 < len(with_ev) < len(train):
        subsets["with_evidence"] = with_ev
    best, log = {}, {}
    for sname, sub in subsets.items():
        print(f"[{sname}] training on {len(sub)} claims")
        models = fit_all(make_models(), sub["claim"], sub["label"])
        for n, pipe in models.items():
            pred = [config.LABELS[i] for i in proba_matrix(pipe, val["claim"]).argmax(1)]
            f1 = baseline.metrics(val["label"].tolist(), pred)["macro_f1"]
            log.setdefault(n, {})[sname] = f1
            if n not in best or f1 > best[n][0]:
                best[n] = (f1, sname, pipe)
    for n in log:
        log[n]["chosen"] = best[n][1]
    return {n: v[2] for n, v in best.items()}, log, subsets


def save(models: dict, path: Path = MODELS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)


def load(path: Path = MODELS_PATH) -> dict:
    return joblib.load(path)
