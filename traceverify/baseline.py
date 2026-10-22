"""Classical text-only baseline: TF-IDF + Logistic Regression (the paper's original classifier)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from . import config


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_features=50000)),
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=config.SEED)),
    ])


def metrics(y_true, y_pred, labels=config.LABELS) -> dict:
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {l: float(v) for l, v in zip(labels, per_class)},
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save(pipe: Pipeline, train_texts, path: Path = config.BASELINE_PATH) -> None:
    mean = np.asarray(pipe.named_steps["tfidf"].transform(list(train_texts)).mean(axis=0)).ravel()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "tfidf_mean": mean}, path)


def load(path: Path = config.BASELINE_PATH) -> dict:
    return joblib.load(path)


def predict(bundle: dict, claim: str) -> dict:
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba([claim])[0]
    classes = list(pipe.named_steps["logistic_regression"].classes_)
    k = int(np.argmax(proba))
    return {"label": classes[k], "confidence": float(proba[k]),
            "probs": {c: float(p) for c, p in zip(classes, proba)}}
