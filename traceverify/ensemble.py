"""Stacking ensemble (ensemble learning): a meta Logistic Regression learns how much to trust each
member model (classical ML + deep network) from their held-out probability outputs."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config

STACK_PATH = config.ARTIFACT_DIR / "ensemble" / "stack.joblib"


class StackingEnsemble:
    def __init__(self, member_names: list[str]):
        self.member_names = list(member_names)
        self.meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED)

    def _stack(self, member_probs: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([np.asarray(member_probs[n]).reshape(-1, len(config.LABELS)) for n in self.member_names])

    def fit(self, member_probs: dict[str, np.ndarray], y_labels) -> "StackingEnsemble":
        self.meta.fit(self._stack(member_probs), list(y_labels))
        return self

    def predict_proba(self, member_probs: dict[str, np.ndarray]) -> np.ndarray:
        p = self.meta.predict_proba(self._stack(member_probs))
        classes = list(self.meta.classes_)
        return p[:, [classes.index(l) for l in config.LABELS]]

    def member_weights(self) -> dict[str, float]:
        """Share of the meta-learner's total absolute weight given to each member (sums to 1)."""
        w = np.abs(self.meta.coef_).sum(axis=0).reshape(len(self.member_names), len(config.LABELS)).sum(axis=1)
        w = w / (w.sum() + 1e-12)
        return {n: float(v) for n, v in zip(self.member_names, w)}

    def save(self, path: Path = STACK_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = STACK_PATH) -> "StackingEnsemble":
        return joblib.load(path)
