"""Word-level SHAP attributions for the linear (TF-IDF + LR) model.

For a linear model with independent features the SHAP values have the closed form
    phi_i = w_i * (x_i - E[x_i])
which is what shap.LinearExplainer returns, without extra dependencies, background sub-sampling or
slowness. (tests/test_core.py cross-checks it against `shap` when that package is installed.)
"""
from __future__ import annotations

import numpy as np


class LinearExplainer:
    def __init__(self, bundle: dict):
        self.tfidf = bundle["pipeline"].named_steps["tfidf"]
        self.lr = bundle["pipeline"].named_steps["logistic_regression"]
        self.mean = np.asarray(bundle["tfidf_mean"]).ravel()
        self.features = self.tfidf.get_feature_names_out()
        self.classes = list(self.lr.classes_)

    def local(self, claim: str, label: str, top_n: int = 8) -> list[tuple[str, float]]:
        x = self.tfidf.transform([claim])
        present = x.nonzero()[1]
        if len(present) == 0:
            return []
        k = self.classes.index(label)
        xd = x.toarray().ravel()
        phi = self.lr.coef_[k][present] * (xd[present] - self.mean[present])
        pairs = sorted(zip(self.features[present], phi.astype(float)), key=lambda p: abs(p[1]), reverse=True)
        return [(str(t), float(v)) for t, v in pairs[:top_n]]

    def global_importance(self, texts, top_n: int = 20, chunk: int = 500) -> dict[str, list[tuple[str, float]]]:
        X = self.tfidf.transform(list(texts))
        out = {}
        for k, cls in enumerate(self.classes):
            acc = np.zeros(X.shape[1], dtype=np.float64)
            for s in range(0, X.shape[0], chunk):
                block = X[s:s + chunk].toarray().astype(np.float32)
                acc += np.abs(self.lr.coef_[k] * (block - self.mean)).sum(axis=0)
            acc /= X.shape[0]
            top = np.argsort(-acc)[:top_n]
            out[cls] = [(str(self.features[i]), float(acc[i])) for i in top]
        return out
