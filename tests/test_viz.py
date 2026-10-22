"""Every figure function must produce a non-empty PNG (synthetic inputs, no models needed)."""
import numpy as np
import pandas as pd

from traceverify import config, viz


def _ok(p):
    assert p.exists() and p.stat().st_size > 5000


def test_all_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    cm = [[60, 20, 20], [15, 110, 25], [25, 30, 95]]
    rows = [{"name": n, "family": f, "accuracy": a, "macro_f1": a - .02, "per_class_f1": {l: a - .05 + .05 * i for i, l in enumerate(config.LABELS)}, "confusion_matrix": cm}
            for n, f, a in (("Logistic Regression", "Classical ML", .52), ("Deep multimodal network", "Deep learning", .61), ("Stacking ensemble", "Ensemble", .64))]
    _ok(viz.model_comparison(rows, tmp_path / "a.png"))
    _ok(viz.per_class_f1(rows, tmp_path / "b.png"))
    _ok(viz.confusion_matrix_fig(cm, "x", tmp_path / "c.png"))
    _ok(viz.confusion_grid({"m1": cm, "m2": cm, "m3": cm, "m4": cm}, tmp_path / "d.png"))
    y = rng.integers(0, 3, 300)
    p = rng.dirichlet([1, 1, 1], 300)
    aucs, best = viz.roc_curves(y, {"A": p, "B": rng.dirichlet([1, 1, 1], 300)}, tmp_path)
    assert best in aucs and (tmp_path / "07_roc_curve_best_model.png").exists()
    _ok(viz.confidence_hist(p, y, "A", tmp_path / "e.png"))
    _ok(viz.top_confusions([config.LABELS[i] for i in y], [config.LABELS[i] for i in p.argmax(1)], "A", tmp_path / "f.png"))
    g = {c: [[f"word{i}", .1 / (i + 1)] for i in range(15)] for c in config.LABELS}
    _ok(viz.shap_global(g, tmp_path / "g.png"))
    _ok(viz.retrieval_recall({"dense": {"1": .3, "5": .5}, "hybrid": {"1": .3, "5": .45}}, tmp_path / "h.png"))
    _ok(viz.training_curves([{"epoch": e, "train_loss": 1 / e, "val": {"accuracy": .5, "macro_f1": .5}} for e in (1, 2, 3)], tmp_path / "i.png"))
    _ok(viz.grouped_metrics({"gold": {"accuracy": .8, "macro_f1": .78}, "retrieved": {"accuracy": .6, "macro_f1": .58}}, "t", tmp_path / "j.png"))
    _ok(viz.modality_contribution({"all_inputs": .6, "text_only": .55, "image_only": .4, "n": 100}, tmp_path / "k.png"))
    _ok(viz.ensemble_weights({"a": .3, "b": .7}, {"a": "A", "b": "B"}, tmp_path / "l.png"))
    _ok(viz.architecture(tmp_path / "m.png"))
    frames = {s: pd.DataFrame({"label": rng.choice(config.LABELS, 50), "image": [["x.jpg"] if i % 2 else [] for i in range(50)],
                               "text_evidence": [["a"] * int(rng.integers(1, 5)) for _ in range(50)]}) for s in ("train", "validation", "test")}
    _ok(viz.dataset_overview(frames, tmp_path / "n.png"))
    res = {"claim": "Photographs show baby copperhead snakes discovered in a potted plant, according to a viral post.",
           "verdict": {"label": "refuted", "confidence": .71, "decided_by": "stacking_ensemble"},
           "evidence": [{"text": "Snopes found the photographs were unrelated to any plant." * 3, "score": .62}],
           "xai": {"shap_terms": [("snakes", .2), ("photographs", -.1), ("plant", .05)], "occlusion_terms": [("copperhead", .3), ("viral", -.1)],
                   "modality_ablation": {"all_inputs": .71, "text_only": .6, "image_only": .3}},
           "clip_similarity": .29, "clip_band": "strong match"}
    _ok(viz.example_panel(res, tmp_path / "o.png", image=np.zeros((60, 80, 3), dtype="uint8") + 90))
