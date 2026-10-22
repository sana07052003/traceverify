"""Generate every figure for the paper / viva as clear PNGs -> artifacts/figures/ (+ INDEX.md with captions).

Run after training. Missing pieces are skipped with a message, so it also works with only the classical models:
    python -m scripts.make_figures
    python -m scripts.make_figures --no-examples        # skip the qualitative example panels (no model loading)
"""
import argparse
import json
import traceback

import numpy as np
import pandas as pd

from traceverify import classical, config, data, images, viz
from traceverify.baseline import metrics as compute_metrics

FIG = config.ARTIFACT_DIR / "figures"
CAPTIONS: dict[str, str] = {}


def done(name: str, caption: str) -> None:
    CAPTIONS[name] = caption
    print("  created", name)


def attempt(label, fn):
    try:
        fn()
    except Exception as exc:                       # keep going; report what was skipped and why
        print(f"  SKIPPED {label}: {type(exc).__name__}: {exc}")
        if "--debug" in __import__("sys").argv:
            traceback.print_exc()


def load_reports():
    r = {}
    for p in config.REPORT_DIR.glob("*.json"):
        r[p.stem] = json.loads(p.read_text())
    tr = config.DEEP_DIR / "training.json"
    if tr.exists():
        r["training"] = json.loads(tr.read_text())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max-modality-claims", type=int, default=1500)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    R = load_reports()
    frames = {s: data.load_split(s) for s in ("train", "validation", "test")}
    test = frames["test"]
    y_idx = test["label"].map(config.LABEL2ID).to_numpy()
    print("Generating figures into", FIG)

    attempt("architecture", lambda: (viz.architecture(FIG / "01_architecture.png"),
            done("01_architecture.png", "Overall architecture of TraceVerify: retrieval, five classifiers, stacking ensemble, XAI and RAG explanation.")))
    attempt("dataset", lambda: (viz.dataset_overview(frames, FIG / "02_dataset_overview.png"),
            done("02_dataset_overview.png", "MOCHEG (official splits): label balance, availability of image evidence, and evidence passages per claim.")))

    # ---- model table (classical + deep + ensemble) from the reports
    rows = []
    for n, r in R.get("classical", {}).items():
        t = r["test"]
        rows.append({"name": classical.NAMES.get(n, n), "family": "Classical ML", **t})
    ens = R.get("ensemble", {})
    if ens.get("member_test", {}).get("deep_multimodal"):
        rows.append({"name": classical.NAMES["deep_multimodal"], "family": "Deep learning", **ens["member_test"]["deep_multimodal"]})
    if ens.get("ensemble_test"):
        rows.append({"name": classical.NAMES["stacking_ensemble"], "family": "Ensemble", **ens["ensemble_test"]})
    if rows:
        attempt("model comparison", lambda: (viz.model_comparison(rows, FIG / "03_model_comparison.png"),
                done("03_model_comparison.png", "Test accuracy and macro-F1 of every algorithm (classical ML, deep learning, stacking ensemble).")))
        attempt("per-class f1", lambda: (viz.per_class_f1(rows, FIG / "04_per_class_f1.png"),
                done("04_per_class_f1.png", "Per-class F1 for Supported, Refuted and Not-enough-info.")))
        cms = {r["name"]: r["confusion_matrix"] for r in rows}
        attempt("confusion grid", lambda: (viz.confusion_grid(cms, FIG / "05_confusion_matrices_all.png"),
                done("05_confusion_matrices_all.png", "Row-normalised confusion matrices of all models on the test split.")))
        best = max(rows, key=lambda r: r["macro_f1"])
        attempt("confusion best", lambda: (viz.confusion_matrix_fig(best["confusion_matrix"], best["name"], FIG / "06_confusion_matrix_best.png"),
                done("06_confusion_matrix_best.png", f"Confusion matrix of the best model by macro-F1 ({best['name']}): counts and per-class recall.")))

    # ---- probabilities for ROC / confidence / error analysis
    probs = {}
    preds = config.REPORT_DIR / "preds_test.npz"
    if preds.exists():
        z = np.load(preds, allow_pickle=False)
        if list(z["claim_id"].astype(str)) == test["claim_id"].tolist():
            probs = {k[3:]: z[k] for k in z.files if k.startswith("p__")}
    if not probs and (config.ARTIFACT_DIR / "classical" / "models.joblib").exists():
        zoo = classical.load()
        probs = {n: classical.proba_matrix(p, test["claim"]) for n, p in zoo.items()}
    if probs:
        disp = {n: classical.NAMES.get(n, n) for n in probs}
        P = {disp[n]: p for n, p in probs.items()}

        def roc():
            aucs, best_name = viz.roc_curves(y_idx, P, FIG)
            pd.DataFrame(aucs).T.rename_axis("model").to_csv(FIG / "auc_table.csv")
            done("07_roc_curve_best_model.png", f"One-vs-rest ROC curves per class for the best model by AUC ({best_name}).")
            done("07b_roc_curves_all_models.png", "Micro-average ROC curves and AUC of every model.")
        attempt("ROC", roc)

        def err():
            name = max(P, key=lambda n: float((P[n].argmax(1) == y_idx).mean()))
            p = P[name]
            pred = p.argmax(1)
            viz.confidence_hist(p, y_idx, name, FIG / "08_confidence_distribution.png")
            done("08_confidence_distribution.png", f"Confidence of correct versus wrong predictions ({name}).")
            viz.top_confusions([config.LABELS[i] for i in y_idx], [config.LABELS[i] for i in pred], name, FIG / "09_top_errors.png")
            done("09_top_errors.png", f"Most frequent error types (true label to predicted label), {name}.")
            wrong = np.where(pred != y_idx)[0]
            pd.DataFrame({"claim": test["claim"].to_numpy()[wrong], "true": [config.LABELS[i] for i in y_idx[wrong]],
                          "predicted": [config.LABELS[i] for i in pred[wrong]], "confidence": p.max(1)[wrong].round(3)}
                         ).sort_values("confidence", ascending=False).to_csv(FIG / "error_analysis_misclassified.csv", index=False)
            done("error_analysis_misclassified.csv", "All misclassified test claims, most confidently wrong first (for the error-analysis discussion).")
        attempt("error analysis", err)

    if R.get("shap_global"):
        attempt("SHAP global", lambda: (viz.shap_global(R["shap_global"], FIG / "10_shap_global_importance.png"),
                done("10_shap_global_importance.png", "Global word importance (mean absolute SHAP value) per class for the Logistic Regression model.")))
    if R.get("retrieval"):
        attempt("retrieval", lambda: (viz.retrieval_recall(R["retrieval"], FIG / "11_retrieval_recall.png"),
                done("11_retrieval_recall.png", "Recall@k of dense versus hybrid (dense + BM25) evidence retrieval on the test split.")))
    if R.get("training", {}).get("history"):
        attempt("training curves", lambda: (viz.training_curves(R["training"]["history"], FIG / "12_deep_training_curves.png"),
                done("12_deep_training_curves.png", "Training loss and validation accuracy / macro-F1 per epoch of the deep multimodal network.")))
    d = R.get("deep", {})
    if d.get("evidence_settings"):
        attempt("evidence settings", lambda: (viz.grouped_metrics(d["evidence_settings"], "Deep model by evidence source (test)", FIG / "13_evidence_source.png"),
                done("13_evidence_source.png", "Deep model with annotated (gold) evidence, retrieved evidence, and retrieved evidence with the claim's own fact-check removed (realistic).")))
    if d.get("modality"):
        mod = {k: v for k, v in d["modality"].items() if isinstance(v, dict) and "accuracy" in v}
        attempt("modality comparison", lambda: (viz.grouped_metrics(mod, "Does the image help? Multimodal vs text-only vs image-only (test)", FIG / "14_modality_comparison.png"),
                done("14_modality_comparison.png", "Multimodal versus text-only versus image-only models, overall and on the subset of claims that have an image.")))
    if ens.get("member_weights"):
        attempt("ensemble weights", lambda: (viz.ensemble_weights(ens["member_weights"], classical.NAMES, FIG / "15_ensemble_weights.png"),
                done("15_ensemble_weights.png", "Share of the stacking meta-learner's weight assigned to each member model.")))

    # ---- model-dependent figures (need the deep model + artifacts)
    if (config.DEEP_DIR / "head.pt").exists() and not args.no_examples:
        def modality_dataset():
            from traceverify import features
            from traceverify.mm_model import MultimodalVerifier
            from traceverify.retrieval import Retriever
            deep = MultimodalVerifier(config.DEEP_DIR)
            img, txt, has = features.load_features("test", test)
            sel = np.where(has)[0][: args.max_modality_claims]
            if len(sel) == 0:
                raise RuntimeError("no test claims with images (run extract_clip_features with images)")
            sub = test.iloc[sel].reset_index(drop=True)
            ev = features.evidence_for_frame(sub, "retrieved", Retriever.load(config.RAG_DIR))
            k = y_idx[sel]
            args_ = (sub["claim"], ev, img[sel], txt[sel])
            full = deep.probs(*args_, np.ones(len(sel), bool))[np.arange(len(sel)), k].mean()
            noimg = deep.probs(*args_, np.zeros(len(sel), bool))[np.arange(len(sel)), k].mean()
            notext = deep.probs([""] * len(sel), [""] * len(sel), img[sel], txt[sel], np.ones(len(sel), bool))[np.arange(len(sel)), k].mean()
            res = {"all_inputs": float(full), "text_only": float(noimg), "image_only": float(notext), "n": int(len(sel))}
            (config.REPORT_DIR / "modality_dataset.json").write_text(json.dumps(res, indent=2))
            viz.modality_contribution(res, FIG / "16_modality_contribution.png")
            done("16_modality_contribution.png", "Dataset-level XAI: mean probability of the true label with all inputs, image masked, and text blanked; and each modality's contribution.")
        attempt("modality contribution", modality_dataset)

        def examples():
            from traceverify.pipeline import TraceVerify
            system = TraceVerify()
            idx = images.index_images(config.IMAGE_DIR)
            for l in config.LABELS:
                row = test[test["label"] == l].sample(1, random_state=7).iloc[0]
                res = system.verify(row["claim"], use_llm=False)
                viz.example_panel(res, FIG / f"17_example_{l}.png")
                done(f"17_example_{l}.png", f"Worked example (true label: {config.DISPLAY[l]}): verdict, SHAP, occlusion, modality ablation and retrieved evidence.")
            for _, row in test.iterrows():
                path = next((idx[str(n).lower()] for n in row["image"] if str(n).lower() in idx), None)
                pil = images.open_image(path) if path else None
                if pil is not None:
                    res = system.verify(row["claim"], image=pil, use_llm=False)
                    viz.example_panel(res, FIG / "18_example_text_plus_image.png", image=pil)
                    done("18_example_text_plus_image.png", "Worked example with text and image: adds CLIP image-claim similarity and the image's contribution.")
                    break
        attempt("example panels", examples)

    lines = ["# Figure index", "", "Every PNG is 200 dpi, white background. Suggested captions:", ""]
    lines += [f"- **{k}**: {v}" for k, v in sorted(CAPTIONS.items())]
    (FIG / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(CAPTIONS)} files written. Index: {FIG / 'INDEX.md'}")


if __name__ == "__main__":
    main()
