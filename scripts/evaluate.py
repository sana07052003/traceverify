"""Step 7: retrieval Recall@k (dense vs hybrid) and deep-model evaluation incl. modality ablation."""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from traceverify import baseline, config, data, features
from traceverify.mm_model import MultimodalVerifier
from traceverify.retrieval import Retriever


def labels_from(p):
    return [config.LABELS[i] for i in p.argmax(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-claims", type=int, default=0, help="0 = whole test set")
    args = ap.parse_args()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    test = data.load_split("test")
    if args.max_claims:
        test = test.sample(min(args.max_claims, len(test)), random_state=config.SEED).reset_index(drop=True)
    y = test["label"].tolist()
    retriever = Retriever.load(config.RAG_DIR)

    # ---- retrieval (claim_id is str on both sides, so this no longer collapses to 0.000)
    retrieval = {"dense": retriever.recall_at_ks(test, hybrid=False),
                 "hybrid": retriever.recall_at_ks(test, hybrid=True), "n_claims": len(test)}
    (config.REPORT_DIR / "retrieval.json").write_text(json.dumps(retrieval, indent=2))
    print("retrieval:", json.dumps(retrieval))

    if not (config.DEEP_DIR / "head.pt").exists():
        print("no deep model found - skipping deep evaluation")
        return
    deep = MultimodalVerifier(config.DEEP_DIR)
    img, txt, has = features.load_features("test", test)
    out = {"n_test": len(test), "n_with_image": int(has.sum())}

    # ---- evidence settings
    probs_by_setting = {}
    for mode in ("gold", "retrieved_with_own", "retrieved"):
        ev = features.evidence_for_frame(test, mode, retriever)
        probs_by_setting[mode] = deep.probs(test["claim"], ev, img, txt, has)
    out["evidence_settings"] = {m: baseline.metrics(y, labels_from(p)) for m, p in probs_by_setting.items()}

    # ---- modality ablation on realistic evidence (own evidence excluded)
    ev = features.evidence_for_frame(test, "retrieved", retriever)
    full = probs_by_setting["retrieved"]
    text_only = deep.probs(test["claim"], ev, img, txt, np.zeros_like(has))
    mod = {"multimodal_all": baseline.metrics(y, labels_from(full)),
           "text_only_all": baseline.metrics(y, labels_from(text_only))}
    if has.any():
        sub = np.where(has)[0]
        ys = [y[i] for i in sub]
        mod["multimodal_image_subset"] = baseline.metrics(ys, labels_from(full[sub]))
        mod["text_only_image_subset"] = baseline.metrics(ys, labels_from(text_only[sub]))
        mod["prediction_change_rate_image_subset"] = float((full[sub].argmax(1) != text_only[sub].argmax(1)).mean())
        print(f"image changes the prediction for {mod['prediction_change_rate_image_subset']:.1%} of image claims")
        # image-only classical baseline: Logistic Regression on frozen CLIP image embeddings
        tr = data.load_split("train")
        t_img, _, t_has = features.load_features("train", tr)
        if t_has.sum() > 10:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                t_img[t_has], tr["label"][t_has])
            mod["image_only_logreg_image_subset"] = baseline.metrics(ys, clf.predict(img[sub]).tolist())
    else:
        mod["note"] = "No images were available; run scripts/extract_clip_features.py with --image-dir."
    out["modality"] = mod
    (config.REPORT_DIR / "deep.json").write_text(json.dumps(out, indent=2))
    for k, v in mod.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"{k:34s} acc {v['accuracy']:.4f} macro-F1 {v['macro_f1']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()
