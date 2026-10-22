"""Step 2: classical ML zoo (LogReg, Linear SVM, Naive Bayes, Random Forest) + SHAP statistics.

Each model is trained on (a) all training claims and (b) only claims that have evidence; the better variant on the
VALIDATION split is kept (see classical.select_by_validation). Both scores are saved for the paper."""
import json

from traceverify import baseline, classical, config, data
from traceverify.explain import LinearExplainer


def main():
    train, val, test = (data.load_split(s) for s in ("train", "validation", "test"))
    n_no_ev = int((train["text_evidence"].apply(len) == 0).sum())
    print(f"train claims: {len(train)} (without evidence: {n_no_ev}); label counts without evidence: "
          f"{train[train['text_evidence'].apply(len) == 0]['label'].value_counts().to_dict()}")
    models, selection, subsets = classical.select_by_validation(train, val)
    classical.save(models)

    report = {}
    for name, pipe in models.items():
        P = {}
        for split, df in (("val", val), ("test", test)):
            pred = [config.LABELS[i] for i in classical.proba_matrix(pipe, df["claim"]).argmax(1)]
            P[split] = baseline.metrics(df["label"].tolist(), pred)
        P["train_subset"] = selection[name]["chosen"]
        report[name] = P
        sel = selection[name]
        print(f"{name:20s} val macro-F1 all={sel['all']:.4f}" +
              (f" with_evidence={sel['with_evidence']:.4f}" if "with_evidence" in sel else "") +
              f" -> {sel['chosen']} | test acc {P['test']['accuracy']:.4f} macro-F1 {P['test']['macro_f1']:.4f}")

    lr_train = subsets[selection["logistic_regression"]["chosen"]]
    baseline.save(models["logistic_regression"], lr_train["claim"])
    glob = LinearExplainer(baseline.load()).global_importance(val["claim"])
    votes = [v["chosen"] for v in selection.values()]
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "classical.json").write_text(json.dumps(report, indent=2))
    (config.REPORT_DIR / "shap_global.json").write_text(json.dumps(glob, indent=2))
    (config.REPORT_DIR / "subset_selection.json").write_text(json.dumps(
        {"per_model": selection, "majority": max(set(votes), key=votes.count), "train_claims_without_evidence": n_no_ev}, indent=2))
    print("saved models, SHAP stats, subset_selection.json and reports")


if __name__ == "__main__":
    main()
