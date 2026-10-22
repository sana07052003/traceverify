"""Step 6: stacking ensemble. Meta-learner is fit on VALIDATION probabilities, scored on TEST."""
import json

import numpy as np

from traceverify import baseline, classical, config, data, features
from traceverify.ensemble import StackingEnsemble
from traceverify.mm_model import MultimodalVerifier
from traceverify.retrieval import Retriever


def member_probs(split, df, zoo, deep, retriever):
    probs = {n: classical.proba_matrix(p, df["claim"]) for n, p in zoo.items()}
    ev = features.evidence_for_frame(df, "retrieved", retriever)   # realistic: unseen-claim evidence
    img, txt, has = features.load_features(split, df)
    probs["deep_multimodal"] = deep.probs(df["claim"], ev, img, txt, has)
    return probs


def main():
    val, test = data.load_split("validation"), data.load_split("test")
    zoo, deep, retriever = classical.load(), MultimodalVerifier(config.DEEP_DIR), Retriever.load(config.RAG_DIR)
    pv, pt = member_probs("validation", val, zoo, deep, retriever), member_probs("test", test, zoo, deep, retriever)

    stack = StackingEnsemble(list(pv)).fit(pv, val["label"])
    stack.save()
    pe = stack.predict_proba(pt)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.REPORT_DIR / "preds_test.npz", claim_id=test["claim_id"].to_numpy().astype(str),
                        **{f"p__{n}": p for n, p in pt.items()}, p__stacking_ensemble=pe)
    pred = [config.LABELS[i] for i in pe.argmax(1)]
    report = {"ensemble_test": baseline.metrics(test["label"].tolist(), pred),
              "member_test": {n: baseline.metrics(test["label"].tolist(), [config.LABELS[i] for i in p.argmax(1)])
                              for n, p in pt.items()},
              "member_weights": stack.member_weights()}
    (config.REPORT_DIR / "ensemble.json").write_text(json.dumps(report, indent=2))
    print("ensemble test acc:", round(report["ensemble_test"]["accuracy"], 4),
          "| macro-F1:", round(report["ensemble_test"]["macro_f1"], 4))
    print("member weights:", {k: round(v, 3) for k, v in report["member_weights"].items()})


if __name__ == "__main__":
    main()
