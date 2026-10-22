"""Step 5: train the deep multimodal network (DistilBERT + CLIP fusion). GPU strongly recommended."""
import argparse
import json

from traceverify import config, data, features
from traceverify.mm_model import train
from traceverify.retrieval import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", choices=["gold", "retrieved"], default="retrieved",
                    help="'retrieved' trains on RAG evidence (own evidence excluded) - matches deployment")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5, help="encoder learning rate")
    ap.add_argument("--head-lr", type=float, default=5e-4, help="fusion-head learning rate")
    ap.add_argument("--max-len", type=int, default=config.MAX_LEN)
    ap.add_argument("--train-subset", choices=["auto", "all", "with_evidence"], default="auto",
                    help="auto = the variant that won most classical models on validation (subset_selection.json)")
    args = ap.parse_args()

    tr, va = data.load_split("train"), data.load_split("validation")
    subset = args.train_subset
    if subset == "auto":
        sel = config.REPORT_DIR / "subset_selection.json"
        subset = json.loads(sel.read_text())["majority"] if sel.exists() else "all"
    if subset == "with_evidence":
        tr = tr[tr["text_evidence"].apply(len) > 0].reset_index(drop=True)
    print(f"training the deep model on the '{subset}' training subset: {len(tr)} claims")
    retriever = Retriever.load(config.RAG_DIR) if args.evidence == "retrieved" else None
    tr_ev = features.evidence_for_frame(tr, args.evidence, retriever)
    va_ev = features.evidence_for_frame(va, args.evidence, retriever)
    result = train(tr, tr_ev, features.load_features("train", tr), va, va_ev,
                   features.load_features("validation", va), config.DEEP_DIR,
                   epochs=args.epochs, lr=args.lr, head_lr=args.head_lr, batch_size=args.batch_size, max_len=args.max_len)
    print("best validation macro-F1:", round(result["best_val_macro_f1"], 4))


if __name__ == "__main__":
    main()
