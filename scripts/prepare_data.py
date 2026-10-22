"""Step 1: read the ORIGINAL MOCHEG release (official splits) and build processed files + evidence corpus."""
import argparse
import json

from traceverify import config, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mocheg-dir", default=str(config.MOCHEG_DIR))
    ap.add_argument("--corpus", choices=["evidence", "sentences"], default="evidence",
                    help="evidence = annotated text evidence of all claims (compact, default); "
                         "sentences = MOCHEG's full sentence-level document corpus (large)")
    args = ap.parse_args()
    train, val, test, corpus, stats = data.load_mocheg(args.mocheg_dir)
    if args.corpus == "sentences":
        import pandas as pd
        corpus = data.load_sentence_corpus(args.mocheg_dir, pd.concat([train, val, test]))
        stats["evidence_passages"] = len(corpus)
    data.save_processed(train, val, test, corpus)
    print(json.dumps(stats, indent=2))
    print(f"saved to {config.PROCESSED_DIR}")
    if not sum(stats["claims_with_image"].values()):
        print("NOTE: no image files were matched. The system will run text-only until images are available.")


if __name__ == "__main__":
    main()
