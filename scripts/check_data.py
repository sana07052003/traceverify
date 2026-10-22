"""Data sanity / leakage report: split overlap, missing evidence, label balance, image coverage."""
import json

from traceverify import config, data


def main():
    fr = {s: data.load_split(s) for s in ("train", "validation", "test")}
    names = list(fr)
    rep = {"claims": {s: len(f) for s, f in fr.items()},
           "labels": {s: f["label"].value_counts().to_dict() for s, f in fr.items()},
           "claims_without_evidence": {s: int((f["text_evidence"].apply(len) == 0).sum()) for s, f in fr.items()},
           "claims_with_image": {s: int((f["image"].apply(len) > 0).sum()) for s, f in fr.items()},
           "overlap_claim_text": {}, "overlap_claim_id": {}}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = names[i], names[j]
            rep["overlap_claim_text"][f"{a}&{b}"] = len(set(fr[a]["claim"]) & set(fr[b]["claim"]))
            rep["overlap_claim_id"][f"{a}&{b}"] = len(set(fr[a]["claim_id"]) & set(fr[b]["claim_id"]))
    rep["leakage_free"] = not any(rep["overlap_claim_text"].values()) and not any(rep["overlap_claim_id"].values())
    rep["label_columns_used"] = "only Corpus2 'cleaned_truthfulness' as the target; ruling_outline / Origin never read"
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "data_checks.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("LEAKAGE CHECK:", "PASSED (no claim shared across splits)" if rep["leakage_free"] else "FAILED")


if __name__ == "__main__":
    main()
