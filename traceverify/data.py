"""Loader for the ORIGINAL MOCHEG release (Yao et al., SIGIR 2023; https://doi.org/10.5281/zenodo.6653771).

Expected layout (found recursively under --mocheg-dir, so the unzip folder name does not matter):
    train/Corpus2.csv  val/Corpus2.csv  test/Corpus2.csv      one row per text-evidence piece
    train/images/  val/images/  test/images/                   image evidence  '<claim_id>-proof-<n>-<desc>'
    supplementary/Corpus3_sentence_level.csv                   (optional) sentence-level retrieval corpus
We use the official splits. `ruling_outline` and `Origin` (the fact-checker's reasoning / full article) are
never read, because they would leak the verdict.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import config, images

SPLIT_DIRS = {"train": ("train",), "validation": ("val", "dev", "validation"), "test": ("test",)}
_ALIASES = {
    "claim_id": ("claim_id", "claimid"),
    "claim": ("claim",),
    "evidence": ("evidence", "text_evidence"),
    "evidence_id": ("evidence_id",),
    "label": ("cleaned_truthfulness", "truthfulness", "label"),
    "snopes_url": ("snopes_url",),
}
_ALL_ALIASES = {a for aliases in _ALIASES.values() for a in aliases}


def _norm(col) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def normalize_label(value) -> str | None:
    s = str(value).strip().lower()
    if s.startswith("support"):
        return "supported"
    if s.startswith("refut"):
        return "refuted"
    if s == "nei" or s.startswith("not enough"):
        return "nei"
    return None


def find_corpus2(root: Path, split: str) -> Path:
    for p in sorted(Path(root).rglob("Corpus2.csv")):
        if p.parent.name.lower() in SPLIT_DIRS[split]:
            return p
    raise FileNotFoundError(
        f"Could not find a '{'/'.join(SPLIT_DIRS[split])}/Corpus2.csv' under {root}. "
        "Download MOCHEG first (python -m scripts.download_mocheg) and check the folder layout.")


def _read_csv(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False,
                               usecols=lambda c: _norm(c) in _ALL_ALIASES)
        except UnicodeDecodeError as exc:
            last = exc
    raise last


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for canon, aliases in _ALIASES.items():
        for c in df.columns:
            if _norm(c) in aliases and c not in rename:
                rename[c] = canon
                break
    return df.rename(columns=rename)


def load_official_split(root: Path, split: str) -> pd.DataFrame:
    """Claim-level frame: claim_id, claim, label, text_evidence(list), snopes_url(list), image(list)."""
    df = _canonical(_read_csv(find_corpus2(root, split)))
    missing = {"claim_id", "claim", "evidence", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Corpus2.csv ({split}) lacks columns {sorted(missing)}; found {list(df.columns)}")
    df["claim_id"] = df["claim_id"].astype(str).str.strip()
    df["claim"] = df["claim"].astype(str).str.strip()
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].notna() & (df["claim"] != "")]
    if "evidence_id" in df.columns:
        df["_eid"] = pd.to_numeric(df["evidence_id"], errors="coerce")
    rows = []
    for cid, g in df.groupby("claim_id", sort=False):
        if "_eid" in g:
            g = g.sort_values("_eid", kind="stable")
        ev = list(dict.fromkeys(e.strip() for e in g["evidence"] if isinstance(e, str) and e.strip()))
        url = g["snopes_url"].iloc[0].strip() if "snopes_url" in g and str(g["snopes_url"].iloc[0]).strip() else ""
        rows.append({"claim_id": cid, "claim": g["claim"].iloc[0], "label": g["label"].iloc[0],
                     "text_evidence": ev, "snopes_url": [url] if url else [], "image": []})
    return pd.DataFrame(rows, columns=["claim_id", "claim", "label", "text_evidence", "snopes_url", "image"])


def load_mocheg(root: Path = config.MOCHEG_DIR, image_dir: Path | None = None):
    """Official train / validation / test frames + evidence corpus + statistics.

    Leak control: a validation/test claim whose exact text already occurs in an earlier split is dropped."""
    root = Path(root)
    frames = {s: load_official_split(root, s) for s in ("train", "validation", "test")}
    stats = {"official_claims": {s: len(f) for s, f in frames.items()}}
    seen: set[str] = set()
    for s in ("train", "validation", "test"):
        f = frames[s].drop_duplicates(subset=["claim"])
        f = f[~f["claim"].isin(seen)]
        seen |= set(f["claim"])
        frames[s] = f.reset_index(drop=True)
    stats["after_dedup"] = {s: len(f) for s, f in frames.items()}

    img_map = images.index_by_claim(image_dir or root)
    for f in frames.values():
        f["image"] = f["claim_id"].map(lambda c: img_map.get(c, []))
    stats["claims_with_image"] = {s: int((f["image"].apply(len) > 0).sum()) for s, f in frames.items()}
    train, val, test = frames["train"], frames["validation"], frames["test"]
    corpus = build_evidence_corpus(pd.concat([train, val, test], ignore_index=True))
    stats["label_counts"] = pd.concat([train, val, test])["label"].value_counts().to_dict()
    stats["evidence_passages"] = len(corpus)
    return train, val, test, corpus, stats


def build_evidence_corpus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        url = r["snopes_url"][0] if r["snopes_url"] else ""
        for i, text in enumerate(r["text_evidence"]):
            rows.append({"evidence_id": f"{r['claim_id']}_{i}", "claim_id": str(r["claim_id"]),
                         "text": text, "source_url": url})
    return pd.DataFrame(rows, columns=["evidence_id", "claim_id", "text", "source_url"])


def load_sentence_corpus(root: Path, claims: pd.DataFrame, min_chars: int = 25) -> pd.DataFrame:
    """Optional, much larger corpus: MOCHEG's sentence-level split of all source documents."""
    p = next(Path(root).rglob("Corpus3_sentence_level.csv"), None)
    if p is None:
        raise FileNotFoundError("Corpus3_sentence_level.csv not found (it lives in MOCHEG's 'supplementary' folder).")
    df = pd.read_csv(p, dtype=str, keep_default_na=False,
                     usecols=lambda c: _norm(c) in {"claim_id", "corpus_id", "paragraph"})
    df.columns = [_norm(c) for c in df.columns]
    df["claim_id"] = df["claim_id"].str.strip()
    url = {c: (u[0] if u else "") for c, u in zip(claims["claim_id"], claims["snopes_url"])}
    df = df[df["claim_id"].isin(url) & (df["paragraph"].str.len() >= min_chars)]
    return pd.DataFrame({"evidence_id": df["corpus_id"], "claim_id": df["claim_id"], "text": df["paragraph"],
                         "source_url": df["claim_id"].map(url)}).reset_index(drop=True)


def save_processed(train, val, test, corpus, out_dir: Path = config.PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        frame.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    corpus.to_csv(out_dir / "evidence.csv", index=False)


def load_split(name: str, out_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    df = pd.read_json(out_dir / f"{name}.jsonl", lines=True, dtype=False)
    df["claim_id"] = df["claim_id"].astype(str)
    return df


def load_all_claims(out_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    return pd.concat([load_split(s, out_dir) for s in ("train", "validation", "test")], ignore_index=True)


def load_evidence_lookup(out_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_csv(out_dir / "evidence.csv", dtype=str, keep_default_na=False)


def gold_evidence_text(evidence: list[str], max_evidence: int = config.CLASSIFIER_EVIDENCE) -> str:
    return " ".join(evidence[:max_evidence])
