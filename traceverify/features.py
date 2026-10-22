"""Cached CLIP features (training) and evidence construction for classifier inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import gold_evidence_text


def feature_path(split: str) -> Path:
    return config.FEATURE_DIR / f"clip_{split}.npz"


def save_features(split: str, claim_ids, img, txt, has) -> None:
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path(split), claim_id=np.asarray(claim_ids, dtype=str),
                        img=np.asarray(img, dtype="float32"), txt=np.asarray(txt, dtype="float32"),
                        has_img=np.asarray(has, dtype=bool))


def load_features(split: str, df: pd.DataFrame):
    """Return (img, txt, has) aligned to df rows. Missing cache => zeros (text-only behaviour)."""
    n, d = len(df), config.CLIP_DIM
    img, txt, has = np.zeros((n, d), "float32"), np.zeros((n, d), "float32"), np.zeros(n, dtype=bool)
    p = feature_path(split)
    if not p.exists():
        return img, txt, has
    z = np.load(p, allow_pickle=False)
    # load each array ONCE: indexing an npz inside the loop re-decompresses the whole array every time
    zid, zimg, ztxt, zhas = z["claim_id"].astype(str), z["img"], z["txt"], z["has_img"]
    pos = {cid: i for i, cid in enumerate(zid)}
    for j, cid in enumerate(df["claim_id"].astype(str)):
        i = pos.get(cid)
        if i is not None:
            img[j], txt[j], has[j] = zimg[i], ztxt[i], bool(zhas[i])
    return img, txt, has


def evidence_for_frame(frame: pd.DataFrame, mode: str, retriever=None,
                       k: int = config.CLASSIFIER_EVIDENCE) -> list[str]:
    """mode: 'gold' (annotated evidence, oracle) | 'retrieved' (RAG, own evidence excluded:
    simulates a genuinely new claim) | 'retrieved_with_own' (RAG over the full corpus)."""
    if mode == "gold":
        return [gold_evidence_text(list(e), k) for e in frame["text_evidence"]]
    if mode not in ("retrieved", "retrieved_with_own"):
        raise ValueError(f"unknown evidence mode: {mode}")
    ex = frame["claim_id"].astype(str).tolist() if mode == "retrieved" else None
    results = retriever.search_batch(frame["claim"].tolist(), k=k, exclude_claim_ids=ex)
    return [" ".join(e["text"] for e in r) for r in results]
