"""Evidence retrieval: dense (FAISS), hybrid (dense + BM25), and CLIP image -> fact-checked claims."""
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from . import config

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def load_embedder(name: str = config.EMBED_MODEL_NAME, device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


class Retriever:
    """Searches the MOCHEG evidence corpus. `embedder` only needs `.encode(list[str], ...)`."""

    def __init__(self, index, lookup: pd.DataFrame, embedder):
        self.index = index
        self.lookup = lookup.reset_index(drop=True)
        self.embedder = embedder
        self._emb = None
        self._bm25 = None

    # ---------------------------------------------------------------- build / persist
    @classmethod
    def build(cls, lookup: pd.DataFrame, embedder, batch_size: int = 64, progress: bool = True):
        texts = lookup["text"].fillna("").tolist()
        emb = embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=progress)
        emb = np.ascontiguousarray(emb, dtype="float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        obj = cls(index, lookup, embedder)
        obj._emb = emb
        return obj

    def save(self, directory: Path = config.RAG_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "evidence.index"))
        self.lookup.to_csv(directory / "evidence_lookup.csv", index=False)

    @classmethod
    def load(cls, directory: Path = config.RAG_DIR, embedder=None):
        directory = Path(directory)
        index = faiss.read_index(str(directory / "evidence.index"))
        lookup = pd.read_csv(directory / "evidence_lookup.csv", dtype=str, keep_default_na=False)
        return cls(index, lookup, embedder or load_embedder())

    # ---------------------------------------------------------------- internals
    def _encode(self, queries: list[str]) -> np.ndarray:
        q = self.embedder.encode(list(queries), convert_to_numpy=True, normalize_embeddings=True)
        return np.ascontiguousarray(q, dtype="float32")

    def _embeddings(self) -> np.ndarray:
        if self._emb is None:
            self._emb = self.index.reconstruct_n(0, self.index.ntotal)
        return self._emb

    def _bm25_model(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(t) for t in self.lookup["text"]])
        return self._bm25

    def _pack(self, scores, idxs, k: int, exclude: str | None) -> list[dict]:
        out = []
        for s, i in zip(scores, idxs):
            if i < 0:
                continue
            row = self.lookup.iloc[int(i)]
            if exclude is not None and str(row["claim_id"]) == str(exclude):
                continue
            out.append({"evidence_id": row["evidence_id"], "claim_id": str(row["claim_id"]),
                        "text": row["text"], "source_url": row.get("source_url", ""),
                        "score": float(s)})
            if len(out) == k:
                break
        return out

    # ---------------------------------------------------------------- public API
    def search_batch(self, queries: list[str], k: int = 5, hybrid: bool = False, alpha: float = 0.5,
                     exclude_claim_ids: list[str] | None = None) -> list[list[dict]]:
        q = self._encode(queries)
        extra = 60 if exclude_claim_ids is not None else 0
        results = []
        if not hybrid:
            scores, idxs = self.index.search(q, k + extra)
            for qi in range(len(queries)):
                ex = exclude_claim_ids[qi] if exclude_claim_ids is not None else None
                results.append(self._pack(scores[qi], idxs[qi], k, ex))
            return results
        emb, bm25 = self._embeddings(), self._bm25_model()
        for qi, query in enumerate(queries):
            dense = emb @ q[qi]
            lex = np.asarray(bm25.get_scores(tokenize(query)), dtype="float32")
            lex = (lex - lex.min()) / (lex.max() - lex.min() + 1e-9)
            combined = alpha * dense + (1 - alpha) * lex
            top = np.argsort(-combined)[: k + extra]
            ex = exclude_claim_ids[qi] if exclude_claim_ids is not None else None
            results.append(self._pack(combined[top], top, k, ex))
        return results

    def search(self, query: str, k: int = 5, hybrid: bool = False, alpha: float = 0.5,
               exclude_claim_id: str | None = None) -> list[dict]:
        ex = [exclude_claim_id] if exclude_claim_id is not None else None
        return self.search_batch([query], k, hybrid, alpha, ex)[0]

    def recall_at_ks(self, claims: pd.DataFrame, ks=(1, 3, 5, 10), hybrid: bool = False) -> dict[str, float]:
        """Recall@k = fraction of claims whose own evidence appears in the top-k (paper Eq. 15)."""
        ids = claims["claim_id"].astype(str).tolist()
        res = self.search_batch(claims["claim"].tolist(), k=max(ks), hybrid=hybrid)
        return {str(k): float(np.mean([any(e["claim_id"] == cid for e in r[:k])
                                       for cid, r in zip(ids, res)])) for k in ks}


class ClaimImageIndex:
    """CLIP text embeddings of every fact-checked claim, searchable with a CLIP *image* embedding.

    Used for image-only input and as extra visual context for text+image input.
    """

    def __init__(self, index, claims: pd.DataFrame):
        self.index = index
        self.claims = claims.reset_index(drop=True)

    @classmethod
    def build(cls, claims: pd.DataFrame, clip_text_embeddings: np.ndarray):
        emb = np.ascontiguousarray(clip_text_embeddings, dtype="float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        keep = claims[["claim_id", "claim", "label"]].copy()
        keep["source_url"] = claims["snopes_url"].apply(lambda u: u[0] if isinstance(u, list) and u else "")
        return cls(index, keep)

    def save(self, directory: Path = config.RAG_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "clip_claims.index"))
        self.claims.to_csv(directory / "clip_claims.csv", index=False)

    @classmethod
    def load(cls, directory: Path = config.RAG_DIR):
        directory = Path(directory)
        index = faiss.read_index(str(directory / "clip_claims.index"))
        claims = pd.read_csv(directory / "clip_claims.csv", dtype=str, keep_default_na=False)
        return cls(index, claims)

    @classmethod
    def exists(cls, directory: Path = config.RAG_DIR) -> bool:
        directory = Path(directory)
        return (directory / "clip_claims.index").exists() and (directory / "clip_claims.csv").exists()

    def search(self, image_embedding: np.ndarray, k: int = 5) -> list[dict]:
        q = np.ascontiguousarray(np.asarray(image_embedding, dtype="float32").reshape(1, -1))
        scores, idxs = self.index.search(q, k)
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            row = self.claims.iloc[int(i)]
            out.append({"claim_id": row["claim_id"], "claim": row["claim"], "label": row["label"],
                        "source_url": row["source_url"], "score": float(s)})
        return out
