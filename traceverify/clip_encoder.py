"""CLIP (ViT-B/32) wrapper used for the visual modality. Image and text share one 512-d space."""
from __future__ import annotations

import numpy as np

from . import config


def _shorten(text: str, max_words: int = 50) -> str:
    # CLIP's text tower accepts 77 tokens; trimming words avoids tokenizer overflow errors.
    return " ".join(str(text).split()[:max_words])


class ClipEncoder:
    def __init__(self, name: str = config.CLIP_MODEL_NAME, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)

    def images(self, pil_images: list, batch_size: int = 32, progress: bool = False) -> np.ndarray:
        emb = self.model.encode(pil_images, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=progress)
        return np.asarray(emb, dtype="float32")

    def texts(self, texts: list[str], batch_size: int = 128, progress: bool = False) -> np.ndarray:
        emb = self.model.encode([_shorten(t) for t in texts], batch_size=batch_size,
                                convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=progress)
        return np.asarray(emb, dtype="float32")


def consistency_band(sim: float) -> str:
    """Rough, heuristic reading of CLIP image-text cosine (typical matches fall around 0.25-0.35)."""
    if sim >= 0.27:
        return "strong match"
    if sim >= 0.20:
        return "moderate match"
    return "weak match"
