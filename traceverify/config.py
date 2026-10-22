"""Central configuration. Secrets are NEVER stored here: they come from env vars / Streamlit secrets."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("TV_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("TV_ARTIFACT_DIR", ROOT / "artifacts"))

PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"          # cached CLIP features (training only)
MOCHEG_DIR = Path(os.getenv("TV_MOCHEG_DIR", DATA_DIR / "mocheg"))   # original MOCHEG release
IMAGE_DIR = Path(os.getenv("TV_IMAGE_DIR", MOCHEG_DIR))                # images are found recursively here
BASELINE_PATH = ARTIFACT_DIR / "baseline" / "tfidf_lr.joblib"
DEEP_DIR = ARTIFACT_DIR / "deep_model"
RAG_DIR = ARTIFACT_DIR / "rag"
REPORT_DIR = ARTIFACT_DIR / "reports"

EMBED_MODEL_NAME = os.getenv("TV_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEEP_BASE_MODEL = os.getenv("TV_DEEP_BASE", "distilbert-base-uncased")
CLIP_MODEL_NAME = "clip-ViT-B-32"
CLIP_DIM = 512
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Alphabetical == sklearn's classes_ ordering, so every model shares one label order.
LABELS = ["nei", "refuted", "supported"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
DISPLAY = {"nei": "Not enough info", "refuted": "Refuted", "supported": "Supported"}

SEED = 42
CLASSIFIER_EVIDENCE = 3         # retrieved passages fed to the deep model
MAX_LEN = 320                   # tokens for claim + evidence
LOW_RELEVANCE_THRESHOLD = 0.45  # top cosine below this => warn "no strong evidence"
