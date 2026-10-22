"""Locate trained artifacts: local folder first, otherwise download from a Hugging Face Hub repo."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config


def artifacts_ready(root: Path = config.ARTIFACT_DIR) -> bool:
    return (Path(root) / "rag" / "evidence.index").exists()


def ensure_artifacts(root: Path = config.ARTIFACT_DIR) -> Path:
    root = Path(root)
    if artifacts_ready(root):
        return root
    repo = os.getenv("TV_ARTIFACT_REPO", "")
    if not repo:
        raise FileNotFoundError(
            f"No trained artifacts in {root}. Run the training scripts (see README) or set "
            "TV_ARTIFACT_REPO to a Hugging Face repo that holds them.")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(root),
                      token=os.getenv("HF_TOKEN") or None)
    if not artifacts_ready(root):
        raise FileNotFoundError(f"Downloaded {repo} but evidence.index is missing.")
    return root


def load_reports(root: Path = config.ARTIFACT_DIR) -> dict:
    out = {}
    rdir = Path(root) / "reports"
    if rdir.exists():
        for p in rdir.glob("*.json"):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return out
