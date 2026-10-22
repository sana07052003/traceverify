"""Image helpers: locate MOCHEG image files and open uploads safely."""
from __future__ import annotations

import re
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_CLAIM_PREFIX = re.compile(r"^(\d+)-")


def index_images(root: Path) -> dict[str, Path]:
    """Map lower-cased file name -> path, searching `root` recursively."""
    root = Path(root)
    if not root.exists():
        return {}
    return {p.name.lower(): p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT}


def index_by_claim(root: Path, proof_only: bool = True) -> dict[str, list[str]]:
    """claim_id -> image file names, using MOCHEG's naming '<claim_id>-proof-<img_id>-<description>'.

    Image evidence in the train/val/test folders is named '...-proof-...'. If a copy of the data uses
    no such marker, all images are used instead."""
    names = [p.name for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT] if Path(root).exists() else []
    has_proof = any("-proof-" in n.lower() for n in names)
    out: dict[str, set[str]] = {}
    for n in names:
        if proof_only and has_proof and "-proof-" not in n.lower():
            continue
        m = _CLAIM_PREFIX.match(n)
        if m:
            out.setdefault(m.group(1), set()).add(n)
    return {k: sorted(v) for k, v in out.items()}


def primary_image_name(value) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def open_image(src):
    """Open a path or file-like object as RGB PIL image; None if unreadable."""
    from PIL import Image
    try:
        img = Image.open(src)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None
