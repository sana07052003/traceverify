"""Download the ORIGINAL MOCHEG release from Zenodo (DOI 10.5281/zenodo.6653771, CC BY 4.0).

The record is one ~41 GB `mocheg_v1.tar.gz`. Saving it and then extracting it would need ~80 GB of disk, so by
default this script STREAMS the archive and extracts only what the project uses:
    * every  <split>/Corpus2.csv            (claims, labels, text evidence)
    * every image inside a train/val/test folder   (image evidence)
    * optionally  Corpus3_sentence_level.csv  (--with-sentences)
It skips the huge full-document corpus, the global image collection and embedding pickles.
Images are shrunk while extracting (longest side 512 px, JPEG) - CLIP only needs 224 px - so the whole subset is
small enough for a free Google Drive. A `.mocheg_extract_complete` marker is written when finished, so a finished
download is never repeated. (A .tar.gz cannot be resumed mid-stream: an interrupted run re-reads the archive from
the start but skips files already saved, so point --out at Google Drive.)

    python -m scripts.download_mocheg --list
    python -m scripts.download_mocheg                 # stream + selective extract (recommended)
    python -m scripts.download_mocheg --no-text-images   # skip images (text-only, much faster to finish)
If the network drops, just re-run: files already extracted are skipped.
Manual alternative: download from https://doi.org/10.5281/zenodo.6653771 and unzip into data/mocheg/.
"""
import argparse
import io
import json
import tarfile
import time
from pathlib import Path, PurePosixPath

import requests

from traceverify import config
from traceverify.images import IMG_EXT

RECORD = "6653771"
SPLIT_NAMES = {"train", "val", "dev", "validation", "test"}


def list_files() -> list[dict]:
    r = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json().get("files", []):
        links = f.get("links") or {}
        out.append({"name": f.get("key") or f.get("filename"), "size": f.get("size", 0),
                    "url": links.get("self") or links.get("download")})
    return out


def wanted(member_name: str, with_images: bool = True, with_sentences: bool = False) -> bool:
    p = PurePosixPath(member_name)
    parts = [x.lower() for x in p.parts]
    base = p.name.lower()
    if base == "corpus2.csv":
        return True
    if with_sentences and base == "corpus3_sentence_level.csv":
        return True
    if with_images and p.suffix.lower() in IMG_EXT and any(x in SPLIT_NAMES for x in parts[:-1]):
        return True
    return False


class _Counting:
    """File-like wrapper that reports download progress."""

    def __init__(self, raw, total: int):
        self.raw, self.total, self.done, self._t = raw, total, 0, time.time()

    def read(self, n=-1):
        chunk = self.raw.read(n)
        self.done += len(chunk)
        if time.time() - self._t > 30:
            self._t = time.time()
            pct = 100 * self.done / self.total if self.total else 0
            print(f"  streamed {self.done / 1e9:6.2f} GB ({pct:4.1f} %)", flush=True)
        return chunk


MARKER = ".mocheg_extract_complete"


def _save_resized(data: bytes, target: Path, max_side: int) -> bool:
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        if max_side:
            img.thumbnail((max_side, max_side))
        target.parent.mkdir(parents=True, exist_ok=True)
        img.save(target, "JPEG", quality=85)
        return True
    except Exception:
        return False


def stream_extract(fileobj, out: Path, with_images: bool = True, with_sentences: bool = False,
                   max_side: int = 512) -> dict:
    out = Path(out)
    kept = {"csv": 0, "images": 0, "unreadable_images": 0, "skipped_existing": 0}
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile() or not wanted(m.name, with_images, with_sentences):
                continue
            if PurePosixPath(m.name).suffix.lower() in IMG_EXT:
                target = (out / m.name).with_suffix(".jpg")
                if target.exists() and target.stat().st_size > 0:
                    kept["skipped_existing"] += 1
                    continue
                if _save_resized(tar.extractfile(m).read(), target, max_side):
                    kept["images"] += 1
                else:
                    kept["unreadable_images"] += 1
                continue
            target = out / m.name
            if target.exists() and target.stat().st_size == m.size:
                kept["skipped_existing"] += 1
                continue
            try:
                tar.extract(m, out, filter="data")      # Python >= 3.12 safe extraction
            except TypeError:
                tar.extract(m, out)
            kept["csv"] += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.MOCHEG_DIR))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--with-sentences", action="store_true", help="also keep Corpus3_sentence_level.csv")
    ap.add_argument("--no-text-images", dest="images", action="store_false", help="do not extract images")
    ap.add_argument("--max-side", type=int, default=512, help="shrink images to this longest side (0 = keep original)")
    ap.add_argument("--force", action="store_true", help="download again even if a completed extraction exists")
    args = ap.parse_args()

    files = list_files()
    for f in files:
        print(f"{f['name']:50s} {f['size'] / 1e9:8.2f} GB")
    if args.list:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / MARKER).exists() and not args.force:
        print(f"Already downloaded and extracted into {out} (marker found). Nothing to do. Use --force to redo.")
        return
    for f in files:
        if not f["name"].lower().endswith((".tar.gz", ".tgz")):
            print("skipping (not a tar.gz):", f["name"])
            continue
        print(f"streaming {f['name']} -> {out} (no 41 GB temp file is written)")
        with requests.get(f["url"], stream=True, timeout=(30, 600)) as r:
            r.raise_for_status()
            kept = stream_extract(_Counting(r.raw, f["size"]), out, args.images, args.with_sentences, args.max_side)
        print("extracted:", kept)
        (out / MARKER).write_text(json.dumps(kept))
    print("done ->", out)


if __name__ == "__main__":
    main()
