"""Step 4 (visual modality): CLIP features for claims + their images, and the image->claim index.

Images come from the ORIGINAL MOCHEG release: each claim's image evidence (files named
'<claim_id>-proof-<n>-<desc>') found under --image-dir. A claim's image vector is the mean of up to 3 of its
images. Claims without a usable image stay text-only (has_img = False).
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from traceverify import config, data, features, images
from traceverify.clip_encoder import ClipEncoder
from traceverify.retrieval import ClaimImageIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default=str(config.IMAGE_DIR))
    ap.add_argument("--max-images", type=int, default=3)
    args = ap.parse_args()

    clip = ClipEncoder()
    idx = images.index_images(args.image_dir)
    print(f"found {len(idx)} image files under {args.image_dir}")
    if not idx:
        print("WARNING: no images found - features will be text-only (the model still trains).")

    all_claims, all_txt = [], []
    for split in ("train", "validation", "test"):
        df = data.load_split(split)
        txt = clip.texts(df["claim"].tolist(), progress=True)
        img = np.zeros_like(txt)
        has = np.zeros(len(df), dtype=bool)
        for pos, names in enumerate(tqdm(df["image"], desc=f"{split} images")):
            pils = []
            for name in list(names or [])[: args.max_images]:
                path = idx.get(str(name).lower())
                pil = images.open_image(path) if path else None
                if pil is not None:
                    pils.append(pil)
            if pils:
                v = clip.images(pils).mean(axis=0)
                img[pos] = v / (np.linalg.norm(v) + 1e-9)
                has[pos] = True
        features.save_features(split, df["claim_id"], img, txt, has)
        print(f"{split}: {int(has.sum())}/{len(df)} claims have a usable image")
        all_claims.append(df)
        all_txt.append(txt)

    claims = pd.concat(all_claims, ignore_index=True)
    ClaimImageIndex.build(claims, np.vstack(all_txt)).save(config.RAG_DIR)
    print("saved CLIP claim index (image -> related fact-checked claims)")


if __name__ == "__main__":
    main()
