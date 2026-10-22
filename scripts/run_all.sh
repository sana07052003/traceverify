#!/usr/bin/env bash
# Full training pipeline. Use a GPU runtime (e.g. Colab T4) for steps 3-5.
set -euo pipefail
python -m scripts.download_mocheg
python -m scripts.prepare_data
python -m scripts.train_classical
python -m scripts.build_index
python -m scripts.extract_clip_features --image-dir "${TV_IMAGE_DIR:-data/mocheg}"
python -m scripts.train_deep --evidence retrieved --epochs 3
python -m scripts.train_ensemble
python -m scripts.evaluate
