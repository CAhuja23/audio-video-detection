#!/usr/bin/env python3
"""
prepare_models_offline.py

One-time online setup:
- downloads Whisper tiny into a local folder
- downloads YAMNet from TF Hub
- saves YAMNet as a local SavedModel
- downloads yamnet_class_map.csv locally

After this finishes, your main runtime can work without internet
as long as it points to these local files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import tensorflow as tf
import tensorflow_hub as hub
import whisper

# -----------------------------
# Local output paths
# -----------------------------
BASE_DIR = Path("local_models")
BASE_DIR.mkdir(exist_ok=True)

WHISPER_DIR = BASE_DIR / "whisper"
WHISPER_DIR.mkdir(exist_ok=True)

YAMNET_DIR = BASE_DIR / "yamnet_saved_model"
YAMNET_CSV = BASE_DIR / "yamnet_class_map.csv"

# -----------------------------
# Official sources
# -----------------------------
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

WHISPER_MODEL_NAME = "tiny"


def download_whisper_local() -> None:
    print(f"[whisper] downloading/loading model: {WHISPER_MODEL_NAME}")
    # This downloads once if missing, then reuses local files on future runs.
    whisper.load_model(WHISPER_MODEL_NAME, download_root=str(WHISPER_DIR))
    print(f"[whisper] ready in: {WHISPER_DIR}")


def download_yamnet_local() -> None:
    print(f"[yamnet] downloading from TF Hub: {YAMNET_HANDLE}")
    model = hub.load(YAMNET_HANDLE)

    if YAMNET_DIR.exists():
        shutil.rmtree(YAMNET_DIR)

    tf.saved_model.save(model, str(YAMNET_DIR))
    print(f"[yamnet] saved model to: {YAMNET_DIR}")

    print("[yamnet] downloading class map csv")
    urlretrieve(YAMNET_CLASS_MAP_URL, str(YAMNET_CSV))
    print(f"[yamnet] class map saved to: {YAMNET_CSV}")


def main() -> None:
    print("[setup] preparing offline model bundle...")
    download_whisper_local()
    download_yamnet_local()

    print("\n[done] offline assets created:")
    print(f"  Whisper cache dir : {WHISPER_DIR}")
    print(f"  YAMNet model dir  : {YAMNET_DIR}")
    print(f"  YAMNet class map  : {YAMNET_CSV}")
    print("\nUse these paths in your main server code.")


if __name__ == "__main__":
    main()