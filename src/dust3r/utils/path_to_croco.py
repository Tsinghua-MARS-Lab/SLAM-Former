# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import sys
import os.path as path
import importlib

HERE_PATH = path.normpath(path.dirname(__file__))
CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, "../../croco"))
CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, "models")
# IMPORTANT:
# Do NOT add `.../src/croco` directly to sys.path, otherwise subfolders like
# `croco/datasets` become a top-level module named `datasets`, which will shadow
# HuggingFace `datasets` and break `accelerate` (and others).
# Instead, add `.../src` so we import as `croco.*`.
SRC_PATH = path.normpath(path.join(HERE_PATH, "../../.."))

if path.isdir(CROCO_MODELS_PATH):

    # Prefer adding the `src` directory; this enables `import croco...` without
    # polluting top-level module names.
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    # In case an old run already inserted CROCO_REPO_PATH, remove it to avoid
    # shadowing top-level modules (e.g., `datasets`).
    while CROCO_REPO_PATH in sys.path:
        sys.path.remove(CROCO_REPO_PATH)

    # Backward-compat: DUSt3R code expects `models.*` to exist as a top-level package
    # (historically achieved by adding CROCO_REPO_PATH to sys.path). We keep that
    # import path working by aliasing `croco.models` to `models` without exposing
    # other top-level names like `datasets`.
    try:
        _croco_models = importlib.import_module("croco.models")
        sys.modules.setdefault("models", _croco_models)
    except Exception:
        # If croco isn't importable yet, downstream import will raise a clearer error.
        pass
else:
    raise ImportError(
        f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )
