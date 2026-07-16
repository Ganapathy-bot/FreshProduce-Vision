#!/usr/bin/env python3
"""Generate the production Kaggle notebook for FreshProduce shelf-life prediction."""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
    }
)

cells = []


def md(source: str):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# =============================================================================
# CELL 0 — Title
# =============================================================================
md(
    r"""
# 🍎🥦 FreshProduce Vision — Remaining Shelf-Life Prediction

**Senior CV / DL pipeline for Kaggle** — multi-task transfer learning that predicts:

| Output | Description |
|--------|-------------|
| **Category** | Fruit / vegetable identity (e.g. Apple, Tomato) |
| **Freshness stage** | Fresh → Semi-Fresh → Early/Mid Ripening → Fully Ripe → Spoiled |
| **Remaining shelf life** | Estimated days remaining (proxy heuristic → lab labels when available) |
| **Confidence** | Softmax confidence for category + freshness |

### Highlights
- **Images folder only** — scans produce subfolders (no annotations.csv / train-val-test trees)
- Expected layout: `images/{Fruit|Vegetable}/{Produce}/{Freshness}/img.jpg`
- Skips corrupted images; stratified train / val / test splits from the folder
- Optimized `tf.data` pipeline (cache / prefetch / parallel map)
- Realistic augmentations that preserve natural color cues for freshness
- Configurable backbones: EfficientNetV2B0 (default), EfficientNetB3, ConvNeXtTiny, DenseNet121, MobileNetV3, ResNet50
- Two-stage training: frozen head → fine-tune upper layers
- AdamW, mixed precision, class weights, EarlyStopping, Checkpoint, ReduceLROnPlateau / Cosine
- Full evaluation suite: curves, CM, classification report, ROC, Grad-CAM, misclass analysis
- Single / multi-image inference with shelf-life decoding
- All artifacts written to `/kaggle/working`

> **Dataset note:** remaining shelf-life is derived from freshness stage folders (proxy heuristic). Treat RSL as approximate until lab Day0→Spoiled labels are available.
"""
)

# =============================================================================
# CELL 1 — Config
# =============================================================================
md("## 1. Configuration")

code(
    r'''
# ============================================================
# GLOBAL CONFIG — edit these knobs for experiments
# ============================================================
from pathlib import Path
import os
import random
import warnings

warnings.filterwarnings("ignore")

# ---- Paths ----
# ONLY source of data: a single folder of produce images with subfolders.
# Layout (this pack):
#   images/
#     Fruit/
#       Apple/
#         Fresh/  *.jpg
#         Spoiled/
#       Banana/ ...
#     Vegetable/
#       Tomato/
#         Fresh/
#         Fully_Ripe/
#         Spoiled/
#
# Also accepted (flatter):
#   images/Apple/Fresh/*.jpg
#   images/Tomato/Spoiled/*.jpg
#
# Set explicitly, or leave None to auto-find an `images` folder under CWD / Kaggle input.
IMAGES_DIR = None  # e.g. Path("/kaggle/input/your-dataset/images") or Path("images")
WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# Stratified splits carved from the images folder (no separate train/val/test dirs)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---- Image / pipeline ----
IMG_SIZE = (224, 224)          # (H, W)
BATCH_SIZE = 32
AUTOTUNE = None                # set after TF import
NUM_PARALLEL_CALLS = None
CACHE_TRAIN = False            # True if dataset fits in RAM / SSD budget
SHUFFLE_BUFFER = 2048

# ---- Backbone (one of: efficientnetv2b0 | efficientnetb3 | convnext_tiny
#                       densenet121 | mobilenetv3 | resnet50) ----
BACKBONE = "efficientnetv2b0"
PRETRAINED = True

# ---- Training schedule ----
EPOCHS_HEAD = 8                # stage 1: frozen backbone
EPOCHS_FINETUNE = 20           # stage 2: unfreeze upper layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-5
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.05
UNFREEZE_RATIO = 0.4           # fraction of backbone layers to unfreeze (from top)
USE_COSINE_LR = False          # True → CosineDecay; False → ReduceLROnPlateau
MIXED_PRECISION = True         # GPU mixed precision (float16 compute)
EARLY_STOP_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ---- Loss weights for multi-task heads ----
LOSS_WEIGHT_CATEGORY = 1.0
LOSS_WEIGHT_FRESHNESS = 1.25   # freshness is primary for shelf-life

# ---- Class / label policy ----
# Primary training target can be:
#   "multitask"  → separate category + freshness heads (recommended)
#   "combined"   → single head over Produce_Stage classes
TASK_MODE = "multitask"

# Freshness stage → remaining shelf life (days). Override if lab RSL available.
# Matches FreshProduce-Vision Image Pack proxy heuristics.
DEFAULT_RSL_MAP = {
    "Fresh": 7.0,
    "Day0": 7.0,
    "Day1": 6.0,
    "Day2": 5.0,
    "Day3": 4.0,
    "Day4": 3.0,
    "Day5": 2.0,
    "Day6": 1.0,
    "Semi_Fresh": 5.0,
    "Early_Ripening": 4.0,
    "Mid_Ripening": 3.0,
    "Fully_Ripe": 2.0,
    "Ripened": 2.0,
    "Spoiled": 0.0,
    "Rotten": 0.0,
}

# Ordered freshness progression for reporting / ordinal logic
FRESHNESS_ORDER = [
    "Fresh", "Day0", "Day1", "Day2", "Day3", "Day4", "Day5", "Day6",
    "Semi_Fresh", "Early_Ripening", "Mid_Ripening", "Fully_Ripe", "Ripened", "Spoiled", "Rotten",
]

# ---- EDA / viz ----
EDA_SAMPLES_PER_CLASS = 3
MAX_MISCLASS_SHOW = 16
GRADCAM_SAMPLES = 8

# ---- QC ----
# Full PIL verify on ~23k images is thorough but slow (~10–20 min).
# Set False to only check file existence (much faster).
VALIDATE_IMAGES = True
MAX_VALIDATE = None  # e.g. 2000 to sample-QC; None = all

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Split ratios must sum to 1.0"

print("Config loaded.")
print(f"  IMAGES_DIR     = {IMAGES_DIR}")
print(f"  BACKBONE       = {BACKBONE}")
print(f"  TASK_MODE      = {TASK_MODE}")
print(f"  IMG_SIZE       = {IMG_SIZE}")
print(f"  BATCH_SIZE     = {BATCH_SIZE}")
print(f"  SPLITS         = train {TRAIN_RATIO:.0%} / val {VAL_RATIO:.0%} / test {TEST_RATIO:.0%}")
print(f"  WORKING_DIR    = {WORKING_DIR}")
'''
)

# =============================================================================
# CELL 2 — Imports & TF setup
# =============================================================================
md("## 2. Environment, Imports & Mixed Precision")

code(
    r'''
import json
import math
import time
import gc
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated JPEGs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetB3,
    DenseNet121,
    MobileNetV3Large,
    ResNet50,
)

# ConvNeXtTiny is available in TF >= 2.11 applications
try:
    from tensorflow.keras.applications import ConvNeXtTiny
    HAS_CONVNEXT = True
except ImportError:
    HAS_CONVNEXT = False
    ConvNeXtTiny = None

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

# Optional Grad-CAM dependency (cv2) — fall back if missing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠ OpenCV not found — Grad-CAM heatmaps will use a pure-TF fallback.")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# Seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Autotune
AUTOTUNE = tf.data.AUTOTUNE
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE

# GPU + mixed precision
gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__}")
print(f"GPUs detected: {len(gpus)} → {[g.name for g in gpus]}")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("Memory growth setup:", e)

if MIXED_PRECISION and gpus:
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy: mixed_float16")
else:
    mixed_precision.set_global_policy("float32")
    print("Mixed precision policy: float32")

print("✓ Environment ready.")
'''
)

# =============================================================================
# CELL 3 — Dataset discovery
# =============================================================================
md(
    r"""
## 3. Images Folder Discovery

**Only** the produce images folder is used. Expected structure:

```text
images/
├── Fruit/
│   ├── Apple/
│   │   ├── Fresh/
│   │   ├── Semi_Fresh/
│   │   └── Spoiled/
│   └── Banana/
│       └── ...
└── Vegetable/
    ├── Tomato/
    │   ├── Fresh/
    │   ├── Fully_Ripe/
    │   └── Spoiled/
    └── Potato/
        └── ...
```

- Top level: optional group folders (`Fruit`, `Vegetable`) **or** produce names directly
- Next: **one subfolder per fruit/vegetable** (Apple, Tomato, …)
- Inside each produce: **freshness stage** folders (`Fresh`, `Day0`, `Spoiled`, …)
- Image files live in those stage folders

Set `IMAGES_DIR` in the config cell, or leave `None` to auto-locate a folder named `images`.
"""
)

code(
    r'''
# ============================================================
# Resolve IMAGES_DIR only (no annotations.csv / train-val-test trees)
# ============================================================

PRODUCE_GROUP_NAMES = {"fruit", "fruits", "vegetable", "vegetables", "produce"}


def _looks_like_images_root(path: Path) -> bool:
    """True if path has produce subfolders (with or without Fruit/Vegetable groups)."""
    if not path.is_dir():
        return False
    subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        return False
    # Case A: Fruit/ / Vegetable/ groups
    groupish = [d for d in subdirs if d.name.lower() in PRODUCE_GROUP_NAMES]
    if groupish:
        for g in groupish:
            produce = [p for p in g.iterdir() if p.is_dir()]
            if produce:
                return True
    # Case B: produce names directly under images/
    for p in subdirs:
        stage_or_imgs = list(p.iterdir())
        if any(c.is_dir() for c in stage_or_imgs):
            return True
        if any(c.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for c in stage_or_imgs if c.is_file()):
            return True
    return False


def _candidate_image_dirs() -> List[Path]:
    cands: List[Path] = []
    if IMAGES_DIR is not None:
        cands.append(Path(IMAGES_DIR))

    cwd = Path.cwd()
    # Common local / Kaggle locations
    for base in [cwd, cwd.parent]:
        cands.append(base / "images")
        cands.append(base)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for ds in sorted(kaggle_input.iterdir()):
            if not ds.is_dir():
                continue
            cands.append(ds / "images")
            cands.append(ds)
            for sub in sorted(ds.iterdir()):
                if sub.is_dir():
                    cands.append(sub / "images")
                    cands.append(sub)

    seen, out = set(), []
    for c in cands:
        try:
            r = c.resolve()
        except Exception:
            r = c
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def find_images_dir() -> Path:
    for path in _candidate_image_dirs():
        if _looks_like_images_root(path):
            print(f"✓ Images folder: {path}")
            return path
    raise FileNotFoundError(
        "Could not find a produce images folder.\n"
        "Set IMAGES_DIR to your folder, e.g.:\n"
        "  IMAGES_DIR = Path('/kaggle/input/your-dataset/images')\n"
        "Expected: images/Fruit/Apple/Fresh/*.jpg  or  images/Apple/Fresh/*.jpg"
    )


IMAGES_DIR = find_images_dir()
print(f"Using IMAGES_DIR = {IMAGES_DIR}")

# Quick tree summary (produce names only)
print("\nProduce subfolders found:")
for child in sorted(IMAGES_DIR.iterdir()):
    if not child.is_dir() or child.name.startswith("."):
        continue
    if child.name.lower() in PRODUCE_GROUP_NAMES:
        produce_dirs = sorted([p for p in child.iterdir() if p.is_dir()])
        print(f"  [{child.name}] → {len(produce_dirs)} produce: "
              + ", ".join(p.name for p in produce_dirs[:12])
              + (" …" if len(produce_dirs) > 12 else ""))
    else:
        n_stages = sum(1 for p in child.iterdir() if p.is_dir())
        print(f"  {child.name}/  ({n_stages} stage subfolders)")
'''
)

# =============================================================================
# CELL 4 — Load metadata
# =============================================================================
md("## 4. Scan Images Folder, Validate, Split")

code(
    r'''
# ============================================================
# Scan ONLY IMAGES_DIR — produce subfolders → freshness stages → images
# ============================================================
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CANONICAL_FRESHNESS = {
    "fresh": "Fresh",
    "semi_fresh": "Semi_Fresh",
    "semifresh": "Semi_Fresh",
    "early_ripening": "Early_Ripening",
    "mid_ripening": "Mid_Ripening",
    "fully_ripe": "Fully_Ripe",
    "ripened": "Fully_Ripe",
    "ripe": "Fully_Ripe",
    "spoiled": "Spoiled",
    "rotten": "Spoiled",
    "day0": "Day0", "day_0": "Day0",
    "day1": "Day1", "day_1": "Day1",
    "day2": "Day2", "day_2": "Day2",
    "day3": "Day3", "day_3": "Day3",
    "day4": "Day4", "day_4": "Day4",
    "day5": "Day5", "day_5": "Day5",
    "day6": "Day6", "day_6": "Day6",
    "day7": "Day7", "day_7": "Day7",
}


def _canon_token(s: str) -> str:
    return str(s).strip().replace("-", "_").replace(" ", "_")


def parse_folder_label(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse folder name into (category, freshness). Used for combined names too."""
    name = _canon_token(folder_name)
    low = name.lower()
    if low in CANONICAL_FRESHNESS:
        return None, CANONICAL_FRESHNESS[low]
    parts = name.split("_")
    if len(parts) >= 2:
        for k in range(1, min(3, len(parts))):
            tail = "_".join(parts[-k:]).lower()
            if tail in CANONICAL_FRESHNESS:
                cat = "_".join(parts[:-k])
                return cat if cat else None, CANONICAL_FRESHNESS[tail]
    return name, None


def canon_freshness(name: str) -> str:
    low = _canon_token(name).lower()
    return CANONICAL_FRESHNESS.get(low, _canon_token(name))


def is_image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
            if min(im.size) < 8:
                return False
        return True
    except Exception:
        return False


def rsl_from_freshness(freshness: str, fallback: Optional[float] = None) -> float:
    if freshness is None:
        return float(fallback) if fallback is not None else float("nan")
    key = _canon_token(freshness)
    if key in DEFAULT_RSL_MAP:
        return float(DEFAULT_RSL_MAP[key])
    for k, v in DEFAULT_RSL_MAP.items():
        if k.lower() == key.lower():
            return float(v)
    return float(fallback) if fallback is not None else float("nan")


def iter_produce_dirs(images_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Yield (group, category, produce_path) for each produce subfolder.

    Supported:
      images/Fruit/Apple/...
      images/Vegetable/Tomato/...
      images/Apple/...          (no group)
    """
    out = []
    for child in sorted(images_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name.lower() in PRODUCE_GROUP_NAMES:
            group = child.name
            for produce in sorted(child.iterdir()):
                if produce.is_dir() and not produce.name.startswith("."):
                    out.append((group, produce.name, produce))
        else:
            # Direct produce folder under images/
            out.append(("Unknown", child.name, child))
    return out


def load_from_images_folder(images_dir: Path) -> pd.DataFrame:
    """
    Walk:
      {IMAGES_DIR}/{Fruit|Vegetable}/{Produce}/{FreshnessStage}/*.{jpg,png,...}
    or:
      {IMAGES_DIR}/{Produce}/{FreshnessStage}/*.{jpg,png,...}
    or images directly under produce if no stage folders.
    """
    rows = []
    produce_list = iter_produce_dirs(images_dir)
    if not produce_list:
        raise RuntimeError(f"No produce subfolders under {images_dir}")

    print(f"Scanning {len(produce_list)} produce folders under {images_dir} …")
    for group, category, produce_path in tqdm(produce_list, desc="Produce folders"):
        # Stage subfolders?
        stage_dirs = [d for d in sorted(produce_path.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        if stage_dirs:
            for stage_dir in stage_dirs:
                freshness = canon_freshness(stage_dir.name)
                # If stage folder is actually a nested unknown structure, still use its name
                for img in stage_dir.rglob("*"):
                    if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    rows.append({
                        "filepath": str(img.resolve()),
                        "image_name": img.name,
                        "group": group,
                        "category": str(category),
                        "freshness": str(freshness),
                        "combined": f"{category}_{freshness}",
                        "remaining_shelf_life": rsl_from_freshness(freshness),
                    })
        else:
            # Images directly under produce folder → freshness Unknown
            for img in produce_path.iterdir():
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                    continue
                freshness = "Unknown"
                rows.append({
                    "filepath": str(img.resolve()),
                    "image_name": img.name,
                    "group": group,
                    "category": str(category),
                    "freshness": freshness,
                    "combined": f"{category}_{freshness}",
                    "remaining_shelf_life": rsl_from_freshness(freshness, fallback=0.0),
                })

    if not rows:
        raise RuntimeError(
            f"No images found under {images_dir}. "
            "Expected: images/Fruit/Apple/Fresh/*.jpg (or images/Apple/Fresh/*.jpg)"
        )
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified train/val/test from a single folder (no pre-made splits)."""
    df = df.copy()
    # First hold out test, then split remainder into train/val
    try:
        train_val, test = train_test_split(
            df, test_size=TEST_RATIO, random_state=SEED, stratify=df["combined"]
        )
    except ValueError:
        train_val, test = train_test_split(df, test_size=TEST_RATIO, random_state=SEED)

    val_frac_of_rest = VAL_RATIO / max(TRAIN_RATIO + VAL_RATIO, 1e-8)
    try:
        train, val = train_test_split(
            train_val, test_size=val_frac_of_rest, random_state=SEED, stratify=train_val["combined"]
        )
    except ValueError:
        train, val = train_test_split(train_val, test_size=val_frac_of_rest, random_state=SEED)

    train = train.copy(); val = val.copy(); test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)


print(f"Loading images from: {IMAGES_DIR}")
meta = load_from_images_folder(IMAGES_DIR)
print(f"Raw image files found: {len(meta):,}")
print("\nBy group:")
print(meta["group"].value_counts().to_string())
print("\nBy category (produce):")
print(meta["category"].value_counts().to_string())
print("\nBy freshness stage:")
print(meta["freshness"].value_counts().to_string())

# ---- Validate images (skip corrupted) ----
print("\nValidating images…")
paths_list = meta["filepath"].tolist()
if not VALIDATE_IMAGES:
    valid_flags = [Path(fp).is_file() for fp in tqdm(paths_list, desc="Exists check")]
else:
    valid_flags = []
    if MAX_VALIDATE is not None and len(paths_list) > MAX_VALIDATE:
        print(f"Sample QC on {MAX_VALIDATE} images (MAX_VALIDATE set).")
        check_idx = set(np.random.RandomState(SEED).choice(len(paths_list), MAX_VALIDATE, replace=False))
    else:
        check_idx = set(range(len(paths_list)))
    for i, fp in enumerate(tqdm(paths_list, desc="QC images")):
        p = Path(fp)
        if not p.is_file():
            valid_flags.append(False)
        elif i in check_idx:
            valid_flags.append(is_image_readable(p))
        else:
            valid_flags.append(True)

meta["is_valid"] = valid_flags
n_bad = int((~meta["is_valid"]).sum())
bad_paths = meta.loc[~meta["is_valid"], "filepath"].tolist()
print(f"Corrupted / missing images skipped: {n_bad:,}")
if n_bad:
    pd.Series(bad_paths).to_csv(WORKING_DIR / "corrupted_images.txt", index=False, header=False)

meta = meta[meta["is_valid"]].reset_index(drop=True)
print(f"Usable images: {len(meta):,}")

# Drop empty labels
meta = meta[
    (meta["category"].astype(str).str.len() > 0)
    & (meta["freshness"].astype(str).str.len() > 0)
    & (meta["freshness"] != "Unknown")
].reset_index(drop=True)
if len(meta) == 0:
    raise RuntimeError("No labeled images left after filtering. Check folder structure.")

# Stratified train / val / test from this folder only
meta = assign_splits(meta)
print("\nFinal split counts:")
print(meta["split"].value_counts().to_string())
print(f"\nCategories: {sorted(meta['category'].unique().tolist())}")
print(f"Freshness : {sorted(meta['freshness'].unique().tolist())}")
meta.head(3)
'''
)

# =============================================================================
# CELL 5 — Label encoders
# =============================================================================
md("## 5. Label Encoders & Class Weights")

code(
    r'''
# ============================================================
# Build label spaces from training split only
# ============================================================
train_meta = meta[meta["split"] == "train"].copy()
val_meta = meta[meta["split"] == "val"].copy()
test_meta = meta[meta["split"] == "test"].copy() if "test" in meta["split"].unique() else val_meta.copy()

CATEGORY_CLASSES = sorted(train_meta["category"].unique().tolist())
FRESHNESS_CLASSES = sorted(
    train_meta["freshness"].unique().tolist(),
    key=lambda x: (FRESHNESS_ORDER.index(x) if x in FRESHNESS_ORDER else 999, x),
)
COMBINED_CLASSES = sorted(train_meta["combined"].unique().tolist())

cat2id = {c: i for i, c in enumerate(CATEGORY_CLASSES)}
fresh2id = {c: i for i, c in enumerate(FRESHNESS_CLASSES)}
comb2id = {c: i for i, c in enumerate(COMBINED_CLASSES)}
id2cat = {i: c for c, i in cat2id.items()}
id2fresh = {i: c for c, i in fresh2id.items()}
id2comb = {i: c for c, i in comb2id.items()}

NUM_CATEGORIES = len(CATEGORY_CLASSES)
NUM_FRESHNESS = len(FRESHNESS_CLASSES)
NUM_COMBINED = len(COMBINED_CLASSES)

print(f"Categories ({NUM_CATEGORIES}): {CATEGORY_CLASSES}")
print(f"Freshness  ({NUM_FRESHNESS}): {FRESHNESS_CLASSES}")
print(f"Combined   ({NUM_COMBINED})")

# Filter val/test to known classes (drop rare unseen if any)
def _filter_known(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["category"].isin(CATEGORY_CLASSES)
        & df["freshness"].isin(FRESHNESS_CLASSES)
        & df["combined"].isin(COMBINED_CLASSES)
    )
    return df[m].reset_index(drop=True)

val_meta = _filter_known(val_meta)
test_meta = _filter_known(test_meta)
print(f"\nTrain={len(train_meta):,}  Val={len(val_meta):,}  Test={len(test_meta):,}")

# Encode
for df_ in (train_meta, val_meta, test_meta, meta):
    df_["category_id"] = df_["category"].map(cat2id)
    df_["freshness_id"] = df_["freshness"].map(fresh2id)
    df_["combined_id"] = df_["combined"].map(comb2id)

# Class weights (sklearn balanced) — for primary evaluation head
def make_class_weights(y: np.ndarray, classes: List[int]) -> Dict[int, float]:
    cw = compute_class_weight(class_weight="balanced", classes=np.array(classes), y=y)
    return {int(c): float(w) for c, w in zip(classes, cw)}

cw_category = make_class_weights(train_meta["category_id"].values, list(range(NUM_CATEGORIES)))
cw_freshness = make_class_weights(train_meta["freshness_id"].values, list(range(NUM_FRESHNESS)))
cw_combined = make_class_weights(train_meta["combined_id"].values, list(range(NUM_COMBINED)))

print("\nFreshness class weights:")
for i, n in id2fresh.items():
    print(f"  {n:20s} → {cw_freshness[i]:.3f}")

# Persist label maps
label_maps = {
    "category_classes": CATEGORY_CLASSES,
    "freshness_classes": FRESHNESS_CLASSES,
    "combined_classes": COMBINED_CLASSES,
    "rsl_map": DEFAULT_RSL_MAP,
    "task_mode": TASK_MODE,
    "backbone": BACKBONE,
    "img_size": list(IMG_SIZE),
}
with open(WORKING_DIR / "label_maps.json", "w", encoding="utf-8") as f:
    json.dump(label_maps, f, indent=2)
print(f"\n✓ Saved {WORKING_DIR / 'label_maps.json'}")
'''
)

# =============================================================================
# CELL 6 — EDA
# =============================================================================
md("## 6. Exploratory Data Analysis")

code(
    r'''
# ============================================================
# Distributions, stats, sample grid
# ============================================================

def plot_value_counts(series, title, ax, top_n=None, color="#2a9d8f"):
    vc = series.value_counts()
    if top_n:
        vc = vc.head(top_n)
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color=color, orient="h")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_value_counts(meta["category"], "Category distribution", axes[0, 0], color="#264653")
plot_value_counts(meta["freshness"], "Freshness stage distribution", axes[0, 1], color="#e76f51")
plot_value_counts(meta["split"], "Split distribution", axes[1, 0], color="#457b9d")
# Remaining shelf life hist
sns.histplot(meta["remaining_shelf_life"].dropna(), bins=15, ax=axes[1, 1], color="#2a9d8f")
axes[1, 1].set_title("Remaining shelf life (days)")
axes[1, 1].set_xlabel("Days")
plt.tight_layout()
fig.savefig(WORKING_DIR / "eda_distributions.png", bbox_inches="tight")
plt.show()

# Combined heatmap: category × freshness
ct = pd.crosstab(meta["category"], meta["freshness"])
# order freshness cols
cols = [c for c in FRESHNESS_ORDER if c in ct.columns] + [c for c in ct.columns if c not in FRESHNESS_ORDER]
ct = ct.reindex(columns=cols)
fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(ct))))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title("Category × Freshness counts")
plt.tight_layout()
fig.savefig(WORKING_DIR / "eda_category_freshness_heatmap.png", bbox_inches="tight")
plt.show()

# Image size statistics (sample up to 800 images for speed)
sample_paths = meta["filepath"].sample(min(800, len(meta)), random_state=SEED).tolist()
widths, heights, aspects = [], [], []
for fp in tqdm(sample_paths, desc="Image stats"):
    try:
        with Image.open(fp) as im:
            w, h = im.size
            widths.append(w); heights.append(h); aspects.append(w / max(h, 1))
    except Exception:
        pass

stats_df = pd.DataFrame({"width": widths, "height": heights, "aspect": aspects})
print("Image size statistics (sample):")
display(stats_df.describe().T)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
sns.histplot(stats_df["width"], ax=axes[0], color="#264653"); axes[0].set_title("Width")
sns.histplot(stats_df["height"], ax=axes[1], color="#2a9d8f"); axes[1].set_title("Height")
sns.histplot(stats_df["aspect"], ax=axes[2], color="#e76f51"); axes[2].set_title("Aspect ratio")
plt.tight_layout()
fig.savefig(WORKING_DIR / "eda_image_stats.png", bbox_inches="tight")
plt.show()

# Sample images grid by freshness
def show_samples(df: pd.DataFrame, group_col: str, n_per: int = 3, max_groups: int = 8):
    groups = list(df[group_col].value_counts().index)[:max_groups]
    fig, axes = plt.subplots(len(groups), n_per, figsize=(3 * n_per, 2.6 * len(groups)))
    if len(groups) == 1:
        axes = np.array([axes])
    for i, g in enumerate(groups):
        subset = df[df[group_col] == g].sample(min(n_per, (df[group_col] == g).sum()), random_state=SEED)
        for j in range(n_per):
            ax = axes[i, j] if n_per > 1 else axes[i]
            ax.axis("off")
            if j < len(subset):
                row = subset.iloc[j]
                try:
                    im = Image.open(row["filepath"]).convert("RGB")
                    ax.imshow(im)
                    ax.set_title(f"{g}\n{row['category']}", fontsize=8)
                except Exception:
                    ax.set_title("load error", fontsize=8)
            if j == 0:
                ax.set_ylabel(g)
    plt.suptitle(f"Sample images by {group_col}", y=1.01)
    plt.tight_layout()
    fig.savefig(WORKING_DIR / f"eda_samples_by_{group_col}.png", bbox_inches="tight")
    plt.show()

show_samples(meta, "freshness", n_per=EDA_SAMPLES_PER_CLASS, max_groups=min(8, NUM_FRESHNESS))
show_samples(meta, "category", n_per=EDA_SAMPLES_PER_CLASS, max_groups=min(8, NUM_CATEGORIES))

# Save summary CSV
summary = (
    meta.groupby(["split", "category", "freshness"], as_index=False)
    .agg(count=("filepath", "count"), mean_rsl=("remaining_shelf_life", "mean"))
    .sort_values(["split", "category", "freshness"])
)
summary.to_csv(WORKING_DIR / "dataset_summary.csv", index=False)
print(f"✓ Saved dataset_summary.csv → {WORKING_DIR}")
'''
)

# =============================================================================
# CELL 7 — Preprocessing & augmentations
# =============================================================================
md(
    r"""
## 7. Preprocessing & Augmentations

Augmentations are **mild** so natural color shifts (browning, yellowing, mold tint) remain informative:

- Random flip H/V, small rotation, zoom, brightness / contrast
- CLAHE-like local contrast (approx. via random adaptive contrast)
- Slight Gaussian blur & Gaussian noise
"""
)

code(
    r'''
# ============================================================
# Preprocess + Augment (graph-mode friendly)
# ============================================================

def get_preprocess_fn(backbone_name: str):
    """Return the Keras application preprocess_input for the backbone."""
    name = backbone_name.lower()
    if name in ("efficientnetv2b0", "efficientnetb3"):
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as pp
        # EfficientNetB3 uses efficientnet preprocess; V2 uses efficientnet_v2
        if name == "efficientnetb3":
            from tensorflow.keras.applications.efficientnet import preprocess_input as pp
        return pp
    if name == "convnext_tiny":
        try:
            from tensorflow.keras.applications.convnext import preprocess_input as pp
            return pp
        except Exception:
            return tf.keras.applications.imagenet_utils.preprocess_input
    if name == "densenet121":
        from tensorflow.keras.applications.densenet import preprocess_input as pp
        return pp
    if name == "mobilenetv3":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as pp
        return pp
    if name == "resnet50":
        from tensorflow.keras.applications.resnet import preprocess_input as pp
        return pp
    # default: scale 0-1 then ImageNet mean/std-ish via efficientnet
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as pp
    return pp


PREPROCESS_FN = get_preprocess_fn(BACKBONE)


def decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    """Load image bytes → RGB float32 [0,255] → resize."""
    raw = tf.io.read_file(path)
    # decode_image supports jpeg/png/gif/bmp; expand_animations=False for static
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = img * 255.0  # back to [0,255] for application preprocess_input
    return img


def augment_image(img: tf.Tensor) -> tf.Tensor:
    """
    Realistic train-time augmentations.
    Input/output: float32 RGB in [0, 255].
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Small rotation via 90-degree k + fine affine approx with crop/pad
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)

    # Random zoom (central crop + resize)
    zoom = tf.random.uniform([], 0.85, 1.0)
    h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * zoom, tf.int32)
    w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * zoom, tf.int32)
    img = tf.image.random_crop(img, size=[h, w, 3])
    img = tf.image.resize(img, IMG_SIZE)

    # Brightness / contrast (keep mild — color is a freshness cue)
    img = tf.image.random_brightness(img, max_delta=0.12 * 255.0)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    # Mild saturation (avoid destroying browning signals)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)

    # Slight Gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=3.0, dtype=img.dtype)
    img = img + noise

    # Approximate slight blur: average pool with random application
    def _blur(x):
        x4 = x[None, ...]
        x4 = tf.nn.avg_pool2d(x4, ksize=3, strides=1, padding="SAME")
        return x4[0]

    img = tf.cond(tf.random.uniform([]) < 0.25, lambda: _blur(img), lambda: img)

    # CLAHE-like: random local contrast via tile-ish rescale of luminance
    def _clahe_approx(x):
        # Convert to YUV-ish: enhance Y channel contrast slightly
        gray = tf.image.rgb_to_grayscale(x / 255.0)
        gray_eq = tf.image.adjust_contrast(gray, contrast_factor=1.15)
        # Blend enhanced luminance back
        yuv = tf.image.rgb_to_yuv(x / 255.0)
        y = gray_eq
        u = yuv[..., 1:2]
        v = yuv[..., 2:3]
        yuv2 = tf.concat([y, u, v], axis=-1)
        rgb = tf.image.yuv_to_rgb(yuv2) * 255.0
        return rgb

    img = tf.cond(tf.random.uniform([]) < 0.35, lambda: _clahe_approx(img), lambda: img)

    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def apply_preprocess(img: tf.Tensor) -> tf.Tensor:
    """Backbone-specific preprocess_input (expects [0,255] RGB)."""
    # preprocess_input may expect numpy; wrap via tf.numpy_function if needed.
    # Keras application preprocess_input works on tensors in TF2.
    return PREPROCESS_FN(img)
'''
)

# =============================================================================
# CELL 8 — tf.data pipeline
# =============================================================================
md("## 8. `tf.data` Pipeline")

code(
    r'''
# ============================================================
# Build tf.data datasets
# ============================================================

def _df_to_arrays(df: pd.DataFrame):
    paths = df["filepath"].astype(str).values
    cat = df["category_id"].astype(np.int32).values
    fresh = df["freshness_id"].astype(np.int32).values
    comb = df["combined_id"].astype(np.int32).values
    rsl = df["remaining_shelf_life"].astype(np.float32).values
    return paths, cat, fresh, comb, rsl


def build_dataset(
    df: pd.DataFrame,
    training: bool = False,
    batch_size: int = BATCH_SIZE,
    task_mode: str = TASK_MODE,
) -> tf.data.Dataset:
    paths, cat, fresh, comb, rsl = _df_to_arrays(df)

    if task_mode == "combined":
        ds = tf.data.Dataset.from_tensor_slices((paths, comb))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, cat, fresh))

    if training:
        ds = ds.shuffle(min(SHUFFLE_BUFFER, len(df)), seed=SEED, reshuffle_each_iteration=True)

    def _load_multitask(path, y_cat, y_fresh):
        img = decode_and_resize(path)
        if training:
            img = augment_image(img)
        img = apply_preprocess(img)
        # one-hot
        y_cat_oh = tf.one_hot(y_cat, NUM_CATEGORIES)
        y_fresh_oh = tf.one_hot(y_fresh, NUM_FRESHNESS)
        return img, {"category": y_cat_oh, "freshness": y_fresh_oh}

    def _load_combined(path, y_comb):
        img = decode_and_resize(path)
        if training:
            img = augment_image(img)
        img = apply_preprocess(img)
        y_oh = tf.one_hot(y_comb, NUM_COMBINED)
        return img, y_oh

    if task_mode == "combined":
        ds = ds.map(_load_combined, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=not training)
    else:
        ds = ds.map(_load_multitask, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=not training)

    if CACHE_TRAIN and training:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = build_dataset(train_meta, training=True)
val_ds = build_dataset(val_meta, training=False)
test_ds = build_dataset(test_meta, training=False)

# Peek one batch
for batch in train_ds.take(1):
    xb, yb = batch
    print("Batch images:", xb.shape, xb.dtype)
    if isinstance(yb, dict):
        for k, v in yb.items():
            print(f"  label[{k}]:", v.shape, v.dtype)
    else:
        print("  labels:", yb.shape, yb.dtype)

print(f"\nSteps/epoch train ≈ {math.ceil(len(train_meta) / BATCH_SIZE)}")
print(f"Steps/epoch val   ≈ {math.ceil(len(val_meta) / BATCH_SIZE)}")
'''
)

# =============================================================================
# CELL 9 — Model
# =============================================================================
md("## 9. Model — Transfer Learning Backbone + Multi-Task Heads")

code(
    r'''
# ============================================================
# Backbone factory + classifier heads
# ============================================================

def build_backbone(
    name: str,
    input_shape: Tuple[int, int, int],
    pretrained: bool = True,
    input_tensor=None,
):
    """Build backbone. Prefer input_tensor so layers are inlined for Grad-CAM."""
    weights = "imagenet" if pretrained else None
    name = name.lower()
    kwargs = dict(include_top=False, weights=weights)
    if input_tensor is not None:
        kwargs["input_tensor"] = input_tensor
    else:
        kwargs["input_shape"] = input_shape

    if name == "efficientnetv2b0":
        base = EfficientNetV2B0(**kwargs)
    elif name == "efficientnetb3":
        base = EfficientNetB3(**kwargs)
    elif name == "convnext_tiny":
        if not HAS_CONVNEXT:
            raise ImportError("ConvNeXtTiny requires TensorFlow >= 2.11")
        base = ConvNeXtTiny(**kwargs)
    elif name == "densenet121":
        base = DenseNet121(**kwargs)
    elif name == "mobilenetv3":
        base = MobileNetV3Large(**kwargs)
    elif name == "resnet50":
        base = ResNet50(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return base


def build_model(
    backbone_name: str = BACKBONE,
    task_mode: str = TASK_MODE,
    dropout: float = DROPOUT_RATE,
    pretrained: bool = PRETRAINED,
) -> keras.Model:
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    # input_tensor inlines backbone layers into the Functional graph (Grad-CAM friendly)
    base = build_backbone(
        backbone_name, (*IMG_SIZE, 3), pretrained=pretrained, input_tensor=inputs
    )
    base.trainable = False  # stage 1

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    shared = layers.Dense(512, activation="relu", name="shared_dense")(x)
    shared = layers.BatchNormalization(name="bn_shared")(shared)
    shared = layers.Dropout(dropout * 0.75, name="dropout_2")(shared)

    # float32 outputs for numeric stability under mixed precision
    if task_mode == "combined":
        outputs = layers.Dense(
            NUM_COMBINED, activation="softmax", dtype="float32", name="combined"
        )(shared)
        model = keras.Model(inputs, outputs, name=f"{backbone_name}_combined")
    else:
        cat_h = layers.Dense(256, activation="relu", name="cat_dense")(shared)
        cat_h = layers.Dropout(dropout * 0.5, name="cat_dropout")(cat_h)
        category = layers.Dense(
            NUM_CATEGORIES, activation="softmax", dtype="float32", name="category"
        )(cat_h)

        fr_h = layers.Dense(256, activation="relu", name="fresh_dense")(shared)
        fr_h = layers.Dropout(dropout * 0.5, name="fresh_dropout")(fr_h)
        freshness = layers.Dense(
            NUM_FRESHNESS, activation="softmax", dtype="float32", name="freshness"
        )(fr_h)

        model = keras.Model(
            inputs,
            {"category": category, "freshness": freshness},
            name=f"{backbone_name}_multitask",
        )

    model._backbone = base  # stash for freeze/unfreeze
    return model


def compile_model(model: keras.Model, lr: float, task_mode: str = TASK_MODE, stage: str = "head"):
    # AdamW
    try:
        opt = optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
    except Exception:
        # Fallback for older TF
        opt = optimizers.Adam(learning_rate=lr)

    # Loss scale for mixed precision is handled automatically by Keras when policy is mixed_float16
    if task_mode == "combined":
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
        metrics = [
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ]
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
    else:
        losses = {
            "category": keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            "freshness": keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        }
        loss_weights = {
            "category": LOSS_WEIGHT_CATEGORY,
            "freshness": LOSS_WEIGHT_FRESHNESS,
        }
        metrics = {
            "category": [keras.metrics.CategoricalAccuracy(name="accuracy")],
            "freshness": [keras.metrics.CategoricalAccuracy(name="accuracy")],
        }
        model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics)
    print(f"Compiled model stage={stage} lr={lr} task_mode={task_mode}")
    return model


def set_backbone_trainable(model: keras.Model, trainable: bool, unfreeze_ratio: float = UNFREEZE_RATIO):
    """Freeze all / unfreeze top fraction of backbone layers.

    With input_tensor inlining, backbone layers live on the full model; we still
    prefer model._backbone.layers when available, else heuristic on early layers.
    """
    base = getattr(model, "_backbone", None)
    if base is None:
        for layer in model.layers:
            if isinstance(layer, keras.Model) and len(layer.layers) > 20:
                base = layer
                break

    # Layer list to freeze: backbone layers if available, else all except head names
    head_prefixes = (
        "gap", "bn_head", "dropout", "shared_dense", "bn_shared",
        "cat_", "fresh_", "category", "freshness", "combined",
    )
    if base is not None and hasattr(base, "layers") and len(base.layers) > 5:
        target_layers = list(base.layers)
    else:
        target_layers = [
            lyr for lyr in model.layers
            if not any(lyr.name.startswith(p) for p in head_prefixes)
            and lyr.name != "image"
        ]

    if not trainable:
        for layer in target_layers:
            layer.trainable = False
        if base is not None:
            base.trainable = False
        print(f"Backbone frozen ({len(target_layers)} layers).")
        return

    if base is not None:
        base.trainable = True
    n = len(target_layers)
    freeze_until = int(n * (1.0 - unfreeze_ratio))
    for i, layer in enumerate(target_layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False  # standard fine-tune practice
    n_train = sum(1 for l in target_layers if l.trainable)
    print(f"Backbone fine-tune: {n_train}/{n} layers trainable (top {unfreeze_ratio:.0%}).")


model = build_model()
model = compile_model(model, lr=LEARNING_RATE_HEAD, stage="head")
model.summary()
'''
)

# =============================================================================
# CELL 10 — Callbacks
# =============================================================================
md("## 10. Callbacks & Class-Weight Mapping")

code(
    r'''
# ============================================================
# Callbacks
# ============================================================
ckpt_path = WORKING_DIR / f"best_model_{BACKBONE}_{TASK_MODE}.keras"

def make_callbacks(stage: str, use_cosine: bool = USE_COSINE_LR, steps_per_epoch: int = 100):
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.CSVLogger(str(WORKING_DIR / f"history_{stage}.csv"), append=False),
        callbacks.TerminateOnNaN(),
    ]
    if use_cosine:
        # Cosine over this stage's epochs
        epochs = EPOCHS_HEAD if stage == "head" else EPOCHS_FINETUNE
        total_steps = max(1, steps_per_epoch * epochs)
        cosine = optimizers.schedules.CosineDecay(
            initial_learning_rate=(LEARNING_RATE_HEAD if stage == "head" else LEARNING_RATE_FINETUNE),
            decay_steps=total_steps,
            alpha=MIN_LR / max(LEARNING_RATE_FINETUNE, 1e-12),
        )
        # LearningRateScheduler wrapping the schedule value each epoch mid-point
        def _lr_cb(epoch, lr):
            # approximate epoch-level cosine
            step = epoch * steps_per_epoch
            return float(cosine(step))
        cbs.append(callbacks.LearningRateScheduler(_lr_cb, verbose=1))
    else:
        cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=1,
            )
        )
    return cbs


# Sample-weight arrays for multi-output training (per-sample class weights)
def build_sample_weights(df: pd.DataFrame, task_mode: str = TASK_MODE):
    if task_mode == "combined":
        w = df["combined_id"].map(cw_combined).astype(np.float32).values
        return w
    w_cat = df["category_id"].map(cw_category).astype(np.float32).values
    w_fr = df["freshness_id"].map(cw_freshness).astype(np.float32).values
    return {"category": w_cat, "freshness": w_fr}


# Inject sample weights into datasets via a re-map
def build_dataset_weighted(df: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
    paths, cat, fresh, comb, rsl = _df_to_arrays(df)
    sw = build_sample_weights(df)

    if TASK_MODE == "combined":
        ds = tf.data.Dataset.from_tensor_slices((paths, comb, sw.astype(np.float32)))
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (paths, cat, fresh, sw["category"].astype(np.float32), sw["freshness"].astype(np.float32))
        )

    if training:
        ds = ds.shuffle(min(SHUFFLE_BUFFER, len(df)), seed=SEED, reshuffle_each_iteration=True)

    def _mt(path, y_cat, y_fresh, w_cat, w_fresh):
        img = decode_and_resize(path)
        if training:
            img = augment_image(img)
        img = apply_preprocess(img)
        y = {
            "category": tf.one_hot(y_cat, NUM_CATEGORIES),
            "freshness": tf.one_hot(y_fresh, NUM_FRESHNESS),
        }
        w = {"category": w_cat, "freshness": w_fresh}
        return img, y, w

    def _cb(path, y_comb, w):
        img = decode_and_resize(path)
        if training:
            img = augment_image(img)
        img = apply_preprocess(img)
        return img, tf.one_hot(y_comb, NUM_COMBINED), w

    if TASK_MODE == "combined":
        ds = ds.map(_cb, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=not training)
    else:
        ds = ds.map(_mt, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=not training)

    if CACHE_TRAIN and training:
        ds = ds.cache()
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


train_ds_w = build_dataset_weighted(train_meta, training=True)
val_ds_w = build_dataset_weighted(val_meta, training=False)
test_ds_w = build_dataset_weighted(test_meta, training=False)

steps_train = math.ceil(len(train_meta) / BATCH_SIZE)
steps_val = math.ceil(len(val_meta) / BATCH_SIZE)
print(f"Pipeline ready. steps_train={steps_train}, steps_val={steps_val}")
print(f"Checkpoint → {ckpt_path}")
'''
)

# =============================================================================
# CELL 11 — Stage 1 training
# =============================================================================
md("## 11. Training Stage 1 — Frozen Backbone (Classification Head)")

code(
    r'''
set_backbone_trainable(model, trainable=False)
model = compile_model(model, lr=LEARNING_RATE_HEAD, stage="head")

t0 = time.time()
history_head = model.fit(
    train_ds_w,
    validation_data=val_ds_w,
    epochs=EPOCHS_HEAD,
    callbacks=make_callbacks("head", steps_per_epoch=steps_train),
    verbose=1,
)
t_head = time.time() - t0
print(f"Stage 1 done in {t_head/60:.1f} min")

# Persist stage-1 weights snapshot
model.save(WORKING_DIR / f"model_stage1_{BACKBONE}.keras")
'''
)

# =============================================================================
# CELL 12 — Stage 2 fine-tune
# =============================================================================
md("## 12. Training Stage 2 — Fine-Tune Upper Backbone Layers")

code(
    r'''
set_backbone_trainable(model, trainable=True, unfreeze_ratio=UNFREEZE_RATIO)
model = compile_model(model, lr=LEARNING_RATE_FINETUNE, stage="finetune")

t0 = time.time()
history_ft = model.fit(
    train_ds_w,
    validation_data=val_ds_w,
    epochs=EPOCHS_FINETUNE,
    callbacks=make_callbacks("finetune", steps_per_epoch=steps_train),
    verbose=1,
)
t_ft = time.time() - t0
print(f"Stage 2 done in {t_ft/60:.1f} min")

# Load best checkpoint if present
if ckpt_path.exists():
    print(f"Loading best checkpoint: {ckpt_path}")
    model = keras.models.load_model(ckpt_path)
    # re-stash backbone ref for Grad-CAM
    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 20:
            model._backbone = layer
            break

model.save(WORKING_DIR / f"model_final_{BACKBONE}_{TASK_MODE}.keras")
print("✓ Final model saved.")
'''
)

# =============================================================================
# CELL 13 — Training curves
# =============================================================================
md("## 13. Training Curves")

code(
    r'''
def merge_histories(*hist_list):
    merged = defaultdict(list)
    for h in hist_list:
        if h is None:
            continue
        for k, v in h.history.items():
            merged[k].extend(list(v))
    return dict(merged)


full_hist = merge_histories(history_head, history_ft)
# Save raw history
with open(WORKING_DIR / "training_history.json", "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in full_hist.items()}, f, indent=2)


def plot_history(hist: dict):
    # Identify keys
    loss_keys = [k for k in hist if "loss" in k and not k.startswith("val_")]
    acc_keys = [k for k in hist if "accuracy" in k and not k.startswith("val_")]

    n_rows = 1 + (1 if acc_keys else 0)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    # Total loss
    ax = axes[0, 0]
    if "loss" in hist:
        ax.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="val")
    ax.set_title("Total loss"); ax.set_xlabel("Epoch"); ax.legend()

    # Per-head losses if present
    ax = axes[0, 1]
    plotted = False
    for k in loss_keys:
        if k == "loss":
            continue
        ax.plot(hist[k], label=k)
        vk = "val_" + k
        if vk in hist:
            ax.plot(hist[vk], label=vk, linestyle="--")
        plotted = True
    if not plotted and "loss" in hist:
        ax.plot(hist["loss"], label="train_loss")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label="val_loss")
    ax.set_title("Loss components"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

    if acc_keys:
        ax = axes[1, 0]
        for k in acc_keys:
            ax.plot(hist[k], label=k)
            vk = "val_" + k
            if vk in hist:
                ax.plot(hist[vk], label=vk, linestyle="--")
        ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

        ax = axes[1, 1]
        # LR if logged
        if "lr" in hist:
            ax.plot(hist["lr"], label="lr")
            ax.set_yscale("log")
            ax.set_title("Learning rate")
        else:
            ax.axis("off")
            ax.text(0.1, 0.5, "LR curve not logged\n(ReduceLROnPlateau / Cosine active)", fontsize=10)

    plt.tight_layout()
    fig.savefig(WORKING_DIR / "training_curves.png", bbox_inches="tight")
    plt.show()


plot_history(full_hist)
'''
)

# =============================================================================
# CELL 14 — Evaluation helpers
# =============================================================================
md("## 14. Evaluation — Predictions, Metrics, Confusion Matrix, ROC")

code(
    r'''
# ============================================================
# Collect predictions
# ============================================================

def predict_dataframe(model: keras.Model, df: pd.DataFrame, task_mode: str = TASK_MODE):
    ds = build_dataset(df, training=False)
    preds = model.predict(ds, verbose=1)
    out = df.copy()

    if task_mode == "combined":
        prob = np.asarray(preds)
        y_true = df["combined_id"].values
        y_pred = prob.argmax(axis=1)
        conf = prob.max(axis=1)
        out["pred_combined_id"] = y_pred
        out["pred_combined"] = [id2comb[i] for i in y_pred]
        out["confidence"] = conf
        # split combined into cat/fresh if possible
        cats, freshes = [], []
        for name in out["pred_combined"]:
            c, f = parse_folder_label(name)
            # parse_folder_label for Apple_Fresh works
            if c is None and "_" in name:
                c, f = name.rsplit("_", 1)[0], name.rsplit("_", 1)[1]
            cats.append(c if c else "Unknown")
            freshes.append(f if f else name)
        out["pred_category"] = cats
        out["pred_freshness"] = freshes
        out["y_true_primary"] = y_true
        out["y_pred_primary"] = y_pred
        out["_prob_primary"] = list(prob)
    else:
        prob_cat = np.asarray(preds["category"])
        prob_fr = np.asarray(preds["freshness"])
        pred_cat = prob_cat.argmax(axis=1)
        pred_fr = prob_fr.argmax(axis=1)
        conf_cat = prob_cat.max(axis=1)
        conf_fr = prob_fr.max(axis=1)
        # joint confidence = geometric mean
        conf = np.sqrt(conf_cat * conf_fr)
        out["pred_category_id"] = pred_cat
        out["pred_freshness_id"] = pred_fr
        out["pred_category"] = [id2cat[i] for i in pred_cat]
        out["pred_freshness"] = [id2fresh[i] for i in pred_fr]
        out["confidence_category"] = conf_cat
        out["confidence_freshness"] = conf_fr
        out["confidence"] = conf
        # primary for CM / ROC = freshness (shelf-life relevant)
        out["y_true_primary"] = df["freshness_id"].values
        out["y_pred_primary"] = pred_fr
        out["_prob_primary"] = list(prob_fr)
        out["y_true_category"] = df["category_id"].values
        out["y_pred_category"] = pred_cat
        out["_prob_category"] = list(prob_cat)

    # Shelf life from predicted freshness
    out["pred_remaining_shelf_life"] = out["pred_freshness"].map(
        lambda x: rsl_from_freshness(x, fallback=0.0)
    )
    return out


print("Running inference on validation set…")
val_pred = predict_dataframe(model, val_meta)
print("Running inference on test set…")
test_pred = predict_dataframe(model, test_meta)

def _save_pred_csv(df: pd.DataFrame, path: Path):
    drop_cols = [c for c in df.columns if c.startswith("_prob")]
    df.drop(columns=drop_cols, errors="ignore").to_csv(path, index=False)

_save_pred_csv(val_pred, WORKING_DIR / "val_predictions.csv")
_save_pred_csv(test_pred, WORKING_DIR / "test_predictions.csv")
print("✓ Prediction CSVs saved.")
'''
)

code(
    r'''
# ============================================================
# Metrics suite
# ============================================================

def evaluate_classification(y_true, y_pred, class_names, title_prefix, prob=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}\n{title_prefix}  |  Accuracy = {acc:.4f}\n{'='*60}")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print(report)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True
    )
    pd.DataFrame(report_dict).T.to_csv(WORKING_DIR / f"{title_prefix.lower().replace(' ', '_')}_classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(class_names)), max(5, 0.5 * len(class_names))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title_prefix} — Confusion Matrix (acc={acc:.3f})")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    safe = title_prefix.lower().replace(" ", "_")
    fig.savefig(WORKING_DIR / f"confusion_matrix_{safe}.png", bbox_inches="tight")
    plt.show()

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        per_class[name] = float((y_pred[mask] == i).mean()) if mask.any() else float("nan")
    pca = pd.Series(per_class).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(pca))))
    sns.barplot(x=pca.values, y=pca.index, ax=ax, color="#2a9d8f", orient="h")
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy"); ax.set_title(f"{title_prefix} — Per-class accuracy")
    plt.tight_layout()
    fig.savefig(WORKING_DIR / f"per_class_accuracy_{safe}.png", bbox_inches="tight")
    plt.show()

    # ROC (one-vs-rest) if probabilities provided
    if prob is not None and len(class_names) >= 2:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        if y_bin.ndim == 1:
            y_bin = np.vstack([1 - y_bin, y_bin]).T
            prob_use = prob
        else:
            prob_use = prob
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(class_names):
            if y_bin.shape[1] <= i:
                continue
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob_use[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{title_prefix} — ROC (OvR)")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        fig.savefig(WORKING_DIR / f"roc_{safe}.png", bbox_inches="tight")
        plt.show()

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class_accuracy": per_class,
    }
    return metrics


primary_names = FRESHNESS_CLASSES if TASK_MODE != "combined" else COMBINED_CLASSES
prob_primary = np.stack(test_pred["_prob_primary"].to_list())

print("\n########## TEST SET — PRIMARY HEAD ##########")
metrics_fresh = evaluate_classification(
    test_pred["y_true_primary"].values.astype(int),
    test_pred["y_pred_primary"].values.astype(int),
    primary_names,
    "Test_Primary",
    prob=prob_primary,
)

if TASK_MODE == "multitask":
    print("\n########## TEST SET — CATEGORY ##########")
    metrics_cat = evaluate_classification(
        test_pred["y_true_category"].values.astype(int),
        test_pred["y_pred_category"].values.astype(int),
        CATEGORY_CLASSES,
        "Test_Category",
        prob=np.stack(test_pred["_prob_category"].to_list()),
    )
else:
    metrics_cat = {}

# Shelf-life regression-style metrics (from mapped days)
rsl_true = test_pred["remaining_shelf_life"].astype(float).values
rsl_pred = test_pred["pred_remaining_shelf_life"].astype(float).values
mae = float(np.mean(np.abs(rsl_true - rsl_pred)))
rmse = float(np.sqrt(np.mean((rsl_true - rsl_pred) ** 2)))
print(f"\nRemaining shelf life  MAE={mae:.3f} days  RMSE={rmse:.3f} days")

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, str):
        return obj
    return obj

metrics_all = {
    "primary": metrics_fresh,
    "category": metrics_cat,
    "shelf_life_mae": mae,
    "shelf_life_rmse": rmse,
    "backbone": BACKBONE,
    "task_mode": TASK_MODE,
    "n_test": int(len(test_pred)),
}
with open(WORKING_DIR / "metrics.json", "w") as f:
    json.dump(_json_safe(metrics_all), f, indent=2)
print(f"✓ metrics.json saved → {WORKING_DIR}")
'''
)

# =============================================================================
# CELL 15 — Misclassified analysis
# =============================================================================
md("## 15. Misclassified Image Analysis")

code(
    r'''
def show_misclassified(pred_df: pd.DataFrame, max_show: int = MAX_MISCLASS_SHOW):
    if TASK_MODE == "multitask":
        wrong = pred_df[pred_df["freshness_id"] != pred_df["pred_freshness_id"]].copy()
        wrong["true_label"] = wrong["freshness"]
        wrong["pred_label"] = wrong["pred_freshness"]
    else:
        wrong = pred_df[pred_df["combined_id"] != pred_df["pred_combined_id"]].copy()
        wrong["true_label"] = wrong["combined"]
        wrong["pred_label"] = wrong["pred_combined"]

    print(f"Misclassified (primary head): {len(wrong):,} / {len(pred_df):,} "
          f"({100*len(wrong)/max(len(pred_df),1):.1f}%)")
    if wrong.empty:
        print("No misclassifications 🎉")
        return wrong

    # highest-confidence mistakes first (most interesting)
    wrong = wrong.sort_values("confidence", ascending=False).head(max_show)

    n = len(wrong)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (_, row) in enumerate(wrong.iterrows()):
        ax = axes[i]
        try:
            im = Image.open(row["filepath"]).convert("RGB")
            ax.imshow(im)
        except Exception:
            pass
        ax.set_title(
            f"T:{row['true_label']}\nP:{row['pred_label']}\n"
            f"{row['category']} conf={row['confidence']:.2f}",
            fontsize=8, color="crimson",
        )
        ax.axis("off")
    plt.suptitle("High-confidence misclassifications", y=1.01)
    plt.tight_layout()
    fig.savefig(WORKING_DIR / "misclassified_samples.png", bbox_inches="tight")
    plt.show()

    wrong.to_csv(WORKING_DIR / "misclassified_test.csv", index=False)
    return wrong


mis_df = show_misclassified(test_pred)
'''
)

# =============================================================================
# CELL 16 — Grad-CAM
# =============================================================================
md("## 16. Grad-CAM Visualization")

code(
    r'''
# ============================================================
# Grad-CAM (nested-backbone safe) + saliency fallback
# ============================================================

def _get_backbone(m: keras.Model) -> Optional[keras.Model]:
    if hasattr(m, "_backbone") and m._backbone is not None:
        return m._backbone
    for layer in m.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 20:
            return layer
    return None


def find_last_conv_layer(m: keras.Model) -> Tuple[keras.Model, str]:
    """Return (owner_model, layer_name) for a 4D feature map."""
    base = _get_backbone(m)
    search = [x for x in (base, m) if x is not None]
    candidates = []
    for owner in search:
        for layer in owner.layers:
            try:
                shape = getattr(layer, "output_shape", None)
            except Exception:
                shape = None
            if isinstance(shape, list):
                shape = shape[0]
            if shape is not None and len(shape) == 4:
                candidates.append((owner, layer.name))
    if not candidates:
        raise ValueError("No conv layer found for Grad-CAM")
    for owner, name in reversed(candidates):
        low = name.lower()
        if any(k in low for k in ["top_conv", "out_conv", "conv_pw", "conv2d", "block"]):
            return owner, name
    return candidates[-1]


def _select_output(preds, output_name: Optional[str], m: keras.Model):
    if isinstance(preds, dict):
        key = output_name or ("freshness" if "freshness" in preds else list(preds.keys())[-1])
        return preds[key]
    if isinstance(preds, (list, tuple)):
        names = getattr(m, "output_names", None)
        if output_name and names and output_name in names:
            return preds[names.index(output_name)]
        return preds[-1]
    return preds


def make_gradcam_heatmap(
    img_array: np.ndarray,
    m: keras.Model,
    last_conv_owner: keras.Model,
    last_conv_name: str,
    pred_index: Optional[int] = None,
    output_name: Optional[str] = None,
):
    """img_array: preprocessed batch (1,H,W,3) → (heatmap HxW, class_index)."""
    img_tensor = tf.convert_to_tensor(img_array)

    # Resolve prediction tensor from multi-output model
    if isinstance(m.output, dict):
        key = output_name or ("freshness" if "freshness" in m.output else list(m.output.keys())[-1])
        pred_tensor = m.output[key]
    elif isinstance(m.outputs, list) and len(m.outputs) > 1:
        names = getattr(m, "output_names", [])
        if output_name and output_name in names:
            pred_tensor = m.outputs[names.index(output_name)]
        else:
            pred_tensor = m.outputs[-1]
    else:
        pred_tensor = m.output

    # Path A: connected Grad-CAM graph (works when backbone is inlined via input_tensor)
    try:
        conv_layer = m.get_layer(last_conv_name)
        grad_model = keras.Model(m.inputs, [conv_layer.output, pred_tensor], name="gradcam_model")
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is not None:
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            heat = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1)
            heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
            return heat.numpy(), int(pred_index)
    except Exception as e:
        print("Grad-CAM connected path note:", e)

    # Path B: backbone feature extractor (may not connect grads to scores)
    try:
        conv_layer = last_conv_owner.get_layer(last_conv_name)
        feat_extractor = keras.Model(
            last_conv_owner.inputs, conv_layer.output, name="gradcam_feat"
        )
        with tf.GradientTape() as tape:
            conv_outputs = feat_extractor(img_tensor, training=False)
            tape.watch(conv_outputs)
            predictions = m(img_tensor, training=False)
            scores = _select_output(predictions, output_name, m)
            if pred_index is None:
                pred_index = int(tf.argmax(scores[0]))
            class_channel = scores[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is not None:
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            heat = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1)
            heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
            return heat.numpy(), int(pred_index)
    except Exception as e:
        print("Grad-CAM feature path note:", e)

    # Path C: input-gradient saliency (always works)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = m(img_tensor, training=False)
        scores = _select_output(predictions, output_name, m)
        if pred_index is None:
            pred_index = int(tf.argmax(scores[0]))
        class_channel = scores[:, pred_index]
    grads = tape.gradient(class_channel, img_tensor)
    sal = tf.reduce_mean(tf.abs(grads[0]), axis=-1)
    sal = sal / (tf.reduce_max(sal) + 1e-8)
    return sal.numpy(), int(pred_index)


def overlay_heatmap_on_image(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    heat = heatmap
    if heat.ndim == 2:
        heat_r = np.array(
            Image.fromarray(np.uint8(255 * heat)).resize((img_uint8.shape[1], img_uint8.shape[0]))
        ).astype(np.float32) / 255.0
    else:
        heat_r = heat
    if HAS_CV2:
        heat_color = cv2.applyColorMap(np.uint8(255 * heat_r), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    else:
        heat_color = np.uint8(255 * plt.cm.jet(heat_r)[:, :, :3])
    return np.uint8(heat_color * alpha + img_uint8 * (1 - alpha))


try:
    LAST_CONV_OWNER, LAST_CONV = find_last_conv_layer(model)
    print(f"Grad-CAM target layer: {LAST_CONV} (owner={LAST_CONV_OWNER.name})")
except Exception as e:
    LAST_CONV_OWNER, LAST_CONV = None, None
    print("Grad-CAM layer discovery failed:", e)


def load_raw_and_preprocessed(path: str):
    raw = np.array(Image.open(path).convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0])))
    tpath = tf.constant(path)
    img = decode_and_resize(tpath)
    img = apply_preprocess(img)
    return raw, np.expand_dims(img.numpy(), 0)


def demo_gradcam(pred_df: pd.DataFrame, n: int = GRADCAM_SAMPLES):
    if LAST_CONV is None:
        print("Skipping Grad-CAM.")
        return
    sample = pred_df.sample(min(n, len(pred_df)), random_state=SEED)
    cols = 4
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    out_name = "freshness" if TASK_MODE == "multitask" else None
    for i, (_, row) in enumerate(sample.iterrows()):
        raw, prep = load_raw_and_preprocessed(row["filepath"])
        try:
            heat, _pidx = make_gradcam_heatmap(
                prep, model, LAST_CONV_OWNER, LAST_CONV,
                pred_index=int(row["y_pred_primary"]),
                output_name=out_name,
            )
            axes[i].imshow(overlay_heatmap_on_image(raw, heat))
            axes[i].set_title(
                f"{row['pred_category']} | {row['pred_freshness']}\n"
                f"RSL={row['pred_remaining_shelf_life']:.0f}d conf={row['confidence']:.2f}",
                fontsize=8,
            )
        except Exception as e:
            axes[i].imshow(raw)
            axes[i].set_title(f"Grad-CAM failed\n{e}", fontsize=7)
        axes[i].axis("off")
    plt.suptitle("Grad-CAM / saliency — freshness head focus", y=1.01)
    plt.tight_layout()
    fig.savefig(WORKING_DIR / "gradcam_samples.png", bbox_inches="tight")
    plt.show()


demo_gradcam(test_pred)
'''
)

# =============================================================================
# CELL 17 — Shelf life prediction API
# =============================================================================
md(
    r"""
## 17. Shelf-Life Prediction API

Convert model outputs → human-readable report:

- Predicted **category**
- Predicted **freshness stage**
- **Remaining shelf life** (days)
- **Confidence** score(s)
"""
)

code(
    r'''
# ============================================================
# Inference helpers
# ============================================================

def preprocess_path_for_model(path: str) -> np.ndarray:
    tpath = tf.constant(str(path))
    img = decode_and_resize(tpath)
    img = apply_preprocess(img)
    return np.expand_dims(img.numpy(), axis=0)


def predict_image(path: str, model: keras.Model = model) -> Dict[str, Any]:
    """Predict shelf-life package for a single image path."""
    path = str(path)
    x = preprocess_path_for_model(path)
    raw = model.predict(x, verbose=0)

    if TASK_MODE == "combined":
        prob = raw[0]
        idx = int(np.argmax(prob))
        label = id2comb[idx]
        conf = float(prob[idx])
        # split label
        cat, fr = parse_folder_label(label)
        if cat is None and "_" in label:
            cat, fr = label.split("_", 1)
        if cat is None:
            cat = "Unknown"
        if fr is None:
            fr = label
        conf_cat = conf
        conf_fr = conf
        topk = np.argsort(prob)[::-1][:3]
        top3 = [(id2comb[i], float(prob[i])) for i in topk]
    else:
        prob_cat = raw["category"][0]
        prob_fr = raw["freshness"][0]
        ic = int(np.argmax(prob_cat))
        ifr = int(np.argmax(prob_fr))
        cat = id2cat[ic]
        fr = id2fresh[ifr]
        conf_cat = float(prob_cat[ic])
        conf_fr = float(prob_fr[ifr])
        conf = float(np.sqrt(conf_cat * conf_fr))
        top3 = [(id2fresh[i], float(prob_fr[i])) for i in np.argsort(prob_fr)[::-1][:3]]

    rsl = float(rsl_from_freshness(fr, fallback=0.0))
    result = {
        "path": path,
        "category": cat,
        "freshness_stage": fr,
        "remaining_shelf_life_days": rsl,
        "confidence": conf,
        "confidence_category": conf_cat,
        "confidence_freshness": conf_fr,
        "top3_freshness": top3,
        "rsl_is_proxy": True,
        "interpretation": (
            "Spoiled — do not consume" if rsl <= 0
            else f"Approximately {rsl:.0f} day(s) of remaining shelf life (proxy estimate)"
        ),
    }
    return result


def predict_images(paths: List[str], show: bool = True) -> pd.DataFrame:
    rows = []
    for p in paths:
        try:
            rows.append(predict_image(p))
        except Exception as e:
            rows.append({"path": p, "error": str(e)})
    res = pd.DataFrame(rows)

    if show and len(rows):
        n = len(rows)
        cols = min(4, n)
        r = math.ceil(n / cols)
        fig, axes = plt.subplots(r, cols, figsize=(3.5 * cols, 3.8 * r))
        axes = np.array(axes).reshape(-1) if n > 1 else [axes]
        for ax in axes:
            ax.axis("off")
        for i, row in enumerate(rows):
            ax = axes[i]
            if "error" in row:
                ax.set_title(row["error"][:40], fontsize=8, color="red")
                continue
            try:
                im = Image.open(row["path"]).convert("RGB")
                ax.imshow(im)
            except Exception:
                pass
            ax.set_title(
                f"{row['category']} | {row['freshness_stage']}\n"
                f"RSL ≈ {row['remaining_shelf_life_days']:.0f}d | conf={row['confidence']:.2f}",
                fontsize=9,
            )
            ax.axis("off")
        plt.suptitle("Shelf-life predictions", y=1.02)
        plt.tight_layout()
        fig.savefig(WORKING_DIR / "inference_batch_preview.png", bbox_inches="tight")
        plt.show()
    return res


# Demo on random test images
demo_paths = test_meta.sample(min(8, len(test_meta)), random_state=SEED)["filepath"].tolist()
demo_results = predict_images(demo_paths, show=True)
display(demo_results[[
    c for c in [
        "path", "category", "freshness_stage", "remaining_shelf_life_days",
        "confidence", "interpretation",
    ] if c in demo_results.columns
]])
demo_results.to_csv(WORKING_DIR / "demo_inference_results.csv", index=False)
'''
)

# =============================================================================
# CELL 18 — Upload inference (Kaggle)
# =============================================================================
md(
    r"""
## 18. Interactive Inference (Upload Images)

- **Kaggle**: use the file upload widget below, or copy images into `/kaggle/working/uploads`
- **Local**: set `UPLOAD_DIR` to a folder of images
"""
)

code(
    r'''
UPLOAD_DIR = WORKING_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Try IPython file upload widget (works in many notebook UIs)
try:
    from IPython.display import display as ipy_display
    import ipywidgets as widgets

    uploader = widgets.FileUpload(accept="image/*", multiple=True, description="Upload images")
    run_btn = widgets.Button(description="Run prediction", button_style="success")
    out_box = widgets.Output()

    def _on_run(_):
        out_box.clear_output()
        with out_box:
            if not uploader.value:
                print("No files uploaded. Place images in", UPLOAD_DIR)
                paths = sorted(
                    [str(p) for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
                )
            else:
                paths = []
                # ipywidgets 7 vs 8 compatibility
                files = uploader.value
                if isinstance(files, dict):
                    items = files.values()
                else:
                    items = files
                for f in items:
                    name = f["metadata"]["name"] if "metadata" in f else f.get("name", "upload.jpg")
                    content = f["content"] if "content" in f else f.get("content", b"")
                    dest = UPLOAD_DIR / name
                    with open(dest, "wb") as fh:
                        fh.write(content)
                    paths.append(str(dest))
            if not paths:
                print("No images found.")
                return
            res = predict_images(paths, show=True)
            display(res)
            res.to_csv(WORKING_DIR / "upload_predictions.csv", index=False)
            print("Saved →", WORKING_DIR / "upload_predictions.csv")

    run_btn.on_click(_on_run)
    ipy_display(widgets.VBox([uploader, run_btn, out_box]))
    print("Upload widget ready. Or copy images into:", UPLOAD_DIR)
except Exception as e:
    print("Widget unavailable:", e)
    print(f"Drop images into {UPLOAD_DIR} and run the next cell.")
'''
)

code(
    r'''
# Fallback: predict everything currently in UPLOAD_DIR
upload_paths = sorted([str(p) for p in UPLOAD_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
if upload_paths:
    print(f"Found {len(upload_paths)} upload(s).")
    upload_res = predict_images(upload_paths, show=True)
    display(upload_res)
    upload_res.to_csv(WORKING_DIR / "upload_predictions.csv", index=False)
else:
    print(f"No uploads yet. Add images to: {UPLOAD_DIR}")
'''
)

# =============================================================================
# CELL 19 — Save all artifacts
# =============================================================================
md(
    r"""
## 19. Save Artifacts to `/kaggle/working`

| Artifact | Description |
|----------|-------------|
| `model_final_*.keras` | Best / final trained model |
| `label_maps.json` | Category, freshness, RSL maps |
| `metrics.json` | Accuracy, P/R/F1, MAE/RMSE |
| `training_history.json` | Merged train/val curves |
| `*_predictions.csv` | Val / test / upload predictions |
| `confusion_matrix_*.png` | Confusion matrices |
| `training_curves.png` | Loss / accuracy plots |
| `gradcam_samples.png` | Explainability |
| `dataset_summary.csv` | Class counts per split |
| `corrupted_images.txt` | Skipped files |
"""
)

code(
    r'''
# ============================================================
# Final packaging
# ============================================================

# Also export SavedModel directory for serving
export_dir = WORKING_DIR / f"savedmodel_{BACKBONE}_{TASK_MODE}"
try:
    model.export(str(export_dir))  # TF 2.13+
    print("Exported SavedModel via model.export →", export_dir)
except Exception:
    try:
        tf.saved_model.save(model, str(export_dir))
        print("Exported SavedModel via tf.saved_model.save →", export_dir)
    except Exception as e:
        print("SavedModel export skipped:", e)

# Weights-only backup
model.save_weights(str(WORKING_DIR / f"weights_{BACKBONE}_{TASK_MODE}.weights.h5"))

# Config dump
run_config = {
    "seed": SEED,
    "backbone": BACKBONE,
    "task_mode": TASK_MODE,
    "img_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_head": EPOCHS_HEAD,
    "epochs_finetune": EPOCHS_FINETUNE,
    "lr_head": LEARNING_RATE_HEAD,
    "lr_finetune": LEARNING_RATE_FINETUNE,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT_RATE,
    "label_smoothing": LABEL_SMOOTHING,
    "unfreeze_ratio": UNFREEZE_RATIO,
    "use_cosine_lr": USE_COSINE_LR,
    "mixed_precision": MIXED_PRECISION,
    "num_categories": NUM_CATEGORIES,
    "num_freshness": NUM_FRESHNESS,
    "num_combined": NUM_COMBINED,
    "images_dir": str(IMAGES_DIR),
    "n_train": len(train_meta),
    "n_val": len(val_meta),
    "n_test": len(test_meta),
}
with open(WORKING_DIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

# Manifest
manifest = sorted([str(p.relative_to(WORKING_DIR)) for p in WORKING_DIR.rglob("*") if p.is_file()])
with open(WORKING_DIR / "artifacts_manifest.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(manifest))

print("\n" + "=" * 60)
print("ARTIFACTS IN", WORKING_DIR)
print("=" * 60)
for m in manifest:
    size = (WORKING_DIR / m).stat().st_size
    print(f"  {m:50s}  {size/1024:10.1f} KB")
print("\n✓ Pipeline complete.")
'''
)

# =============================================================================
# CELL 20 — Quick reload utility
# =============================================================================
md(
    r"""
## 20. Reload Model for Future Sessions

```python
import json
from pathlib import Path
from tensorflow import keras

WORKING_DIR = Path("/kaggle/working")
with open(WORKING_DIR / "label_maps.json") as f:
    label_maps = json.load(f)
model = keras.models.load_model(WORKING_DIR / "best_model_efficientnetv2b0_multitask.keras")
# then call predict_image(...) after redefining helpers or re-running inference cells
```
"""
)

code(
    r'''
print("=" * 60)
print("FreshProduce Vision — training notebook finished successfully.")
print(f"Backbone : {BACKBONE}")
print(f"Task mode: {TASK_MODE}")
print(f"Outputs  : {WORKING_DIR}")
print("=" * 60)
'''
)

nb["cells"] = cells
out_path = Path(__file__).resolve().parents[1] / "FreshProduce_ShelfLife_TransferLearning.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path} with {len(cells)} cells")
