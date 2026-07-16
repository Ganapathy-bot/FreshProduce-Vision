"""
Retrain FreshProduce multitask model with the REAL images/ tree and export
a Streamlit-ready bundle (label_maps.json + SavedModel).

Fixes the broken export that used /kaggle path → classes "input"/"datasets"
and 1-logit heads (always 100% confidence).

Usage (from repo root or anywhere):
  python scripts/retrain_streamlit_bundle.py

CPU-friendly defaults: stratified cap per produce×stage, short schedule.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", REPO / "images")).resolve()
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO / "streamlit_model_bundle")).resolve()
WORK = Path(os.environ.get("WORK_DIR", REPO / "working_retrain")).resolve()
WORK.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
BACKBONE = os.environ.get("BACKBONE", "efficientnetv2b0")
# Cap samples per category×freshness for CPU retrain (set 0 = use all)
MAX_PER_COMBINED = int(os.environ.get("MAX_PER_COMBINED", "120"))
EPOCHS_HEAD = int(os.environ.get("EPOCHS_HEAD", "3"))
EPOCHS_FINETUNE = int(os.environ.get("EPOCHS_FINETUNE", "2"))
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
DROPOUT = 0.35
LABEL_SMOOTHING = 0.05
UNFREEZE_RATIO = 0.3
SHUFFLE_BUFFER = 1024
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PRODUCE_GROUP_NAMES = {"fruit", "fruits", "vegetable", "vegetables", "produce"}

FRESHNESS_ORDER = [
    "Fresh",
    "Semi_Fresh",
    "Early_Ripening",
    "Mid_Ripening",
    "Fully_Ripe",
    "Spoiled",
]
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
}


def canon_freshness(name: str) -> str:
    low = str(name).strip().replace("-", "_").replace(" ", "_").lower()
    return CANONICAL_FRESHNESS.get(low, str(name).strip().replace("-", "_").replace(" ", "_"))


def rsl_from_freshness(freshness: str) -> float:
    key = str(freshness)
    if key in DEFAULT_RSL_MAP:
        return float(DEFAULT_RSL_MAP[key])
    for k, v in DEFAULT_RSL_MAP.items():
        if k.lower() == key.lower():
            return float(v)
    return float("nan")


def iter_produce_dirs(images_dir: Path) -> List[Tuple[str, str, Path]]:
    out = []
    for child in sorted(images_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name.lower() in PRODUCE_GROUP_NAMES:
            for produce in sorted(child.iterdir()):
                if produce.is_dir() and not produce.name.startswith("."):
                    out.append((child.name, produce.name, produce))
        else:
            out.append(("Unknown", child.name, child))
    return out


def load_from_images_folder(images_dir: Path) -> pd.DataFrame:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"IMAGES_DIR not found: {images_dir}")
    rows = []
    produce_list = iter_produce_dirs(images_dir)
    if not produce_list:
        raise RuntimeError(f"No produce folders under {images_dir}")
    print(f"Scanning {len(produce_list)} produce folders under {images_dir} …")
    for group, category, produce_path in produce_list:
        stage_dirs = [
            d for d in sorted(produce_path.iterdir()) if d.is_dir() and not d.name.startswith(".")
        ]
        if not stage_dirs:
            continue
        for stage_dir in stage_dirs:
            freshness = canon_freshness(stage_dir.name)
            for img in stage_dir.rglob("*"):
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                    continue
                rows.append(
                    {
                        "filepath": str(img.resolve()),
                        "group": group,
                        "category": str(category),
                        "freshness": str(freshness),
                        "combined": f"{category}_{freshness}",
                        "remaining_shelf_life": rsl_from_freshness(freshness),
                    }
                )
    if not rows:
        raise RuntimeError(f"No images found under {images_dir}")
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        train_val, test = train_test_split(
            df, test_size=TEST_RATIO, random_state=SEED, stratify=df["combined"]
        )
    except ValueError:
        train_val, test = train_test_split(df, test_size=TEST_RATIO, random_state=SEED)
    val_frac = VAL_RATIO / max(TRAIN_RATIO + VAL_RATIO, 1e-8)
    try:
        train, val = train_test_split(
            train_val, test_size=val_frac, random_state=SEED, stratify=train_val["combined"]
        )
    except ValueError:
        train, val = train_test_split(train_val, test_size=val_frac, random_state=SEED)
    train, val, test = train.copy(), val.copy(), test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)


def cap_per_combined(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if max_n <= 0:
        return df
    parts = []
    for _, g in df.groupby("combined", sort=False):
        if len(g) > max_n:
            parts.append(g.sample(n=max_n, random_state=SEED))
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main():
    t0 = time.time()
    print("=" * 60)
    print("FreshProduce retrain → Streamlit bundle")
    print("=" * 60)
    print(f"IMAGES_DIR = {IMAGES_DIR}")
    print(f"OUT_DIR    = {OUT_DIR}")
    print(f"WORK       = {WORK}")
    print(f"BACKBONE   = {BACKBONE}")
    print(f"MAX_PER_COMBINED = {MAX_PER_COMBINED}")
    print(f"EPOCHS     = head {EPOCHS_HEAD} + finetune {EPOCHS_FINETUNE}")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV3Large

    tf.keras.utils.set_random_seed(SEED)
    # Mixed precision often slower/unstable on CPU
    try:
        keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass

    # ---- data ----
    meta = load_from_images_folder(IMAGES_DIR)
    print(f"Raw images: {len(meta):,}")
    meta = meta[meta["freshness"].astype(str).str.len() > 0].reset_index(drop=True)
    meta = meta[meta["freshness"] != "Unknown"].reset_index(drop=True)

    before = len(meta)
    meta = cap_per_combined(meta, MAX_PER_COMBINED)
    print(f"After cap: {len(meta):,} (from {before:,})")

    meta = assign_splits(meta)
    train_meta = meta[meta["split"] == "train"].copy()
    val_meta = meta[meta["split"] == "val"].copy()
    test_meta = meta[meta["split"] == "test"].copy()

    CATEGORY_CLASSES = sorted(train_meta["category"].unique().tolist())
    FRESHNESS_CLASSES = sorted(
        train_meta["freshness"].unique().tolist(),
        key=lambda x: (FRESHNESS_ORDER.index(x) if x in FRESHNESS_ORDER else 999, x),
    )
    COMBINED_CLASSES = sorted(train_meta["combined"].unique().tolist())
    cat2id = {c: i for i, c in enumerate(CATEGORY_CLASSES)}
    fresh2id = {c: i for i, c in enumerate(FRESHNESS_CLASSES)}
    NUM_CATEGORIES = len(CATEGORY_CLASSES)
    NUM_FRESHNESS = len(FRESHNESS_CLASSES)

    if NUM_CATEGORIES < 2 or NUM_FRESHNESS < 2:
        raise RuntimeError(
            f"Too few classes: categories={NUM_CATEGORIES} freshness={NUM_FRESHNESS}. "
            "Check IMAGES_DIR points at images/Fruit|Vegetable/..."
        )
    if set(CATEGORY_CLASSES) <= {"input", "datasets"} or set(FRESHNESS_CLASSES) <= {
        "input",
        "datasets",
    }:
        raise RuntimeError("Labels still look like wrong Kaggle path. Fix IMAGES_DIR.")

    print(f"Categories ({NUM_CATEGORIES}): {CATEGORY_CLASSES}")
    print(f"Freshness  ({NUM_FRESHNESS}): {FRESHNESS_CLASSES}")

    for df_ in (train_meta, val_meta, test_meta):
        df_["category_id"] = df_["category"].map(cat2id).astype(np.int32)
        df_["freshness_id"] = df_["freshness"].map(fresh2id).astype(np.int32)

    # drop rows with NaN ids (unseen in train)
    val_meta = val_meta.dropna(subset=["category_id", "freshness_id"]).reset_index(drop=True)
    test_meta = test_meta.dropna(subset=["category_id", "freshness_id"]).reset_index(drop=True)
    print(f"Train={len(train_meta):,} Val={len(val_meta):,} Test={len(test_meta):,}")

    label_maps = {
        "category_classes": CATEGORY_CLASSES,
        "freshness_classes": FRESHNESS_CLASSES,
        "combined_classes": COMBINED_CLASSES,
        "rsl_map": DEFAULT_RSL_MAP,
        "task_mode": "multitask",
        "backbone": BACKBONE,
        "img_size": list(IMG_SIZE),
    }
    (WORK / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    # class weights
    def make_cw(y, n):
        cw = compute_class_weight("balanced", classes=np.arange(n), y=y)
        return {int(i): float(w) for i, w in enumerate(cw)}

    cw_cat = make_cw(train_meta["category_id"].values, NUM_CATEGORIES)
    cw_fr = make_cw(train_meta["freshness_id"].values, NUM_FRESHNESS)

    # ---- preprocess ----
    if BACKBONE.lower() == "mobilenetv3":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    AUTOTUNE = tf.data.AUTOTUNE

    def decode_and_resize(path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        return img * 255.0

    def augment_image(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1 * 255.0)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img

    def build_ds(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
        paths = df["filepath"].astype(str).values
        cat = df["category_id"].astype(np.int32).values
        fr = df["freshness_id"].astype(np.int32).values
        w_cat = df["category_id"].map(cw_cat).astype(np.float32).values
        w_fr = df["freshness_id"].map(cw_fr).astype(np.float32).values
        ds = tf.data.Dataset.from_tensor_slices((paths, cat, fr, w_cat, w_fr))
        if training:
            ds = ds.shuffle(min(SHUFFLE_BUFFER, len(df)), seed=SEED, reshuffle_each_iteration=True)

        def _map(path, y_cat, y_fr, wc, wf):
            img = decode_and_resize(path)
            if training:
                img = augment_image(img)
            img = preprocess_input(img)
            y = {
                "category": tf.one_hot(y_cat, NUM_CATEGORIES),
                "freshness": tf.one_hot(y_fr, NUM_FRESHNESS),
            }
            w = {"category": wc, "freshness": wf}
            return img, y, w

        ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=not training)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_ds = build_ds(train_meta, True)
    val_ds = build_ds(val_meta, False)

    # ---- model ----
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    if BACKBONE.lower() == "mobilenetv3":
        base = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=inputs)
    else:
        base = EfficientNetV2B0(include_top=False, weights="imagenet", input_tensor=inputs)
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(DROPOUT, name="dropout_1")(x)
    shared = layers.Dense(512, activation="relu", name="shared_dense")(x)
    shared = layers.BatchNormalization(name="bn_shared")(shared)
    shared = layers.Dropout(DROPOUT * 0.75, name="dropout_2")(shared)

    cat_h = layers.Dense(256, activation="relu", name="cat_dense")(shared)
    cat_h = layers.Dropout(DROPOUT * 0.5, name="cat_dropout")(cat_h)
    category = layers.Dense(NUM_CATEGORIES, activation="softmax", name="category")(cat_h)

    fr_h = layers.Dense(256, activation="relu", name="fresh_dense")(shared)
    fr_h = layers.Dropout(DROPOUT * 0.5, name="fresh_dropout")(fr_h)
    freshness = layers.Dense(NUM_FRESHNESS, activation="softmax", name="freshness")(fr_h)

    model = keras.Model(
        inputs, {"category": category, "freshness": freshness}, name=f"{BACKBONE}_multitask"
    )
    model._backbone = base

    def compile_model(m, lr):
        try:
            opt = optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
        except Exception:
            opt = optimizers.Adam(learning_rate=lr)
        m.compile(
            optimizer=opt,
            loss={
                "category": keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                "freshness": keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            },
            loss_weights={"category": 1.0, "freshness": 1.25},
            metrics={
                "category": [keras.metrics.CategoricalAccuracy(name="accuracy")],
                "freshness": [keras.metrics.CategoricalAccuracy(name="accuracy")],
            },
        )
        return m

    ckpt = WORK / f"best_model_{BACKBONE}_multitask.keras"
    cbs = [
        callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True, mode="min", verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
    ]

    print("\n--- Stage 1: train head (frozen backbone) ---")
    model = compile_model(model, LR_HEAD)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cbs, verbose=1)

    print("\n--- Stage 2: fine-tune top backbone ---")
    # unfreeze top fraction
    layers_list = list(base.layers)
    n = len(layers_list)
    freeze_until = int(n * (1.0 - UNFREEZE_RATIO))
    base.trainable = True
    for i, layer in enumerate(layers_list):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    model = compile_model(model, LR_FINETUNE)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=cbs, verbose=1)

    if ckpt.is_file():
        model = keras.models.load_model(str(ckpt), compile=False)
        print(f"Loaded best checkpoint: {ckpt}")

    # quick val metrics
    print("\n--- Quick validation eval ---")
    model = compile_model(model, LR_FINETUNE)
    hist = model.evaluate(val_ds, verbose=1, return_dict=True)
    print(hist)

    # ---- export bundle ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # remove old broken savedmodel
    for old in OUT_DIR.glob("savedmodel_*"):
        if old.is_dir():
            print(f"Removing old {old}")
            shutil.rmtree(old, ignore_errors=True)

    export_dir = OUT_DIR / f"savedmodel_{BACKBONE}_multitask"
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    try:
        model.export(str(export_dir))
        print("Exported via model.export →", export_dir)
    except Exception as e1:
        print("model.export failed:", e1)
        tf.saved_model.save(model, str(export_dir))
        print("Exported via tf.saved_model.save →", export_dir)

    # also save keras
    keras_path = OUT_DIR / f"best_model_{BACKBONE}_multitask.keras"
    try:
        model.save(str(keras_path))
        print("Saved Keras model →", keras_path)
    except Exception as e:
        print("Keras save skipped:", e)

    (OUT_DIR / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    run_config = {
        "seed": SEED,
        "backbone": BACKBONE,
        "task_mode": "multitask",
        "img_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs_head": EPOCHS_HEAD,
        "epochs_finetune": EPOCHS_FINETUNE,
        "max_per_combined": MAX_PER_COMBINED,
        "num_categories": NUM_CATEGORIES,
        "num_freshness": NUM_FRESHNESS,
        "num_combined": len(COMBINED_CLASSES),
        "images_dir": str(IMAGES_DIR),
        "n_train": len(train_meta),
        "n_val": len(val_meta),
        "n_test": len(test_meta),
        "val_metrics": {k: float(v) for k, v in hist.items()} if isinstance(hist, dict) else {},
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # sanity: inference shape
    print("\n--- Sanity check SavedModel ---")
    loaded = tf.saved_model.load(str(export_dir))
    sig = (
        loaded.signatures["serving_default"]
        if "serving_default" in loaded.signatures
        else list(loaded.signatures.values())[0]
    )
    ink = list(sig.structured_input_signature[1].keys())[0]
    x = tf.zeros((1, *IMG_SIZE, 3), dtype=tf.float32)
    out = sig(**{ink: x})
    for k, v in out.items():
        arr = np.asarray(v[0]).reshape(-1)
        print(f"  {k}: shape={arr.shape} sum={arr.sum():.4f} argmax={arr.argmax()}")
        assert arr.shape[0] > 1, f"{k} still single-class!"

    print(f"\n✓ Bundle ready at {OUT_DIR}")
    print(f"  categories={NUM_CATEGORIES} freshness={NUM_FRESHNESS}")
    print(f"  elapsed {time.time()-t0:.1f}s")
    print("Restart Streamlit (or clear cache) and reload http://localhost:8501/")


if __name__ == "__main__":
    main()
