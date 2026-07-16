"""
FreshProduce Vision — Streamlit inference app

Features:
  - Choose / upload one or more fruit & vegetable images
  - Predict category, freshness stage, remaining shelf life, confidence
  - Top-k class probabilities for both heads (multitask model)

Run:
  streamlit run app.py

Place model artifacts next to this file (or set MODEL_DIR):
  streamlit_model_bundle/
    label_maps.json
    run_config.json
    model_multitask.onnx          # preferred (Streamlit Cloud — no TensorFlow)
    onnx_meta.json
    # optional local: best_model_*.keras / SavedModel (needs TensorFlow)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths — edit MODEL_DIR if your bundle lives elsewhere
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
# Default: ../streamlit_model_bundle relative to this app
DEFAULT_MODEL_DIR = APP_DIR.parent / "streamlit_model_bundle"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()

st.set_page_config(
    page_title="FreshProduce Shelf-Life",
    page_icon="🍎",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_file(name: str, root: Path) -> Optional[Path]:
    direct = root / name
    if direct.is_file():
        return direct
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def find_savedmodel_dir(root: Path) -> Optional[Path]:
    """Directory that contains saved_model.pb (+ variables/)."""
    for pb in root.rglob("saved_model.pb"):
        d = pb.parent
        if (d / "variables").is_dir() or any(d.glob("variables*")):
            return d
    for pb in root.rglob("saved_model.pb"):
        return pb.parent
    return None


def find_keras_model(root: Path) -> Optional[Path]:
    for pat in ("*.keras", "*.h5"):
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def find_onnx_model(root: Path) -> Optional[Path]:
    direct = root / "model_multitask.onnx"
    if direct.is_file():
        return direct
    hits = sorted(root.rglob("*.onnx"))
    return hits[0] if hits else None


def preprocess_numpy_rgb255(arr: np.ndarray) -> np.ndarray:
    """Keep RGB floats in [0, 255].

    This project's EfficientNetV2 multitask model already includes Rescaling +
    Normalization inside the graph (include_preprocessing). Feeding mode='tf'
    [-1,1] double-preprocesses and hurts accuracy.
    """
    return arr.astype(np.float32, copy=False)


def get_preprocess_fn(backbone: str):
    """Pure NumPy preprocess — no TensorFlow required on Cloud."""
    _ = backbone  # reserved if future backbones need different scaling
    return preprocess_numpy_rgb255


def preprocess_pil(img: Image.Image, img_size: Tuple[int, int], preprocess_fn) -> np.ndarray:
    """RGB PIL → batch (1,H,W,3) float32 ready for model."""
    img = img.convert("RGB").resize((img_size[1], img_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)  # [0, 255]
    arr = preprocess_fn(arr)
    return np.expand_dims(arr, axis=0)


def topk(probs: np.ndarray, class_names: List[str], k: int = 5) -> List[Tuple[str, float]]:
    k = min(k, len(probs))
    idx = np.argsort(probs)[::-1][:k]
    names = class_names if len(class_names) == len(probs) else [f"class_{i}" for i in range(len(probs))]
    return [(names[i] if i < len(names) else f"class_{i}", float(probs[i])) for i in idx]


# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(model_dir: str):
    root = Path(model_dir)
    if not root.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {root}")

    label_path = find_file("label_maps.json", root)
    config_path = find_file("run_config.json", root)
    onnx_meta_path = find_file("onnx_meta.json", root)
    label_maps = json.loads(label_path.read_text(encoding="utf-8")) if label_path else {}
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    onnx_meta = json.loads(onnx_meta_path.read_text(encoding="utf-8")) if onnx_meta_path else {}

    img_size = tuple(label_maps.get("img_size") or run_config.get("img_size") or [224, 224])
    backbone = label_maps.get("backbone") or run_config.get("backbone") or "efficientnetv2b0"
    task_mode = label_maps.get("task_mode") or run_config.get("task_mode") or "multitask"
    rsl_map = label_maps.get("rsl_map") or {
        "Fresh": 7.0,
        "Semi_Fresh": 5.0,
        "Early_Ripening": 4.0,
        "Mid_Ripening": 3.0,
        "Fully_Ripe": 2.0,
        "Spoiled": 0.0,
    }

    category_classes = list(label_maps.get("category_classes") or [])
    freshness_classes = list(label_maps.get("freshness_classes") or [])
    combined_classes = list(label_maps.get("combined_classes") or [])

    preprocess_fn = get_preprocess_fn(backbone)

    backend = None
    model = None
    serve_fn = None
    output_keys: List[str] = []
    input_key: Optional[str] = None
    onnx_session = None
    onnx_input_name: Optional[str] = None
    onnx_output_map: Dict[str, str] = {}

    onnx_path = find_onnx_model(root)
    keras_path = find_keras_model(root)
    saved_dir = find_savedmodel_dir(root)

    # ---- Preferred: ONNX Runtime (Streamlit Cloud friendly) ----
    if onnx_path is not None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        backend = "onnx"
        onnx_input_name = (
            onnx_meta.get("input_name")
            or onnx_session.get_inputs()[0].name
        )
        output_keys = list(
            onnx_meta.get("output_names")
            or [o.name for o in onnx_session.get_outputs()]
        )
        onnx_output_map = dict(onnx_meta.get("output_map") or {})
        if not onnx_output_map:
            # default semantic names = onnx names
            onnx_output_map = {n: n for n in output_keys}
        model = onnx_session

    # ---- Optional: TensorFlow backends (local only) ----
    elif keras_path is not None or saved_dir is not None:
        try:
            import tensorflow as tf
        except ImportError as e:
            raise FileNotFoundError(
                "No ONNX model found and TensorFlow is not installed. "
                "Add model_multitask.onnx to streamlit_model_bundle/ "
                "or install tensorflow for local .keras/SavedModel inference."
            ) from e

        if keras_path is not None:
            model = tf.keras.models.load_model(str(keras_path), compile=False)
            backend = "keras"
            if isinstance(model.output, dict):
                output_keys = list(model.output.keys())
            elif getattr(model, "output_names", None):
                output_keys = list(model.output_names)
            else:
                output_keys = ["output"]
        else:
            if not (saved_dir / "variables").exists():
                raise FileNotFoundError(
                    f"Incomplete SavedModel at {saved_dir}: missing variables/ folder."
                )
            loaded = tf.saved_model.load(str(saved_dir))
            backend = "saved_model"
            if not loaded.signatures:
                model = loaded
                serve_fn = None
            else:
                sig_name = (
                    "serving_default"
                    if "serving_default" in loaded.signatures
                    else list(loaded.signatures.keys())[0]
                )
                serve_fn = loaded.signatures[sig_name]
                spec = serve_fn.structured_input_signature[1]
                input_key = list(spec.keys())[0]
                output_keys = list(serve_fn.structured_outputs.keys())
                model = loaded
    else:
        raise FileNotFoundError(
            f"No model found under {root}. Expected model_multitask.onnx "
            f"(preferred), a .keras file, or a SavedModel folder."
        )

    meta = {
        "model_dir": str(root),
        "backend": backend,
        "keras_path": str(keras_path) if keras_path else None,
        "saved_dir": str(saved_dir) if saved_dir else None,
        "onnx_path": str(onnx_path) if onnx_path else None,
        "img_size": img_size,
        "backbone": backbone,
        "task_mode": task_mode,
        "category_classes": category_classes,
        "freshness_classes": freshness_classes,
        "combined_classes": combined_classes,
        "rsl_map": rsl_map,
        "output_keys": output_keys,
        "input_key": input_key,
        "onnx_input_name": onnx_input_name,
        "onnx_output_map": onnx_output_map,
        "label_maps_path": str(label_path) if label_path else None,
    }
    return model, serve_fn, preprocess_fn, meta


def run_inference(batch: np.ndarray, model, serve_fn, meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return dict of output_name -> probability vector (1D for single image batch[0])."""
    if meta["backend"] == "onnx":
        sess = model
        in_name = meta["onnx_input_name"] or sess.get_inputs()[0].name
        outs = sess.run(None, {in_name: batch.astype(np.float32)})
        names = [o.name for o in sess.get_outputs()]
        raw = {names[i]: np.asarray(outs[i][0]) for i in range(len(outs))}
        # Normalize to semantic keys when possible
        out_map = meta.get("onnx_output_map") or {}
        if out_map:
            # out_map is semantic -> onnx_name; invert for convenience
            inv = {v: k for k, v in out_map.items()}
            mapped = {}
            for oname, vec in raw.items():
                mapped[inv.get(oname, oname)] = vec
            return mapped
        return raw

    import tensorflow as tf

    x = tf.convert_to_tensor(batch)

    if meta["backend"] == "keras":
        preds = model(x, training=False)
        if isinstance(preds, dict):
            return {k: np.asarray(v[0]) for k, v in preds.items()}
        if isinstance(preds, (list, tuple)):
            keys = meta["output_keys"] or [f"out_{i}" for i in range(len(preds))]
            return {keys[i]: np.asarray(preds[i][0]) for i in range(len(preds))}
        return {"output": np.asarray(preds[0])}

    if serve_fn is not None:
        key = meta["input_key"]
        out = serve_fn(**{key: x})
        return {k: np.asarray(v[0]) for k, v in out.items()}

    preds = model(x)
    if isinstance(preds, dict):
        return {k: np.asarray(v[0]) for k, v in preds.items()}
    return {"output": np.asarray(preds[0])}


def labels_look_broken(meta: Dict[str, Any]) -> bool:
    """True when export used wrong IMAGES_DIR (e.g. Kaggle path → input/datasets)."""
    cats = meta.get("category_classes") or []
    frs = meta.get("freshness_classes") or []
    bad_names = {"input", "datasets", "kaggle", "working"}
    if len(cats) <= 1 or len(frs) <= 1:
        return True
    if set(c.lower() for c in cats) <= bad_names:
        return True
    if set(f.lower() for f in frs) <= bad_names:
        return True
    return False


def resolve_rsl(freshness: str, rsl_map: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """Return (days, known). Unknown stage names must NOT default to spoiled (0)."""
    if freshness in rsl_map:
        return float(rsl_map[freshness]), True
    lower = {str(k).lower(): float(v) for k, v in rsl_map.items()}
    key = str(freshness).lower()
    if key in lower:
        return lower[key], True
    return None, False


def decode_prediction(raw: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Map model outputs → human-readable shelf-life report."""
    keys_l = {k.lower(): k for k in raw.keys()}
    cat_key = keys_l.get("category") or keys_l.get("category_output")
    fr_key = keys_l.get("freshness") or keys_l.get("freshness_output")
    comb_key = keys_l.get("combined") or keys_l.get("output")

    rsl_map = meta["rsl_map"]
    cat_names = meta["category_classes"]
    fr_names = meta["freshness_classes"]
    comb_names = meta["combined_classes"]

    result: Dict[str, Any] = {
        "raw_keys": list(raw.keys()),
        "labels_broken": labels_look_broken(meta),
        "single_class_heads": False,
        "rsl_known": True,
    }

    if cat_key and fr_key:
        p_cat = np.asarray(raw[cat_key]).reshape(-1)
        p_fr = np.asarray(raw[fr_key]).reshape(-1)
        result["single_class_heads"] = len(p_cat) <= 1 or len(p_fr) <= 1
        ic = int(np.argmax(p_cat))
        ifr = int(np.argmax(p_fr))
        category = cat_names[ic] if ic < len(cat_names) else f"class_{ic}"
        freshness = fr_names[ifr] if ifr < len(fr_names) else f"class_{ifr}"
        conf_cat = float(p_cat[ic])
        conf_fr = float(p_fr[ifr])
        conf = float(np.sqrt(max(conf_cat, 0.0) * max(conf_fr, 0.0)))
        rsl, rsl_known = resolve_rsl(freshness, rsl_map)
        result.update(
            {
                "mode": "multitask",
                "category": category,
                "freshness_stage": freshness,
                "remaining_shelf_life_days": rsl if rsl is not None else float("nan"),
                "rsl_known": rsl_known,
                "confidence": conf,
                "confidence_category": conf_cat,
                "confidence_freshness": conf_fr,
                "top_category": topk(p_cat, cat_names, k=5),
                "top_freshness": topk(p_fr, fr_names, k=5),
            }
        )
    else:
        key = comb_key or list(raw.keys())[0]
        p = np.asarray(raw[key]).reshape(-1)
        result["single_class_heads"] = len(p) <= 1
        i = int(np.argmax(p))
        names = comb_names if len(comb_names) == len(p) else fr_names if len(fr_names) == len(p) else cat_names
        label = names[i] if i < len(names) else f"class_{i}"
        conf = float(p[i])
        if "_" in label:
            category, freshness = label.split("_", 1)
        else:
            category, freshness = "Unknown", label
        rsl, rsl_known = resolve_rsl(freshness, rsl_map)
        result.update(
            {
                "mode": "single",
                "category": category,
                "freshness_stage": freshness,
                "remaining_shelf_life_days": rsl if rsl is not None else float("nan"),
                "rsl_known": rsl_known,
                "confidence": conf,
                "confidence_category": conf,
                "confidence_freshness": conf,
                "top_category": [],
                "top_freshness": topk(p, names if names else [f"class_{j}" for j in range(len(p))], k=5),
            }
        )

    if result["labels_broken"] or result["single_class_heads"]:
        result["interpretation"] = (
            "Model misconfigured — not a real freshness call. "
            "This bundle has 1 class per head and/or broken labels. "
            "Retrain with IMAGES_DIR pointing at your real images/ folder."
        )
    elif not result["rsl_known"]:
        result["interpretation"] = (
            f"Predicted stage `{result['freshness_stage']}` is not in the shelf-life map — "
            "cannot estimate remaining days. Check label_maps.json."
        )
    else:
        rsl = float(result["remaining_shelf_life_days"])
        stage = result["freshness_stage"]
        if rsl <= 0 or str(stage).lower() in {"spoiled", "rotten"}:
            result["interpretation"] = (
                f"Spoiled / reject — do not consume "
                f"(model is {result['confidence']*100:.0f}% confident this is **{stage}**)"
            )
        elif rsl <= 2:
            result["interpretation"] = (
                f"Use soon — about {rsl:.0f} day(s) left "
                f"(proxy; predicted **{stage}**, conf {result['confidence']*100:.0f}%)"
            )
        else:
            result["interpretation"] = (
                f"About {rsl:.0f} day(s) remaining shelf life "
                f"(proxy; predicted **{stage}**, conf {result['confidence']*100:.0f}%)"
            )
    return result


def conf_color(c: float) -> str:
    if c >= 0.75:
        return "green"
    if c >= 0.45:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🍎🥦 FreshProduce — Shelf-Life Predictor")
st.caption("Upload fruit/vegetable images → category, freshness stage, remaining days, confidence")

with st.sidebar:
    st.header("Settings")
    model_dir_in = st.text_input("Model directory", value=str(MODEL_DIR))
    top_k = st.slider("Top-K classes to show", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        """
**Expected bundle**
```
streamlit_model_bundle/
  label_maps.json
  run_config.json
  model_multitask.onnx
  onnx_meta.json
```
"""
    )
    st.markdown("Or set env `MODEL_DIR=/path/to/bundle`")

try:
    model, serve_fn, preprocess_fn, meta = load_artifacts(model_dir_in)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

with st.sidebar:
    st.success("Model loaded")
    st.write(f"**Backend:** `{meta['backend']}`")
    st.write(f"**Backbone:** `{meta['backbone']}`")
    st.write(f"**Task:** `{meta['task_mode']}`")
    st.write(f"**Input size:** `{meta['img_size']}`")
    st.write(f"**Outputs:** `{meta['output_keys']}`")
    n_cat = len(meta["category_classes"])
    n_fr = len(meta["freshness_classes"])
    st.write(f"**Categories:** {n_cat}  |  **Freshness:** {n_fr}")
    st.caption(f"Classes: cat={meta['category_classes'][:8]}…  fr={meta['freshness_classes'][:8]}…")
    if labels_look_broken(meta):
        st.error(
            "**Broken model export.** Labels are wrong (`input`/`datasets` or only 1 class). "
            "Retrain with `IMAGES_DIR` → your real `images/` folder, then re-export the bundle."
        )

st.subheader("1. Choose image(s)")
col_up, col_cam = st.columns(2)
with col_up:
    uploads = st.file_uploader(
        "Upload image files",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Select one or more produce photos",
    )
with col_cam:
    camera = st.camera_input("Or take a photo")

images: List[Tuple[str, Image.Image]] = []
if uploads:
    for f in uploads:
        try:
            images.append((f.name, Image.open(f).convert("RGB")))
        except Exception as ex:
            st.warning(f"Could not open {f.name}: {ex}")
if camera is not None:
    images.append(("camera.jpg", Image.open(camera).convert("RGB")))

if not images:
    st.info("👆 Upload or capture an image to run prediction.")
    st.stop()

st.subheader("2. Predictions")
run = st.button("Predict shelf life", type="primary", use_container_width=True)

if not run:
    cols = st.columns(min(4, len(images)))
    for i, (name, im) in enumerate(images[:8]):
        with cols[i % len(cols)]:
            st.image(im, caption=name, use_container_width=True)
    st.caption("Click **Predict shelf life** to run the model.")
    st.stop()

for name, im in images:
    st.markdown("---")
    left, right = st.columns([1, 1.2])
    with left:
        st.image(im, caption=name, use_container_width=True)

    try:
        batch = preprocess_pil(im, tuple(meta["img_size"]), preprocess_fn)
        raw = run_inference(batch, model, serve_fn, meta)
        pred = decode_prediction(raw, meta)
    except Exception as e:
        with right:
            st.error(f"Inference failed: {e}")
        continue

    with right:
        st.markdown(f"### Results — `{name}`")
        if pred.get("labels_broken") or pred.get("single_class_heads"):
            st.error(
                "This prediction is **invalid**: broken labels or single-class heads."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Category", pred["category"])
        m2.metric("Freshness", pred["freshness_stage"])
        rsl_val = pred["remaining_shelf_life_days"]
        m3.metric(
            "Shelf life (days)",
            "—" if (rsl_val is None or (isinstance(rsl_val, float) and np.isnan(rsl_val))) else f"{rsl_val:.0f}",
        )
        m4.metric("Confidence", f"{pred['confidence']*100:.1f}%")
        st.caption(
            "Confidence = how sure the model is about its **class pick**, "
            "not how fresh the produce is."
        )

        st.markdown(f"**Interpretation:** {pred['interpretation']}")
        st.progress(min(max(pred["confidence"], 0.0), 1.0))

        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Category confidence: **{pred['confidence_category']*100:.1f}%**")
            st.markdown(
                f"<span style='color:{conf_color(pred['confidence_category'])}'>●</span> "
                f"{'High' if pred['confidence_category']>=0.75 else 'Medium' if pred['confidence_category']>=0.45 else 'Low'}",
                unsafe_allow_html=True,
            )
        with c2:
            st.caption(f"Freshness confidence: **{pred['confidence_freshness']*100:.1f}%**")
            st.markdown(
                f"<span style='color:{conf_color(pred['confidence_freshness'])}'>●</span> "
                f"{'High' if pred['confidence_freshness']>=0.75 else 'Medium' if pred['confidence_freshness']>=0.45 else 'Low'}",
                unsafe_allow_html=True,
            )

        st.markdown("#### Probability features")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Top categories**")
            rows = pred["top_category"][:top_k]
            if rows:
                st.dataframe(
                    {"class": [r[0] for r in rows], "prob": [round(r[1], 4) for r in rows]},
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.write("—")
        with t2:
            st.markdown("**Top freshness stages**")
            rows = pred["top_freshness"][:top_k]
            if rows:
                st.dataframe(
                    {"class": [r[0] for r in rows], "prob": [round(r[1], 4) for r in rows]},
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.write("—")

        with st.expander("Raw model outputs"):
            st.json(
                {
                    "mode": pred["mode"],
                    "backend": meta["backend"],
                    "category": pred["category"],
                    "freshness_stage": pred["freshness_stage"],
                    "remaining_shelf_life_days": pred["remaining_shelf_life_days"],
                    "confidence": pred["confidence"],
                    "confidence_category": pred["confidence_category"],
                    "confidence_freshness": pred["confidence_freshness"],
                    "raw_keys": pred["raw_keys"],
                    "interpretation": pred["interpretation"],
                }
            )

st.markdown("---")
st.caption(
    "Remaining shelf life is a **proxy** mapped from the predicted freshness stage. "
    "Confidence is model certainty about the predicted class — not a freshness percentage. "
    "Not a medical/food-safety guarantee."
)
