"""Export Keras multitask model to ONNX for Streamlit Cloud (no TensorFlow runtime)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "streamlit_model_bundle"
KERAS = BUNDLE / "best_model_efficientnetv2b0_multitask.keras"
ONNX = BUNDLE / "model_multitask.onnx"
META = BUNDLE / "onnx_meta.json"


def main() -> None:
    print("Loading", KERAS)
    model = tf.keras.models.load_model(str(KERAS), compile=False)
    model.summary()
    print("inputs:", model.inputs)
    print("outputs:", model.outputs)
    print("output_names:", getattr(model, "output_names", None))

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    # Prefer named outputs if multi-output
    output_names = list(getattr(model, "output_names", None) or [])
    if not output_names and isinstance(model.output, dict):
        output_names = list(model.output.keys())

    print("Converting to ONNX…")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(ONNX),
    )
    print("Wrote", ONNX, "size_mb=", round(ONNX.stat().st_size / 1e6, 2))

    # Probe output names from ONNX
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    inps = [(i.name, i.shape) for i in sess.get_inputs()]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    print("ORT inputs:", inps)
    print("ORT outputs:", outs)

    # Sanity check vs Keras
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(1, 224, 224, 3)).astype(np.float32)
    k_pred = model(x, training=False)
    if isinstance(k_pred, dict):
        k_map = {k: np.asarray(v) for k, v in k_pred.items()}
    elif isinstance(k_pred, (list, tuple)):
        names = output_names or [f"out_{i}" for i in range(len(k_pred))]
        k_map = {names[i]: np.asarray(k_pred[i]) for i in range(len(k_pred))}
    else:
        k_map = {"output": np.asarray(k_pred)}

    feed = {sess.get_inputs()[0].name: x}
    o_pred = sess.run(None, feed)
    o_map = {sess.get_outputs()[i].name: o_pred[i] for i in range(len(o_pred))}
    print("Keras keys:", list(k_map.keys()))
    print("ONNX keys:", list(o_map.keys()))
    for kn, kv in k_map.items():
        # find best matching onnx output by shape
        best = None
        best_diff = 1e9
        for on, ov in o_map.items():
            if ov.shape == kv.shape:
                d = float(np.max(np.abs(ov - kv)))
                if d < best_diff:
                    best_diff = d
                    best = on
        print(f"  {kn} ~ {best}: max_abs_diff={best_diff:.6g}")

    meta = {
        "onnx_file": ONNX.name,
        "input_name": sess.get_inputs()[0].name,
        "input_shape": list(sess.get_inputs()[0].shape),
        "output_names": [o.name for o in sess.get_outputs()],
        "keras_output_names": output_names,
        "preprocess": "efficientnet_v2_tf",  # scale to [-1, 1]
        "img_size": [224, 224],
    }
    # Map onnx outputs to semantic heads if possible
    semantic = {}
    for kn in k_map:
        best = None
        best_diff = 1e9
        for on, ov in o_map.items():
            if ov.shape == k_map[kn].shape:
                d = float(np.max(np.abs(ov - k_map[kn])))
                if d < best_diff:
                    best_diff = d
                    best = on
        if best:
            semantic[kn] = best
    meta["output_map"] = semantic  # semantic_name -> onnx_output_name
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", META)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
