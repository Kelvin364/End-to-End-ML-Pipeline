"""
src/prediction.py
Inference logic. No quantization patch needed when
tensorflow==2.16.1 and keras==3.3.3 match the Colab save version.
"""

import os
import logging
import numpy as np
from src.preprocessing import prepare_for_inference, CLASSES

logger           = logging.getLogger(__name__)
MODEL_KERAS_PATH = os.path.join("models", "blood_cell_best.keras")
MODEL_PKL_PATH   = os.path.join("models", "blood_cell_best.pkl")

_model_cache = None


def _load_keras_model():
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_KERAS_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_KERAS_PATH}'."
            )
        logger.info(f"Loading model from {MODEL_KERAS_PATH}")
        _model_cache = tf.keras.models.load_model(
            MODEL_KERAS_PATH, compile=False
        )
        _model_cache.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model loaded successfully")
    return _model_cache


def reload_model():
    global _model_cache
    _model_cache = None
    logger.info("Model cache cleared")


def predict(image_bytes: bytes) -> dict:
    model = _load_keras_model()
    batch = prepare_for_inference(image_bytes)
    probs = model.predict(batch, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return {
        "label"      : CLASSES[idx],
        "confidence" : round(float(probs[idx]) * 100, 2),
        "all_scores" : {
            cls: round(float(probs[i]) * 100, 2)
            for i, cls in enumerate(CLASSES)
        }
    }


def get_model_metadata() -> dict:
    model_exists = os.path.exists(MODEL_KERAS_PATH)
    pkl_exists   = os.path.exists(MODEL_PKL_PATH)
    base = {
        "model_name"  : "Custom CNN",
        "class_names" : CLASSES,
        "img_size"    : [96, 96],
        "accuracy"    : 99.84,
        "f1"          : 99.84,
        "model_file"  : MODEL_KERAS_PATH,
        "model_exists": model_exists,
        "pkl_exists"  : pkl_exists,
    }
    if pkl_exists:
        try:
            import joblib
            wrapper = joblib.load(MODEL_PKL_PATH)
            base.update({
                "model_name": getattr(wrapper, "model_name", "Custom CNN"),
                "accuracy"  : wrapper.metrics.get("Accuracy", 99.84),
                "f1"        : wrapper.metrics.get("F1", 99.84),
            })
        except Exception:
            pass
    return base
