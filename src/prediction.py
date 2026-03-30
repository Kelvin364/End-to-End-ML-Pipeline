"""
src/prediction.py
Inference logic with Keras quantization_config patch.
"""

import os
import logging
import numpy as np

from src.preprocessing import prepare_for_inference, CLASSES

logger = logging.getLogger(__name__)

MODEL_KERAS_PATH = os.path.join("models", "blood_cell_best.keras")
MODEL_PKL_PATH   = os.path.join("models", "blood_cell_best.pkl")

_model_cache   = None
_wrapper_cache = None


def _load_keras_model():
    """Load model with patch for quantization_config deserialization error."""
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        from tensorflow import keras

        if not os.path.exists(MODEL_KERAS_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_KERAS_PATH}'. "
                "Place blood_cell_best.keras in models/ directory."
            )

        logger.info(f"Loading Keras model from {MODEL_KERAS_PATH}")

        # ── Patch Dense to accept quantization_config ──────────────────────
        # The model was saved with Keras 3.x which added quantization_config
        # to Dense layer config. Older Keras versions do not recognise it.
        # This patch makes Dense silently ignore the unknown argument.
        original_dense_init = keras.layers.Dense.__init__

        def patched_dense_init(self, *args, quantization_config=None, **kwargs):
            original_dense_init(self, *args, **kwargs)

        keras.layers.Dense.__init__ = patched_dense_init

        try:
            _model_cache = tf.keras.models.load_model(
                MODEL_KERAS_PATH,
                compile=False
            )
            # Recompile with basic settings for inference
            _model_cache.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            logger.info("Model loaded successfully with quantization patch")
        finally:
            # Restore original Dense init regardless of success or failure
            keras.layers.Dense.__init__ = original_dense_init

    return _model_cache


def _load_wrapper():
    """Load the pkl wrapper for metadata."""
    global _wrapper_cache
    if _wrapper_cache is None:
        if not os.path.exists(MODEL_PKL_PATH):
            return None
        try:
            import joblib
            _wrapper_cache = joblib.load(MODEL_PKL_PATH)
        except Exception as e:
            logger.warning(f"Could not load pkl wrapper: {e}")
            return None
    return _wrapper_cache


def reload_model():
    """Force reload model from disk after retraining."""
    global _model_cache, _wrapper_cache
    _model_cache   = None
    _wrapper_cache = None
    logger.info("Model cache cleared — will reload on next request")


def predict(image_bytes: bytes) -> dict:
    """
    Run inference on a single image.
    Returns label, confidence, and all class scores.
    """
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
    """Return model metadata for the /status endpoint."""
    model_exists = os.path.exists(MODEL_KERAS_PATH)
    pkl_exists   = os.path.exists(MODEL_PKL_PATH)

    base = {
        "model_name"  : "Custom CNN",
        "class_names" : CLASSES,
        "img_size"    : [96, 96],
        "accuracy"    : None,
        "f1"          : None,
        "model_file"  : MODEL_KERAS_PATH,
        "model_exists": model_exists,
        "pkl_exists"  : pkl_exists,
    }

    wrapper = _load_wrapper()
    if wrapper:
        try:
            base.update({
                "model_name" : getattr(wrapper, "model_name", "Custom CNN"),
                "class_names": getattr(wrapper, "class_names", CLASSES),
                "img_size"   : list(getattr(wrapper, "img_size", [96, 96])),
                "accuracy"   : wrapper.metrics.get("Accuracy"),
                "f1"         : wrapper.metrics.get("F1"),
            })
        except Exception as e:
            logger.warning(f"Could not read wrapper metadata: {e}")

    return base
