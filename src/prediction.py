"""
src/prediction.py
-----------------
Inference logic for the blood cell classification pipeline.

Loads blood_cell_best.pkl (which bundles model weights + preprocessing config)
and exposes a single predict() function used by api.py.

The pkl wrapper handles preprocessing automatically — the caller only needs
to pass raw image bytes.
"""

import os
import logging
import joblib
import numpy as np

from src.preprocessing import prepare_for_inference, CLASSES

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PKL_PATH   = os.path.join("models", "blood_cell_best.pkl")
MODEL_KERAS_PATH = os.path.join("models", "blood_cell_best.keras")


# ── Model loader (singleton) ───────────────────────────────────────────────────
_model_cache = None
_wrapper_cache = None


def _load_wrapper():
    """Load and cache the pkl wrapper. Reloads if called again after retraining."""
    global _wrapper_cache
    if _wrapper_cache is None:
        if not os.path.exists(MODEL_PKL_PATH):
            raise FileNotFoundError(
                f"Model wrapper not found at '{MODEL_PKL_PATH}'. "
                "Place blood_cell_best.pkl in the models/ directory."
            )
        logger.info(f"Loading pkl wrapper from {MODEL_PKL_PATH}")
        _wrapper_cache = joblib.load(MODEL_PKL_PATH)
    return _wrapper_cache


def _load_keras_model():
    """
    Load and cache the Keras model directly from the .keras file.
    Used as the primary inference path — faster than reconstructing from pkl.
    """
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_KERAS_PATH):
            raise FileNotFoundError(
                f"Keras model not found at '{MODEL_KERAS_PATH}'. "
                "Place blood_cell_best.keras in the models/ directory."
            )
        logger.info(f"Loading Keras model from {MODEL_KERAS_PATH}")
        _model_cache = tf.keras.models.load_model(MODEL_KERAS_PATH)
    return _model_cache


def reload_model():
    """
    Force-clear the in-memory model cache so the next prediction call
    reloads from disk. Called by api.py after a successful retraining run.
    """
    global _model_cache, _wrapper_cache
    _model_cache   = None
    _wrapper_cache = None
    logger.info("Model cache cleared — next prediction will reload from disk")


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes) -> dict:
    """
    Run inference on a single image.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image (JPEG, PNG, etc.).

    Returns
    -------
    dict
        {
          "label"      : str,    # predicted class name
          "confidence" : float,  # confidence of top prediction (0–100)
          "all_scores" : dict    # {class_name: confidence} for all 4 classes
        }

    Raises
    ------
    FileNotFoundError
        If neither the .keras nor the .pkl model file is found.
    ValueError
        If the image cannot be decoded.
    """
    model    = _load_keras_model()
    batch    = prepare_for_inference(image_bytes)   # shape (1, 96, 96, 3)
    probs    = model.predict(batch, verbose=0)[0]   # shape (4,)
    idx      = int(np.argmax(probs))

    return {
        "label"      : CLASSES[idx],
        "confidence" : round(float(probs[idx]) * 100, 2),
        "all_scores" : {
            cls: round(float(probs[i]) * 100, 2)
            for i, cls in enumerate(CLASSES)
        }
    }


def get_model_metadata() -> dict:
    """
    Return metadata about the currently loaded model.
    Used by the /status endpoint and the UI uptime tab.

    Returns
    -------
    dict
        {
          "model_name"      : str,
          "class_names"     : list,
          "img_size"        : tuple,
          "accuracy"        : float,
          "f1"              : float,
          "model_file"      : str,
          "model_exists"    : bool,
          "pkl_exists"      : bool
        }
    """
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

    if pkl_exists:
        try:
            wrapper      = _load_wrapper()
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
