"""
src/model.py
------------
Model architecture definition and retraining logic for the blood cell
classification pipeline.

This module is responsible for:
  1. Building the Custom CNN architecture (used if retraining from scratch)
  2. Loading a saved model from disk
  3. Running the retraining loop on new uploaded images
  4. Saving the updated model back to disk if performance improves
"""

import os
import gc
import time
import logging
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing import prepare_batch_for_retraining, IMG_SIZE, NUM_CLASSES

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_KERAS_PATH = os.path.join("models", "blood_cell_best.keras")
MODEL_PKL_PATH   = os.path.join("models", "blood_cell_best.pkl")
CLASSES          = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


# ── Architecture ───────────────────────────────────────────────────────────────

def build_custom_cnn(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build the Custom CNN architecture used in the original training pipeline.

    Architecture: 4 Conv blocks (32 → 64 → 128 → 256 filters) with
    BatchNormalization and MaxPooling, followed by GlobalAveragePooling
    and a Dense classification head with Dropout regularisation.

    This is used when retraining from scratch is required.
    For fine-tuning, use load_model() instead.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 4 (EOSINOPHIL, LYMPHOCYTE,
        MONOCYTE, NEUTROPHIL).

    Returns
    -------
    tf.keras.Model
        Uncompiled model ready for .compile() and .fit().
    """
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x   = tf.keras.layers.Dense(512, activation="relu")(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax",
                                 dtype="float32")(x)

    return tf.keras.Model(inp, out, name="Custom_CNN")


def load_model(path: str = MODEL_KERAS_PATH) -> tf.keras.Model:
    """
    Load a trained Keras model from disk.

    Parameters
    ----------
    path : str
        Path to the .keras file.

    Returns
    -------
    tf.keras.Model
        Loaded model ready for inference or fine-tuning.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Ensure blood_cell_best.keras is placed in the models/ directory."
        )
    logger.info(f"Loading model from {path}")
    return tf.keras.models.load_model(path)


# ── Retraining ─────────────────────────────────────────────────────────────────

def retrain(
    new_images:     list,
    new_labels:     list,
    val_images:     list,
    val_labels:     list,
    epochs:         int   = 5,
    batch_size:     int   = 32,
    learning_rate:  float = 1e-4,
    model_path:     str   = MODEL_KERAS_PATH,
    pkl_path:       str   = MODEL_PKL_PATH,
) -> dict:
    """
    Fine-tune the saved model on newly uploaded images and save only if
    weighted F1 improves on the validation set.

    This function is called both by the manual /retrain endpoint and by the
    autonomous background scheduler in api.py.

    Parameters
    ----------
    new_images : list of np.ndarray
        Raw uint8 image arrays to train on.
    new_labels : list of int
        Integer class indices (0–3) for each training image.
    val_images : list of np.ndarray
        Raw uint8 image arrays for validation.
    val_labels : list of int
        Integer class indices for validation images.
    epochs : int
        Maximum number of fine-tuning epochs (EarlyStopping may fire earlier).
    batch_size : int
        Batch size for the retraining loop.
    learning_rate : float
        Learning rate for fine-tuning. Keep small (1e-4 or lower) to avoid
        overwriting previously learned features.
    model_path : str
        Path to the .keras checkpoint to load and conditionally overwrite.
    pkl_path : str
        Path to the .pkl wrapper to update if model improves.

    Returns
    -------
    dict
        {
          "f1_before"     : float,
          "f1_after"      : float,
          "acc_before"    : float,
          "acc_after"     : float,
          "improved"      : bool,
          "epochs_run"    : int,
          "images_used"   : int,
          "duration_s"    : float,
          "saved_path"    : str or None
        }
    """
    start = time.time()
    logger.info(f"Retraining started — {len(new_images)} images, {epochs} epochs max")

    # ── Load current model ────────────────────────────────────────────────────
    model = load_model(model_path)

    # ── Evaluate BEFORE ───────────────────────────────────────────────────────
    X_val, y_val = prepare_batch_for_retraining(
        val_images, val_labels, apply_augmentation=False
    )
    probs_before  = model.predict(X_val, batch_size=batch_size, verbose=0)
    preds_before  = np.argmax(probs_before, axis=1)
    true_val      = np.argmax(y_val, axis=1)
    f1_before     = f1_score(true_val, preds_before, average="weighted")
    acc_before    = accuracy_score(true_val, preds_before)
    logger.info(f"BEFORE — Accuracy: {acc_before*100:.2f}%  F1: {f1_before:.4f}")

    # ── Prepare training data ─────────────────────────────────────────────────
    X_train, y_train = prepare_batch_for_retraining(
        new_images, new_labels, apply_augmentation=True
    )

    # ── Compile with small LR to avoid catastrophic forgetting ───────────────
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5,
            clipnorm=1.0
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    epochs_run = len(history.history["loss"])

    # ── Evaluate AFTER ────────────────────────────────────────────────────────
    probs_after = model.predict(X_val, batch_size=batch_size, verbose=0)
    preds_after = np.argmax(probs_after, axis=1)
    f1_after    = f1_score(true_val, preds_after, average="weighted")
    acc_after   = accuracy_score(true_val, preds_after)
    logger.info(f"AFTER  — Accuracy: {acc_after*100:.2f}%  F1: {f1_after:.4f}")

    # ── Conditionally save ────────────────────────────────────────────────────
    improved   = f1_after >= f1_before
    saved_path = None

    if improved:
        model.save(model_path)
        saved_path = model_path
        logger.info(f"Model improved ({f1_before:.4f} → {f1_after:.4f}) — saved to {model_path}")

        # Update the pkl wrapper as well so prediction.py stays in sync
        if os.path.exists(pkl_path):
            wrapper = joblib.load(pkl_path)
            wrapper.model_json    = model.to_json()
            wrapper.model_weights = model.get_weights()
            wrapper.metrics.update({
                "Accuracy" : round(acc_after * 100, 2),
                "F1"       : round(f1_after  * 100, 2),
            })
            joblib.dump(wrapper, pkl_path, compress=3)
            logger.info(f"PKL wrapper updated at {pkl_path}")
    else:
        logger.info(f"No improvement ({f1_before:.4f} → {f1_after:.4f}) — original model kept")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, X_train, y_train, X_val, y_val
    gc.collect()

    duration = round(time.time() - start, 1)
    logger.info(f"Retraining complete in {duration}s")

    return {
        "f1_before"  : round(f1_before,  4),
        "f1_after"   : round(f1_after,   4),
        "acc_before" : round(acc_before * 100, 2),
        "acc_after"  : round(acc_after  * 100, 2),
        "improved"   : improved,
        "epochs_run" : epochs_run,
        "images_used": len(new_images),
        "duration_s" : duration,
        "saved_path" : saved_path,
    }
