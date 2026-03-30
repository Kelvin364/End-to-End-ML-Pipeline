"""
src/model.py
Model loading and retraining logic for the blood cell classification pipeline.
"""

import os
import gc
import time
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from src.preprocessing import prepare_batch_for_retraining, IMG_SIZE, NUM_CLASSES

logger           = logging.getLogger(__name__)
MODEL_KERAS_PATH = os.path.join("models", "blood_cell_best.keras")
MODEL_PKL_PATH   = os.path.join("models", "blood_cell_best.pkl")
CLASSES          = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


def build_custom_cnn(num_classes=NUM_CLASSES):
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


def load_model(path=MODEL_KERAS_PATH):
    """
    Load model with quantization_config patch.
    The model was saved with Keras 3.x which added quantization_config
    to Dense layer config. This patch makes Dense silently ignore it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Place blood_cell_best.keras in models/ directory."
        )
    logger.info(f"Loading model from {path}")

    from tensorflow import keras
    original_init = keras.layers.Dense.__init__

    def patched_init(self, *args, quantization_config=None, **kwargs):
        original_init(self, *args, **kwargs)

    keras.layers.Dense.__init__ = patched_init
    try:
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model loaded successfully with quantization patch")
    finally:
        keras.layers.Dense.__init__ = original_init

    return model


def retrain(
    new_images,
    new_labels,
    val_images,
    val_labels,
    epochs=3,
    batch_size=8,
    learning_rate=1e-4,
    model_path=MODEL_KERAS_PATH,
    pkl_path=MODEL_PKL_PATH,
):
    """
    Fine-tune the saved model on new images.
    Saves only if weighted F1 improves.
    Memory-optimised for Render free tier (512MB RAM).
    """
    start = time.time()
    logger.info(f"Retraining: {len(new_images)} images, "
                f"batch={batch_size}, epochs={epochs}")

    model = load_model(model_path)

    # Evaluate BEFORE
    X_val, y_val = prepare_batch_for_retraining(
        val_images, val_labels, apply_augmentation=False
    )
    probs_before = model.predict(X_val, batch_size=4, verbose=0)
    preds_before = np.argmax(probs_before, axis=1)
    true_val     = np.argmax(y_val, axis=1)
    f1_before    = f1_score(true_val, preds_before, average="weighted")
    acc_before   = accuracy_score(true_val, preds_before)
    logger.info(f"BEFORE — F1: {f1_before:.4f}  Acc: {acc_before*100:.2f}%")

    del probs_before, preds_before
    gc.collect()

    # Prepare training data
    X_train, y_train = prepare_batch_for_retraining(
        new_images, new_labels, apply_augmentation=True
    )

    # Compile with small LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2,
                restore_best_weights=True, verbose=1
            )
        ],
        verbose=1
    )
    epochs_run = len(history.history["loss"])

    del X_train, y_train
    gc.collect()

    # Evaluate AFTER
    probs_after = model.predict(X_val, batch_size=4, verbose=0)
    preds_after = np.argmax(probs_after, axis=1)
    f1_after    = f1_score(true_val, preds_after, average="weighted")
    acc_after   = accuracy_score(true_val, preds_after)
    logger.info(f"AFTER  — F1: {f1_after:.4f}  Acc: {acc_after*100:.2f}%")

    # Save only if improved
    improved   = f1_after >= f1_before
    saved_path = None

    if improved:
        model.save(model_path)
        saved_path = model_path
        logger.info(f"Model saved — F1 {f1_before:.4f} -> {f1_after:.4f}")

        if os.path.exists(pkl_path):
            try:
                import joblib
                wrapper = joblib.load(pkl_path)
                wrapper.model_json    = model.to_json()
                wrapper.model_weights = model.get_weights()
                wrapper.metrics.update({
                    "Accuracy": round(acc_after * 100, 2),
                    "F1"      : round(f1_after  * 100, 2),
                })
                joblib.dump(wrapper, pkl_path, compress=3)
            except Exception as e:
                logger.warning(f"Could not update pkl: {e}")
    else:
        logger.info("No improvement — original model kept")

    del model, X_val, y_val
    gc.collect()

    duration = round(time.time() - start, 1)

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
