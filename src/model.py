"""
src/model.py
Retraining logic that reuses the already-loaded model from prediction.py
to avoid loading TensorFlow twice on Render free tier (512MB RAM).
"""

import os
import gc
import time
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

logger           = logging.getLogger(__name__)
MODEL_KERAS_PATH = os.path.join("models", "blood_cell_best.keras")
MODEL_PKL_PATH   = os.path.join("models", "blood_cell_best.pkl")
CLASSES          = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


def retrain(
    new_images,
    new_labels,
    val_images,
    val_labels,
    epochs=2,
    batch_size=4,
    learning_rate=1e-4,
    model_path=MODEL_KERAS_PATH,
    pkl_path=MODEL_PKL_PATH,
):
    """
    Fine-tune using the already-cached model from prediction.py.
    This avoids loading TensorFlow a second time which causes OOM
    on Render free tier (512MB RAM).

    Key design: imports _load_keras_model from prediction.py which
    returns the cached singleton — no new TF instance is created.
    """
    import tensorflow as tf

    # ── Reuse the already-loaded model — do NOT call load_model() again ───────
    # prediction.py caches the model after the first /predict call.
    # We import that cache directly to avoid a second TF model load.
    from src.prediction import _load_keras_model
    model = _load_keras_model()
    logger.info("Retrain: reusing cached model from prediction.py")

    start = time.time()
    logger.info(f"Retraining: {len(new_images)} images, "
                f"batch={batch_size}, epochs={epochs}")

    from src.preprocessing import prepare_batch_for_retraining

    # Evaluate BEFORE — tiny batch to save memory
    X_val, y_val = prepare_batch_for_retraining(
        val_images, val_labels, apply_augmentation=False
    )
    probs_before = model.predict(X_val, batch_size=2, verbose=0)
    preds_before = np.argmax(probs_before, axis=1)
    true_val     = np.argmax(y_val, axis=1)
    f1_before    = f1_score(true_val, preds_before, average="weighted")
    acc_before   = accuracy_score(true_val, preds_before)
    logger.info(f"BEFORE — F1: {f1_before:.4f}")

    del probs_before, preds_before
    gc.collect()

    # Prepare training data — tiny batches
    X_train, y_train = prepare_batch_for_retraining(
        new_images, new_labels, apply_augmentation=True
    )

    # Compile with very small LR to avoid disrupting existing weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train — 2 epochs max, batch size 4 to stay in 512MB
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=1,
                restore_best_weights=True, verbose=1
            )
        ],
        verbose=1
    )
    epochs_run = len(history.history["loss"])

    del X_train, y_train
    gc.collect()

    # Evaluate AFTER
    probs_after = model.predict(X_val, batch_size=2, verbose=0)
    preds_after = np.argmax(probs_after, axis=1)
    f1_after    = f1_score(true_val, preds_after, average="weighted")
    acc_after   = accuracy_score(true_val, preds_after)
    logger.info(f"AFTER  — F1: {f1_after:.4f}")

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
                wrapper.metrics.update({
                    "Accuracy": round(acc_after * 100, 2),
                    "F1"      : round(f1_after  * 100, 2),
                })
                joblib.dump(wrapper, pkl_path, compress=3)
            except Exception as e:
                logger.warning(f"Could not update pkl: {e}")
    else:
        logger.info("No improvement — original model kept")

    del X_val, y_val
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
