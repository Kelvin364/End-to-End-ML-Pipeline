"""
scripts/retrain_job.py
Bypasses keras model config deserialization entirely.
Rebuilds architecture from code, loads weights from .keras zip.
This works regardless of keras version differences.
"""

import os, io, sys, gc, time, zipfile, tempfile
import numpy as np

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
STORAGE_BUCKET = os.environ.get("SUPABASE_BUCKET", "cell-images")
CLASSES        = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE       = (96, 96)
MIN_IMAGES     = 5
MODEL_LOCAL    = "/tmp/blood_cell_best.keras"
LOG_FILE       = "/tmp/retrain_log.txt"
log_lines      = []

def log(msg):
    print(msg)
    log_lines.append(str(msg))

def save_log():
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_lines))

# ── Connect ────────────────────────────────────────────────────────────────────
try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    log("Connected to Supabase")
except Exception as e:
    log(f"Supabase connection failed: {e}")
    save_log()
    sys.exit(1)

# ── Step 1: Check pending ──────────────────────────────────────────────────────
log("\nStep 1: Checking pending images...")
try:
    response = (
        supabase.table("uploaded_images")
        .select("id, storage_path, label")
        .eq("retrained", False)
        .execute()
    )
    pending = response.data
    count   = len(pending)
    log(f"  Pending: {count}")
except Exception as e:
    log(f"  DB query failed: {e}")
    save_log()
    sys.exit(1)

if count < MIN_IMAGES:
    log(f"  Not enough images (need {MIN_IMAGES}). Exiting cleanly.")
    save_log()
    sys.exit(0)

# ── Step 2: Download model ─────────────────────────────────────────────────────
log("\nStep 2: Downloading model...")
try:
    model_bytes = supabase.storage.from_(STORAGE_BUCKET).download(
        "models/blood_cell_best.keras"
    )
    with open(MODEL_LOCAL, "wb") as f:
        f.write(model_bytes)
    log(f"  Downloaded: {len(model_bytes)/1e6:.1f} MB")
except Exception as e:
    log(f"  Download failed: {e}")
    if os.path.exists("models/blood_cell_best.keras"):
        MODEL_LOCAL = "models/blood_cell_best.keras"
        log("  Using repo model file")
    else:
        log("  No model available. Exiting.")
        save_log()
        sys.exit(1)

# ── Step 3: Build architecture + load weights ──────────────────────────────────
# This completely bypasses keras model config deserialization.
# We rebuild the exact same architecture from Python code,
# then load only the weights from the .keras zip file.
log("\nStep 3: Rebuilding model architecture and loading weights...")
try:
    import tensorflow as tf

    def build_model():
        inp = tf.keras.Input(shape=(96, 96, 3))
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
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        out = tf.keras.layers.Dense(4, activation="softmax", dtype="float32")(x)
        return tf.keras.Model(inp, out, name="Custom_CNN")

    model = build_model()

    # Keras 3.x stores weights in model.weights.h5 inside the .keras zip.
    # The correct way to load them is to pass the .keras file directly
    # to load_weights — Keras handles the extraction internally.
    with zipfile.ZipFile(MODEL_LOCAL, "r") as z:
        log(f"  Files in .keras: {z.namelist()}")

    # load_weights accepts .keras files directly in Keras 3.x
    model.load_weights(MODEL_LOCAL)
    log("  Weights loaded successfully from .keras file")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    log(f"  Model ready: {model.count_params():,} parameters")

except Exception as e:
    log(f"  Model setup failed: {e}")
    import traceback
    log(traceback.format_exc())
    save_log()
    sys.exit(1)

# ── Step 4: Download images ────────────────────────────────────────────────────
log(f"\nStep 4: Downloading {count} images...")
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

images, labels, used_ids = [], [], []
for i, record in enumerate(pending):
    try:
        raw   = supabase.storage.from_(STORAGE_BUCKET).download(
            record["storage_path"]
        )
        img   = np.array(
            Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
        ).astype(np.float32) / 255.0
        label = record.get("label", "")
        if label not in CLASSES:
            continue
        images.append(img)
        labels.append(CLASSES.index(label))
        used_ids.append(record["id"])
        if (i + 1) % 10 == 0:
            log(f"  Progress: {i+1}/{count}")
    except Exception as e:
        log(f"  Skipped {record.get('id','?')}: {e}")

log(f"  Loaded: {len(images)} images")
if len(images) < MIN_IMAGES:
    log("  Not enough valid images. Exiting.")
    save_log()
    sys.exit(0)

# ── Step 5: Prepare ────────────────────────────────────────────────────────────
X = np.array(images)
y = np.eye(len(CLASSES))[labels]
if len(X) >= 10:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_val = X, X
    y_train, y_val = y, y
log(f"\nStep 5: Train={len(X_train)} Val={len(X_val)}")

# ── Step 6: Evaluate BEFORE ────────────────────────────────────────────────────
log("\nStep 6: Evaluating BEFORE...")
preds_before = np.argmax(model.predict(X_val, verbose=0), axis=1)
true_val     = np.argmax(y_val, axis=1)
f1_before    = f1_score(true_val, preds_before, average="weighted", zero_division=0)
log(f"  F1: {f1_before:.4f}")

# ── Step 7: Retrain ────────────────────────────────────────────────────────────
log("\nStep 7: Retraining...")
start = time.time()
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5, batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )],
    verbose=1
)
duration = round(time.time() - start, 1)

# ── Step 8: Evaluate AFTER ─────────────────────────────────────────────────────
log("\nStep 8: Evaluating AFTER...")
preds_after = np.argmax(model.predict(X_val, verbose=0), axis=1)
f1_after    = f1_score(true_val, preds_after, average="weighted", zero_division=0)
improved    = f1_after >= f1_before
log(f"  F1: {f1_after:.4f} | Improved: {improved}")

# ── Step 9: Save if improved ───────────────────────────────────────────────────
if improved:
    log("\nStep 9: Saving improved model...")
    model.save(MODEL_LOCAL)
    with open(MODEL_LOCAL, "rb") as f:
        new_bytes = f.read()
    try:
        supabase.storage.from_(STORAGE_BUCKET).update(
            "models/blood_cell_best.keras", new_bytes,
            {"content-type": "application/octet-stream"}
        )
    except Exception:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            "models/blood_cell_best.keras", new_bytes,
            {"content-type": "application/octet-stream"}
        )
    log(f"  Saved {len(new_bytes)/1e6:.1f} MB to Supabase")
else:
    log("\nStep 9: No improvement — original model kept")

# ── Step 10: Update DB ─────────────────────────────────────────────────────────
if used_ids:
    supabase.table("uploaded_images").update(
        {"retrained": True}
    ).in_("id", used_ids).execute()
    log(f"\nStep 10: Marked {len(used_ids)} images as retrained")

supabase.table("retraining_runs").insert({
    "triggered_by": "github_actions_autonomous",
    "images_used" : len(images),
    "f1_before"   : round(f1_before, 4),
    "f1_after"    : round(f1_after,  4),
    "improved"    : improved,
    "duration_s"  : duration,
    "epochs_run"  : 5,
}).execute()
log("  Run logged to database")

log(f"\n{'='*50}\nRETRAINING COMPLETE\nF1: {f1_before:.4f} → {f1_after:.4f}\nImproved: {improved}\n{'='*50}")
save_log()
