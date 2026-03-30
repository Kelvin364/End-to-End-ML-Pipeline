"""
scripts/retrain_job.py
Runs inside GitHub Actions (7GB RAM, free).
No human action required — triggered automatically by Render.
"""

import os, io, sys, gc, time, json
import numpy as np

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
STORAGE_BUCKET = os.environ.get("SUPABASE_BUCKET", "cell-images")
CLASSES        = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE       = (96, 96)
MIN_IMAGES     = 5
MODEL_LOCAL    = "/tmp/blood_cell_best.keras"
LOG_FILE       = "/tmp/retrain_log.txt"

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
log("Connected to Supabase")

# ── Step 1: Check pending images ──────────────────────────────────────────────
log("\nStep 1: Checking pending images...")
response = (
    supabase.table("uploaded_images")
    .select("id, storage_path, label")
    .eq("retrained", False)
    .execute()
)
pending = response.data
count   = len(pending)
log(f"  Pending: {count}")

if count < MIN_IMAGES:
    log(f"  Not enough images (need {MIN_IMAGES}). Exiting.")
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_lines))
    sys.exit(0)

# ── Step 2: Download model ────────────────────────────────────────────────────
log("\nStep 2: Downloading model from Supabase Storage...")
try:
    model_bytes = supabase.storage.from_(STORAGE_BUCKET).download(
        "models/blood_cell_best.keras"
    )
    with open(MODEL_LOCAL, "wb") as f:
        f.write(model_bytes)
    log(f"  Downloaded: {len(model_bytes)/1e6:.1f} MB")
except Exception as e:
    log(f"  Supabase download failed: {e}")
    if os.path.exists("models/blood_cell_best.keras"):
        MODEL_LOCAL = "models/blood_cell_best.keras"
        log("  Using local model file")
    else:
        log("  No model found. Exiting.")
        sys.exit(1)

# ── Step 3: Load model ────────────────────────────────────────────────────────
log("\nStep 3: Loading model...")
import tensorflow as tf
from tensorflow import keras

original_init = keras.layers.Dense.__init__
def patched_init(self, *args, quantization_config=None, **kwargs):
    original_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_init
try:
    model = tf.keras.models.load_model(MODEL_LOCAL, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    log(f"  Loaded: {model.count_params():,} parameters")
finally:
    keras.layers.Dense.__init__ = original_init

# ── Step 4: Download images ───────────────────────────────────────────────────
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

        label_name = record.get("label", "")
        if label_name not in CLASSES:
            log(f"  Skipped unknown label: {label_name}")
            continue

        images.append(img)
        labels.append(CLASSES.index(label_name))
        used_ids.append(record["id"])

        if (i + 1) % 10 == 0:
            log(f"  Progress: {i+1}/{count}")
    except Exception as e:
        log(f"  Skipped {record['id']}: {e}")

log(f"  Loaded: {len(images)} images")

if len(images) < MIN_IMAGES:
    log("  Not enough valid images after download. Exiting.")
    sys.exit(0)

# ── Step 5: Prepare data ──────────────────────────────────────────────────────
log("\nStep 5: Preparing data...")
X = np.array(images)
y = np.eye(len(CLASSES))[labels]

if len(X) >= 10:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_val = X, X
    y_train, y_val = y, y

log(f"  Train: {len(X_train)} | Val: {len(X_val)}")

# ── Step 6: Evaluate BEFORE ───────────────────────────────────────────────────
log("\nStep 6: Evaluating BEFORE...")
preds_before = np.argmax(model.predict(X_val, verbose=0), axis=1)
true_val     = np.argmax(y_val, axis=1)
f1_before    = f1_score(true_val, preds_before,
                         average="weighted", zero_division=0)
acc_before   = accuracy_score(true_val, preds_before)
log(f"  F1: {f1_before:.4f}  Acc: {acc_before*100:.2f}%")

# ── Step 7: Retrain ───────────────────────────────────────────────────────────
log("\nStep 7: Retraining...")
start = time.time()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        )
    ],
    verbose=1
)
duration = round(time.time() - start, 1)

# ── Step 8: Evaluate AFTER ────────────────────────────────────────────────────
log("\nStep 8: Evaluating AFTER...")
preds_after = np.argmax(model.predict(X_val, verbose=0), axis=1)
f1_after    = f1_score(true_val, preds_after,
                        average="weighted", zero_division=0)
acc_after   = accuracy_score(true_val, preds_after)
improved    = f1_after >= f1_before
log(f"  F1: {f1_after:.4f}  Acc: {acc_after*100:.2f}%")
log(f"  Improved: {improved}")

# ── Step 9: Save to Supabase if improved ──────────────────────────────────────
if improved:
    log("\nStep 9: Saving improved model to Supabase Storage...")
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
    log(f"  Saved: {len(new_bytes)/1e6:.1f} MB")
else:
    log("\nStep 9: No improvement — original model kept")

# ── Step 10: Update DB ────────────────────────────────────────────────────────
log("\nStep 10: Updating database...")
if used_ids:
    supabase.table("uploaded_images").update(
        {"retrained": True}
    ).in_("id", used_ids).execute()
    log(f"  Marked {len(used_ids)} images as retrained")

supabase.table("retraining_runs").insert({
    "triggered_by": "github_actions_autonomous",
    "images_used" : len(images),
    "f1_before"   : round(f1_before, 4),
    "f1_after"    : round(f1_after,  4),
    "improved"    : improved,
    "duration_s"  : duration,
    "epochs_run"  : 5,
}).execute()
log("  Run logged to retraining_runs table")

summary = f"""
{'='*50}
RETRAINING COMPLETE
  Images        : {len(images)}
  F1 before     : {f1_before:.4f}
  F1 after      : {f1_after:.4f}
  Improved      : {improved}
  Duration      : {duration}s
{'='*50}
"""
log(summary)

with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
