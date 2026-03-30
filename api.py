"""
api.py
------
FastAPI application for the blood cell classification pipeline.

Endpoints:
  GET  /                  — health check
  GET  /status            — model metadata + uptime
  POST /predict           — single image prediction
  POST /upload            — bulk image upload to Supabase Storage
  POST /retrain           — manual retraining trigger
  GET  /results           — experiment metrics from pkl
  GET  /history           — retraining history from Supabase

Autonomous retraining:
  APScheduler runs a background job every hour.
  If the number of unprocessed uploaded images since the last retraining
  exceeds RETRAIN_THRESHOLD (default 50), retraining is triggered automatically
  with no human intervention required.
"""

import os
import io
import gc
import time
import logging
import asyncio
import datetime
import numpy as np

from PIL import Image
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

import joblib
from supabase import create_client, Client

# ── Local modules ──────────────────────────────────────────────────────────────
from src.prediction   import predict, get_model_metadata, reload_model
from src.model        import retrain
from src.preprocessing import bytes_to_array, CLASSES

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Supabase ───────────────────────────────────────────────────────────────────
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
STORAGE_BUCKET  = os.getenv("SUPABASE_BUCKET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Autonomous retraining config ───────────────────────────────────────────────
RETRAIN_THRESHOLD   = int(os.getenv("RETRAIN_THRESHOLD", "50"))
RETRAIN_CHECK_HOURS = int(os.getenv("RETRAIN_CHECK_HOURS", "1"))

# ── App startup time (for uptime calculation) ──────────────────────────────────
APP_START_TIME = time.time()

# ── Retraining lock (prevents concurrent retraining jobs) ─────────────────────
_retrain_lock = asyncio.Lock()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Blood Cell Classification API",
    description=(
        "End-to-end ML pipeline for white blood cell differential counting. "
        "Classifies EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL from microscope images."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = AsyncIOScheduler()


# ══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS RETRAINING JOB
# ══════════════════════════════════════════════════════════════════════════════

async def autonomous_retrain_job():
    """
    Background job executed every RETRAIN_CHECK_HOURS hours.

    Logic:
      1. Count images in Supabase `uploaded_images` table with
         retrained=False (not yet used in any retraining run).
      2. If count >= RETRAIN_THRESHOLD, download those images from
         Supabase Storage and trigger retraining.
      3. Mark processed images as retrained=True in the database.
      4. Log the result to the `retraining_runs` table.

    This runs completely autonomously — no button press required.
    The only human action needed is uploading images via the UI.
    """
    if _retrain_lock.locked():
        logger.info("[AUTO-RETRAIN] Skipping — another retrain job is running")
        return

    async with _retrain_lock:
        try:
            logger.info("[AUTO-RETRAIN] Checking for new images...")

            # Count unprocessed images
            response = (
                supabase.table("uploaded_images")
                .select("id, storage_path, label", count="exact")
                .eq("retrained", False)
                .execute()
            )
            pending_records = response.data
            count           = len(pending_records)

            logger.info(f"[AUTO-RETRAIN] Pending images: {count} / threshold: {RETRAIN_THRESHOLD}")

            if count < RETRAIN_THRESHOLD:
                logger.info("[AUTO-RETRAIN] Threshold not reached — skipping")
                return

            logger.info(f"[AUTO-RETRAIN] Threshold reached — triggering retraining on {count} images")

            # Download images from Supabase Storage
            train_images, train_labels = [], []
            failed_ids = []

            for record in pending_records:
                try:
                    storage_path = record["storage_path"]
                    label_name   = record.get("label")

                    if label_name not in CLASSES:
                        logger.warning(f"[AUTO-RETRAIN] Unknown label '{label_name}' — skipping")
                        continue

                    raw = supabase.storage.from_(STORAGE_BUCKET).download(storage_path)
                    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
                    train_images.append(img)
                    train_labels.append(CLASSES.index(label_name))

                except Exception as e:
                    logger.error(f"[AUTO-RETRAIN] Failed to download {record['id']}: {e}")
                    failed_ids.append(record["id"])

            if len(train_images) < 10:
                logger.warning("[AUTO-RETRAIN] Too few valid images after download — skipping")
                return

            # Use 80/20 split for train/val within the new images
            split_idx   = max(1, int(len(train_images) * 0.8))
            val_images  = train_images[split_idx:]
            val_labels  = train_labels[split_idx:]
            train_images = train_images[:split_idx]
            train_labels = train_labels[:split_idx]

            if len(val_images) < 1:
                # If too few images for a split, use train set as val
                val_images = train_images
                val_labels = train_labels

            # Run retraining
            result = retrain(
                new_images    = train_images,
                new_labels    = train_labels,
                val_images    = val_images,
                val_labels    = val_labels,
                epochs        = 5,
                batch_size    = 32,
                learning_rate = 1e-4,
            )

            # If model improved, reload it in memory
            if result["improved"]:
                reload_model()
                logger.info("[AUTO-RETRAIN] Model reloaded after improvement")

            # Mark images as retrained in database
            processed_ids = [r["id"] for r in pending_records if r["id"] not in failed_ids]
            if processed_ids:
                supabase.table("uploaded_images").update(
                    {"retrained": True}
                ).in_("id", processed_ids).execute()

            # Log retraining run
            supabase.table("retraining_runs").insert({
                "triggered_by" : "autonomous_scheduler",
                "images_used"  : result["images_used"],
                "f1_before"    : result["f1_before"],
                "f1_after"     : result["f1_after"],
                "improved"     : result["improved"],
                "duration_s"   : result["duration_s"],
                "epochs_run"   : result["epochs_run"],
            }).execute()

            logger.info(f"[AUTO-RETRAIN] Complete — improved={result['improved']}, "
                        f"F1: {result['f1_before']:.4f} → {result['f1_after']:.4f}")

        except Exception as e:
            logger.error(f"[AUTO-RETRAIN] Job failed: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    """Start the autonomous retraining scheduler when the API starts."""
    scheduler.add_job(
        autonomous_retrain_job,
        trigger=IntervalTrigger(hours=RETRAIN_CHECK_HOURS),
        id="auto_retrain",
        name="Autonomous Retraining Job",
        replace_existing=True
    )
    scheduler.start()
    logger.info(
        f"Autonomous retraining scheduler started. "
        f"Checks every {RETRAIN_CHECK_HOURS}h, "
        f"triggers at {RETRAIN_THRESHOLD} new images."
    )


@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown(wait=False)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """Basic health check."""
    return {
        "status"   : "ok",
        "service"  : "Blood Cell Classification API",
        "uptime_s" : round(time.time() - APP_START_TIME, 1)
    }


@app.get("/status", tags=["Health"])
async def status():
    """
    Returns model metadata and API uptime.
    Used by the Streamlit Model Uptime tab.
    """
    meta       = get_model_metadata()
    uptime_s   = round(time.time() - APP_START_TIME, 1)
    uptime_str = str(datetime.timedelta(seconds=int(uptime_s)))

    # Get pending images count from Supabase
    try:
        pending = (
            supabase.table("uploaded_images")
            .select("id", count="exact")
            .eq("retrained", False)
            .execute()
        )
        pending_count = len(pending.data)
    except Exception:
        pending_count = -1

    return {
        **meta,
        "uptime_seconds"           : uptime_s,
        "uptime_human"             : uptime_str,
        "retrain_threshold"        : RETRAIN_THRESHOLD,
        "pending_images_for_retrain": pending_count,
        "scheduler_running"        : scheduler.running,
        "next_auto_retrain"        : str(
            scheduler.get_job("auto_retrain").next_run_time
            if scheduler.get_job("auto_retrain") else "N/A"
        ),
    }


@app.post("/predict", tags=["Inference"])
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict the white blood cell type from a single uploaded image.

    Accepts: JPEG, PNG
    Returns: predicted class, confidence (%), and scores for all 4 classes.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPEG or PNG."
        )

    try:
        image_bytes = await file.read()
        result      = predict(image_bytes)
        return {
            "filename"   : file.filename,
            "prediction" : result,
            "status"     : "ok"
        }
    except Exception as e:
        logger.error(f"/predict error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Data"])
async def upload_images(
    files:      List[UploadFile] = File(...),
    label:      str = "UNKNOWN",
    background: BackgroundTasks = None
):
    """
    Upload one or more labelled images to Supabase Storage for retraining.

    Parameters
    ----------
    files : List[UploadFile]
        Images to upload (JPEG or PNG).
    label : str
        Class label for all images in this batch.
        Must be one of: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL.

    After upload, the autonomous scheduler will detect these images within
    its next check cycle and trigger retraining if the threshold is met.
    """
    if label not in CLASSES and label != "UNKNOWN":
        raise HTTPException(
            status_code=422,
            detail=f"Invalid label '{label}'. Choose from {CLASSES}."
        )

    results       = []
    uploaded_ids  = []

    for file in files:
        try:
            image_bytes  = await file.read()
            storage_path = f"uploads/{label}/{int(time.time())}_{file.filename}"

            # Upload to Supabase Storage
            supabase.storage.from_(STORAGE_BUCKET).upload(
                storage_path,
                image_bytes,
                {"content-type": file.content_type or "image/jpeg"}
            )

            # Log to database
            record = supabase.table("uploaded_images").insert({
                "filename"     : file.filename,
                "storage_path" : storage_path,
                "label"        : label,
                "retrained"    : False,
            }).execute()

            uploaded_ids.append(record.data[0]["id"])
            results.append({"filename": file.filename, "status": "uploaded"})

        except Exception as e:
            logger.error(f"Upload failed for {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "failed", "error": str(e)})

    # Check if threshold is now met and trigger immediately if so
    try:
        pending_count = len(
            supabase.table("uploaded_images")
            .select("id", count="exact")
            .eq("retrained", False)
            .execute().data
        )
        auto_triggered = pending_count >= RETRAIN_THRESHOLD
        if auto_triggered and not _retrain_lock.locked():
            # Schedule immediate execution in background
            scheduler.add_job(
                autonomous_retrain_job,
                id="immediate_retrain",
                replace_existing=True
            )
    except Exception:
        pending_count  = -1
        auto_triggered = False

    return {
        "uploaded"          : len([r for r in results if r["status"] == "uploaded"]),
        "failed"            : len([r for r in results if r["status"] == "failed"]),
        "results"           : results,
        "pending_total"     : pending_count,
        "auto_retrain_triggered": auto_triggered,
        "retrain_threshold" : RETRAIN_THRESHOLD,
    }


@app.post("/retrain", tags=["Training"])
async def manual_retrain():
    """
    Manually trigger retraining on all pending (not yet retrained) images.

    This endpoint is called by the Streamlit 'Trigger Retrain' button.
    The autonomous scheduler also calls the same underlying logic automatically.

    Returns a summary of the retraining result.
    """
    if _retrain_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"detail": "Retraining already in progress. Please wait."}
        )

    async with _retrain_lock:
        # Fetch pending images
        response = (
            supabase.table("uploaded_images")
            .select("id, storage_path, label")
            .eq("retrained", False)
            .execute()
        )
        pending = response.data

        if len(pending) < 5:
            return {
                "status" : "skipped",
                "reason" : f"Only {len(pending)} pending images. Minimum is 5.",
                "pending": len(pending)
            }

        # Download images
        train_images, train_labels, used_ids = [], [], []
        for record in pending:
            try:
                raw = supabase.storage.from_(STORAGE_BUCKET).download(record["storage_path"])
                img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
                label_name = record.get("label", "")
                if label_name not in CLASSES:
                    continue
                train_images.append(img)
                train_labels.append(CLASSES.index(label_name))
                used_ids.append(record["id"])
            except Exception as e:
                logger.error(f"Download failed for {record['id']}: {e}")

        if len(train_images) < 5:
            return {"status": "skipped", "reason": "Too few valid images after download."}

        split_idx    = max(1, int(len(train_images) * 0.8))
        val_images   = train_images[split_idx:]
        val_labels   = train_labels[split_idx:]
        train_images = train_images[:split_idx]
        train_labels = train_labels[:split_idx]

        if not val_images:
            val_images = train_images
            val_labels = train_labels

        # Retrain
        result = retrain(
            new_images    = train_images,
            new_labels    = train_labels,
            val_images    = val_images,
            val_labels    = val_labels,
            epochs        = 5,
            batch_size    = 32,
            learning_rate = 1e-4,
        )

        if result["improved"]:
            reload_model()

        # Mark as retrained
        if used_ids:
            supabase.table("uploaded_images").update(
                {"retrained": True}
            ).in_("id", used_ids).execute()

        # Log run
        supabase.table("retraining_runs").insert({
            "triggered_by": "manual_ui",
            "images_used" : result["images_used"],
            "f1_before"   : result["f1_before"],
            "f1_after"    : result["f1_after"],
            "improved"    : result["improved"],
            "duration_s"  : result["duration_s"],
            "epochs_run"  : result["epochs_run"],
        }).execute()

        return {"status": "complete", **result}


@app.get("/results", tags=["Metrics"])
async def get_results():
    """
    Return the experiment comparison metrics stored in blood_cell_results.pkl.
    Used by the Streamlit Results tab.
    """
    pkl_path = os.path.join("data", "blood_cell_results.pkl")
    if not os.path.exists(pkl_path):
        raise HTTPException(status_code=404, detail="Results file not found.")
    try:
        metadata = joblib.load(pkl_path)
        return {"status": "ok", "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["Training"])
async def get_retrain_history():
    """
    Return all past retraining runs from Supabase.
    Includes both manual and autonomous runs.
    """
    try:
        response = (
            supabase.table("retraining_runs")
            .select("*")
            .order("triggered_at", desc=True)
            .limit(50)
            .execute()
        )
        return {"status": "ok", "runs": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
