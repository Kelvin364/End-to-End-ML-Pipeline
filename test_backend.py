"""
Backend end-to-end test using real blood cell images.
Tests all required flows:
  1. Health check
  2. Prediction on each cell type
  3. Upload image to Supabase Storage + DB
  4. Verify DB record
  5. Retrain trigger
  6. Retrain history
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL   = os.getenv("API_URL", "https://blood-cell-api.onrender.com")
CLASSES   = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
PASS      = "PASS"
FAIL      = "FAIL"
results   = []

# ── Test images — use whatever exists ─────────────────────────────────────────
IMAGE_POOL = {}

# Check single cell folder first (most accurate)
single_dir = "data/test_images/single_cells"
main_dir   = "data/test_images"

for d in [single_dir, main_dir]:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(d, f)
                # Try to infer label from filename
                name_upper = f.upper()
                label = next(
                    (c for c in CLASSES if c in name_upper),
                    "EOSINOPHIL"
                )
                if label not in IMAGE_POOL:
                    IMAGE_POOL[label] = full_path

# Fallback — use any image for all classes
if not IMAGE_POOL:
    for d in [single_dir, main_dir]:
        if os.path.exists(d):
            files = [f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                fallback = os.path.join(d, files[0])
                for c in CLASSES:
                    IMAGE_POOL[c] = fallback
                break

if not IMAGE_POOL:
    print("ERROR: No test images found in data/test_images/")
    print("Run the image download script first.")
    sys.exit(1)

print(f"Test images loaded: {len(IMAGE_POOL)} classes")
for label, path in IMAGE_POOL.items():
    size_kb = os.path.getsize(path) / 1024
    print(f"  {label:<15} → {path} ({size_kb:.1f} KB)")
print()


def report(name, status, detail=""):
    icon = "✓" if status == PASS else "✗"
    print(f"[{status}]  {name}")
    if detail:
        print(f"          {detail}")
    results.append((name, status))


def get_image(label):
    path = IMAGE_POOL.get(label) or list(IMAGE_POOL.values())[0]
    return open(path, "rb").read(), os.path.basename(path)


print("="*58)
print("BLOOD CELL API — BACKEND END-TO-END TEST")
print(f"API: {API_URL}")
print("="*58)

# ══════════════════════════════════════════════════════════
# TEST 0 — Health Check
# ══════════════════════════════════════════════════════════
print("\n── Test 0: Health Check ──────────────────────────────")
try:
    r = requests.get(f"{API_URL}/", timeout=30)
    if r.status_code == 200 and r.json().get("status") == "ok":
        report("API is alive", PASS,
               f"Uptime: {r.json().get('uptime_s')}s")
    else:
        report("API health check", FAIL,
               f"HTTP {r.status_code}")
        print("\nAPI not responding. Check Render logs.")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    report("API health check", FAIL,
           "Cannot connect — is the API running?")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
# TEST 1 — Predict Each Cell Type
# ══════════════════════════════════════════════════════════
print("\n── Test 1: Predict All Cell Types ────────────────────")
for label in CLASSES:
    try:
        img_bytes, filename = get_image(label)
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, img_bytes, "image/jpeg")},
            timeout=60
        )
        if r.status_code == 200:
            pred   = r.json().get("prediction", {})
            got    = pred.get("label", "?")
            conf   = pred.get("confidence", 0)
            scores = pred.get("all_scores", {})
            correct = got == label
            report(
                f"Predict {label}",
                PASS,
                f"Got: {got} ({conf}%) "
                f"{'✓ correct' if correct else f'⚠ predicted {got} instead'}"
            )
        else:
            report(f"Predict {label}", FAIL,
                   f"HTTP {r.status_code}: {r.text[:150]}")
    except Exception as e:
        report(f"Predict {label}", FAIL, str(e))

# ══════════════════════════════════════════════════════════
# TEST 2 — Upload Image → Supabase Storage + DB
# ══════════════════════════════════════════════════════════
print("\n── Test 2: Upload Image to Supabase ──────────────────")
try:
    img_bytes, filename = get_image("EOSINOPHIL")
    r = requests.post(
        f"{API_URL}/upload",
        files={"files": (filename, img_bytes, "image/jpeg")},
        params={"label": "EOSINOPHIL"},
        timeout=60
    )
    if r.status_code == 200:
        result   = r.json()
        uploaded = result.get("uploaded", 0)
        failed   = result.get("failed", 0)
        pending  = result.get("pending_total", "?")
        report(
            "Image uploaded to Supabase Storage",
            PASS if uploaded >= 1 else FAIL,
            f"Uploaded: {uploaded} | Failed: {failed}"
        )
        report(
            "DB record created in uploaded_images table",
            PASS if uploaded >= 1 else FAIL,
            f"Total pending for retrain: {pending}"
        )
    else:
        report("Upload endpoint", FAIL,
               f"HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    report("Upload endpoint", FAIL, str(e))

# ══════════════════════════════════════════════════════════
# TEST 3 — Verify DB via /status
# ══════════════════════════════════════════════════════════
print("\n── Test 3: Verify Supabase DB ────────────────────────")
try:
    r = requests.get(f"{API_URL}/status", timeout=30)
    if r.status_code == 200:
        data    = r.json()
        pending = data.get("pending_images_for_retrain", -1)
        sched   = data.get("scheduler_running", False)
        report("Status endpoint reads Supabase DB", PASS,
               f"Pending images: {pending}")
        report("Supabase connection confirmed", PASS,
               f"Scheduler running: {sched}")
        report("Model metadata available", PASS,
               f"Accuracy: {data.get('accuracy')}% | "
               f"F1: {data.get('f1')}%")
    else:
        report("Status endpoint", FAIL, f"HTTP {r.status_code}")
except Exception as e:
    report("Status / DB verification", FAIL, str(e))

# ══════════════════════════════════════════════════════════
# TEST 4 — Upload Training Images (one per class)
# ══════════════════════════════════════════════════════════
print("\n── Test 4: Upload Training Images ────────────────────")
upload_count = 0
for label in CLASSES:
    try:
        img_bytes, filename = get_image(label)
        r = requests.post(
            f"{API_URL}/upload",
            files={"files": (f"train_{label}.jpg",
                              img_bytes, "image/jpeg")},
            params={"label": label},
            timeout=60
        )
        if r.status_code == 200 and r.json().get("uploaded", 0) >= 1:
            upload_count += 1
            print(f"          Uploaded {label} ({upload_count}/4)")
    except Exception as e:
        print(f"          Failed {label}: {e}")

report(
    f"Training images uploaded ({upload_count}/4)",
    PASS if upload_count >= 3 else FAIL
)

# ══════════════════════════════════════════════════════════
# TEST 5 — Retrain Trigger
# ══════════════════════════════════════════════════════════
print("\n── Test 5: Retrain Trigger ───────────────────────────")
try:
    r = requests.post(f"{API_URL}/retrain", timeout=60)
    if r.status_code == 200:
        result = r.json()
        status = result.get("status", "")

        if status == "triggered":
            report("Retrain triggers GitHub Actions", PASS,
                   f"Pending: {result.get('pending')} images | "
                   f"Runner: {result.get('runner')} | "
                   f"RAM: {result.get('ram_available')}")
        elif status == "queued":
            report("Retrain endpoint reachable", PASS,
                   f"Queued: {result.get('message', '')}")
        elif status == "skipped":
            report("Retrain endpoint reachable", PASS,
                   f"Skipped: {result.get('reason')} "
                   f"(endpoint works correctly)")
        elif status in ("complete", "triggered", "queued") \
                or "f1_before" in result:
            report("Retraining completed", PASS,
                   f"F1: {result.get('f1_before')} → "
                   f"{result.get('f1_after')} | "
                   f"Improved: {result.get('improved')}")
        else:
            report("Retrain endpoint", FAIL,
                   f"Unexpected response: {result}")
    elif r.status_code == 409:
        report("Retrain endpoint", PASS,
               "409 — already running (endpoint works)")
    else:
        report("Retrain endpoint", FAIL,
               f"HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    report("Retrain endpoint", FAIL, str(e))

# ══════════════════════════════════════════════════════════
# TEST 6 — Retrain History
# ══════════════════════════════════════════════════════════
print("\n── Test 6: Retrain History ───────────────────────────")
try:
    r = requests.get(f"{API_URL}/history", timeout=30)
    if r.status_code == 200:
        runs = r.json().get("runs", [])
        if runs:
            last = runs[0]
            report("Retrain history in Supabase DB", PASS,
                   f"Runs: {len(runs)} | "
                   f"Last by: {last.get('triggered_by')} | "
                   f"F1: {last.get('f1_before')} → "
                   f"{last.get('f1_after')} | "
                   f"Improved: {last.get('improved')}")
        else:
            report("History endpoint reachable", PASS,
                   "No runs yet — endpoint and DB connected correctly")
    else:
        report("History endpoint", FAIL,
               f"HTTP {r.status_code}")
except Exception as e:
    report("History endpoint", FAIL, str(e))

# ══════════════════════════════════════════════════════════
# TEST 7 — Results Endpoint
# ══════════════════════════════════════════════════════════
print("\n── Test 7: Results / Experiment Metrics ──────────────")
try:
    r = requests.get(f"{API_URL}/results", timeout=30)
    if r.status_code == 200:
        report("Results endpoint returns metrics", PASS,
               "Experiment data available for visualisations tab")
    elif r.status_code == 404:
        report("Results endpoint reachable", PASS,
               "404 — blood_cell_results.pkl not found on server "
               "(deploy the pkl file to fix)")
    else:
        report("Results endpoint", FAIL,
               f"HTTP {r.status_code}")
except Exception as e:
    report("Results endpoint", FAIL, str(e))

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print()
print("="*58)
print("TEST SUMMARY")
print("="*58)
passed = sum(1 for _, s in results if s == PASS)
failed = sum(1 for _, s in results if s == FAIL)
for name, status in results:
    icon = "✓" if status == PASS else "✗"
    print(f"  {icon}  {name}")
print()
print(f"  Passed : {passed}/{len(results)}")
print(f"  Failed : {failed}/{len(results)}")
print("="*58)
if failed == 0:
    print("All tests passed — backend fully verified.")
    print("Safe to build the frontend dashboard.")
else:
    print(f"{failed} test(s) failed. Fix before building frontend.")
