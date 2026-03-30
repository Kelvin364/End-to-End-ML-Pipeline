"""
locustfile.py
-------------
Locust load test for the blood cell classification API.

Usage:
  # Against local Docker container (1 container)
  locust -f locustfile.py --host http://localhost:8000

  # Against deployed Render URL
  locust -f locustfile.py --host https://your-app.onrender.com

  # Headless mode (no UI) — for CI/CD
  locust -f locustfile.py --host http://localhost:8000 \\
    --users 50 --spawn-rate 5 --run-time 60s --headless

Docker container scaling for comparison:
  1 container : docker-compose up --scale api=1
  2 containers: docker-compose up --scale api=2
  3 containers: docker-compose up --scale api=3

  Then run Locust at each scale and compare latency/RPS in the report.
"""

import os
import io
import random
import requests
import numpy as np
from PIL import Image
from locust import HttpUser, task, between, events


# ── Generate synthetic test images ─────────────────────────────────────────────
# Creates in-memory JPEG images so Locust does not depend on local image files.

def _make_synthetic_cell_image() -> bytes:
    """
    Generate a synthetic 96×96 RGB image that mimics a stained cell image.
    The image has a coloured circular nucleus on a light pink background —
    sufficient to test API throughput without requiring real images.
    """
    img = np.ones((96, 96, 3), dtype=np.uint8) * 230  # pale background

    # Random nucleus colour (approximate stain variation)
    colours = [
        [180, 100, 120],   # eosinophil — pinkish
        [60,  60,  160],   # lymphocyte — dark purple
        [120, 80,  100],   # monocyte   — medium purple
        [140, 120, 160],   # neutrophil — pale purple
    ]
    colour = random.choice(colours)

    # Draw nucleus (filled circle)
    cx, cy = 48, 48
    r      = random.randint(20, 35)
    Y, X   = np.ogrid[:96, :96]
    mask   = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask] = colour

    # Add slight noise
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# Pre-generate a small pool of synthetic images to reuse across tasks
_IMAGE_POOL = [_make_synthetic_cell_image() for _ in range(20)]


# ══════════════════════════════════════════════════════════════════════════════
# USER BEHAVIOURS
# ══════════════════════════════════════════════════════════════════════════════

class BloodCellAPIUser(HttpUser):
    """
    Simulates a typical user of the blood cell classification API.

    Task weights reflect realistic usage patterns:
      - Prediction requests are the most frequent (weight 5)
      - Status checks happen regularly (weight 3)
      - History reads are less frequent (weight 2)
      - Uploads happen occasionally (weight 1)
    """
    wait_time = between(0.5, 2.0)   # seconds between requests per user

    @task(5)
    def predict(self):
        """
        POST /predict — single image classification.
        This is the primary endpoint and will receive the most traffic.
        """
        image_bytes = random.choice(_IMAGE_POOL)
        with self.client.post(
            "/predict",
            files={"file": ("cell.jpg", image_bytes, "image/jpeg")},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if "prediction" in body and "label" in body["prediction"]:
                    response.success()
                else:
                    response.failure(f"Unexpected response shape: {body}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    @task(3)
    def check_status(self):
        """
        GET /status — model metadata and uptime check.
        Simulates the Streamlit UI polling for model status.
        """
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def get_history(self):
        """
        GET /history — retraining run history.
        Simulates users viewing the History tab in the UI.
        """
        with self.client.get("/history", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_results(self):
        """
        GET /results — experiment metrics.
        Simulates users viewing the Visualisations tab.
        """
        with self.client.get("/results", catch_response=True) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def upload_single_image(self):
        """
        POST /upload — single image upload to simulate data collection.
        Weight 1 — much less frequent than predictions.
        """
        image_bytes = random.choice(_IMAGE_POOL)
        label       = random.choice(["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"])
        with self.client.post(
            "/upload",
            files={"files": ("cell.jpg", image_bytes, "image/jpeg")},
            params={"label": label},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    def on_start(self):
        """Called once per simulated user when the test starts."""
        # Verify the API is reachable before running tasks
        try:
            r = self.client.get("/")
            if r.status_code != 200:
                print(f"Warning: API health check returned {r.status_code}")
        except Exception as e:
            print(f"Warning: Could not reach API on start: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HEAVY LOAD USER — for stress testing /predict only
# ══════════════════════════════════════════════════════════════════════════════

class HeavyPredictUser(HttpUser):
    """
    Simulates a heavy-load scenario where all requests are predictions.
    Use this class with --tags heavy to isolate prediction throughput.
    """
    wait_time = between(0.1, 0.5)

    @task
    def predict_only(self):
        image_bytes = random.choice(_IMAGE_POOL)
        self.client.post(
            "/predict",
            files={"file": ("cell.jpg", image_bytes, "image/jpeg")}
        )


# ══════════════════════════════════════════════════════════════════════════════
# EVENT HOOKS — printed at test completion
# ══════════════════════════════════════════════════════════════════════════════

@events.quitting.add_listener
def on_locust_quit(environment, **kwargs):
    stats = environment.stats.total
    print("\n" + "="*55)
    print("LOAD TEST SUMMARY")
    print("="*55)
    print(f"  Total requests     : {stats.num_requests}")
    print(f"  Total failures     : {stats.num_failures}")
    print(f"  Median response    : {stats.median_response_time:.1f} ms")
    print(f"  95th percentile    : {stats.get_response_time_percentile(0.95):.1f} ms")
    print(f"  99th percentile    : {stats.get_response_time_percentile(0.99):.1f} ms")
    print(f"  Max response       : {stats.max_response_time:.1f} ms")
    print(f"  Requests/second    : {stats.current_rps:.1f}")
    print("="*55)
    print("\nRecord these values for each Docker container count (1, 2, 3)")
    print("and include the comparison in your submission report.\n")
