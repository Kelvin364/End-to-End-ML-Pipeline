"""
locustfile.py
Blood Cell Classification API — Load Test
Tests against the deployed Render API.

Run commands:
  # With UI (recommended for graphs):
  locust -f locustfile.py --host https://blood-cell-api.onrender.com

  # Headless with HTML report:
  locust -f locustfile.py --host https://blood-cell-api.onrender.com \
    --users 10 --spawn-rate 2 --run-time 60s --headless --html reports/report_1_container.html

Usage for rubric:
  Run once per Docker container count (1, 2, 3)
  Record: median response time, 95th percentile, RPS, failures
"""

import io
import os
import random
from PIL import Image
import numpy as np
from locust import HttpUser, task, between, events


# ── Load real test image if available, else generate synthetic ────────────────
def _load_test_image() -> bytes:
    test_path = "data/test_images/eosinophil.jpeg"
    if os.path.exists(test_path):
        return open(test_path, "rb").read()
    # Fallback: generate synthetic cell image
    img   = np.ones((96, 96, 3), dtype=np.uint8) * 220
    cx, cy, r = 48, 48, 25
    Y, X  = np.ogrid[:96, :96]
    mask  = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask] = [180, 80, 100]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    return buf.getvalue()


IMAGE_BYTES = _load_test_image()
CLASSES     = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
print(f"Test image loaded: {len(IMAGE_BYTES)/1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD USER — realistic mixed traffic
# ══════════════════════════════════════════════════════════════════════════════
class BloodCellUser(HttpUser):
    """
    Simulates realistic user traffic.
    Task weights reflect real usage patterns:
      predict=5 (most common), status=3, history=2, results=1
    """
    wait_time = between(1, 3)

    @task(5)
    def predict(self):
        """POST /predict — primary endpoint, highest weight"""
        with self.client.post(
            "/predict",
            files={"file": ("cell.jpg", IMAGE_BYTES, "image/jpeg")},
            catch_response=True
        ) as r:
            if r.status_code == 200:
                body = r.json()
                if "prediction" in body and "label" in body["prediction"]:
                    r.success()
                else:
                    r.failure(f"Bad response shape: {body}")
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(3)
    def status(self):
        """GET /status — UI polls this for uptime display"""
        with self.client.get("/status", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(2)
    def history(self):
        """GET /history — History tab polling"""
        with self.client.get("/history", catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def results(self):
        """GET /results — Visualisations tab"""
        with self.client.get("/results", catch_response=True) as r:
            if r.status_code in (200, 404):
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}")

    def on_start(self):
        r = self.client.get("/")
        if r.status_code != 200:
            print(f"WARNING: API health check returned {r.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# HEAVY USER — predict only, for stress testing
# ══════════════════════════════════════════════════════════════════════════════
class HeavyPredictUser(HttpUser):
    """
    All requests go to /predict.
    Use for measuring pure prediction throughput.
    """
    wait_time = between(0.5, 1.5)

    @task
    def predict_only(self):
        self.client.post(
            "/predict",
            files={"file": ("cell.jpg", IMAGE_BYTES, "image/jpeg")}
        )


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY HOOK — printed when test ends
# ══════════════════════════════════════════════════════════════════════════════
@events.quitting.add_listener
def on_quit(environment, **kwargs):
    stats = environment.stats.total
    print("\n" + "="*55)
    print("LOCUST LOAD TEST SUMMARY")
    print("="*55)
    print(f"  Total requests      : {stats.num_requests}")
    print(f"  Total failures      : {stats.num_failures}")
    print(f"  Failure rate        : {stats.fail_ratio*100:.1f}%")
    print(f"  Median response     : {stats.median_response_time:.0f} ms")
    print(f"  Average response    : {stats.avg_response_time:.0f} ms")
    print(f"  95th percentile     : {stats.get_response_time_percentile(0.95):.0f} ms")
    print(f"  99th percentile     : {stats.get_response_time_percentile(0.99):.0f} ms")
    print(f"  Max response        : {stats.max_response_time:.0f} ms")
    print(f"  Requests/second     : {stats.current_rps:.1f}")
    print("="*55)
    print("\nRecord these values in your README table.")
    print("Run again with 2 and 3 Docker containers to compare.\n")
