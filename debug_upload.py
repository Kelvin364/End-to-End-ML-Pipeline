"""
Standalone upload test to see the exact Supabase error.
"""
import os, requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://blood-cell-api.onrender.com")

image_bytes = open("data/test_images/eosinophil.jpg", "rb").read()

print(f"Testing upload against: {API_URL}")
print(f"Image size: {len(image_bytes)/1024:.1f} KB")

r = requests.post(
    f"{API_URL}/upload",
    files={"files": ("eosinophil.jpg", image_bytes, "image/jpeg")},
    params={"label": "EOSINOPHIL"},
    timeout=60
)

print(f"\nHTTP Status: {r.status_code}")
print(f"Response: {r.text}")
