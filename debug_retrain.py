import os, requests
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL", "https://blood-cell-api.onrender.com")

print("Triggering retrain and reading full response...")
r = requests.post(f"{API_URL}/retrain", timeout=600)
print(f"HTTP Status : {r.status_code}")
print(f"Response    : {r.text[:2000]}")
