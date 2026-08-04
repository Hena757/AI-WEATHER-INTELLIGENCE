"""Test the Flask REST API endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.app import create_app

# Sample weather observation with all required features
SAMPLE_INPUT = {
    "Location": "Sydney",
    "MinTemp": 12.0,
    "MaxTemp": 22.0,
    "Rainfall": 0.0,
    "Evaporation": 5.0,
    "Sunshine": 8.0,
    "WindGustDir": "W",
    "WindGustSpeed": 30.0,
    "WindDir9am": "W",
    "WindDir3pm": "W",
    "WindSpeed9am": 10.0,
    "WindSpeed3pm": 15.0,
    "Humidity9am": 70.0,
    "Humidity3pm": 55.0,
    "Pressure9am": 1015.0,
    "Pressure3pm": 1012.0,
    "Cloud9am": 4.0,
    "Cloud3pm": 5.0,
    "Temp9am": 15.0,
    "Temp3pm": 20.0,
    "RainToday": "No",
    "TemperatureDifference": 10.0,
    "AverageTemperature": 17.0,
    "HumidityIndex": 62.5,
    "PressureDifference": 3.0,
    "Month": 1,
    "Quarter": 1,
    "DayOfYear": 15,
    "Season": "Summer",
    "IsWeekend": 0,
    "WindIntensityCategory": "Moderate",
    "RainfallIndicator": 0,
    "RainTodayBinary": 0,
}

print("=" * 70)
print("FLASK REST API TEST")
print("=" * 70)

app = create_app("testing")
client = app.test_client()

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}" + (f" - {detail}" if detail and not condition else ""))


# 1. Health check
print("\n[1] Health Check")
resp = client.get("/api/health")
data = resp.get_json()
check("GET /api/health returns 200", resp.status_code == 200, f"got {resp.status_code}")
check("Response has success=True", data.get("success") is True)
check("Model is loaded", data.get("data", {}).get("model_loaded") is True)
check("Model type is XGBClassifier", data.get("data", {}).get("model_type") == "XGBClassifier")

# 2. API documentation
print("\n[2] API Documentation")
resp = client.get("/")
data = resp.get_json()
check("GET / returns 200", resp.status_code == 200)
check("Has API name", "name" in data.get("data", {}))
check("Has 5 endpoints", len(data.get("data", {}).get("endpoints", [])) == 5)

# 3. Model metadata
print("\n[3] Model Metadata")
resp = client.get("/api/v1/model")
data = resp.get_json()
check("GET /api/v1/model returns 200", resp.status_code == 200)
check("Has model_type", "model_type" in data.get("data", {}))
check("Has feature_columns", "feature_columns" in data.get("data", {}))
check("Has 33 features", len(data.get("data", {}).get("feature_columns", [])) == 33)

# 4. Feature schema
print("\n[4] Feature Schema")
resp = client.get("/api/v1/features")
data = resp.get_json()
check("GET /api/v1/features returns 200", resp.status_code == 200)
check("Has feature_columns", "feature_columns" in data.get("data", {}))
check("Has target_column", data.get("data", {}).get("target_column") == "RainTomorrow")

# 5. Single prediction
print("\n[5] Single Prediction")
resp = client.post("/api/v1/predict", json=SAMPLE_INPUT)
data = resp.get_json()
check("POST /api/v1/predict returns 200", resp.status_code == 200, f"got {resp.status_code}")
check("Has prediction", "prediction" in data.get("data", {}))
check("Has prediction_label", "prediction_label" in data.get("data", {}))
check("Has probability", "probability" in data.get("data", {}))
check("Has confidence", "confidence" in data.get("data", {}))
check("Has probabilities", "probabilities" in data.get("data", {}))
check("Prediction is 0 or 1", data.get("data", {}).get("prediction") in (0, 1))
check("Probability is between 0 and 1", 0 <= data.get("data", {}).get("probability", -1) <= 1)

# 6. Batch prediction
print("\n[6] Batch Prediction")
batch_input = [SAMPLE_INPUT, {**SAMPLE_INPUT, "Location": "Melbourne", "Rainfall": 5.0}]
resp = client.post("/api/v1/predict/batch", json=batch_input)
data = resp.get_json()
check("POST /api/v1/predict/batch returns 200", resp.status_code == 200)
check("Has count=2", data.get("data", {}).get("count") == 2)
check("Has 2 results", len(data.get("data", {}).get("results", [])) == 2)

# 7. Invalid input - missing fields
print("\n[7] Invalid Input - Missing Fields")
invalid_input = {"Location": "Sydney", "MinTemp": 12.0}
resp = client.post("/api/v1/predict", json=invalid_input)
data = resp.get_json()
check("Returns 400", resp.status_code == 400)
check("Has error code INVALID_INPUT", data.get("error", {}).get("code") == "INVALID_INPUT")
check("Error message mentions missing fields", "Missing required fields" in data.get("error", {}).get("message", ""))

# 8. Invalid input - empty body
print("\n[8] Invalid Input - Empty Body")
resp = client.post("/api/v1/predict", json={})
data = resp.get_json()
check("Returns 400", resp.status_code == 400)
check("Has error code INVALID_INPUT", data.get("error", {}).get("code") == "INVALID_INPUT")

# 9. Invalid input - wrong content type
print("\n[9] Invalid Input - Wrong Content Type")
resp = client.post("/api/v1/predict", data="not json", content_type="text/plain")
data = resp.get_json()
check("Returns 400", resp.status_code == 400)
check("Has error code INVALID_INPUT", data.get("error", {}).get("code") == "INVALID_INPUT")

# 10. Invalid input - batch not array
print("\n[10] Invalid Input - Batch Not Array")
resp = client.post("/api/v1/predict/batch", json={"not": "array"})
data = resp.get_json()
check("Returns 400", resp.status_code == 400)
check("Has error code INVALID_INPUT", data.get("error", {}).get("code") == "INVALID_INPUT")

# 11. 404 Not Found
print("\n[11] 404 Not Found")
resp = client.get("/api/v1/nonexistent")
data = resp.get_json()
check("Returns 404", resp.status_code == 404)
check("Has error code NOT_FOUND", data.get("error", {}).get("code") == "NOT_FOUND")

# 12. 405 Method Not Allowed
print("\n[12] 405 Method Not Allowed")
resp = client.put("/api/v1/predict")
data = resp.get_json()
check("Returns 405", resp.status_code == 405)
check("Has error code METHOD_NOT_ALLOWED", data.get("error", {}).get("code") == "METHOD_NOT_ALLOWED")

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed > 0:
    sys.exit(1)