"""Test the live Flask API endpoints with real HTTP requests."""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

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


def make_request(method: str, path: str, body: dict | list | None = None) -> tuple[int, dict]:
    """Make an HTTP request and return (status_code, json_response)."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


print("=" * 70)
print("LIVE API TEST")
print("=" * 70)

# 1. Health check
print("\n[1] Health Check")
status, data = make_request("GET", "/api/health")
check("GET /api/health returns 200", status == 200, f"got {status}")
check("Status is healthy", data.get("data", {}).get("status") == "healthy")
check("Model is loaded", data.get("data", {}).get("model_loaded") is True)
check("Model type is XGBClassifier", data.get("data", {}).get("model_type") == "XGBClassifier")

# 2. API documentation
print("\n[2] API Documentation")
status, data = make_request("GET", "/")
check("GET / returns 200", status == 200, f"got {status}")
check("Has API name", "name" in data.get("data", {}))
check("Has 5 endpoints", len(data.get("data", {}).get("endpoints", [])) == 5)

# 3. Model metadata
print("\n[3] Model Metadata")
status, data = make_request("GET", "/api/v1/model")
check("GET /api/v1/model returns 200", status == 200, f"got {status}")
check("Has model_type", "model_type" in data.get("data", {}))
check("Has 33 features", len(data.get("data", {}).get("feature_columns", [])) == 33)

# 4. Feature schema
print("\n[4] Feature Schema")
status, data = make_request("GET", "/api/v1/features")
check("GET /api/v1/features returns 200", status == 200, f"got {status}")
check("Has target_column", data.get("data", {}).get("target_column") == "RainTomorrow")

# 5. Single prediction
print("\n[5] Single Prediction")
status, data = make_request("POST", "/api/v1/predict", SAMPLE_INPUT)
check("POST /api/v1/predict returns 200", status == 200, f"got {status}")
result = data.get("data", {})
check("Has prediction", "prediction" in result)
check("Has prediction_label", "prediction_label" in result)
check("Has probability", "probability" in result)
check("Has confidence", "confidence" in result)
check("Has probabilities", "probabilities" in result)
check("Prediction is 0 or 1", result.get("prediction") in (0, 1))
check("Probability is between 0 and 1", 0 <= result.get("probability", -1) <= 1)
print(f"       Prediction: {result.get('prediction_label')} (prob={result.get('probability'):.4f}, conf={result.get('confidence'):.4f})")

# 6. Batch prediction
print("\n[6] Batch Prediction")
batch_input = [SAMPLE_INPUT, {**SAMPLE_INPUT, "Location": "Melbourne", "Rainfall": 5.0}]
status, data = make_request("POST", "/api/v1/predict/batch", batch_input)
check("POST /api/v1/predict/batch returns 200", status == 200, f"got {status}")
check("Has count=2", data.get("data", {}).get("count") == 2)
check("Has 2 results", len(data.get("data", {}).get("results", [])) == 2)

# 7. Invalid input - missing fields
print("\n[7] Invalid Input - Missing Fields")
status, data = make_request("POST", "/api/v1/predict", {"Location": "Sydney", "MinTemp": 12.0})
check("Returns 400", status == 400, f"got {status}")
check("Has error code INVALID_INPUT", data.get("error", {}).get("code") == "INVALID_INPUT")

# 8. Invalid input - empty body
print("\n[8] Invalid Input - Empty Body")
status, data = make_request("POST", "/api/v1/predict", {})
check("Returns 400", status == 400, f"got {status}")
check("Has error code INVALID_INPUT", data.get("error", {}).get("code") == "INVALID_INPUT")

# 9. 404 Not Found
print("\n[9] 404 Not Found")
status, data = make_request("GET", "/api/v1/nonexistent")
check("Returns 404", status == 404, f"got {status}")
check("Has error code NOT_FOUND", data.get("error", {}).get("code") == "NOT_FOUND")

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed > 0:
    sys.exit(1)