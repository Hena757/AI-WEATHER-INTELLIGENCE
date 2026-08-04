"""Test the dashboard syntax and interactive prediction functionality."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 70)
print("DASHBOARD & INTERACTIVE PREDICTION TEST")
print("=" * 70)

# 1. Verify dashboard syntax
print("\n[1] Dashboard Syntax")
try:
    with open(BASE_DIR / "dashboard" / "app.py", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)
    print("  [OK] dashboard/app.py parses successfully")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 2. Test explain_prediction with a sample input
print("\n[2] Interactive Prediction Test")
try:
    from src.explainability import explain_prediction

    # Build a sample input matching the feature columns
    input_data = {
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
    input_df = pd.DataFrame([input_data])

    result = explain_prediction(
        input_df,
        model_path=BASE_DIR / "models" / "best_model.joblib",
        output_dir=BASE_DIR / "reports" / "explanations",
        background_size=50,
        max_display=20,
    )

    print(f"  [OK] Prediction: {result['prediction_label']}")
    print(f"  [OK] Probability: {result['probability']:.4f}")
    print(f"  [OK] Contribution table: {len(result['contribution_table'])} rows")
    print(f"  [OK] Waterfall plot: {result['artifacts']['waterfall_plot'].exists()}")
    print(f"  [OK] Force plot: {result['artifacts']['force_plot'].exists()}")
    print(f"  [OK] Local contributions: {result['artifacts']['local_contributions'].exists()}")

    # Show top 5 contributions
    print("\n  Top 5 feature contributions:")
    top5 = result["contribution_table"].head(5)
    for _, row in top5.iterrows():
        print(f"    {row['feature']}: {row['shap_value']:.4f} ({row['direction']})")

except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)