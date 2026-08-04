"""Test the professional Streamlit dashboard syntax and functionality."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 70)
print("PROFESSIONAL DASHBOARD TEST")
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

# 2. Verify imports
print("\n[2] Required Imports")
try:
    import plotly.express
    import plotly.graph_objects
    import requests
    import streamlit
    print("  [OK] plotly, requests, streamlit imports available")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 3. Test data loading
print("\n[3] Data Loading")
try:
    import pandas as pd
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_weather_dataset.csv", parse_dates=["Date"])
    print(f"  [OK] Loaded {len(df):,} records with {df.shape[1]} columns")
    print(f"  [OK] {df['Location'].nunique()} unique locations")
    print(f"  [OK] Date range: {df['Date'].min()} to {df['Date'].max()}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 4. Test build_input_data function
print("\n[4] build_input_data Function")
try:
    from datetime import datetime
    from dashboard.app import build_input_data, WIND_DIRS, SEASONS

    input_df = build_input_data(
        location="Sydney",
        min_temp=12.0,
        max_temp=22.0,
        rainfall=0.0,
        humidity9am=70.0,
        humidity3pm=55.0,
        pressure9am=1015.0,
        pressure3pm=1012.0,
        wind_speed=15.0,
        wind_gust_speed=30.0,
        cloud3pm=5.0,
        temp3pm=20.0,
    )
    print(f"  [OK] Input DataFrame shape: {input_df.shape}")
    print(f"  [OK] Columns: {len(input_df.columns)}")
    print(f"  [OK] Location: {input_df['Location'].iloc[0]}")
    print(f"  [OK] TemperatureDifference: {input_df['TemperatureDifference'].iloc[0]}")
    print(f"  [OK] Season: {input_df['Season'].iloc[0]}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 5. Test prediction with SHAP
print("\n[5] Prediction with SHAP")
try:
    from src.explainability import explain_prediction

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
    print(f"  [OK] Waterfall plot exists: {result['artifacts']['waterfall_plot'].exists()}")
    print(f"  [OK] Force plot exists: {result['artifacts']['force_plot'].exists()}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 6. Test live weather API
print("\n[6] Live Weather API (Open-Meteo)")
try:
    from dashboard.app import fetch_live_weather

    weather = fetch_live_weather("Sydney")
    if weather:
        current = weather.get("current", {})
        print(f"  [OK] Live weather fetched for Sydney")
        print(f"  [OK] Temperature: {current.get('temperature_2m')}°C")
        print(f"  [OK] Humidity: {current.get('relative_humidity_2m')}%")
        print(f"  [OK] Pressure: {current.get('pressure_msl')}hPa")
    else:
        print("  [WARN] Could not fetch live weather (network may be unavailable)")
except Exception as exc:
    print(f"  [WARN] Live weather test: {exc}")

# 7. Verify SHAP artifacts exist
print("\n[7] SHAP Artifacts")
reports_dir = BASE_DIR / "reports" / "explanations"
artifacts = [
    "shap_summary_plot.png",
    "shap_global_importance.png",
    "shap_dependence_plot.png",
    "shap_waterfall_plot.png",
    "shap_force_plot.png",
    "shap_contributions.csv",
    "shap_local_contributions.csv",
]
all_exist = True
for artifact in artifacts:
    path = reports_dir / artifact
    exists = path.exists()
    if not exists:
        all_exist = False
    print(f"  [{'OK' if exists else 'MISSING'}] {artifact}")

# 8. Verify model comparison data
print("\n[8] Model Comparison Data")
comparison_path = BASE_DIR / "models" / "model_comparison_results.csv"
if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)
    print(f"  [OK] Loaded {len(comparison_df)} model comparisons")
    print(f"  [OK] Models: {', '.join(comparison_df['model'].tolist())}")
else:
    print("  [MISSING] model_comparison_results.csv")

print("\n" + "=" * 70)
print("ALL CHECKS COMPLETED")
print("=" * 70)