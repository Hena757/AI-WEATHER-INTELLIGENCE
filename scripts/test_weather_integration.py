"""Test the OpenWeather API integration and dashboard functionality."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 70)
print("OPENWEATHER INTEGRATION TEST")
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

# 2. Verify weather_service module
print("\n[2] Weather Service Module")
try:
    from src.weather_service import (
        OpenWeatherService,
        WeatherServiceError,
        APIKeyError,
        WeatherFetchError,
        get_weather_service,
        CITY_COORDS,
    )
    print("  [OK] weather_service imports successful")
    print(f"  [OK] {len(CITY_COORDS)} cities in CITY_COORDS")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 3. Test API key management
print("\n[3] API Key Management")
service = OpenWeatherService()
has_key = service.has_api_key
print(f"  [OK] has_api_key = {has_key}")
print(f"  [OK] API key {'configured' if has_key else 'not configured - using fallback'}")

# 4. Test weather fetching (Open-Meteo fallback)
print("\n[4] Weather Fetching (Open-Meteo fallback)")
try:
    weather = service.fetch_open_meteo("Sydney")
    print(f"  [OK] Weather fetched via Open-Meteo")
    print(f"  [OK] Temperature: {weather.get('temperature')}°C")
    print(f"  [OK] Humidity: {weather.get('humidity')}%")
    print(f"  [OK] Pressure: {weather.get('pressure')}hPa")
    print(f"  [OK] Wind Speed: {weather.get('wind_speed')} km/h")
    print(f"  [OK] Cloud: {weather.get('cloud_cover')}%")
    print(f"  [OK] Forecast days: {len(weather.get('daily', []))}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 5. Test get_weather (unified)
print("\n[5] Unified Weather Fetching")
try:
    weather = service.get_weather_with_forecast("Melbourne")
    print(f"  [OK] Source: {weather.get('source')}")
    print(f"  [OK] City: {weather.get('city')}")
    print(f"  [OK] Temp: {weather.get('temperature')}°C")
    print(f"  [OK] Humidity: {weather.get('humidity')}%")
    print(f"  [OK] Forecast entries: {len(weather.get('daily', []))}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 6. Test build_model_input
print("\n[6] build_model_input Preprocessing")
try:
    input_df = service.build_model_input(weather)
    print(f"  [OK] Input shape: {input_df.shape}")
    print(f"  [OK] Columns: {len(input_df.columns)}")
    print(f"  [OK] Location: {input_df['Location'].iloc[0]}")
    print(f"  [OK] MinTemp: {input_df['MinTemp'].iloc[0]:.2f}")
    print(f"  [OK] MaxTemp: {input_df['MaxTemp'].iloc[0]:.2f}")
    print(f"  [OK] HumidityIndex: {input_df['HumidityIndex'].iloc[0]:.2f}")
    print(f"  [OK] PressureDifference: {input_df['PressureDifference'].iloc[0]:.2f}")
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 7. Test prediction with live data
print("\n[7] Prediction with Live Weather Data")
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
except Exception as exc:
    print(f"  [FAIL] {exc}")
    sys.exit(1)

# 8. Test caching
print("\n[8] Caching")
try:
    cache_key = f"openmeteo_sydney"
    cached_1 = service._get_cached(cache_key)
    print(f"  [OK] Cache entry exists: {cached_1 is not None}")
    if cached_1:
        service2 = OpenWeatherService()
        cached_2 = service2._get_cached(cache_key)
        print(f"  [OK] Cache persisted across instances: {cached_2 is not None}")
except Exception as exc:
    print(f"  [FAIL] {exc}")

# 9. Test dashboard fetch_live_weather
print("\n[9] Dashboard fetch_live_weather")
try:
    from dashboard.app import fetch_live_weather, weather_service
    weather = fetch_live_weather("Brisbane")
    if weather:
        print(f"  [OK] fetch_live_weather returned data for Brisbane")
        print(f"  [OK] Source: {weather.get('source')}")
        print(f"  [OK] Temperature: {weather.get('temperature')}°C")
    else:
        print("  [WARN] fetch_live_weather returned None (network issue?)")
except Exception as exc:
    print(f"  [WARN] {exc}")

# 10. Multi-city comparison
print("\n[10] Multi-City Comparison")
try:
    cities = ["Sydney", "Melbourne", "Perth"]
    for city in cities:
        w = service.get_weather(city)
        if w:
            print(f"  [OK] {city}: {w.get('temperature')}°C, {w.get('humidity')}% humidity")
except Exception as exc:
    print(f"  [FAIL] {exc}")

print("\n" + "=" * 70)
print("ALL CHECKS COMPLETED")
print("=" * 70)