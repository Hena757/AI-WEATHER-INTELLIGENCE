"""OpenWeather API integration service for the AI Weather Intelligence platform.

This module provides a modular, production-ready integration with the
OpenWeather API. It handles:

- API key management (via environment variable or config file)
- Real-time weather data fetching for cities
- 5-day / 3-hour forecast data
- Automatic preprocessing of live data into model-ready features
- Caching to reduce API calls
- Comprehensive error handling
- Fallback to Open-Meteo API when OpenWeather is unavailable
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# OpenWeather API endpoints
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_GEO = "https://api.openweathermap.org/geo/1.0"

# Open-Meteo fallback API
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# Default city coordinates (used as fallback if geocoding fails)
CITY_COORDS = {
    "Sydney": {"lat": -33.8688, "lon": 151.2093},
    "Melbourne": {"lat": -37.8136, "lon": 144.9631},
    "Brisbane": {"lat": -27.4698, "lon": 153.0251},
    "Perth": {"lat": -31.9505, "lon": 115.8605},
    "Adelaide": {"lat": -34.9285, "lon": 138.6007},
    "Canberra": {"lat": -35.2809, "lon": 149.1300},
    "Hobart": {"lat": -42.8821, "lon": 147.3272},
    "Darwin": {"lat": -12.4634, "lon": 130.8456},
    "Gold Coast": {"lat": -28.0167, "lon": 153.4000},
    "Newcastle": {"lat": -32.9283, "lon": 151.7817},
    "Wollongong": {"lat": -34.4240, "lon": 150.8931},
    "Cairns": {"lat": -16.9186, "lon": 145.7781},
    "Townsville": {"lat": -19.2589, "lon": 146.8169},
    "Alice Springs": {"lat": -23.6980, "lon": 133.8807},
    "Launceston": {"lat": -41.4388, "lon": 147.1347},
}

# Cache settings
CACHE_TTL_SECONDS = 300  # 5 minutes
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "weather_cache.json"

# Wind direction mapping
WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

# Season mapping
SEASONS = ["Summer", "Autumn", "Winter", "Spring"]


class WeatherServiceError(Exception):
    """Base exception for weather service errors."""


class APIKeyError(WeatherServiceError):
    """Raised when the OpenWeather API key is missing or invalid."""


class WeatherFetchError(WeatherServiceError):
    """Raised when weather data cannot be fetched."""


class WeatherDataError(WeatherServiceError):
    """Raised when weather data is invalid or incomplete."""


class OpenWeatherService:
    """Service for fetching and preprocessing weather data from OpenWeather API.

    The service manages API keys, caches responses, and provides both
    OpenWeather and Open-Meteo fallback data sources.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = CACHE_TTL_SECONDS,
        cache_dir: str | Path = CACHE_DIR,
    ) -> None:
        """Initialize the weather service.

        Parameters
        ----------
        api_key : str | None
            OpenWeather API key. If None, reads from OPENWEATHER_API_KEY env var.
        cache_ttl : int
            Cache time-to-live in seconds.
        cache_dir : str | Path
            Directory for the cache file.
        """
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY", "")
        self.cache_ttl = cache_ttl
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "weather_cache.json"
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    # ------------------------------------------------------------------
    # API Key Management
    # ------------------------------------------------------------------
    @property
    def has_api_key(self) -> bool:
        """Return True if an OpenWeather API key is configured."""
        return bool(self.api_key) and self.api_key != "your_openweather_api_key_here"

    def set_api_key(self, api_key: str) -> None:
        """Set the OpenWeather API key at runtime."""
        self.api_key = api_key.strip()
        logger.info("OpenWeather API key updated")

    def save_api_key(self, api_key: str, config_path: str | Path = ".env") -> None:
        """Persist the API key to a .env file for future sessions."""
        config_path = Path(config_path)
        lines = []
        if config_path.exists():
            lines = config_path.read_text(encoding="utf-8").splitlines()

        # Remove existing OPENWEATHER_API_KEY line
        lines = [line for line in lines if not line.startswith("OPENWEATHER_API_KEY=")]

        # Add new key
        lines.append(f"OPENWEATHER_API_KEY={api_key}")

        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("OpenWeather API key saved to %s", config_path)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load the weather cache from disk."""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load weather cache, starting fresh")
            return {}

    def _save_cache(self) -> None:
        """Persist the weather cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save weather cache: %s", exc)

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached response if it's still fresh."""
        entry = self._cache.get(key)
        if not entry:
            return None
        timestamp = entry.get("timestamp", 0)
        if time.time() - timestamp > self.cache_ttl:
            return None
        return entry.get("data")

    def _set_cached(self, key: str, data: Dict[str, Any]) -> None:
        """Cache a response with a timestamp."""
        self._cache[key] = {
            "timestamp": time.time(),
            "data": data,
        }
        self._save_cache()

    def clear_cache(self) -> None:
        """Clear the weather cache."""
        self._cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Weather cache cleared")

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------
    def geocode_city(self, city: str) -> Tuple[float, float]:
        """Get latitude and longitude for a city.

        Uses OpenWeather geocoding API if key is available, otherwise
        falls back to the built-in city coordinates.
        """
        # Check built-in coordinates first
        if city in CITY_COORDS:
            coords = CITY_COORDS[city]
            return coords["lat"], coords["lon"]

        # Try OpenWeather geocoding
        if self.has_api_key:
            try:
                resp = requests.get(
                    f"{OPENWEATHER_GEO}/direct",
                    params={"q": city, "limit": 1, "appid": self.api_key},
                    timeout=10,
                )
                resp.raise_for_status()
                results = resp.json()
                if results:
                    return results[0]["lat"], results[0]["lon"]
            except Exception as exc:
                logger.warning("Geocoding failed for %s: %s", city, exc)

        raise WeatherDataError(f"Could not find coordinates for city: {city}")

    # ------------------------------------------------------------------
    # OpenWeather API
    # ------------------------------------------------------------------
    def fetch_current_weather(self, city: str) -> Dict[str, Any]:
        """Fetch current weather data from OpenWeather API.

        Returns a normalized dictionary with weather metrics.
        """
        cache_key = f"current_{city.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug("Using cached weather for %s", city)
            return cached

        if not self.has_api_key:
            raise APIKeyError(
                "OpenWeather API key not configured. Set OPENWEATHER_API_KEY "
                "environment variable or provide it in the sidebar."
            )

        lat, lon = self.geocode_city(city)

        try:
            resp = requests.get(
                f"{OPENWEATHER_BASE}/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            normalized = self._normalize_current_weather(data, city)
            self._set_cached(cache_key, normalized)
            return normalized

        except requests.exceptions.RequestException as exc:
            raise WeatherFetchError(f"Failed to fetch weather for {city}: {exc}") from exc

    def fetch_forecast(self, city: str) -> Dict[str, Any]:
        """Fetch 5-day / 3-hour forecast from OpenWeather API."""
        cache_key = f"forecast_{city.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        if not self.has_api_key:
            raise APIKeyError("OpenWeather API key not configured")

        lat, lon = self.geocode_city(city)

        try:
            resp = requests.get(
                f"{OPENWEATHER_BASE}/forecast",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            normalized = self._normalize_forecast(data, city)
            self._set_cached(cache_key, normalized)
            return normalized

        except requests.exceptions.RequestException as exc:
            raise WeatherFetchError(f"Failed to fetch forecast for {city}: {exc}") from exc

    def _normalize_current_weather(self, data: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Normalize OpenWeather current weather response."""
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        weather = data.get("weather", [{}])[0] if data.get("weather") else {}

        return {
            "source": "openweather",
            "city": city,
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "cloud_cover": clouds.get("all"),
            "description": weather.get("description", ""),
            "icon": weather.get("icon", ""),
            "timestamp": data.get("dt"),
            "sunrise": data.get("sys", {}).get("sunrise"),
            "sunset": data.get("sys", {}).get("sunset"),
        }

    def _normalize_forecast(self, data: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Normalize OpenWeather forecast response."""
        forecast_list = data.get("list", [])
        daily: Dict[str, Dict[str, Any]] = {}

        for item in forecast_list:
            dt = datetime.fromtimestamp(item.get("dt", 0))
            date_key = dt.strftime("%Y-%m-%d")

            if date_key not in daily:
                daily[date_key] = {
                    "date": date_key,
                    "temp_min": item["main"].get("temp_min", 0),
                    "temp_max": item["main"].get("temp_max", 0),
                    "humidity": item["main"].get("humidity", 0),
                    "pressure": item["main"].get("pressure", 0),
                    "wind_speed": item["wind"].get("speed", 0),
                    "cloud_cover": item["clouds"].get("all", 0),
                    "description": item["weather"][0].get("description", "") if item.get("weather") else "",
                    "precipitation": item.get("rain", {}).get("3h", 0) if "rain" in item else 0,
                }
            else:
                daily[date_key]["temp_min"] = min(daily[date_key]["temp_min"], item["main"].get("temp_min", 0))
                daily[date_key]["temp_max"] = max(daily[date_key]["temp_max"], item["main"].get("temp_max", 0))
                daily[date_key]["precipitation"] += item.get("rain", {}).get("3h", 0) if "rain" in item else 0

        return {
            "source": "openweather",
            "city": city,
            "daily": list(daily.values())[:5],
        }

    # ------------------------------------------------------------------
    # Open-Meteo Fallback
    # ------------------------------------------------------------------
    def fetch_open_meteo(self, city: str) -> Dict[str, Any]:
        """Fetch weather data from Open-Meteo API as fallback."""
        cache_key = f"openmeteo_{city.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        coords = CITY_COORDS.get(city)
        if not coords:
            raise WeatherDataError(f"Could not find coordinates for city: {city}")

        try:
            resp = requests.get(
                OPEN_METEO_BASE,
                params={
                    "latitude": coords["lat"],
                    "longitude": coords["lon"],
                    "current": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,cloud_cover",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "Australia/Sydney",
                    "forecast_days": 7,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            current = data.get("current", {})
            daily = data.get("daily", {})

            normalized = {
                "source": "openmeteo",
                "city": city,
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "pressure": current.get("pressure_msl"),
                "wind_speed": current.get("wind_speed_10m"),
                "wind_direction": current.get("wind_direction_10m"),
                "cloud_cover": current.get("cloud_cover"),
                "precipitation": current.get("precipitation", 0),
                "temp_min": daily.get("temperature_2m_min", [None])[0],
                "temp_max": daily.get("temperature_2m_max", [None])[0],
                "daily": [
                    {
                        "date": d,
                        "temp_min": tmin,
                        "temp_max": tmax,
                        "precipitation": precip,
                    }
                    for d, tmin, tmax, precip in zip(
                        daily.get("time", [])[:7],
                        daily.get("temperature_2m_min", [])[:7],
                        daily.get("temperature_2m_max", [])[:7],
                        daily.get("precipitation_sum", [])[:7],
                    )
                ],
            }
            self._set_cached(cache_key, normalized)
            return normalized

        except requests.exceptions.RequestException as exc:
            raise WeatherFetchError(f"Failed to fetch Open-Meteo data for {city}: {exc}") from exc

    # ------------------------------------------------------------------
    # Unified weather fetching
    # ------------------------------------------------------------------
    def get_weather(self, city: str) -> Dict[str, Any]:
        """Fetch weather data, trying OpenWeather first, then Open-Meteo.

        Returns a unified weather dictionary with all metrics needed for
        the prediction model.
        """
        # Try OpenWeather first
        if self.has_api_key:
            try:
                return self.fetch_current_weather(city)
            except (APIKeyError, WeatherFetchError) as exc:
                logger.warning("OpenWeather failed for %s, falling back to Open-Meteo: %s", city, exc)

        # Fallback to Open-Meteo
        try:
            return self.fetch_open_meteo(city)
        except WeatherFetchError as exc:
            raise WeatherFetchError(f"All weather sources failed for {city}: {exc}") from exc

    def get_weather_with_forecast(self, city: str) -> Dict[str, Any]:
        """Fetch current weather and forecast together."""
        weather = self.get_weather(city)

        # Try to get forecast
        try:
            if self.has_api_key:
                forecast = self.fetch_forecast(city)
                weather["daily"] = forecast.get("daily", [])
            else:
                meteo = self.fetch_open_meteo(city)
                weather["daily"] = meteo.get("daily", [])
        except Exception as exc:
            logger.warning("Failed to fetch forecast for %s: %s", city, exc)
            weather["daily"] = []

        return weather

    # ------------------------------------------------------------------
    # Preprocessing for model input
    # ------------------------------------------------------------------
    def build_model_input(
        self,
        weather: Dict[str, Any],
        rain_today: str = "No",
        wind_gust_dir: str = "W",
        wind_dir9am: str = "W",
        wind_dir3pm: str = "W",
    ) -> pd.DataFrame:
        """Convert live weather data into model-ready feature DataFrame.

        Parameters
        ----------
        weather : Dict[str, Any]
            Normalized weather data from get_weather().
        rain_today : str
            Whether it rained today ("Yes" or "No").
        wind_gust_dir : str
            Wind gust direction.
        wind_dir9am : str
            Wind direction at 9am.
        wind_dir3pm : str
            Wind direction at 3pm.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with all 33 model features.
        """
        city = weather.get("city", "Sydney")
        temp = weather.get("temperature", 20.0)
        temp_min = weather.get("temp_min", temp - 5)
        temp_max = weather.get("temp_max", temp + 5)
        humidity = weather.get("humidity", 60)
        pressure = weather.get("pressure", 1013)
        wind_speed = weather.get("wind_speed", 15)
        cloud = weather.get("cloud_cover", 30)
        precipitation = weather.get("precipitation", 0)

        # Convert wind direction degrees to compass direction
        wind_deg = weather.get("wind_direction")
        if wind_deg is not None:
            wind_dir = WIND_DIRS[int((wind_deg % 360) / 22.5) % 16]
        else:
            wind_dir = wind_gust_dir

        now = datetime.now()
        month = now.month
        quarter = (month - 1) // 3 + 1
        season = SEASONS[(month - 1) // 3]

        # Estimate 9am/3pm values from current conditions
        humidity9am = min(100, humidity + 10)
        humidity3pm = max(0, humidity - 5)
        pressure9am = pressure + 1.5
        pressure3pm = pressure - 1.5
        temp9am = temp - 3
        temp3pm = temp + 2
        wind_speed9am = max(0, wind_speed - 5)
        wind_gust_speed = wind_speed * 1.6

        input_data = {
            "Location": city,
            "MinTemp": temp_min,
            "MaxTemp": temp_max,
            "Rainfall": precipitation,
            "Evaporation": 5.0,
            "Sunshine": 8.0,
            "WindGustDir": wind_gust_dir,
            "WindGustSpeed": wind_gust_speed,
            "WindDir9am": wind_dir9am,
            "WindDir3pm": wind_dir3pm,
            "WindSpeed9am": wind_speed9am,
            "WindSpeed3pm": wind_speed,
            "Humidity9am": humidity9am,
            "Humidity3pm": humidity3pm,
            "Pressure9am": pressure9am,
            "Pressure3pm": pressure3pm,
            "Cloud9am": min(8, max(0, cloud / 12.5)),
            "Cloud3pm": min(8, max(0, cloud / 12.5)),
            "Temp9am": temp9am,
            "Temp3pm": temp3pm,
            "RainToday": rain_today,
            "TemperatureDifference": temp_max - temp_min,
            "AverageTemperature": (temp_min + temp_max) / 2.0,
            "HumidityIndex": (humidity9am + humidity3pm) / 2.0,
            "PressureDifference": pressure9am - pressure3pm,
            "Month": month,
            "Quarter": quarter,
            "DayOfYear": now.timetuple().tm_yday,
            "Season": season,
            "IsWeekend": 1 if now.weekday() >= 5 else 0,
            "WindIntensityCategory": "Low" if wind_speed < 10 else ("Moderate" if wind_speed < 25 else ("High" if wind_speed < 40 else "VeryHigh")),
            "RainfallIndicator": 1 if precipitation > 0 else 0,
            "RainTodayBinary": 1 if rain_today == "Yes" else 0,
        }
        return pd.DataFrame([input_data])

    def get_city_weather_data(self, city: str) -> Dict[str, Any]:
        """Get complete weather data for a city with forecast.

        This is the main entry point for the dashboard.
        """
        return self.get_weather_with_forecast(city)


# Singleton instance for the dashboard
_weather_service: Optional[OpenWeatherService] = None


def get_weather_service() -> OpenWeatherService:
    """Get or create the singleton weather service instance."""
    global _weather_service
    if _weather_service is None:
        _weather_service = OpenWeatherService()
    return _weather_service