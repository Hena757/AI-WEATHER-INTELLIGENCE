"""Configuration settings for the weather prediction REST API."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration for the Flask API."""

    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.joblib"))
    PREPROCESSING_METADATA_PATH = os.getenv(
        "PREPROCESSING_METADATA_PATH",
        str(BASE_DIR / "models" / "preprocessing_metadata.joblib"),
    )
    CLEANED_DATA_PATH = os.getenv(
        "CLEANED_DATA_PATH",
        str(BASE_DIR / "data" / "processed" / "cleaned_weather_dataset.csv"),
    )

    # API settings
    API_TITLE = "AI Weather Intelligence API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = (
        "Production-ready REST API for the weather prediction model. "
        "Predicts whether it will rain tomorrow in Australia using an "
        "XGBoost classifier with SHAP explainability."
    )

    # Request validation
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB max request body

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # CORS (if needed)
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}