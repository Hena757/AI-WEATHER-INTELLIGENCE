"""Production-ready Flask REST API for the AI Weather Intelligence platform.

This module provides a RESTful API for the weather prediction model with:

- ``GET  /api/health``          - Health check endpoint
- ``GET  /api/v1/model``        - Model metadata endpoint
- ``POST /api/v1/predict``      - Single prediction endpoint
- ``POST /api/v1/predict/batch`` - Batch prediction endpoint
- ``GET  /api/v1/features``     - Feature schema endpoint
- ``GET  /``                    - API documentation

The API follows REST best practices:
- Structured JSON responses with consistent envelope format
- Proper HTTP status codes
- Request validation with clear error messages
- Comprehensive logging
- Centralised exception handling
- CORS support
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure project root is on the path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.config import config_by_name
from api.services import (
    InvalidInputError,
    ModelNotFoundError,
    ModelServiceError,
    PredictionError,
    WeatherModelService,
)

logger = logging.getLogger(__name__)


def create_app(config_name: str = "default") -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, app.config["LOG_LEVEL"], logging.INFO),
        format=app.config["LOG_FORMAT"],
    )

    # Enable CORS
    CORS(app, origins=app.config["CORS_ORIGINS"])

    # Initialize model service
    model_service = WeatherModelService(
        model_path=app.config["MODEL_PATH"],
        metadata_path=app.config["PREPROCESSING_METADATA_PATH"],
    )

    # Load model at startup
    try:
        model_service.load()
        logger.info("Model loaded successfully at startup")
    except ModelNotFoundError as exc:
        logger.error("Failed to load model at startup: %s", exc)

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    @app.errorhandler(InvalidInputError)
    def handle_invalid_input(exc: InvalidInputError):
        logger.warning("Invalid input: %s", exc)
        return jsonify({
            "success": False,
            "error": {"code": "INVALID_INPUT", "message": str(exc)},
        }), 400

    @app.errorhandler(ModelNotFoundError)
    def handle_model_not_found(exc: ModelNotFoundError):
        logger.error("Model not found: %s", exc)
        return jsonify({
            "success": False,
            "error": {"code": "MODEL_NOT_FOUND", "message": str(exc)},
        }), 503

    @app.errorhandler(PredictionError)
    def handle_prediction_error(exc: PredictionError):
        logger.error("Prediction error: %s", exc)
        return jsonify({
            "success": False,
            "error": {"code": "PREDICTION_ERROR", "message": str(exc)},
        }), 500

    @app.errorhandler(ModelServiceError)
    def handle_model_service_error(exc: ModelServiceError):
        logger.error("Model service error: %s", exc)
        return jsonify({
            "success": False,
            "error": {"code": "MODEL_SERVICE_ERROR", "message": str(exc)},
        }), 500

    @app.errorhandler(404)
    def handle_not_found(exc):
        return jsonify({
            "success": False,
            "error": {"code": "NOT_FOUND", "message": "The requested resource was not found"},
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(exc):
        return jsonify({
            "success": False,
            "error": {"code": "METHOD_NOT_ALLOWED", "message": "The HTTP method is not allowed for this endpoint"},
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(exc):
        logger.exception("Unhandled exception: %s", exc)
        return jsonify({
            "success": False,
            "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
        }), 500

    @app.errorhandler(413)
    def handle_payload_too_large(exc):
        return jsonify({
            "success": False,
            "error": {"code": "PAYLOAD_TOO_LARGE", "message": "Request body exceeds the maximum allowed size"},
        }), 413

    # ------------------------------------------------------------------
    # Request/response helpers
    # ------------------------------------------------------------------
    def success_response(data: Any, status: int = 200) -> Any:
        return jsonify({"success": True, "data": data}), status

    def get_json_body() -> Dict[str, Any]:
        if not request.is_json:
            raise InvalidInputError("Content-Type must be application/json")
        data = request.get_json(silent=True)
        if data is None:
            raise InvalidInputError("Request body must be valid JSON")
        return data

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def api_documentation():
        return success_response({
            "name": app.config["API_TITLE"],
            "version": app.config["API_VERSION"],
            "description": app.config["API_DESCRIPTION"],
            "endpoints": [
                {"method": "GET", "path": "/api/health", "description": "Health check for the API and model"},
                {"method": "GET", "path": "/api/v1/model", "description": "Get model metadata"},
                {"method": "POST", "path": "/api/v1/predict", "description": "Make a single weather prediction"},
                {"method": "POST", "path": "/api/v1/predict/batch", "description": "Make multiple weather predictions"},
                {"method": "GET", "path": "/api/v1/features", "description": "Get the feature schema expected by the model"},
            ],
        })

    @app.route("/api/health", methods=["GET"])
    def health_check():
        model_loaded = model_service.is_loaded
        status = "healthy" if model_loaded else "degraded"
        status_code = 200 if model_loaded else 503
        return success_response({
            "status": status,
            "model_loaded": model_loaded,
            "model_type": model_service.model_type if model_loaded else None,
            "timestamp": time.time(),
        }, status=status_code)

    @app.route("/api/v1/model", methods=["GET"])
    def model_metadata():
        if not model_service.is_loaded:
            raise ModelNotFoundError("Model is not loaded")
        metadata = model_service.get_model_metadata()
        metadata["api_version"] = app.config["API_VERSION"]
        return success_response(metadata)

    @app.route("/api/v1/features", methods=["GET"])
    def feature_schema():
        if not model_service.is_loaded:
            raise ModelNotFoundError("Model is not loaded")
        return success_response({
            "feature_columns": model_service.feature_columns,
            "n_features": len(model_service.feature_columns),
            "target_column": model_service.target_column,
        })

    @app.route("/api/v1/predict", methods=["POST"])
    def predict():
        if not model_service.is_loaded:
            raise ModelNotFoundError("Model is not loaded")
        data = get_json_body()
        result = model_service.predict(data)
        return success_response(result)

    @app.route("/api/v1/predict/batch", methods=["POST"])
    def predict_batch():
        if not model_service.is_loaded:
            raise ModelNotFoundError("Model is not loaded")
        data = get_json_body()
        if not isinstance(data, list):
            raise InvalidInputError("Request body must be an array of weather observations")
        results = model_service.predict_batch(data)
        return success_response({"count": len(results), "results": results})

    return app


# Create the default app instance for WSGI servers
app = create_app(os.getenv("FLASK_ENV", "default"))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)