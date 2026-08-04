"""Model service layer for the weather prediction REST API.

This module encapsulates all model-related operations: loading the trained
pipeline, preprocessing input data, making predictions, and computing
confidence scores. It is designed to be framework-agnostic so it can be
reused by the Flask API, CLI tools, or other consumers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Base exception for model service errors."""


class ModelNotFoundError(ModelServiceError):
    """Raised when the model pipeline cannot be loaded."""


class InvalidInputError(ModelServiceError):
    """Raised when the input data is invalid or missing required columns."""


class PredictionError(ModelServiceError):
    """Raised when prediction fails."""


class WeatherModelService:
    """Service for loading the model and making predictions.

    The service loads the trained sklearn Pipeline (which includes a
    preprocessing ColumnTransformer and an XGBoost classifier) once at
    startup and reuses it for all requests. This avoids the overhead of
    reloading the model on every prediction.
    """

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._pipeline: Optional[Any] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._feature_columns: Optional[List[str]] = None
        self._target_column: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> "WeatherModelService":
        """Load the model pipeline and preprocessing metadata."""
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model pipeline not found at {self.model_path}")

        if not self.metadata_path.exists():
            raise ModelNotFoundError(f"Preprocessing metadata not found at {self.metadata_path}")

        try:
            self._pipeline = joblib.load(self.model_path)
            self._metadata = joblib.load(self.metadata_path)
        except Exception as exc:
            raise ModelNotFoundError(f"Failed to load model artifacts: {exc}") from exc

        if not hasattr(self._pipeline, "named_steps") or "model" not in self._pipeline.named_steps:
            raise ModelNotFoundError("Loaded object is not a valid sklearn Pipeline with a 'model' step")

        self._feature_columns = self._metadata.get("feature_columns", [])
        self._target_column = self._metadata.get("target_column", "RainTomorrow")

        logger.info(
            "Model service initialized: %s features, target=%s",
            len(self._feature_columns),
            self._target_column,
        )
        return self

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._pipeline is not None

    @property
    def feature_columns(self) -> List[str]:
        """Return the list of feature columns expected by the model."""
        if self._feature_columns is None:
            raise ModelServiceError("Model service has not been loaded")
        return self._feature_columns

    @property
    def target_column(self) -> str:
        """Return the target column name."""
        if self._target_column is None:
            raise ModelServiceError("Model service has not been loaded")
        return self._target_column

    @property
    def model_type(self) -> str:
        """Return the type of the underlying model."""
        if self._pipeline is None:
            raise ModelServiceError("Model service has not been loaded")
        return type(self._pipeline.named_steps["model"]).__name__

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Validate and convert input data to a DataFrame.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of feature values.

        Returns
        -------
        pd.DataFrame
            A single-row DataFrame with the required feature columns.

        Raises
        ------
        InvalidInputError
            If required columns are missing or values are invalid.
        """
        if not isinstance(data, dict):
            raise InvalidInputError("Request body must be a JSON object")

        if not data:
            raise InvalidInputError("Request body cannot be empty")

        feature_columns = self.feature_columns
        missing = [col for col in feature_columns if col not in data]
        if missing:
            raise InvalidInputError(
                f"Missing required fields: {', '.join(missing)}. "
                f"Expected fields: {', '.join(feature_columns)}"
            )

        # Build DataFrame with only the required feature columns
        try:
            df = pd.DataFrame([{col: data[col] for col in feature_columns}])
        except Exception as exc:
            raise InvalidInputError(f"Failed to construct input DataFrame: {exc}") from exc

        # Check for NaN values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df[numeric_cols].isnull().any().any():
            nan_cols = df[numeric_cols].columns[df[numeric_cols].isnull().any()].tolist()
            raise InvalidInputError(f"Invalid numeric values in fields: {', '.join(nan_cols)}")

        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction for a single weather observation.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of feature values.

        Returns
        -------
        Dict[str, Any]
            Prediction result with class, probability, and confidence.
        """
        if self._pipeline is None:
            raise ModelServiceError("Model service has not been loaded")

        df = self.validate_input(data)

        try:
            # Preprocess and predict
            X_processed = self._pipeline.named_steps["preprocessor"].transform(df)
            model = self._pipeline.named_steps["model"]

            prediction = int(model.predict(X_processed)[0])
            probabilities = model.predict_proba(X_processed)[0]

            # Determine class labels
            classes = model.classes_
            proba_dict = {
                str(int(cls)): float(prob) for cls, prob in zip(classes, probabilities)
            }

            # Confidence is the probability of the predicted class
            confidence = float(probabilities[np.where(classes == prediction)[0][0]])

            return {
                "prediction": prediction,
                "prediction_label": "Rain" if prediction == 1 else "No Rain",
                "probability": float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]),
                "confidence": confidence,
                "probabilities": proba_dict,
            }
        except InvalidInputError:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            raise PredictionError(f"Prediction failed: {exc}") from exc

    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple weather observations."""
        if not isinstance(data_list, list) or not data_list:
            raise InvalidInputError("Request body must be a non-empty array of weather observations")

        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.predict(data)
                result["instance_id"] = i
                results.append(result)
            except InvalidInputError as exc:
                results.append({"instance_id": i, "error": str(exc)})

        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_model_metadata(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        if self._pipeline is None:
            raise ModelServiceError("Model service has not been loaded")

        model = self._pipeline.named_steps["model"]
        preprocessor = self._pipeline.named_steps["preprocessor"]

        return {
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "target_column": self.target_column,
            "classes": [int(c) for c in model.classes_],
            "preprocessor_transformers": list(preprocessor.named_transformers_.keys()),
            "model_path": str(self.model_path),
        }