"""Explainable AI (XAI) utilities for the weather prediction model.

This module integrates SHAP (SHapley Additive exPlanations) into the trained
weather prediction pipeline. It provides reusable functions to:

- Load the trained model pipeline and preprocessing artifacts.
- Compute SHAP values for global and local explanations.
- Generate summary plots, dependence plots, and waterfall plots.
- Save per-prediction contribution tables for dashboard visualisation.
- Explain individual predictions with transparent feature attributions.

The module is designed to work with the sklearn Pipeline saved as
``models/best_model.joblib``, which wraps a preprocessing ``ColumnTransformer``
and an ``XGBClassifier`` (or any other supported classifier).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/best_model.joblib")
DEFAULT_PREPROCESSING_METADATA_PATH = Path("models/preprocessing_metadata.joblib")
DEFAULT_EXPLANATIONS_DIR = Path("reports/explanations")
DEFAULT_BACKGROUND_SIZE = 100
DEFAULT_MAX_DISPLAY = 20


# ---------------------------------------------------------------------------
# Model & data loading helpers
# ---------------------------------------------------------------------------
def load_model_pipeline(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load the trained sklearn Pipeline from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {model_path}")
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "named_steps") or "model" not in pipeline.named_steps:
        raise ValueError("The loaded object is not a valid sklearn Pipeline with a 'model' step")
    logger.info("Loaded model pipeline from %s", model_path)
    return pipeline


def load_preprocessing_metadata(
    metadata_path: str | Path = DEFAULT_PREPROCESSING_METADATA_PATH,
) -> Dict[str, Any]:
    """Load the preprocessing metadata (feature columns, target column)."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Preprocessing metadata not found at {metadata_path}")
    metadata = joblib.load(metadata_path)
    logger.info("Loaded preprocessing metadata from %s", metadata_path)
    return metadata


def get_processed_feature_names(pipeline: Any) -> List[str]:
    """Return the feature names produced by the pipeline's preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - fallback for older sklearn
        logger.warning("Could not retrieve feature names from preprocessor; using generic names")
        return [f"feature_{i}" for i in range(preprocessor.transformers_[0][2].__len__())]


def load_background_data(
    pipeline: Any,
    sample_size: int = DEFAULT_BACKGROUND_SIZE,
    cleaned_data_path: str | Path = "data/processed/cleaned_weather_dataset.csv",
) -> np.ndarray:
    """Load a background sample of processed features for SHAP explainer.

    The background dataset is used by SHAP to estimate the expected model
    output (base value) and to compute feature attributions. We sample a
    subset of the cleaned dataset, drop the target column, and run it
    through the pipeline's preprocessor.
    """
    cleaned_path = Path(cleaned_data_path)
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    metadata = load_preprocessing_metadata()
    feature_columns = metadata.get("feature_columns", [])
    target_col = metadata.get("target_column", "RainTomorrow")

    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        raise ValueError("No feature columns from metadata were found in the cleaned dataset")

    X = df[available_features].copy()
    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    # Sample a representative background set
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"]
    X_processed = preprocessor.transform(X)
    logger.info("Loaded background data with shape %s", X_processed.shape)
    return X_processed


# ---------------------------------------------------------------------------
# SHAP explainer construction
# ---------------------------------------------------------------------------
def build_shap_explainer(
    pipeline: Any,
    background_data: Optional[np.ndarray] = None,
    sample_size: int = DEFAULT_BACKGROUND_SIZE,
) -> shap.Explainer:
    """Build a SHAP explainer for the trained model pipeline.

    For tree-based models (XGBoost, LightGBM, RandomForest, etc.) we use the
    fast ``shap.TreeExplainer``. For other models we fall back to
    ``shap.Explainer`` with a background sample.
    """
    model = pipeline.named_steps["model"]
    model_name = type(model).__name__.lower()

    if any(name in model_name for name in ("xgb", "lgbm", "lightgbm", "randomforest", "gradientboosting", "decisiontree", "catboost")):
        logger.info("Using TreeExplainer for %s", model_name)
        explainer = shap.TreeExplainer(model)
    else:
        if background_data is None:
            background_data = load_background_data(pipeline, sample_size=sample_size)
        logger.info("Using generic Explainer for %s", model_name)
        explainer = shap.Explainer(model, background_data)

    return explainer


def compute_shap_values(
    pipeline: Any,
    X_processed: np.ndarray,
    explainer: Optional[shap.Explainer] = None,
    background_data: Optional[np.ndarray] = None,
) -> shap.Explanation:
    """Compute SHAP values for the provided processed feature matrix."""
    if explainer is None:
        explainer = build_shap_explainer(pipeline, background_data=background_data)

    shap_values = explainer(X_processed)
    logger.info("Computed SHAP values with shape %s", shap_values.shape)
    return shap_values


# ---------------------------------------------------------------------------
# Global explanation visualisations
# ---------------------------------------------------------------------------
def generate_summary_plot(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: str | Path,
    max_display: int = DEFAULT_MAX_DISPLAY,
) -> Path:
    """Generate and save a SHAP summary (beeswarm) plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary plot to %s", output_path)
    return output_path


def generate_dependence_plot(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: str | Path,
    feature_index: int = 0,
    interaction_index: Optional[int] = "auto",
) -> Path:
    """Generate and save a SHAP dependence plot for a given feature."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the raw feature data from the Explanation object
    feature_data = np.asarray(shap_values.data)
    if feature_data.ndim == 3:
        feature_data = feature_data.reshape(feature_data.shape[0], -1)

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_index,
        np.asarray(shap_values.values),
        feature_data,
        feature_names=feature_names,
        interaction_index=interaction_index,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved dependence plot to %s", output_path)
    return output_path


def generate_global_feature_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: str | Path,
    max_display: int = DEFAULT_MAX_DISPLAY,
) -> Path:
    """Generate and save a SHAP bar plot of global feature importance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved global feature importance plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Local explanation visualisations
# ---------------------------------------------------------------------------
def generate_waterfall_plot(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: str | Path,
    instance_index: int = 0,
    max_display: int = DEFAULT_MAX_DISPLAY,
) -> Path:
    """Generate and save a SHAP waterfall plot for a single prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap_values[instance_index],
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved waterfall plot to %s", output_path)
    return output_path


def generate_force_plot(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: str | Path,
    instance_index: int = 0,
) -> Path:
    """Generate and save a SHAP force plot for a single prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    shap.force_plot(
        shap_values.base_values[instance_index],
        shap_values.values[instance_index],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved force plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Contribution tables
# ---------------------------------------------------------------------------
def build_contribution_table(
    shap_values: shap.Explanation,
    feature_names: List[str],
    instance_index: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame of per-feature SHAP contributions for one instance."""
    values = np.asarray(shap_values.values[instance_index]).flatten()
    base_value = float(np.asarray(shap_values.base_values[instance_index]).flatten()[0])

    contribution_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": values,
            "abs_shap_value": np.abs(values),
        }
    )
    contribution_df = contribution_df.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
    contribution_df["direction"] = np.where(contribution_df["shap_value"] >= 0, "Increases rain probability", "Decreases rain probability")
    contribution_df["base_value"] = base_value
    contribution_df["prediction"] = float(np.asarray(shap_values.data[instance_index]).flatten()[0]) if hasattr(shap_values, "data") else None
    return contribution_df


def save_contributions_table(
    shap_values: shap.Explanation,
    feature_names: List[str],
    output_path: str | Path,
    max_instances: int = 50,
) -> Path:
    """Save a long-format CSV of SHAP contributions for multiple instances."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_instances = min(max_instances, shap_values.shape[0])
    rows: List[Dict[str, Any]] = []
    for i in range(n_instances):
        base_value = float(np.asarray(shap_values.base_values[i]).flatten()[0])
        values = np.asarray(shap_values.values[i]).flatten()
        for feature, value in zip(feature_names, values):
            rows.append(
                {
                    "instance_id": i,
                    "feature": feature,
                    "shap_value": float(value),
                    "abs_shap_value": float(abs(value)),
                    "base_value": base_value,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved contributions table to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------
def generate_global_explanations(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_dir: str | Path = DEFAULT_EXPLANATIONS_DIR,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    max_display: int = DEFAULT_MAX_DISPLAY,
    cleaned_data_path: str | Path = "data/processed/cleaned_weather_dataset.csv",
) -> Dict[str, Path]:
    """Generate all global SHAP explanation artifacts.

    Produces:
    - ``shap_summary_plot.png``        : beeswarm summary plot
    - ``shap_global_importance.png``   : bar plot of mean |SHAP| importance
    - ``shap_dependence_plot.png``     : dependence plot for the top feature
    - ``shap_contributions.csv``       : per-instance contribution table
    """
    pipeline = load_model_pipeline(model_path)
    feature_names = get_processed_feature_names(pipeline)
    background_data = load_background_data(pipeline, sample_size=background_size, cleaned_data_path=cleaned_data_path)

    explainer = build_shap_explainer(pipeline, background_data=background_data)
    shap_values = compute_shap_values(pipeline, background_data, explainer=explainer)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {
        "summary_plot": generate_summary_plot(
            shap_values, feature_names, output_path / "shap_summary_plot.png", max_display=max_display
        ),
        "global_importance": generate_global_feature_importance(
            shap_values, feature_names, output_path / "shap_global_importance.png", max_display=max_display
        ),
        "dependence_plot": generate_dependence_plot(
            shap_values, feature_names, output_path / "shap_dependence_plot.png", feature_index=0
        ),
        "contributions": save_contributions_table(
            shap_values, feature_names, output_path / "shap_contributions.csv"
        ),
    }

    logger.info("Global explanations saved to %s", output_path)
    return artifacts


def explain_prediction(
    input_df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_dir: str | Path = DEFAULT_EXPLANATIONS_DIR,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    max_display: int = DEFAULT_MAX_DISPLAY,
) -> Dict[str, Any]:
    """Explain a single prediction with SHAP.

    Parameters
    ----------
    input_df : pd.DataFrame
        A single-row DataFrame with the raw feature columns expected by the
        preprocessing pipeline (e.g. Location, MinTemp, MaxTemp, ...).
    model_path : str | Path
        Path to the saved sklearn Pipeline.
    output_dir : str | Path
        Directory where explanation artifacts will be saved.
    background_size : int
        Number of background samples used to build the explainer.
    max_display : int
        Maximum number of features to display in plots.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the prediction, probability, SHAP values,
        contribution table, and paths to generated plots.
    """
    pipeline = load_model_pipeline(model_path)
    feature_names = get_processed_feature_names(pipeline)
    metadata = load_preprocessing_metadata()
    feature_columns = metadata.get("feature_columns", [])

    # Ensure the input has all required feature columns
    missing = [col for col in feature_columns if col not in input_df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    X_input = input_df[feature_columns].copy()
    preprocessor = pipeline.named_steps["preprocessor"]
    X_processed = preprocessor.transform(X_input)

    # Make prediction
    model = pipeline.named_steps["model"]
    prediction = int(model.predict(X_processed)[0])
    probability = float(model.predict_proba(X_processed)[0, 1])

    # Build explainer and compute SHAP values
    background_data = load_background_data(pipeline, sample_size=background_size)
    explainer = build_shap_explainer(pipeline, background_data=background_data)
    shap_values = compute_shap_values(pipeline, X_processed, explainer=explainer)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate local explanation artifacts
    waterfall_path = generate_waterfall_plot(
        shap_values, feature_names, output_path / "shap_waterfall_plot.png", instance_index=0, max_display=max_display
    )
    force_path = generate_force_plot(
        shap_values, feature_names, output_path / "shap_force_plot.png", instance_index=0
    )
    contribution_df = build_contribution_table(shap_values, feature_names, instance_index=0)
    contribution_path = output_path / "shap_local_contributions.csv"
    contribution_df.to_csv(contribution_path, index=False)

    return {
        "prediction": prediction,
        "probability": probability,
        "prediction_label": "Rain" if prediction == 1 else "No Rain",
        "shap_values": shap_values,
        "contribution_table": contribution_df,
        "artifacts": {
            "waterfall_plot": waterfall_path,
            "force_plot": force_path,
            "local_contributions": contribution_path,
        },
    }


def generate_all_explanations(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_dir: str | Path = DEFAULT_EXPLANATIONS_DIR,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    max_display: int = DEFAULT_MAX_DISPLAY,
    cleaned_data_path: str | Path = "data/processed/cleaned_weather_dataset.csv",
) -> Dict[str, Any]:
    """Generate both global and local explanation artifacts in one call."""
    global_artifacts = generate_global_explanations(
        model_path=model_path,
        output_dir=output_dir,
        background_size=background_size,
        max_display=max_display,
        cleaned_data_path=cleaned_data_path,
    )

    # Use the first background instance as a representative local example
    pipeline = load_model_pipeline(model_path)
    feature_names = get_processed_feature_names(pipeline)
    background_data = load_background_data(pipeline, sample_size=background_size, cleaned_data_path=cleaned_data_path)
    explainer = build_shap_explainer(pipeline, background_data=background_data)
    shap_values = compute_shap_values(pipeline, background_data[:1], explainer=explainer)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    local_artifacts = {
        "waterfall_plot": generate_waterfall_plot(
            shap_values, feature_names, output_path / "shap_waterfall_plot.png", instance_index=0, max_display=max_display
        ),
        "force_plot": generate_force_plot(
            shap_values, feature_names, output_path / "shap_force_plot.png", instance_index=0
        ),
    }

    return {
        "global": global_artifacts,
        "local": local_artifacts,
        "output_dir": output_path,
    }