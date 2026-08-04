"""Training pipeline for weather prediction models.

This module trains and compares multiple classification models on the
preprocessed weather dataset. It supports feature selection, cross-validation,
hyperparameter tuning, evaluation, and artifact generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import (
    DEFAULT_TARGET_COL,
    build_preprocessing_pipeline,
    clean_dataset,
    load_dataset,
    prepare_feature_matrix,
    save_preprocessing_artifacts,
    split_dataset,
)
from src.evaluate import (
    build_classification_report,
    build_confusion_matrix,
    calculate_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_evaluation_artifacts,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_model_definitions() -> Dict[str, Dict[str, Any]]:
    """Create the model definitions and tuning grids for comparison."""
    definitions: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=3000, random_state=42, solver="liblinear"),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 8, None],
                "min_samples_split": [2, 5],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        },
    }

    if XGBClassifier is not None:
        definitions["xgboost"] = {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=200,
                use_label_encoder=False,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        }

    if LGBMClassifier is not None:
        definitions["lightgbm"] = {
            "estimator": LGBMClassifier(random_state=42, n_estimators=200, verbose=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        }

    if CatBoostClassifier is not None:
        definitions["catboost"] = {
            "estimator": CatBoostClassifier(
                iterations=200,
                depth=4,
                learning_rate=0.1,
                loss_function="Logloss",
                verbose=False,
                random_seed=42,
            ),
            "params": {
                "iterations": [100, 200],
                "depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
        }

    return definitions


def _build_pipeline_for_model(model: Any) -> Pipeline:
    """Create an sklearn Pipeline that combines preprocessing and model fitting."""
    numeric_features = [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Evaporation",
        "Sunshine",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm",
        "TemperatureDifference",
        "AverageTemperature",
        "HumidityIndex",
        "PressureDifference",
        "Month",
        "Quarter",
        "DayOfYear",
        "IsWeekend",
        "RainfallIndicator",
        "RainTodayBinary",
    ]
    categorical_features = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "Season", "WindIntensityCategory", "RainToday"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate_models(
    input_path: str = "data/raw/weatherAUS.csv",
    output_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train, tune, and evaluate multiple classification models."""
    raw_df = load_dataset(input_path)
    cleaned_df = clean_dataset(raw_df, target_col=DEFAULT_TARGET_COL)
    X, y = prepare_feature_matrix(cleaned_df, target_col=DEFAULT_TARGET_COL)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, random_state=random_state)

    model_definitions = _build_model_definitions()
    results_rows: list[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None

    for model_name, config in model_definitions.items():
        logger.info("Training model: %s", model_name)
        estimator = config["estimator"]
        params = config["params"]

        try:
            if model_name in {"catboost"}:
                pipeline = Pipeline([("model", estimator)])
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=[{f"model__{key}": values for key, values in params.items()}],
                    scoring=make_scorer(roc_auc_score),
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                pipeline = _build_pipeline_for_model(estimator)
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=[{f"model__{key}": values for key, values in params.items()}],
                    scoring=make_scorer(roc_auc_score),
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            metrics = calculate_metrics(y_test, y_pred, y_proba)
            confusion = build_confusion_matrix(y_test, y_pred)
            classification_report_df = build_classification_report(y_test, y_pred)

            result = {
                "model": model_name,
                **metrics,
                "cv_score": round(float(grid.best_score_), 4),
                "best_params": grid.best_params_,
                "confusion_matrix": confusion,
                "classification_report": classification_report_df,
                "model_instance": best_model,
            }
            results_rows.append(result)

            if best_result is None or metrics["roc_auc"] is not None and metrics["roc_auc"] > best_result["roc_auc"]:
                best_result = result

            logger.info("Completed %s with ROC-AUC %.4f", model_name, metrics.get("roc_auc", 0))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Model training failed for %s: %s", model_name, exc)

    if not results_rows:
        raise RuntimeError("No models were trained successfully")

    results_df = pd.DataFrame(
        [
            {
                "model": row["model"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "cv_score": row["cv_score"],
                "best_params": row["best_params"],
            }
            for row in results_rows
        ]
    )
    results_df = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(reports_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    for row in results_rows:
        model_name = row["model"]
        model = row["model_instance"]
        plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], report_path / f"roc_curve_{model_name}.png", model_name)
        plot_precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1], report_path / f"pr_curve_{model_name}.png", model_name)
        plot_confusion_matrix(y_test, model.predict(X_test), report_path / f"confusion_matrix_{model_name}.png", model_name)
        plot_feature_importance(model, list(X.columns), report_path / f"feature_importance_{model_name}.png", model_name)

    if best_result is None:
        raise RuntimeError("No best model could be determined")

    artifact_paths = save_evaluation_artifacts(results_df, best_result["model_instance"], output_dir=output_dir)
    joblib.dump(best_result["model_instance"], output_path / "best_model.joblib")

    summary_path = report_path / "model_comparison_summary.csv"
    results_df.to_csv(summary_path, index=False)

    return {
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "results_df": results_df,
        "best_result": best_result,
        "artifact_paths": artifact_paths,
        "report_path": summary_path,
    }
