"""Model evaluation utilities for the weather prediction pipeline.

This module provides reusable helpers for computing standard classification
metrics, generating diagnostic plots, and persisting evaluation artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute binary classification metrics."""
    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
    else:
        metrics["roc_auc"] = None
    return metrics


def build_confusion_matrix(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, labels: Optional[list[int]] = None) -> np.ndarray:
    """Return the confusion matrix for the predictions."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def build_classification_report(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a classification report as a DataFrame."""
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return pd.DataFrame(report).T


def plot_roc_curve(y_true: pd.Series | np.ndarray, y_proba: np.ndarray, output_path: Path, label: str) -> Path:
    """Save the ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_precision_recall_curve(y_true: pd.Series | np.ndarray, y_proba: np.ndarray, output_path: Path, label: str) -> Path:
    """Save the Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_confusion_matrix(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, output_path: Path, label: str) -> Path:
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.colorbar()
    tick_labels = ["No Rain", "Rain"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_feature_importance(model: Any, feature_names: list[str], output_path: Path, label: str) -> Path:
    """Save a feature importance plot when the model supports it."""
    if not hasattr(model, "feature_importances_"):
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.info("Feature importance is not available for %s", label)
            return output_path
    else:
        importances = model.feature_importances_

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title(f"Feature Importance - {label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_evaluation_artifacts(results_df: pd.DataFrame, best_model: Any, output_dir: str = "models") -> Dict[str, Path]:
    """Persist the evaluation dataframe and best model to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    best_model_path = output_path / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    return {"results": results_path, "best_model": best_model_path}
