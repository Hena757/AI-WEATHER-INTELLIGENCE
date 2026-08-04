"""Data preprocessing utilities for the AI Weather Intelligence platform.

This module provides reusable functions for loading, inspecting, cleaning,
feature engineering, and preparing the weather dataset for downstream model
training. It combines deterministic preprocessing steps with scikit-learn
pipelines to support reproducible experiments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import engineer_weather_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TARGET_COL = "RainTomorrow"


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the weather dataset from disk."""
    try:
        logger.info("Loading dataset from %s", filepath)
        df = pd.read_csv(filepath)
        logger.info("Successfully loaded dataset with shape %s", df.shape)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error loading dataset: %s", exc)
        raise


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic information about the dataset."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "size_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "columns": df.columns.tolist(),
        "index_name": df.index.name,
        "total_cells": df.shape[0] * df.shape[1],
    }


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing-values counts and percentages."""
    missing_counts = df.isnull().sum()
    missing_data = pd.DataFrame(
        {
            "Column": missing_counts.index,
            "Missing_Count": missing_counts.values,
            "Missing_Percentage": (100 * missing_counts / len(df)).round(2),
            "Data_Type": df.dtypes.loc[missing_counts.index].values,
        }
    )
    return missing_data[missing_data["Missing_Count"] > 0].sort_values(
        "Missing_Count", ascending=False
    ).reset_index(drop=True)


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Report duplicate rows present in the dataset."""
    total_duplicates = int(df.duplicated().sum())
    return {
        "total_duplicates": total_duplicates,
        "duplicate_percentage": round(100 * total_duplicates / len(df), 2) if len(df) else 0.0,
        "duplicates_df": df[df.duplicated(keep=False)].sort_values(by=list(df.columns)) if total_duplicates else pd.DataFrame(),
    }


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.describe().T


def analyze_target_variable(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> Dict[str, Any]:
    """Summarize the target variable distribution."""
    if target_col not in df.columns:
        return {}
    value_counts = df[target_col].value_counts(dropna=False)
    percentages = round(100 * df[target_col].value_counts(normalize=True, dropna=False), 2)
    imbalance_ratio = None
    if len(value_counts) == 2:
        imbalance_ratio = max(value_counts) / min(value_counts)
    return {
        "value_counts": value_counts,
        "percentages": percentages,
        "imbalance_ratio": imbalance_ratio,
        "missing_values": int(df[target_col].isnull().sum()),
    }


def get_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """Categorize columns by their data types."""
    return {
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime64"]).columns.tolist(),
    }


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level data quality metrics."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    completeness_score = round(100 * (1 - missing_cells / total_cells), 2) if total_cells else 100.0
    uniqueness_score = round(100 * (1 - duplicate_rows / len(df)), 2) if len(df) else 100.0
    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "total_cells": int(total_cells),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "completeness_score": completeness_score,
        "uniqueness_score": uniqueness_score,
        "quality_score": round((completeness_score + uniqueness_score) / 2, 2),
    }


def clean_dataset(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """Remove duplicates and engineer weather features while preserving the target column."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if "Date" in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")

    for column in [
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
    ]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = engineer_weather_features(cleaned, target_col=target_col)
    cleaned = treat_outliers(cleaned)

    if target_col in cleaned.columns:
        cleaned = cleaned.dropna(subset=[target_col]).reset_index(drop=True)

    return cleaned


def detect_outliers(df: pd.DataFrame, columns: Optional[list[str]] = None, threshold: float = 1.5) -> Dict[str, Dict[str, float]]:
    """Detect outliers for numeric columns using the IQR rule."""
    numeric_columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    bounds: Dict[str, Dict[str, float]] = {}
    for column in numeric_columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        bounds[column] = {"lower_bound": float(lower_bound), "upper_bound": float(upper_bound)}
    return bounds


def treat_outliers(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    threshold: float = 1.5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip extreme numeric values to the IQR-based bounds."""
    treated = df.copy()
    numeric_columns = columns or treated.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column not in treated.columns:
            continue
        series = pd.to_numeric(treated[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        lower_bound = series.quantile(lower_quantile) if pd.notna(lower_bound) else lower_bound
        upper_bound = series.quantile(upper_quantile) if pd.notna(upper_bound) else upper_bound
        treated[column] = series.clip(lower_bound, upper_bound)
    return treated


def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target vector from the cleaned dataset."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' was not found in the cleaned dataset")

    features = df.drop(columns=[target_col]).copy()
    features = features.drop(columns=["Date"], errors="ignore")

    target = df[target_col].astype("string").str.strip().str.title()
    target = target.replace({"Yes": 1, "No": 0})
    target = pd.to_numeric(target, errors="coerce")

    valid_rows = target.notna()
    features = features.loc[valid_rows].reset_index(drop=True)
    target = target.loc[valid_rows].astype(int).reset_index(drop=True)
    return features, target


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train, validation, and test splits using stratification."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must lie between 0 and 1")

    combined_test_size = test_size + validation_size
    if combined_test_size >= 1:
        raise ValueError("The sum of test_size and validation_size must be less than 1")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=combined_test_size,
        random_state=random_state,
        stratify=y,
    )

    validation_ratio = validation_size / combined_test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(
    X_train: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[list[str], Pipeline]:
    """Build a reproducible preprocessing pipeline with column transformers."""
    feature_columns = [column for column in X_train.columns if column != target_col]

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(X_train[column])
    ]
    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    transformers: list[Tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline([("preprocessor", preprocessor)])
    pipeline.fit(X_train)
    return feature_columns, pipeline


def generate_preprocessing_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None,
) -> Path:
    """Generate a markdown report and visualization plots for preprocessing quality."""
    output_path = Path(output_dir or "reports")
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    missing_before = check_missing_values(original_df)
    missing_after = check_missing_values(cleaned_df)

    feature_columns = [
        column
        for column in ["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm", "TemperatureDifference", "AverageTemperature", "PressureDifference"]
        if column in cleaned_df.columns and column in original_df.columns
    ]
    if not feature_columns:
        feature_columns = [column for column in ["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm"] if column in cleaned_df.columns]

    summary_before = original_df[feature_columns].describe().T[["mean", "std", "min", "max"]]
    summary_after = cleaned_df[feature_columns].describe().T[["mean", "std", "min", "max"]]
    comparison = pd.concat([summary_before.add_suffix("_before"), summary_after.add_suffix("_after")], axis=1)

    report_lines = []
    report_lines.append("# Preprocessing Report")
    report_lines.append("")
    report_lines.append("## Dataset Summary")
    report_lines.append(f"- Original rows: {len(original_df)}")
    report_lines.append(f"- Cleaned rows: {len(cleaned_df)}")
    report_lines.append(f"- Training rows: {len(X_train)}")
    report_lines.append(f"- Validation rows: {len(X_val)}")
    report_lines.append(f"- Test rows: {len(X_test)}")
    report_lines.append("")
    report_lines.append("## Missing Value Summary")
    report_lines.append("### Before")
    report_lines.append(missing_before.head(10).to_string(index=False))
    report_lines.append("")
    report_lines.append("### After")
    report_lines.append(missing_after.head(10).to_string(index=False))
    report_lines.append("")
    report_lines.append("## Before vs After Statistics")
    report_lines.append(comparison.to_string())
    report_lines.append("")
    report_lines.append("## Feature Distribution")
    feature_plot_path = plots_dir / "feature_distributions.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, column in zip(axes.flatten(), feature_columns[:4]):
        cleaned_df[column].dropna().hist(bins=30, ax=axis, edgecolor="black")
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(feature_plot_path, dpi=200)
    plt.close(fig)
    report_lines.append(f"- Feature distribution plot: {feature_plot_path}")

    report_lines.append("")
    report_lines.append("## Class Balance")
    balance_plot_path = plots_dir / "target_balance.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y_train.value_counts().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Training target balance")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    y_test.value_counts().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Test target balance")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(balance_plot_path, dpi=200)
    plt.close(fig)
    report_lines.append(f"- Class balance plot: {balance_plot_path}")

    report_path = output_path / "preprocessing_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Preprocessing report saved to %s", report_path)
    return report_path


def save_preprocessing_artifacts(
    cleaned_df: pd.DataFrame,
    pipeline: Pipeline,
    output_dir: Optional[str] = None,
    feature_columns: Optional[list[str]] = None,
) -> Dict[str, Path]:
    """Persist the cleaned dataset and preprocessing pipeline artifacts to disk."""
    output_path = Path(output_dir or "models")
    output_path.mkdir(parents=True, exist_ok=True)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_dataset_path = processed_dir / "cleaned_weather_dataset.csv"
    cleaned_df.to_csv(cleaned_dataset_path, index=False)

    pipeline_path = output_path / "weather_preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    if "cat" in preprocessor.named_transformers_:
        categorical_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        joblib.dump(categorical_encoder, output_path / "categorical_encoder.joblib")
    if "num" in preprocessor.named_transformers_:
        numeric_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        joblib.dump(numeric_scaler, output_path / "numeric_scaler.joblib")

    metadata = {
        "feature_columns": feature_columns or cleaned_df.columns.tolist(),
        "target_column": DEFAULT_TARGET_COL,
    }
    joblib.dump(metadata, output_path / "preprocessing_metadata.joblib")

    return {
        "cleaned_dataset": cleaned_dataset_path,
        "pipeline": pipeline_path,
        "categorical_encoder": output_path / "categorical_encoder.joblib",
        "numeric_scaler": output_path / "numeric_scaler.joblib",
        "metadata": output_path / "preprocessing_metadata.joblib",
    }


def run_full_preprocessing_pipeline(
    input_path: str,
    output_dir: Optional[str] = None,
    reports_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute the complete preprocessing workflow from raw CSV to model-ready splits."""
    raw_df = load_dataset(input_path)
    cleaned_df = clean_dataset(raw_df, target_col=DEFAULT_TARGET_COL)

    X, y = prepare_feature_matrix(cleaned_df, target_col=DEFAULT_TARGET_COL)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X,
        y,
        random_state=random_state,
    )

    feature_columns, pipeline = build_preprocessing_pipeline(X_train, target_col=DEFAULT_TARGET_COL)
    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    saved_artifacts = save_preprocessing_artifacts(
        cleaned_df,
        pipeline,
        output_dir=output_dir,
        feature_columns=feature_columns,
    )
    report_path = generate_preprocessing_report(
        raw_df,
        cleaned_df,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        output_dir=reports_dir or "reports",
    )

    return {
        "cleaned_dataset": cleaned_df,
        "feature_columns": feature_columns,
        "train_features": X_train,
        "validation_features": X_val,
        "test_features": X_test,
        "train_target": y_train,
        "validation_target": y_val,
        "test_target": y_test,
        "processed_train": X_train_processed,
        "processed_validation": X_val_processed,
        "processed_test": X_test_processed,
        "artifacts": saved_artifacts,
        "report_path": report_path,
    }


def print_dataset_overview(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> None:
    """Print a concise overview of the dataset and its data quality."""
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW".center(80))
    print("=" * 80 + "\n")

    info = get_dataset_info(df)
    print(f"Shape: {info['shape'][0]:,} rows × {info['shape'][1]:,} columns")
    print(f"Memory Usage: {info['size_mb']} MB\n")

    quality = check_data_quality(df)
    print(f"Data Quality Score: {quality['quality_score']}%")
    print(f"  - Completeness: {quality['completeness_score']}%")
    print(f"  - Uniqueness: {quality['uniqueness_score']}%\n")

    col_types = get_column_types(df)
    print("Column Types:")
    print(f"  - Numeric: {len(col_types['numeric'])}")
    print(f"  - Categorical: {len(col_types['categorical'])}")
    print(f"  - Datetime: {len(col_types['datetime'])}\n")

    missing = check_missing_values(df)
    if not missing.empty:
        print(f"Columns with Missing Values: {len(missing)}")
        print(missing.head().to_string(index=False))
    else:
        print("No missing values found!\n")

    dup_info = check_duplicates(df)
    print(f"\nDuplicate Rows: {dup_info['total_duplicates']} ({dup_info['duplicate_percentage']}%)")

    if target_col in df.columns:
        target_info = analyze_target_variable(df, target_col)
        print(f"\nTarget Variable ('{target_col}'):")
        print(target_info["value_counts"].to_string())
        print(f"Distribution: {target_info['percentages'].to_string()}")

    print("\n" + "=" * 80 + "\n")
