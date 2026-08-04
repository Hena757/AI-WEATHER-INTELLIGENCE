"""Feature engineering utilities for the AI Weather Intelligence platform."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def engineer_weather_features(df: pd.DataFrame, target_col: Optional[str] = "RainTomorrow") -> pd.DataFrame:
    """Create weather-specific features from the raw weather dataset.

    The function performs lightweight, deterministic feature engineering suitable
    for downstream machine-learning models. It derives temperature-based metrics,
    humidity and pressure relationships, seasonal indicators, wind intensity
    categories, and rainfall flags from the original weather measurements.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weather dataset.
    target_col : str | None, default "RainTomorrow"
        Target column name if present. When available, it is normalized to a
        consistent string representation without changing the raw values.

    Returns
    -------
    pd.DataFrame
        Copy of the input data with engineered features appended.
    """
    engineered = df.copy()

    if "Date" in engineered.columns:
        engineered["Date"] = pd.to_datetime(engineered["Date"], errors="coerce")
        engineered = engineered.sort_values("Date").reset_index(drop=True)

    numeric_columns = [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "WindSpeed9am",
        "WindSpeed3pm",
        "WindGustSpeed",
    ]
    for column in numeric_columns:
        if column in engineered.columns:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    if {"MinTemp", "MaxTemp"}.issubset(engineered.columns):
        engineered["TemperatureDifference"] = engineered["MaxTemp"] - engineered["MinTemp"]
        engineered["AverageTemperature"] = (engineered["MinTemp"] + engineered["MaxTemp"]) / 2.0

    if {"Humidity9am", "Humidity3pm"}.issubset(engineered.columns):
        engineered["HumidityIndex"] = (engineered["Humidity9am"] + engineered["Humidity3pm"]) / 2.0

    if {"Pressure9am", "Pressure3pm"}.issubset(engineered.columns):
        engineered["PressureDifference"] = engineered["Pressure9am"] - engineered["Pressure3pm"]

    if "Date" in engineered.columns and pd.api.types.is_datetime64_any_dtype(engineered["Date"]):
        engineered["Month"] = engineered["Date"].dt.month
        engineered["Quarter"] = engineered["Date"].dt.quarter
        engineered["DayOfYear"] = engineered["Date"].dt.dayofyear
        engineered["Season"] = engineered["Month"].map(
            {
                12: "Summer",
                1: "Summer",
                2: "Summer",
                3: "Autumn",
                4: "Autumn",
                5: "Autumn",
                6: "Winter",
                7: "Winter",
                8: "Winter",
                9: "Spring",
                10: "Spring",
                11: "Spring",
            }
        )
        engineered["IsWeekend"] = engineered["Date"].dt.dayofweek.isin([5, 6]).astype(int)

    if {"WindSpeed3pm"}.issubset(engineered.columns):
        wind_speed = engineered["WindSpeed3pm"].fillna(0)
        engineered["WindIntensityCategory"] = np.select(
            [wind_speed < 10, wind_speed < 25, wind_speed < 40],
            ["Low", "Moderate", "High"],
            default="VeryHigh",
        )

    if {"Rainfall"}.issubset(engineered.columns):
        engineered["RainfallIndicator"] = (engineered["Rainfall"] > 0).astype(int)

    if {"RainToday"}.issubset(engineered.columns):
        engineered["RainTodayBinary"] = (
            engineered["RainToday"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"Yes": 1, "No": 0})
            .astype("Float64")
        )

    if target_col and target_col in engineered.columns:
        normalized_target = engineered[target_col].astype("string").str.strip().str.title()
        normalized_target = normalized_target.where(
            normalized_target.notna() & ~normalized_target.isin(["Nan", "None", ""]),
            pd.NA,
        )
        engineered[target_col] = normalized_target

    return engineered
