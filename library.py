"""
Helper functions for the Jakarta AQI Streamlit application.

This module handles:
- Loading the trained model
- Loading the cleaned Excel dataset
- Standardising user inputs
- Predicting AQI
- Calculating pollutant contributions
- Giving a simple AQI interpretation
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Display names used in the Streamlit interface
FEATURE_LABELS = {
    "pm10": "PM10",
    "so2": "SO₂",
    "co": "CO",
    "o3": "O₃",
    "no2": "NO₂",
}


def load_model(model_path: str) -> dict:
    """
    Load the saved multiple linear regression model.

    Parameters
    ----------
    model_path : str
        Path to the .npz model file.

    Returns
    -------
    dict
        Dictionary containing coefficients, feature means,
        feature standard deviations and feature names.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model_data = np.load(
        path,
        allow_pickle=True
    )

    return {
        "coefficients": model_data["coefficients"],
        "feature_mean": model_data["feature_mean"],
        "feature_std": model_data["feature_std"],
        "feature_names": model_data["feature_names"].tolist(),
    }


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the cleaned Jakarta air-quality Excel dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the cleaned .xlsx file.

    Returns
    -------
    pandas.DataFrame
        Cleaned air-quality dataset.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )

    dataset = pd.read_excel(
        path,
        engine="openpyxl"
    )

    required_columns = {
        "tanggal",
        "pm10",
        "so2",
        "co",
        "o3",
        "no2",
        "max",
    }

    missing_columns = required_columns.difference(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataset


def standardize_features(
    feature_values: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """
    Standardise pollutant values using the statistics
    from the training dataset.
    """
    safe_std = np.where(
        feature_std == 0,
        1,
        feature_std
    )

    scaled_values = (
        feature_values - feature_mean
    ) / safe_std

    return scaled_values


def predict_aqi(
    feature_values: np.ndarray,
    coefficients: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Predict AQI using the trained multiple linear regression model.

    Returns
    -------
    tuple
        Predicted AQI and the standardised feature values.
    """
    scaled_values = standardize_features(
        feature_values=feature_values,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    # Add the intercept value of 1 at the beginning
    model_input = np.insert(
        scaled_values,
        0,
        1.0
    )

    predicted_aqi = float(
        model_input @ coefficients
    )

    # Prevent impossible negative AQI predictions
    predicted_aqi = max(
        0.0,
        predicted_aqi
    )

    return predicted_aqi, scaled_values


def calculate_contributions(
    scaled_values: np.ndarray,
    coefficients: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Estimate each pollutant's contribution to the prediction.

    Contribution =
    standardised pollutant value × regression coefficient
    """
    contribution_values = (
        scaled_values * coefficients[1:]
    )

    contribution_table = pd.DataFrame({
        "Pollutant": [
            FEATURE_LABELS.get(
                feature,
                feature.upper()
            )
            for feature in feature_names
        ],
        "Contribution": contribution_values,
    })

    contribution_table[
        "Absolute contribution"
    ] = contribution_table[
        "Contribution"
    ].abs()

    return contribution_table


def get_aqi_interpretation(
    predicted_aqi: float,
) -> tuple[str, str]:
    """
    Return a simple interpretation for the predicted AQI.

    These descriptions are for project demonstration only
    and should not replace official government advisories.
    """
    if predicted_aqi <= 50:
        return (
            "Good",
            "Estimated air quality is relatively low-risk."
        )

    if predicted_aqi <= 100:
        return (
            "Moderate",
            "Sensitive groups may monitor official advisories."
        )

    if predicted_aqi <= 150:
        return (
            "Unhealthy for sensitive groups",
            "Vulnerable groups may need targeted precautions."
        )

    if predicted_aqi <= 200:
        return (
            "Unhealthy",
            "Stronger public-health guidance may be required."
        )

    return (
        "Very unhealthy",
        "Immediate monitoring and intervention may be required."
    )


def get_possible_source_message(
    pollutant: str,
) -> str:
    """
    Return a broad possible-source message for the pollutant.

    This is not a confirmed source attribution.
    """
    possible_sources = {
        "PM10": (
            "Possible sources may include road dust, "
            "construction activity and combustion."
        ),
        "SO₂": (
            "Possible sources may include industrial fuel "
            "combustion and fossil-fuel burning."
        ),
        "CO": (
            "Possible sources may include incomplete combustion "
            "and motor-vehicle emissions."
        ),
        "O₃": (
            "Ground-level ozone forms through atmospheric "
            "reactions involving precursor pollutants."
        ),
        "NO₂": (
            "Possible sources may include road traffic and "
            "industrial combustion."
        ),
    }

    return possible_sources.get(
        pollutant,
        "Further investigation is required to identify the source."
    )