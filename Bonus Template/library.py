
"""Shared helpers for the Jakarta AQI Streamlit multipage app."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

FEATURE_LABELS = {
    "pm10": "PM₁₀",
    "so2": "SO₂",
    "co": "CO",
    "o3": "O₃",
    "no2": "NO₂",
}

MODEL_FILE = "aqi_model.npz"
DATA_FILE = "jakarta_air_quality_cleaned.xlsx"


def apply_page_config(page_title: str, page_icon: str = "🌿") -> None:
    """Apply consistent Streamlit page settings."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def apply_global_styles() -> None:
    """Apply the shared visual style used by every page."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

            :root {
                --background: #F4F7F2;
                --surface: #FFFFFF;
                --text: #18221B;
                --muted: #657168;
                --primary: #1E6B43;
                --primary-dark: #154C31;
                --accent: #B7D86C;
                --border: rgba(24, 34, 27, 0.10);
            }

            html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
            h1, h2, h3 { font-family: "Space Grotesk", sans-serif; letter-spacing: -0.03em; }

            .stApp {
                background: radial-gradient(circle at top right, rgba(183,216,108,.16), transparent 28%), var(--background);
            }

            .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 4rem;
            }

            #MainMenu, footer { visibility: hidden; }
            header[data-testid="stHeader"] { background: transparent; }

            .hero-card {
                position: relative;
                overflow: hidden;
                padding: 3rem;
                border-radius: 30px;
                background: linear-gradient(135deg, #0F5132, #1E6B45);
                color: white;
                margin-bottom: 2.2rem;
                box-shadow: 0 24px 60px rgba(30,107,67,.18);
            }

            .hero-card::after {
                content: "";
                position: absolute;
                width: 300px;
                height: 300px;
                border-radius: 50%;
                right: -90px;
                top: -130px;
                background: rgba(183,216,108,.22);
            }

            .hero-card h1 {
                position: relative;
                z-index: 1;
                margin: 1rem 0 .6rem 0;
                color: white;
                font-size: clamp(2.2rem, 5vw, 4rem);
            }

            .hero-card p {
                position: relative;
                z-index: 1;
                max-width: 760px;
                margin: 0;
                color: rgba(255,255,255,.88);
                font-size: 1.06rem;
                line-height: 1.65;
            }

            .hero-pill {
                position: relative;
                z-index: 1;
                display: inline-block;
                padding: .48rem .9rem;
                border-radius: 999px;
                background: rgba(255,255,255,.12);
                border: 1px solid rgba(255,255,255,.22);
                font-weight: 700;
                letter-spacing: .08em;
                font-size: .78rem;
            }

            .section-label {
                color: var(--primary);
                font-size: .78rem;
                font-weight: 700;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin-bottom: .3rem;
            }

            .section-title {
                color: var(--text);
                font-family: "Space Grotesk", sans-serif;
                font-size: 1.9rem;
                font-weight: 650;
                margin-bottom: .35rem;
            }

            .section-copy { color: var(--muted); margin-bottom: 1.25rem; }

            div[data-testid="stForm"],
            div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--border) !important;
                border-radius: 22px !important;
                background: rgba(255,255,255,.86) !important;
                box-shadow: 0 14px 34px rgba(24,34,27,.055) !important;
            }

            div[data-testid="stNumberInput"] input {
                border-radius: 12px;
                min-height: 2.9rem;
                background: #F7F9F6;
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1.2rem 1.25rem;
                min-height: 132px;
                box-shadow: 0 12px 30px rgba(24,34,27,.055);
            }

            div[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
            div[data-testid="stMetricValue"] { color: var(--text); font-size: 2.15rem; }

            div[data-testid="stFormSubmitButton"] > button {
                width: 100%;
                min-height: 3.2rem;
                border: none;
                border-radius: 14px;
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                font-size: 1rem;
                font-weight: 700;
                box-shadow: 0 10px 22px rgba(30,107,67,.22);
            }

            .insight-card {
                padding: 1.35rem 1.5rem;
                border-radius: 20px;
                background: linear-gradient(135deg, #EFF7E9, #F8FAF5);
                border: 1px solid rgba(30,107,67,.13);
                margin-top: 1rem;
                margin-bottom: 1.2rem;
            }

            .source-card, .workflow-step, .limitation-item {
                height: 100%;
                padding: 1.2rem;
                border-radius: 16px;
                background: #F8FAF7;
                border: 1px solid rgba(24,34,27,.08);
            }

            .workflow-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                background: #DDECD9;
                color: #154C31;
                font-weight: 700;
                margin-bottom: .8rem;
            }

            .workflow-title { font-weight: 700; margin-bottom: .35rem; }
            .workflow-copy { color: var(--muted); font-size: .91rem; line-height: 1.5; margin: 0; }

            .model-pill {
                display: inline-block;
                margin: .25rem .35rem .25rem 0;
                padding: .45rem .75rem;
                border-radius: 999px;
                background: #EDF4EA;
                color: #24563A;
                font-size: .86rem;
                font-weight: 600;
            }

            .limitations-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: .8rem;
            }

            div[data-testid="stAlert"], div[data-testid="stExpander"] { border-radius: 16px; }
            hr { margin: 2rem 0; }

            @media (max-width: 760px) {
                .limitations-grid { grid-template-columns: 1fr; }
                .hero-card { padding: 2rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the main dashboard header."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-pill">ENVIRONMENTAL DECISION SUPPORT</div>
            <h1>Jakarta AQI Intelligence Dashboard</h1>
            <p>
                Estimate daily air quality from pollutant concentrations,
                identify the strongest statistical contributor, and translate
                predictions into actionable monitoring and public-health recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_model(model_path: str = MODEL_FILE) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_data = np.load(path, allow_pickle=True)
    return {
        "coefficients": model_data["coefficients"],
        "feature_mean": model_data["feature_mean"],
        "feature_std": model_data["feature_std"],
        "feature_names": model_data["feature_names"].tolist(),
    }


def load_dataset(dataset_path: str = DATA_FILE) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    dataset = pd.read_excel(path, engine="openpyxl")
    required = {"tanggal", "pm10", "so2", "co", "o3", "no2", "max"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"The dataset is missing required columns: {sorted(missing)}")
    return dataset


@st.cache_resource
def get_model() -> dict:
    return load_model()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset()


def load_app_resources() -> tuple[dict, pd.DataFrame]:
    """Load shared model and dataset, showing a helpful UI error if missing."""
    try:
        return get_model(), get_dataset()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info(
            "Place `aqi_model.npz` and `jakarta_air_quality_cleaned.xlsx` "
            "inside the Bonus Template folder."
        )
        st.stop()


def standardize_features(feature_values, feature_mean, feature_std):
    safe_std = np.where(feature_std == 0, 1, feature_std)
    return (feature_values - feature_mean) / safe_std


def predict_aqi(feature_values, coefficients, feature_mean, feature_std):
    scaled_values = standardize_features(feature_values, feature_mean, feature_std)
    model_input = np.insert(scaled_values, 0, 1.0)
    predicted_aqi = max(0.0, float(model_input @ coefficients))
    return predicted_aqi, scaled_values


def calculate_contributions(scaled_values, coefficients, feature_names):
    values = scaled_values * coefficients[1:]
    table = pd.DataFrame({
        "Pollutant": [FEATURE_LABELS.get(name, name.upper()) for name in feature_names],
        "Contribution": values,
    })
    table["Absolute contribution"] = table["Contribution"].abs()
    return table


def get_aqi_interpretation(predicted_aqi: float) -> tuple[str, str]:
    if predicted_aqi <= 50:
        return "Good", "Estimated air quality is relatively low-risk."
    if predicted_aqi <= 100:
        return "Moderate", "Sensitive groups may monitor official advisories."
    if predicted_aqi <= 150:
        return "Unhealthy for sensitive groups", "Vulnerable groups may need targeted precautions."
    if predicted_aqi <= 200:
        return "Unhealthy", "Stronger public-health guidance may be required."
    return "Very unhealthy", "Immediate monitoring and intervention may be required."


def get_possible_source_message(pollutant: str) -> str:
    messages = {
        "PM10": "Possible sources may include road dust, construction activity and combustion.",
        "SO₂": "Possible sources may include industrial fuel combustion and fossil-fuel burning.",
        "CO": "Possible sources may include incomplete combustion and motor-vehicle emissions.",
        "O₃": "Ground-level ozone forms through atmospheric reactions involving precursor pollutants.",
        "NO₂": "Possible sources may include road traffic and industrial combustion.",
    }
    return messages.get(pollutant, "Further investigation is required to identify the source.")


def get_risk_level(predicted_aqi: float) -> str:
    if predicted_aqi <= 50:
        return "Low"
    if predicted_aqi <= 100:
        return "Watch"
    if predicted_aqi <= 150:
        return "Elevated"
    if predicted_aqi <= 200:
        return "High"
    return "Severe"


def get_recommendations(predicted_aqi: float, strongest_pollutant: str) -> list[str]:
    recommendations = [
        f"Prioritise monitoring of {strongest_pollutant} and verify readings with nearby stations.",
        "Compare the result with traffic, industrial and weather data before identifying an emission source.",
    ]
    if predicted_aqi <= 50:
        recommendations.append("Continue routine monitoring and maintain current controls.")
    elif predicted_aqi <= 100:
        recommendations.append("Sensitive groups should monitor official air-quality updates.")
    elif predicted_aqi <= 150:
        recommendations.append("Consider targeted guidance for children, older adults and people with respiratory conditions.")
    else:
        recommendations.append("Consider stronger public-health communication and rapid investigation of elevated readings.")
    return recommendations
