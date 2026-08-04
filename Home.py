"""
Jakarta AQI Intelligence Dashboard

This Streamlit application predicts Jakarta's AQI using
a multiple linear regression model and explains the
relative contribution of each pollutant.
"""

from time import sleep

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from library import (
    FEATURE_LABELS,
    calculate_contributions,
    get_aqi_interpretation,
    get_possible_source_message,
    load_dataset,
    load_model,
    predict_aqi,
)


MODEL_FILE = "aqi_model.npz"
DATA_FILE = "jakarta_air_quality_cleaned.xlsx"


st.set_page_config(
    page_title="Jakarta AQI Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        @import url(
            'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap'
        );

        :root {
            --background: #F4F7F2;
            --surface: #FFFFFF;
            --surface-soft: #EEF4EC;
            --text: #18221B;
            --muted: #657168;
            --primary: #1E6B43;
            --primary-dark: #154C31;
            --accent: #B7D86C;
            --warning: #E9A23B;
            --danger: #C94B43;
            --border: rgba(24, 34, 27, 0.10);
        }

        html, body, [class*="css"] {
            font-family: "DM Sans", sans-serif;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.03em;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at top right,
                    rgba(183, 216, 108, 0.16),
                    transparent 28%
                ),
                var(--background);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.8rem 3rem;
            border-radius: 28px;
            background:
                linear-gradient(
                    135deg,
                    #163F2C 0%,
                    #1E6B43 62%,
                    #397C4D 100%
                );
            color: white;
            box-shadow:
                0 24px 60px rgba(30, 107, 67, 0.18);
            margin-bottom: 2rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            right: -80px;
            top: -100px;
            background: rgba(183, 216, 108, 0.22);
            filter: blur(2px);
        }

        .hero-kicker {
            display: inline-block;
            padding: 0.42rem 0.85rem;
            margin-bottom: 1rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.16);
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            margin: 0;
            font-size: clamp(2.3rem, 5vw, 4rem);
            line-height: 1.02;
            max-width: 760px;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 720px;
            margin-top: 1rem;
            margin-bottom: 0;
            color: rgba(255, 255, 255, 0.86);
            font-size: 1.05rem;
            line-height: 1.65;
        }

        .section-label {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .section-title {
            color: var(--text);
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.85rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 1.25rem;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.82);
            box-shadow:
                0 14px 35px rgba(24, 34, 27, 0.06);
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
            box-shadow:
                0 12px 30px rgba(24, 34, 27, 0.055);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-family: "Space Grotesk", sans-serif;
            font-size: 2.15rem;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 3.2rem;
            border: none;
            border-radius: 14px;
            background:
                linear-gradient(
                    135deg,
                    var(--primary),
                    var(--primary-dark)
                );
            color: white;
            font-size: 1rem;
            font-weight: 700;
            box-shadow:
                0 10px 22px rgba(30, 107, 67, 0.22);
            transition:
                transform 0.18s ease,
                box-shadow 0.18s ease;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow:
                0 14px 28px rgba(30, 107, 67, 0.28);
        }

        .insight-card {
            padding: 1.35rem 1.5rem;
            border-radius: 20px;
            background:
                linear-gradient(
                    135deg,
                    #EFF7E9,
                    #F8FAF5
                );
            border: 1px solid rgba(30, 107, 67, 0.13);
            margin-top: 1rem;
            margin-bottom: 1.2rem;
        }

        .insight-card strong {
            color: var(--primary-dark);
        }

        .source-card {
            height: 100%;
            padding: 1.35rem;
            border-radius: 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow:
                0 12px 28px rgba(24, 34, 27, 0.05);
        }

        .source-card h4 {
            font-family: "Space Grotesk", sans-serif;
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
        }

        .source-card p {
            color: var(--muted);
            margin-bottom: 0;
            line-height: 1.55;
        }

        .model-pill {
            display: inline-block;
            margin: 0.25rem 0.35rem 0.25rem 0;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: #EDF4EA;
            color: #24563A;
            font-size: 0.86rem;
            font-weight: 600;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 12px;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Bordered Streamlit containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(24, 34, 27, 0.10) !important;
            border-radius: 22px !important;
            background: rgba(255, 255, 255, 0.86) !important;
            box-shadow: 0 14px 34px rgba(24, 34, 27, 0.055) !important;
            padding: 0.45rem 0.6rem 0.7rem 0.6rem !important;
        }

        .workflow-step {
            height: 100%;
            padding: 1.15rem;
            border-radius: 16px;
            background: #F7FAF5;
            border: 1px solid rgba(30, 107, 67, 0.10);
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
            margin-bottom: 0.8rem;
        }

        .workflow-title {
            color: var(--text);
            font-family: "Space Grotesk", sans-serif;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .workflow-copy {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.5;
            margin: 0;
        }

        .limitations-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .limitation-item {
            padding: 1rem;
            border-radius: 14px;
            background: #F8FAF7;
            border: 1px solid rgba(24, 34, 27, 0.08);
            color: #4A554E;
            line-height: 1.5;
        }

        hr {
            margin: 2rem 0;
        }
        .hero-card{
            background:linear-gradient(135deg,#0F5132,#1E6B45);
            padding:50px;
            border-radius:30px;
            color:white;
            margin-bottom:40px;
            position:relative;
            overflow:hidden;
            }

            .hero-card h1{
            font-size:52px;
            font-weight:800;
            margin-top:25px;
            margin-bottom:15px;
            color:white;
            }

            .hero-card p{
            font-size:22px;
            line-height:1.8;
            max-width:900px;
            opacity:.95;
            }

            .hero-pill{
            display:inline-block;
            padding:10px 22px;
            border-radius:999px;
            background:rgba(255,255,255,.12);
            border:1px solid rgba(255,255,255,.25);
            font-weight:700;
            letter-spacing:.08em;
            font-size:14px;
            }

        @media (max-width: 760px) {
            .limitations-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource
def get_model() -> dict:
    """Load and cache the trained model."""
    return load_model(MODEL_FILE)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load and cache the cleaned dataset."""
    return load_dataset(DATA_FILE)


try:
    model = get_model()
    dataset = get_dataset()

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))

    st.info(
        """
        Place these files in the same folder as `Home.py`:

        - `aqi_model.npz`
        - `jakarta_air_quality_cleaned.xlsx`
        """
    )

    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_risk_level(predicted_aqi: float) -> str:
    """Return a short risk label."""
    if predicted_aqi <= 50:
        return "Low"
    if predicted_aqi <= 100:
        return "Watch"
    if predicted_aqi <= 150:
        return "Elevated"
    if predicted_aqi <= 200:
        return "High"
    return "Severe"


def get_recommendations(
    predicted_aqi: float,
    strongest_pollutant: str,
) -> list[str]:
    """Generate simple recommendations for the prediction."""
    recommendations = [
        (
            f"Prioritise monitoring of {strongest_pollutant} "
            "and verify readings with nearby stations."
        ),
        (
            "Compare the result with traffic, industrial and "
            "weather data before identifying an emission source."
        ),
    ]

    if predicted_aqi <= 50:
        recommendations.append(
            "Continue routine monitoring and maintain current controls."
        )

    elif predicted_aqi <= 100:
        recommendations.append(
            "Sensitive groups should monitor official air-quality updates."
        )

    elif predicted_aqi <= 150:
        recommendations.append(
            "Consider targeted guidance for children, older adults "
            "and people with respiratory conditions."
        )

    else:
        recommendations.append(
            "Consider stronger public-health communication and "
            "rapid investigation of elevated pollutant readings."
        )

    return recommendations


st.markdown("""
<div class="hero-card">

<div class="hero-pill">
ENVIRONMENTAL DECISION SUPPORT
</div>

<h1>Jakarta AQI Intelligence Dashboard</h1>

<p>
Estimate daily air quality from pollutant concentrations,
identify the strongest statistical contributor,
and translate predictions into actionable monitoring and
public-health recommendations.
</p>

</div>
""", unsafe_allow_html=True)

predictor_tab, model_tab, dataset_tab = st.tabs(
    [
        "Analyse air quality",
        "Model and methodology",
        "Dataset transparency",
    ]
)


# =========================================================
# PREDICTOR TAB
# =========================================================

with predictor_tab:

    st.markdown(
        """
        <div class="section-label">Step 1</div>
        <div class="section-title">Enter pollutant measurements</div>
        <div class="section-copy">
            Use current or hypothetical pollutant concentrations.
            Default values represent the median of the cleaned dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("aqi_form"):

        top_row = st.columns(3)
        bottom_row = st.columns(2)

        input_slots = [
            top_row[0],
            top_row[1],
            top_row[2],
            bottom_row[0],
            bottom_row[1],
        ]

        user_values = []

        for index, feature_name in enumerate(
            model["feature_names"]
        ):
            feature_data = dataset[feature_name]

            minimum_value = float(feature_data.min())
            maximum_value = float(feature_data.max())
            default_value = float(feature_data.median())

            display_name = FEATURE_LABELS.get(
                feature_name,
                feature_name.upper(),
            )

            with input_slots[index]:
                value = st.number_input(
                    label=display_name,
                    min_value=minimum_value,
                    max_value=maximum_value,
                    value=default_value,
                    step=1.0,
                    help=(
                        f"Historical range: "
                        f"{minimum_value:.0f}–{maximum_value:.0f}"
                    ),
                    key=f"input_{feature_name}",
                )

            user_values.append(value)

        submitted = st.form_submit_button(
            "Analyse air quality",
            type="primary",
            use_container_width=True,
        )


    if submitted:

        with st.spinner(
            "Analysing pollutant interactions and estimating AQI..."
        ):
            sleep(0.8)

            input_array = np.array(
                user_values,
                dtype=float,
            )

            predicted_aqi, scaled_values = predict_aqi(
                feature_values=input_array,
                coefficients=model["coefficients"],
                feature_mean=model["feature_mean"],
                feature_std=model["feature_std"],
            )

            category, health_message = get_aqi_interpretation(
                predicted_aqi
            )

            contribution_table = calculate_contributions(
                scaled_values=scaled_values,
                coefficients=model["coefficients"],
                feature_names=model["feature_names"],
            )

        strongest_index = contribution_table[
            "Absolute contribution"
        ].idxmax()

        strongest_pollutant = str(
            contribution_table.loc[
                strongest_index,
                "Pollutant",
            ]
        )

        strongest_contribution = float(
            contribution_table.loc[
                strongest_index,
                "Contribution",
            ]
        )

        risk_level = get_risk_level(predicted_aqi)

        st.divider()

        st.markdown(
            """
            <div class="section-label">Step 2</div>
            <div class="section-title">Prediction overview</div>
            """,
            unsafe_allow_html=True,
        )

        metric_one, metric_two, metric_three = st.columns(3)

        with metric_one:
            st.metric(
                label="Predicted AQI",
                value=f"{predicted_aqi:.1f}",
            )

        with metric_two:
            st.metric(
                label="Estimated category",
                value=category,
            )

        with metric_three:
            st.metric(
                label="Strongest contributor",
                value=strongest_pollutant,
            )

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Public-health signal:</strong>
                {health_message}
                <br><br>
                <strong>Current risk level:</strong> {risk_level}
            </div>
            """,
            unsafe_allow_html=True,
        )



        st.markdown(
            """
            <div class="section-label">Step 3</div>
            <div class="section-title">Relative pollutant influence</div>
            <div class="section-copy">
                Each bar shows how the entered pollutant value shifts
                the AQI prediction away from the model baseline.
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_data = contribution_table[
            [
                "Pollutant",
                "Contribution",
            ]
        ].copy()

        chart_data["Absolute"] = chart_data[
            "Contribution"
        ].abs()

        chart_data["Direction"] = np.where(
            chart_data["Contribution"] >= 0,
            "Raises predicted AQI",
            "Lowers predicted AQI",
        )

        chart_data["Value label"] = chart_data[
            "Contribution"
        ].map(lambda value: f"{value:+.2f}")

        bars = (
            alt.Chart(chart_data)
            .mark_bar(
                cornerRadiusEnd=8,
                height=18,
            )
            .encode(
                y=alt.Y(
                    "Pollutant:N",
                    sort=alt.SortField(
                        field="Absolute",
                        order="descending",
                    ),
                    title=None,
                    axis=alt.Axis(
                        labelFontSize=14,
                        labelFontWeight=600,
                        labelPadding=12,
                        ticks=False,
                        domain=False,
                    ),
                ),
                x=alt.X(
                    "Contribution:Q",
                    title="Contribution relative to baseline",
                    axis=alt.Axis(
                        grid=True,
                        gridColor="#E4EAE2",
                        domain=False,
                        tickColor="#C9D2C8",
                    ),
                ),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(
                        domain=[
                            "Raises predicted AQI",
                            "Lowers predicted AQI",
                        ],
                        range=[
                            "#C94B43",
                            "#2C7A61",
                        ],
                    ),
                    legend=alt.Legend(
                        orient="top",
                        title=None,
                        labelFontSize=12,
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Pollutant:N",
                        title="Pollutant",
                    ),
                    alt.Tooltip(
                        "Contribution:Q",
                        title="Contribution",
                        format=".3f",
                    ),
                    alt.Tooltip(
                        "Direction:N",
                        title="Effect",
                    ),
                ],
            )
            .properties(
                height=290,
            )
        )

        zero_line = (
            alt.Chart(
                pd.DataFrame({"zero": [0]})
            )
            .mark_rule(
                color="#657168",
                strokeDash=[4, 4],
            )
            .encode(
                x="zero:Q"
            )
        )

        st.altair_chart(
            bars + zero_line,
            use_container_width=True,
        )

        direction_word = (
            "raised"
            if strongest_contribution > 0
            else "lowered"
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Key insight:</strong>
                {strongest_pollutant} had the largest statistical
                influence for this input and {direction_word} the
                predicted AQI relative to the model baseline.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            """
            A negative contribution does not mean a pollutant is healthy.
            It means the entered value is below the model's training
            average and therefore lowers this particular prediction
            relative to its baseline.
            """
        )


        # =================================================
        # SOURCE AND RECOMMENDATIONS
        # =================================================

        st.markdown(
            """
            <div class="section-label">Step 4</div>
            <div class="section-title">Decision-support insight</div>
            <div class="section-copy">
                Translate the model output into possible investigation
                priorities without claiming confirmed causation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_message = get_possible_source_message(
            strongest_pollutant
        )

        source_col, action_col = st.columns(
            [1, 1.2]
        )

        with source_col:
            st.markdown(
                f"""
                <div class="source-card">
                    <h4>🔎 Possible source investigation</h4>
                    <p>{source_message}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with action_col:
            recommendations = get_recommendations(
                predicted_aqi,
                strongest_pollutant,
            )

            recommendation_html = "".join(
                f"<li>{item}</li>"
                for item in recommendations
            )

            st.markdown(
                f"""
                <div class="source-card">
                    <h4>✓ Recommended next steps</h4>
                    <ul>{recommendation_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.warning(
            """
            This model identifies statistical relationships only.
            Confirmed source attribution requires location, traffic,
            industrial, meteorological and emissions data.
            """
        )



        with st.expander(
            "View entered values and detailed contributions"
        ):

            detail_one, detail_two = st.columns(2)

            with detail_one:
                st.markdown("#### Entered concentrations")

                entered_table = pd.DataFrame({
                    "Pollutant": [
                        FEATURE_LABELS.get(
                            feature,
                            feature.upper(),
                        )
                        for feature in model["feature_names"]
                    ],
                    "Concentration": user_values,
                })

                st.dataframe(
                    entered_table,
                    use_container_width=True,
                    hide_index=True,
                )

            with detail_two:
                st.markdown("#### Model contributions")

                display_contributions = contribution_table[
                    [
                        "Pollutant",
                        "Contribution",
                    ]
                ].copy()

                display_contributions["Contribution"] = (
                    display_contributions[
                        "Contribution"
                    ].round(3)
                )

                st.dataframe(
                    display_contributions,
                    use_container_width=True,
                    hide_index=True,
                )



with model_tab:

    st.markdown(
        """
        <div class="section-label">Technical overview</div>
        <div class="section-title">How the prediction is produced</div>
        <div class="section-copy">
            The application uses the trained multiple linear regression
            model created in the DDW notebook.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep the full workflow inside one visual container.
    with st.container(border=True):
        st.markdown("### Model workflow")

        workflow_steps = [
            (
                "Enter measurements",
                "The user provides PM10, SO₂, CO, O₃ and NO₂ concentrations.",
            ),
            (
                "Standardise inputs",
                "Values are scaled using the mean and standard deviation from the training set.",
            ),
            (
                "Predict AQI",
                "The saved regression coefficients calculate the estimated AQI.",
            ),
            (
                "Explain the result",
                "The app calculates the relative contribution of every pollutant.",
            ),
            (
                "Support decisions",
                "The strongest contribution is translated into a possible monitoring priority.",
            ),
        ]

        first_row = st.columns(3, gap="medium")
        second_row = st.columns(2, gap="medium")
        workflow_columns = list(first_row) + list(second_row)

        for index, (title, description) in enumerate(workflow_steps):
            with workflow_columns[index]:
                st.markdown(
                    f"""
                    <div class="workflow-step">
                        <div class="workflow-number">{index + 1}</div>
                        <div class="workflow-title">{title}</div>
                        <p class="workflow-copy">{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Keep all performance information in one visual container.
    with st.container(border=True):
        st.markdown("### Performance summary")

        st.markdown(
            """
            <span class="model-pill">Multiple Linear Regression</span>
            <span class="model-pill">Gradient Descent</span>
            <span class="model-pill">80% Training</span>
            <span class="model-pill">20% Testing</span>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        metric_a, metric_b, metric_c = st.columns(3, gap="medium")

        with metric_a:
            st.metric(
                label="Test MSE",
                value="337.03",
                help="Lower values indicate smaller squared prediction errors.",
            )

        with metric_b:
            st.metric(
                label="Test R²",
                value="0.8178",
                help="The proportion of AQI variation explained by the model.",
            )

        with metric_c:
            st.metric(
                label="Variation explained",
                value="81.78%",
            )

        st.info(
            """
            A test R² of 0.8178 means that the selected pollutant
            concentrations explain approximately 81.78% of the
            variation in AQI within the unseen test data.
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Put limitations together in one container rather than leaving
    # them as a loose list on the page.
    with st.container(border=True):
        st.markdown("### Limitations")

        st.markdown(
            """
            <div class="limitations-grid">
                <div class="limitation-item">
                    <strong>Linear assumption</strong><br>
                    The model assumes a linear relationship between pollutant concentrations and AQI.
                </div>
                <div class="limitation-item">
                    <strong>Missing weather variables</strong><br>
                    Wind, rainfall, humidity and other meteorological conditions were not included.
                </div>
                <div class="limitation-item">
                    <strong>No proof of causation</strong><br>
                    Contribution values describe the model prediction, not confirmed real-world causation.
                </div>
                <div class="limitation-item">
                    <strong>Decision support only</strong><br>
                    The results should not replace official environmental monitoring or health advisories.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



with dataset_tab:

    st.markdown(
        """
        <div class="section-label">Data transparency</div>
        <div class="section-title">Cleaned Jakarta air-quality dataset</div>
        <div class="section-copy">
            The application uses this file to establish realistic
            input ranges. Predictions use the saved trained model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_metric_one, dataset_metric_two = st.columns(2)

    with dataset_metric_one:
        st.metric(
            label="Cleaned observations",
            value=f"{len(dataset):,}",
        )

    with dataset_metric_two:
        st.metric(
            label="Model predictors",
            value="5",
        )

    display_columns = [
        "tanggal",
        "pm10",
        "so2",
        "co",
        "o3",
        "no2",
        "max",
    ]

    with st.expander(
        "Preview the first 20 cleaned observations"
    ):
        st.dataframe(
            dataset[display_columns].head(20),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "View descriptive statistics"
    ):
        summary = dataset[
            [
                "pm10",
                "so2",
                "co",
                "o3",
                "no2",
                "max",
            ]
        ].describe().T

        st.dataframe(
            summary,
            use_container_width=True,
        )