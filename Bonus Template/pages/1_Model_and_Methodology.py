"""Model and methodology page."""

import streamlit as st

from library import apply_global_styles, apply_page_config

apply_page_config("Model and Methodology", "⚙️")
apply_global_styles()

st.markdown(
    """
    <div class="section-label">Technical overview</div>
    <div class="section-title">How the prediction is produced</div>
    <div class="section-copy">
        The application uses the trained multiple linear regression model
        created in the DDW notebook.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### Model workflow")
    workflow_steps = [
        ("Enter measurements", "The user provides PM10, SO₂, CO, O₃ and NO₂ concentrations."),
        ("Standardise inputs", "Values are scaled using the training-set mean and standard deviation."),
        ("Predict AQI", "The saved regression coefficients calculate the estimated AQI."),
        ("Explain the result", "The app calculates the relative contribution of every pollutant."),
        ("Support decisions", "The strongest contribution is translated into a possible monitoring priority."),
    ]

    first_row = st.columns(3, gap="medium")
    second_row = st.columns(2, gap="medium")
    columns = list(first_row) + list(second_row)

    for index, (title, description) in enumerate(workflow_steps):
        with columns[index]:
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

    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("Test MSE", "337.03", help="Lower values mean smaller squared prediction errors.")
    with metric_b:
        st.metric("Test R²", "0.8178", help="The proportion of AQI variation explained by the model.")
    with metric_c:
        st.metric("Variation explained", "81.78%")

    st.info(
        "A test R² of 0.8178 means the selected pollutant concentrations explain "
        "approximately 81.78% of AQI variation in the unseen test data."
    )

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("### Limitations")
    st.markdown(
        """
        <div class="limitations-grid">
            <div class="limitation-item"><strong>Linear assumption</strong><br>The model assumes a linear relationship between pollutants and AQI.</div>
            <div class="limitation-item"><strong>Missing weather variables</strong><br>Wind, rainfall, humidity and other meteorological conditions were not included.</div>
            <div class="limitation-item"><strong>No proof of causation</strong><br>Contribution values describe the model prediction, not confirmed real-world causation.</div>
            <div class="limitation-item"><strong>Decision support only</strong><br>The results should not replace official monitoring or health advisories.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
