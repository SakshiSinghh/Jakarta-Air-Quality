"""Dataset transparency page."""

import streamlit as st

from library import apply_global_styles, apply_page_config, load_app_resources

apply_page_config("Dataset Transparency", "📊")
apply_global_styles()
_, dataset = load_app_resources()

st.markdown(
    """
    <div class="section-label">Data transparency</div>
    <div class="section-title">Cleaned Jakarta air-quality dataset</div>
    <div class="section-copy">
        The application uses this file to establish realistic input ranges.
        Predictions use the saved trained model.
    </div>
    """,
    unsafe_allow_html=True,
)

metric_one, metric_two, metric_three = st.columns(3)
with metric_one:
    st.metric("Cleaned observations", f"{len(dataset):,}")
with metric_two:
    st.metric("Model predictors", "5")
with metric_three:
    st.metric("Target variable", "AQI (max)")

columns = ["tanggal", "pm10", "so2", "co", "o3", "no2", "max"]

with st.expander("Preview the first 20 cleaned observations", expanded=True):
    st.dataframe(dataset[columns].head(20), use_container_width=True, hide_index=True)

with st.expander("View descriptive statistics"):
    summary = dataset[["pm10", "so2", "co", "o3", "no2", "max"]].describe().T
    st.dataframe(summary, use_container_width=True)
