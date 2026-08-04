"""Main AQI prediction page."""

from time import sleep

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from library import (
    FEATURE_LABELS,
    apply_global_styles,
    apply_page_config,
    calculate_contributions,
    get_aqi_interpretation,
    get_possible_source_message,
    get_recommendations,
    get_risk_level,
    load_app_resources,
    predict_aqi,
    render_hero,
)

apply_page_config("Jakarta AQI Intelligence")
apply_global_styles()
model, dataset = load_app_resources()
render_hero()

st.markdown(
    """
    <div class="section-label">Step 1</div>
    <div class="section-title">Enter pollutant measurements</div>
    <div class="section-copy">
        Use current or hypothetical pollutant concentrations. Default values
        represent the median of the cleaned dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("aqi_form"):
    top_row = st.columns(3)
    bottom_row = st.columns(2)
    input_slots = [top_row[0], top_row[1], top_row[2], bottom_row[0], bottom_row[1]]
    user_values = []

    for index, feature_name in enumerate(model["feature_names"]):
        feature_data = dataset[feature_name]
        display_name = FEATURE_LABELS.get(feature_name, feature_name.upper())
        with input_slots[index]:
            value = st.number_input(
                label=display_name,
                min_value=float(feature_data.min()),
                max_value=float(feature_data.max()),
                value=float(feature_data.median()),
                step=1.0,
                help=f"Historical range: {feature_data.min():.0f}–{feature_data.max():.0f}",
                key=f"input_{feature_name}",
            )
        user_values.append(value)

    submitted = st.form_submit_button("Analyse air quality", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Analysing pollutant interactions and estimating AQI..."):
        sleep(0.6)
        input_array = np.array(user_values, dtype=float)
        predicted_aqi, scaled_values = predict_aqi(
            input_array,
            model["coefficients"],
            model["feature_mean"],
            model["feature_std"],
        )
        category, health_message = get_aqi_interpretation(predicted_aqi)
        contribution_table = calculate_contributions(
            scaled_values,
            model["coefficients"],
            model["feature_names"],
        )

    strongest_index = contribution_table["Absolute contribution"].idxmax()
    strongest_pollutant = str(contribution_table.loc[strongest_index, "Pollutant"])
    strongest_contribution = float(contribution_table.loc[strongest_index, "Contribution"])
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
        st.metric("Predicted AQI", f"{predicted_aqi:.1f}")
    with metric_two:
        st.metric("Estimated category", category)
    with metric_three:
        st.metric("Strongest contributor", strongest_pollutant)

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Public-health signal:</strong> {health_message}<br><br>
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
            Each bar shows how the entered pollutant value shifts the AQI
            prediction away from the model baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_data = contribution_table[["Pollutant", "Contribution"]].copy()
    chart_data["Absolute"] = chart_data["Contribution"].abs()
    chart_data["Direction"] = np.where(
        chart_data["Contribution"] >= 0,
        "Raises predicted AQI",
        "Lowers predicted AQI",
    )

    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=8, size=24)
        .encode(
            y=alt.Y(
                "Pollutant:N",
                sort=alt.SortField(field="Absolute", order="descending"),
                title=None,
                axis=alt.Axis(labelFontSize=14, labelFontWeight=600, labelPadding=12, ticks=False, domain=False),
            ),
            x=alt.X(
                "Contribution:Q",
                title="Contribution relative to baseline",
                axis=alt.Axis(grid=True, gridColor="#E4EAE2", domain=False),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Raises predicted AQI", "Lowers predicted AQI"],
                    range=["#C94B43", "#2C7A61"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Contribution:Q", format=".3f"),
                alt.Tooltip("Direction:N"),
            ],
        )
        .properties(height=320)
    )

    zero_line = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
        color="#657168", strokeDash=[4, 4]
    ).encode(x="zero:Q")

    st.altair_chart(bars + zero_line, use_container_width=True)

    direction_word = "raised" if strongest_contribution > 0 else "lowered"
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Key insight:</strong> {strongest_pollutant} had the largest
            statistical influence and {direction_word} the predicted AQI
            relative to the model baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "A negative contribution does not mean a pollutant is healthy. It means "
        "the entered value is below the model's training average."
    )

    st.markdown(
        """
        <div class="section-label">Step 4</div>
        <div class="section-title">Decision-support insight</div>
        <div class="section-copy">
            Translate the output into possible investigation priorities without
            claiming confirmed causation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_col, action_col = st.columns([1, 1.2])
    with source_col:
        st.markdown(
            f"""
            <div class="source-card">
                <h4>🔎 Possible source investigation</h4>
                <p>{get_possible_source_message(strongest_pollutant)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        recommendation_html = "".join(
            f"<li>{item}</li>" for item in get_recommendations(predicted_aqi, strongest_pollutant)
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
        "This model identifies statistical relationships only. Confirmed source "
        "attribution requires location, traffic, industrial, weather and emissions data."
    )

    with st.expander("View entered values and detailed contributions"):
        detail_one, detail_two = st.columns(2)
        with detail_one:
            st.markdown("#### Entered concentrations")
            entered_table = pd.DataFrame({
                "Pollutant": [FEATURE_LABELS.get(name, name.upper()) for name in model["feature_names"]],
                "Concentration": user_values,
            })
            st.dataframe(entered_table, use_container_width=True, hide_index=True)
        with detail_two:
            st.markdown("#### Model contributions")
            display_table = contribution_table[["Pollutant", "Contribution"]].copy()
            display_table["Contribution"] = display_table["Contribution"].round(3)
            st.dataframe(display_table, use_container_width=True, hide_index=True)
