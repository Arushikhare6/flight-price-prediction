import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import time


# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered",
)


# ── Load Model (cached so it's not reloaded on every interaction) ──
@st.cache_resource
def load_model():
    MODEL_DIR = Path("models")
    return joblib.load(MODEL_DIR / "final_tuned_model.pkl")


model_pipeline = load_model()
preprocessor = model_pipeline.named_steps["preprocessor"]
model = model_pipeline.named_steps["model"]


# ── Feature Engineering (must mirror Notebook 4 exactly) ────
def duration_to_minutes(hours, minutes):
    return hours * 60 + minutes


def build_feature_row(
    airline,
    source,
    destination,
    total_stops,
    journey_date,
    dep_time,
    arr_time,
    duration_hours,
    duration_minutes,
):
    row = {
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Total_Stops": total_stops,
    "Journey_Day": journey_date.day,
    "Journey_Month": journey_date.month,
    "Journey_Weekday": journey_date.weekday(),  
    "Departure_Hour": dep_time.hour,
    "Departure_Minute": dep_time.minute,
    "Arrival_Hour": arr_time.hour,
    "Arrival_Minute": arr_time.minute,
    "Duration_Minutes": duration_to_minutes(duration_hours, duration_minutes),
}
    return pd.DataFrame([row])


# ── Header ───────────────────────────────────────────────
st.title("✈️ Flight Price Prediction")
st.write(
    "Enter your flight details below to get a predicted ticket price, "
    "along with an explanation of what's driving that price."
)


# ── Input Form ───────────────────────────────────────────
with st.form("flight_form"):

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "Airline",
            [
                "IndiGo", "Air India", "Jet Airways", "SpiceJet",
                "Multiple carriers", "GoAir", "Vistara", "Air Asia",
            ],
        )
        source = st.selectbox(
            "Source",
            ["Delhi", "Kolkata", "Mumbai", "Chennai", "Bangalore"],
        )
        destination = st.selectbox(
            "Destination",
            ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"],
        )
        total_stops = st.selectbox(
            "Total Stops",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: "Non-stop" if x == 0 else f"{x} stop(s)",
        )

    with col2:
        journey_date = st.date_input("Journey Date")
        dep_time = st.time_input("Departure Time", value=time(10, 0))
        arr_time = st.time_input("Arrival Time", value=time(13, 0))
        duration_hours = st.number_input("Duration (hours)", min_value=0, max_value=30, value=2)
        duration_minutes = st.number_input("Duration (minutes)", min_value=0, max_value=59, value=30)

    submitted = st.form_submit_button("Predict Price")


# ── Prediction & Explanation ─────────────────────────────
if submitted:

    if source == destination:
        st.error("Source and Destination cannot be the same. Please correct your input.")
        st.stop()

    input_df = build_feature_row(
        airline, source, destination, total_stops,
        journey_date, dep_time, arr_time,
        duration_hours, duration_minutes,
    )

    predicted_price = model_pipeline.predict(input_df)[0]

    st.success(f"### Predicted Price: ₹{predicted_price:,.0f}")

    # ── SHAP Explanation for this single prediction ──
    with st.spinner("Generating explanation..."):

        input_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()

        input_transformed_df = pd.DataFrame(
            input_transformed, columns=feature_names,
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_transformed_df)

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_transformed_df.values[0],
            feature_names=feature_names,
        )

    st.subheader("Why this price?")

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig, bbox_inches="tight")

    # ── Top 3 contributing factors as plain text ──
    contributions = pd.Series(shap_values[0], index=feature_names)
    top_factors = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    ).head(3)

    st.write("**Top factors influencing this price:**")
    for feature, value in top_factors.items():
        direction = "increased" if value > 0 else "decreased"
        st.write(f"- `{feature}` {direction} the price by ₹{abs(value):,.0f}")


# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption("Built with scikit-learn, XGBoost/LightGBM/CatBoost, and SHAP.")
