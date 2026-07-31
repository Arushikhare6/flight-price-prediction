import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import time, datetime, timedelta

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Predictor & AI Explainer",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load Model (cached) ──────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("models") / "final_tuned_model.pkl"
    if not model_path.exists():
        model_path = Path("../models") / "final_tuned_model.pkl"
    return joblib.load(model_path)


# ── Load SHAP Explainer (cached) ──────────────────────────
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    model_pipeline = load_model()
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    explainer = get_shap_explainer(model)

    # Dynamic model name resolution
    raw_model_name = type(model).__name__
    model_display_names = {
        "RandomForestRegressor": "Tuned Random Forest Regressor",
        "XGBRegressor": "Tuned XGBoost Regressor",
        "LGBMRegressor": "Tuned LightGBM Regressor",
        "CatBoostRegressor": "Tuned CatBoost Regressor",
    }
    active_model_name = model_display_names.get(raw_model_name, raw_model_name)

except Exception as e:
    st.error(f"Error loading trained machine learning model: {e}")
    st.stop()


# ── Helper Functions ─────────────────────────────────────
def duration_to_minutes(hours, minutes):
    return int(hours) * 60 + int(minutes)


def calculate_arrival_time(dep_time, duration_hours, duration_minutes):
    """Automatically calculate arrival time based on departure time and flight duration."""
    dummy_dt = datetime(2026, 1, 1, dep_time.hour, dep_time.minute)
    arr_dt = dummy_dt + timedelta(hours=int(duration_hours), minutes=int(duration_minutes))
    return arr_dt.time()


def build_feature_row(
    airline,
    source,
    destination,
    total_stops,
    journey_date,
    dep_time,
    duration_hours,
    duration_minutes,
):
    arr_time = calculate_arrival_time(dep_time, duration_hours, duration_minutes)

    row = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Total_Stops": int(total_stops),
        "Journey_Day": journey_date.day,
        "Journey_Month": journey_date.month,
        "Journey_Weekday": journey_date.weekday(),
        "Departure_Hour": dep_time.hour,
        "Departure_Minute": dep_time.minute,
        "Arrival_Hour": arr_time.hour,
        "Arrival_Minute": arr_time.minute,
        "Duration_Minutes": duration_to_minutes(duration_hours, duration_minutes),
    }
    return pd.DataFrame([row]), arr_time


# ── Header ───────────────────────────────────────────────
st.title("✈️ Flight Price Predictor")
st.markdown(
    f"Enter your travel details below to estimate ticket prices powered by **{active_model_name}** and **SHAP AI Explainability**."
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📌 Model Benchmark")
    st.markdown(f"""
    - **Algorithm:** {active_model_name}
    - **Dataset:** 10,680+ Flight Records
    - **Final Test R² Accuracy:** `0.815` (81.5%)
    - **Final Test MAE:** `₹1,169`
    - **Explainability:** SHAP TreeExplainer
    """)
    
    st.markdown("---")
    st.subheader("💡 Travel Insights")
    st.info("Direct non-stop flights and business class options heavily impact ticket pricing. Weekend travel also exhibits higher demand curves.")


# ── Flight Search Input Form ──────────────────────────────
with st.form("flight_form", clear_on_submit=False):

    st.subheader("1️⃣ Route & Carrier")
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "Airline Carrier",
            options=[
                "IndiGo",
                "Air India",
                "Jet Airways",
                "SpiceJet",
                "Multiple carriers",
                "GoAir",
                "Vistara",
                "Air Asia",
                "Multiple carriers Premium economy",
                "Jet Airways Business",
                "Vistara Premium economy",
                "Trujet",
            ],
            index=0,
        )

        source_options = ["Delhi", "Kolkata", "Banglore", "Mumbai", "Chennai"]
        source_display = {
            "Delhi": "Delhi (DEL)",
            "Kolkata": "Kolkata (CCU)",
            "Banglore": "Bangalore (BLR)",
            "Mumbai": "Mumbai (BOM)",
            "Chennai": "Chennai (MAA)",
        }

        source = st.selectbox(
            "Departure City (Source)",
            options=source_options,
            format_func=lambda x: source_display.get(x, x),
            index=0,
        )

    with col2:
        total_stops = st.selectbox(
            "Flight Layovers",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: "Non-stop" if x == 0 else f"{x} Stop(s)",
            index=1,
        )

        destination_options = ["Cochin", "Banglore", "Delhi", "New Delhi", "Hyderabad", "Kolkata"]
        destination_display = {
            "Cochin": "Cochin (COK)",
            "Banglore": "Bangalore (BLR)",
            "Delhi": "Delhi (DEL)",
            "New Delhi": "New Delhi (DEL)",
            "Hyderabad": "Hyderabad (HYD)",
            "Kolkata": "Kolkata (CCU)",
        }

        destination = st.selectbox(
            "Arrival City (Destination)",
            options=destination_options,
            format_func=lambda x: destination_display.get(x, x),
            index=0,
        )

    st.subheader("2️⃣ Schedule & Duration")
    sc1, sc2 = st.columns(2)

    with sc1:
        today = datetime.today()
        max_booking_date = today + timedelta(days=365)
        journey_date = st.date_input("Date of Departure", min_value=today, max_value=max_booking_date)
        dep_time = st.time_input("Departure Time", value=time(10, 0))

    with sc2:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=30, value=2, step=1)
        with d_col2:
            duration_minutes = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=30, step=1)

        # Compute preview arrival time for user
        calc_arr = calculate_arrival_time(dep_time, duration_hours, duration_minutes)
        st.caption(f"🕒 **Estimated Arrival Time:** `{calc_arr.strftime('%I:%M %p')}`")

    submitted = st.form_submit_button("💰 Estimate Ticket Fare", use_container_width=True)


# ── Prediction & Explanation Output ──────────────────────
if submitted:

    # 1. Validation guards
    if source == destination:
        st.error("⚠️ Source and Destination cities cannot be identical. Please select different cities.")
        st.stop()

    if duration_hours == 0 and duration_minutes == 0:
        st.error("⚠️ Flight duration must be greater than 0 minutes.")
        st.stop()

    # 2. Build feature DataFrame
    input_df, calculated_arr_time = build_feature_row(
        airline,
        source,
        destination,
        total_stops,
        journey_date,
        dep_time,
        duration_hours,
        duration_minutes,
    )

    # 3. Model Prediction
    try:
        predicted_price = model_pipeline.predict(input_df)[0]
        predicted_price = max(0, float(predicted_price))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # 4. Display Price Result Metric
    st.markdown("---")
    st.subheader("🎉 Prediction Result")

    src_label = source_display.get(source, source).split("(")[0].strip()
    dest_label = destination_display.get(destination, destination).split("(")[0].strip()

    st.metric(
        label=f"Estimated Fare ({airline} • {src_label} ➔ {dest_label})",
        value=f"₹{predicted_price:,.0f}",
        delta=f"Dep: {dep_time.strftime('%I:%M %p')} | Arr: {calculated_arr_time.strftime('%I:%M %p')} | Duration: {duration_hours}h {duration_minutes}m",
        delta_color="off",
    )

    # 5. SHAP AI Explainability Section
    st.markdown("---")
    st.subheader("🔍 AI Fare Breakdown & Explanation")
    st.caption("Here is how your selected flight specifications influenced the base fare price:")

    with st.spinner("Calculating SHAP feature attributions..."):
        try:
            input_transformed = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()

            input_transformed_df = pd.DataFrame(
                input_transformed,
                columns=feature_names,
            )

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

            # Highlighting Top Influencing Factors with native Streamlit metric columns
            contributions = pd.Series(shap_values[0], index=feature_names)
            top_factors = contributions.reindex(
                contributions.abs().sort_values(ascending=False).index
            ).head(4)

            cols = st.columns(len(top_factors))
            for idx, (feature, val) in enumerate(top_factors.items()):
                clean_feat = feature.replace("categorical__", "").replace("remainder__", "")
                sign_str = f"+₹{val:,.0f}" if val > 0 else f"-₹{abs(val):,.0f}"
                # In financial context, price increase = red/inverse, price decrease = green/normal
                cols[idx].metric(
                    label=clean_feat,
                    value=sign_str,
                    delta="Increased Fare" if val > 0 else "Decreased Fare",
                    delta_color="inverse" if val > 0 else "normal",
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Render SHAP Waterfall Plot
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            shap.plots.waterfall(explanation, show=False)
            plt.title("SHAP Waterfall Feature Contribution Chart", fontsize=11, fontweight="bold", pad=12)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)

        except Exception as err:
            st.warning(f"Could not render SHAP explanation chart: {err}")

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption(f"✈️ Flight Price Prediction System • Built with Streamlit, scikit-learn, {active_model_name} & SHAP")
