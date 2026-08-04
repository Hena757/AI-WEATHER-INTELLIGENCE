"""Professional Streamlit dashboard for the AI Weather Intelligence Platform.

Features:
- Interactive weather prediction with confidence scores
- SHAP explanations (waterfall, force, contribution tables)
- Historical weather charts and trends
- Feature importance visualizations
- Multi-city weather comparison
- Live weather data via OpenWeather API (with Open-Meteo fallback)
- API key management, caching, and error handling
- Modern responsive layout with custom CSS
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on the path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.explainability import (
    explain_prediction,
    load_model_pipeline,
    load_preprocessing_metadata,
)
from src.weather_service import (
    APIKeyError,
    OpenWeatherService,
    WeatherFetchError,
    get_weather_service,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Weather Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for modern responsive design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container */
    .main {
        padding: 0 1.5rem;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90c2 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-container h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .header-container p {
        margin: 0.5rem 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #2d6a9f;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }

    /* Prediction card */
    .prediction-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .prediction-rain {
        background: linear-gradient(135deg, #d32f2f, #e57373);
    }
    .prediction-no-rain {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
    }
    .prediction-card .prediction-label {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .prediction-card .prediction-prob {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    /* Section headers */
    .section-header {
        border-left: 4px solid #2d6a9f;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem;
        color: #1e3a5f;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        padding: 2rem 0 1rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
REPORTS_DIR = BASE_DIR / "reports" / "explanations"
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_weather_dataset.csv"

# City coordinates for live weather (from weather_service)
from src.weather_service import CITY_COORDS

# Feature columns expected by the model
FEATURE_COLUMNS = [
    "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday",
    "TemperatureDifference", "AverageTemperature", "HumidityIndex",
    "PressureDifference", "Month", "Quarter", "DayOfYear", "Season",
    "IsWeekend", "WindIntensityCategory", "RainfallIndicator", "RainTodayBinary",
]

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SEASONS = ["Summer", "Autumn", "Winter", "Spring"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned weather dataset."""
    return pd.read_csv(CLEANED_DATA_PATH, parse_dates=["Date"])


# Initialize the weather service (singleton)
weather_service = get_weather_service()


def fetch_live_weather(city: str) -> dict | None:
    """Fetch live weather data using OpenWeatherService with fallback."""
    try:
        return weather_service.get_city_weather_data(city)
    except Exception:
        return None


def build_input_data(
    location: str,
    min_temp: float,
    max_temp: float,
    rainfall: float,
    humidity9am: float,
    humidity3pm: float,
    pressure9am: float,
    pressure3pm: float,
    wind_speed: float,
    wind_gust_speed: float,
    cloud3pm: float,
    temp3pm: float,
    wind_gust_dir: str = "W",
    wind_dir9am: str = "W",
    wind_dir3pm: str = "W",
    rain_today: str = "No",
    evaporation: float = 5.0,
    sunshine: float = 8.0,
    wind_speed9am: float = 10.0,
    cloud9am: float = 4.0,
    temp9am: float = 15.0,
) -> pd.DataFrame:
    """Build the input DataFrame with all required feature columns."""
    month = datetime.now().month
    quarter = (month - 1) // 3 + 1
    season = SEASONS[(month - 1) // 3]

    input_data = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir9am,
        "WindDir3pm": wind_dir3pm,
        "WindSpeed9am": wind_speed9am,
        "WindSpeed3pm": wind_speed,
        "Humidity9am": humidity9am,
        "Humidity3pm": humidity3pm,
        "Pressure9am": pressure9am,
        "Pressure3pm": pressure3pm,
        "Cloud9am": cloud9am,
        "Cloud3pm": cloud3pm,
        "Temp9am": temp9am,
        "Temp3pm": temp3pm,
        "RainToday": rain_today,
        "TemperatureDifference": max_temp - min_temp,
        "AverageTemperature": (min_temp + max_temp) / 2.0,
        "HumidityIndex": (humidity9am + humidity3pm) / 2.0,
        "PressureDifference": pressure9am - pressure3pm,
        "Month": month,
        "Quarter": quarter,
        "DayOfYear": datetime.now().timetuple().tm_yday,
        "Season": season,
        "IsWeekend": 1 if datetime.now().weekday() >= 5 else 0,
        "WindIntensityCategory": "Low" if wind_speed < 10 else ("Moderate" if wind_speed < 25 else ("High" if wind_speed < 40 else "VeryHigh")),
        "RainfallIndicator": 1 if rainfall > 0 else 0,
        "RainTodayBinary": 1 if rain_today == "Yes" else 0,
    }
    return pd.DataFrame([input_data])


def render_prediction_card(prediction: int, probability: float) -> None:
    """Render a styled prediction result card."""
    if prediction == 1:
        card_class = "prediction-rain"
        label = "🌧️ Rain Expected"
    else:
        card_class = "prediction-no-rain"
        label = "☀️ No Rain Expected"

    st.markdown(
        f"""
        <div class="prediction-card {card_class}">
            <div class="prediction-label">{label}</div>
            <div class="prediction-prob">{probability:.1%}</div>
            <div>Rain probability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(value: str, label: str) -> None:
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <h1>🌦️ AI Weather Intelligence Platform</h1>
        <p>Predict rain probability with explainable AI • Powered by XGBoost + SHAP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select page",
        ["🏠 Dashboard", "🔮 Predict & Explain", "📊 Historical Analysis", "🌍 Multi-City Comparison", "📈 Model Insights"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### ⚙️ Settings")
    st.caption(f"Model: XGBoost Classifier")
    st.caption(f"Data: {load_cleaned_data().shape[0]:,} records")
    st.caption(f"Locations: {load_cleaned_data()['Location'].nunique()} cities")

    st.divider()

    # OpenWeather API Key Management
    st.markdown("### 🔑 OpenWeather API")
    
    # Show API key status
    if weather_service.has_api_key:
        st.success("✅ API key configured")
        st.caption(f"Source: OpenWeather API")
    else:
        st.warning("⚠️ No API key - using Open-Meteo fallback")
        st.caption("Set OPENWEATHER_API_KEY env var or enter a key below")

    # API key input
    api_key_input = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="Enter OpenWeather API key",
        help="Get a free key at openweathermap.org/api",
    )

    if api_key_input and st.button("💾 Save API Key", use_container_width=True):
        weather_service.set_api_key(api_key_input)
        weather_service.save_api_key(api_key_input)
        weather_service.clear_cache()
        st.success("API key saved to .env")
        st.rerun()

    if st.button("🗑️ Clear Weather Cache", use_container_width=True):
        weather_service.clear_cache()
        st.success("Cache cleared")
        st.rerun()

    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        "This dashboard provides transparent, explainable weather predictions "
        "using SHAP (SHapley Additive exPlanations)."
    )

# ---------------------------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown('<div class="section-header"><h2>📊 Weather Overview</h2></div>', unsafe_allow_html=True)

    df = load_cleaned_data()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(f"{len(df):,}", "Total Records")
    with col2:
        render_metric_card(f"{df['Location'].nunique()}", "Cities")
    with col3:
        rain_pct = (df["RainTomorrow"] == "Yes").mean() * 100
        render_metric_card(f"{rain_pct:.1f}%", "Rain Tomorrow Rate")
    with col4:
        avg_temp = df["AverageTemperature"].mean()
        render_metric_card(f"{avg_temp:.1f}°C", "Avg Temperature")

    st.divider()

    # Recent weather trends
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Temperature Trends by City")
        city = st.selectbox("Select city", sorted(df["Location"].unique()), key="dash_city")
        city_df = df[df["Location"] == city].sort_values("Date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=city_df["Date"], y=city_df["MaxTemp"],
            name="Max Temp", line=dict(color="#e74c3c", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=city_df["Date"], y=city_df["MinTemp"],
            name="Min Temp", line=dict(color="#3498db", width=2)
        ))
        fig.update_layout(
            title=f"Temperature Range - {city}",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            height=400,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌧️ Rainfall Distribution")
        rain_df = df[df["Location"] == city].sort_values("Date")
        fig = px.bar(
            rain_df.tail(90),
            x="Date",
            y="Rainfall",
            title=f"Daily Rainfall - {city} (Last 90 days)",
            color_discrete_sequence=["#2e86de"],
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Rainfall (mm)",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Rain probability by month
    st.subheader("📅 Rain Probability by Month")
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    monthly_rain = df.groupby("Month")["RainTomorrow"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
    monthly_rain["Month"] = monthly_rain["Month"].map({
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    })

    fig = px.bar(
        monthly_rain,
        x="Month",
        y="RainTomorrow",
        title="Rain Probability by Month",
        labels={"RainTomorrow": "Rain Probability (%)"},
        color="RainTomorrow",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Rain Probability (%)",
        height=350,
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2: Predict & Explain
# ---------------------------------------------------------------------------
elif page == "🔮 Predict & Explain":
    st.markdown('<div class="section-header"><h2>🔮 Weather Prediction with SHAP Explanation</h2></div>', unsafe_allow_html=True)

    # Input mode selection
    input_mode = st.radio(
        "Input mode",
        ["Manual Input", "Live Weather (API)"],
        horizontal=True,
    )

    if input_mode == "Live Weather (API)":
        source_label = "OpenWeather API" if weather_service.has_api_key else "Open-Meteo API (fallback)"
        st.info(f"🌐 Fetching live weather data from **{source_label}**")

        col1, col2 = st.columns([1, 2])
        with col1:
            live_city = st.selectbox("Select city", list(CITY_COORDS.keys()))
            fetch_btn = st.button("📡 Fetch Live Weather", type="primary", use_container_width=True)

        if fetch_btn:
            with st.spinner(f"Fetching live weather for {live_city}..."):
                weather = fetch_live_weather(live_city)

            if weather:
                # Display live weather metrics (normalized format)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card(f"{weather.get('temperature', 'N/A')}°C", "Current Temp")
                with col2:
                    render_metric_card(f"{weather.get('humidity', 'N/A')}%", "Humidity")
                with col3:
                    render_metric_card(f"{weather.get('precipitation', 0)}mm", "Precipitation")
                with col4:
                    render_metric_card(f"{weather.get('pressure', 'N/A')}hPa", "Pressure")

                # Show weather description if available
                if weather.get("description"):
                    st.markdown(f"**Conditions:** {weather['description'].title()}")

                # Build model input from normalized live data
                input_df = weather_service.build_model_input(weather)

                st.session_state["input_df"] = input_df
                st.session_state["input_city"] = live_city
                st.success(f"✅ Live weather data fetched for {live_city}")

                # Show forecast if available
                daily = weather.get("daily", [])
                if daily:
                    st.subheader("📅 Weather Forecast")
                    forecast_df = pd.DataFrame(daily)
                    # Rename columns for display
                    col_map = {
                        "date": "Date",
                        "temp_max": "Max Temp (°C)",
                        "temp_min": "Min Temp (°C)",
                        "precipitation": "Precipitation (mm)",
                        "humidity": "Humidity (%)",
                        "wind_speed": "Wind Speed (km/h)",
                        "description": "Conditions",
                    }
                    forecast_df = forecast_df.rename(columns=col_map)
                    display_cols = [c for c in col_map.values() if c in forecast_df.columns]
                    st.dataframe(forecast_df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.error("Failed to fetch live weather data. Please try manual input.")
        else:
            st.info("Click 'Fetch Live Weather' to get current conditions and make a prediction.")

    else:
        st.subheader("🌡️ Weather Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            location = st.selectbox("📍 Location", sorted(load_cleaned_data()["Location"].unique()))
            min_temp = st.number_input("Min Temperature (°C)", value=12.0, step=0.5)
            max_temp = st.number_input("Max Temperature (°C)", value=22.0, step=0.5)
            rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.1)

        with col2:
            humidity9am = st.number_input("Humidity 9am (%)", value=70.0, step=1.0)
            humidity3pm = st.number_input("Humidity 3pm (%)", value=55.0, step=1.0)
            pressure9am = st.number_input("Pressure 9am (hPa)", value=1015.0, step=0.1)
            pressure3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0, step=0.1)

        with col3:
            wind_speed = st.number_input("Wind Speed 3pm (km/h)", value=15.0, step=1.0)
            wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", value=30.0, step=1.0)
            cloud3pm = st.number_input("Cloud 3pm (oktas)", value=5.0, step=1.0)
            temp3pm = st.number_input("Temp 3pm (°C)", value=20.0, step=0.5)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            wind_gust_dir = st.selectbox("Wind Gust Direction", WIND_DIRS, index=14)
            wind_dir9am = st.selectbox("Wind Direction 9am", WIND_DIRS, index=14)
        with col2:
            wind_dir3pm = st.selectbox("Wind Direction 3pm", WIND_DIRS, index=14)
            rain_today = st.selectbox("Rain Today", ["No", "Yes"])

        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
            input_df = build_input_data(
                location=location,
                min_temp=min_temp,
                max_temp=max_temp,
                rainfall=rainfall,
                humidity9am=humidity9am,
                humidity3pm=humidity3pm,
                pressure9am=pressure9am,
                pressure3pm=pressure3pm,
                wind_speed=wind_speed,
                wind_gust_speed=wind_gust_speed,
                cloud3pm=cloud3pm,
                temp3pm=temp3pm,
                wind_gust_dir=wind_gust_dir,
                wind_dir9am=wind_dir9am,
                wind_dir3pm=wind_dir3pm,
                rain_today=rain_today,
            )
            st.session_state["input_df"] = input_df
            st.session_state["input_city"] = location

    # Make prediction if input is available
    if "input_df" in st.session_state:
        input_df = st.session_state["input_df"]
        input_city = st.session_state.get("input_city", "Sydney")

        try:
            with st.spinner("Computing prediction and SHAP explanation..."):
                result = explain_prediction(
                    input_df,
                    model_path=MODEL_PATH,
                    output_dir=REPORTS_DIR,
                    background_size=100,
                    max_display=20,
                )

            st.divider()

            # Prediction result
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("🎯 Prediction Result")
                render_prediction_card(result["prediction"], result["probability"])

                # Confidence metrics
                st.divider()
                st.markdown("### 📊 Confidence Metrics")
                render_metric_card(f"{result['probability']:.1%}", "Rain Probability")
                st.markdown("")
                render_metric_card(f"{1 - result['probability']:.1%}", "No Rain Probability")

            with col2:
                st.subheader("📊 Feature Contributions")
                contrib_df = result["contribution_table"].head(10)

                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=contrib_df["shap_value"],
                    y=contrib_df["feature"],
                    orientation="h",
                    marker_color=["#e74c3c" if v >= 0 else "#3498db" for v in contrib_df["shap_value"]],
                ))
                fig.update_layout(
                    title="Top 10 Feature Contributions (SHAP Values)",
                    xaxis_title="SHAP Value",
                    yaxis_title="",
                    height=400,
                    template="plotly_white",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # SHAP visualizations
            col1, col2 = st.columns(2)

            with col1:
                waterfall_path = result["artifacts"]["waterfall_plot"]
                if waterfall_path.exists():
                    st.subheader("💧 Waterfall Plot")
                    st.image(str(waterfall_path), use_container_width=True)
                    st.caption("How each feature contributed to this prediction")

            with col2:
                force_path = result["artifacts"]["force_plot"]
                if force_path.exists():
                    st.subheader("⚡ Force Plot")
                    st.image(str(force_path), use_container_width=True)
                    st.caption("Red pushes toward rain, blue pushes away")

            st.divider()

            # Full contribution table
            st.subheader("📋 Full Feature Contribution Table")
            display_cols = ["feature", "shap_value", "abs_shap_value", "direction"]
            st.dataframe(
                result["contribution_table"][display_cols],
                use_container_width=True,
                height=400,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"An error occurred: {exc}")

# ---------------------------------------------------------------------------
# Page 3: Historical Analysis
# ---------------------------------------------------------------------------
elif page == "📊 Historical Analysis":
    st.markdown('<div class="section-header"><h2>📊 Historical Weather Analysis</h2></div>', unsafe_allow_html=True)

    df = load_cleaned_data()
    df["Date"] = pd.to_datetime(df["Date"])

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cities = st.multiselect("Select cities", sorted(df["Location"].unique()), default=["Sydney", "Melbourne"])
    with col2:
        start_date = st.date_input("Start date", df["Date"].min().date())
    with col3:
        end_date = st.date_input("End date", df["Date"].max().date())

    filtered = df[
        (df["Location"].isin(cities)) &
        (df["Date"] >= pd.Timestamp(start_date)) &
        (df["Date"] <= pd.Timestamp(end_date))
    ]

    if filtered.empty:
        st.warning("No data found for the selected filters.")
    else:
        # Temperature trends
        st.subheader("🌡️ Temperature Trends")
        fig = go.Figure()
        for city in cities:
            city_df = filtered[filtered["Location"] == city]
            fig.add_trace(go.Scatter(
                x=city_df["Date"], y=city_df["AverageTemperature"],
                name=city, mode="lines"
            ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Temperature (°C)",
            height=400,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rainfall comparison
        st.subheader("🌧️ Rainfall Comparison")
        fig = px.box(
            filtered,
            x="Location",
            y="Rainfall",
            color="Location",
            title="Rainfall Distribution by City",
        )
        fig.update_layout(
            xaxis_title="City",
            yaxis_title="Rainfall (mm)",
            height=400,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Humidity vs Temperature scatter
        st.subheader("💧 Humidity vs Temperature")
        fig = px.scatter(
            filtered.sample(min(5000, len(filtered))),
            x="AverageTemperature",
            y="Humidity3pm",
            color="Location",
            opacity=0.6,
            title="Humidity vs Temperature Relationship",
        )
        fig.update_layout(
            xaxis_title="Average Temperature (°C)",
            yaxis_title="Humidity 3pm (%)",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("📋 Summary Statistics")
        summary = filtered.groupby("Location").agg({
            "MinTemp": "mean",
            "MaxTemp": "mean",
            "Rainfall": "mean",
            "Humidity3pm": "mean",
            "Pressure3pm": "mean",
            "WindSpeed3pm": "mean",
        }).round(2)
        summary.columns = ["Min Temp (°C)", "Max Temp (°C)", "Rainfall (mm)", "Humidity 3pm (%)", "Pressure 3pm (hPa)", "Wind Speed 3pm (km/h)"]
        st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 4: Multi-City Comparison
# ---------------------------------------------------------------------------
elif page == "🌍 Multi-City Comparison":
    st.markdown('<div class="section-header"><h2>🌍 Multi-City Weather Comparison</h2></div>', unsafe_allow_html=True)

    source_label = "OpenWeather API" if weather_service.has_api_key else "Open-Meteo API (fallback)"
    st.info(f"Compare weather predictions across multiple cities using live data from **{source_label}**")

    cities = st.multiselect(
        "Select cities to compare",
        list(CITY_COORDS.keys()),
        default=["Sydney", "Melbourne", "Brisbane"],
    )

    if st.button("📡 Fetch & Compare", type="primary", use_container_width=True):
        results = []
        with st.spinner("Fetching live weather data for all cities..."):
            for city in cities:
                weather = fetch_live_weather(city)
                if weather:
                    temp = weather.get("temperature", 20)
                    temp_min = weather.get("temp_min", temp - 5)
                    temp_max = weather.get("temp_max", temp + 5)
                    rainfall = weather.get("precipitation", 0)
                    humidity = weather.get("humidity", 60)
                    wind_speed = weather.get("wind_speed", 15)

                    input_df = weather_service.build_model_input(weather)

                    try:
                        result = explain_prediction(
                            input_df,
                            model_path=MODEL_PATH,
                            output_dir=REPORTS_DIR,
                            background_size=50,
                            max_display=10,
                        )
                        results.append({
                            "City": city,
                            "Temperature": f"{max_temp:.1f}°C",
                            "Humidity": f"{humidity:.0f}%",
                            "Rainfall": f"{rainfall:.1f}mm",
                            "Wind Speed": f"{wind_speed:.0f} km/h",
                            "Rain Probability": f"{result['probability']:.1%}",
                            "Prediction": result["prediction_label"],
                        })
                    except Exception:
                        results.append({
                            "City": city,
                            "Temperature": f"{max_temp:.1f}°C",
                            "Humidity": f"{humidity:.0f}%",
                            "Rainfall": f"{rainfall:.1f}mm",
                            "Wind Speed": f"{wind_speed:.0f} km/h",
                            "Rain Probability": "N/A",
                            "Prediction": "Error",
                        })

        if results:
            results_df = pd.DataFrame(results)

            # Comparison table
            st.subheader("📋 City Comparison Table")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Rain probability comparison chart
            st.subheader("🌧️ Rain Probability Comparison")
            fig = go.Figure(go.Bar(
                x=results_df["City"],
                y=[float(p.strip("%")) for p in results_df["Rain Probability"]] if results_df["Rain Probability"].iloc[0] != "N/A" else [0] * len(results_df),
                marker_color=["#e74c3c" if p == "Rain" else "#2ecc71" for p in results_df["Prediction"]],
                text=[f"{p}" for p in results_df["Rain Probability"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Rain Probability by City",
                xaxis_title="City",
                yaxis_title="Rain Probability (%)",
                height=400,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Temperature comparison
            st.subheader("🌡️ Temperature Comparison")
            fig = go.Figure(go.Bar(
                x=results_df["City"],
                y=[float(t.replace("°C", "")) for t in results_df["Temperature"]],
                marker_color="#3498db",
                text=[t for t in results_df["Temperature"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Current Temperature by City",
                xaxis_title="City",
                yaxis_title="Temperature (°C)",
                height=400,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 5: Model Insights
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="section-header"><h2>📈 Model Insights & Feature Importance</h2></div>', unsafe_allow_html=True)

    # Model performance metrics
    st.subheader("🎯 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("0.8911", "ROC-AUC")
    with col2:
        render_metric_card("85.78%", "Accuracy")
    with col3:
        render_metric_card("0.6295", "F1 Score")
    with col4:
        render_metric_card("XGBoost", "Best Model")

    st.divider()

    # Global SHAP explanations
    st.subheader("🌍 Global SHAP Explanations")

    col1, col2 = st.columns(2)

    with col1:
        summary_path = REPORTS_DIR / "shap_summary_plot.png"
        if summary_path.exists():
            st.image(str(summary_path), use_container_width=True)
            st.caption("SHAP Summary (Beeswarm) Plot - shows feature impact across all predictions")
        else:
            st.info("Summary plot not found. Run scripts/generate_explanations.py")

    with col2:
        importance_path = REPORTS_DIR / "shap_global_importance.png"
        if importance_path.exists():
            st.image(str(importance_path), use_container_width=True)
            st.caption("Global Feature Importance - mean absolute SHAP values")
        else:
            st.info("Global importance plot not found.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        dependence_path = REPORTS_DIR / "shap_dependence_plot.png"
        if dependence_path.exists():
            st.subheader("📈 Dependence Plot")
            st.image(str(dependence_path), use_container_width=True)
            st.caption("Shows how the top feature's SHAP value changes with its value")
        else:
            st.info("Dependence plot not found.")

    with col2:
        # Feature importance from contributions
        contrib_path = REPORTS_DIR / "shap_contributions.csv"
        if contrib_path.exists():
            st.subheader("📊 Top Features by Mean |SHAP|")
            contrib_df = pd.read_csv(contrib_path)
            top_features = contrib_df.groupby("feature")["abs_shap_value"].mean().sort_values(ascending=False).head(15)

            fig = go.Figure(go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation="h",
                marker_color="#2d6a9f",
            ))
            fig.update_layout(
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="",
                height=400,
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Model comparison
    st.subheader("📊 Model Comparison")
    comparison_path = BASE_DIR / "models" / "model_comparison_results.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df["model"],
            y=comparison_df["roc_auc"],
            name="ROC-AUC",
            marker_color="#2d6a9f",
        ))
        fig.add_trace(go.Bar(
            x=comparison_df["model"],
            y=comparison_df["accuracy"],
            name="Accuracy",
            marker_color="#27ae60",
        ))
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        AI Weather Intelligence Platform • Powered by XGBoost, SHAP, and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)