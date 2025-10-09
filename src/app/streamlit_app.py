import streamlit as st
import requests
from datetime import datetime, date, time, timedelta

# Define the FastAPI service URL
# NOTE: Ensure your main.py service is running on this address.
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AQI Assistant", layout="wide")
st.title("🌫️ AQI Prediction Assistant")
st.markdown("Enter your question and select a **future date/time** for the forecast.")

# --- Fixed Location (Delhi) ---
LAT = 28.6139
LON = 77.2090
st.info(f"📍 Forecasting for Delhi, India (Lat: {LAT}, Lon: {LON})")

# --- Dynamic Date and Time Input ---
col1, col2 = st.columns(2)

with col1:
    prediction_date = st.date_input(
        "Select Prediction Date",
        # Default to tomorrow
        value=date.today() + timedelta(days=1)
    )

with col2:
    prediction_time = st.time_input(
        "Select Prediction Time (IST)",
        value=time(12, 0) # Default to noon
    )

# --- User Query ---
user_query = st.text_area(
    "Your question/query about air quality:",
    value="What will the AQI be like next Friday at 9 AM? Is it safe to visit the market?",
    height=150
)

# Combine date and time into a datetime object
prediction_dt = datetime.combine(prediction_date, prediction_time)
# Format to ISO 8601 string (e.g., "2025-10-02T12:00:00")
dt_iso_str = prediction_dt.isoformat()

st.markdown(f"**Selected Forecast Time (ISO):** `{dt_iso_str}`")

submit = st.button("🚀 Get AQI Advice", type="primary")

if submit:
    # --- Prepare Payload ---
    payload = {
        "lat": LAT,
        "lon": LON,
        "dt_iso": dt_iso_str,
        "user_query": user_query
    }

    with st.spinner(f"Fetching predictions for {prediction_dt.strftime('%Y-%m-%d %H:%M')}..."):
        try:
            # --- Call FastAPI Backend ---
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()

            st.success("✅ Advice fetched successfully!")
            
            st.divider()
            st.subheader("🤖 LLM Advice")
            
            # Display the LLM's response using markdown for proper formatting
            advice_text = data.get("advice", "No advice returned. Check the server logs for errors.")
            st.markdown(advice_text)

            st.divider()
            
            # Show raw data for debugging purposes
            with st.expander("Show Raw API Response"):
                 st.json(data)

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection Error: Could not connect to the FastAPI server at {API_URL}. Please ensure your 'main.py' service is running locally.")
        except requests.exceptions.HTTPError as e:
            # Handle errors returned by FastAPI (e.g., 500 from a pipeline failure)
            error_details = e.response.json().get('detail', 'No details available.')
            st.error(f"❌ API Error ({e.response.status_code}): Pipeline execution failed. Details: {error_details}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ An unexpected error occurred: {e}")