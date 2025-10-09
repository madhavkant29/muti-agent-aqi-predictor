import openmeteo_requests
from openmeteo_sdk.Variable import Variable
import pandas as pd

# Initialize client (direct, no cache, no retry)
openmeteo = openmeteo_requests.Client()

# API endpoint
url = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Request GFS Seamless ensemble forecast
params = {
    "latitude": 28.61,      # Example: New Delhi
    "longitude": 77.20,
    "hourly": ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "wind_speed_100m",
        "wind_direction_100m",
        "pressure_msl",
        "surface_pressure"
    ]),
    "models": "gfs_seamless"
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# Extract hourly data
hourly = response.Hourly()

# Pick the first forecasted hour (index = 0)
hour_index = 0
forecast_time = pd.to_datetime(hourly.Time(), unit="s", utc=True) + pd.to_timedelta(hour_index, "h")

# Helper to get a value for a variable
def get_variable_value(hourly, var_id, altitude=None, member=0):
    for i in range(hourly.VariablesLength()):
        v = hourly.Variables(i)
        if v.Variable() == var_id and (altitude is None or v.Altitude() == altitude) and v.EnsembleMember() == member:
            return v.ValuesAsNumpy()[hour_index]
    return None

# Collect variables (convert wind speed from m/s → km/h)
result = {
    "time": str(forecast_time),
    "temperature_2m_C": get_variable_value(hourly, Variable.temperature, altitude=2),
    "relative_humidity_2m_%": get_variable_value(hourly, Variable.relative_humidity, altitude=2),
    "rain_mm": get_variable_value(hourly, Variable.precipitation),
    "wind_speed_100m_kmh": get_variable_value(hourly, Variable.wind_speed, altitude=100) * 3.6,
    "wind_direction_100m_deg": get_variable_value(hourly, Variable.wind_direction, altitude=100),
    "pressure_msl_hPa": get_variable_value(hourly, Variable.pressure_msl),
    "surface_pressure_hPa": get_variable_value(hourly, Variable.surface_pressure)
}

print(result)



import sys
import os
from llama_cpp import Llama
from contextlib import contextmanager
from datetime import datetime

# --- Context manager to suppress stderr ---
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# --- Load context from file ---
context_path = r"C:\Users\madha\.vscode\python\aqi project\src\context_corpus.txt"
with open(context_path, encoding="utf-8") as f:
    context = f.read()

# --- Load MLP and DML outputs ---
# Example: Replace with real calls to your agents
mlp_output = {
    "co": 2000.0,
    "no": 1.5,
    "no2": 50.0,
    "o3": 10.0,
    "so2": 30.0,
    "pm2_5": 120.0,
    "pm10": 180.0,
    "nh3": 20.0
}

dml_output = {
    "temperature_2m": {"co": 5.0, "no": 0.1, "no2": 0.2, "o3": -0.1, "so2": 0.3, "pm2_5": 1.0, "pm10": 1.5, "nh3": 0.2},
    "rain": {"co": -2.0, "no": -0.05, "no2": -0.1, "o3": 0.0, "so2": -0.2, "pm2_5": -0.5, "pm10": -0.7, "nh3": -0.1},
    # Add all other treatments...
}

# Format outputs for LLM context
mlp_str = "\n".join([f"{k}: {v:.2f}" for k, v in mlp_output.items()])
dml_str = "\n".join([
    f"{treatment} -> " + ", ".join([f"{pollutant}: {effect:.2f}" for pollutant, effect in effects.items()])
    for treatment, effects in dml_output.items()
])

# --- Initialize model ---
with suppress_output():
    llm = Llama(
        model_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'),
        n_ctx=4096,       
        n_gpu_layers=32,
        n_threads=4,
        n_batch=16
    )

# --- User prompt ---
user_prompt = "Should I wear a mask today? It's 150 AQI in my area."
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Build full prompt with MLP & DML outputs ---
full_prompt = f"""[INST] You are an AQI assistant. Answer **directly**. Include:
- AQI category
- Health risks for vulnerable groups
- Practical precautions (mask type, outdoor activity limits)
- Short-term mitigation tips
- Predictions for this week (current month/week)
- Seasonal tips (e.g., monsoon cleans air, summer dust increases, etc.)
Keep it concise, clear, and practical.

Current timestamp: {current_time}
AQI data (weekly/monthly): {context}

MLP model predictions (current pollutants levels):
{mlp_str}

DML model treatment effects (causal impact of weather on pollutants):
{dml_str}

User query: {user_prompt}
[/INST]"""

# --- Run inference ---
output = llm(
    full_prompt,
    max_tokens=800,
    stop=["[/INST]"], 
    temperature=0.7,
    top_p=0.9
)

# --- Print clean output ---
response_text = output["choices"][0]["text"].strip()
print("\n--- AQI Assistant Response ---\n")
print(response_text)
print("\n-------------------------------\n")

