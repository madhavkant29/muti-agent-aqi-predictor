import openmeteo_requests
from openmeteo_sdk.Variable import Variable
from datetime import datetime
import pandas as pd
from typing import Any
from src.app.schema import WeatherDataPoint, WeatherPayload, DEFAULT_WEATHER
import logging

log = logging.getLogger(__name__)

# Map Open-Meteo variable names to model feature names
VAR_MAP = {
    Variable.temperature: "temperature_2m",
    Variable.relative_humidity: "relative_humidity_2m",
    Variable.precipitation: "rain",
    Variable.wind_speed: "wind_speed_100m",
    Variable.wind_direction: "wind_direction_100m",
    Variable.pressure_msl: "pressure_msl",
    Variable.surface_pressure: "surface_pressure"
}

# For wind speed conversion m/s -> km/h
WIND_SPEED_CONV = 3.6

def _get_variable_value(hourly_data: Any, var_id: Variable, altitude: int = None, member: int = 0) -> float:
    """
    Helper to extract the value for a specific variable, altitude, and ensemble member 
    from the Open-Meteo Hourly response structure, applying unit conversions.
    Returns the default value if the variable is not found or is None.
    """
    default_name = VAR_MAP.get(var_id)
    # Get default from Pydantic schema defaults
    default = DEFAULT_WEATHER.get(default_name, 0.0)
    
    for i in range(hourly_data.VariablesLength()):
        v = hourly_data.Variables(i)
        
        # Check if the variable matches the ID, optional altitude, and ensemble member (0 for mean)
        is_match = (
            v.Variable() == var_id and 
            (altitude is None or v.Altitude() == altitude) and 
            v.EnsembleMember() == member
        )

        if is_match:
            # We only care about the first hour (index 0) of the forecast
            val = v.ValuesAsNumpy()[0] 
            
            if val is None:
                return default
            
            # Convert units if wind speed (m/s -> km/h)
            if default_name == "wind_speed_100m":
                return float(val) * WIND_SPEED_CONV
                
            return float(val)
            
    # If the loop completes without finding the variable, return the default
    log.warning("Variable %s (alt: %s) not found in Open-Meteo response. Using default: %s", default_name, altitude, default)
    return default

def fetch_openmeteo_hour(lat: float, lon: float, dt_hour: datetime) -> WeatherPayload:
    """
    Fetches hourly weather from Open-Meteo Ensemble API and normalizes it
    to the exact feature names and units expected by MLP/DML models.
    Missing values are filled from DEFAULT_WEATHER.
    """
    client = openmeteo_requests.Client()
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    
    # Get the list of required string variable names by taking the VALUES of VAR_MAP.
    hourly_vars = list(VAR_MAP.values())
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        # Use gfs_seamless for the best hourly data coverage
        "models": "gfs_seamless" 
    }

    try:
        responses = client.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
    except Exception as e:
        log.error(f"Failed to fetch ensemble data: {e}")
        # Return default WeatherPayload if API fails
        weather_point = WeatherDataPoint(timestamp=dt_hour, **DEFAULT_WEATHER)
        return WeatherPayload(
            source_timestamp=dt_hour,
            source="openmeteo_ensemble_fallback",
            weather=[weather_point],
            raw_response={}
        )

    # Convert times from Unix timestamp (s) to pandas datetime object(s)
    times = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    
    # FIX: Check if 'times' is a Series (multiple points) or a single Timestamp object.
    if isinstance(times, pd.Timestamp):
        # If it's a single Timestamp (not subscriptable), use it directly
        forecast_time = times.to_pydatetime()
    elif len(times) > 0:
        # If it's a Series/array, take the first element (the time requested)
        forecast_time = times[0].to_pydatetime()
    else:
        # Fallback if no times were returned
        forecast_time = dt_hour
        log.warning("No forecast time returned from Open-Meteo. Using requested time.")


    # Build WeatherDataPoint using the new helper function
    weather_point = WeatherDataPoint(
        timestamp=forecast_time,
        temperature_2m=_get_variable_value(hourly, Variable.temperature, altitude=2),
        relative_humidity_2m=_get_variable_value(hourly, Variable.relative_humidity, altitude=2),
        rain=_get_variable_value(hourly, Variable.precipitation),
        wind_speed_100m=_get_variable_value(hourly, Variable.wind_speed, altitude=100),
        wind_direction_100m=_get_variable_value(hourly, Variable.wind_direction, altitude=100),
        pressure_msl=_get_variable_value(hourly, Variable.pressure_msl),
        surface_pressure=_get_variable_value(hourly, Variable.surface_pressure),
        # Assuming pm2_5 is not available via ensemble API and uses default
        pm2_5=DEFAULT_WEATHER["pm2_5"] 
    )

    return WeatherPayload(
        source_timestamp=dt_hour,
        source="openmeteo_ensemble",
        weather=[weather_point],
        raw_response={}
    )
