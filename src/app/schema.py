import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

class IngestRequest(BaseModel):
    lat: float
    lon: float
    dt_iso: str
    user_query: str

class WeatherDataPoint(BaseModel):
    timestamp: datetime
    temperature_2m: Optional[float]       # °C
    relative_humidity_2m: Optional[float] # %
    rain: Optional[float]                 # mm
    wind_speed_100m: Optional[float]      # km/h
    wind_direction_100m: Optional[float]  # °
    pressure_msl: Optional[float]         # hPa
    surface_pressure: Optional[float]     # hPa
    pm2_5: Optional[float] = None         # μg/m³


class WeatherPayload(BaseModel):
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_timestamp: datetime
    source: str  # "redis" or "openmeteo"
    weather: List[WeatherDataPoint]
    raw_response: Optional[Dict[str, Any]] = None

# Default values for the features defined in WeatherDataPoint (used as model inputs)
DEFAULT_WEATHER_FEATURES = {
    "temperature_2m": 13.0,       # °C
    "relative_humidity_2m": 75.0, # %
    "rain": 0.0,                  # mm
    "wind_speed_100m": 17.0,      # km/h
    "wind_direction_100m": 105.0, # °
    "pressure_msl": 1016.0,       # hPa
    "surface_pressure": 988.0,    # hPa
    "pm2_5": 364.0,               # Default pm2_5 must be here as it's defined in WeatherDataPoint
}

# Default values for the pollutant OUTPUTS from the model (used downstream)
DEFAULT_POLLUTANT_OUTPUTS = {
    "co": 2616.0,
    "no": 2.0,
    "no2": 70.0,
    "o3": 13.0,
    "so2": 38.0,
    "pm10": 411.0,
    "nh3": 28.0
}

# The name used by other modules (IngestionAgent, openmeteo) for weather defaults
DEFAULT_WEATHER = DEFAULT_WEATHER_FEATURES

# Log a warning if DEFAULT_WEATHER_FEATURES is missing a key expected by WeatherDataPoint
if not all(k in DEFAULT_WEATHER_FEATURES for k in WeatherDataPoint.__annotations__.keys() if k != 'timestamp'):
    log.warning("DEFAULT_WEATHER_FEATURES is missing some keys defined in WeatherDataPoint!")
