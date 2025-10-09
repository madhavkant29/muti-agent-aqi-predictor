import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from src.app.schema import IngestRequest, WeatherPayload, WeatherDataPoint
from src.app.utils import ymdh_from_iso
from src.app.clients import redis_client
from src.app.clients.openmeteo import fetch_openmeteo_hour
import asyncio

log = logging.getLogger(__name__)

class IngestionAgent:
    @staticmethod
    def _extract_features(w: WeatherDataPoint) -> Dict[str, Any]:
        """
        Extracts and normalizes the required feature dictionary from a WeatherDataPoint.
        This ensures the feature set is consistent whether retrieved from cache or API.
        """
        return {
            "temperature_2m": w.temperature_2m,
            "relative_humidity_2m": w.relative_humidity_2m,
            "rain": w.rain,
            "wind_speed_100m": w.wind_speed_100m,
            "wind_direction_100m": w.wind_direction_100m,
            "pressure_msl": w.pressure_msl,
            "surface_pressure": w.surface_pressure
        }

    @staticmethod
    async def ingest(req: IngestRequest) -> Tuple[WeatherPayload, Dict[str, Any]]:
        """
        Process ingestion request:
        - Normalize datetime
        - Try Redis cache
        - Fallback to Open-Meteo API
        - Cache result if needed
        - Ensure features have required keys for MLP/DML

        Returns:
            Tuple[WeatherPayload, Dict[str, Any]]: 
                - Normalized weather payload
                - Features dictionary ready for MLP/DML
        """
        # Normalize datetime -> (ymd, hour, dt_obj)
        ymd, hour, dt_obj = ymdh_from_iso(req.dt_iso)

        # 1. Try Redis cache first
        cached: Optional[WeatherPayload] = await redis_client.get_cached(req.lat, req.lon, ymd, hour)
        if cached:
            log.info("Cache hit for lat=%s lon=%s ymd=%s hour=%s", req.lat, req.lon, ymd, hour)
            
            # Use helper to extract features consistently
            if cached.weather:
                features = IngestionAgent._extract_features(cached.weather[0])
                return cached, features
            
            # Fallback if cached payload somehow lacked weather data
            log.error("Cached payload found but is empty. Proceeding to API fetch.")


        log.info("Cache miss for lat=%s lon=%s ymd=%s hour=%s", req.lat, req.lon, ymd, hour)

        # 2. Fetch from Open-Meteo
        try:
            # Note: We use asyncio.to_thread for the synchronous API call
            payload: WeatherPayload = await asyncio.to_thread(fetch_openmeteo_hour, req.lat, req.lon, dt_obj)
        except Exception as e:
            log.exception("Failed to fetch or process weather from Open-Meteo: %s", e)
            raise

        features: Dict[str, Any] = {}
        
        # 2a. Extract features
        if payload.weather:
            w = payload.weather[0]
            # Use helper to extract features consistently
            features = IngestionAgent._extract_features(w)
        else:
            log.warning("No weather data available for lat=%s lon=%s dt=%s", req.lat, req.lon, dt_obj)

        # 3. Cache result
        # Only cache if we actually got data to avoid caching empty results
        if payload.weather:
            try:
                await redis_client.set_cached(req.lat, req.lon, ymd, hour, payload)
            except Exception as e:
                log.warning("Failed to cache weather payload: %s", e)

        # 4. Return payload and features
        return payload, features
