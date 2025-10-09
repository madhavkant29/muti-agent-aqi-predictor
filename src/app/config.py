import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL_SECONDS: int = 60 * 60 * 36  # 36 hours
    REDIS_INDEX_TTL_SECONDS: int = 60 * 60 * 48  # 48 hours

    # Open-Meteo API Settings
    OPENMETEO_BASE_URL: str = "https://api.open-meteo.com"
    OPENMETEO_VERSION: str = "v1"
    OPENMETEO_FORECAST_ENDPOINT: str = "forecast"

    @property
    def OPENMETEO_URL(self) -> str:
        return f"{self.OPENMETEO_BASE_URL}/{self.OPENMETEO_VERSION}/{self.OPENMETEO_FORECAST_ENDPOINT}"

# Instantiate settings for easy import
settings = Settings()