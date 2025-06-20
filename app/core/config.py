# app/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "ql+asyncpg://postgres:admin@localhost:5432/Presence"
    DEBUG_MODE: bool = False

settings = Settings()
