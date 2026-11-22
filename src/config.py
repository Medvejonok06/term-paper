from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/football_predictor"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
