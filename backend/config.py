from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    featherless_api_key: str = ""
    featherless_base_url: str = ""
    openai_api_key: str = ""
    claude_api_key: str = ""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_enabled: bool = True

    language: Literal["id", "en"] = "id"

    stt_provider: Literal["openai", "google", "local", "elevenlabs"] = "openai"
    tts_provider: Literal["openai", "google"] = "openai"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "nova"
    tts_speed: float = 1.5
    tts_device_index: int = 1

    elevenlabs_api_key: str = ""

    pinecone_api_key: str = ""
    pinecone_index_name: str = "mitsubishi-customers"
    pinecone_namespace: str = "customers-data"

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = ""

    app_mode: Literal["demo", "production"] = "demo"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
