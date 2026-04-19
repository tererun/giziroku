from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str = Field(default="", alias="HF_TOKEN")
    api_keys_raw: str = Field(default="", alias="API_KEYS")

    whisper_model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    whisper_compute_type: str = Field(default="int8_float16", alias="WHISPER_COMPUTE_TYPE")
    device: str = Field(default="cuda", alias="DEVICE")
    default_language: str = Field(default="ja", alias="DEFAULT_LANGUAGE")

    max_queue_size: int = Field(default=50, alias="MAX_QUEUE_SIZE")
    job_ttl: int = Field(default=3600, alias="JOB_TTL")

    stream_chunk_seconds: float = Field(default=5.0, alias="STREAM_CHUNK_SECONDS")
    stream_sample_rate: int = Field(default=16000, alias="STREAM_SAMPLE_RATE")

    @property
    def api_keys(self) -> List[str]:
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
