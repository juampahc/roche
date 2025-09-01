from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Triton interal name for model (i.e. the repo folder)
    MODEL_NAME: str = Field(default='formula-recognition', env='NER_MODEL_NAME')
    TRITON_URL: str = Field(default='localhost:8001', env="TRITON_URL")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

def get_settings():
    return Settings()