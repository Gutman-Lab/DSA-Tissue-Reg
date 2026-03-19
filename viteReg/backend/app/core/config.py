"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # DSA Configuration
    DSA_BASE_URL: str = "http://bdsa.pathology.emory.edu:8080/api/v1"
    DSAKEY: str = ""
    
    # Case Collection - folder containing all case folders
    CASE_COLLECTION_ID: str = "695d6a148c871f3a02969b00"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:80",
        "http://localhost:3000",
    ]
    
    # Cache
    CACHE_DIR: str = "/app/.npCacheDir"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env (like VITE_* variables)


settings = Settings()

