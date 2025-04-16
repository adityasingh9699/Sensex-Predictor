from pydantic_settings import BaseSettings
from datetime import time
import pytz
from typing import List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sensex Predictor"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Database Settings
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    
    # Model Settings
    MODEL_SAVE_PATH: str = "app/models/saved_models"
    SEQUENCE_LENGTH: int = 60
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    
    # Hugging Face Settings
    HUGGINGFACE_API_KEY: str
    USE_GPU: bool = False
    SENTIMENT_MODEL: str = "finiteautomata/bertweet-base-sentiment-analysis"
    MAX_NEWS_LENGTH: int = 512
    
    # Data Collection Settings
    YEARS_OF_HISTORICAL_DATA: int = 10
    LIVE_DATA_UPDATE_INTERVAL: int = 5  # minutes
    NEWS_UPDATE_INTERVAL: int = 60  # minutes
    MARKET_CLOSE_TIME: time = time(15, 30)  # 3:30 PM IST
    TIMEZONE: str = "Asia/Kolkata"
    
    # News API Settings
    NEWS_API_KEY: str
    NEWS_SOURCES: List[str] = [
        "reuters.com",
        "bloomberg.com",
        "economictimes.indiatimes.com",
        "moneycontrol.com",
        "livemint.com",
        "business-standard.com",
        "thehindubusinessline.com"
    ]
    
    # Training Settings
    RECENT_DATA_WEIGHT: float = 0.7  # Weight for recent data (last 3 months)
    OLD_DATA_WEIGHT: float = 0.3  # Weight for older data
    
    # Market Settings
    MARKET_OPEN_TIME: time = time(9, 15)  # 9:15 AM IST
    MARKET_CLOSE_TIME: time = time(15, 30)  # 3:30 PM IST
    MARKET_SYMBOL: str = "^BSESN"  # BSE SENSEX symbol
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        env_nested_delimiter = '__'

settings = Settings() 