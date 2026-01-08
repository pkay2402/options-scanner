"""
Configuration management for the options trading platform
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    
    # Schwab API Configuration
    SCHWAB_CLIENT_ID: str = Field(default="", env="SCHWAB_CLIENT_ID")
    SCHWAB_CLIENT_SECRET: str = Field(default="", env="SCHWAB_CLIENT_SECRET")
    SCHWAB_AUTH_CODE: Optional[str] = Field(None, env="SCHWAB_AUTH_CODE")
    SCHWAB_REDIRECT_URI: str = Field("https://localhost:8080", env="SCHWAB_REDIRECT_URI")
    
    # Database Configuration
    DATABASE_URL: str = Field("sqlite:///./data/options_trading.db", env="DATABASE_URL")
    POSTGRES_USER: Optional[str] = Field(None, env="POSTGRES_USER")
    POSTGRES_PASSWORD: Optional[str] = Field(None, env="POSTGRES_PASSWORD")
    POSTGRES_HOST: Optional[str] = Field("localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: Optional[int] = Field(5432, env="POSTGRES_PORT")
    POSTGRES_DB: Optional[str] = Field(None, env="POSTGRES_DB")
    
    # Redis Configuration (for caching and real-time data)
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_DB: int = Field(0, env="REDIS_DB")
    
    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field("./logs/options_trading.log", env="LOG_FILE")
    
    # API Rate Limiting
    API_RATE_LIMIT: int = Field(100, env="API_RATE_LIMIT")  # requests per minute
    API_RETRY_ATTEMPTS: int = Field(3, env="API_RETRY_ATTEMPTS")
    API_RETRY_DELAY: float = Field(1.0, env="API_RETRY_DELAY")  # seconds
    
    # Analysis Configuration
    DEFAULT_LOOKBACK_DAYS: int = Field(30, env="DEFAULT_LOOKBACK_DAYS")
    MIN_OPTION_VOLUME: int = Field(100, env="MIN_OPTION_VOLUME")
    MIN_OPTION_OI: int = Field(500, env="MIN_OPTION_OI")  # Open Interest
    UNUSUAL_VOLUME_THRESHOLD: float = Field(2.0, env="UNUSUAL_VOLUME_THRESHOLD")  # Multiple of average
    
    # Market Data Configuration
    MARKET_DATA_UPDATE_INTERVAL: int = Field(60, env="MARKET_DATA_UPDATE_INTERVAL")  # seconds
    REAL_TIME_DATA_ENABLED: bool = Field(True, env="REAL_TIME_DATA_ENABLED")
    
    # Monitoring and Alerts
    ENABLE_ALERTS: bool = Field(True, env="ENABLE_ALERTS")
    ALERT_EMAIL: Optional[str] = Field(None, env="ALERT_EMAIL")
    SLACK_WEBHOOK_URL: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")
    DISCORD_WEBHOOK_URL: Optional[str] = Field(None, env="DISCORD_WEBHOOK_URL")
    
    # Discord Bot Configuration
    DISCORD_BOT_TOKEN: Optional[str] = Field(None, env="DISCORD_BOT_TOKEN")
    
    # TOS Email Alerts (for Discord bot TOS alerts command)
    TOS_EMAIL_ADDRESS: Optional[str] = Field(None, env="TOS_EMAIL_ADDRESS")
    TOS_EMAIL_PASSWORD: Optional[str] = Field(None, env="TOS_EMAIL_PASSWORD")
    
    # Dashboard Configuration
    DASHBOARD_HOST: str = Field("localhost", env="DASHBOARD_HOST")
    DASHBOARD_PORT: int = Field(8050, env="DASHBOARD_PORT")
    DASHBOARD_DEBUG: bool = Field(False, env="DASHBOARD_DEBUG")
    
    # Data Storage
    DATA_RETENTION_DAYS: int = Field(365, env="DATA_RETENTION_DAYS")
    BACKUP_ENABLED: bool = Field(True, env="BACKUP_ENABLED")
    BACKUP_INTERVAL_HOURS: int = Field(24, env="BACKUP_INTERVAL_HOURS")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(30, env="REQUEST_TIMEOUT")  # seconds
    
    # Security
    SECRET_KEY: str = Field("your-secret-key-here", env="SECRET_KEY")
    ALLOWED_HOSTS: list = Field(["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")

    # Optional Web Push / API Server settings (tolerate extra env vars)
    VAPID_PUBLIC_KEY: Optional[str] = Field(None, env="VAPID_PUBLIC_KEY")
    VAPID_PRIVATE_KEY: Optional[str] = Field(None, env="VAPID_PRIVATE_KEY")
    API_HOST: Optional[str] = Field(None, env="API_HOST")
    API_PORT: Optional[int] = Field(None, env="API_PORT")
    
    # Development/Testing
    DEBUG: bool = Field(False, env="DEBUG")
    TESTING: bool = Field(False, env="TESTING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """
    Get application settings (singleton pattern)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings():
    """
    Reload settings (useful for testing or configuration changes)
    """
    global _settings
    _settings = None
    return get_settings()

# Market hours configuration
MARKET_HOURS = {
    "NYSE": {
        "open": "09:30",
        "close": "16:00",
        "timezone": "America/New_York"
    },
    "NASDAQ": {
        "open": "09:30",
        "close": "16:00", 
        "timezone": "America/New_York"
    },
    "OPTIONS": {
        "open": "09:30",
        "close": "16:15",
        "timezone": "America/New_York"
    }
}

# Options expiration patterns
OPTIONS_EXPIRATION = {
    "MONTHLY": "third_friday",
    "WEEKLY": "every_friday",
    "QUARTERLY": "last_trading_day_of_quarter"
}

# Common stock symbols for analysis
MAJOR_INDICES = ["SPY", "QQQ", "IWM", "DIA"]
HIGH_VOLUME_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
    "AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "IBM", "TXN"
]

# Options Greeks thresholds
GREEKS_THRESHOLDS = {
    "high_gamma": 0.1,
    "high_delta": 0.7,
    "high_theta": -0.05,
    "high_vega": 0.3,
    "high_rho": 0.1
}

# Volume and Open Interest thresholds
VOLUME_THRESHOLDS = {
    "unusual_volume_ratio": 2.0,  # 2x average volume
    "high_volume_absolute": 1000,
    "min_oi_for_analysis": 100
}

# Volatility thresholds
VOLATILITY_THRESHOLDS = {
    "low_iv": 0.15,   # 15%
    "normal_iv": 0.25,  # 25%
    "high_iv": 0.40,   # 40%
    "extreme_iv": 0.60  # 60%
}

# Big trade thresholds
BIG_TRADE_THRESHOLDS = {
    "min_premium": 50000,      # $50k minimum premium
    "min_volume": 500,         # 500 contracts minimum
    "min_notional": 1000000,   # $1M notional value
    "unusual_size_ratio": 5.0   # 5x normal size
}

def get_discord_webhook() -> Optional[str]:
    """
    Get Discord webhook URL from Streamlit secrets or environment variable.
    Returns None if not configured.
    """
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'alerts' in st.secrets:
            return st.secrets['alerts'].get('discord_webhook')
    except:
        pass
    
    # Fallback to environment variable
    webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if webhook:
        return webhook
    
    # Try settings
    settings = get_settings()
    return settings.DISCORD_WEBHOOK_URL