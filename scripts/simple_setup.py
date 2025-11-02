#!/usr/bin/env python3
"""
Simple setup script for Options Trading Platform
Creates directories and basic configuration files
"""

import os
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "logs", 
        "config",
        "exports",
        "backups",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_template():
    """Create .env template file"""
    env_template = """# Schwab API Configuration
SCHWAB_CLIENT_ID=your_client_id_here
SCHWAB_CLIENT_SECRET=your_client_secret_here
SCHWAB_AUTH_CODE=your_auth_code_here
SCHWAB_REDIRECT_URI=https://localhost:8080

# Database Configuration
DATABASE_URL=sqlite:///./data/options_trading.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/options_trading.log

# Analysis Configuration
DEFAULT_LOOKBACK_DAYS=30
MIN_OPTION_VOLUME=100
MIN_OPTION_OI=500
UNUSUAL_VOLUME_THRESHOLD=2.0

# Market Data Configuration
MARKET_DATA_UPDATE_INTERVAL=60
REAL_TIME_DATA_ENABLED=true

# Monitoring and Alerts
ENABLE_ALERTS=true

# Dashboard Configuration
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=false

# Security
SECRET_KEY=your-secret-key-here-change-this

# Development/Testing
DEBUG=false
TESTING=false
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_template)
        print("✓ Created .env template file")
        print("⚠️  Please update .env with your actual configuration values")
    else:
        print("✓ .env file already exists")

def create_sample_config():
    """Create sample configuration file"""
    config_content = """# Options Trading Platform Configuration
# This file contains sample configurations and can be customized

# Symbols to monitor by default
DEFAULT_MONITOR_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",  # Major indices
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"  # High volume stocks
]

# Alert thresholds
ALERT_THRESHOLDS = {
    "volume_spike_multiplier": 3.0,
    "big_trade_min_premium": 100000,  # $100k
    "iv_surge_threshold": 0.5,  # 50%
    "unusual_flow_ratio": 3.0
}

# Analysis parameters
ANALYSIS_PARAMS = {
    "short_term_lookback_hours": 4,
    "mid_term_lookback_days": 30,
    "min_confidence_threshold": 0.6
}
"""
    
    config_file = Path("config/sample_config.py")
    if not config_file.exists():
        with open(config_file, "w") as f:
            f.write(config_content)
        print("✓ Created sample configuration file")
    else:
        print("✓ Sample configuration file already exists")

def run_setup():
    """Run the setup process"""
    print("🚀 Starting Options Trading Platform Setup")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create environment template
    print("\n⚙️  Setting up environment configuration...")
    create_env_template()
    
    # Create sample config
    print("\n📋 Creating sample configuration...")
    create_sample_config()
    
    print("\n" + "=" * 50)
    print("✅ Basic setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Update the .env file with your Schwab API credentials")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the main application: python3 main.py")
    print("\n📖 For more information, see the README.md file")

if __name__ == "__main__":
    run_setup()