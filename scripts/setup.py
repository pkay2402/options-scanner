#!/usr/bin/env python3
"""
Setup script for Options Trading Platform
Initializes the environment, creates necessary directories, and sets up the database
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config import get_settings
from src.data.database import DatabaseManager

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "setup.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

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
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/options_trading
# POSTGRES_USER=your_postgres_user
# POSTGRES_PASSWORD=your_postgres_password
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=options_trading

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

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
ALERT_EMAIL=your_email@example.com
# SLACK_WEBHOOK_URL=your_slack_webhook_url
# DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Dashboard Configuration
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=false

# Data Storage
DATA_RETENTION_DAYS=365
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30

# Security
SECRET_KEY=your-secret-key-here-change-this
ALLOWED_HOSTS=localhost,127.0.0.1

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

def setup_database():
    """Initialize the database"""
    try:
        print("Setting up database...")
        db_manager = DatabaseManager()
        
        # Database tables are created automatically by DatabaseManager
        stats = db_manager.get_database_stats()
        print(f"✓ Database initialized successfully")
        print(f"  - Tables created and ready")
        
        return True
        
    except Exception as e:
        print(f"✗ Error setting up database: {str(e)}")
        return False

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

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_dirs = [
        "src",
        "src/api",
        "src/analysis", 
        "src/data",
        "src/monitoring",
        "src/utils",
        "src/visualization",
        "tests"
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✓ Created {init_file}")

def run_setup():
    """Run the complete setup process"""
    logger = setup_logging()
    
    print("🚀 Starting Options Trading Platform Setup")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create init files
    print("\n🐍 Creating Python package files...")
    create_init_files()
    
    # Create environment template
    print("\n⚙️  Setting up environment configuration...")
    create_env_template()
    
    # Create sample config
    print("\n📋 Creating sample configuration...")
    create_sample_config()
    
    # Setup database
    print("\n🗄️  Setting up database...")
    db_success = setup_database()
    
    print("\n" + "=" * 50)
    
    if db_success:
        print("✅ Setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Update the .env file with your Schwab API credentials")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Test the setup: python -c 'from src.api.schwab_client import SchwabClient; print(\"Import successful!\")'")
        print("4. Start monitoring: python -m src.monitoring.options_flow")
        
    else:
        print("❌ Setup completed with errors!")
        print("Please check the logs and fix any database issues.")
    
    print("\n📖 For more information, see the README.md file")

if __name__ == "__main__":
    run_setup()