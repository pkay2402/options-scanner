<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Options Trading Platform

This is a comprehensive Python-based options trading and market dynamics analysis platform.

### Key Features
- **Market Dynamics Analysis**: Short-term and mid-term market analysis using options data
- **Big Trades Detection**: Identify and analyze significant options trades
- **Real-time Monitoring**: Live options flow monitoring with customizable alerts
- **Individual Stock Analysis**: Deep dive analysis for specific stocks
- **Schwab API Integration**: Direct integration with Schwab's trading API
- **Interactive Dashboard**: Web-based visualization and monitoring dashboard
- **Database Management**: SQLite/PostgreSQL storage for historical data

### Project Structure
```
src/
├── api/                 # Schwab API integration
├── analysis/            # Market analysis and big trades detection
├── data/                # Database management
├── monitoring/          # Real-time monitoring and alerts
├── utils/               # Configuration and utilities
└── visualization/       # Dashboard and charts
```

### Development Guidelines
- Use async/await for API calls when possible
- Follow dataclasses for structured data
- Implement comprehensive error handling
- Use logging for monitoring and debugging
- Write tests for all major functionality
- Follow type hints for better code clarity

### Configuration
The platform uses environment variables and pydantic settings for configuration. Update `.env` with your Schwab API credentials and other settings.

### Usage Examples
```python
# Market analysis
from src.analysis.market_dynamics import MarketDynamicsAnalyzer
analyzer = MarketDynamicsAnalyzer()
result = analyzer.analyze_short_term_dynamics()

# Big trades detection
from src.analysis.big_trades import BigTradesDetector
detector = BigTradesDetector()
big_trades = detector.scan_for_big_trades(['SPY', 'QQQ'])

# Real-time monitoring
from src.monitoring.options_flow import OptionsFlowMonitor
monitor = OptionsFlowMonitor()
monitor.start_monitoring()
```

The platform provides both programmatic APIs and an interactive CLI interface via `main.py`.