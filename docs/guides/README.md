# Options Trading & Market Dynamics Analysis Platform

A comprehensive Python-based platform for analyzing options market dynamics, individual stock behavior, and detecting significant options trades using the Schwab API.

## 🚀 Features

### Market Dynamics Analysis
- **Short-term Analysis**: Real-time options flow, gamma exposure, and volatility patterns
- **Mid-term Analysis**: Options positioning trends, put/call ratios, and sentiment indicators
- **Individual Stock Dynamics**: Stock-specific options analysis and unusual activity detection

### Big Trades Detection
- Identify significant options trades based on premium, volume, and notional value
- Analyze trade sentiment and market impact
- Real-time alerts for unusual trading activity

### Real-time Monitoring
- Live options flow monitoring with customizable alerts
- Volume spikes, IV surges, and unusual flow detection
- Multiple alert channels (console, logs, webhooks)

### Data Sources
- Schwab API integration for real-time and historical options data
- Market data aggregation and processing
- Comprehensive database storage for historical analysis

### Visualization & Dashboard
- Interactive web dashboard with Plotly/Dash
- Options flow charts, put/call ratios, gamma exposure
- Big trades visualization and unusual activity timelines

## 📁 Project Structure

```
options-trading-platform/
├── src/
│   ├── api/                    # Schwab API clients (sync/async)
│   ├── analysis/               # Market analysis modules
│   │   ├── market_dynamics.py  # Short/mid-term market analysis
│   │   └── big_trades.py       # Big trades detection
│   ├── data/                   # Database management
│   ├── monitoring/             # Real-time monitoring & alerts
│   ├── visualization/          # Charts and dashboards
│   └── utils/                  # Configuration and utilities
├── config/                     # Configuration files
├── tests/                      # Unit and integration tests
├── scripts/                    # Setup and utility scripts
├── data/                       # Data storage
├── logs/                       # Application logs
├── main.py                     # Main application entry point
└── requirements.txt            # Dependencies
```

## 🛠️ Installation

1. **Clone and Setup**
   ```bash
   cd options-trading-platform
   python3 scripts/simple_setup.py
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   - Update `.env` file with your Schwab API credentials
   - Configure database settings (SQLite by default)
   - Set monitoring and alert preferences

4. **Verify Installation**
   ```bash
   python3 main.py --mode analysis --symbols SPY
   ```

## 🔧 Configuration

### Schwab API Setup
```bash
# Required in .env file
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_client_secret
SCHWAB_AUTH_CODE=your_auth_code
SCHWAB_REDIRECT_URI=https://localhost:8080
```

### Database Configuration
```bash
# SQLite (default)
DATABASE_URL=sqlite:///./data/options_trading.db

# PostgreSQL (optional)
DATABASE_URL=postgresql://user:password@localhost:5432/options_trading
```

### Monitoring Settings
```bash
MARKET_DATA_UPDATE_INTERVAL=60
ENABLE_ALERTS=true
ALERT_EMAIL=your_email@example.com
```

## 🚀 Usage

### Interactive Mode
```bash
python3 main.py --mode interactive
```

### Market Analysis
```bash
# Short-term analysis
python3 main.py --mode analysis --analysis-type short --symbols SPY QQQ

# Mid-term analysis  
python3 main.py --mode analysis --analysis-type mid --symbols AAPL TSLA
```

### Real-time Monitoring
```bash
python3 main.py --mode monitor --symbols SPY QQQ IWM DIA
```

### Web Dashboard
```bash
python3 main.py --mode dashboard
# Open http://localhost:8050 in your browser
```

### Programmatic Usage
```python
from src.analysis.market_dynamics import MarketDynamicsAnalyzer
from src.api.schwab_client import SchwabClient

# Initialize components
client = SchwabClient()
analyzer = MarketDynamicsAnalyzer(client)

# Analyze short-term market dynamics
analysis = analyzer.analyze_short_term_dynamics(['SPY'])
print(f"Put/Call Ratio: {analysis.sentiment.put_call_ratio:.2f}")
print(f"Sentiment Score: {analysis.sentiment.sentiment_score:.2f}")

# Get recommendations
for rec in analysis.recommendations:
    print(f"• {rec}")
```

### Big Trades Detection
```python
from src.analysis.big_trades import BigTradesDetector

detector = BigTradesDetector()
big_trades = detector.scan_for_big_trades(['SPY', 'QQQ'], min_premium=100000)

for trade in big_trades[:5]:  # Top 5 trades
    print(f"{trade.symbol} {trade.contract_type} ${trade.strike}")
    print(f"Volume: {trade.volume:,}, Premium: ${trade.premium * trade.volume * 100:,.0f}")
    print(f"Sentiment: {trade.sentiment}, Size Score: {trade.size_score}/10")
```

### Real-time Monitoring
```python
from src.monitoring.options_flow import OptionsFlowMonitor

# Create monitor with custom alerts
monitor = OptionsFlowMonitor()

# Add custom alert handler
def my_alert_handler(alert):
    print(f"CUSTOM ALERT: {alert.message}")
    # Send to Slack, Discord, etc.

monitor.add_alert_handler(my_alert_handler)
monitor.start_monitoring(['SPY', 'AAPL', 'TSLA'])
```

## 📊 Analysis Features

### Market Dynamics
- **Put/Call Ratios**: Track market sentiment indicators
- **Gamma Exposure**: Identify potential volatility zones
- **VIX Analysis**: Volatility environment assessment
- **Key Levels**: Support/resistance from options data

### Individual Stock Analysis
- Options volume and open interest analysis
- Implied volatility trends
- Max pain calculations
- Unusual activity detection
- Sentiment scoring with confidence levels

### Big Trades Detection
- Premium-based filtering (configurable thresholds)
- Volume and notional value analysis
- Trade sentiment classification
- Urgency and size scoring
- Market impact assessment

## 🔔 Alert System

### Alert Types
- **Volume Spikes**: Unusual options volume activity
- **Big Trades**: Large premium or notional value trades
- **IV Surges**: Implied volatility increases
- **Unusual Flow**: Extreme put/call ratios

### Alert Channels
- Console output
- Log files
- Email notifications (configurable)
- Webhook support (Slack, Discord)
- Custom handlers

## 📈 Dashboard Features

- **Real-time Options Flow**: Live volume and premium tracking
- **Put/Call Ratio Charts**: Historical trend analysis
- **Big Trades Visualization**: Size and sentiment mapping
- **Gamma Exposure Maps**: Strike-level analysis
- **IV Surface Plots**: Volatility structure analysis
- **Unusual Activity Timeline**: Pattern recognition

## 🧪 Testing

```bash
# Run all tests
python3 tests/test_basic.py

# Run specific test category
python3 -m unittest tests.test_basic.TestMarketDynamicsAnalyzer
```

## 📝 Development

### Adding New Analysis Modules
1. Create module in `src/analysis/`
2. Follow dataclass patterns for data structures
3. Implement proper error handling and logging
4. Add unit tests in `tests/`
5. Update documentation

### Custom Alert Rules
```python
from src.monitoring.options_flow import AlertRule

# Create custom rule
custom_rule = AlertRule(
    name="My Custom Rule",
    condition_type="volume_spike",
    parameters={
        'min_volume': 2000,
        'spike_multiplier': 4.0
    },
    cooldown_minutes=20
)

monitor.add_alert_rule(custom_rule)
```

## 🔒 Security

- Store API credentials in environment variables
- Use secure database connections
- Implement rate limiting for API calls
- Log security events
- Regular credential rotation

## 📚 Dependencies

- **Core**: pandas, numpy, requests, asyncio
- **Database**: SQLAlchemy, psycopg2-binary
- **Visualization**: plotly, dash, matplotlib
- **API**: aiohttp, websockets
- **Configuration**: pydantic, python-dotenv
- **Testing**: pytest, unittest

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- Check the logs in `logs/` directory
- Review configuration in `.env` file
- Ensure Schwab API credentials are valid
- Test database connectivity
- Verify Python dependencies are installed

For issues and questions, please check the documentation or create an issue in the repository.