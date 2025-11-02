# Options Trading Platform Configuration
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
