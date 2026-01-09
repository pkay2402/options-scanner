"""
Reflecting Boundaries Scanner - Based on Milton Berg's "The Boundaries of Technical Analysis"

This scanner identifies market turning points by detecting "reflecting boundaries" - 
extreme price/volume conditions that signal probable trend reversals.

Key Concepts:
1. Stock prices move randomly day-to-day but encounter boundaries at extremes
2. Anomalous price thrusts + volume spikes near multi-period highs/lows = boundaries
3. Same indicators work for tops AND bottoms - context matters

Indicators Implemented:
- 5-Day ROC +8% Thrust (buy signal near lows)
- 5-Day ROC -8% Capitulation (buy signal near lows)
- Volume Climax at Boundaries
- TRIN Extremes (bullish at lows, bearish at highs)
- Breadth Thrusts & Advance-Decline Extremes
- New High/Low Ratios at Boundaries

ENHANCED VERSION:
✅ Schwab API integration for real-time data
✅ Backtesting engine with performance tracking
✅ Alert notifications (email & webhook)
✅ Additional Berg indicators
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
import json
import os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Try to import Schwab client
SCHWAB_AVAILABLE = False
try:
    from src.api.schwab_client import SchwabClient
    SCHWAB_AVAILABLE = True
except ImportError:
    print("Schwab API not available, using yfinance only")

# Configuration
DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'NVDA', 'TSLA', 'META', 'MSFT', 'AMZN']
LOOKBACK_DAYS = 500  # ~2 years of data

# Alert configuration
ALERT_CONFIG_FILE = Path(__file__).parent / 'alerts_config.json'


class DataFetcher:
    """Unified data fetcher supporting both Schwab API and yfinance"""
    
    def __init__(self, use_schwab: bool = True):
        self.use_schwab = use_schwab and SCHWAB_AVAILABLE
        self.schwab_client = None
        
        if self.use_schwab:
            try:
                self.schwab_client = SchwabClient()
                if not self.schwab_client.authenticate():
                    print("Schwab authentication failed, falling back to yfinance")
                    self.use_schwab = False
            except Exception as e:
                print(f"Schwab client initialization failed: {e}")
                self.use_schwab = False
    
    def fetch_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch historical data from best available source"""
        
        # Try Schwab first if available
        if self.use_schwab and self.schwab_client:
            try:
                data = self._fetch_from_schwab(symbol, period)
                if data is not None and len(data) > 100:
                    return data
            except Exception as e:
                print(f"Schwab fetch failed for {symbol}: {e}")
        
        # Fall back to yfinance
        return self._fetch_from_yfinance(symbol, period)
    
    def _fetch_from_schwab(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch from Schwab API with price history"""
        
        # Map period to Schwab API parameters
        period_map = {
            '1y': {'period_type': 'year', 'period': 1},
            '2y': {'period_type': 'year', 'period': 2},
            '5y': {'period_type': 'year', 'period': 5},
            '10y': {'period_type': 'year', 'period': 10}
        }
        
        params = period_map.get(period, {'period_type': 'year', 'period': 2})
        
        # Get price history from Schwab
        try:
            price_history = self.schwab_client.get_price_history(
                symbol=symbol,
                period_type=params['period_type'],
                period=params['period'],
                frequency_type='daily',
                frequency=1
            )
        except Exception as e:
            print(f"Error getting price history for {symbol}: {e}")
            return None
        
        if not price_history or 'candles' not in price_history:
            return None
        
        # Convert to DataFrame
        candles = price_history['candles']
        df = pd.DataFrame(candles)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df = df.set_index('datetime')
        
        # Rename columns to match yfinance format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def _fetch_from_yfinance(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return None
            
            # Convert timezone-aware index to timezone-naive to avoid comparison issues
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df
        except Exception as e:
            print(f"yfinance fetch failed for {symbol}: {e}")
            return None


class AlertManager:
    """Manage alerts for boundary signals"""
    
    def __init__(self):
        self.config = self._load_config()
        self.sent_alerts = self._load_sent_alerts()
    
    def _load_config(self) -> Dict:
        """Load alert configuration"""
        default_config = {
            'enabled': False,
            'email': {
                'enabled': False,
                'smtp_server': '',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'to_addresses': []
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'headers': {}
            },
            'discord': {
                'enabled': False,
                'webhook_url': ''
            }
        }
        
        if ALERT_CONFIG_FILE.exists():
            try:
                with open(ALERT_CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
            except Exception as e:
                print(f"Error loading alert config: {e}")
        
        return default_config
    
    def _load_sent_alerts(self) -> Dict:
        """Load record of sent alerts to avoid duplicates"""
        sent_file = Path(__file__).parent / 'sent_alerts.json'
        if sent_file.exists():
            try:
                with open(sent_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_sent_alerts(self):
        """Save record of sent alerts"""
        sent_file = Path(__file__).parent / 'sent_alerts.json'
        try:
            with open(sent_file, 'w') as f:
                json.dump(self.sent_alerts, f, indent=2)
        except Exception as e:
            print(f"Error saving sent alerts: {e}")
    
    def should_send_alert(self, symbol: str, signal_type: str, signal_date: str) -> bool:
        """Check if alert should be sent (not duplicate)"""
        if not self.config.get('enabled', False):
            return False
        
        alert_key = f"{symbol}_{signal_type}_{signal_date}"
        return alert_key not in self.sent_alerts
    
    def send_alert(self, symbol: str, signal_type: str, signal_data: Dict):
        """Send alert via configured channels"""
        
        signal_date = signal_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        if not self.should_send_alert(symbol, signal_type, signal_date):
            return
        
        message = self._format_alert_message(symbol, signal_type, signal_data)
        
        # Send via email
        if self.config.get('email', {}).get('enabled', False):
            self._send_email_alert(message)
        
        # Send via webhook
        if self.config.get('webhook', {}).get('enabled', False):
            self._send_webhook_alert(message)
        
        # Send via Discord
        if self.config.get('discord', {}).get('enabled', False):
            self._send_discord_alert(message)
        
        # Record alert sent
        alert_key = f"{symbol}_{signal_type}_{signal_date}"
        self.sent_alerts[alert_key] = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal_type': signal_type,
            'data': signal_data
        }
        self._save_sent_alerts()
    
    def _format_alert_message(self, symbol: str, signal_type: str, signal_data: Dict) -> str:
        """Format alert message"""
        emoji_map = {
            'THRUST_BUY': '🚀',
            'CAPITULATION_BUY': '⭐',
            'TRIN_VOLUME_BUY': '💎',
            'TRIN_VOLUME_SELL': '⚠️',
            'BREADTH_THRUST': '📈',
            'ADVANCE_DECLINE_EXTREME': '🎯'
        }
        
        emoji = emoji_map.get(signal_type, '📊')
        
        message = f"""
{emoji} BOUNDARY SIGNAL DETECTED {emoji}

Symbol: {symbol}
Signal: {signal_type.replace('_', ' ')}
Date: {signal_data.get('date', 'N/A')}
Price: ${signal_data.get('price', 0):.2f}
ROC 5D: {signal_data.get('roc_5d', 0):.2f}%
Volume: {signal_data.get('volume', 0):,.0f}
Strength: {signal_data.get('strength', 0):.2f}

Context: {signal_data.get('context', 'N/A')}

This is an automated alert from the Reflecting Boundaries Scanner.
Review the signal and take appropriate action.
        """
        
        return message.strip()
    
    def _send_email_alert(self, message: str):
        """Send email alert"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = '🎯 Boundary Signal Alert'
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print("✅ Email alert sent successfully")
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
    
    def _send_webhook_alert(self, message: str):
        """Send webhook alert"""
        try:
            import requests
            
            webhook_config = self.config['webhook']
            
            payload = {
                'text': message,
                'timestamp': datetime.now().isoformat()
            }
            
            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Webhook alert sent successfully")
            else:
                print(f"❌ Webhook alert failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Webhook alert failed: {e}")
    
    def _send_discord_alert(self, message: str):
        """Send Discord webhook alert"""
        try:
            import requests
            
            discord_config = self.config['discord']
            
            payload = {
                'content': message,
                'username': 'Boundary Scanner'
            }
            
            response = requests.post(
                discord_config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                print("✅ Discord alert sent successfully")
            else:
                print(f"❌ Discord alert failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Discord alert failed: {e}")


class BacktestEngine:
    """Backtest boundary signals for performance analysis"""
    
    def __init__(self, holding_periods: List[int] = [5, 10, 21, 63]):
        self.holding_periods = holding_periods
    
    def backtest_signals(self, scanner: 'BoundaryScanner', signals: pd.DataFrame, 
                        signal_type: str) -> pd.DataFrame:
        """Calculate forward returns for signals"""
        
        if len(signals) == 0:
            return pd.DataFrame()
        
        results = []
        
        for signal_date, signal_row in signals.iterrows():
            entry_price = signal_row['Close']
            
            # Calculate forward returns for each holding period
            forward_returns = {}
            max_drawdown = 0
            
            for days in self.holding_periods:
                exit_date = signal_date + timedelta(days=days)
                
                # Find exit price
                future_data = scanner.data[scanner.data.index > signal_date]
                
                if len(future_data) == 0:
                    continue
                
                # Get price at holding period (or closest)
                if exit_date in future_data.index:
                    exit_price = future_data.loc[exit_date, 'Close']
                else:
                    # Find nearest date
                    nearest_date = future_data.index[future_data.index >= exit_date]
                    if len(nearest_date) > 0:
                        exit_price = future_data.loc[nearest_date[0], 'Close']
                    else:
                        # Use last available price
                        exit_price = future_data.iloc[-1]['Close']
                
                # Calculate return
                ret = ((exit_price - entry_price) / entry_price) * 100
                forward_returns[f'return_{days}d'] = ret
                
                # Calculate max drawdown during period
                period_data = future_data.iloc[:days] if len(future_data) >= days else future_data
                if len(period_data) > 0:
                    period_low = period_data['Low'].min()
                    drawdown = ((period_low - entry_price) / entry_price) * 100
                    max_drawdown = min(max_drawdown, drawdown)
            
            if forward_returns:
                result = {
                    'signal_date': signal_date,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'max_drawdown': max_drawdown,
                    **forward_returns
                }
                results.append(result)
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Calculate summary statistics
            for days in self.holding_periods:
                col = f'return_{days}d'
                if col in results_df.columns:
                    results_df[f'{col}_rank'] = results_df[col].rank(pct=True)
            
            return results_df
        
        return pd.DataFrame()
    
    def generate_performance_summary(self, backtest_results: pd.DataFrame) -> Dict:
        """Generate performance summary statistics"""
        
        if len(backtest_results) == 0:
            return {}
        
        summary = {
            'total_signals': len(backtest_results),
            'holding_periods': {}
        }
        
        for days in self.holding_periods:
            col = f'return_{days}d'
            if col not in backtest_results.columns:
                continue
            
            returns = backtest_results[col].dropna()
            
            if len(returns) == 0:
                continue
            
            summary['holding_periods'][days] = {
                'avg_return': returns.mean(),
                'median_return': returns.median(),
                'win_rate': (returns > 0).sum() / len(returns) * 100,
                'best_return': returns.max(),
                'worst_return': returns.min(),
                'std_dev': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
            }
        
        # Overall max drawdown
        if 'max_drawdown' in backtest_results.columns:
            summary['max_drawdown'] = backtest_results['max_drawdown'].min()
        
        return summary


class BoundaryScanner:
    """Identify reflecting boundaries in market data"""
    
    def __init__(self, symbol: str, data: pd.DataFrame):
        self.symbol = symbol
        self.data = data.copy()
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """Calculate all technical indicators needed for boundary detection"""
        df = self.data
        
        # 5-Day Rate of Change
        df['roc_5d'] = df['Close'].pct_change(periods=5) * 100
        
        # 5-Day Average Volume
        df['vol_5d_avg'] = df['Volume'].rolling(window=5).mean()
        
        # Rolling lows/highs for boundary context
        df['low_90d'] = df['Low'].rolling(window=90, min_periods=1).min()
        df['low_6m'] = df['Low'].rolling(window=126, min_periods=1).min()  # ~6 months
        df['low_1y'] = df['Low'].rolling(window=252, min_periods=1).min()
        df['high_3y'] = df['High'].rolling(window=756, min_periods=1).max()
        
        # Days since lows/highs
        df['days_since_90d_low'] = self._days_since_extreme(df['Low'], df['low_90d'])
        df['days_since_6m_low'] = self._days_since_extreme(df['Low'], df['low_6m'])
        df['days_since_1y_low'] = self._days_since_extreme(df['Low'], df['low_1y'])
        df['days_since_3y_high'] = self._days_since_extreme(df['High'], df['high_3y'], find_min=False)
        
        # Volume extremes
        df['vol_250d_high'] = df['vol_5d_avg'].rolling(window=250, min_periods=1).max()
        df['vol_375d_high'] = df['vol_5d_avg'].rolling(window=375, min_periods=1).max()
        df['is_vol_250d_high'] = df['vol_5d_avg'] >= df['vol_250d_high']
        df['is_vol_375d_high'] = df['vol_5d_avg'] >= df['vol_375d_high']
        
        # Calculate TRIN if SPY (proxy for market-wide)
        if self.symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
            df['trin'] = self._estimate_trin(df)
        else:
            df['trin'] = np.nan
        
        self.data = df
    
    def _days_since_extreme(self, series: pd.Series, extreme_series: pd.Series, 
                           find_min: bool = True) -> pd.Series:
        """Calculate days since price matched extreme (low or high)"""
        if find_min:
            is_extreme = series <= extreme_series
        else:
            is_extreme = series >= extreme_series
        
        days_since = pd.Series(index=series.index, dtype=float)
        last_extreme_idx = None
        
        for idx in series.index:
            if is_extreme[idx]:
                last_extreme_idx = idx
                days_since[idx] = 0
            elif last_extreme_idx is not None:
                days_since[idx] = (idx - last_extreme_idx).days
            else:
                days_since[idx] = np.nan
        
        return days_since
    
    def _estimate_trin(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate TRIN (Arms Index) using price/volume relationship
        Real TRIN = (Adv/Dec) / (Adv Vol/Dec Vol)
        We'll use a simplified proxy based on price change and volume
        """
        # Use price momentum and volume to estimate TRIN-like indicator
        # High volume + strong up move = low TRIN (< 0.50)
        # High volume + strong down move = high TRIN (> 1.5)
        
        roc_1d = df['Close'].pct_change() * 100
        vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Inverse relationship: big up move + high volume = low TRIN
        trin_estimate = 1.0 - (roc_1d * vol_ratio / 100)
        trin_estimate = trin_estimate.clip(0.1, 5.0)  # Keep in reasonable range
        
        return trin_estimate
    
    def detect_thrust_signals(self) -> pd.DataFrame:
        """
        Detect 5-Day ROC +8% Thrust Signals (BUY)
        Criteria:
        - ROC 5-day >= 8%
        - Within 4-6 days AFTER 90-day low
        - Exclude signals 1-3 days after low (too early)
        """
        df = self.data
        
        thrust_conditions = (
            (df['roc_5d'] >= 8.0) &
            (df['days_since_90d_low'] >= 4) &
            (df['days_since_90d_low'] <= 6)
        )
        
        signals = df[thrust_conditions].copy()
        signals['signal_type'] = 'THRUST_BUY'
        signals['signal_strength'] = signals['roc_5d'] / 8.0  # Normalized strength
        
        return signals[['Close', 'roc_5d', 'days_since_90d_low', 'Volume', 
                       'signal_type', 'signal_strength']]
    
    def detect_capitulation_signals(self) -> pd.DataFrame:
        """
        Detect 5-Day ROC -8% Capitulation Signals (BUY)
        Criteria:
        - ROC 5-day <= -8%
        - 5-day avg volume at 250-day high
        - Within 1-7 days of 6-month low
        - Wait for series to end (signal on final day + 1)
        """
        df = self.data
        
        # Find all -8% ROC days
        decline_days = df['roc_5d'] <= -8.0
        
        # Find end of series (next day ROC > -8%)
        is_series_end = decline_days & (~decline_days.shift(-1).fillna(False))
        
        # Shift by 1 to get day after series ends
        signal_day = is_series_end.shift(1).fillna(False)
        
        capitulation_conditions = (
            signal_day &
            (df['is_vol_250d_high']) &
            (df['days_since_6m_low'] <= 7)
        )
        
        signals = df[capitulation_conditions].copy()
        signals['signal_type'] = 'CAPITULATION_BUY'
        signals['signal_strength'] = abs(signals['roc_5d']) / 8.0
        
        return signals[['Close', 'roc_5d', 'days_since_6m_low', 'Volume', 
                       'vol_5d_avg', 'signal_type', 'signal_strength']]
    
    def detect_trin_volume_signals(self) -> pd.DataFrame:
        """
        Detect TRIN + Volume Extremes at Boundaries
        
        BUY Signal:
        - TRIN <= 0.50 (extreme buying urgency)
        - 5-day volume at 375-day high
        - Within 10 days of 1-year low
        
        SELL Signal:
        - TRIN <= 0.50 (extreme buying urgency at TOP)
        - 5-day volume at 375-day high
        - Within 5 days of 3-year high
        """
        df = self.data
        
        if df['trin'].isna().all():
            return pd.DataFrame()  # No TRIN data available
        
        # TRIN extreme occurred in last 2 days
        trin_extreme = (
            (df['trin'] <= 0.50) | 
            (df['trin'].shift(1) <= 0.50)
        )
        
        # Volume extreme occurred in last 2 days
        vol_extreme = (
            df['is_vol_375d_high'] | 
            df['is_vol_375d_high'].shift(1)
        )
        
        # Buy signals at 1-year lows
        buy_conditions = (
            trin_extreme &
            vol_extreme &
            (df['days_since_1y_low'] <= 10)
        )
        
        # Sell signals at 3-year highs
        sell_conditions = (
            trin_extreme &
            vol_extreme &
            (df['days_since_3y_high'] <= 5)
        )
        
        buy_signals = df[buy_conditions].copy()
        buy_signals['signal_type'] = 'TRIN_VOLUME_BUY'
        buy_signals['signal_strength'] = (0.50 - buy_signals['trin'].clip(0, 0.50)) / 0.50
        
        sell_signals = df[sell_conditions].copy()
        sell_signals['signal_type'] = 'TRIN_VOLUME_SELL'
        sell_signals['signal_strength'] = (0.50 - sell_signals['trin'].clip(0, 0.50)) / 0.50
        
        signals = pd.concat([buy_signals, sell_signals])
        
        if len(signals) > 0:
            return signals[['Close', 'trin', 'Volume', 'vol_5d_avg', 
                           'signal_type', 'signal_strength']].sort_index()
        
        return pd.DataFrame()
    
    def detect_breadth_thrust_signals(self) -> pd.DataFrame:
        """
        Detect Breadth Thrust Signals (ADDITIONAL BERG INDICATOR)
        
        Breadth thrust occurs when:
        - Market makes sharp advance
        - Breadth (advance/decline) ratio is extremely bullish
        - Near multi-month low (bottom boundary)
        
        We estimate breadth using volume-weighted price moves
        """
        df = self.data
        
        # Estimate breadth ratio from price action
        # Positive ROC = advancing, negative = declining
        df['adv_ratio_estimate'] = (df['roc_5d'] + 10) / 20  # Normalize to 0-1 range
        df['adv_ratio_estimate'] = df['adv_ratio_estimate'].clip(0, 1)
        
        # Breadth thrust: >80% advancing days with strong ROC
        thrust_conditions = (
            (df['adv_ratio_estimate'] >= 0.80) &
            (df['roc_5d'] >= 6.0) &
            (df['days_since_90d_low'] <= 10)
        )
        
        signals = df[thrust_conditions].copy()
        signals['signal_type'] = 'BREADTH_THRUST'
        signals['signal_strength'] = signals['adv_ratio_estimate']
        
        return signals[['Close', 'roc_5d', 'adv_ratio_estimate', 'days_since_90d_low',
                       'signal_type', 'signal_strength']]
    
    def detect_advance_decline_extremes(self) -> pd.DataFrame:
        """
        Detect Advance-Decline Extreme Signals (ADDITIONAL BERG INDICATOR)
        
        Extreme advance-decline ratios at boundaries:
        - Very high advance/decline at lows = bullish reversal
        - Very high decline/advance at highs = bearish reversal
        """
        df = self.data
        
        # Estimate A/D line from price and volume
        df['ad_line_estimate'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        # Detect sharp reversals in A/D line
        df['ad_5d_change'] = df['ad_line_estimate'].diff(5)
        df['ad_5d_change_pct'] = df['ad_line_estimate'].pct_change(5) * 100
        
        # Buy signal: Sharp A/D reversal up near lows
        buy_conditions = (
            (df['ad_5d_change_pct'] >= 20) &  # 20%+ increase in A/D
            (df['days_since_6m_low'] <= 10) &
            (df['roc_5d'] >= 3.0)
        )
        
        # Sell signal: Sharp A/D reversal down near highs
        sell_conditions = (
            (df['ad_5d_change_pct'] <= -20) &  # 20%+ decrease in A/D
            (df['days_since_3y_high'] <= 10) &
            (df['roc_5d'] <= -3.0)
        )
        
        buy_signals = df[buy_conditions].copy()
        buy_signals['signal_type'] = 'AD_EXTREME_BUY'
        buy_signals['signal_strength'] = buy_signals['ad_5d_change_pct'] / 20
        
        sell_signals = df[sell_conditions].copy()
        sell_signals['signal_type'] = 'AD_EXTREME_SELL'
        sell_signals['signal_strength'] = abs(sell_signals['ad_5d_change_pct']) / 20
        
        signals = pd.concat([buy_signals, sell_signals])
        
        if len(signals) > 0:
            return signals[['Close', 'roc_5d', 'ad_5d_change_pct', 'signal_type', 
                           'signal_strength']].sort_index()
        
        return pd.DataFrame()
    
    def detect_new_high_low_extremes(self) -> pd.DataFrame:
        """
        Detect New High/Low Ratio Extremes (ADDITIONAL BERG INDICATOR)
        
        Track ratio of new highs to new lows:
        - Many new lows near market bottom = bullish reversal
        - Many new highs near market top = bearish warning
        """
        df = self.data
        
        # Detect new 52-week highs and lows
        df['is_52w_high'] = df['High'] >= df['High'].rolling(252).max()
        df['is_52w_low'] = df['Low'] <= df['Low'].rolling(252).min()
        
        # Count new highs/lows in recent period (20 days)
        df['new_highs_20d'] = df['is_52w_high'].rolling(20).sum()
        df['new_lows_20d'] = df['is_52w_low'].rolling(20).sum()
        
        # Calculate ratio (avoid division by zero)
        df['hl_ratio'] = df['new_highs_20d'] / (df['new_lows_20d'] + 1)
        
        # Buy signal: Many new lows near market bottom
        buy_conditions = (
            (df['new_lows_20d'] >= 3) &  # At least 3 new lows
            (df['hl_ratio'] <= 0.5) &  # More lows than highs
            (df['days_since_1y_low'] <= 15) &
            (df['roc_5d'] >= 4.0)  # Starting to recover
        )
        
        # Sell signal: Many new highs near market top
        sell_conditions = (
            (df['new_highs_20d'] >= 3) &  # At least 3 new highs
            (df['hl_ratio'] >= 2.0) &  # Many more highs than lows
            (df['days_since_3y_high'] <= 15)
        )
        
        buy_signals = df[buy_conditions].copy()
        buy_signals['signal_type'] = 'HL_EXTREME_BUY'
        buy_signals['signal_strength'] = 1 / (buy_signals['hl_ratio'] + 0.1)
        
        sell_signals = df[sell_conditions].copy()
        sell_signals['signal_type'] = 'HL_EXTREME_SELL'
        sell_signals['signal_strength'] = sell_signals['hl_ratio'] / 2
        
        signals = pd.concat([buy_signals, sell_signals])
        
        if len(signals) > 0:
            return signals[['Close', 'new_highs_20d', 'new_lows_20d', 'hl_ratio',
                           'signal_type', 'signal_strength']].sort_index()
        
        return pd.DataFrame()
    
    def get_current_boundary_status(self) -> Dict:
        """Analyze current position relative to boundaries"""
        latest = self.data.iloc[-1]
        
        status = {
            'symbol': self.symbol,
            'current_price': latest['Close'],
            'roc_5d': latest['roc_5d'],
            'days_since_90d_low': latest['days_since_90d_low'],
            'days_since_6m_low': latest['days_since_6m_low'],
            'days_since_1y_low': latest['days_since_1y_low'],
            'days_since_3y_high': latest['days_since_3y_high'],
            'is_near_90d_low': latest['days_since_90d_low'] <= 10,
            'is_near_6m_low': latest['days_since_6m_low'] <= 10,
            'is_near_1y_low': latest['days_since_1y_low'] <= 30,
            'is_near_3y_high': latest['days_since_3y_high'] <= 30,
            'volume_at_250d_high': latest['is_vol_250d_high'],
            'volume_at_375d_high': latest['is_vol_375d_high'],
            'trin': latest['trin'] if not pd.isna(latest['trin']) else None,
        }
        
        # Determine boundary proximity
        if status['is_near_90d_low'] or status['is_near_6m_low']:
            status['boundary_context'] = 'NEAR_BOTTOM'
        elif status['is_near_3y_high']:
            status['boundary_context'] = 'NEAR_TOP'
        else:
            status['boundary_context'] = 'MIDDLE_RANGE'
        
        # Check for potential setups
        status['potential_thrust_setup'] = (
            status['roc_5d'] >= 5.0 and  # Approaching threshold
            status['is_near_90d_low']
        )
        
        status['potential_capitulation_setup'] = (
            status['roc_5d'] <= -5.0 and  # Declining sharply
            status['is_near_6m_low'] and
            status['volume_at_250d_high']
        )
        
        return status


def fetch_data(symbol: str, period: str = "2y", data_fetcher: Optional[DataFetcher] = None) -> Optional[pd.DataFrame]:
    """Fetch historical data from best available source"""
    try:
        if data_fetcher is None:
            data_fetcher = DataFetcher(use_schwab=False)
        
        df = data_fetcher.fetch_historical_data(symbol, period)
        
        if df is None or df.empty:
            st.error(f"No data available for {symbol}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None


def create_boundary_chart(scanner: BoundaryScanner, 
                         thrust_signals: pd.DataFrame,
                         cap_signals: pd.DataFrame,
                         trin_signals: pd.DataFrame) -> go.Figure:
    """Create comprehensive chart showing boundaries and signals"""
    
    df = scanner.data
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{scanner.symbol} Price & Signals',
            '5-Day Rate of Change (%)',
            'Volume Analysis',
            'TRIN (if available)'
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Row 1: Price with signals
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add 90-day low line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['low_90d'],
            name='90-Day Low',
            line=dict(color='orange', dash='dot', width=1),
            opacity=0.5
        ),
        row=1, col=1
    )
    
    # Add signals to price chart
    if len(thrust_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=thrust_signals.index,
                y=thrust_signals['Close'],
                mode='markers',
                name='Thrust Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='lime',
                    line=dict(color='darkgreen', width=2)
                )
            ),
            row=1, col=1
        )
    
    if len(cap_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=cap_signals.index,
                y=cap_signals['Close'],
                mode='markers',
                name='Capitulation Buy',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='cyan',
                    line=dict(color='darkblue', width=2)
                )
            ),
            row=1, col=1
        )
    
    if len(trin_signals) > 0:
        buy_trin = trin_signals[trin_signals['signal_type'] == 'TRIN_VOLUME_BUY']
        sell_trin = trin_signals[trin_signals['signal_type'] == 'TRIN_VOLUME_SELL']
        
        if len(buy_trin) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_trin.index,
                    y=buy_trin['Close'],
                    mode='markers',
                    name='TRIN Buy',
                    marker=dict(
                        symbol='diamond',
                        size=15,
                        color='yellow',
                        line=dict(color='orange', width=2)
                    )
                ),
                row=1, col=1
            )
        
        if len(sell_trin) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_trin.index,
                    y=sell_trin['Close'],
                    mode='markers',
                    name='TRIN Sell',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='red',
                        line=dict(color='darkred', width=2)
                    )
                ),
                row=1, col=1
            )
    
    # Row 2: ROC with thresholds
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['roc_5d'],
            name='5-Day ROC',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=8, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-8, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Row 3: Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['vol_5d_avg'],
            name='5-Day Avg',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['vol_250d_high'],
            name='250-Day High',
            line=dict(color='red', dash='dot')
        ),
        row=3, col=1
    )
    
    # Row 4: TRIN
    if not df['trin'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['trin'],
                name='TRIN Estimate',
                line=dict(color='orange')
            ),
            row=4, col=1
        )
        fig.add_hline(y=0.50, line_dash="dash", line_color="green", row=4, col=1)
        fig.add_hline(y=1.50, line_dash="dash", line_color="red", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


def display_signal_card(signal_date, signal_data, signal_type):
    """Display formatted signal card"""
    
    color_map = {
        'THRUST_BUY': '#00ff00',
        'CAPITULATION_BUY': '#00ffff',
        'TRIN_VOLUME_BUY': '#ffff00',
        'TRIN_VOLUME_SELL': '#ff0000',
        'BREADTH_THRUST': '#00ff88',
        'AD_EXTREME_BUY': '#88ff00',
        'AD_EXTREME_SELL': '#ff8800',
        'HL_EXTREME_BUY': '#0088ff',
        'HL_EXTREME_SELL': '#ff0088'
    }
    
    emoji_map = {
        'THRUST_BUY': '🚀',
        'CAPITULATION_BUY': '⭐',
        'TRIN_VOLUME_BUY': '💎',
        'TRIN_VOLUME_SELL': '⚠️',
        'BREADTH_THRUST': '📈',
        'AD_EXTREME_BUY': '🟢',
        'AD_EXTREME_SELL': '🔴',
        'HL_EXTREME_BUY': '⬆️',
        'HL_EXTREME_SELL': '⬇️'
    }
    
    color = color_map.get(signal_type, '#ffffff')
    emoji = emoji_map.get(signal_type, '📊')
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 24px;">{emoji}</span>
                <span style="font-size: 18px; font-weight: bold; margin-left: 10px;">
                    {signal_type.replace('_', ' ')}
                </span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 20px; font-weight: bold; color: {color};">
                    ${signal_data['Close']:.2f}
                </div>
                <div style="font-size: 12px; color: #888;">
                    {signal_date.strftime('%Y-%m-%d')}
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 14px; color: #bbb;">
            ROC 5D: <strong>{signal_data.get('roc_5d', 0):.2f}%</strong> | 
            {'Volume: <strong>' + f"{signal_data.get('Volume', 0):,.0f}" + '</strong> | ' if 'Volume' in signal_data.index else ''}
            Strength: <strong>{signal_data.get('signal_strength', 0):.2f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Reflecting Boundaries Scanner", layout="wide")
    
    st.title("🎯 Reflecting Boundaries Scanner")
    st.markdown("""
    Based on Milton Berg's "The Boundaries of Technical Analysis" - identifies market turning points 
    by detecting extreme price/volume conditions near multi-period highs/lows.
    
    ### 🟢 BUY Signals (Bottom Boundaries):
    - 🚀 **Thrust Buy**: +8% in 5 days within 4-6 days AFTER 90-day low (major bottoms)
    - ⭐ **Capitulation Buy**: -8% in 5 days + 250-day volume high near 6-month low (panic exhaustion)
    - 💎 **TRIN Buy**: TRIN ≤0.50 + 375-day volume high near 1-year low (extreme buying at bottom)
    - 📈 **Breadth Thrust**: 80%+ advance ratio + 6%+ gain near 90-day low
    - 🟢 **A/D Buy**: 20%+ A/D line spike near 6-month low
    - ⬆️ **H/L Buy**: New lows outnumber highs 2:1 near 1-year low (reversal setup)
    
    ### 🔴 SELL Signals (Top Boundaries):
    - ⚠️ **TRIN Sell**: TRIN ≤0.50 + 375-day volume high near 3-year high ⚠️ (euphoria warning)
    - � **A/D Sell**: 20%+ A/D line drop near 3-year high (distribution)
    - ⬇️ **H/L Sell**: New highs outnumber lows 2:1 near 3-year high (exhaustion warning)
    
    💡 **Key Insight**: Same indicators work for BOTH tops and bottoms - context determines meaning!
    """)
    
    # Display data source status
    data_source_status = "🟢 Schwab API" if SCHWAB_AVAILABLE else "🟡 yfinance (Schwab unavailable)"
    st.sidebar.info(f"**Data Source:** {data_source_status}")
    
    # Sidebar controls
    st.sidebar.header("Scanner Settings")
    
    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Single Symbol", "Multi-Symbol Scan"]
    )
    
    if scan_mode == "Single Symbol":
        symbol = st.sidebar.text_input("Symbol", value="SPY").upper()
        symbols = [symbol]
    else:
        default_symbols_str = ", ".join(DEFAULT_SYMBOLS)
        symbols_input = st.sidebar.text_area(
            "Symbols (comma-separated)",
            value=default_symbols_str
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    
    lookback_period = st.sidebar.selectbox(
        "Lookback Period",
        ["1y", "2y", "5y", "10y"],
        index=1
    )
    
    show_charts = st.sidebar.checkbox("Show Charts", value=True)
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        use_schwab = st.checkbox("Use Schwab API (if available)", value=SCHWAB_AVAILABLE)
        run_backtest = st.checkbox("Run Backtest", value=True)
        enable_alerts = st.checkbox("Enable Alerts", value=False)
        
        if enable_alerts:
            st.info("Configure alerts in alerts_config.json")
        
        show_additional_signals = st.checkbox("Show Additional Berg Indicators", value=True)
    
    if st.sidebar.button("🔍 Scan for Boundaries", type="primary"):
        
        all_results = []
        
        # Initialize components
        data_fetcher = DataFetcher(use_schwab=use_schwab)
        alert_manager = AlertManager() if enable_alerts else None
        backtest_engine = BacktestEngine() if run_backtest else None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(symbols):
            status_text.text(f"Scanning {symbol}...")
            
            # Fetch data
            data = fetch_data(symbol, period=lookback_period, data_fetcher=data_fetcher)
            
            if data is None or len(data) < 100:
                st.warning(f"Insufficient data for {symbol}")
                continue
            
            # Create scanner
            scanner = BoundaryScanner(symbol, data)
            
            # Detect all signals
            thrust_signals = scanner.detect_thrust_signals()
            cap_signals = scanner.detect_capitulation_signals()
            trin_signals = scanner.detect_trin_volume_signals()
            
            # Additional Berg indicators
            breadth_signals = pd.DataFrame()
            ad_signals = pd.DataFrame()
            hl_signals = pd.DataFrame()
            
            if show_additional_signals:
                breadth_signals = scanner.detect_breadth_thrust_signals()
                ad_signals = scanner.detect_advance_decline_extremes()
                hl_signals = scanner.detect_new_high_low_extremes()
            
            # Get current status
            current_status = scanner.get_current_boundary_status()
            
            # Check for recent signals (last 5 days) and send alerts
            if alert_manager:
                recent_date = pd.Timestamp(datetime.now() - timedelta(days=5))
                
                for sig_df, sig_type in [
                    (thrust_signals, 'THRUST_BUY'),
                    (cap_signals, 'CAPITULATION_BUY'),
                    (trin_signals, None),  # Type in dataframe
                    (breadth_signals, 'BREADTH_THRUST'),
                    (ad_signals, None),  # Type in dataframe
                    (hl_signals, None)  # Type in dataframe
                ]:
                    if len(sig_df) > 0:
                        # Handle timezone-aware vs timezone-naive comparison
                        try:
                            if hasattr(sig_df.index, 'tz') and sig_df.index.tz is not None:
                                if recent_date.tz is None:
                                    recent_date = recent_date.tz_localize(sig_df.index.tz)
                            recent_signals = sig_df[sig_df.index >= recent_date]
                        except (TypeError, AttributeError):
                            # Fallback: convert to tz-naive for comparison
                            sig_df_copy = sig_df.copy()
                            sig_df_copy.index = sig_df_copy.index.tz_localize(None) if hasattr(sig_df_copy.index, 'tz') else sig_df_copy.index
                            recent_date_naive = recent_date.tz_localize(None) if hasattr(recent_date, 'tz') and recent_date.tz else recent_date
                            recent_signals = sig_df_copy[sig_df_copy.index >= recent_date_naive]
                        for sig_date, sig_row in recent_signals.iterrows():
                            signal_type = sig_type if sig_type else sig_row.get('signal_type', 'UNKNOWN')
                            alert_data = {
                                'date': sig_date.strftime('%Y-%m-%d'),
                                'price': sig_row.get('Close', 0),
                                'roc_5d': sig_row.get('roc_5d', 0),
                                'volume': sig_row.get('Volume', 0),
                                'strength': sig_row.get('signal_strength', 0),
                                'context': current_status['boundary_context']
                            }
                            alert_manager.send_alert(symbol, signal_type, alert_data)
            
            # Run backtests
            backtest_results = {}
            if backtest_engine:
                if len(thrust_signals) > 0:
                    backtest_results['thrust'] = backtest_engine.backtest_signals(
                        scanner, thrust_signals, 'THRUST_BUY'
                    )
                if len(cap_signals) > 0:
                    backtest_results['capitulation'] = backtest_engine.backtest_signals(
                        scanner, cap_signals, 'CAPITULATION_BUY'
                    )
                if len(trin_signals) > 0:
                    trin_buy = trin_signals[trin_signals['signal_type'] == 'TRIN_VOLUME_BUY']
                    if len(trin_buy) > 0:
                        backtest_results['trin_buy'] = backtest_engine.backtest_signals(
                            scanner, trin_buy, 'TRIN_VOLUME_BUY'
                        )
            
            # Store results
            result = {
                'symbol': symbol,
                'scanner': scanner,
                'thrust_signals': thrust_signals,
                'cap_signals': cap_signals,
                'trin_signals': trin_signals,
                'breadth_signals': breadth_signals,
                'ad_signals': ad_signals,
                'hl_signals': hl_signals,
                'current_status': current_status,
                'backtest_results': backtest_results
            }
            all_results.append(result)
            
            progress_bar.progress((idx + 1) / len(symbols))
        
        status_text.text("Scan complete!")
        
        # Display results
        st.header("📊 Scan Results")
        
        for result in all_results:
            symbol = result['symbol']
            current_status = result['current_status']
            
            with st.expander(f"**{symbol}** - {current_status['boundary_context']}", expanded=(len(symbols) == 1)):
                
                # Current status summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_status['current_price']:.2f}")
                    st.metric("5-Day ROC", f"{current_status['roc_5d']:.2f}%")
                
                with col2:
                    st.metric("Days Since 90D Low", f"{current_status['days_since_90d_low']:.0f}")
                    st.metric("Days Since 6M Low", f"{current_status['days_since_6m_low']:.0f}")
                
                with col3:
                    st.metric("Days Since 1Y Low", f"{current_status['days_since_1y_low']:.0f}")
                    st.metric("Days Since 3Y High", f"{current_status['days_since_3y_high']:.0f}")
                
                with col4:
                    vol_250 = "✅" if current_status['volume_at_250d_high'] else "❌"
                    vol_375 = "✅" if current_status['volume_at_375d_high'] else "❌"
                    st.metric("Vol 250D High", vol_250)
                    st.metric("Vol 375D High", vol_375)
                
                # Signal Summary - Buy vs Sell
                st.markdown("---")
                st.subheader("🎯 Signal Summary")
                
                # Count buy vs sell signals
                buy_count = 0
                sell_count = 0
                
                # Count from all signal types
                buy_count += len(result['thrust_signals'])
                buy_count += len(result['cap_signals'])
                
                # TRIN signals
                if len(result['trin_signals']) > 0:
                    buy_count += len(result['trin_signals'][result['trin_signals']['signal_type'] == 'TRIN_VOLUME_BUY'])
                    sell_count += len(result['trin_signals'][result['trin_signals']['signal_type'] == 'TRIN_VOLUME_SELL'])
                
                # Additional signals
                if show_additional_signals:
                    buy_count += len(result['breadth_signals'])
                    
                    if len(result['ad_signals']) > 0:
                        buy_count += len(result['ad_signals'][result['ad_signals']['signal_type'] == 'AD_EXTREME_BUY'])
                        sell_count += len(result['ad_signals'][result['ad_signals']['signal_type'] == 'AD_EXTREME_SELL'])
                    
                    if len(result['hl_signals']) > 0:
                        buy_count += len(result['hl_signals'][result['hl_signals']['signal_type'] == 'HL_EXTREME_BUY'])
                        sell_count += len(result['hl_signals'][result['hl_signals']['signal_type'] == 'HL_EXTREME_SELL'])
                
                col_buy, col_sell, col_bias = st.columns(3)
                
                with col_buy:
                    st.metric("🟢 BUY Signals", buy_count)
                
                with col_sell:
                    st.metric("🔴 SELL Signals", sell_count)
                
                with col_bias:
                    if buy_count > sell_count:
                        bias = "🟢 BULLISH"
                        bias_color = "green"
                    elif sell_count > buy_count:
                        bias = "🔴 BEARISH"
                        bias_color = "red"
                    else:
                        bias = "⚪ NEUTRAL"
                        bias_color = "gray"
                    
                    st.markdown(f"<h3 style='color: {bias_color};'>{bias}</h3>", unsafe_allow_html=True)
                
                # Potential setups
                if current_status['potential_thrust_setup']:
                    st.success("🚀 **Potential Thrust Setup** - Price rising sharply near 90-day low")
                
                if current_status['potential_capitulation_setup']:
                    st.success("⭐ **Potential Capitulation Setup** - Price declining with volume spike near 6-month low")
                
                st.markdown("---")
                
                # Historical signals
                st.subheader("📜 Historical Signals Detail")
                
                tabs = ["Thrust", "Capitulation", "TRIN"]
                if show_additional_signals:
                    tabs.extend(["Breadth", "A/D Extremes", "H/L Extremes"])
                
                if run_backtest:
                    tabs.append("📊 Backtest Results")
                
                tab_objects = st.tabs(tabs)
                
                with tab_objects[0]:  # Thrust
                    thrust_signals = result['thrust_signals']
                    if len(thrust_signals) > 0:
                        st.success(f"Found {len(thrust_signals)} thrust signals")
                        for date, signal in thrust_signals.iterrows():
                            display_signal_card(date, signal, 'THRUST_BUY')
                    else:
                        st.info("No thrust signals found")
                
                with tab_objects[1]:  # Capitulation
                    cap_signals = result['cap_signals']
                    if len(cap_signals) > 0:
                        st.success(f"Found {len(cap_signals)} capitulation signals")
                        for date, signal in cap_signals.iterrows():
                            display_signal_card(date, signal, 'CAPITULATION_BUY')
                    else:
                        st.info("No capitulation signals found")
                
                with tab_objects[2]:  # TRIN
                    trin_signals = result['trin_signals']
                    if len(trin_signals) > 0:
                        st.success(f"Found {len(trin_signals)} TRIN signals")
                        for date, signal in trin_signals.iterrows():
                            signal_type = signal['signal_type']
                            display_signal_card(date, signal, signal_type)
                    else:
                        st.info("No TRIN signals found (or not applicable for this symbol)")
                
                # Additional signals tabs
                if show_additional_signals:
                    with tab_objects[3]:  # Breadth
                        breadth_signals = result['breadth_signals']
                        if len(breadth_signals) > 0:
                            st.success(f"Found {len(breadth_signals)} breadth thrust signals")
                            for date, signal in breadth_signals.iterrows():
                                display_signal_card(date, signal, 'BREADTH_THRUST')
                        else:
                            st.info("No breadth thrust signals found")
                    
                    with tab_objects[4]:  # A/D
                        ad_signals = result['ad_signals']
                        if len(ad_signals) > 0:
                            st.success(f"Found {len(ad_signals)} A/D extreme signals")
                            for date, signal in ad_signals.iterrows():
                                signal_type = signal['signal_type']
                                display_signal_card(date, signal, signal_type)
                        else:
                            st.info("No A/D extreme signals found")
                    
                    with tab_objects[5]:  # H/L
                        hl_signals = result['hl_signals']
                        if len(hl_signals) > 0:
                            st.success(f"Found {len(hl_signals)} new high/low extreme signals")
                            for date, signal in hl_signals.iterrows():
                                signal_type = signal['signal_type']
                                display_signal_card(date, signal, signal_type)
                        else:
                            st.info("No new high/low extreme signals found")
                
                # Backtest results tab
                if run_backtest:
                    backtest_tab_idx = len(tabs) - 1
                    with tab_objects[backtest_tab_idx]:
                        st.subheader("📊 Backtest Performance")
                        
                        backtest_results = result['backtest_results']
                        
                        if not backtest_results:
                            st.info("No backtest data available")
                        else:
                            for signal_name, bt_df in backtest_results.items():
                                if len(bt_df) == 0:
                                    continue
                                
                                st.markdown(f"### {signal_name.replace('_', ' ').title()} Signals")
                                
                                # Generate performance summary
                                perf_summary = backtest_engine.generate_performance_summary(bt_df)
                                
                                if 'holding_periods' in perf_summary:
                                    cols = st.columns(len(perf_summary['holding_periods']))
                                    
                                    for idx, (days, stats) in enumerate(perf_summary['holding_periods'].items()):
                                        with cols[idx]:
                                            st.metric(f"{days}-Day Return", f"{stats['avg_return']:.2f}%")
                                            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                                            st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
                                
                                # Show detailed results table
                                with st.expander("View Detailed Results"):
                                    display_df = bt_df.copy()
                                    display_df['signal_date'] = display_df['signal_date'].dt.strftime('%Y-%m-%d')
                                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Chart
                if show_charts:
                    st.subheader("Interactive Chart")
                    fig = create_boundary_chart(
                        result['scanner'],
                        result['thrust_signals'],
                        result['cap_signals'],
                        result['trin_signals']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        if len(all_results) > 1:
            st.header("📋 Summary Table")
            
            summary_data = []
            for result in all_results:
                status = result['current_status']
                
                # Count buy and sell signals
                buy_count = len(result['thrust_signals']) + len(result['cap_signals'])
                sell_count = 0
                
                if len(result['trin_signals']) > 0:
                    buy_count += len(result['trin_signals'][result['trin_signals']['signal_type'] == 'TRIN_VOLUME_BUY'])
                    sell_count += len(result['trin_signals'][result['trin_signals']['signal_type'] == 'TRIN_VOLUME_SELL'])
                
                if show_additional_signals:
                    buy_count += len(result['breadth_signals'])
                    
                    if len(result['ad_signals']) > 0:
                        buy_count += len(result['ad_signals'][result['ad_signals']['signal_type'] == 'AD_EXTREME_BUY'])
                        sell_count += len(result['ad_signals'][result['ad_signals']['signal_type'] == 'AD_EXTREME_SELL'])
                    
                    if len(result['hl_signals']) > 0:
                        buy_count += len(result['hl_signals'][result['hl_signals']['signal_type'] == 'HL_EXTREME_BUY'])
                        sell_count += len(result['hl_signals'][result['hl_signals']['signal_type'] == 'HL_EXTREME_SELL'])
                
                # Determine bias
                if buy_count > sell_count:
                    bias = '🟢 BULLISH'
                elif sell_count > buy_count:
                    bias = '🔴 BEARISH'
                else:
                    bias = '⚪ NEUTRAL'
                
                summary_data.append({
                    'Symbol': status['symbol'],
                    'Price': f"${status['current_price']:.2f}",
                    'ROC 5D': f"{status['roc_5d']:.1f}%",
                    'Context': status['boundary_context'],
                    '🟢 BUY': buy_count,
                    '🔴 SELL': sell_count,
                    'Bias': bias,
                    'Near Low': '✅' if status['is_near_90d_low'] else '❌',
                    'Vol Extreme': '✅' if status['volume_at_250d_high'] else '❌'
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
