#!/usr/bin/env python3
"""
AI Newsletter Scanner Service - Runs Every 30 Minutes
Collects stock scores, options flow, and stores in database for AI Copilot

Improvements over original:
1. RSI integration for momentum scoring
2. Options flow sentiment from Schwab API
3. IV Rank for options premium analysis  
4. Gamma wall detection for support/resistance
5. Multiple timeframe analysis (weekly, monthly, quarterly plays)
6. Bearish setup detection (not just bullish)
7. Historical score tracking for momentum detection
"""

import sys
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yfinance as yf
import pandas as pd
import numpy as np

# Try imports
try:
    from src.api.schwab_client import SchwabClient
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False

from src.theme_tracker import THEMES

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'newsletter_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "newsletter_scanner.db"
HISTORY_FILE = Path(__file__).parent.parent / "data" / "newsletter_scan_history.json"


class NewsletterScanner:
    """Enhanced stock scanner with options flow integration"""
    
    def __init__(self):
        self.schwab_client = None
        if SCHWAB_AVAILABLE:
            try:
                self.schwab_client = SchwabClient()
                logger.info("✅ Schwab client initialized")
            except Exception as e:
                logger.warning(f"Schwab client unavailable: {e}")
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ticker TEXT NOT NULL,
                    theme TEXT,
                    description TEXT,
                    
                    -- Scores
                    opportunity_score REAL,
                    technical_score REAL,
                    momentum_score REAL,
                    volume_score REAL,
                    
                    -- Price data
                    current_price REAL,
                    week_return REAL,
                    month_return REAL,
                    quarter_return REAL,
                    
                    -- Technical indicators
                    rsi REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    above_sma20 INTEGER,
                    above_sma50 INTEGER,
                    above_sma200 INTEGER,
                    
                    -- Volume
                    volume_ratio REAL,
                    avg_volume REAL,
                    
                    -- Volatility
                    volatility_ratio REAL,
                    near_52w_high INTEGER,
                    distance_from_high REAL,
                    
                    -- Options data (if available)
                    call_volume INTEGER,
                    put_volume INTEGER,
                    put_call_ratio REAL,
                    options_sentiment TEXT,
                    iv_rank REAL,
                    
                    -- Gamma walls
                    gamma_support REAL,
                    gamma_resistance REAL,
                    
                    -- Setup type
                    setup_type TEXT,  -- BULLISH, BEARISH, NEUTRAL
                    timeframe TEXT,   -- WEEKLY, MONTHLY, QUARTERLY
                    
                    -- Catalysts
                    has_earnings_soon INTEGER,
                    days_to_earnings INTEGER,
                    has_upgrade INTEGER,
                    has_downgrade INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scans_ticker_time 
                ON scans(ticker, scan_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scans_score 
                ON scans(opportunity_score DESC)
            """)
            
            # Summary table for quick access
            conn.execute("""
                CREATE TABLE IF NOT EXISTS latest_scores (
                    ticker TEXT PRIMARY KEY,
                    theme TEXT,
                    description TEXT,
                    opportunity_score REAL,
                    setup_type TEXT,
                    timeframe TEXT,
                    current_price REAL,
                    week_return REAL,
                    rsi REAL,
                    options_sentiment TEXT,
                    last_updated TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {DB_PATH}")
    
    def fetch_stock_data(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch stock data with caching"""
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        if df is None or len(df) < period + 1:
            return None
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
        except:
            return None
    
    def calculate_technical_score(self, ticker: str, df: pd.DataFrame) -> Dict:
        """Enhanced technical scoring with RSI and multiple timeframes"""
        if df is None or len(df) < 50:
            return {'score': 0, 'signals': {}}
        
        signals = {}
        score = 0
        
        current = float(df['Close'].iloc[-1])
        
        # Moving averages
        sma_20 = float(df['Close'].rolling(20).mean().iloc[-1])
        sma_50 = float(df['Close'].rolling(50).mean().iloc[-1])
        sma_200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else sma_50
        
        signals['current_price'] = current
        signals['sma_20'] = sma_20
        signals['sma_50'] = sma_50
        signals['sma_200'] = sma_200
        signals['above_sma20'] = current > sma_20
        signals['above_sma50'] = current > sma_50
        signals['above_sma200'] = current > sma_200
        
        # 1. Trend strength (25 points)
        if current > sma_20 > sma_50 > sma_200:
            score += 25
            signals['trend'] = "STRONG_UPTREND"
        elif current > sma_20 > sma_50:
            score += 18
            signals['trend'] = "UPTREND"
        elif current > sma_50:
            score += 10
            signals['trend'] = "MODERATE_UP"
        elif current < sma_20 < sma_50 < sma_200:
            score -= 5  # Penalty for downtrend
            signals['trend'] = "STRONG_DOWNTREND"
        else:
            signals['trend'] = "NEUTRAL"
        
        # 2. RSI Analysis (20 points)
        rsi = self.calculate_rsi(df)
        signals['rsi'] = rsi
        
        if rsi:
            if 50 <= rsi <= 70:  # Bullish but not overbought
                score += 20
                signals['rsi_signal'] = "BULLISH_MOMENTUM"
            elif 30 <= rsi < 50:  # Potential reversal
                score += 10
                signals['rsi_signal'] = "POTENTIAL_REVERSAL"
            elif rsi > 70:  # Overbought
                score += 5
                signals['rsi_signal'] = "OVERBOUGHT"
            elif rsi < 30:  # Oversold - could be bullish reversal
                score += 15
                signals['rsi_signal'] = "OVERSOLD"
        
        # 3. Momentum across timeframes (20 points)
        week_return = ((current / df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
        month_return = ((current / df['Close'].iloc[-21]) - 1) * 100 if len(df) >= 21 else 0
        quarter_return = ((current / df['Close'].iloc[-63]) - 1) * 100 if len(df) >= 63 else 0
        
        signals['week_return'] = week_return
        signals['month_return'] = month_return
        signals['quarter_return'] = quarter_return
        
        # Weekly momentum
        if week_return > 5:
            score += 10
        elif week_return > 2:
            score += 5
        elif week_return < -5:
            score -= 5
        
        # Monthly momentum
        if month_return > 10:
            score += 10
        elif month_return > 5:
            score += 5
        
        # 4. Volume analysis (15 points)
        avg_volume = float(df['Volume'].rolling(20).mean().iloc[-1])
        recent_volume = float(df['Volume'].iloc[-1])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        signals['volume_ratio'] = volume_ratio
        signals['avg_volume'] = avg_volume
        
        if volume_ratio > 2:
            score += 15
            signals['volume_signal'] = "SURGE"
        elif volume_ratio > 1.5:
            score += 10
            signals['volume_signal'] = "ELEVATED"
        elif volume_ratio > 1.2:
            score += 5
            signals['volume_signal'] = "ABOVE_AVG"
        else:
            signals['volume_signal'] = "NORMAL"
        
        # 5. Volatility compression (10 points)
        recent_vol = df['Close'].pct_change().tail(10).std()
        hist_vol = df['Close'].pct_change().tail(60).std()
        vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1
        
        signals['volatility_ratio'] = vol_ratio
        
        if 0.5 < vol_ratio < 0.8:
            score += 10
            signals['volatility_signal'] = "COMPRESSION"
        elif vol_ratio > 1.5:
            score += 5
            signals['volatility_signal'] = "EXPANDING"
        else:
            signals['volatility_signal'] = "NORMAL"
        
        # 6. Distance from 52-week high (10 points)
        high_52w = float(df['High'].max())
        distance_from_high = ((high_52w - current) / high_52w) * 100
        
        signals['distance_from_high'] = distance_from_high
        signals['near_52w_high'] = distance_from_high < 5
        
        if distance_from_high < 3:
            score += 10
            signals['breakout_signal'] = "AT_HIGHS"
        elif distance_from_high < 10:
            score += 5
            signals['breakout_signal'] = "NEAR_HIGHS"
        elif distance_from_high > 30:
            signals['breakout_signal'] = "BEATEN_DOWN"
        else:
            signals['breakout_signal'] = "MID_RANGE"
        
        signals['score'] = max(0, min(100, score))
        
        return signals
    
    def get_options_data(self, ticker: str) -> Dict:
        """Get options flow data from Schwab"""
        if not self.schwab_client:
            return {}
        
        try:
            chain = self.schwab_client.get_options_chain(ticker)
            if not chain:
                return {}
            
            underlying_price = chain.get('underlyingPrice', 0)
            calls = chain.get('callExpDateMap', {})
            puts = chain.get('putExpDateMap', {})
            
            total_call_vol = 0
            total_put_vol = 0
            ivs = []
            gamma_strikes = {}
            
            for exp_date, strikes in calls.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    for opt in options:
                        vol = opt.get('totalVolume', 0)
                        oi = opt.get('openInterest', 0)
                        iv = opt.get('volatility', 0)
                        dte = opt.get('daysToExpiration', 0)
                        
                        if dte <= 90:  # Only near-term
                            total_call_vol += vol
                            if iv > 0 and underlying_price * 0.95 <= strike <= underlying_price * 1.05:
                                ivs.append(iv)
                            
                            gamma_strikes[strike] = gamma_strikes.get(strike, 0) + oi
            
            for exp_date, strikes in puts.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    for opt in options:
                        vol = opt.get('totalVolume', 0)
                        oi = opt.get('openInterest', 0)
                        dte = opt.get('daysToExpiration', 0)
                        
                        if dte <= 90:
                            total_put_vol += vol
                            gamma_strikes[strike] = gamma_strikes.get(strike, 0) + oi
            
            # Calculate metrics
            put_call_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 1
            
            if put_call_ratio < 0.7:
                sentiment = "BULLISH"
            elif put_call_ratio > 1.0:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            # IV Rank (simplified)
            current_iv = sum(ivs) / len(ivs) if ivs else 0
            if current_iv < 25:
                iv_rank = 20
            elif current_iv < 40:
                iv_rank = 40
            elif current_iv < 55:
                iv_rank = 60
            else:
                iv_rank = 80
            
            # Gamma walls
            sorted_strikes = sorted(gamma_strikes.items(), key=lambda x: x[1], reverse=True)[:5]
            support = [s[0] for s in sorted_strikes if s[0] < underlying_price]
            resistance = [s[0] for s in sorted_strikes if s[0] > underlying_price]
            
            return {
                'call_volume': total_call_vol,
                'put_volume': total_put_vol,
                'put_call_ratio': put_call_ratio,
                'sentiment': sentiment,
                'iv_rank': iv_rank,
                'current_iv': current_iv,
                'gamma_support': support[0] if support else None,
                'gamma_resistance': resistance[0] if resistance else None
            }
        except Exception as e:
            logger.error(f"Options error for {ticker}: {e}")
            return {}
    
    def get_earnings_info(self, ticker: str) -> Dict:
        """Get earnings date"""
        try:
            t = yf.Ticker(ticker)
            calendar = t.calendar
            
            if calendar and 'Earnings Date' in calendar:
                earnings_list = calendar['Earnings Date']
                if earnings_list:
                    next_earnings = earnings_list[0] if isinstance(earnings_list, list) else earnings_list
                    if hasattr(next_earnings, 'date'):
                        next_date = next_earnings if isinstance(next_earnings, datetime) else datetime.combine(next_earnings, datetime.min.time())
                    else:
                        next_date = next_earnings
                    
                    days = (next_date.date() if hasattr(next_date, 'date') else next_date) - datetime.now().date()
                    days_to_earnings = days.days if hasattr(days, 'days') else days
                    
                    return {
                        'has_earnings_soon': days_to_earnings <= 14,
                        'days_to_earnings': days_to_earnings
                    }
        except:
            pass
        return {'has_earnings_soon': False, 'days_to_earnings': None}
    
    def determine_setup_type(self, signals: Dict, options: Dict) -> Tuple[str, str]:
        """Determine if setup is bullish/bearish and timeframe"""
        score = signals.get('score', 0)
        trend = signals.get('trend', 'NEUTRAL')
        rsi = signals.get('rsi', 50)
        week_ret = signals.get('week_return', 0)
        month_ret = signals.get('month_return', 0)
        options_sentiment = options.get('sentiment', 'NEUTRAL')
        
        # Setup type
        bullish_signals = 0
        bearish_signals = 0
        
        if 'UPTREND' in trend:
            bullish_signals += 2
        elif 'DOWNTREND' in trend:
            bearish_signals += 2
        
        if rsi and rsi > 50:
            bullish_signals += 1
        elif rsi and rsi < 50:
            bearish_signals += 1
        
        if week_ret > 2:
            bullish_signals += 1
        elif week_ret < -2:
            bearish_signals += 1
        
        if options_sentiment == 'BULLISH':
            bullish_signals += 1
        elif options_sentiment == 'BEARISH':
            bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            setup_type = 'BULLISH'
        elif bearish_signals > bullish_signals + 1:
            setup_type = 'BEARISH'
        else:
            setup_type = 'NEUTRAL'
        
        # Timeframe
        if signals.get('volatility_signal') == 'COMPRESSION' or week_ret > 5:
            timeframe = 'WEEKLY'
        elif month_ret > 10 and score > 70:
            timeframe = 'MONTHLY'
        else:
            timeframe = 'QUARTERLY'
        
        return setup_type, timeframe
    
    def scan_stock(self, ticker: str, theme: str, description: str) -> Dict:
        """Complete scan of a single stock"""
        logger.debug(f"Scanning {ticker}...")
        
        # Get price data
        df = self.fetch_stock_data(ticker)
        if df is None or len(df) < 50:
            return None
        
        # Technical analysis
        signals = self.calculate_technical_score(ticker, df)
        
        # Options data (if available)
        options = self.get_options_data(ticker)
        
        # Earnings
        earnings = self.get_earnings_info(ticker)
        
        # Determine setup
        setup_type, timeframe = self.determine_setup_type(signals, options)
        
        # Calculate final opportunity score
        base_score = signals.get('score', 0)
        
        # Bonus for options confluence
        if options.get('sentiment') == 'BULLISH' and setup_type == 'BULLISH':
            base_score += 10
        elif options.get('sentiment') == 'BEARISH' and setup_type == 'BEARISH':
            base_score += 10
        
        # Earnings bonus/penalty
        if earnings.get('has_earnings_soon'):
            base_score += 5  # Catalyst
        
        opportunity_score = max(0, min(100, base_score))
        
        return {
            'ticker': ticker,
            'theme': theme,
            'description': description,
            'opportunity_score': opportunity_score,
            'technical_score': signals.get('score', 0),
            'current_price': signals.get('current_price'),
            'week_return': signals.get('week_return'),
            'month_return': signals.get('month_return'),
            'quarter_return': signals.get('quarter_return'),
            'rsi': signals.get('rsi'),
            'sma_20': signals.get('sma_20'),
            'sma_50': signals.get('sma_50'),
            'sma_200': signals.get('sma_200'),
            'above_sma20': signals.get('above_sma20'),
            'above_sma50': signals.get('above_sma50'),
            'above_sma200': signals.get('above_sma200'),
            'volume_ratio': signals.get('volume_ratio'),
            'avg_volume': signals.get('avg_volume'),
            'volatility_ratio': signals.get('volatility_ratio'),
            'near_52w_high': signals.get('near_52w_high'),
            'distance_from_high': signals.get('distance_from_high'),
            'call_volume': options.get('call_volume'),
            'put_volume': options.get('put_volume'),
            'put_call_ratio': options.get('put_call_ratio'),
            'options_sentiment': options.get('sentiment'),
            'iv_rank': options.get('iv_rank'),
            'gamma_support': options.get('gamma_support'),
            'gamma_resistance': options.get('gamma_resistance'),
            'setup_type': setup_type,
            'timeframe': timeframe,
            'has_earnings_soon': earnings.get('has_earnings_soon'),
            'days_to_earnings': earnings.get('days_to_earnings'),
            'trend': signals.get('trend'),
            'rsi_signal': signals.get('rsi_signal'),
            'volume_signal': signals.get('volume_signal')
        }
    
    def run_full_scan(self) -> List[Dict]:
        """Run full scan of all theme stocks"""
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"Starting newsletter scan at {datetime.now()}")
        logger.info("=" * 60)
        
        results = []
        all_stocks = []
        
        # Collect all stocks
        for theme, stocks in THEMES.items():
            for ticker, desc in stocks.items():
                all_stocks.append((ticker, theme, desc))
        
        logger.info(f"Scanning {len(all_stocks)} stocks across {len(THEMES)} themes")
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.scan_stock, ticker, theme, desc): ticker
                for ticker, theme, desc in all_stocks
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error scanning {ticker}: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Save to database
        self._save_results(results)
        
        # Update JSON history (for backward compatibility)
        self._update_json_history(results)
        
        elapsed = time.time() - start_time
        logger.info(f"Scan complete: {len(results)} stocks in {elapsed:.1f}s")
        logger.info(f"Top 5: {', '.join([f\"{r['ticker']}({r['opportunity_score']})\" for r in results[:5]])}")
        
        return results
    
    def _save_results(self, results: List[Dict]):
        """Save results to database"""
        with sqlite3.connect(DB_PATH) as conn:
            for r in results:
                # Insert into scans table
                conn.execute("""
                    INSERT INTO scans (
                        ticker, theme, description, opportunity_score, technical_score,
                        current_price, week_return, month_return, quarter_return,
                        rsi, sma_20, sma_50, sma_200, above_sma20, above_sma50, above_sma200,
                        volume_ratio, avg_volume, volatility_ratio, near_52w_high, distance_from_high,
                        call_volume, put_volume, put_call_ratio, options_sentiment, iv_rank,
                        gamma_support, gamma_resistance, setup_type, timeframe,
                        has_earnings_soon, days_to_earnings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r['ticker'], r['theme'], r['description'], r['opportunity_score'], r['technical_score'],
                    r['current_price'], r['week_return'], r['month_return'], r.get('quarter_return'),
                    r['rsi'], r['sma_20'], r['sma_50'], r['sma_200'], r['above_sma20'], r['above_sma50'], r.get('above_sma200'),
                    r['volume_ratio'], r['avg_volume'], r['volatility_ratio'], r['near_52w_high'], r['distance_from_high'],
                    r.get('call_volume'), r.get('put_volume'), r.get('put_call_ratio'), r.get('options_sentiment'), r.get('iv_rank'),
                    r.get('gamma_support'), r.get('gamma_resistance'), r['setup_type'], r['timeframe'],
                    r.get('has_earnings_soon'), r.get('days_to_earnings')
                ))
                
                # Upsert into latest_scores
                conn.execute("""
                    INSERT OR REPLACE INTO latest_scores (
                        ticker, theme, description, opportunity_score, setup_type, timeframe,
                        current_price, week_return, rsi, options_sentiment, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r['ticker'], r['theme'], r['description'], r['opportunity_score'], r['setup_type'], r['timeframe'],
                    r['current_price'], r['week_return'], r['rsi'], r.get('options_sentiment'), datetime.now()
                ))
            
            conn.commit()
        logger.info(f"Saved {len(results)} results to database")
    
    def _update_json_history(self, results: List[Dict]):
        """Update JSON history file for backward compatibility"""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            else:
                history = {}
            
            today = datetime.now().strftime('%Y-%m-%d')
            today_scores = {}
            
            for r in results:
                today_scores[r['ticker']] = {
                    'score': r['opportunity_score'],
                    'price': r['current_price'],
                    'week_return': r['week_return'],
                    'month_return': r['month_return'],
                    'theme': r['theme'],
                    'description': r['description'],
                    'setup_type': r['setup_type'],
                    'timeframe': r['timeframe']
                }
            
            history[today] = today_scores
            
            # Keep last 30 days
            dates = sorted(history.keys(), reverse=True)[:30]
            history = {d: history[d] for d in dates}
            
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            logger.info(f"Updated JSON history: {len(history)} days")
        except Exception as e:
            logger.error(f"Error updating JSON history: {e}")
    
    def get_latest_scores(self) -> List[Dict]:
        """Get latest scores from database"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM latest_scores 
                ORDER BY opportunity_score DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_improving_stocks(self, days: int = 7) -> List[Dict]:
        """Get stocks with improving scores over time"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get average scores from last N days vs previous N days
            cursor = conn.execute("""
                WITH recent AS (
                    SELECT ticker, AVG(opportunity_score) as recent_avg
                    FROM scans
                    WHERE scan_time >= datetime('now', '-3 days')
                    GROUP BY ticker
                ),
                older AS (
                    SELECT ticker, AVG(opportunity_score) as older_avg
                    FROM scans  
                    WHERE scan_time >= datetime('now', '-7 days')
                      AND scan_time < datetime('now', '-3 days')
                    GROUP BY ticker
                )
                SELECT 
                    r.ticker,
                    r.recent_avg,
                    o.older_avg,
                    (r.recent_avg - o.older_avg) as improvement
                FROM recent r
                JOIN older o ON r.ticker = o.ticker
                WHERE r.recent_avg > o.older_avg
                ORDER BY improvement DESC
                LIMIT 20
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_bullish_setups(self, min_score: int = 70) -> List[Dict]:
        """Get current bullish setups"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE setup_type = 'BULLISH' AND opportunity_score >= ?
                ORDER BY opportunity_score DESC
            """, (min_score,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_bearish_setups(self, min_score: int = 60) -> List[Dict]:
        """Get current bearish setups (for shorts/puts)"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE setup_type = 'BEARISH' AND opportunity_score >= ?
                ORDER BY opportunity_score DESC
            """, (min_score,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_weekly_plays(self) -> List[Dict]:
        """Get stocks suitable for weekly options plays"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE timeframe = 'WEEKLY' AND opportunity_score >= 70
                ORDER BY opportunity_score DESC
                LIMIT 10
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_monthly_plays(self) -> List[Dict]:
        """Get stocks suitable for monthly swing trades"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM latest_scores
                WHERE timeframe = 'MONTHLY' AND opportunity_score >= 65
                ORDER BY opportunity_score DESC
                LIMIT 15
            """)
            return [dict(row) for row in cursor.fetchall()]


def main():
    """Main entry point for service"""
    scanner = NewsletterScanner()
    results = scanner.run_full_scan()
    
    print(f"\n📊 Scan Summary:")
    print(f"Total stocks: {len(results)}")
    print(f"Bullish setups: {len([r for r in results if r['setup_type'] == 'BULLISH'])}")
    print(f"Bearish setups: {len([r for r in results if r['setup_type'] == 'BEARISH'])}")
    print(f"Weekly plays: {len([r for r in results if r['timeframe'] == 'WEEKLY'])}")
    
    print(f"\n🔥 Top 10 Opportunities:")
    for i, r in enumerate(results[:10], 1):
        print(f"{i}. {r['ticker']:5} | Score: {r['opportunity_score']:3.0f} | {r['setup_type']:8} | {r['timeframe']:10} | ${r['current_price']:.2f}")
    
    return results


if __name__ == "__main__":
    main()
