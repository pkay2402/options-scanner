"""
AI Trading Copilot - Synthesizes market data and provides actionable insights
Uses Groq's free Llama 3.1 API
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import requests

# Try to import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try to import Schwab client for live data
try:
    from src.api.schwab_client import SchwabClient
    SCHWAB_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.api.schwab_client import SchwabClient
        SCHWAB_AVAILABLE = True
    except:
        SCHWAB_AVAILABLE = False


class TradingCopilot:
    """AI-powered trading assistant that synthesizes multiple data sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = "llama-3.3-70b-versatile"  # Updated model - free on Groq!
        self.project_root = Path(__file__).parent.parent.parent
        
        if GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
        
        # Initialize Schwab client for live data
        self.schwab_client = None
        if SCHWAB_AVAILABLE:
            try:
                self.schwab_client = SchwabClient()
            except:
                pass
    
    def is_available(self) -> bool:
        """Check if AI is properly configured"""
        return self.client is not None and self.api_key is not None
    
    # ==================== LIVE DATA FETCHING ====================
    
    def get_live_quote(self, symbol: str) -> Dict:
        """Fetch live quote data from Schwab API"""
        if not self.schwab_client:
            return {}
        
        try:
            quote_data = self.schwab_client.get_quote(symbol.upper())
            if quote_data and symbol.upper() in quote_data:
                q = quote_data[symbol.upper()]
                # Extract key fields from quote
                quote = q.get('quote', q)
                return {
                    'symbol': symbol.upper(),
                    'price': quote.get('lastPrice', quote.get('mark', 0)),
                    'change': quote.get('netChange', 0),
                    'change_pct': quote.get('netPercentChangeInDouble', 0),
                    'volume': quote.get('totalVolume', 0),
                    'high': quote.get('highPrice', quote.get('52WkHigh', 0)),
                    'low': quote.get('lowPrice', quote.get('52WkLow', 0)),
                    'open': quote.get('openPrice', 0),
                    'prev_close': quote.get('closePrice', 0),
                    'bid': quote.get('bidPrice', 0),
                    'ask': quote.get('askPrice', 0),
                    'pe_ratio': quote.get('peRatio', 0),
                    'week_52_high': quote.get('52WkHigh', 0),
                    'week_52_low': quote.get('52WkLow', 0),
                }
        except Exception as e:
            pass
        return {}
    
    def get_price_history(self, symbol: str, days: int = 30) -> Dict:
        """Fetch price history for technical analysis"""
        if not self.schwab_client:
            return {}
        
        try:
            history = self.schwab_client.get_price_history(
                symbol.upper(),
                period_type="month",
                period=1,
                frequency_type="daily",
                frequency=1
            )
            if history and 'candles' in history:
                candles = history['candles']
                if candles:
                    # Calculate basic technical indicators
                    closes = [c['close'] for c in candles]
                    highs = [c['high'] for c in candles]
                    lows = [c['low'] for c in candles]
                    volumes = [c['volume'] for c in candles]
                    
                    # Simple moving averages
                    sma_10 = sum(closes[-10:]) / min(10, len(closes)) if closes else 0
                    sma_20 = sum(closes[-20:]) / min(20, len(closes)) if closes else 0
                    
                    # Recent performance
                    current = closes[-1] if closes else 0
                    week_ago = closes[-5] if len(closes) >= 5 else closes[0] if closes else current
                    month_ago = closes[0] if closes else current
                    
                    # RSI (simplified)
                    gains = []
                    losses = []
                    for i in range(1, min(15, len(closes))):
                        change = closes[i] - closes[i-1]
                        if change > 0:
                            gains.append(change)
                        else:
                            losses.append(abs(change))
                    avg_gain = sum(gains) / 14 if gains else 0.001
                    avg_loss = sum(losses) / 14 if losses else 0.001
                    rs = avg_gain / avg_loss if avg_loss > 0 else 1
                    rsi = 100 - (100 / (1 + rs))
                    
                    return {
                        'current': current,
                        'sma_10': round(sma_10, 2),
                        'sma_20': round(sma_20, 2),
                        'above_sma10': current > sma_10,
                        'above_sma20': current > sma_20,
                        'week_return': round((current - week_ago) / week_ago * 100, 2) if week_ago else 0,
                        'month_return': round((current - month_ago) / month_ago * 100, 2) if month_ago else 0,
                        'rsi': round(rsi, 1),
                        'avg_volume': round(sum(volumes) / len(volumes)) if volumes else 0,
                        'high_30d': max(highs) if highs else 0,
                        'low_30d': min(lows) if lows else 0,
                        'near_high': current >= max(highs) * 0.95 if highs else False,
                        'near_low': current <= min(lows) * 1.05 if lows else False,
                    }
        except Exception as e:
            pass
        return {}
    
    def get_live_stock_analysis(self, symbol: str) -> str:
        """Get comprehensive live analysis for a stock"""
        quote = self.get_live_quote(symbol)
        technicals = self.get_price_history(symbol)
        
        if not quote and not technicals:
            return f"Unable to fetch live data for {symbol}. Please check if the symbol is valid."
        
        analysis = f"**Live Data for {symbol.upper()}**\n\n"
        
        if quote:
            analysis += f"**Current Quote:**\n"
            analysis += f"- Price: ${quote.get('price', 0):.2f}\n"
            analysis += f"- Change: {quote.get('change', 0):+.2f} ({quote.get('change_pct', 0):+.2f}%)\n"
            analysis += f"- Volume: {quote.get('volume', 0):,}\n"
            analysis += f"- Bid/Ask: ${quote.get('bid', 0):.2f} / ${quote.get('ask', 0):.2f}\n"
            analysis += f"- 52w Range: ${quote.get('week_52_low', 0):.2f} - ${quote.get('week_52_high', 0):.2f}\n"
            if quote.get('pe_ratio'):
                analysis += f"- P/E Ratio: {quote.get('pe_ratio', 0):.1f}\n"
        
        if technicals:
            analysis += f"\n**Technical Indicators:**\n"
            analysis += f"- RSI(14): {technicals.get('rsi', 0):.1f}"
            if technicals.get('rsi', 50) > 70:
                analysis += " (Overbought)\n"
            elif technicals.get('rsi', 50) < 30:
                analysis += " (Oversold)\n"
            else:
                analysis += " (Neutral)\n"
            analysis += f"- SMA(10): ${technicals.get('sma_10', 0):.2f} {'✅ Above' if technicals.get('above_sma10') else '❌ Below'}\n"
            analysis += f"- SMA(20): ${technicals.get('sma_20', 0):.2f} {'✅ Above' if technicals.get('above_sma20') else '❌ Below'}\n"
            analysis += f"- Week Return: {technicals.get('week_return', 0):+.2f}%\n"
            analysis += f"- Month Return: {technicals.get('month_return', 0):+.2f}%\n"
            analysis += f"- 30d Range: ${technicals.get('low_30d', 0):.2f} - ${technicals.get('high_30d', 0):.2f}\n"
            
            if technicals.get('near_high'):
                analysis += f"- ⚠️ Near 30-day high\n"
            if technicals.get('near_low'):
                analysis += f"- ⚠️ Near 30-day low\n"
        
        return analysis
    
    # ==================== DATA AGGREGATION ====================
    
    def load_newsletter_history(self) -> dict:
        """Load newsletter scan history"""
        history_file = self.project_root / "data" / "newsletter_scan_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def get_improving_stocks_summary(self) -> str:
        """Get summary of improving stocks from newsletter"""
        history = self.load_newsletter_history()
        if not history:
            return "No historical data available yet."
        
        dates = sorted(history.keys(), reverse=True)
        if len(dates) < 2:
            return "Need more scans to identify trends (only 1 scan so far)."
        
        # Find improving stocks
        improving = []
        latest = history[dates[0]]
        previous = history[dates[1]]
        
        for ticker, data in latest.items():
            if ticker in previous:
                improvement = data['score'] - previous[ticker]['score']
                if improvement > 0:
                    improving.append({
                        'ticker': ticker,
                        'current_score': data['score'],
                        'improvement': improvement,
                        'theme': data.get('theme', 'Unknown'),
                        'week_return': data.get('week_return', 0)
                    })
        
        improving.sort(key=lambda x: x['improvement'], reverse=True)
        
        if not improving:
            return "No improving stocks detected between last two scans."
        
        summary = f"Top improving stocks (comparing {dates[0]} vs {dates[1]}):\n"
        for stock in improving[:10]:
            summary += f"- {stock['ticker']}: Score {stock['current_score']} (+{stock['improvement']:.0f}), {stock['theme']}, Week: {stock['week_return']:+.1f}%\n"
        
        return summary
    
    def get_top_opportunities_summary(self) -> str:
        """Get summary of top scoring stocks"""
        history = self.load_newsletter_history()
        if not history:
            return "No scan data available."
        
        dates = sorted(history.keys(), reverse=True)
        latest = history[dates[0]]
        
        # Sort by score
        top_stocks = sorted(
            [(ticker, data) for ticker, data in latest.items()],
            key=lambda x: x[1]['score'],
            reverse=True
        )[:15]
        
        summary = f"Top opportunities from {dates[0]}:\n"
        for ticker, data in top_stocks:
            summary += f"- {ticker}: Score {data['score']}, {data.get('theme', 'Unknown')}, ${data.get('price', 0):.2f}, Week: {data.get('week_return', 0):+.1f}%\n"
        
        return summary
    
    def get_market_context(self) -> str:
        """Get current market context"""
        try:
            import yfinance as yf
            
            # Get major indices
            spy = yf.Ticker("SPY")
            vix = yf.Ticker("^VIX")
            qqq = yf.Ticker("QQQ")
            
            spy_hist = spy.history(period="5d")
            vix_hist = vix.history(period="5d")
            qqq_hist = qqq.history(period="5d")
            
            spy_change = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100
            qqq_change = ((qqq_hist['Close'].iloc[-1] / qqq_hist['Close'].iloc[0]) - 1) * 100
            vix_level = vix_hist['Close'].iloc[-1]
            
            context = f"""Market Context:
- SPY: ${spy_hist['Close'].iloc[-1]:.2f} ({spy_change:+.1f}% this week)
- QQQ: ${qqq_hist['Close'].iloc[-1]:.2f} ({qqq_change:+.1f}% this week)  
- VIX: {vix_level:.1f} ({'Elevated fear' if vix_level > 20 else 'Low fear' if vix_level < 15 else 'Normal'})
- Market Bias: {'Bullish' if spy_change > 1 else 'Bearish' if spy_change < -1 else 'Neutral'}
"""
            return context
        except Exception as e:
            return f"Market data unavailable: {str(e)}"
    
    def aggregate_all_data(self) -> str:
        """Aggregate all available data sources into context"""
        sections = []
        
        # Market context
        sections.append(self.get_market_context())
        
        # Top opportunities
        sections.append(self.get_top_opportunities_summary())
        
        # Improving stocks
        sections.append(self.get_improving_stocks_summary())
        
        return "\n\n".join(sections)
    
    # ==================== AI INTERACTIONS ====================
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the trading copilot"""
        return """You are an expert AI trading assistant analyzing options flow, technical setups, and market dynamics.

Your role:
1. Synthesize multiple data sources to identify high-probability trade setups
2. Provide clear, actionable insights (not financial advice)
3. Highlight confluences where multiple signals align
4. Assess risk levels and key support/resistance levels
5. Be concise but thorough

When analyzing:
- Higher scores (80+) indicate stronger technical setups
- Improving scores suggest building momentum
- Look for confluence: high score + improving + bullish flow = stronger signal
- Always mention key risk factors

Format responses clearly with bullet points and sections when appropriate.
Never give specific buy/sell recommendations - present data-driven observations."""
    
    def chat(self, user_message: str, include_context: bool = True) -> str:
        """Send a message to the AI and get a response"""
        if not self.is_available():
            return "❌ AI not configured. Please add your GROQ_API_KEY to use this feature."
        
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Add market context if requested
        if include_context:
            context = self.aggregate_all_data()
            messages.append({
                "role": "system", 
                "content": f"Current market data and analysis:\n\n{context}"
            })
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error communicating with AI: {str(e)}"
    
    def generate_morning_brief(self) -> str:
        """Generate a morning market brief"""
        prompt = """Generate a concise morning trading brief based on the data provided. Include:

1. **Market Sentiment** (1-2 sentences on overall market)
2. **Top 3 Setups** (stocks with best confluence of signals)
3. **Watch List** (3-5 stocks showing improving momentum)
4. **Risk Factors** (any concerns to be aware of)

Be specific with tickers and scores. Keep it under 300 words."""
        
        return self.chat(prompt, include_context=True)
    
    def analyze_stock(self, ticker: str) -> str:
        """Deep analysis of a specific stock"""
        ticker = ticker.upper().strip()
        
        # Get stock-specific data from history
        history = self.load_newsletter_history()
        stock_history = []
        
        for date in sorted(history.keys(), reverse=True)[:7]:
            if ticker in history[date]:
                data = history[date][ticker]
                stock_history.append(f"{date}: Score {data['score']}, ${data.get('price', 0):.2f}")
        
        history_context = "\n".join(stock_history) if stock_history else "No newsletter history for this ticker"
        
        # Get LIVE data from Schwab API
        live_data = self.get_live_stock_analysis(ticker)
        
        prompt = f"""Analyze {ticker} based on the available data.

{live_data}

Newsletter Historical scores for {ticker}:
{history_context}

Provide a comprehensive analysis:
1. **Current Setup** - Based on price action and technicals, is this bullish, bearish, or neutral?
2. **Technical Analysis** - RSI, moving averages, support/resistance
3. **Momentum** - Is the stock building momentum or losing steam?
4. **Key Levels** - Where are support and resistance based on the data?
5. **Risk Assessment** - What could go wrong? Position sizing guidance
6. **Trade Idea** - What's the thesis? Entry, target, stop-loss levels if applicable

Be specific with numbers and actionable."""
        
        return self.chat(prompt, include_context=True)
    
    def find_best_setups(self) -> str:
        """Find the best trade setups with confluence"""
        prompt = """Based on the current data, identify the TOP 5 trade setups right now.

For each setup, explain:
1. **Why this stock?** (score, improvement, theme)
2. **Confluence factors** (what multiple signals align?)
3. **Risk/Reward** (what's the potential vs risk?)
4. **Timing** (is it ready now or needs confirmation?)

Rank them by conviction level (highest first).
Focus on stocks with scores 75+ that are also improving."""
        
        return self.chat(prompt, include_context=True)


# Quick test
if __name__ == "__main__":
    copilot = TradingCopilot()
    print("Copilot available:", copilot.is_available())
    print("\n--- Aggregated Data ---")
    print(copilot.aggregate_all_data())
