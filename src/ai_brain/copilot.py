"""
AI Trading Copilot - Synthesizes market data and provides actionable insights
Uses Groq's free Llama 3.1 API
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import requests

# Try to import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class TradingCopilot:
    """AI-powered trading assistant that synthesizes multiple data sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = "llama-3.1-70b-versatile"  # Free on Groq!
        self.project_root = Path(__file__).parent.parent.parent
        
        if GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if AI is properly configured"""
        return self.client is not None and self.api_key is not None
    
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
        # Get stock-specific data from history
        history = self.load_newsletter_history()
        stock_history = []
        
        for date in sorted(history.keys(), reverse=True)[:7]:
            if ticker.upper() in history[date]:
                data = history[date][ticker.upper()]
                stock_history.append(f"{date}: Score {data['score']}, ${data.get('price', 0):.2f}")
        
        history_context = "\n".join(stock_history) if stock_history else "No historical data for this ticker"
        
        prompt = f"""Analyze {ticker.upper()} based on the available data.

Historical scores for {ticker.upper()}:
{history_context}

Provide:
1. **Setup Quality** - Is this a good setup? Why?
2. **Momentum** - Is the score improving or declining?
3. **Key Levels** - Where might support/resistance be?
4. **Risk Assessment** - What could go wrong?
5. **Trade Idea** - If bullish/bearish, what's the thesis?

Be specific and actionable."""
        
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
