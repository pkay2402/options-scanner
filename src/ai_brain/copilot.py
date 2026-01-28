"""
AI Trading Copilot - Synthesizes market data and provides actionable insights
Uses Groq's free Llama 3.1 API
"""

import os
import json
import re
from html import unescape
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import requests

# Try to import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try to import yfinance for earnings data and news
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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
    
    # ==================== NEWS & ANALYST RATINGS ====================
    
    def get_yfinance_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Fetch news for a stock from Yahoo Finance via yfinance"""
        if not YFINANCE_AVAILABLE:
            return []
        
        try:
            ticker = yf.Ticker(symbol.upper())
            news = ticker.news
            
            if not news:
                return []
            
            articles = []
            for article in news[:limit]:
                # Determine sentiment from title keywords
                title = article.get('title', '')
                title_lower = title.lower()
                
                sentiment = 0
                bullish_words = ['upgrade', 'buy', 'bullish', 'outperform', 'beat', 'surge', 'jump', 'soar', 'rally']
                bearish_words = ['downgrade', 'sell', 'bearish', 'underperform', 'miss', 'drop', 'fall', 'plunge', 'cut']
                
                for word in bullish_words:
                    if word in title_lower:
                        sentiment = 0.5
                        break
                for word in bearish_words:
                    if word in title_lower:
                        sentiment = -0.5
                        break
                
                articles.append({
                    'title': title,
                    'source': article.get('publisher', 'Yahoo Finance'),
                    'url': article.get('link', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M'),
                    'sentiment': sentiment,
                    'type': article.get('type', 'STORY'),
                })
            
            return articles
        except Exception as e:
            return []
    
    def get_stock_news(self, symbol: str) -> Dict:
        """Get recent news for a specific stock using MarketAux + yfinance"""
        symbol = symbol.upper()
        
        # Try MarketAux first
        marketaux_news = self.get_marketaux_news(symbol, limit=10)
        
        # Get yfinance news as supplement
        yf_news = self.get_yfinance_news(symbol, limit=10)
        
        # Combine and categorize
        all_news = []
        upgrades = []
        downgrades = []
        
        # Process MarketAux news
        if marketaux_news and marketaux_news.get('articles'):
            for article in marketaux_news['articles']:
                all_news.append(article)
                title_lower = article.get('title', '').lower()
                if any(word in title_lower for word in ['upgrade', 'buy rating', 'outperform', 'price target raised']):
                    upgrades.append(article)
                elif any(word in title_lower for word in ['downgrade', 'sell rating', 'underperform', 'price target cut', 'price target lowered']):
                    downgrades.append(article)
        
        # Process yfinance news
        for article in yf_news:
            all_news.append(article)
            title_lower = article.get('title', '').lower()
            if any(word in title_lower for word in ['upgrade', 'buy rating', 'outperform', 'price target raised']):
                upgrades.append(article)
            elif any(word in title_lower for word in ['downgrade', 'sell rating', 'underperform', 'price target cut', 'price target lowered']):
                downgrades.append(article)
        
        return {
            'symbol': symbol,
            'has_upgrade': len(upgrades) > 0,
            'has_downgrade': len(downgrades) > 0,
            'stock_upgrades': upgrades,
            'stock_downgrades': downgrades,
            'all_news': all_news[:15],
            'sentiment_avg': marketaux_news.get('sentiment_avg', 0) if marketaux_news else 0
        }
    
    def get_news_summary(self) -> Dict:
        """Get summary of recent news for major market tickers using MarketAux + yfinance"""
        # Expanded list of major tickers to scan for news
        # Includes: Indices, Mag7, Healthcare, Finance, Energy, Semis, Consumer
        major_tickers = [
            # Market ETFs
            'SPY', 'QQQ', 'IWM', 'DIA',
            # Mag 7 + Big Tech
            'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMD', 'META', 'GOOGL', 'AMZN', 'NFLX',
            # Healthcare (often earnings movers)
            'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK',
            # Finance
            'JPM', 'BAC', 'GS', 'V', 'MA',
            # Energy
            'XOM', 'CVX', 'COP',
            # Other large caps
            'WMT', 'HD', 'COST', 'DIS', 'BA', 'CAT',
            # High beta / Memes
            'COIN', 'PLTR', 'SOFI', 'RIVN',
        ]
        
        upgraded_tickers = {}
        downgraded_tickers = {}
        major_news = {}  # Track significant news by ticker
        all_news = []
        
        # Keywords for significant news (beyond just upgrades/downgrades)
        bullish_keywords = ['upgrade', 'buy rating', 'outperform', 'price target raised', 'price target increase', 
                          'beat', 'beats', 'surge', 'soars', 'rally', 'breakthrough', 'approval']
        bearish_keywords = ['downgrade', 'sell rating', 'underperform', 'price target cut', 'price target lowered', 
                          'price target decrease', 'miss', 'misses', 'plunge', 'crash', 'fall', 'drops', 
                          'investigation', 'probe', 'lawsuit', 'warning', 'disappoints']
        
        for ticker in major_tickers:
            try:
                # Try MarketAux first (more reliable sentiment)
                marketaux_news = self.get_marketaux_news(ticker, limit=5)
                
                if marketaux_news and marketaux_news.get('articles'):
                    for article in marketaux_news['articles']:
                        title = article.get('title', '')
                        title_lower = title.lower()
                        sentiment = article.get('sentiment', 0)
                        
                        all_news.append({'ticker': ticker, 'title': title, 'source': 'MarketAux', 'sentiment': sentiment})
                        
                        # Check for upgrades/downgrades
                        if any(word in title_lower for word in bullish_keywords):
                            if ticker not in upgraded_tickers:
                                upgraded_tickers[ticker] = []
                            upgraded_tickers[ticker].append(title[:100])
                        elif any(word in title_lower for word in bearish_keywords):
                            if ticker not in downgraded_tickers:
                                downgraded_tickers[ticker] = []
                            downgraded_tickers[ticker].append(title[:100])
                        
                        # Track significant news (high sentiment magnitude)
                        if abs(sentiment) > 0.3:
                            if ticker not in major_news:
                                major_news[ticker] = []
                            major_news[ticker].append({'title': title, 'sentiment': sentiment})
                else:
                    # Fallback to yfinance
                    yf_news = self.get_yfinance_news(ticker, limit=5)
                    for article in yf_news:
                        title = article.get('title', '')
                        title_lower = title.lower()
                        
                        all_news.append({'ticker': ticker, 'title': title, 'source': 'Yahoo Finance', 'sentiment': 0})
                        
                        if any(word in title_lower for word in bullish_keywords):
                            if ticker not in upgraded_tickers:
                                upgraded_tickers[ticker] = []
                            upgraded_tickers[ticker].append(title[:100])
                        elif any(word in title_lower for word in bearish_keywords):
                            if ticker not in downgraded_tickers:
                                downgraded_tickers[ticker] = []
                            downgraded_tickers[ticker].append(title[:100])
            except Exception:
                continue
        
        return {
            'all_news': all_news,
            'upgraded_tickers': upgraded_tickers,
            'downgraded_tickers': downgraded_tickers,
            'major_news': major_news,
            'total_upgrades': len(upgraded_tickers),
            'total_downgrades': len(downgraded_tickers)
        }
    
    def get_news_analysis(self, symbol: str) -> str:
        """Get formatted news analysis for a stock using MarketAux + yfinance combined"""
        symbol = symbol.upper()
        
        # Fetch from BOTH sources for comprehensive coverage
        marketaux_news = self.get_marketaux_news(symbol, limit=10)
        yf_news = self.get_yfinance_news(symbol, limit=10)
        
        analysis = f"**📰 NEWS & ANALYST RATINGS for {symbol}**\n\n"
        
        all_articles = []
        sentiment_scores = []
        
        # Process MarketAux news
        if marketaux_news and marketaux_news.get('articles'):
            for article in marketaux_news['articles']:
                sentiment = article.get('sentiment', 0)
                sentiment_scores.append(sentiment)
                all_articles.append({
                    'title': article.get('title', ''),
                    'source': f"[MarketAux] {article.get('source', '')}",
                    'published': article.get('published', '')[:16],
                    'sentiment': sentiment,
                })
        
        # Process yfinance news (always include for broader coverage)
        for article in yf_news:
            # Avoid duplicates by checking title similarity
            title = article.get('title', '')
            if not any(title[:50] in a['title'] for a in all_articles):
                all_articles.append({
                    'title': title,
                    'source': f"[Yahoo] {article.get('source', 'Yahoo Finance')}",
                    'published': article.get('published', ''),
                    'sentiment': article.get('sentiment', 0),
                })
                sentiment_scores.append(article.get('sentiment', 0))
        
        # Calculate overall sentiment
        if sentiment_scores:
            sentiment_avg = sum(sentiment_scores) / len(sentiment_scores)
            if sentiment_avg > 0.15:
                analysis += f"📈 **Overall News Sentiment: POSITIVE** (score: {sentiment_avg:.2f})\n\n"
            elif sentiment_avg < -0.15:
                analysis += f"📉 **Overall News Sentiment: NEGATIVE** (score: {sentiment_avg:.2f})\n\n"
            else:
                analysis += f"😐 **Overall News Sentiment: NEUTRAL** (score: {sentiment_avg:.2f})\n\n"
        
        if all_articles:
            analysis += f"**Recent Headlines ({len(all_articles)} articles from MarketAux + Yahoo Finance):**\n\n"
            for article in all_articles[:8]:  # Show more articles
                title = article['title'][:120]
                source = article['source']
                sentiment = article.get('sentiment', 0)
                published = article.get('published', '')
                
                # More visible sentiment indicators
                if sentiment > 0.2:
                    sentiment_icon = "🟢 BULLISH"
                elif sentiment < -0.2:
                    sentiment_icon = "🔴 BEARISH"
                elif sentiment > 0.05:
                    sentiment_icon = "🟡 Slightly Positive"
                elif sentiment < -0.05:
                    sentiment_icon = "🟠 Slightly Negative"
                else:
                    sentiment_icon = "⚪ Neutral"
                
                analysis += f"• **{title}**\n"
                analysis += f"  {source} | {published} | {sentiment_icon}\n\n"
        else:
            analysis += f"⚠️ No recent news found for {symbol} - this could indicate limited coverage or API issues\n\n"
        
        # Check for specific upgrade/downgrade keywords in news
        upgrade_keywords = ['upgrade', 'buy rating', 'outperform', 'price target raised', 'price target increase']
        downgrade_keywords = ['downgrade', 'sell rating', 'underperform', 'price target cut', 'price target lowered', 'price target decrease']
        major_event_keywords = ['investigation', 'probe', 'lawsuit', 'resign', 'ceo', 'earnings', 'guidance', 'warning']
        
        upgrades_found = []
        downgrades_found = []
        major_events = []
        
        for article in all_articles:
            title_lower = article['title'].lower()
            if any(kw in title_lower for kw in upgrade_keywords):
                upgrades_found.append(article['title'])
            if any(kw in title_lower for kw in downgrade_keywords):
                downgrades_found.append(article['title'])
            if any(kw in title_lower for kw in major_event_keywords):
                major_events.append(article['title'])
        
        if upgrades_found:
            analysis += f"🔼 **UPGRADES/BULLISH NEWS DETECTED ({len(upgrades_found)}):**\n"
            for headline in upgrades_found[:3]:
                analysis += f"  • {headline[:100]}\n"
            analysis += "\n"
        
        if downgrades_found:
            analysis += f"🔽 **DOWNGRADES/BEARISH NEWS DETECTED ({len(downgrades_found)}):**\n"
            for headline in downgrades_found[:3]:
                analysis += f"  • {headline[:100]}\n"
            analysis += "\n"
        
        if major_events:
            analysis += f"⚠️ **MAJOR EVENTS DETECTED ({len(major_events)}):**\n"
            for headline in major_events[:3]:
                analysis += f"  • {headline[:100]}\n"
            analysis += "\n"
        
        return analysis
    
    def get_marketaux_news(self, symbol: str, limit: int = 10) -> Optional[Dict]:
        """Fetch news for a stock from MarketAux API"""
        # Get API key from environment or Streamlit secrets
        api_key = os.environ.get("MARKETAUX_API_KEY")
        
        # Try Streamlit secrets if not in env
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("MARKETAUX_API_KEY") or st.secrets.get("alerts", {}).get("MARKETAUX_API_KEY")
            except:
                pass
        
        if not api_key:
            return None
        
        try:
            # MarketAux API endpoint
            url = "https://api.marketaux.com/v1/news/all"
            params = {
                'api_token': api_key,
                'symbols': symbol.upper(),
                'filter_entities': 'true',
                'language': 'en',
                'limit': limit,
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data.get('data'):
                return None
            
            articles = []
            sentiment_scores = []
            
            for article in data['data']:
                # Get sentiment for the specific symbol
                symbol_sentiment = 0
                for entity in article.get('entities', []):
                    if entity.get('symbol', '').upper() == symbol.upper():
                        symbol_sentiment = entity.get('sentiment_score', 0)
                        break
                
                sentiment_scores.append(symbol_sentiment)
                
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'snippet': article.get('snippet', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published': article.get('published_at', ''),
                    'sentiment': symbol_sentiment,
                })
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                'symbol': symbol.upper(),
                'articles': articles,
                'sentiment_avg': avg_sentiment,
                'total_found': data.get('meta', {}).get('found', len(articles)),
            }
            
        except Exception as e:
            return None
    
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
    
    def get_options_flow(self, symbol: str) -> Dict:
        """Fetch and analyze options flow data"""
        if not self.schwab_client:
            return {}
        
        try:
            chain = self.schwab_client.get_options_chain(symbol.upper())
            if not chain:
                return {}
            
            underlying_price = chain.get('underlyingPrice', 0)
            calls = chain.get('callExpDateMap', {})
            puts = chain.get('putExpDateMap', {})
            
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            
            # Track unusual activity
            high_volume_calls = []
            high_volume_puts = []
            
            # Near-term expirations (next 30 days)
            near_term_call_vol = 0
            near_term_put_vol = 0
            
            for exp_date, strikes in calls.items():
                for strike, options in strikes.items():
                    for opt in options:
                        vol = opt.get('totalVolume', 0)
                        oi = opt.get('openInterest', 0)
                        dte = opt.get('daysToExpiration', 0)
                        
                        # Only process options within 90 days
                        if dte > 90:
                            continue
                        
                        total_call_volume += vol
                        total_call_oi += oi
                        
                        if dte <= 30:
                            near_term_call_vol += vol
                        
                        # Flag unusual volume (volume > 2x open interest)
                        if oi > 0 and vol > oi * 2 and vol > 500:
                            high_volume_calls.append({
                                'strike': float(strike),
                                'expiry': exp_date.split(':')[0],
                                'volume': vol,
                                'oi': oi,
                                'dte': dte,
                                'ratio': round(vol / oi, 1)
                            })
            
            for exp_date, strikes in puts.items():
                for strike, options in strikes.items():
                    for opt in options:
                        vol = opt.get('totalVolume', 0)
                        oi = opt.get('openInterest', 0)
                        dte = opt.get('daysToExpiration', 0)
                        
                        # Only process options within 90 days
                        if dte > 90:
                            continue
                        
                        total_put_volume += vol
                        total_put_oi += oi
                        
                        if dte <= 30:
                            near_term_put_vol += vol
                        
                        if oi > 0 and vol > oi * 2 and vol > 500:
                            high_volume_puts.append({
                                'strike': float(strike),
                                'expiry': exp_date.split(':')[0],
                                'volume': vol,
                                'oi': oi,
                                'dte': dte,
                                'ratio': round(vol / oi, 1)
                            })
            
            # Sort by volume
            high_volume_calls.sort(key=lambda x: x['volume'], reverse=True)
            high_volume_puts.sort(key=lambda x: x['volume'], reverse=True)
            
            # Calculate ratios
            put_call_ratio = round(total_put_volume / max(total_call_volume, 1), 2)
            
            return {
                'underlying_price': underlying_price,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'put_call_ratio': put_call_ratio,
                'near_term_call_vol': near_term_call_vol,
                'near_term_put_vol': near_term_put_vol,
                'sentiment': 'Bullish' if put_call_ratio < 0.7 else 'Bearish' if put_call_ratio > 1.0 else 'Neutral',
                'unusual_calls': high_volume_calls[:5],  # Top 5
                'unusual_puts': high_volume_puts[:5],
            }
        except Exception as e:
            return {}
    
    def get_options_analysis(self, symbol: str) -> str:
        """Get formatted options flow analysis"""
        flow = self.get_options_flow(symbol)
        
        if not flow:
            return f"Unable to fetch options data for {symbol}."
        
        analysis = f"**Options Flow for {symbol.upper()}**\n\n"
        analysis += f"**Volume Summary:**\n"
        analysis += f"- Call Volume: {flow.get('total_call_volume', 0):,}\n"
        analysis += f"- Put Volume: {flow.get('total_put_volume', 0):,}\n"
        analysis += f"- Put/Call Ratio: {flow.get('put_call_ratio', 0):.2f} ({flow.get('sentiment', 'N/A')})\n"
        analysis += f"- Near-term Calls (30d): {flow.get('near_term_call_vol', 0):,}\n"
        analysis += f"- Near-term Puts (30d): {flow.get('near_term_put_vol', 0):,}\n"
        
        analysis += f"\n**Open Interest:**\n"
        analysis += f"- Call OI: {flow.get('total_call_oi', 0):,}\n"
        analysis += f"- Put OI: {flow.get('total_put_oi', 0):,}\n"
        
        # Unusual activity
        unusual_calls = flow.get('unusual_calls', [])
        unusual_puts = flow.get('unusual_puts', [])
        
        if unusual_calls:
            analysis += f"\n**🔥 Unusual Call Activity (Strike @ Expiry):**\n"
            for c in unusual_calls[:3]:
                analysis += f"- ${c['strike']} @ {c['expiry']} expiry: {c['volume']:,} vol vs {c['oi']:,} OI ({c['ratio']}x normal)\n"
        
        if unusual_puts:
            analysis += f"\n**🔥 Unusual Put Activity (Strike @ Expiry):**\n"
            for p in unusual_puts[:3]:
                analysis += f"- ${p['strike']} @ {p['expiry']} expiry: {p['volume']:,} vol vs {p['oi']:,} OI ({p['ratio']}x normal)\n"
        
        return analysis
    
    # ==================== PRO TRADER DATA ====================
    
    def get_iv_rank(self, symbol: str) -> Dict:
        """Calculate IV Rank - is premium cheap or expensive?"""
        if not self.schwab_client:
            return {}
        
        try:
            chain = self.schwab_client.get_options_chain(symbol.upper())
            if not chain:
                return {}
            
            # Get current IV from the chain
            current_iv = chain.get('volatility', 0)
            
            # Get all IVs from near-term ATM options
            underlying_price = chain.get('underlyingPrice', 0)
            calls = chain.get('callExpDateMap', {})
            
            ivs = []
            for exp_date, strikes in calls.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    # Focus on near-the-money options
                    if underlying_price * 0.95 <= strike <= underlying_price * 1.05:
                        for opt in options:
                            iv = opt.get('volatility', 0)
                            if iv > 0:
                                ivs.append(iv)
            
            if ivs:
                current_iv = sum(ivs) / len(ivs)
            
            # For IV Rank, we'd need historical data
            # Using a simplified approach: compare to typical ranges
            # Low IV: < 20%, Medium: 20-40%, High: 40-60%, Very High: > 60%
            if current_iv < 20:
                iv_rank = 15
                iv_status = "LOW - Options are CHEAP, good for buying"
            elif current_iv < 30:
                iv_rank = 35
                iv_status = "BELOW AVERAGE - Slightly cheap"
            elif current_iv < 45:
                iv_rank = 50
                iv_status = "AVERAGE - Fair pricing"
            elif current_iv < 60:
                iv_rank = 70
                iv_status = "ELEVATED - Consider selling premium"
            else:
                iv_rank = 85
                iv_status = "HIGH - Options expensive, SELL premium"
            
            return {
                'current_iv': round(current_iv, 1),
                'iv_rank': iv_rank,
                'iv_status': iv_status
            }
        except:
            return {}
    
    def get_gamma_walls(self, symbol: str) -> Dict:
        """Find Gamma Walls - highest OI strikes act as support/resistance"""
        if not self.schwab_client:
            return {}
        
        try:
            chain = self.schwab_client.get_options_chain(symbol.upper())
            if not chain:
                return {}
            
            underlying_price = chain.get('underlyingPrice', 0)
            calls = chain.get('callExpDateMap', {})
            puts = chain.get('putExpDateMap', {})
            
            # Aggregate OI by strike across all expirations
            strike_oi = {}
            call_oi_by_strike = {}
            put_oi_by_strike = {}
            
            for exp_date, strikes in calls.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    for opt in options:
                        oi = opt.get('openInterest', 0)
                        strike_oi[strike] = strike_oi.get(strike, 0) + oi
                        call_oi_by_strike[strike] = call_oi_by_strike.get(strike, 0) + oi
            
            for exp_date, strikes in puts.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    for opt in options:
                        oi = opt.get('openInterest', 0)
                        strike_oi[strike] = strike_oi.get(strike, 0) + oi
                        put_oi_by_strike[strike] = put_oi_by_strike.get(strike, 0) + oi
            
            # Find top 5 gamma walls (highest OI)
            sorted_strikes = sorted(strike_oi.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Separate into support (below price) and resistance (above price)
            support_levels = []
            resistance_levels = []
            
            for strike, oi in sorted_strikes:
                if strike < underlying_price:
                    call_oi = call_oi_by_strike.get(strike, 0)
                    put_oi = put_oi_by_strike.get(strike, 0)
                    support_levels.append({
                        'strike': strike,
                        'total_oi': oi,
                        'call_oi': call_oi,
                        'put_oi': put_oi
                    })
                else:
                    call_oi = call_oi_by_strike.get(strike, 0)
                    put_oi = put_oi_by_strike.get(strike, 0)
                    resistance_levels.append({
                        'strike': strike,
                        'total_oi': oi,
                        'call_oi': call_oi,
                        'put_oi': put_oi
                    })
            
            # Sort by proximity to current price
            support_levels.sort(key=lambda x: x['strike'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strike'])
            
            return {
                'current_price': underlying_price,
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3]
            }
        except:
            return {}
    
    def _get_limited_options_flow(self, symbol: str) -> Dict:
        """Get options flow using limited chain for large ETFs like SPY/QQQ"""
        if not self.schwab_client:
            return {}
        
        try:
            chain = self.get_limited_options_chain(symbol.upper())
            if not chain:
                return {}
            
            underlying_price = chain.get('underlyingPrice', 0)
            calls = chain.get('callExpDateMap', {})
            puts = chain.get('putExpDateMap', {})
            
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0
            
            for exp_date, strikes in calls.items():
                for strike, options in strikes.items():
                    for opt in options:
                        total_call_volume += opt.get('totalVolume', 0)
                        total_call_oi += opt.get('openInterest', 0)
            
            for exp_date, strikes in puts.items():
                for strike, options in strikes.items():
                    for opt in options:
                        total_put_volume += opt.get('totalVolume', 0)
                        total_put_oi += opt.get('openInterest', 0)
            
            put_call_ratio = round(total_put_volume / max(total_call_volume, 1), 2)
            
            return {
                'underlying_price': underlying_price,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'put_call_ratio': put_call_ratio,
                'sentiment': 'Bullish' if put_call_ratio < 0.7 else 'Bearish' if put_call_ratio > 1.0 else 'Neutral',
            }
        except:
            return {}
    
    def _get_limited_gamma_walls(self, symbol: str) -> Dict:
        """Get gamma walls using limited chain for large ETFs like SPY/QQQ"""
        if not self.schwab_client:
            return {}
        
        try:
            chain = self.get_limited_options_chain(symbol.upper())
            if not chain:
                return {}
            
            underlying_price = chain.get('underlyingPrice', 0)
            calls = chain.get('callExpDateMap', {})
            puts = chain.get('putExpDateMap', {})
            
            strike_oi = {}
            call_oi_by_strike = {}
            put_oi_by_strike = {}
            
            for exp_date, strikes in calls.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    for opt in options:
                        oi = opt.get('openInterest', 0)
                        strike_oi[strike] = strike_oi.get(strike, 0) + oi
                        call_oi_by_strike[strike] = call_oi_by_strike.get(strike, 0) + oi
            
            for exp_date, strikes in puts.items():
                for strike_str, options in strikes.items():
                    strike = float(strike_str)
                    for opt in options:
                        oi = opt.get('openInterest', 0)
                        strike_oi[strike] = strike_oi.get(strike, 0) + oi
                        put_oi_by_strike[strike] = put_oi_by_strike.get(strike, 0) + oi
            
            sorted_strikes = sorted(strike_oi.items(), key=lambda x: x[1], reverse=True)[:5]
            
            support_levels = []
            resistance_levels = []
            
            for strike, oi in sorted_strikes:
                level = {
                    'strike': strike,
                    'total_oi': oi,
                    'call_oi': call_oi_by_strike.get(strike, 0),
                    'put_oi': put_oi_by_strike.get(strike, 0)
                }
                if strike < underlying_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
            
            support_levels.sort(key=lambda x: x['strike'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strike'])
            
            return {
                'current_price': underlying_price,
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3]
            }
        except:
            return {}
    
    def get_earnings_info(self, symbol: str) -> Dict:
        """Get earnings date from yfinance"""
        if not YFINANCE_AVAILABLE:
            return {}
        
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Method 1: Use calendar (most reliable for upcoming earnings)
            calendar = ticker.calendar
            if calendar and 'Earnings Date' in calendar:
                earnings_list = calendar['Earnings Date']
                if earnings_list:
                    # Can be a list or single date
                    next_earnings = earnings_list[0] if isinstance(earnings_list, list) else earnings_list
                    
                    # Convert to date if needed
                    if hasattr(next_earnings, 'date'):
                        next_earnings_date = next_earnings if isinstance(next_earnings, datetime) else datetime.combine(next_earnings, datetime.min.time())
                    else:
                        next_earnings_date = next_earnings
                    
                    days_to_earnings = (next_earnings_date.date() if hasattr(next_earnings_date, 'date') else next_earnings_date) - datetime.now().date()
                    days_to_earnings = days_to_earnings.days if hasattr(days_to_earnings, 'days') else days_to_earnings
                    
                    # Earnings warning
                    if days_to_earnings <= 7:
                        warning = "⚠️ EARNINGS IMMINENT - High IV crush risk!"
                    elif days_to_earnings <= 14:
                        warning = "⚠️ Earnings within 2 weeks - Watch IV"
                    elif days_to_earnings <= 30:
                        warning = "Earnings approaching - Consider position timing"
                    else:
                        warning = "Earnings not imminent - Safe for swing trades"
                    
                    return {
                        'earnings_date': str(next_earnings),
                        'days_to_earnings': days_to_earnings,
                        'earnings_warning': warning
                    }
            
            # Method 2: Fallback to earnings_dates
            earnings_dates = ticker.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                future_earnings = [d for d in earnings_dates.index if d.date() > datetime.now().date()]
                if future_earnings:
                    next_earnings = future_earnings[0]
                    days_to_earnings = (next_earnings.date() - datetime.now().date()).days
                    
                    if days_to_earnings <= 7:
                        warning = "⚠️ EARNINGS IMMINENT - High IV crush risk!"
                    elif days_to_earnings <= 14:
                        warning = "⚠️ Earnings within 2 weeks - Watch IV"
                    elif days_to_earnings <= 30:
                        warning = "Earnings approaching - Consider position timing"
                    else:
                        warning = "Earnings not imminent - Safe for swing trades"
                    
                    return {
                        'earnings_date': next_earnings.strftime('%Y-%m-%d'),
                        'days_to_earnings': days_to_earnings,
                        'earnings_warning': warning
                    }
        except Exception as e:
            pass
        return {'earnings_date': 'Unknown', 'days_to_earnings': None, 'earnings_warning': 'Earnings date not available'}
    
    def get_volume_analysis(self, symbol: str) -> Dict:
        """Compare today's volume to average - is something happening?"""
        quote = self.get_live_quote(symbol)
        technicals = self.get_price_history(symbol)
        
        if not quote or not technicals:
            return {}
        
        today_volume = quote.get('volume', 0)
        avg_volume = technicals.get('avg_volume', 1)
        
        if avg_volume == 0:
            avg_volume = 1
        
        volume_ratio = today_volume / avg_volume
        
        if volume_ratio >= 3:
            volume_signal = "🔥 EXTREME volume (3x+) - Major activity!"
        elif volume_ratio >= 2:
            volume_signal = "📈 HIGH volume (2x+) - Something's happening"
        elif volume_ratio >= 1.5:
            volume_signal = "Above average volume - Increased interest"
        elif volume_ratio >= 0.7:
            volume_signal = "Normal volume"
        else:
            volume_signal = "Low volume - Light trading"
        
        return {
            'today_volume': today_volume,
            'avg_volume': avg_volume,
            'volume_ratio': round(volume_ratio, 2),
            'volume_signal': volume_signal
        }
    
    def get_pro_trader_analysis(self, symbol: str) -> str:
        """Get all pro trader data combined"""
        analysis = f"**Pro Trader Data for {symbol.upper()}**\n\n"
        
        # IV Rank
        iv_data = self.get_iv_rank(symbol)
        if iv_data:
            analysis += f"**IV Analysis:**\n"
            analysis += f"- Current IV: {iv_data.get('current_iv', 0)}%\n"
            analysis += f"- IV Rank: {iv_data.get('iv_rank', 0)} ({iv_data.get('iv_status', 'N/A')})\n"
        
        # Volume Analysis
        vol_data = self.get_volume_analysis(symbol)
        if vol_data:
            analysis += f"\n**Volume Analysis:**\n"
            analysis += f"- Today: {vol_data.get('today_volume', 0):,}\n"
            analysis += f"- Average: {vol_data.get('avg_volume', 0):,}\n"
            analysis += f"- Ratio: {vol_data.get('volume_ratio', 0)}x - {vol_data.get('volume_signal', 'N/A')}\n"
        
        # Gamma Walls
        gamma_data = self.get_gamma_walls(symbol)
        if gamma_data:
            analysis += f"\n**Gamma Walls (OI-Based Support/Resistance):**\n"
            analysis += f"- Current Price: ${gamma_data.get('current_price', 0):.2f}\n"
            
            support = gamma_data.get('support_levels', [])
            if support:
                analysis += f"- Support Levels:\n"
                for s in support[:2]:
                    analysis += f"  • ${s['strike']:.0f} ({s['total_oi']:,} OI)\n"
            
            resistance = gamma_data.get('resistance_levels', [])
            if resistance:
                analysis += f"- Resistance Levels:\n"
                for r in resistance[:2]:
                    analysis += f"  • ${r['strike']:.0f} ({r['total_oi']:,} OI)\n"
        
        # Earnings
        earnings_data = self.get_earnings_info(symbol)
        if earnings_data and earnings_data.get('days_to_earnings') is not None:
            analysis += f"\n**Earnings:**\n"
            analysis += f"- Next Earnings: {earnings_data.get('earnings_date', 'Unknown')}\n"
            analysis += f"- Days Away: {earnings_data.get('days_to_earnings', 'Unknown')}\n"
            analysis += f"- {earnings_data.get('earnings_warning', '')}\n"
        
        return analysis
    
    # ==================== NEWSLETTER SCANNER INTEGRATION (REMOTE API) ====================
    
    # Droplet API URL - Newsletter scanner runs on droplet
    DROPLET_API_URL = "http://138.197.210.166:8001"
    
    def _fetch_from_droplet(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Fetch data from droplet API"""
        try:
            url = f"{self.DROPLET_API_URL}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            return None
    
    def get_scanner_data(self, query_type: str = "all", min_score: int = 60) -> List[Dict]:
        """Get data from newsletter scanner API on droplet"""
        result = self._fetch_from_droplet(
            "/api/newsletter/plays",
            {"query_type": query_type, "min_score": min_score, "limit": 30}
        )
        if result and result.get("status") == "success":
            return result.get("data", [])
        return []
    
    def get_improving_from_scanner(self) -> List[Dict]:
        """Get improving stocks from scanner API on droplet"""
        result = self._fetch_from_droplet("/api/newsletter/improving", {"limit": 15})
        if result and result.get("status") == "success":
            return result.get("data", [])
        return []
    
    def get_scanner_summary(self) -> Dict:
        """Get scanner summary from droplet"""
        result = self._fetch_from_droplet("/api/newsletter/summary")
        if result and result.get("status") == "success":
            return result
        return {}
    
    def get_scanner_stock_data(self, symbol: str) -> Dict:
        """Get scanner data for specific stock from droplet"""
        result = self._fetch_from_droplet(f"/api/newsletter/stock/{symbol.upper()}")
        if result and result.get("status") == "success":
            return result
        return {}
    
    def format_scanner_plays(self, plays: List[Dict], play_type: str = "Opportunities") -> str:
        """Format scanner plays for AI prompt with live prices"""
        if not plays:
            return f"No {play_type.lower()} found in scanner data."
        
        result = f"**{play_type}:**\n"
        for p in plays[:10]:
            ticker = p.get('ticker', 'N/A')
            score = p.get('opportunity_score', 0)
            setup = p.get('setup_type', 'N/A')
            timeframe = p.get('timeframe', 'N/A')
            
            # Try to get live price if Schwab client is available
            live_price = None
            if self.schwab_client:
                quote = self.get_live_quote(ticker)
                if quote:
                    live_price = quote.get('price')
            
            # Use live price if available, otherwise scanner price
            price = live_price if live_price else p.get('current_price', 0)
            
            week_ret = p.get('week_return', 0)
            rsi = p.get('rsi', 0)
            sentiment = p.get('options_sentiment', 'N/A')
            
            result += f"- {ticker}: Score {score:.0f}, {setup}, {timeframe}, "
            result += f"${price:.2f}"
            if live_price:
                result += " (LIVE)"
            result += f", Week: {week_ret:+.1f}%"
            if rsi:
                result += f", RSI: {rsi:.0f}"
            if sentiment:
                result += f", Options: {sentiment}"
            result += "\n"
        
        return result
    
    def get_ai_trade_recommendations(self) -> str:
        """Generate comprehensive trade recommendations using scanner data + LLM"""
        # Gather all scanner data
        bullish = self.get_scanner_data("bullish", 65)
        bearish = self.get_scanner_data("bearish", 55)
        weekly = self.get_scanner_data("weekly")
        monthly = self.get_scanner_data("monthly")
        improving = self.get_improving_from_scanner()
        
        # Check if we have data
        if not bullish and not bearish and not weekly:
            return "⚠️ No scanner data available. The scanner may not have run recently. Check if the newsletter-scanner service is running on the droplet."
        
        # Get live prices for top picks
        live_prices = {}
        top_tickers = list(set([p['ticker'] for p in (bullish[:5] + bearish[:3] + weekly[:5])]))
        
        if self.schwab_client:
            for ticker in top_tickers[:10]:  # Limit to avoid too many API calls
                quote = self.get_live_quote(ticker)
                if quote:
                    live_prices[ticker] = {
                        'price': quote.get('price', 0),
                        'change_pct': quote.get('change_pct', 0)
                    }
        
        # Format for prompt with live data
        context = "=== SCANNER DATA WITH LIVE PRICES ===\n\n"
        
        # Add bullish plays with live prices
        if bullish:
            context += "**🟢 BULLISH SETUPS (High Score)**\n"
            for p in bullish[:8]:
                ticker = p['ticker']
                live = live_prices.get(ticker, {})
                price = live.get('price') or p.get('current_price', 0)
                change = live.get('change_pct', 0)
                context += f"- {ticker}: Score {p['opportunity_score']:.0f}, RSI {p.get('rsi', 0):.0f}, "
                context += f"${price:.2f} ({change:+.1f}% today), Week: {p.get('week_return', 0):+.1f}%\n"
        
        context += "\n"
        
        # Add bearish plays
        if bearish:
            context += "**🔴 BEARISH SETUPS (For Puts)**\n"
            for p in bearish[:5]:
                ticker = p['ticker']
                live = live_prices.get(ticker, {})
                price = live.get('price') or p.get('current_price', 0)
                change = live.get('change_pct', 0)
                context += f"- {ticker}: Score {p['opportunity_score']:.0f}, RSI {p.get('rsi', 0):.0f}, "
                context += f"${price:.2f} ({change:+.1f}% today), Week: {p.get('week_return', 0):+.1f}%\n"
        
        context += "\n"
        
        # Weekly plays
        if weekly:
            context += "**⚡ WEEKLY OPTIONS PLAYS**\n"
            for p in weekly[:6]:
                ticker = p['ticker']
                live = live_prices.get(ticker, {})
                price = live.get('price') or p.get('current_price', 0)
                context += f"- {ticker}: Score {p['opportunity_score']:.0f}, ${price:.2f}, {p['setup_type']}\n"
        
        context += "\n"
        
        # Improving momentum
        if improving:
            context += "**📈 IMPROVING MOMENTUM**\n"
            for imp in improving[:5]:
                context += f"- {imp['ticker']}: Score improving +{imp['improvement']:.1f} pts\n"
        
        # Get news context
        news = self.get_news_summary()
        context += "\n**📰 TODAY'S NEWS**\n"
        if news['upgraded_tickers']:
            context += f"Upgrades: {', '.join(list(news['upgraded_tickers'].keys())[:5])}\n"
        if news['downgraded_tickers']:
            context += f"Downgrades: {', '.join(list(news['downgraded_tickers'].keys())[:5])}\n"
        
        # Build the AI prompt - more focused and actionable
        prompt = f"""Based on the scanner data below, provide SPECIFIC and ACTIONABLE trade recommendations.

{context}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## 🔥 TOP 3 WEEKLY PLAYS (0-5 days)

**1. [TICKER] - [CALL/PUT]**
- Current Price: $XXX
- Entry: $XXX (at market or on pullback to $XXX)
- Strike/Expiry: $XXX strike, [DATE] expiry
- Target: $XXX (+X%)
- Stop Loss: $XXX (-X%)
- Why: [Brief reason - RSI, momentum, score]

**2. [TICKER]...**

**3. [TICKER]...**

## 📅 SWING TRADES (2-4 weeks)

[Same format for 2-3 trades]

## 🐻 BEARISH PLAYS

[If any bearish setups exist, list 1-2 with put recommendations]

## ⚠️ AVOID

[List 2-3 tickers to avoid and why - overbought, extended, etc.]

## 📊 MARKET OUTLOOK

[2-3 sentences on overall market bias based on bullish vs bearish ratio]

IMPORTANT RULES:
- Use REAL prices from the data above
- Pick strikes near the money (within 5% of current price)
- Use expiries 7-21 days out for weekly, 30-45 days for swings
- Keep stop losses tight (5-10% for options)
- NO LaTeX or special formatting
- Be SPECIFIC - no vague recommendations"""

        return self.chat(prompt, include_context=False)
    
    # ==================== MARKET FORECAST ====================
    
    def get_limited_options_chain(self, symbol: str) -> Optional[Dict]:
        """Get limited options chain for large ETFs like SPY/QQQ to avoid timeouts"""
        if not self.schwab_client:
            return None
        
        try:
            # Calculate date range - next 14 days only
            from datetime import datetime, timedelta
            today = datetime.now()
            from_date = today.strftime('%Y-%m-%d')
            to_date = (today + timedelta(days=14)).strftime('%Y-%m-%d')
            
            # Request with limited strikes (20 strikes around ATM) and limited date range
            chain = self.schwab_client.get_options_chain(
                symbol.upper(),
                strike_count=20,  # Only 20 strikes (10 above, 10 below ATM)
                from_date=from_date,
                to_date=to_date
            )
            return chain
        except Exception as e:
            return None
    
    def get_market_forecast_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for forecast"""
        data = {
            'symbol': symbol,
            'quote': {},
            'technicals': {},
            'options': {},
            'gamma_levels': {},
            'expected_move': None
        }
        
        # Get live quote
        quote = self.get_live_quote(symbol)
        if quote:
            data['quote'] = quote
        
        # Get technicals
        technicals = self.get_price_history(symbol)
        if technicals:
            data['technicals'] = technicals
        
        # For SPY/QQQ use limited options chain to avoid timeouts
        is_large_etf = symbol.upper() in ['SPY', 'QQQ', 'IWM', 'DIA']
        
        # Get options flow (uses limited chain for large ETFs)
        if is_large_etf:
            options = self._get_limited_options_flow(symbol)
        else:
            options = self.get_options_flow(symbol)
        if options:
            data['options'] = options
        
        # Get gamma walls (uses limited chain for large ETFs)
        if is_large_etf:
            gamma = self._get_limited_gamma_walls(symbol)
        else:
            gamma = self.get_gamma_walls(symbol)
        if gamma:
            data['gamma_levels'] = gamma
        
        # Calculate expected move from options
        if self.schwab_client:
            try:
                # Use limited chain for large ETFs
                if is_large_etf:
                    chain = self.get_limited_options_chain(symbol.upper())
                else:
                    chain = self.schwab_client.get_options_chain(symbol.upper())
                    
                if chain:
                    underlying = chain.get('underlyingPrice', 0)
                    # Find nearest weekly expiry ATM straddle
                    calls = chain.get('callExpDateMap', {})
                    puts = chain.get('putExpDateMap', {})
                    
                    # Get first expiry (usually weekly)
                    if calls:
                        first_exp = list(calls.keys())[0]
                        exp_calls = calls[first_exp]
                        exp_puts = puts.get(first_exp, {})
                        
                        # Find ATM strike
                        strikes = [float(s) for s in exp_calls.keys()]
                        atm_strike = min(strikes, key=lambda x: abs(x - underlying))
                        
                        # Get ATM call and put prices
                        atm_call = exp_calls.get(str(atm_strike), [{}])[0]
                        atm_put = exp_puts.get(str(atm_strike), [{}])[0]
                        
                        call_mid = (atm_call.get('bid', 0) + atm_call.get('ask', 0)) / 2
                        put_mid = (atm_put.get('bid', 0) + atm_put.get('ask', 0)) / 2
                        
                        straddle = call_mid + put_mid
                        expected_move_pct = (straddle / underlying) * 100 if underlying else 0
                        
                        data['expected_move'] = {
                            'expiry': first_exp.split(':')[0],
                            'atm_strike': atm_strike,
                            'straddle_price': round(straddle, 2),
                            'expected_move_pct': round(expected_move_pct, 2),
                            'upper_range': round(underlying + straddle, 2),
                            'lower_range': round(underlying - straddle, 2),
                            'call_iv': atm_call.get('volatility', 0),
                            'put_iv': atm_put.get('volatility', 0)
                        }
            except:
                pass
        
        return data
    
    def get_5_day_market_forecast(self) -> str:
        """Generate 5-day market forecast for SPY and QQQ"""
        # Gather data for SPY and QQQ
        spy_data = self.get_market_forecast_data('SPY')
        qqq_data = self.get_market_forecast_data('QQQ')
        
        # Also get VIX for fear gauge
        vix_quote = self.get_live_quote('^VIX') or self.get_live_quote('VIX')
        
        # Build comprehensive context
        context = "=== MARKET FORECAST DATA ===\n\n"
        
        # SPY Analysis
        context += "## SPY (S&P 500 ETF)\n"
        if spy_data['quote']:
            q = spy_data['quote']
            context += f"- Current Price: ${q.get('price', 0):.2f}\n"
            context += f"- Today's Change: {q.get('change_pct', 0):+.2f}%\n"
            context += f"- 52w Range: ${q.get('week_52_low', 0):.2f} - ${q.get('week_52_high', 0):.2f}\n"
        
        if spy_data['technicals']:
            t = spy_data['technicals']
            context += f"- RSI(14): {t.get('rsi', 0):.1f}\n"
            context += f"- SMA(10): ${t.get('sma_10', 0):.2f} {'(Above)' if t.get('above_sma10') else '(Below)'}\n"
            context += f"- SMA(20): ${t.get('sma_20', 0):.2f} {'(Above)' if t.get('above_sma20') else '(Below)'}\n"
            context += f"- Week Return: {t.get('week_return', 0):+.2f}%\n"
            context += f"- Month Return: {t.get('month_return', 0):+.2f}%\n"
        
        if spy_data['options']:
            o = spy_data['options']
            context += f"- Put/Call Ratio: {o.get('put_call_ratio', 0):.2f} ({o.get('sentiment', 'N/A')})\n"
            context += f"- Call Volume: {o.get('total_call_volume', 0):,}\n"
            context += f"- Put Volume: {o.get('total_put_volume', 0):,}\n"
        
        if spy_data['expected_move']:
            em = spy_data['expected_move']
            context += f"- Expected Move (to {em['expiry']}): ±{em['expected_move_pct']:.1f}%\n"
            context += f"- Expected Range: ${em['lower_range']:.2f} - ${em['upper_range']:.2f}\n"
            context += f"- ATM IV: {em.get('call_iv', 0):.1f}%\n"
        
        if spy_data['gamma_levels']:
            g = spy_data['gamma_levels']
            if g.get('support_levels'):
                supports = [f"${s['strike']:.0f}" for s in g['support_levels'][:2]]
                context += f"- Gamma Support: {', '.join(supports)}\n"
            if g.get('resistance_levels'):
                resistances = [f"${r['strike']:.0f}" for r in g['resistance_levels'][:2]]
                context += f"- Gamma Resistance: {', '.join(resistances)}\n"
        
        context += "\n"
        
        # QQQ Analysis
        context += "## QQQ (Nasdaq 100 ETF)\n"
        if qqq_data['quote']:
            q = qqq_data['quote']
            context += f"- Current Price: ${q.get('price', 0):.2f}\n"
            context += f"- Today's Change: {q.get('change_pct', 0):+.2f}%\n"
            context += f"- 52w Range: ${q.get('week_52_low', 0):.2f} - ${q.get('week_52_high', 0):.2f}\n"
        
        if qqq_data['technicals']:
            t = qqq_data['technicals']
            context += f"- RSI(14): {t.get('rsi', 0):.1f}\n"
            context += f"- SMA(10): ${t.get('sma_10', 0):.2f} {'(Above)' if t.get('above_sma10') else '(Below)'}\n"
            context += f"- SMA(20): ${t.get('sma_20', 0):.2f} {'(Above)' if t.get('above_sma20') else '(Below)'}\n"
            context += f"- Week Return: {t.get('week_return', 0):+.2f}%\n"
            context += f"- Month Return: {t.get('month_return', 0):+.2f}%\n"
        
        if qqq_data['options']:
            o = qqq_data['options']
            context += f"- Put/Call Ratio: {o.get('put_call_ratio', 0):.2f} ({o.get('sentiment', 'N/A')})\n"
            context += f"- Call Volume: {o.get('total_call_volume', 0):,}\n"
            context += f"- Put Volume: {o.get('total_put_volume', 0):,}\n"
        
        if qqq_data['expected_move']:
            em = qqq_data['expected_move']
            context += f"- Expected Move (to {em['expiry']}): ±{em['expected_move_pct']:.1f}%\n"
            context += f"- Expected Range: ${em['lower_range']:.2f} - ${em['upper_range']:.2f}\n"
            context += f"- ATM IV: {em.get('call_iv', 0):.1f}%\n"
        
        if qqq_data['gamma_levels']:
            g = qqq_data['gamma_levels']
            if g.get('support_levels'):
                supports = [f"${s['strike']:.0f}" for s in g['support_levels'][:2]]
                context += f"- Gamma Support: {', '.join(supports)}\n"
            if g.get('resistance_levels'):
                resistances = [f"${r['strike']:.0f}" for r in g['resistance_levels'][:2]]
                context += f"- Gamma Resistance: {', '.join(resistances)}\n"
        
        context += "\n"
        
        # VIX
        if vix_quote:
            context += f"## VIX (Fear Index)\n"
            context += f"- Current: {vix_quote.get('price', 0):.2f}\n"
            context += f"- Today's Change: {vix_quote.get('change_pct', 0):+.2f}%\n"
            vix_level = vix_quote.get('price', 15)
            if vix_level < 15:
                context += "- Interpretation: LOW FEAR - Complacency, possible pullback risk\n"
            elif vix_level < 20:
                context += "- Interpretation: NORMAL - Healthy market conditions\n"
            elif vix_level < 25:
                context += "- Interpretation: ELEVATED - Caution warranted\n"
            else:
                context += "- Interpretation: HIGH FEAR - Potential buying opportunity on dips\n"
        
        # Get current date info - skip weekends (Saturday=5, Sunday=6)
        from datetime import datetime, timedelta
        today = datetime.now()
        next_5_trading_days = []
        days_ahead = 1
        while len(next_5_trading_days) < 5:
            next_day = today + timedelta(days=days_ahead)
            # Skip weekends (Monday=0, Sunday=6)
            if next_day.weekday() < 5:  # Monday-Friday only
                next_5_trading_days.append(next_day.strftime('%A, %b %d'))
            days_ahead += 1
        
        context += f"\n## Next 5 Trading Days\n"
        for i, day in enumerate(next_5_trading_days, 1):
            context += f"- Day {i}: {day}\n"
        
        # Build the AI prompt
        prompt = f"""Based on the comprehensive market data below, provide a DAY-BY-DAY 5-day market forecast.

{context}

ANALYZE THESE KEY FACTORS:
1. RSI levels - overbought (>70) or oversold (<30)?
2. Price vs Moving Averages - bullish or bearish trend?
3. Put/Call Ratio - sentiment reading
4. Expected Move from options - what's priced in?
5. Gamma levels - where are the support/resistance walls?
6. VIX level - fear or complacency?

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

## 📊 5-DAY MARKET FORECAST

### Overall Bias: [BULLISH / BEARISH / NEUTRAL] (X% confidence)

**Key Levels to Watch:**
- SPY Support: $XXX, $XXX
- SPY Resistance: $XXX, $XXX
- QQQ Support: $XXX, $XXX  
- QQQ Resistance: $XXX, $XXX

---

### Day 1 - {next_5_trading_days[0]}
**Forecast: [UP / DOWN / FLAT]**
- SPY Expected: $XXX - $XXX
- QQQ Expected: $XXX - $XXX
- Reasoning: [1-2 sentences]
- Trade Idea: [Specific actionable trade if any]

### Day 2 - {next_5_trading_days[1]}
**Forecast: [UP / DOWN / FLAT]**
- SPY Expected: $XXX - $XXX
- QQQ Expected: $XXX - $XXX
- Reasoning: [1-2 sentences]
- Trade Idea: [Specific actionable trade if any]

### Day 3 - {next_5_trading_days[2]}
**Forecast: [UP / DOWN / FLAT]**
- SPY Expected: $XXX - $XXX
- QQQ Expected: $XXX - $XXX
- Reasoning: [1-2 sentences]

### Day 4 - {next_5_trading_days[3]}
**Forecast: [UP / DOWN / FLAT]**
- SPY Expected: $XXX - $XXX
- QQQ Expected: $XXX - $XXX
- Reasoning: [1-2 sentences]

### Day 5 - {next_5_trading_days[4]}
**Forecast: [UP / DOWN / FLAT]**
- SPY Expected: $XXX - $XXX
- QQQ Expected: $XXX - $XXX
- Reasoning: [1-2 sentences]

---

## 🎯 WEEKLY OPTIONS PLAYS

**SPY Play:**
- Direction: [CALL/PUT]
- Strike: $XXX
- Expiry: [This Friday or next]
- Entry: Current price or on pullback to $XXX
- Target: $XXX
- Stop: $XXX

**QQQ Play:**  
- Direction: [CALL/PUT]
- Strike: $XXX
- Expiry: [This Friday or next]
- Entry: Current price or on pullback to $XXX
- Target: $XXX
- Stop: $XXX

---

## ⚠️ RISKS TO WATCH
- [List 2-3 specific risks that could invalidate the forecast]

IMPORTANT:
- Use the ACTUAL price data provided above
- Stay within the Expected Move range for predictions
- Reference specific gamma levels for support/resistance
- NO LaTeX - use plain text only
- Be specific with price levels"""

        return self.chat(prompt, include_context=False)
    
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
    
    def chat(self, user_message: str, include_context: bool = True, return_usage: bool = False) -> str:
        """Send a message to the AI and get a response
        
        Args:
            user_message: The user's question or prompt
            include_context: Whether to include market data context
            return_usage: If True, returns tuple (response, usage_dict)
        """
        if not self.is_available():
            error_msg = "❌ AI not configured. Please add your GROQ_API_KEY to use this feature."
            return (error_msg, None) if return_usage else error_msg
        
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
            
            # Extract token usage
            usage_dict = None
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                usage_dict = {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                }
                print(f"📊 Tokens - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
            content = response.choices[0].message.content
            
            if return_usage:
                return content, usage_dict
            return content
        except Exception as e:
            error_msg = f"❌ Error communicating with AI: {str(e)}"
            return (error_msg, None) if return_usage else error_msg
    
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
        
        # Get OPTIONS FLOW data
        options_data = self.get_options_analysis(ticker)
        
        # Get PRO TRADER data (IV Rank, Gamma Walls, Earnings, Volume)
        pro_data = self.get_pro_trader_analysis(ticker)
        
        # Get NEWS & ANALYST RATINGS
        news_data = self.get_news_analysis(ticker)
        
        prompt = f"""Analyze {ticker} based on the available data.

{live_data}

{options_data}

{pro_data}

{news_data}

Newsletter Historical scores for {ticker}:
{history_context}

Provide a comprehensive analysis with EXACTLY these sections:

## Current Setup
Describe the current setup as bullish, bearish, or neutral based on recent price action and technical indicators. Mention recent returns (week, month) and where it's trading relative to its range.

## Technical Analysis
Analyze RSI, SMA(10), SMA(20) relative to current price. Identify key support and resistance levels based on options OI (gamma walls). Mention the 30-day trading range. Clearly state if the trend is up or down based on price vs moving averages.

## Options Flow Sentiment
State the put/call ratio and overall sentiment. List ALL unusual options activity from the data above in detail:
- For each unusual call: Strike, Expiry Date, Volume vs OI, how many times normal
- For each unusual put: Strike, Expiry Date, Volume vs OI, how many times normal
This is critical information - don't skip any unusual activity.

## Trade Idea
Provide a SPECIFIC trade recommendation:
- Strategy type (calls, puts, spread, etc.)
- Exact strikes and expiry dates
- Entry price range
- Target price
- Stop-loss price
Base targets on gamma resistance levels and stops on gamma support levels.

## Risk Assessment
List key risks for the trade:
- Mention earnings if within 2 weeks
- Note any recent downgrades or negative news
- Address unusual options activity that contradicts the trade
- Recommend position sizing (2-3% of portfolio)
- Explain how to manage risk

IMPORTANT RULES:
- Use plain text only - NO LaTeX or special formatting
- Write prices with dollar sign like $125.50
- Be specific with strikes and dates - no vague recommendations
- Include all unusual options activity - this is important intel"""
        
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
