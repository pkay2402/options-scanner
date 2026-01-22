"""
AI-Powered Stock Opportunity Screener
Identify stocks with potential for significant moves based on technical, fundamental, and catalyst analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import theme tracker for context
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from theme_tracker import THEMES


class StockScreener:
    """Intelligent stock screener for identifying potential big movers"""
    
    def __init__(self):
        self.opportunities = []
        
    def get_stock_data(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch stock data with error handling"""
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                return pd.DataFrame()
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
        except:
            return pd.DataFrame()
    
    def calculate_technical_score(self, ticker: str) -> Tuple[float, Dict]:
        """Calculate technical momentum score (0-100)"""
        df = self.get_stock_data(ticker, period="6mo")
        if df.empty or len(df) < 50:
            return 0, {}
        
        signals = {}
        score = 0
        
        # Current price and MAs
        current = df['Close'].iloc[-1]
        ma_20 = df['Close'].rolling(20).mean().iloc[-1]
        ma_50 = df['Close'].rolling(50).mean().iloc[-1]
        ma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else ma_50
        
        # 1. Trend strength (30 points)
        if current > ma_20 > ma_50:
            score += 30
            signals['trend'] = "Strong uptrend (above all MAs)"
        elif current > ma_50:
            score += 15
            signals['trend'] = "Moderate uptrend"
        else:
            signals['trend'] = "Downtrend or consolidation"
        
        # 2. Recent momentum (25 points)
        week_return = ((current / df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
        month_return = ((current / df['Close'].iloc[-21]) - 1) * 100 if len(df) >= 21 else 0
        
        if week_return > 5:
            score += 15
            signals['momentum'] = f"Strong weekly momentum (+{week_return:.1f}%)"
        elif week_return > 2:
            score += 8
            signals['momentum'] = f"Positive momentum (+{week_return:.1f}%)"
        else:
            signals['momentum'] = f"Weak momentum ({week_return:.1f}%)"
        
        if month_return > 10:
            score += 10
        elif month_return > 5:
            score += 5
        
        # 3. Volume surge (20 points)
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        recent_volume = df['Volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 2:
            score += 20
            signals['volume'] = f"Volume surge ({volume_ratio:.1f}x average)"
        elif volume_ratio > 1.5:
            score += 10
            signals['volume'] = f"Above average volume ({volume_ratio:.1f}x)"
        else:
            signals['volume'] = "Normal volume"
        
        # 4. Volatility setup (15 points) - looking for compression before expansion
        recent_volatility = df['Close'].pct_change().tail(10).std()
        historical_volatility = df['Close'].pct_change().tail(60).std()
        vol_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        
        if 0.5 < vol_ratio < 0.8:  # Compressed volatility
            score += 15
            signals['volatility'] = "Volatility compression (potential breakout setup)"
        elif vol_ratio > 1.5:  # Expanding volatility
            score += 10
            signals['volatility'] = "Volatility expansion (move in progress)"
        else:
            signals['volatility'] = "Normal volatility"
        
        # 5. Breakout potential (10 points)
        high_52w = df['High'].rolling(252).max().iloc[-1] if len(df) >= 252 else df['High'].max()
        distance_from_high = ((high_52w - current) / high_52w) * 100
        
        if distance_from_high < 5:
            score += 10
            signals['breakout'] = f"Near 52-week high (within {distance_from_high:.1f}%)"
        elif distance_from_high < 10:
            score += 5
            signals['breakout'] = f"Testing resistance ({distance_from_high:.1f}% from high)"
        else:
            signals['breakout'] = f"{distance_from_high:.1f}% below 52-week high"
        
        signals['score'] = score
        signals['current_price'] = current
        signals['week_return'] = week_return
        signals['month_return'] = month_return
        
        return score, signals
    
    def get_fundamental_catalyst(self, ticker: str) -> Dict:
        """Identify fundamental catalysts"""
        catalysts = {
            'has_catalyst': False,
            'catalyst_type': [],
            'description': []
        }
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check for upcoming earnings
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty:
                    catalysts['has_catalyst'] = True
                    catalysts['catalyst_type'].append('Earnings')
                    catalysts['description'].append('Earnings report upcoming')
            except:
                pass
            
            # High short interest
            if 'shortPercentOfFloat' in info:
                short_pct = info.get('shortPercentOfFloat', 0) * 100
                if short_pct > 15:
                    catalysts['has_catalyst'] = True
                    catalysts['catalyst_type'].append('Short Squeeze')
                    catalysts['description'].append(f'High short interest ({short_pct:.1f}%)')
            
            # Insider buying
            if 'heldPercentInsiders' in info:
                insider_pct = info.get('heldPercentInsiders', 0) * 100
                if insider_pct > 10:
                    catalysts['catalyst_type'].append('Insider Ownership')
                    catalysts['description'].append(f'High insider ownership ({insider_pct:.1f}%)')
            
            # Strong revenue growth
            if 'revenueGrowth' in info:
                rev_growth = info.get('revenueGrowth', 0) * 100
                if rev_growth > 20:
                    catalysts['has_catalyst'] = True
                    catalysts['catalyst_type'].append('Growth')
                    catalysts['description'].append(f'Strong revenue growth ({rev_growth:.1f}%)')
            
            # Analyst upgrades (based on recommendation)
            if 'recommendationKey' in info:
                rec = info.get('recommendationKey', '')
                if rec in ['strong_buy', 'buy']:
                    catalysts['catalyst_type'].append('Analyst Rating')
                    catalysts['description'].append(f'Analyst rating: {rec}')
            
        except Exception as e:
            pass
        
        return catalysts
    
    def screen_stock(self, ticker: str, description: str = "") -> Dict:
        """Complete screening analysis for a stock"""
        tech_score, tech_signals = self.calculate_technical_score(ticker)
        catalysts = self.get_fundamental_catalyst(ticker)
        
        # Calculate overall opportunity score
        opportunity_score = tech_score
        if catalysts['has_catalyst']:
            opportunity_score += 20  # Bonus for catalyst
        
        result = {
            'ticker': ticker,
            'description': description,
            'opportunity_score': min(opportunity_score, 100),
            'technical_score': tech_score,
            'technical_signals': tech_signals,
            'catalysts': catalysts,
            'current_price': tech_signals.get('current_price', 0),
            'week_return': tech_signals.get('week_return', 0),
            'month_return': tech_signals.get('month_return', 0)
        }
        
        return result
    
    def screen_themes(self, min_score: int = 60) -> List[Dict]:
        """Screen all stocks in themes for opportunities"""
        opportunities = []
        
        print("🔍 Scanning thematic stocks for opportunities...")
        print("=" * 70)
        
        for theme_name, stocks in THEMES.items():
            print(f"\nScanning: {theme_name}...")
            for ticker, desc in stocks.items():
                result = self.screen_stock(ticker, desc)
                if result['opportunity_score'] >= min_score:
                    result['theme'] = theme_name
                    opportunities.append(result)
                    print(f"  ✓ {ticker}: Score {result['opportunity_score']}")
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return opportunities
    
    def generate_opportunity_report(self, opportunities: List[Dict], top_n: int = 10):
        """Generate formatted report of top opportunities"""
        print("\n" + "=" * 70)
        print("🎯 TOP STOCK OPPORTUNITIES - AI SCREENING RESULTS")
        print("=" * 70)
        print(f"\nIdentified {len(opportunities)} high-probability setups")
        print(f"Showing top {min(top_n, len(opportunities))} opportunities\n")
        
        for i, opp in enumerate(opportunities[:top_n], 1):
            print(f"\n{'='*70}")
            print(f"#{i} - {opp['ticker']} ({opp['description']})")
            print(f"{'='*70}")
            print(f"Theme: {opp['theme']}")
            print(f"Opportunity Score: {opp['opportunity_score']}/100")
            print(f"Current Price: ${opp['current_price']:.2f}")
            print(f"Performance: 1W: {opp['week_return']:+.1f}% | 1M: {opp['month_return']:+.1f}%")
            
            print(f"\n📊 Technical Setup:")
            for key, value in opp['technical_signals'].items():
                if key not in ['score', 'current_price', 'week_return', 'month_return']:
                    print(f"  • {key.title()}: {value}")
            
            if opp['catalysts']['catalyst_type']:
                print(f"\n🔥 Catalysts:")
                for cat_type, desc in zip(opp['catalysts']['catalyst_type'], 
                                         opp['catalysts']['description']):
                    print(f"  • {cat_type}: {desc}")
            
            print(f"\n💡 Why It Could Move:")
            # Generate reasoning
            reasons = []
            
            if 'Strong uptrend' in opp['technical_signals'].get('trend', ''):
                reasons.append("Established uptrend with price above key moving averages")
            
            if 'Strong weekly momentum' in opp['technical_signals'].get('momentum', ''):
                reasons.append(f"Strong price momentum (+{opp['week_return']:.1f}% this week)")
            
            if 'surge' in opp['technical_signals'].get('volume', '').lower():
                reasons.append("Unusual volume indicating institutional interest")
            
            if 'compression' in opp['technical_signals'].get('volatility', '').lower():
                reasons.append("Volatility compression often precedes significant moves")
            
            if 'Near 52-week high' in opp['technical_signals'].get('breakout', ''):
                reasons.append("Approaching breakout above 52-week highs")
            
            if opp['catalysts']['has_catalyst']:
                for desc in opp['catalysts']['description']:
                    reasons.append(desc)
            
            for reason in reasons[:4]:  # Top 4 reasons
                print(f"  • {reason}")
            
            print(f"\n⚠️  Risk Level: {'HIGH' if opp['opportunity_score'] < 70 else 'MODERATE'}")
            print(f"📈 Potential: {'High conviction setup' if opp['opportunity_score'] >= 80 else 'Speculative opportunity'}")
    
    def generate_newsletter_section(self, opportunities: List[Dict], top_n: int = 5) -> str:
        """Generate newsletter-ready markdown content"""
        content = f"""## 🎯 AI-Identified Stock Opportunities

*Using technical momentum, volume analysis, and fundamental catalysts to identify potential movers*

This week's screening identified **{len(opportunities)} high-probability setups** across our thematic baskets. Here are the top opportunities:

"""
        
        for i, opp in enumerate(opportunities[:top_n], 1):
            content += f"""### {i}. **{opp['ticker']}** - {opp['description']}
**Theme**: {opp['theme']} | **Score**: {opp['opportunity_score']}/100 | **Price**: ${opp['current_price']:.2f}

**Performance**: 1W: {opp['week_return']:+.1f}% | 1M: {opp['month_return']:+.1f}%

**Technical Setup**:
"""
            # Add key signals
            if 'Strong uptrend' in opp['technical_signals'].get('trend', ''):
                content += f"- ✅ {opp['technical_signals']['trend']}\n"
            if opp['week_return'] > 3:
                content += f"- ✅ {opp['technical_signals']['momentum']}\n"
            if 'surge' in opp['technical_signals'].get('volume', '').lower():
                content += f"- ✅ {opp['technical_signals']['volume']}\n"
            
            content += f"\n**Why It Could Move**: "
            
            # Generate concise reasoning
            if opp['opportunity_score'] >= 80:
                content += f"Multiple bullish signals converging - technical momentum, "
                if 'surge' in opp['technical_signals'].get('volume', '').lower():
                    content += "unusual volume, "
                if opp['catalysts']['has_catalyst']:
                    content += f"plus {', '.join(opp['catalysts']['catalyst_type'][:2])} catalyst(s)"
                else:
                    content += "strong chart setup"
            else:
                content += "Emerging setup with positive momentum"
            
            content += f"\n\n**Risk/Reward**: {'High conviction' if opp['opportunity_score'] >= 80 else 'Speculative'} - {'Monitor closely' if opp['opportunity_score'] < 75 else 'Attractive risk/reward'}\n\n---\n\n"
        
        content += """
*Note: These are technical setups, not buy recommendations. Always do your own research, consider your risk tolerance, and use proper position sizing.*
"""
        
        return content


def main():
    """Run the AI screening system"""
    print("=" * 70)
    print("AI-POWERED STOCK OPPORTUNITY SCANNER")
    print(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    screener = StockScreener()
    
    # Screen all thematic stocks (score >= 60)
    opportunities = screener.screen_themes(min_score=60)
    
    # Generate report
    screener.generate_opportunity_report(opportunities, top_n=10)
    
    # Generate newsletter content
    print("\n\n" + "=" * 70)
    print("📧 NEWSLETTER SECTION")
    print("=" * 70)
    newsletter_content = screener.generate_newsletter_section(opportunities, top_n=5)
    print(newsletter_content)
    
    return opportunities


if __name__ == "__main__":
    opportunities = main()
