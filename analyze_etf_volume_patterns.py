#!/usr/bin/env python3
"""
Analyze volume patterns in leveraged ETFs that lead to 10%+ price moves.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class VolumePatternAnalyzer:
    """Analyze volume patterns that precede significant price moves."""
    
    def __init__(self, symbols_file: str = 'extracted_symbols.csv', lookback_days: int = 30):
        self.symbols_file = symbols_file
        self.lookback_days = lookback_days
        self.symbols = self._load_symbols()
        self.results = []
        
    def _load_symbols(self) -> List[str]:
        """Load ETF symbols from CSV."""
        df = pd.read_csv(self.symbols_file, encoding='utf-8-sig')
        symbols = df['Symbol'].tolist()
        print(f"✅ Loaded {len(symbols)} ETF symbols")
        return symbols
    
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty or len(data) < 10:
                return pd.DataFrame()
            
            return data
        except Exception as e:
            print(f"⚠️  Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def identify_big_moves(self, data: pd.DataFrame, threshold: float = 10.0) -> List[Dict]:
        """Identify days with 10%+ moves."""
        moves = []
        
        # Calculate daily returns
        data['Return'] = data['Close'].pct_change() * 100
        
        # Find 10%+ single day moves
        for idx in range(1, len(data)):
            daily_return = data['Return'].iloc[idx]
            
            if abs(daily_return) >= threshold:
                moves.append({
                    'date': data.index[idx],
                    'return': daily_return,
                    'direction': 'UP' if daily_return > 0 else 'DOWN',
                    'close': data['Close'].iloc[idx],
                    'volume': data['Volume'].iloc[idx],
                    'move_type': 'single_day'
                })
        
        # Find 10%+ moves over 2-3 days
        for idx in range(2, len(data)):
            two_day_return = ((data['Close'].iloc[idx] / data['Close'].iloc[idx-2]) - 1) * 100
            
            if abs(two_day_return) >= threshold and abs(data['Return'].iloc[idx]) < threshold:
                moves.append({
                    'date': data.index[idx],
                    'return': two_day_return,
                    'direction': 'UP' if two_day_return > 0 else 'DOWN',
                    'close': data['Close'].iloc[idx],
                    'volume': data['Volume'].iloc[idx],
                    'move_type': '2_day'
                })
        
        return moves
    
    def analyze_volume_pattern(self, data: pd.DataFrame, move_date, lookback: int = 5) -> Dict:
        """Analyze volume pattern before a big move."""
        try:
            move_idx = data.index.get_loc(move_date)
            
            if move_idx < lookback:
                return {}
            
            # Get volume data before the move
            pre_move_volume = data['Volume'].iloc[move_idx-lookback:move_idx].values
            move_volume = data['Volume'].iloc[move_idx]
            
            # Calculate volume metrics
            avg_volume = np.mean(pre_move_volume)
            volume_surge = (move_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
            
            # Check for consecutive volume increases
            volume_increases = 0
            for i in range(len(pre_move_volume) - 1):
                if pre_move_volume[i+1] > pre_move_volume[i]:
                    volume_increases += 1
            
            # Calculate relative volume trend
            recent_3d_avg = np.mean(pre_move_volume[-3:]) if len(pre_move_volume) >= 3 else avg_volume
            earlier_vol_avg = np.mean(pre_move_volume[:-3]) if len(pre_move_volume) > 3 else avg_volume
            volume_trend = (recent_3d_avg / earlier_vol_avg - 1) * 100 if earlier_vol_avg > 0 else 0
            
            # Get price action before move
            pre_move_returns = data['Close'].iloc[move_idx-lookback:move_idx].pct_change().values[1:]
            avg_pre_move_return = np.mean(pre_move_returns) * 100
            
            return {
                'avg_volume_5d': int(avg_volume),
                'move_volume': int(move_volume),
                'volume_surge_pct': round(volume_surge, 1),
                'consecutive_increases': volume_increases,
                'volume_trend_pct': round(volume_trend, 1),
                'avg_return_5d': round(avg_pre_move_return, 2),
                'volume_pattern': self._classify_pattern(volume_surge, volume_increases, volume_trend)
            }
        except Exception as e:
            return {}
    
    def _classify_pattern(self, surge: float, increases: int, trend: float) -> str:
        """Classify the volume pattern."""
        if surge > 100:
            return "EXPLOSIVE_SURGE"
        elif surge > 50:
            return "STRONG_SURGE"
        elif increases >= 3 and trend > 20:
            return "BUILDING_MOMENTUM"
        elif trend > 30:
            return "STEADY_INCREASE"
        elif surge < -20:
            return "DRY_UP"
        else:
            return "NORMAL"
    
    def analyze_symbol(self, symbol: str) -> List[Dict]:
        """Analyze a single symbol for volume patterns."""
        data = self.fetch_data(symbol)
        
        if data.empty:
            return []
        
        moves = self.identify_big_moves(data)
        
        if not moves:
            return []
        
        results = []
        for move in moves:
            volume_analysis = self.analyze_volume_pattern(data, move['date'])
            
            if volume_analysis:
                result = {
                    'symbol': symbol,
                    'date': move['date'].strftime('%Y-%m-%d'),
                    'return': round(move['return'], 2),
                    'direction': move['direction'],
                    'move_type': move['move_type'],
                    **volume_analysis
                }
                results.append(result)
        
        return results
    
    def analyze_all_symbols(self, max_symbols: int = None):
        """Analyze all symbols for volume patterns."""
        symbols_to_analyze = self.symbols[:max_symbols] if max_symbols else self.symbols
        
        print(f"\n📊 Analyzing {len(symbols_to_analyze)} ETFs for 10%+ moves...")
        print("=" * 80)
        
        for i, symbol in enumerate(symbols_to_analyze, 1):
            print(f"[{i}/{len(symbols_to_analyze)}] Analyzing {symbol}...", end=' ')
            
            symbol_results = self.analyze_symbol(symbol)
            
            if symbol_results:
                self.results.extend(symbol_results)
                print(f"✅ Found {len(symbol_results)} big move(s)")
            else:
                print("⚪ No significant moves")
        
        print("=" * 80)
        print(f"\n✅ Analysis complete! Found {len(self.results)} total big moves\n")
    
    def generate_report(self) -> pd.DataFrame:
        """Generate analysis report."""
        if not self.results:
            print("❌ No results to report")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # Sort by volume surge
        df = df.sort_values('volume_surge_pct', ascending=False)
        
        print("\n" + "="*80)
        print("📈 VOLUME PATTERN ANALYSIS - TOP FINDINGS")
        print("="*80)
        
        # Pattern distribution
        print("\n🔍 VOLUME PATTERN DISTRIBUTION:")
        pattern_counts = df['volume_pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {pattern:20} : {count:3} occurrences ({pct:.1f}%)")
        
        # Average metrics by pattern
        print("\n📊 AVERAGE METRICS BY PATTERN:")
        pattern_stats = df.groupby('volume_pattern').agg({
            'volume_surge_pct': 'mean',
            'return': 'mean',
            'consecutive_increases': 'mean'
        }).round(2)
        print(pattern_stats)
        
        # Top volume surges
        print("\n🚀 TOP 20 VOLUME SURGES BEFORE 10%+ MOVES:")
        print("-" * 80)
        top_surges = df.nlargest(20, 'volume_surge_pct')[
            ['symbol', 'date', 'return', 'direction', 'volume_surge_pct', 'volume_pattern']
        ]
        print(top_surges.to_string(index=False))
        
        # Building momentum examples
        momentum_patterns = df[df['volume_pattern'] == 'BUILDING_MOMENTUM']
        if not momentum_patterns.empty:
            print(f"\n🔥 BUILDING MOMENTUM PATTERNS ({len(momentum_patterns)} found):")
            print("-" * 80)
            print(momentum_patterns.nlargest(10, 'return')[
                ['symbol', 'date', 'return', 'consecutive_increases', 'volume_trend_pct']
            ].to_string(index=False))
        
        # Direction analysis
        print("\n📈 DIRECTION ANALYSIS:")
        direction_stats = df.groupby('direction').agg({
            'return': ['mean', 'count'],
            'volume_surge_pct': 'mean'
        }).round(2)
        print(direction_stats)
        
        # Save to CSV
        output_file = 'etf_volume_analysis_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return df
    
    def print_key_insights(self, df: pd.DataFrame):
        """Print key insights from the analysis."""
        if df.empty:
            return
        
        print("\n" + "="*80)
        print("💡 KEY INSIGHTS")
        print("="*80)
        
        # Best pattern for big moves
        upward_moves = df[df['direction'] == 'UP']
        if not upward_moves.empty:
            best_pattern = upward_moves.groupby('volume_pattern')['return'].mean().idxmax()
            best_avg_return = upward_moves.groupby('volume_pattern')['return'].mean().max()
            print(f"\n✅ Best pattern for upward moves: {best_pattern}")
            print(f"   Average return: {best_avg_return:.2f}%")
        
        # Volume surge threshold
        avg_surge_up = df[df['direction'] == 'UP']['volume_surge_pct'].mean()
        avg_surge_down = df[df['direction'] == 'DOWN']['volume_surge_pct'].mean()
        print(f"\n📊 Average volume surge before moves:")
        print(f"   UP moves:   {avg_surge_up:.1f}%")
        print(f"   DOWN moves: {avg_surge_down:.1f}%")
        
        # Consecutive increases impact
        high_momentum = df[df['consecutive_increases'] >= 3]
        if not high_momentum.empty:
            avg_return_high_momentum = abs(high_momentum['return']).mean()
            print(f"\n🔥 ETFs with 3+ consecutive volume increases:")
            print(f"   Average absolute return: {avg_return_high_momentum:.2f}%")
            print(f"   Occurrences: {len(high_momentum)}")
        
        # Most active ETFs
        print(f"\n🎯 Most active ETFs (most 10%+ moves):")
        top_active = df['symbol'].value_counts().head(10)
        for symbol, count in top_active.items():
            print(f"   {symbol}: {count} big moves")
        
        print("\n" + "="*80)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("🔍 ETF VOLUME PATTERN ANALYZER")
    print("="*80)
    print("Analyzing volume patterns that lead to 10%+ moves in leveraged ETFs")
    print("Lookback period: Last 30 days")
    print("="*80 + "\n")
    
    analyzer = VolumePatternAnalyzer(lookback_days=30)
    
    # Analyze all symbols (or limit for testing)
    analyzer.analyze_all_symbols(max_symbols=None)  # Set to 20 for quick test
    
    # Generate and display report
    df = analyzer.generate_report()
    
    # Print insights
    if not df.empty:
        analyzer.print_key_insights(df)
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
