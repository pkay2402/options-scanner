"""
Dark Pool / FINRA Short Sale Data Utility
Fetches and calculates buy/sell ratios from FINRA data
"""

import requests
import pandas as pd
import io
from datetime import datetime, timedelta
from typing import Optional, Dict
import streamlit as st


def download_finra_short_sale_data(date: str) -> Optional[str]:
    """Download FINRA short sale data for a specific date"""
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    try:
        response = requests.get(url, timeout=5)
        return response.text if response.status_code == 200 else None
    except:
        return None


def process_finra_short_sale_data(data: Optional[str]) -> pd.DataFrame:
    """Process raw FINRA data into DataFrame"""
    if not data:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(data), delimiter="|")
        return df[df["Symbol"].str.len() <= 4]
    except:
        return pd.DataFrame()


def calculate_metrics(row: pd.Series, total_volume: float) -> dict:
    """Calculate buy/sell metrics from FINRA data"""
    short_volume = row.get('ShortVolume', 0)
    short_exempt_volume = row.get('ShortExemptVolume', 0)
    bought_volume = short_volume + short_exempt_volume
    sold_volume = total_volume - bought_volume
    buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')
    short_volume_ratio = bought_volume / total_volume if total_volume > 0 else 0
    return {
        'total_volume': total_volume,
        'bought_volume': bought_volume,
        'sold_volume': sold_volume,
        'buy_to_sell_ratio': round(buy_to_sell_ratio, 2),
        'short_volume_ratio': round(short_volume_ratio, 4)
    }


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_7day_dark_pool_sentiment(symbol: str) -> Dict:
    """
    Get 7-day cumulative dark pool sentiment for a symbol
    
    Returns:
        dict: {
            'ratio': float,  # Cumulative buy/sell ratio
            'sentiment': str,  # 'BULLISH', 'BEARISH', or 'NEUTRAL'
            'total_bought': int,
            'total_sold': int,
            'days_available': int
        }
    """
    total_bought = 0
    total_sold = 0
    days_found = 0
    
    # Try to get last 7 days of data
    for i in range(7):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date_str)
        
        if data:
            df = process_finra_short_sale_data(data)
            symbol_data = df[df['Symbol'] == symbol.upper()]
            
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                
                total_bought += metrics['bought_volume']
                total_sold += metrics['sold_volume']
                days_found += 1
    
    # Calculate cumulative ratio
    if total_sold > 0:
        ratio = total_bought / total_sold
    else:
        ratio = 0
    
    # Determine sentiment
    if ratio > 1.2:
        sentiment = 'BULLISH'
    elif ratio > 1.0:
        sentiment = 'NEUTRAL-BULL'
    elif ratio > 0.8:
        sentiment = 'NEUTRAL'
    else:
        sentiment = 'BEARISH'
    
    return {
        'ratio': round(ratio, 2),
        'sentiment': sentiment,
        'total_bought': int(total_bought),
        'total_sold': int(total_sold),
        'days_available': days_found
    }


def format_dark_pool_display(sentiment_data: Dict) -> tuple:
    """
    Format dark pool sentiment for display
    
    Returns:
        tuple: (display_text, color, icon)
    """
    ratio = sentiment_data['ratio']
    sentiment = sentiment_data['sentiment']
    days = sentiment_data['days_available']
    
    # Determine color and icon
    if sentiment == 'BULLISH':
        color = '#22c55e'  # green
        icon = '🟢'
    elif sentiment == 'NEUTRAL-BULL':
        color = '#10b981'  # light green
        icon = '🟡'
    elif sentiment == 'NEUTRAL':
        color = '#f59e0b'  # orange
        icon = '🟡'
    else:  # BEARISH
        color = '#ef4444'  # red
        icon = '🔴'
    
    # Format display text
    if ratio == float('inf'):
        display_text = f"{icon} Dark Pool (7d): ∞ ({sentiment})"
    else:
        display_text = f"{icon} Dark Pool (7d): {ratio:.2f} ({sentiment})"
    
    return display_text, color, icon
