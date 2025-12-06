"""
Helper module for Streamlit to fetch data from droplet API
Use this in your Streamlit pages instead of direct API calls
"""

import requests
from typing import List, Dict, Optional
import streamlit as st

class DropletAPI:
    """Client for fetching data from droplet API"""
    
    def __init__(self, base_url: str = None):
        """
        Initialize API client
        
        Args:
            base_url: API base URL (e.g., 'http://138.197.210.166:8000')
        """
        if base_url is None:
            # Try to get from Streamlit secrets, fallback to env var
            base_url = st.secrets.get("DROPLET_API_URL", "http://138.197.210.166:8000")
        
        self.base_url = base_url.rstrip('/')
        self.timeout = 10  # seconds
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make GET request to API"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"API Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_watchlist(self, order_by: str = 'daily_change_pct', limit: int = 20) -> List[Dict]:
        """
        Get watchlist data
        
        Args:
            order_by: Sort column (daily_change_pct, volume, symbol, price)
            limit: Number of results (default 20)
        
        Returns:
            List of watchlist items
        """
        params = {'order_by': order_by, 'limit': limit}
        result = self._get('/api/watchlist', params=params)
        return result.get('data', []) if result.get('success') else []
    
    def get_whale_flows(self, sort_by: str = 'score', limit: int = 10, 
                        hours: int = 6) -> List[Dict]:
        """
        Get whale flows
        
        Args:
            sort_by: 'score' or 'time'
            limit: Number of results
            hours: Lookback hours
        
        Returns:
            List of whale flows
        """
        params = {
            'sort_by': sort_by,
            'limit': limit,
            'hours': hours
        }
        result = self._get('/api/whale_flows', params=params)
        return result.get('data', []) if result.get('success') else []
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        result = self._get('/api/stats')
        return result.get('stats', {}) if result.get('success') else {}
    
    def get_last_update(self) -> Dict:
        """Get last update times"""
        result = self._get('/api/last_update')
        if result.get('success'):
            return {
                'watchlist': result.get('watchlist'),
                'whale_flows': result.get('whale_flows')
            }
        return {}
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            result = self._get('/health')
            return result.get('status') == 'healthy'
        except:
            return False


# Cached version for better performance
@st.cache_data(ttl=300, show_spinner=False)
def fetch_watchlist(order_by: str = 'daily_change_pct', limit: int = 20) -> List[Dict]:
    """Cached watchlist fetcher (5 min TTL)"""
    api = DropletAPI()
    return api.get_watchlist(order_by=order_by, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_whale_flows(sort_by: str = 'score', limit: int = 10, 
                      hours: int = 6) -> List[Dict]:
    """Cached whale flows fetcher (5 min TTL)"""
    api = DropletAPI()
    return api.get_whale_flows(sort_by=sort_by, limit=limit, hours=hours)


# Example usage in Streamlit:
"""
from src.utils.droplet_api import fetch_watchlist, fetch_whale_flows, DropletAPI

# Simple usage with caching
watchlist = fetch_watchlist(order_by='daily_change_pct')
whale_flows = fetch_whale_flows(sort_by='score', limit=10)

# Or use the client directly
api = DropletAPI()
stats = api.get_stats()
last_update = api.get_last_update()
"""
