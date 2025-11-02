"""
Schwab API Client
Handles authentication and API calls to Schwab's trading platform
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from authlib.integrations.httpx_client import OAuth2Client
import asyncio
import aiohttp
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

class SchwabAPIError(Exception):
    """Custom exception for Schwab API errors"""
    pass

class SchwabClient:
    """
    Schwab API Client for options and market data
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.schwabapi.com"
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.session = requests.Session()
        
    def authenticate(self) -> bool:
        """
        Authenticate with Schwab API using OAuth2
        """
        try:
            auth_url = f"{self.base_url}/oauth/token"
            
            auth_data = {
                'grant_type': 'authorization_code',
                'code': self.settings.SCHWAB_AUTH_CODE,
                'redirect_uri': self.settings.SCHWAB_REDIRECT_URI,
                'client_id': self.settings.SCHWAB_CLIENT_ID
            }
            
            auth = HTTPBasicAuth(
                self.settings.SCHWAB_CLIENT_ID,
                self.settings.SCHWAB_CLIENT_SECRET
            )
            
            response = self.session.post(
                auth_url,
                data=auth_data,
                auth=auth,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data['refresh_token']
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = time.time() + expires_in
                
                # Set default headers
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                })
                
                logger.info("Successfully authenticated with Schwab API")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using refresh token
        """
        try:
            refresh_url = f"{self.base_url}/oauth/token"
            
            refresh_data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.settings.SCHWAB_CLIENT_ID
            }
            
            auth = HTTPBasicAuth(
                self.settings.SCHWAB_CLIENT_ID,
                self.settings.SCHWAB_CLIENT_SECRET
            )
            
            response = requests.post(
                refresh_url,
                data=refresh_data,
                auth=auth,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = time.time() + expires_in
                
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                
                logger.info("Successfully refreshed access token")
                return True
            else:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            return False
    
    def _ensure_authenticated(self):
        """
        Ensure we have a valid access token
        """
        if not self.access_token:
            if not self.authenticate():
                raise SchwabAPIError("Failed to authenticate")
        elif self.token_expires_at and time.time() >= self.token_expires_at - 300:  # Refresh 5 min early
            if not self.refresh_access_token():
                if not self.authenticate():
                    raise SchwabAPIError("Failed to refresh or re-authenticate")
    
    def get_options_chain(self, symbol: str, contract_type: str = "ALL", 
                         strike_count: int = 20, include_quotes: bool = True,
                         strategy: str = "SINGLE", interval: str = None,
                         strike: float = None, range_type: str = "ALL",
                         from_date: str = None, to_date: str = None,
                         volatility: float = None, underlying_price: float = None,
                         interest_rate: float = None, days_to_expiration: int = None,
                         exp_month: str = "ALL", option_type: str = "ALL") -> Dict:
        """
        Get options chain for a symbol
        """
        self._ensure_authenticated()
        
        params = {
            'symbol': symbol,
            'contractType': contract_type,
            'strikeCount': strike_count,
            'includeQuotes': include_quotes,
            'strategy': strategy,
            'range': range_type,
            'expMonth': exp_month,
            'optionType': option_type
        }
        
        # Add optional parameters
        if interval:
            params['interval'] = interval
        if strike:
            params['strike'] = strike
        if from_date:
            params['fromDate'] = from_date
        if to_date:
            params['toDate'] = to_date
        if volatility:
            params['volatility'] = volatility
        if underlying_price:
            params['underlyingPrice'] = underlying_price
        if interest_rate:
            params['interestRate'] = interest_rate
        if days_to_expiration:
            params['daysToExpiration'] = days_to_expiration
        
        url = f"{self.base_url}/marketdata/v1/chains"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting options chain for {symbol}: {str(e)}")
            raise SchwabAPIError(f"Failed to get options chain: {str(e)}")
    
    def get_option_quote(self, symbol: str) -> Dict:
        """
        Get quote for a specific option symbol
        """
        self._ensure_authenticated()
        
        url = f"{self.base_url}/marketdata/v1/{symbol}/quotes"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting option quote for {symbol}: {str(e)}")
            raise SchwabAPIError(f"Failed to get option quote: {str(e)}")
    
    def get_market_hours(self, markets: List[str] = None, date: str = None) -> Dict:
        """
        Get market hours for specified markets
        """
        self._ensure_authenticated()
        
        params = {}
        if markets:
            params['markets'] = ','.join(markets)
        if date:
            params['date'] = date
        
        url = f"{self.base_url}/marketdata/v1/markets"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting market hours: {str(e)}")
            raise SchwabAPIError(f"Failed to get market hours: {str(e)}")
    
    def get_movers(self, market: str, direction: str = "up", change: str = "percent") -> Dict:
        """
        Get market movers
        """
        self._ensure_authenticated()
        
        params = {
            'direction': direction,
            'change': change
        }
        
        url = f"{self.base_url}/marketdata/v1/movers/{market}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting movers for {market}: {str(e)}")
            raise SchwabAPIError(f"Failed to get movers: {str(e)}")
    
    def get_price_history(self, symbol: str, period_type: str = "day",
                         period: int = 10, frequency_type: str = "minute",
                         frequency: int = 1, start_date: int = None,
                         end_date: int = None, need_extended_hours_data: bool = True) -> Dict:
        """
        Get price history for a symbol
        """
        self._ensure_authenticated()
        
        params = {
            'periodType': period_type,
            'period': period,
            'frequencyType': frequency_type,
            'frequency': frequency,
            'needExtendedHoursData': need_extended_hours_data
        }
        
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date
        
        url = f"{self.base_url}/marketdata/v1/pricehistory"
        
        try:
            response = self.session.get(url, params={**params, 'symbol': symbol})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting price history for {symbol}: {str(e)}")
            raise SchwabAPIError(f"Failed to get price history: {str(e)}")
    
    def search_instruments(self, symbol: str, projection: str = "symbol-search") -> Dict:
        """
        Search for instruments
        """
        self._ensure_authenticated()
        
        params = {
            'symbol': symbol,
            'projection': projection
        }
        
        url = f"{self.base_url}/marketdata/v1/instruments"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching instruments for {symbol}: {str(e)}")
            raise SchwabAPIError(f"Failed to search instruments: {str(e)}")
    
    def get_quotes(self, symbols: List[str], fields: str = None) -> Dict:
        """
        Get quotes for multiple symbols
        """
        self._ensure_authenticated()
        
        params = {
            'symbols': ','.join(symbols)
        }
        
        if fields:
            params['fields'] = fields
        
        url = f"{self.base_url}/marketdata/v1/quotes"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting quotes for {symbols}: {str(e)}")
            raise SchwabAPIError(f"Failed to get quotes: {str(e)}")

# Async version for high-performance operations
class AsyncSchwabClient:
    """
    Async version of Schwab API Client for high-performance operations
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.schwabapi.com"
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
    async def authenticate(self) -> bool:
        """
        Async authentication with Schwab API
        """
        try:
            auth_url = f"{self.base_url}/oauth/token"
            
            auth_data = {
                'grant_type': 'authorization_code',
                'code': self.settings.SCHWAB_AUTH_CODE,
                'redirect_uri': self.settings.SCHWAB_REDIRECT_URI,
                'client_id': self.settings.SCHWAB_CLIENT_ID
            }
            
            auth = aiohttp.BasicAuth(
                self.settings.SCHWAB_CLIENT_ID,
                self.settings.SCHWAB_CLIENT_SECRET
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    auth_url,
                    data=auth_data,
                    auth=auth,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as response:
                    
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data['access_token']
                        self.refresh_token = token_data['refresh_token']
                        expires_in = token_data.get('expires_in', 3600)
                        self.token_expires_at = time.time() + expires_in
                        
                        logger.info("Successfully authenticated with Schwab API (async)")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Async authentication failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Async authentication error: {str(e)}")
            return False
    
    async def get_multiple_options_chains(self, symbols: List[str], **kwargs) -> Dict[str, Dict]:
        """
        Get options chains for multiple symbols concurrently
        """
        if not self.access_token:
            await self.authenticate()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        async def fetch_chain(session, symbol):
            params = {'symbol': symbol, **kwargs}
            url = f"{self.base_url}/marketdata/v1/chains"
            
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return symbol, await response.json()
                    else:
                        logger.error(f"Error fetching options chain for {symbol}: {response.status}")
                        return symbol, None
            except Exception as e:
                logger.error(f"Exception fetching options chain for {symbol}: {str(e)}")
                return symbol, None
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_chain(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            
            return {symbol: data for symbol, data in results if data is not None}