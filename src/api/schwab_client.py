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
import pandas as pd
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

class SchwabAPIError(Exception):
    """Custom exception for Schwab API errors"""
    pass

class SchwabClient:
    """
    Schwab API Client for options and market data using OAuth2Client
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.filepath = Path(__file__).parent.parent.parent / 'schwab_client.json'
        self.TOKEN_ENDPOINT = 'https://api.schwabapi.com/v1/oauth/token'
        self.base_url = "https://api.schwabapi.com"
        self.session: OAuth2Client = None
        
        # Try to get credentials from Streamlit secrets if environment vars are empty
        client_id = self.settings.SCHWAB_CLIENT_ID
        client_secret = self.settings.SCHWAB_CLIENT_SECRET
        redirect_uri = self.settings.SCHWAB_REDIRECT_URI
        
        if not client_id or not client_secret:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'schwab' in st.secrets:
                    client_id = st.secrets['schwab'].get('app_key', '')
                    client_secret = st.secrets['schwab'].get('app_secret', '')
                    redirect_uri = st.secrets['schwab'].get('redirect_uri', 'https://127.0.0.1:8182')
            except:
                pass
        
        self.config = {
            'client': {
                'api_key': client_id,
                'app_secret': client_secret,
                'callback': redirect_uri,
                'setup': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            },
            'token': {}
        }
        self.load()
        
    def setup(self) -> bool:
        """Setup OAuth2 authentication flow"""
        try:
            oauth = OAuth2Client(
                self.config['client']['api_key'], 
                redirect_uri=self.config['client']['callback']
            )
            authorization_url, state = oauth.create_authorization_url(
                'https://api.schwabapi.com/v1/oauth/authorize'
            )
            print('Click the link below:')
            print(authorization_url)
            
            redirected_url = input('Paste URL: ').strip()
            
            self.config['token'] = oauth.fetch_token(
                self.TOKEN_ENDPOINT,
                authorization_response=redirected_url,
                client_id=self.config['client']['api_key'],
                auth=(self.config['client']['api_key'], self.config['client']['app_secret'])
            )
            
            self.save()
            self.load_session()
            logger.info("Successfully authenticated with Schwab API")
            return True
            
        except Exception as e:
            logger.error(f'Setup failed: {e}')
            return False

    def authenticate(self) -> bool:
        """
        Authenticate with Schwab API using stored tokens or setup flow
        """
        try:
            # Check if we have a valid session
            if self.ensure_valid_session():
                return True
                
            # Try to load existing session
            if self.config.get('token') and self.load_session():
                if self.ensure_valid_session():
                    return True
            
            # If no valid session, need to setup
            logger.info("No valid session found, running setup...")
            return self.setup()
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False

    def ensure_valid_session(self) -> bool:
        """Ensure we have a valid session, refresh if needed"""
        if not self.check_session():
            logger.info("Session check failed, attempting to refresh...")
            if not self.refresh_token():
                logger.warning("Token refresh failed, need to re-authenticate")
                return False
        return True

    def check_session(self) -> bool:
        """Check if current session is valid"""
        try:
            if self.session is None:
                self.load_session()
            
            if not self.session or not self.session.token:
                return False
                
            # Check if token exists
            if 'expires_at' not in self.session.token:
                logger.warning("No expiration time found in token")
                return False
                
            expires = datetime.fromtimestamp(int(self.session.token['expires_at']), timezone.utc)
            current = datetime.now(timezone.utc)
            
            # Refresh token if it expires within 5 minutes (300 seconds buffer)
            if (expires - current).total_seconds() <= 300:
                logger.info(f"Token expires at {expires}, refreshing...")
                return self.refresh_token()
            
            return True
            
        except Exception as e:
            logger.error(f'Checking session failed: {e}')
            return False

    def refresh_token(self) -> bool:
        """Refresh the access token using refresh token"""
        try:
            if self.session is None:
                self.load_session()
                
            token = self.config['token']
            
            # Check if refresh token exists
            if 'refresh_token' not in token:
                logger.warning("No refresh token available, need to re-authenticate")
                return False
                
            logger.info("Refreshing access token...")
            new_token = self.session.fetch_token(
                self.TOKEN_ENDPOINT,
                grant_type='refresh_token',
                refresh_token=token['refresh_token']
            )
            
            # Update token with new expiration time
            new_token['expires_at'] = int(time.time()) + new_token.get('expires_in', 1800)
            
            self.config['token'] = new_token
            self.save()
            self.load_session()
            logger.info('Token refreshed successfully')
            return True
            
        except Exception as e:
            logger.error(f'Token could not be refreshed: {e}')
            logger.warning("You may need to re-authenticate")
            return False

    def write_token(self, token, *args, **kwargs):
        """Callback function for token updates"""
        try:
            self.config['token'] = token
            self.save()
            self.load_session()
        except Exception as e:
            logger.error(f'Token could not be loaded: {e}')

    def save(self):
        """Save configuration to file"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f'Configuration could not be saved: {e}')

    def load(self):
        """Load configuration from Streamlit secrets or file"""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'schwab' in st.secrets:
                    logger.info("Loading tokens from Streamlit secrets")
                    schwab_secrets = st.secrets['schwab']
                    
                    # Build config from secrets
                    self.config = {
                        'client': {
                            'api_key': schwab_secrets['app_key'],
                            'app_secret': schwab_secrets['app_secret'],
                            'callback': schwab_secrets.get('redirect_uri', 'https://127.0.0.1:8182'),
                            'setup': schwab_secrets.get('setup', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
                        },
                        'token': {
                            'access_token': schwab_secrets['access_token'],
                            'refresh_token': schwab_secrets['refresh_token'],
                            'id_token': schwab_secrets['id_token'],
                            'token_type': schwab_secrets.get('token_type', 'Bearer'),
                            'expires_in': schwab_secrets.get('expires_in', 1800),
                            'scope': schwab_secrets.get('scope', 'api'),
                            'refresh_token_expires_in': schwab_secrets.get('refresh_token_expires_in', 604800),
                        }
                    }
                    
                    # Add expires_at if not present
                    if 'expires_at' in schwab_secrets:
                        self.config['token']['expires_at'] = schwab_secrets['expires_at']
                    else:
                        # Calculate from expires_in
                        self.config['token']['expires_at'] = int(time.time()) + self.config['token']['expires_in']
                    
                    # Add refresh token created timestamp
                    if 'refresh_token_created_at' in schwab_secrets:
                        self.config['token']['refresh_token_created_at'] = schwab_secrets['refresh_token_created_at']
                    
                    logger.info("Successfully loaded tokens from Streamlit secrets")
                    return
            except ImportError:
                # Streamlit not available, fallback to file
                pass
            except Exception as e:
                logger.debug(f"Streamlit secrets not available: {e}")
            
            # Fallback to local file (for local development)
            if not self.filepath.exists():
                logger.info('Config file not found, will create on first authentication')
                return
            with open(self.filepath, 'r') as f:
                self.config = json.load(f)
                logger.info("Loaded tokens from local file")
        except Exception as e:
            logger.error(f'Configuration could not be loaded: {e}')

    def load_session(self) -> bool:
        """Load OAuth2 session"""
        try:
            if 'client' not in self.config or 'api_key' not in self.config['client'] or 'app_secret' not in self.config['client']:
                raise Exception('API Key or App Secret missing in configuration')
                
            token = self.config.get('token')
            if not token:
                return False
                
            self.session = OAuth2Client(
                self.config['client']['api_key'],
                self.config['client']['app_secret'],
                token=token,
                token_endpoint=self.TOKEN_ENDPOINT,
                update_token=self.write_token
            )
            return True
            
        except Exception as e:
            logger.error(f'Could not load session: {e}')
            return False

    def clean_symbol(self, symbol: str) -> str:
        """
        Clean the symbol format
        Note: Preserves $ prefix for index symbols like $SPX, $DJX, etc.
        Note: Preserves / prefix for futures symbols like /ES, /NQ, etc.
        """
        # Remove .X suffix but preserve special prefixes ($ for indices, / for futures)
        return symbol.replace('.X', '')

    def get_quote(self, underlying: str) -> dict:
        """Get quote for a stock symbol (also supports futures like /ES, /NQ)"""
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')
            
            underlying = self.clean_symbol(underlying)
            endpoint = f'https://api.schwabapi.com/marketdata/v1/quotes'
            params = {'symbols': underlying}
            
            logger.info(f'Fetching quote for symbol: {underlying}')
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            result = response.json()
            logger.info(f'Quote response keys: {list(result.keys()) if result else None}')
            return result
            
        except Exception as e:
            logger.error(f'Could not get quote for {underlying}: {e}')
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f'Response status: {e.response.status_code}')
                logger.error(f'Response text: {e.response.text}')
            return None

    def get_options_chain(self, symbol: str, contract_type: str = "ALL", 
                         strike_count: int = None, include_quotes: bool = True,
                         strategy: str = "SINGLE", interval: str = None,
                         strike: float = None, range_type: str = "ALL",
                         from_date: str = None, to_date: str = None,
                         volatility: float = None, underlying_price: float = None,
                         interest_rate: float = None, days_to_expiration: int = None) -> Dict:
        """
        Get options chain for a symbol
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')
                
            symbol = self.clean_symbol(symbol)
            logger.info(f'Getting options chain for {symbol}')
            
            endpoint = 'https://api.schwabapi.com/marketdata/v1/chains'
            params = {
                'symbol': symbol,
                'contractType': contract_type,
                'includeQuotes': 'TRUE'  # Request greeks and quotes
            }
            
            # Add optional parameters
            if strike_count:
                params['strikeCount'] = strike_count
            if from_date:
                params['fromDate'] = from_date
            if to_date:
                params['toDate'] = to_date
                
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f'Could not get options chain for {symbol}: {e}')
            return None

    def get_option_quote(self, symbol: str) -> Dict:
        """
        Get quote for a specific option symbol
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')

            url = f"{self.base_url}/marketdata/v1/{symbol}/quotes"
            
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting option quote for {symbol}: {str(e)}")
            raise SchwabAPIError(f"Failed to get option quote: {str(e)}")

    def get_market_hours(self, markets: str = 'equity', date: str = None) -> Dict:
        """
        Get market hours for specified markets
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')

            params = {'markets': markets}
            if date:
                params['date'] = date

            url = f"{self.base_url}/marketdata/v1/markets"
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting market hours: {str(e)}")
            raise SchwabAPIError(f"Failed to get market hours: {str(e)}")

    def get_movers(self, market: str, direction: str = "up", change: str = "percent") -> Dict:
        """
        Get market movers
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')

            params = {
                'sort': direction,
                'frequency': change
            }

            url = f"{self.base_url}/marketdata/v1/{market}/movers"
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting movers for {market}: {str(e)}")
            raise SchwabAPIError(f"Failed to get movers: {str(e)}")

    def get_price_history(self, symbol: str, period_type: str = "day",
                         period: int = 10, frequency_type: str = "minute",
                         frequency: int = 1, start_date: int = None,
                         end_date: int = None, need_extended_hours: bool = True,
                         need_previous_close: bool = False) -> Dict:
        """
        Get price history for a symbol
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')

            symbol = self.clean_symbol(symbol)

            params = {
                'symbol': symbol,
                'periodType': period_type,
                'period': period,
                'frequencyType': frequency_type,
                'frequency': frequency,
                'needExtendedHoursData': str(need_extended_hours).lower(),
                'needPreviousClose': str(need_previous_close).lower()
            }

            if start_date:
                params['startDate'] = start_date
            if end_date:
                params['endDate'] = end_date

            url = f"{self.base_url}/marketdata/v1/pricehistory"
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {str(e)}")
            raise SchwabAPIError(f"Failed to get price history: {str(e)}")

    def get_quotes(self, symbols: List[str], fields: str = None) -> Dict:
        """
        Get quotes for multiple symbols
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')

            # Clean symbols
            clean_symbols = [self.clean_symbol(s) for s in symbols]
            
            params = {
                'symbols': ','.join(clean_symbols)
            }
            
            if fields:
                params['fields'] = fields

            url = f"{self.base_url}/marketdata/v1/quotes"
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting quotes: {str(e)}")
            raise SchwabAPIError(f"Failed to get quotes: {str(e)}")

    def options_chain_to_dataframe(self, options_chain: dict) -> tuple:
        """Convert options chain to DataFrame"""
        try:
            spot_price = options_chain.get('underlyingPrice', None)
            if spot_price is None:
                raise ValueError("underlyingPrice not found in options_chain")
                
            calls = options_chain.get('callExpDateMap', {})
            puts = options_chain.get('putExpDateMap', {})
                        
            call_data = [option for exp_date in calls for strike in calls[exp_date] for option in calls[exp_date][strike]]
            put_data = [option for exp_date in puts for strike in puts[exp_date] for option in puts[exp_date][strike]]
            
            calls_df = pd.DataFrame(call_data)
            puts_df = pd.DataFrame(put_data)
        
            calls_df['spotprice'] = spot_price
            puts_df['spotprice'] = spot_price
            options_df = pd.concat([calls_df, puts_df], ignore_index=True)
            
            # Convert and verify data types
            if not options_df.empty:
                options_df['strike'] = pd.to_numeric(options_df['strikePrice'])
                options_df['openInterest'] = pd.to_numeric(options_df['openInterest'])
                options_df['volatility'] = pd.to_numeric(options_df['volatility'])
                options_df['gamma'] = pd.to_numeric(options_df['gamma'])
                options_df['daysToExpiration'] = pd.to_numeric(options_df['daysToExpiration'])
                        
            return options_df, spot_price
            
        except Exception as e:
            logger.error(f'Could not convert options chain to DataFrame: {e}')
            return pd.DataFrame(), None

    def get_token_status(self) -> dict:
        """Get current token status"""
        try:
            if not self.config.get('token'):
                return {"status": "no_token", "message": "No token found"}
                
            token = self.config['token']
            if 'expires_at' not in token:
                return {"status": "invalid", "message": "Invalid token format"}
                
            expires = datetime.fromtimestamp(int(token['expires_at']), timezone.utc)
            current = datetime.now(timezone.utc)
            seconds_left = (expires - current).total_seconds()
            
            if seconds_left <= 0:
                return {"status": "expired", "message": "Token expired", "seconds_left": seconds_left}
            elif seconds_left <= 300:  # 5 minutes
                return {"status": "expiring", "message": "Token expiring soon", "seconds_left": seconds_left}
            else:
                return {"status": "valid", "message": "Token valid", "seconds_left": seconds_left}
                
        except Exception as e:
            return {"status": "error", "message": f"Error checking token: {e}"}
    
    def get_instrument_fundamental(self, symbol: str, projection: str = "fundamental") -> Dict:
        """
        Get instrument fundamental data by symbol
        
        Args:
            symbol: Stock ticker symbol
            projection: Type of data to return. Options: 'symbol-search', 'symbol-regex', 'desc-search', 'desc-regex', 'search', 'fundamental'
        
        Returns:
            Dictionary containing instrument fundamental data including:
            - symbol, high52, low52
            - dividendAmount, dividendYield, dividendDate
            - peRatio, pegRatio, pbRatio, prRatio, pcfRatio
            - grossMarginTTM, netProfitMarginTTM, operatingMarginTTM
            - returnOnEquity, returnOnAssets, returnOnInvestment
            - quickRatio, currentRatio, interestCoverage
            - totalDebtToCapital, ltDebtToEquity, totalDebtToEquity
            - epsTTM, epsChangePercentTTM, epsChangeYear
            - revChangeYear, revChangeTTM
            - sharesOutstanding, marketCapFloat, marketCap
            - bookValuePerShare, shortIntToFloat, shortIntDayToCover
            - beta, vol1DayAvg, vol10DayAvg, vol3MonthAvg
            - and more...
        """
        try:
            if not self.ensure_valid_session():
                raise Exception('No valid session available')
            
            symbol = self.clean_symbol(symbol)
            endpoint = f'https://api.schwabapi.com/marketdata/v1/instruments'
            params = {
                'symbol': symbol,
                'projection': projection
            }
            
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # The API returns a dictionary with the symbol as key
            if 'instruments' in data and len(data['instruments']) > 0:
                return data['instruments'][0]
            elif isinstance(data, dict) and symbol.upper() in data:
                return data[symbol.upper()]
            else:
                return data
            
        except Exception as e:
            logger.error(f'Could not get fundamental data for {symbol}: {e}')
            return None

# Async client for concurrent operations
class AsyncSchwabClient:
    """Async version of Schwab client for concurrent operations"""
    
    def __init__(self, base_client: SchwabClient):
        self.base_client = base_client
        
    async def get_multiple_options_chains(self, symbols: List[str], **kwargs) -> Dict[str, Dict]:
        """Get options chains for multiple symbols concurrently"""
        async def get_single_chain(symbol):
            try:
                # Use the sync client's method in a thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.base_client.get_options_chain(symbol, **kwargs)
                )
            except Exception as e:
                logger.error(f"Error getting options chain for {symbol}: {e}")
                return None
        
        tasks = [get_single_chain(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result for symbol, result in zip(symbols, results)
            if result is not None and not isinstance(result, Exception)
        }
        
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols concurrently"""
        try:
            # Use the sync client's method
            loop = asyncio.get_event_loop()
            quotes = await loop.run_in_executor(
                None, 
                lambda: self.base_client.get_quotes(symbols)
            )
            return quotes
        except Exception as e:
            logger.error(f"Error getting multiple quotes: {e}")
            return {}