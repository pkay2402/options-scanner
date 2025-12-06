"""
Background Worker - Fetches market data and updates cache
Run this as a separate process to continuously update the database
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schwab_client import SchwabClient
from src.data.market_cache import MarketCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketDataWorker:
    """Background worker that fetches and caches market data"""
    
    def __init__(self):
        self.client = SchwabClient()
        self.cache = MarketCache()
        
        # Watchlist - Comprehensive Growth Tech + Value Stocks
        self.watchlist = [
            # === MAJOR INDICES & ETFs ===
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
            
            # === GROWTH TECH STOCKS ===
            # Mega Cap Tech (FAANG+)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
            
            # AI & Machine Learning
            'PLTR', 'AI', 'SOUN', 'BBAI', 'SMCI',
            
            # Cloud & SaaS
            'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'CFLT', 'S', 'MDB', 'ESTC',
            'NOW', 'CRM', 'ADBE', 'WDAY', 'VEEV', 'HUBS', 'ZM', 'DOCN', 'FROG',
            
            # Semiconductors & Hardware
            'AMD', 'AVGO', 'TSM', 'QCOM', 'MU', 'INTC', 'MRVL', 'ARM', 'AMAT', 
            'LRCX', 'ASML', 'KLAC', 'MPWR', 'ON',
            
            # Cybersecurity
            'PANW', 'FTNT', 'CRWD', 'ZS', 'S', 'CHKP', 'TENB',
            
            # Fintech & Digital Payments
            'SQ', 'COIN', 'PYPL', 'AFRM', 'SOFI', 'NU', 'HOOD', 'UPST',
            
            # E-commerce & Digital
            'SHOP', 'MELI', 'BABA', 'SE', 'CPNG', 'ABNB', 'DASH', 'UBER', 'LYFT',
            
            # Software & Dev Tools
            'U', 'GTLB', 'TEAM', 'ATLR', 'MNDY', 'BILL', 'PATH',
            
            # EVs & Battery Tech
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'PLUG', 'CHPT',
            
            # Other High Growth
            'RBLX', 'PINS', 'SNAP', 'TWLO', 'ROKU', 'SPOT', 'NFLX', 'DIS',
            
            # === VALUE STOCKS ===
            # Financials (Banks & Insurance)
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'USB', 'PNC',
            'BRK.B', 'AXP', 'MET', 'PRU', 'AFL',
            
            # Payment Networks
            'V', 'MA', 'AXP',
            
            # Healthcare & Pharma
            'UNH', 'JNJ', 'ABBV', 'LLY', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'HUM',
            
            # Consumer Staples & Retail
            'WMT', 'COST', 'TGT', 'HD', 'LOW', 'PG', 'KO', 'PEP', 'MCD', 'SBUX',
            'NKE', 'TJX', 'DG', 'DLTR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO',
            
            # Industrials & Manufacturing
            'BA', 'CAT', 'DE', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'EMR',
            
            # Telecom & Media
            'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR',
            
            # REITs & Real Estate
            'AMT', 'PLD', 'EQIX', 'PSA', 'SPG', 'O',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP'
        ]
        
        # Whale stocks - High volume subset only (40 stocks to limit API calls)
        # With 2 expiries = 80 API calls per scan cycle
        self.whale_stocks = [
            # Major Indices
            'SPY', 'QQQ', 'IWM', 'DIA',
            # Mega Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # High Volume Growth
            'PLTR', 'AMD', 'CRWD', 'SNOW', 'COIN', 'HOOD', 'SOFI',
            'SHOP', 'UBER', 'ABNB', 'DASH', 'RBLX',
            # Semiconductors
            'TSM', 'AVGO', 'ARM', 'MU', 'QCOM', 'SMCI',
            # Cloud/SaaS
            'NOW', 'CRM', 'NET', 'ZS', 'DDOG',
            # Financials
            'JPM', 'BAC', 'V', 'MA',
            # Other High Volume
            'NFLX', 'DIS', 'BABA', 'RIVN'
        ]
        
    def get_next_fridays(self, n=4):
        """Get next N Fridays"""
        fridays = []
        today = datetime.now().date()
        days_ahead = 4 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        
        for i in range(n):
            friday = today + timedelta(days=days_ahead + (i * 7))
            fridays.append(friday)
        
        return fridays
    
    def get_next_friday(self):
        """Get next Friday"""
        today = datetime.now().date()
        days_ahead = 4 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def fetch_watchlist_item(self, symbol):
        """Fetch single watchlist item"""
        try:
            next_friday = self.get_next_friday()
            exp_date_str = next_friday.strftime('%Y-%m-%d')
            
            # Get quote
            quote = self.client.get_quote(symbol)
            if not quote or symbol not in quote:
                return None
            
            price = quote[symbol]['quote']['lastPrice']
            prev_close = quote[symbol]['quote'].get('closePrice', price)
            volume = quote[symbol]['quote'].get('totalVolume', 0)
            
            daily_change = price - prev_close
            daily_change_pct = (daily_change / prev_close * 100) if prev_close else 0
            
            return {
                'symbol': symbol,
                'price': price,
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'volume': volume
            }
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def update_watchlist(self):
        """Update watchlist data in cache with rate limiting"""
        logger.info(f"Updating watchlist ({len(self.watchlist)} symbols)...")
        
        if not self.client.authenticate():
            logger.error("Failed to authenticate with Schwab API")
            return
        
        watchlist_data = []
        
        # Batch quotes in groups of 50 to avoid rate limit
        batch_size = 50
        for i in range(0, len(self.watchlist), batch_size):
            batch = self.watchlist[i:i+batch_size]
            logger.info(f"Processing watchlist batch {i//batch_size + 1}/{(len(self.watchlist) + batch_size - 1)//batch_size}")
            
            # Parallel fetch
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self.fetch_watchlist_item, symbol): symbol 
                          for symbol in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        watchlist_data.append(result)
            
            # Small delay between batches
            if i + batch_size < len(self.watchlist):
                time.sleep(1)
        
        # Update cache
        if watchlist_data:
            self.cache.upsert_watchlist(watchlist_data)
            self.cache.set_last_update_time('watchlist')
            logger.info(f"Updated {len(watchlist_data)} watchlist items")
        else:
            logger.warning("No watchlist data fetched")
    
    def scan_whale_flow(self, symbol, friday):
        """Scan single symbol/expiry for whale activity"""
        flows = []
        try:
            exp_date_str = friday.strftime('%Y-%m-%d')
            
            # Get quote
            quote = self.client.get_quote(symbol)
            if not quote or symbol not in quote:
                return flows
            
            price = quote[symbol]['quote']['lastPrice']
            underlying_volume = quote[symbol]['quote'].get('totalVolume', 0)
            
            if underlying_volume == 0:
                return flows
            
            # Get options chain
            options = self.client.get_options_chain(
                symbol=symbol,
                contract_type='ALL',
                from_date=exp_date_str,
                to_date=exp_date_str
            )
            
            if not options or 'callExpDateMap' not in options:
                return flows
            
            # Process calls
            if 'callExpDateMap' in options:
                for exp_date, strikes in options['callExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        strike = float(strike_str)
                        
                        # Filter ATM ±5%
                        if abs(strike - price) / price > 0.05:
                            continue
                        
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                            mark = contract.get('mark', 0)
                            delta = contract.get('delta', 0)
                            iv = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                            
                            if volume == 0 or mark == 0 or delta == 0:
                                continue
                            
                            # Calculate whale score
                            leverage = delta * price
                            leverage_ratio = leverage / mark
                            valr = leverage_ratio * iv
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark * 100
                            dvolume_und = price * underlying_volume
                            dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                            whale_score = round(valr * vol_oi * dvolume_ratio * 1000, 0)
                            
                            if vol_oi < 1.5 or whale_score < 100:
                                continue
                            
                            dte = (friday - datetime.now().date()).days
                            
                            flows.append({
                                'symbol': symbol,
                                'type': 'CALL',
                                'strike': strike,
                                'whale_score': whale_score,
                                'volume': volume,
                                'open_interest': oi,
                                'vol_oi': vol_oi,
                                'premium': mark,
                                'delta': delta,
                                'expiry': friday,
                                'dte': dte,
                                'timestamp': datetime.now()
                            })
            
            # Process puts
            if 'putExpDateMap' in options:
                for exp_date, strikes in options['putExpDateMap'].items():
                    for strike_str, contracts in strikes.items():
                        strike = float(strike_str)
                        
                        # Filter ATM ±5%
                        if abs(strike - price) / price > 0.05:
                            continue
                        
                        for contract in contracts:
                            volume = contract.get('totalVolume', 0)
                            oi = contract.get('openInterest', 1) if contract.get('openInterest', 0) > 0 else 1
                            mark = contract.get('mark', 0)
                            delta = contract.get('delta', 0)
                            iv = contract.get('volatility', 0) / 100 if contract.get('volatility', 0) else 0.01
                            
                            if volume == 0 or mark == 0 or delta == 0:
                                continue
                            
                            # Calculate whale score
                            leverage = abs(delta) * price
                            leverage_ratio = leverage / mark
                            valr = leverage_ratio * iv
                            vol_oi = volume / oi
                            dvolume_opt = volume * mark * 100
                            dvolume_und = price * underlying_volume
                            dvolume_ratio = dvolume_opt / dvolume_und if dvolume_und > 0 else 0
                            whale_score = round(valr * vol_oi * dvolume_ratio * 1000, 0)
                            
                            if vol_oi < 1.5 or whale_score < 100:
                                continue
                            
                            dte = (friday - datetime.now().date()).days
                            
                            flows.append({
                                'symbol': symbol,
                                'type': 'PUT',
                                'strike': strike,
                                'whale_score': whale_score,
                                'volume': volume,
                                'open_interest': oi,
                                'vol_oi': vol_oi,
                                'premium': mark,
                                'delta': delta,
                                'expiry': friday,
                                'dte': dte,
                                'timestamp': datetime.now()
                            })
        
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
        
        return flows
    
    def update_whale_flows(self):
        """Update whale flows data in cache with rate limiting"""
        logger.info(f"Scanning whale flows ({len(self.whale_stocks)} symbols x 2 expiries)...")
        
        if not self.client.authenticate():
            logger.error("Failed to authenticate with Schwab API")
            return
        
        # Only scan next 2 expiries to reduce API calls
        fridays = self.get_next_fridays(2)
        all_flows = []
        
        # Calculate total API calls: len(whale_stocks) * 2 expiries = ~80 calls
        # Rate limit: 120/min, so we're safe with batch processing
        logger.info(f"Total API calls: {len(self.whale_stocks) * len(fridays)} (under 120/min limit)")
        
        # Batch process in groups of 40 to stay under rate limit
        batch_size = 40
        tasks = [(symbol, friday) for symbol in self.whale_stocks for friday in fridays]
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(self.scan_whale_flow, symbol, friday): (symbol, friday)
                          for symbol, friday in batch}
                
                for future in as_completed(futures):
                    flows = future.result()
                    all_flows.extend(flows)
            
            # Small delay between batches to avoid rate limit
            if i + batch_size < len(tasks):
                time.sleep(2)
        
        # Insert flows
        if all_flows:
            self.cache.insert_whale_flows(all_flows)
            self.cache.set_last_update_time('whale_flows')
            logger.info(f"Inserted {len(all_flows)} whale flows")
        else:
            logger.warning("No whale flows detected")
    
    def cleanup(self):
        """Cleanup old data"""
        logger.info("Cleaning up old whale flows...")
        self.cache.cleanup_old_whale_flows(days_to_keep=1)
    
    def run_cycle(self):
        """Run one complete update cycle"""
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("Starting market data update cycle")
            
            # Update watchlist
            self.update_watchlist()
            
            # Wait a bit to avoid rate limits
            time.sleep(10)
            
            # Update whale flows
            self.update_whale_flows()
            
            # Cleanup old data
            self.cleanup()
            
            # Show stats
            stats = self.cache.get_cache_stats()
            logger.info(f"Cache stats: {stats}")
            
            elapsed = time.time() - start_time
            logger.info(f"Update cycle completed in {elapsed:.1f}s")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in update cycle: {e}", exc_info=True)
    
    def run_forever(self, interval_minutes=5):
        """Run worker continuously"""
        logger.info(f"Market data worker started (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                self.run_cycle()
                
                # Sleep until next cycle
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                logger.info("Waiting 60 seconds before retry...")
                time.sleep(60)

if __name__ == '__main__':
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Start worker
    worker = MarketDataWorker()
    
    # Run once immediately, then every 5 minutes
    worker.run_cycle()
    worker.run_forever(interval_minutes=5)
