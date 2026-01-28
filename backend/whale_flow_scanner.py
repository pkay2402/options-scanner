#!/usr/bin/env python3
"""
Whale Flow Scanner - Runs every 15 minutes on droplet
Detects significant options flows and sends alerts to Telegram

Key Detection Logic:
1. TIER 1 FLOWS: Premium > $1M (major institutional activity)
2. TIER 2 FLOWS: Premium > $500K (significant positioning)
3. SWEEPS: Tight spread (<5%) + high volume (>1000) = aggressive buying
4. UNUSUAL ACTIVITY: Vol/OI > 3x = fresh positioning
5. CONVICTION SIGNAL: Multiple flows same direction on same stock
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/whale_flow_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Telegram Config (set via environment variables)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Stock Watchlist - 50 high-activity stocks
WATCHLIST = [
    # Mega Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Semi & AI
    'AMD', 'AVGO', 'MU', 'INTC', 'QCOM', 'ARM', 'SMCI', 'MRVL',
    # Growth Tech
    'NFLX', 'CRM', 'ORCL', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'CRWD',
    # Crypto Adjacent
    'COIN', 'MSTR', 'IBIT', 'MARA', 'RIOT', 'IREN',
    # Finance
    'JPM', 'GS', 'MS', 'BAC', 'V', 'MA',
    # Healthcare
    'UNH', 'LLY', 'JNJ', 'PFE', 'ABBV',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'ARKK',
    # High Beta
    'BA', 'TXN', 'NBIS', 'LRCX', 'ZM', 'CRWV'
]

# Alert Thresholds
TIER1_THRESHOLD = 1_000_000      # $1M+ = Tier 1
TIER2_THRESHOLD = 500_000        # $500K+ = Tier 2
MIN_ALERT_PREMIUM = 250_000      # Minimum premium to consider
SWEEP_SPREAD_MAX = 5.0           # Max spread % for sweep detection
SWEEP_VOLUME_MIN = 1000          # Min volume for sweep
UNUSUAL_VOL_OI_RATIO = 3.0       # Vol/OI > 3x is unusual
STRIKE_RANGE_PCT = 0.15          # ±15% from current price

# State file to track sent alerts (avoid duplicates)
STATE_FILE = '/tmp/whale_flow_state.json'

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WhaleFlow:
    symbol: str
    strike: float
    option_type: str  # CALL or PUT
    expiry: str
    premium_total: float
    volume: int
    open_interest: int
    vol_oi_ratio: float
    mark_price: float
    spread_pct: float
    delta: float
    iv: float
    underlying_price: float
    distance_pct: float
    tier: str
    is_sweep: bool
    is_unusual: bool
    timestamp: str
    
    def to_alert_key(self) -> str:
        """Unique key for deduplication"""
        return f"{self.symbol}_{self.strike}_{self.option_type}_{self.expiry}_{self.tier}"

# ============================================================================
# SCHWAB CLIENT
# ============================================================================

def get_schwab_client():
    """Get Schwab API client"""
    try:
        from src.utils.cached_client import get_client
        return get_client()
    except Exception as e:
        logger.error(f"Failed to get Schwab client: {e}")
        return None

def get_next_two_fridays() -> List[str]:
    """Get next 2 Friday expiry dates"""
    today = datetime.now().date()
    weekday = today.weekday()
    days_to_friday = (4 - weekday) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    
    first_friday = today + timedelta(days=days_to_friday)
    second_friday = first_friday + timedelta(days=7)
    
    return [
        first_friday.strftime('%Y-%m-%d'),
        second_friday.strftime('%Y-%m-%d')
    ]

# ============================================================================
# FLOW DETECTION LOGIC
# ============================================================================

def scan_stock_for_flows(client, symbol: str, expiry_dates: List[str]) -> List[WhaleFlow]:
    """
    Scan a single stock for whale flows across given expiry dates
    
    Detection Logic:
    1. Filter strikes within ±15% of current price
    2. Calculate total premium (volume × mark × 100)
    3. Detect sweeps: tight spread + high volume
    4. Detect unusual: Vol/OI > 3x
    5. Classify tiers based on premium
    """
    flows = []
    
    try:
        # Get quote
        quote_response = client.get_quotes([symbol])
        if not quote_response or symbol not in quote_response:
            return flows
        
        quote = quote_response[symbol].get('quote', {})
        underlying_price = quote.get('lastPrice', 0)
        
        if underlying_price == 0:
            return flows
        
        # Scan each expiry date
        for expiry_date in expiry_dates:
            try:
                options_response = client.get_options_chain(
                    symbol=symbol,
                    contract_type='ALL',
                    from_date=expiry_date,
                    to_date=expiry_date
                )
                
                if not options_response:
                    continue
                
                # Process calls and puts
                for opt_type, exp_map_key in [('CALL', 'callExpDateMap'), ('PUT', 'putExpDateMap')]:
                    exp_map = options_response.get(exp_map_key, {})
                    
                    for exp_date_key, strikes_map in exp_map.items():
                        exp_date = exp_date_key.split(':')[0]
                        
                        for strike_str, contracts in strikes_map.items():
                            if not contracts:
                                continue
                            
                            contract = contracts[0]
                            strike = float(strike_str)
                            
                            # Filter by strike range (±15%)
                            distance_pct = ((strike - underlying_price) / underlying_price) * 100
                            if abs(distance_pct) > STRIKE_RANGE_PCT * 100:
                                continue
                            
                            # Extract contract data
                            volume = contract.get('totalVolume', 0)
                            oi = max(contract.get('openInterest', 0), 1)
                            mark = contract.get('mark', contract.get('last', 0))
                            bid = contract.get('bid', 0)
                            ask = contract.get('ask', 0)
                            delta = abs(contract.get('delta', 0))
                            iv = contract.get('volatility', 0)
                            
                            # Skip if no meaningful data
                            if volume == 0 or mark == 0:
                                continue
                            
                            # Calculate metrics
                            premium_total = volume * mark * 100
                            vol_oi_ratio = volume / oi
                            spread = ask - bid if bid > 0 else 0
                            spread_pct = (spread / mark * 100) if mark > 0 else 100
                            
                            # Skip if below minimum threshold
                            if premium_total < MIN_ALERT_PREMIUM:
                                continue
                            
                            # Tier classification
                            if premium_total >= TIER1_THRESHOLD:
                                tier = 'TIER 1'
                            elif premium_total >= TIER2_THRESHOLD:
                                tier = 'TIER 2'
                            else:
                                tier = 'TIER 3'
                            
                            # Sweep detection
                            is_sweep = (
                                spread_pct < SWEEP_SPREAD_MAX and 
                                volume > SWEEP_VOLUME_MIN and 
                                mark > 0.50
                            )
                            
                            # Unusual activity detection
                            is_unusual = vol_oi_ratio >= UNUSUAL_VOL_OI_RATIO
                            
                            # Only alert on significant flows
                            if tier in ['TIER 1', 'TIER 2'] or (is_sweep and is_unusual):
                                flow = WhaleFlow(
                                    symbol=symbol,
                                    strike=strike,
                                    option_type=opt_type,
                                    expiry=exp_date,
                                    premium_total=premium_total,
                                    volume=volume,
                                    open_interest=oi,
                                    vol_oi_ratio=vol_oi_ratio,
                                    mark_price=mark,
                                    spread_pct=spread_pct,
                                    delta=delta,
                                    iv=iv,
                                    underlying_price=underlying_price,
                                    distance_pct=distance_pct,
                                    tier=tier,
                                    is_sweep=is_sweep,
                                    is_unusual=is_unusual,
                                    timestamp=datetime.now().isoformat()
                                )
                                flows.append(flow)
                                
            except Exception as e:
                logger.debug(f"Error scanning {symbol} for {expiry_date}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
    
    return flows

def analyze_flows(flows: List[WhaleFlow]) -> Dict:
    """
    Analyze all flows to identify patterns and conviction signals
    
    Returns summary with:
    - Tier 1 flows (highest priority)
    - Conviction signals (multiple flows same direction)
    - Sweep activity
    - Overall market bias
    """
    if not flows:
        return None
    
    # Group by symbol
    by_symbol = defaultdict(list)
    for flow in flows:
        by_symbol[flow.symbol].append(flow)
    
    # Identify conviction signals (multiple flows same direction)
    conviction_signals = []
    for symbol, symbol_flows in by_symbol.items():
        calls = [f for f in symbol_flows if f.option_type == 'CALL']
        puts = [f for f in symbol_flows if f.option_type == 'PUT']
        
        call_premium = sum(f.premium_total for f in calls)
        put_premium = sum(f.premium_total for f in puts)
        
        # Strong conviction: 3+ flows same direction OR $2M+ same direction
        if len(calls) >= 3 or call_premium >= 2_000_000:
            conviction_signals.append({
                'symbol': symbol,
                'direction': 'BULLISH',
                'flow_count': len(calls),
                'total_premium': call_premium,
                'flows': calls
            })
        if len(puts) >= 3 or put_premium >= 2_000_000:
            conviction_signals.append({
                'symbol': symbol,
                'direction': 'BEARISH',
                'flow_count': len(puts),
                'total_premium': put_premium,
                'flows': puts
            })
    
    # Separate by tier
    tier1_flows = [f for f in flows if f.tier == 'TIER 1']
    tier2_flows = [f for f in flows if f.tier == 'TIER 2']
    sweep_flows = [f for f in flows if f.is_sweep]
    unusual_flows = [f for f in flows if f.is_unusual]
    
    # Overall market bias
    total_call_premium = sum(f.premium_total for f in flows if f.option_type == 'CALL')
    total_put_premium = sum(f.premium_total for f in flows if f.option_type == 'PUT')
    
    if total_call_premium > total_put_premium * 1.5:
        market_bias = 'BULLISH'
    elif total_put_premium > total_call_premium * 1.5:
        market_bias = 'BEARISH'
    else:
        market_bias = 'NEUTRAL'
    
    return {
        'tier1': tier1_flows,
        'tier2': tier2_flows,
        'sweeps': sweep_flows,
        'unusual': unusual_flows,
        'conviction': conviction_signals,
        'market_bias': market_bias,
        'total_call_premium': total_call_premium,
        'total_put_premium': total_put_premium,
        'total_flows': len(flows)
    }

# ============================================================================
# TELEGRAM MESSAGING
# ============================================================================

def send_telegram(message: str, parse_mode: str = 'Markdown') -> bool:
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram error: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def format_flow_alert(flow: WhaleFlow) -> str:
    """Format a single flow for Telegram"""
    direction_emoji = "🟢" if flow.option_type == 'CALL' else "🔴"
    sweep_flag = "🔥" if flow.is_sweep else ""
    unusual_flag = "⚡" if flow.is_unusual else ""
    
    premium_str = f"${flow.premium_total/1e6:.2f}M" if flow.premium_total >= 1e6 else f"${flow.premium_total/1e3:.0f}K"
    
    return (
        f"{direction_emoji} *{flow.symbol}* {flow.option_type} ${flow.strike:.1f} "
        f"({flow.expiry}) {sweep_flag}{unusual_flag}\n"
        f"   💰 {premium_str} | Vol: {flow.volume:,} | Vol/OI: {flow.vol_oi_ratio:.1f}x"
    )

def format_conviction_alert(signal: Dict) -> str:
    """Format conviction signal for Telegram"""
    emoji = "🚀" if signal['direction'] == 'BULLISH' else "📉"
    premium_str = f"${signal['total_premium']/1e6:.2f}M" if signal['total_premium'] >= 1e6 else f"${signal['total_premium']/1e3:.0f}K"
    
    return (
        f"{emoji} *{signal['symbol']}* - {signal['direction']} CONVICTION\n"
        f"   {signal['flow_count']} flows | {premium_str} total"
    )

def build_alert_message(analysis: Dict) -> str:
    """Build complete alert message for Telegram"""
    now = datetime.now().strftime('%I:%M %p ET')
    
    parts = [f"🐋 *WHALE FLOW ALERT* - {now}\n"]
    
    # Market Bias
    bias_emoji = "🟢" if analysis['market_bias'] == 'BULLISH' else "🔴" if analysis['market_bias'] == 'BEARISH' else "⚪"
    call_prem = f"${analysis['total_call_premium']/1e6:.1f}M"
    put_prem = f"${analysis['total_put_premium']/1e6:.1f}M"
    parts.append(f"{bias_emoji} Market Bias: *{analysis['market_bias']}*")
    parts.append(f"📊 Calls: {call_prem} | Puts: {put_prem}\n")
    
    # Tier 1 Flows (always show)
    if analysis['tier1']:
        parts.append("━━━━━━━━━━━━━━━━━━━━")
        parts.append("🏆 *TIER 1 FLOWS (>$1M)*")
        for flow in sorted(analysis['tier1'], key=lambda x: x.premium_total, reverse=True)[:5]:
            parts.append(format_flow_alert(flow))
        parts.append("")
    
    # Conviction Signals
    if analysis['conviction']:
        parts.append("━━━━━━━━━━━━━━━━━━━━")
        parts.append("🎯 *CONVICTION SIGNALS*")
        for signal in sorted(analysis['conviction'], key=lambda x: x['total_premium'], reverse=True)[:3]:
            parts.append(format_conviction_alert(signal))
        parts.append("")
    
    # Tier 2 Flows (top 5 only)
    if analysis['tier2']:
        parts.append("━━━━━━━━━━━━━━━━━━━━")
        parts.append("💎 *TIER 2 FLOWS (>$500K)*")
        for flow in sorted(analysis['tier2'], key=lambda x: x.premium_total, reverse=True)[:5]:
            parts.append(format_flow_alert(flow))
        parts.append("")
    
    # Sweep Activity Summary
    if analysis['sweeps']:
        sweep_count = len(analysis['sweeps'])
        sweep_calls = len([s for s in analysis['sweeps'] if s.option_type == 'CALL'])
        sweep_puts = sweep_count - sweep_calls
        parts.append(f"🔥 Sweeps: {sweep_count} detected ({sweep_calls}C/{sweep_puts}P)")
    
    # Footer
    parts.append(f"\n_Total flows scanned: {analysis['total_flows']}_")
    
    return "\n".join(parts)

# ============================================================================
# STATE MANAGEMENT (Deduplication)
# ============================================================================

def load_state() -> Dict:
    """Load previous alert state"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {'sent_alerts': [], 'last_run': None}

def save_state(state: Dict):
    """Save alert state"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def filter_new_flows(flows: List[WhaleFlow], state: Dict) -> List[WhaleFlow]:
    """Filter out flows that were already alerted"""
    sent_keys = set(state.get('sent_alerts', []))
    new_flows = []
    
    for flow in flows:
        key = flow.to_alert_key()
        if key not in sent_keys:
            new_flows.append(flow)
    
    return new_flows

def update_state_with_flows(state: Dict, flows: List[WhaleFlow]):
    """Update state with newly alerted flows"""
    sent_keys = set(state.get('sent_alerts', []))
    
    for flow in flows:
        sent_keys.add(flow.to_alert_key())
    
    # Keep only last 500 alerts to prevent unlimited growth
    state['sent_alerts'] = list(sent_keys)[-500:]
    state['last_run'] = datetime.now().isoformat()

# ============================================================================
# MAIN SCANNER
# ============================================================================

def run_scanner():
    """Main scanner function - runs every 15 minutes"""
    logger.info("=" * 60)
    logger.info("Starting Whale Flow Scanner")
    logger.info("=" * 60)
    
    # Check market hours (optional - can run 24/7 for futures)
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()
    
    # Skip weekends
    if weekday >= 5:
        logger.info("Weekend - skipping scan")
        return
    
    # Only scan during extended hours (4am - 8pm ET)
    if hour < 4 or hour >= 20:
        logger.info("Outside market hours - skipping scan")
        return
    
    # Get Schwab client
    client = get_schwab_client()
    if not client:
        logger.error("Failed to get Schwab client")
        return
    
    # Get expiry dates
    expiry_dates = get_next_two_fridays()
    logger.info(f"Scanning expiries: {expiry_dates}")
    
    # Load previous state
    state = load_state()
    
    # Scan all stocks
    all_flows = []
    
    for i, symbol in enumerate(WATCHLIST):
        logger.info(f"Scanning {symbol} ({i+1}/{len(WATCHLIST)})")
        flows = scan_stock_for_flows(client, symbol, expiry_dates)
        all_flows.extend(flows)
    
    logger.info(f"Total flows detected: {len(all_flows)}")
    
    if not all_flows:
        logger.info("No significant flows detected")
        return
    
    # Filter out already-sent alerts
    new_flows = filter_new_flows(all_flows, state)
    logger.info(f"New flows (not previously alerted): {len(new_flows)}")
    
    if not new_flows:
        logger.info("No new flows to alert")
        return
    
    # Analyze flows
    analysis = analyze_flows(new_flows)
    
    if not analysis:
        return
    
    # Only send if we have Tier 1/2 flows or conviction signals
    if not analysis['tier1'] and not analysis['tier2'] and not analysis['conviction']:
        logger.info("No significant flows to alert")
        return
    
    # Build and send alert
    message = build_alert_message(analysis)
    
    if send_telegram(message):
        # Update state with sent flows
        update_state_with_flows(state, new_flows)
        save_state(state)
        logger.info("Alert sent and state updated")
    else:
        logger.error("Failed to send alert")
    
    logger.info("Scanner complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_scanner()
