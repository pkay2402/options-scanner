import sys
sys.path.insert(0, '/root/options-scanner')
from src.data.market_cache import MarketCache
from datetime import datetime

# Test the exact bot logic
cache = MarketCache()
db_flows = cache.get_whale_flows(sort_by='time', limit=50, hours_lookback=0.25)

whale_score_threshold = 50
sent_whale_alerts = {}  # Fresh start
whale_alerts = []

print(f"Checking {len(db_flows)} flows from database...")

for flow in db_flows:
    whale_score = flow.get('whale_score', 0)
    
    if whale_score < whale_score_threshold:
        continue
    
    alert_key = f"{flow.get('symbol')}_{flow.get('strike')}_{flow.get('type')}_{flow.get('expiry')}"
    
    now = datetime.now()
    if alert_key in sent_whale_alerts:
        last_sent = sent_whale_alerts[alert_key]
        if (now - last_sent).total_seconds() < 3600:
            continue
    
    underlying_price = flow.get('underlying_price', flow.get('strike', 0))
    
    whale_alerts.append({
        'symbol': flow.get('symbol'),
        'strike': flow.get('strike'),
        'type': flow.get('type', 'CALL'),
        'whale_score': whale_score
    })
    sent_whale_alerts[alert_key] = now

print(f"\n✅ Would send {len(whale_alerts)} alerts:")
for alert in whale_alerts[:10]:
    print(f"  {alert['symbol']} ${alert['strike']} {alert['type']} (score: {int(alert['whale_score'])})")
