# Streaming Infrastructure

## Overview
Real-time WebSocket streaming for Schwab options data. Supports Level 1 quotes and Level 2 order book data.

## Components

### 1. `schwab_streamer.py`
WebSocket client for Schwab streaming API.

**Key Features:**
- Async WebSocket connection management
- Automatic reconnection handling
- Multiple service subscriptions (Level 1, Level 2, etc.)
- Callback-based event handling
- Thread-safe message queue

**Usage:**
```python
from src.streaming.schwab_streamer import SchwabStreamer

# Create streamer
streamer = SchwabStreamer(access_token, streamer_info)

# Start in background
streamer.start()

# Subscribe to options
streamer.subscribe_level_one_options(
    symbols=['SPY_241129C450'],
    callback=lambda data: print(data)
)

# Subscribe to order book
streamer.subscribe_options_book(
    symbols=['SPY_241129C450'],
    callback=lambda data: print(data)
)

# Stop when done
streamer.stop()
```

### 2. `models.py`
Data models for options flow and order book.

**Classes:**
- `OptionsFlow`: Individual options trade with analytics
- `OrderBookLevel`: Single price level in order book
- `OptionsBook`: Full order book with depth metrics

**Flow Analytics:**
- Automatic whale detection (>$100k)
- Moneyness calculation (ITM/ATM/OTM)
- Trade type classification (SWEEP/BLOCK/SPLIT)
- Days to expiry calculation
- Premium formatting

**Book Analytics:**
- Bid/ask imbalance ratio
- Total notional on each side
- Spread calculation ($ and %)
- Cumulative depth metrics

### 3. `integration.py`
Helper functions for Streamlit integration.

**Functions:**
- `get_schwab_client()`: Load credentials and create API client
- `get_streamer_info()`: Fetch streaming connection info
- `create_streamer_connection()`: One-step streamer setup

**Streamlit Pattern:**
```python
from src.streaming.integration import create_streamer_connection

if 'streamer' not in st.session_state:
    streamer, client = create_streamer_connection()
    if streamer:
        st.session_state.streamer = streamer
        streamer.start()
```

## Pages

### `6_Live_Flow_Feed.py`
Real-time options flow scanner with whale detection.

**Features:**
- Live streaming options trades
- Whale trade detection and alerts
- Multi-symbol watchlist
- Premium/type/moneyness filters
- Trade classification (sweep/block/split)
- 5-minute rolling statistics
- P/C ratio tracking
- Audio alerts (configurable)

**UI Components:**
- Real-time metrics dashboard
- Flow cards with badges
- Filter controls in sidebar
- Auto-refresh with 1s updates

### `7_Level2_Book.py`
Level 2 order book visualizer for options.

**Features:**
- Real-time order book depth
- Bid/ask imbalance indicator
- Cumulative depth charts
- Spread monitoring with alerts
- Top-of-book metrics
- Configurable depth levels (5-50)

**UI Components:**
- Split view (bid side | ask side)
- Combined depth chart
- Imbalance bar with gradient
- Spread indicator with alerts
- Notional value display

## Data Flow

```
Schwab API → WebSocket → SchwabStreamer → Callbacks → Streamlit UI
                ↓
         Message Queue → Polling → DataFrame → Charts
```

## Authentication

Streamer requires:
1. OAuth access token (from Schwab API)
2. Streamer info from user preferences:
   - `streamerSocketUrl`: WebSocket endpoint
   - `schwabClientAccountId`: Account ID
   - `schwabClientCorrelId`: Correlation ID
   - `schwabClientChannel`: Channel identifier
   - `schwabClientFunctionId`: Function identifier

## Services Available

### LEVELONE_OPTIONS
Real-time Level 1 options quotes:
- Bid/ask prices
- Volume
- Open interest
- Greeks (delta, gamma, theta, vega)
- Implied volatility
- Last trade price/size

### OPTIONS_BOOK
Level 2 order book:
- Bid levels (price, size, # orders)
- Ask levels (price, size, # orders)
- Up to 50 levels deep
- Real-time updates on changes

### LEVELONE_EQUITIES
Underlying stock quotes (for context)

### CHART_EQUITY
Minute-bar data for charting

## Error Handling

- Automatic reconnection on disconnect
- Callback error isolation (one bad callback won't break others)
- Graceful degradation if streaming unavailable
- Demo data generators for testing without live connection

## Performance

- Runs in background thread (non-blocking)
- Async I/O for WebSocket
- Message queue for decoupling
- Configurable update rates
- Efficient DataFrame operations

## Testing

Both pages include demo data generators:
- Click "Generate Test Flows" for flow feed
- Click "Generate Test Book" for order book
- Works without live connection
- Useful for UI development

## Next Steps

1. **Add Real Connection Logic**
   - Integrate with actual Schwab API client
   - Handle authentication flow
   - Test with live data

2. **Enhance Analytics**
   - Volume profile analysis
   - Price level clustering
   - Historical flow patterns
   - Unusual activity detection

3. **Audio Alerts**
   - Browser audio API integration
   - Customizable alert sounds
   - Volume control
   - Alert history

4. **Data Persistence**
   - Save flows to database
   - Historical playback
   - Export to CSV/JSON
   - Session recordings

5. **Advanced Filters**
   - IV percentile filters
   - Delta-based filtering
   - Multi-leg detection
   - Institution vs retail classification
