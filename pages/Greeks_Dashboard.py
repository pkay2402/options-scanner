import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from src.api.schwab_client import SchwabClient
from datetime import datetime, timedelta

st.set_page_config(page_title="Greeks Dashboard", layout="wide")
st.title("Options Greeks Dashboard")

# --- Helper Functions ---
def extract_price(data, symbol):
    """Extract price from Schwab API quote response"""
    price_keys = ['lastPrice', 'mark', 'closePrice', 'bidPrice', 'askPrice']
    
    # Check if response has symbol key
    if symbol in data and isinstance(data[symbol], dict):
        quote_data = data[symbol].get('quote', data[symbol])
    else:
        quote_data = data.get('quote', data)
    
    # Try each price key
    for key in price_keys:
        if key in quote_data and quote_data[key] is not None:
            return float(quote_data[key])
    
    return None

def parse_options_chain(chain, symbol, current_price, expiry_range_weeks=12):
    """Parse Schwab options chain and extract relevant data"""
    strike_min = current_price * 0.8
    strike_max = current_price * 1.2
    expiry_today = datetime.now().date()
    expiry_limit = expiry_today + timedelta(weeks=expiry_range_weeks)
    
    options = []
    
    for option_type in ['callExpDateMap', 'putExpDateMap']:
        if option_type not in chain:
            continue
            
        is_call = (option_type == 'callExpDateMap')
        
        for exp_date_str, strikes_data in chain[option_type].items():
            # Parse expiry date (format: "2024-12-20:45")
            try:
                expiry_date = datetime.strptime(exp_date_str.split(':')[0], "%Y-%m-%d").date()
            except (ValueError, IndexError):
                continue
                
            # Filter by expiry range
            if not (expiry_today <= expiry_date <= expiry_limit):
                continue
            
            # Parse strikes
            for strike_str, contracts in strikes_data.items():
                if not contracts:
                    continue
                    
                try:
                    strike = float(strike_str)
                except (ValueError, TypeError):
                    continue
                
                # Filter by strike range
                if not (strike_min <= strike <= strike_max):
                    continue
                
                contract = contracts[0]
                
                # Only include options with valid Greek data
                if contract.get('delta') is None:
                    continue
                
                options.append({
                    'expiry': expiry_date,
                    'strike': strike,
                    'type': 'CALL' if is_call else 'PUT',
                    'delta': contract.get('delta', 0),
                    'gamma': contract.get('gamma', 0),
                    'theta': contract.get('theta', 0),
                    'vega': contract.get('vega', 0),
                    'bid': contract.get('bid', 0),
                    'ask': contract.get('ask', 0),
                    'last': contract.get('last', 0),
                    'openInterest': contract.get('openInterest', 0),
                    'volume': contract.get('totalVolume', 0),
                    'impliedVolatility': contract.get('volatility', 0),
                })
    
    return options

# --- User Input ---
symbol = st.text_input("Enter Stock Symbol", value="AAPL").upper()

# Initialize client
client = SchwabClient()

# --- Fetch Quote ---
with st.spinner("Fetching quote..."):
    try:
        quote = client.get_quote(symbol)
        if not quote:
            st.error("Could not fetch quote. Please check the symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching quote: {str(e)}")
        st.stop()

# Extract current price
current_price = extract_price(quote, symbol)

if current_price is None:
    st.error("Could not extract price from quote data.")
    with st.expander("Debug: Quote Response"):
        st.json(quote)
    st.stop()

# Display current price
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${current_price:.2f}")

# --- Fetch Option Chain ---
with st.spinner("Fetching options chain..."):
    try:
        chain = client.get_options_chain(
            symbol=symbol,
            contract_type='ALL',
            strike_count=50
        )
        
        if not chain or ('callExpDateMap' not in chain and 'putExpDateMap' not in chain):
            st.error("No options data available for this symbol.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error fetching option chain: {str(e)}")
        st.stop()

# --- Parse Options ---
options = parse_options_chain(chain, symbol, current_price)

if not options:
    st.warning("No options found in the selected range (±20% strikes, next 12 weeks).")
    with st.expander("Debug: Chain Response"):
        st.json(chain)
    st.stop()

df = pd.DataFrame(options)

# Display summary metrics
with col2:
    st.metric("Total Contracts", len(df))
with col3:
    st.metric("Expiration Dates", df['expiry'].nunique())

# --- Options Data Table ---
st.subheader("Options Data")
display_df = df.copy()
display_df['expiry'] = display_df['expiry'].astype(str)

# Add formatting
for col in ['delta', 'gamma', 'theta', 'vega', 'impliedVolatility']:
    if col in display_df.columns:
        display_df[col] = display_df[col].round(4)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# --- Aggregate Greeks by Expiry ---
st.subheader("Aggregate Greeks by Expiry")
greek_cols = ['delta', 'gamma', 'theta', 'vega']
aggs = df.groupby('expiry')[greek_cols].sum().round(2)
st.dataframe(aggs, use_container_width=True)

# --- Aggregate by Type ---
st.subheader("Aggregate Greeks by Type")
type_aggs = df.groupby('type')[greek_cols].sum().round(2)
st.dataframe(type_aggs, use_container_width=True)

# --- Trading Insights ---
st.markdown("---")
st.header("📊 Trading Insights from Greeks")

insights = []

# Pin risk: high gamma
if not df.empty and 'gamma' in df.columns:
    high_gamma = df.loc[df['gamma'].idxmax()]
    insights.append({
        "type": "🎯 Pin Risk",
        "message": f"Highest gamma at ${high_gamma['strike']:.2f} strike ({high_gamma['type']}) expiring {high_gamma['expiry']}. Expect potential pin risk and increased volatility near this level."
    })

# Directional bias: net delta
net_delta = df['delta'].sum()
if net_delta > 50:
    insights.append({
        "type": "📈 Bullish Bias",
        "message": f"Net delta: {net_delta:.2f}. Market makers are positioned with positive delta, suggesting bullish positioning."
    })
elif net_delta < -50:
    insights.append({
        "type": "📉 Bearish Bias",
        "message": f"Net delta: {net_delta:.2f}. Market makers are positioned with negative delta, suggesting bearish positioning."
    })
else:
    insights.append({
        "type": "➡️ Neutral",
        "message": f"Net delta: {net_delta:.2f}. Market shows no strong directional bias."
    })

# Premium decay: high theta
high_theta = df.loc[df['theta'].idxmin()]  # Most negative theta
insights.append({
    "type": "⏰ Theta Decay",
    "message": f"Highest theta decay at ${high_theta['strike']:.2f} strike ({high_theta['type']}) expiring {high_theta['expiry']}. Premium selling strategies may be attractive here."
})

# Volatility sensitivity: high vega
high_vega = df.loc[df['vega'].idxmax()]
insights.append({
    "type": "📊 Volatility Play",
    "message": f"Highest vega at ${high_vega['strike']:.2f} strike ({high_vega['type']}) expiring {high_vega['expiry']}. Most sensitive to IV changes (vega: {high_vega['vega']:.4f})."
})

# Display insights in columns
cols = st.columns(2)
for idx, insight in enumerate(insights):
    with cols[idx % 2]:
        st.info(f"**{insight['type']}**\n\n{insight['message']}")

# --- Download Data ---
st.markdown("---")
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Options Data as CSV",
    data=csv,
    file_name=f"{symbol}_greeks_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)