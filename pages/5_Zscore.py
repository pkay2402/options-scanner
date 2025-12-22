#!/usr/bin/env python3
"""
Zscore page - simple page to compute and display price z-score
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(page_title="Zscore", layout="wide")

st.title("Zscore")
st.markdown("Enter a stock symbol and view its price z-score relative to a moving window.")

col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("Symbol", value="AAPL", max_chars=10)
with col2:
    lookback = st.slider("Lookback (days)", min_value=5, max_value=120, value=20)

# Auto-refresh control (enabled by default)
auto_refresh = st.checkbox("Auto-refresh every 3 minutes", value=True)
if auto_refresh:
    js = """
    <script>
    if (!window._zscore_autorefresh) {
      window._zscore_autorefresh = true;
      setInterval(function(){ window.location.reload(); }, 180000);
    }
    </script>
    """
    components.html(js, height=0)

# Note: inputs will trigger Streamlit reruns automatically; no explicit rerun call.

@st.cache_data(ttl=300)
def fetch_price_history(symbol: str, period="1mo"):
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")
        if hist.empty:
            return None
        df = hist.reset_index()[[hist.index.name or 'Date', "Open", "High", "Low", "Close", "Volume"]]
        # Normalize column name to `datetime`
        df.rename(columns={hist.index.name or 'Date': "datetime", "Close": "close"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        # Handle tz-aware vs tz-naive datetimes
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        else:
            df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
        return df
    except Exception as e:
        st.error(f"Error fetching price history: {e}")
        return None


def create_zscore_figure(df: pd.DataFrame, lookback: int):
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    df["ma"] = df["close"].rolling(window=lookback, min_periods=1).mean()
    df["std"] = df["close"].rolling(window=lookback, min_periods=1).std(ddof=0).replace(0, 1e-8)
    df["zscore"] = (df["close"] - df["ma"]) / df["std"]
    df["zclip"] = df["zscore"].clip(-10, 10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["close"], mode="lines", name="Close Price",
        line=dict(color="#7fdbca", width=2), hovertemplate="%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["zclip"], mode="lines+markers", name="Z-score",
        line=dict(color="#ff6b6b", width=2), yaxis="y2",
        hovertemplate="%{x|%b %d}<br>Z: %{y:.2f}<extra></extra>"
    ))

    # Horizontal lines for thresholds on secondary axis
    for level, dash, color in [(-3, 'solid', '#8b5cf6'), (-2, 'dash', '#fbbf24'), (2, 'dash', '#fbbf24'), (3, 'solid', '#8b5cf6')]:
        fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1.5, yref='y2')

    fig.update_layout(
        template='plotly_dark',
        height=600,
        margin=dict(t=30, r=60, l=60, b=40),
        xaxis=dict(type='date', tickformat='%b %d'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Z-score', overlaying='y', side='right', range=[-3.5, 3.5], showgrid=False, tickmode='array', tickvals=[-3, -2, -1, 0, 1, 2, 3]),
        hovermode='x unified'
    )

    return fig

# Main flow
if not symbol:
    st.info("Enter a symbol to begin")
else:
    with st.spinner("Fetching data..."):
        df = fetch_price_history(symbol)
    if df is None or df.empty:
        st.error("No price history available for that symbol.")
    else:
        fig = create_zscore_figure(df, lookback)
        st.plotly_chart(fig, use_container_width=True)
        # Show most recent values and computed z-score
        latest = df.iloc[-1]
        ma_latest = df['close'].rolling(window=lookback, min_periods=1).mean().iloc[-1]
        std_latest = df['close'].rolling(window=lookback, min_periods=1).std(ddof=0).replace(0, 1e-8).iloc[-1]
        z_latest = (latest['close'] - ma_latest) / std_latest
        st.write(f"Latest close: ${latest['close']:.2f} | Z-score: {z_latest:.2f}")
