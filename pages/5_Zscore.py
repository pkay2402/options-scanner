#!/usr/bin/env python3
"""
Zscore page - simple page to compute and display price z-score
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import concurrent.futures
import streamlit.components.v1 as components
from datetime import timedelta

try:
    from src.api.schwab_client import SchwabClient
except Exception:
    SchwabClient = None


@st.cache_data(ttl=900)
def compute_gex_timeseries(symbol: str, dates: list, max_workers: int = 8):
    """Fetch options chains for each date in `dates` in parallel and compute net GEX per date.

    Returns (dates_out, gex_list, price_list) where dates_out are parsed datetimes for successful rows.
    """
    try:
        client = SchwabClient()
        if not client.authenticate():
            return None

        gex_results = [None] * len(dates)
        price_results = [None] * len(dates)

        def fetch_for_date(i, d):
            try:
                # First try with explicit from/to date
                chain = client.get_options_chain(symbol=symbol, from_date=d, to_date=d, contract_type='ALL')
                if chain:
                    return i, chain
                # Fallback: some symbols/dates return 400 for historical chains — try without date filters
                chain = client.get_options_chain(symbol=symbol, contract_type='ALL')
                return i, chain
            except Exception as e:
                # If Schwab returned a 400 for date-scoped request, try without dates as fallback
                try:
                    chain = client.get_options_chain(symbol=symbol, contract_type='ALL')
                    return i, chain
                except Exception:
                    return i, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch_for_date, i, d) for i, d in enumerate(dates)]
            for fut in concurrent.futures.as_completed(futures):
                i, chain = fut.result()
                if not chain:
                    gex_results[i] = None
                    price_results[i] = None
                    continue

                spot = chain.get('underlyingPrice') if isinstance(chain, dict) else None
                total_call_gex = 0.0
                total_put_gex = 0.0

                for option_type in ['callExpDateMap', 'putExpDateMap']:
                    if option_type not in chain:
                        continue
                    for exp_date, strikes in chain[option_type].items():
                        for strike_str, contracts in strikes.items():
                            if not contracts:
                                continue
                            contract = contracts[0]
                            try:
                                gamma = float(contract.get('gamma', 0) or 0)
                                oi = float(contract.get('openInterest', 0) or 0)
                            except Exception:
                                continue
                            if oi <= 0 or gamma == 0 or not spot:
                                continue
                            gex = gamma * oi * 100 * spot * spot * 0.01
                            if 'call' in option_type:
                                total_call_gex += gex
                            else:
                                total_put_gex += gex

                net_gex = total_call_gex - total_put_gex
                gex_results[i] = net_gex
                price_results[i] = spot

        # Build output aligned to input dates
        parsed_dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        return parsed_dates, gex_results, price_results

    except Exception as e:
        return None

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

# Choose metric source
metric = st.radio("Metric", options=["Price z-score", "Options GEX z-score"], index=0)


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

    # Annotations: crossings of ±2 and ±3, and latest direction arrow
    df['z_prev'] = df['zscore'].shift(1)
    # crossings above +2 and +3
    cross_p2 = df[(df['z_prev'] <= 2) & (df['zscore'] > 2)]
    cross_p3 = df[(df['z_prev'] <= 3) & (df['zscore'] > 3)]
    # crossings below -2 and -3
    cross_m2 = df[(df['z_prev'] >= -2) & (df['zscore'] < -2)]
    cross_m3 = df[(df['z_prev'] >= -3) & (df['zscore'] < -3)]

    def add_cross_annotations(df_cross, text, arrow_color):
        for idx, row in df_cross.iterrows():
            fig.add_annotation(x=row['datetime'], y=row['zclip'], text=text,
                               showarrow=True, arrowhead=3, ax=0, ay=-30,
                               font=dict(color=arrow_color), arrowcolor=arrow_color, yshift=0)

    add_cross_annotations(cross_p2, '+2σ', '#f59e0b')
    add_cross_annotations(cross_p3, '+3σ', '#8b5cf6')
    add_cross_annotations(cross_m2, '-2σ', '#f59e0b')
    add_cross_annotations(cross_m3, '-3σ', '#8b5cf6')

    # Latest z-score arrow label
    if not df.empty:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last
        direction = 'up' if last['zscore'] > prev['zscore'] else ('down' if last['zscore'] < prev['zscore'] else 'flat')
        arrow_color = '#16a34a' if direction == 'up' else ('#ef4444' if direction == 'down' else '#9ca3af')
        fig.add_annotation(x=last['datetime'], y=last['zclip'], text=f"Latest {last['zscore']:.2f}",
                           showarrow=True, arrowhead=4, ax=40 if direction == 'up' else -40, ay=-40 if direction == 'up' else 40,
                           font=dict(color=arrow_color), arrowcolor=arrow_color)

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
        if metric == "Price z-score":
            fig = create_zscore_figure(df, lookback)
            st.plotly_chart(fig, use_container_width=True)
            # Show most recent values and computed z-score
            latest = df.iloc[-1]
            ma_latest = df['close'].rolling(window=lookback, min_periods=1).mean().iloc[-1]
            std_latest = df['close'].rolling(window=lookback, min_periods=1).std(ddof=0).replace(0, 1e-8).iloc[-1]
            z_latest = (latest['close'] - ma_latest) / std_latest
            st.write(f"Latest close: ${latest['close']:.2f} | Z-score: {z_latest:.2f}")
            # Build alerts table for recent crossings (±2, ±3)
            # Ensure zscore and z_prev exist on this df
            df = df.sort_values('datetime').reset_index(drop=True)
            df['ma'] = df['close'].rolling(window=lookback, min_periods=1).mean()
            df['std'] = df['close'].rolling(window=lookback, min_periods=1).std(ddof=0).replace(0, 1e-8)
            df['zscore'] = (df['close'] - df['ma']) / df['std']
            df['z_prev'] = df['zscore'].shift(1)

            crosses = []
            for lvl, label in [(3, '+3σ'), (2, '+2σ'), (-2, '-2σ'), (-3, '-3σ')]:
                if lvl > 0:
                    rows = df[(df['z_prev'] <= lvl) & (df['zscore'] > lvl)]
                else:
                    rows = df[(df['z_prev'] >= lvl) & (df['zscore'] < lvl)]
                for _, r in rows.iterrows():
                    action = 'Consider trim/lock profits' if abs(lvl) >= 3 else ('Take profits if reversal; confirm with volume' if lvl>0 else 'Buy on confirmation')
                    crosses.append({'date': r['datetime'].date(), 'level': label, 'z': round(r['zscore'], 2), 'price': round(r['close'], 2), 'action': action})

            if crosses:
                alerts_df = pd.DataFrame(crosses).sort_values('date', ascending=False)
                st.subheader('Recent z-score crossings (alerts)')
                st.dataframe(alerts_df)
                csv = alerts_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download alerts CSV', csv, file_name=f'{symbol}_zscore_alerts.csv')

        else:
            # Options-based GEX z-score
            if SchwabClient is None:
                st.error("Schwab client not available in this environment. Cannot fetch options data.")
            else:
                client = SchwabClient()
                if not client.authenticate():
                    st.error("Schwab authentication failed. Ensure credentials/tokens are configured.")
                else:
                    # Build GEX time series by fetching option chains for past `lookback` days
                    days = min(lookback, 60)
                    dates = [ (datetime.now().date() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days-1, -1, -1) ]
                    gex_series = []
                    price_series = []
                    with st.spinner("Fetching options chains (may take a while)..."):
                        res = compute_gex_timeseries(symbol, dates)
                        if res is None:
                            st.error("Failed to compute GEX timeseries (Schwab auth may have failed)")
                            st.stop()
                        parsed_dates, gex_series, price_series = res
                    # Build DataFrame
                    gdf = pd.DataFrame({ 'date': pd.to_datetime(parsed_dates), 'gex': gex_series, 'price': price_series })
                    gdf = gdf.dropna().reset_index(drop=True)
                    if gdf.empty:
                        st.error("No options GEX data available for the requested dates/symbol.")
                    else:
                        gdf['ma'] = gdf['gex'].rolling(window=lookback, min_periods=1).mean()
                        gdf['std'] = gdf['gex'].rolling(window=lookback, min_periods=1).std(ddof=0).replace(0,1e-8)
                        gdf['zscore'] = (gdf['gex'] - gdf['ma']) / gdf['std']
                        gdf['zclip'] = gdf['zscore'].clip(-10,10)

                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=gdf['date'], y=gdf['price'], mode='lines', name='Close Price', line=dict(color='#7fdbca', width=2)))
                        fig2.add_trace(go.Scatter(x=gdf['date'], y=gdf['zclip'], mode='lines+markers', name='GEX z-score', line=dict(color='#ff6b6b', width=2), yaxis='y2'))
                        for level, dash, color in [(-3, 'solid', '#8b5cf6'), (-2, 'dash', '#fbbf24'), (2, 'dash', '#fbbf24'), (3, 'solid', '#8b5cf6')]:
                            fig2.add_hline(y=level, line_dash=dash, line_color=color, line_width=1.5, yref='y2')
                        fig2.update_layout(template='plotly_dark', height=600, margin=dict(t=30,r=60,l=60,b=40), xaxis=dict(type='date', tickformat='%b %d'), yaxis=dict(title='Price'), yaxis2=dict(title='GEX z-score', overlaying='y', side='right', range=[-3.5,3.5], showgrid=False, tickmode='array', tickvals=[-3,-2,-1,0,1,2,3]), hovermode='x unified')
                        st.plotly_chart(fig2, use_container_width=True)
                        # Alerts table for GEX z-score crossings
                        crosses = []
                        for lvl, label in [(3, '+3σ'), (2, '+2σ'), (-2, '-2σ'), (-3, '-3σ')]:
                            if lvl > 0:
                                rows = gdf[(gdf['zscore'].shift(1) <= lvl) & (gdf['zscore'] > lvl)]
                            else:
                                rows = gdf[(gdf['zscore'].shift(1) >= lvl) & (gdf['zscore'] < lvl)]
                            for _, r in rows.iterrows():
                                action = 'Consider trim/lock profits' if abs(lvl) >= 3 else ('Take profits if reversal; confirm with volume' if lvl>0 else 'Buy on confirmation')
                                crosses.append({'date': r['date'].date(), 'level': label, 'z': round(r['zscore'],2), 'price': round(r['price'],2), 'action': action})

                        if crosses:
                            alerts_df = pd.DataFrame(crosses).sort_values('date', ascending=False)
                            st.subheader('Recent GEX z-score crossings (alerts)')
                            st.dataframe(alerts_df)
                            csv = alerts_df.to_csv(index=False).encode('utf-8')
                            st.download_button('Download GEX alerts CSV', csv, file_name=f'{symbol}_gex_alerts.csv')
