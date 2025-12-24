# scripts/rrg_yf.py
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Config: benchmark + 11 sector ETFs (common tickers)
BENCH = "SPY"
SECTORS = {
    "XLB":"Materials", "XLE":"Energy", "XLF":"Financials", "XLI":"Industrials",
    "XLK":"Technology", "XLP":"Consumer Staples", "XLU":"Utilities", "XLV":"Healthcare",
    "XLY":"Consumer Discretionary", "XLRE":"Real Estate", "XLC":"Communication Services"
}
TICKERS = [BENCH] + list(SECTORS.keys())

# --- Parameters (tune as needed)
# DAILY RRG params (in trading days)
daily_ratio_period = 252   # 1-year RS-Ratio
daily_mom_period   = 63    # ~quarter RS-Momentum
daily_tail_points  = 30    # show last 30 days of tail

# WEEKLY RRG params (in weeks)
weekly_ratio_period = 52
weekly_mom_period   = 13
weekly_tail_points  = 12

# Fetch data function
def fetch_close(tickers, period="2y", interval="1d"):
    df = yf.download(tickers, period=period, interval=interval, progress=False, threads=True)["Close"]
    # ensure DataFrame even for single ticker
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    df = df.dropna(how="all")
    return df

# Build RS-based x,y series
def compute_rrg_metrics(price_df, benchmark, ratio_period, mom_period):
    # price_df: DataFrame of closes with columns for tickers
    bench = price_df[benchmark].reindex(price_df.index)
    rs = price_df.div(bench, axis=0)   # RS series (symbol / benchmark)
    # RS-Ratio: pct change over ratio_period
    rs_ratio = rs.pct_change(periods=ratio_period) * 100.0
    # RS-Momentum: pct change over mom_period
    rs_mom = rs.pct_change(periods=mom_period) * 100.0
    return rs_ratio, rs_mom

# Build RRG plot
def plot_rrg(rs_ratio, rs_mom, tickers, tail_points=20, title="RRG"):
    # Use the last available date and tail_points lookback (if available)
    last_idx = rs_ratio.index.get_loc(rs_ratio.index[-1])
    # Prepare figure
    fig = go.Figure()
    annotations = []
    colors = {
        "Leading":"#16a34a", "Weakening":"#f59e0b", "Lagging":"#ef4444", "Improving":"#60a5fa"
    }
    for sym in tickers:
        # collect tail
        ratio_series = rs_ratio[sym].fillna(0)
        mom_series = rs_mom[sym].fillna(0)
        tail_start = max(0, len(ratio_series) - tail_points)
        x_tail = ratio_series.iloc[tail_start:].values
        y_tail = mom_series.iloc[tail_start:].values
        dates_tail = ratio_series.index[tail_start:]
        # latest point
        x0 = x_tail[-1]
        y0 = y_tail[-1]
        # quadrant
        if x0 > 0 and y0 > 0:
            quad = "Leading"
        elif x0 > 0 and y0 < 0:
            quad = "Weakening"
        elif x0 < 0 and y0 < 0:
            quad = "Lagging"
        else:
            quad = "Improving"
        color = colors[quad]
        # tail line
        fig.add_trace(go.Scatter(
            x=x_tail, y=y_tail, mode="lines+markers", name=sym,
            line=dict(color=color, width=1), marker=dict(size=6), hoverinfo="text",
            text=[f"{sym}<br>{d.date()}: x={xx:.2f}% y={yy:.2f}%" for d,xx,yy in zip(dates_tail, x_tail, y_tail)]
        ))
        # label last point
        annotations.append(dict(x=x0, y=y0, xanchor="left", yanchor="bottom",
                                text=f"<b>{sym}</b>", showarrow=False, font=dict(color=color)))
    # quadrant background lines
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_layout(
        title=title,
        xaxis_title="RS-Ratio (%)",
        yaxis_title="RS-Momentum (%)",
        template="plotly_dark",
        annotations=annotations,
        legend=dict(itemsizing='constant', orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig

# Build summary table for latest values
def build_summary_table(rs_ratio, rs_mom, tickers):
    latest_ratio = rs_ratio.iloc[-1].round(2)
    latest_mom = rs_mom.iloc[-1].round(2)
    df = pd.DataFrame({
        "symbol": tickers,
        "RS_Ratio(%)": latest_ratio[tickers].values,
        "RS_Mom(%)": latest_mom[tickers].values
    })
    def get_quadrant(r):
        if r["RS_Ratio(%)"] > 0 and r["RS_Mom(%)"] > 0:
            return "Leading"
        elif r["RS_Ratio(%)"] > 0 and r["RS_Mom(%)"] < 0:
            return "Weakening"
        elif r["RS_Ratio(%)"] < 0 and r["RS_Mom(%)"] < 0:
            return "Lagging"
        else:
            return "Improving"
    df["Quadrant"] = df.apply(get_quadrant, axis=1)
    df = df.sort_values(["Quadrant", "RS_Ratio(%)"], ascending=[True, False])
    return df

# --- MAIN: daily RRG
print("Fetching daily data...")
daily_prices = fetch_close(TICKERS, period="2y", interval="1d")
daily_ratio, daily_mom = compute_rrg_metrics(daily_prices, BENCH, daily_ratio_period, daily_mom_period)
daily_fig = plot_rrg(daily_ratio, daily_mom, list(SECTORS.keys()), tail_points=daily_tail_points, title="Daily RRG (vs SPY)")
daily_table = build_summary_table(daily_ratio, daily_mom, list(SECTORS.keys()))
daily_fig.show()
print("\nDaily summary:")
print(daily_table.to_string(index=False))

# --- WEEKLY RRG
print("\nBuilding weekly RRG (resample weekly closes)...")
weekly_prices = daily_prices.resample('W-FRI').last().dropna(how='all')
weekly_ratio, weekly_mom = compute_rrg_metrics(weekly_prices, BENCH, weekly_ratio_period, weekly_mom_period)
weekly_fig = plot_rrg(weekly_ratio, weekly_mom, list(SECTORS.keys()), tail_points=weekly_tail_points, title="Weekly RRG (vs SPY)")
weekly_table = build_summary_table(weekly_ratio, weekly_mom, list(SECTORS.keys()))
weekly_fig.show()
print("\nWeekly summary:")
print(weekly_table.to_string(index=False))