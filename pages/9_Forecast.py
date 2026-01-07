import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Price Forecast", page_icon="📈", layout="wide")

# Try to import optional libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="1y"):
    """Fetch historical stock data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data['Close'].rolling(window=window).mean()

def simple_forecast(df, forecast_days=30):
    """Simple trend-based forecast using moving averages and momentum."""
    df = df.copy()
    df['SMA_20'] = calculate_sma(df, 20)
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)
    
    recent_prices = df['Close'].tail(30).values
    if len(recent_prices) < 30:
        return None
    
    x = np.arange(len(recent_prices))
    coeffs = np.polyfit(x, recent_prices, 1)
    trend_slope = coeffs[0]
    
    returns = df['Close'].pct_change()
    volatility = returns.std()
    
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )
    
    forecast_values = []
    lower_bounds = []
    upper_bounds = []
    
    for i in range(forecast_days):
        damping_factor = 1 - (i / (forecast_days * 2))
        daily_change = trend_slope * damping_factor
        
        sma_50 = df['SMA_50'].iloc[-1]
        mean_reversion = (sma_50 - last_price) * 0.02 * (i / forecast_days)
        
        forecast_price = last_price + (daily_change * (i + 1)) + mean_reversion
        forecast_values.append(forecast_price)
        
        std_dev = last_price * volatility * np.sqrt(i + 1)
        lower_bounds.append(forecast_price - 1.96 * std_dev)
        upper_bounds.append(forecast_price + 1.96 * std_dev)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_values,
        'Lower_Bound': lower_bounds,
        'Upper_Bound': upper_bounds,
        'Model': 'Simple Trend'
    })
    forecast_df.set_index('Date', inplace=True)
    
    return forecast_df

def arima_forecast(df, forecast_days=30):
    """ARIMA time series forecast."""
    if not ARIMA_AVAILABLE:
        return None
    
    try:
        # Use only close prices
        prices = df['Close'].values
        
        # Fit ARIMA model (auto-selected order)
        model = ARIMA(prices, order=(5, 1, 0))s_dict, ticker):
    """Create an interactive chart with historical and multiple forecast models."""
    
    fig = go.Figure()
    
    # Historical candlesticks for last 60 days
    recent_hist = historical_df.tail(60)
    fig.add_trace(go.Candlestick(
        x=recent_hist.index,
        open=recent_hist['Open'],
        high=recent_hist['High'],
        low=recent_hist['Low'],
        close=recent_hist['Close'],
        name='Historical',
        increasing_line_color='#00C853',
        decreasing_line_color='#FF1744'
    ))
    
    # Historical line (full period)
    fig.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#4CAF50', width=1.5),
        opacity=0.7
    ))
    
    # Colors for different models
    colors = {
        'Simple Trend': '#FF9800',
        'ARIMA': '#2196F3',
        'Prophet': '#9C27B0',
        'LSTM': '#F44336',
        'Gann': '#00BCD4',
        'Ensemble': '#4CAF50'
    }
    
    # Plot each forecast
    for model_name, forecast_df in forecasts_dict.items():
        if forecast_df is not None:
            color = colors.get(model_name, '#666666')
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast'],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=2, dash='dash')
            ))
            
            # Confidence interval (only for selected model to avoid clutter)
            if model_name in forecasts_dict and len(forecasts_dict) == 1:
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['Upper_Bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['Lower_Bound'],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(width=0),
                    fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                    fill='tonexty'
                ))
    
    fig.update_layout(
        title=f'{ticker} Price Forecast - Multi-Model Comparison',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=650,
        plot_bgcolor='#FFE6F0',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        with st.spinner("Training LSTM model (this may take a minute)..."):
            model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
        # Generate forecast
        last_sequence = scaled_data[-sequence_length:]
        forecast_values = []
        
        for _ in range(forecast_days):
            pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
            forecast_values.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)
        
        # Inverse transform
        forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))
        
        # Calculate simple confidence intervals
        volatility = df['Close'].pct_change().std()
        last_price = df['Close'].iloc[-1]
        
        last_date = df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values.flatten(),
            'Lower_Bound': forecast_values.flatten() - (last_price * volatility * np.sqrt(np.arange(1, forecast_days+1))),
            'Upper_Bound': forecast_values.flatten() + (last_price * volatility * np.sqrt(np.arange(1, forecast_days+1))),
            'Model': 'LSTM'
        })
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df
    except Exception as e:
        st.warning(f"LSTM model failed: {e}")
        return None

def gann_forecast(df, forecast_days=30):
    """Gann analysis forecast using geometric angles and cycles."""
    try:
        last_price = df['Close'].iloc[-1]
        last_date = df.index[-1]
        
        # Find recent significant high and low
        lookback = min(len(df), 90)
        recent_high = df['High'].tail(lookback).max()
        recent_low = df['Low'].tail(lookback).min()
        
        # Gann angles (1x1 is 45 degrees, represents 1 unit price per 1 unit time)
        # Calculate daily price change for 1x1 angle
        price_range = recent_high - recent_low
        time_range = lookback
        gann_1x1_slope = price_range / time_range
        
        # Determine if we're in uptrend or downtrend
        sma_20 = df['Close'].tail(20).mean()
        trend_up = last_price > sma_20
        
        # Gann angles: 1x8, 1x4, 1x3, 1x2, 1x1, 2x1, 3x1, 4x1, 8x1
        angles = {
            '1x8': gann_1x1_slope / 8,
            '1x4': gann_1x1_slope / 4,
            '1x2': gann_1x1_slope / 2,
            '1x1': gann_1x1_slope,
            '2x1': gann_1x1_slope * 2,
            '4x1': gann_1x1_slope * 4,
        }
        
        # Use 1x1 angle for primary forecast
        primary_angle = angles['1x1'] if trend_up else -angles['1x1']
        
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast_values = []
        upper_bounds = []
        lower_bounds = []
        
        for i in range(forecast_days):
            # Primary forecast using 1x1 angle
            forecast_price = last_price + (primary_angle * (i + 1))
            forecast_values.append(forecast_price)
            
            # Upper bound using 2x1 angle
            if trend_up:
                upper_bound = last_price + (angles['2x1'] * (i + 1))
                lower_bound = last_price + (angles['1x2'] * (i + 1))
            else:
                upper_bound = last_price - (angles['1x2'] * (i + 1))
                lower_bound = last_price - (angles['2x1'] * (i + 1))
            
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds,
            'Model': 'Gann'
        })
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df
    except Exception as e:
        st.warning(f"Gann model failed: {e}")
        return None

def ensemble_forecast(forecasts):
    """Combine multiple forecasts using weighted average."""
    if not forecasts:
        return None
    
    # Weight models (you can adjust these)
    weights = {
        'Simple Trend': 0.2,
        'ARIMA': 0.25,
        'Prophet': 0.3,
        'LSTM': 0.15,
        'Gann': 0.1
    }
    
    # Combine forecasts
    ensemble_df = None
    total_weight = 0
    
    for forecast_df in forecasts:
        if forecast_df is not None:
            model_name = forecast_df['Model'].iloc[0]
            weight = weights.get(model_name, 0.2)
            
            if ensemble_df is None:
                ensemble_df = forecast_df[['Forecast', 'Lower_Bound', 'Upper_Bound']].copy() * weight
            else:
                ensemble_df['Forecast'] += forecast_df['Forecast'] * weight
                ensemble_df['Lower_Bound'] += forecast_df['Lower_Bound'] * weight
                ensemble_df['Upper_Bound'] += forecast_df['Upper_Bound'] * weight
            
            total_weight += weight
    
    if ensemble_df is not None and total_weight > 0:
        ensemble_df = ensemble_df / total_weight
        ensemble_df['Model'] = 'Ensemble'
        return ensemble_df
    
    return None

def create_forecast_chart(historical_df, forecast_df, ticker):
    """Create an interactive chart with historical and forecast data."""
    
    fig = go.Figure()
    
    # Historical candlesticks for last 60 days
    recent_hist = historical_df.tail(60)
    fig.add_trace(go.Candlestick(
        x=recent_hist.index,
        open=recent_hist['Open'],
        high=recent_hist['High'],
        low=recent_hist['Low'],
        close=recent_hist['Close'],
        name='Historical',
        increasing_line_color='#00C853',
        decreasing_line_color='#FF1744'
    ))
    
    # Historical line (full period)
    fig.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#4CAF50', width=1.5),
        opacity=0.7
    ))
    
    # Moving averages
    if 'SMA_20' in historical_df.columns:
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1, dash='dot'),
            opacity=0.5
        ))
    
    if 'SMA_50' in historical_df.columns:
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=1, dash='dot'),
            opacity=0.5
        ))
    
    # Forecast line
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#4CAF50', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Upper_Bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Lower_Bound'],
            mode='lines',
            name='Confidence Interval',
            line=dict(width=0),
            fillcolor='rgba(76, 175, 80, 0.2)',
            fill='tonexty'
        ))
    
    fig.update_layout(
        title=f'{ticker} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=600,
        plot_bgcolor='#FFE6F0',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig

def main():
    st.title("📈 Price Forecast")
    st.caption("AI-powered price forecasting using historical data and technical indicators")
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Enter a stock ticker (e.g., SPY, AAPL, TSLA)"
        ).upper()
        
        period = st.selectbox(
            "Historical Data Period",
            options=["6mo", "1y", "2y", "5y"],
            index=1,
            help="Amount of historical data to use for training"
        )
        
        forecast_days = st.slider(
            "Forecast Days Ahead",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This forecast uses:
        - Moving average trends
        - Momentum analysis
        - Volatility-based confidence intervals
        - Mean reversion factors
        """)Models")
        
        models_available = {
            'Simple Trend': True,
            'ARIMA': ARIMA_AVAILABLE,
            'Prophet': PROPHET_AVAILABLE,
            'LSTM': LSTM_AVAILABLE,
            'Gann': True,
            'Ensemble': True
        }
        
        selected_models = []
        for model_name, available in models_available.items():
            if available:
                if st.checkbox(model_name, value=(model_name in ['Simple Trend', 'ARIMA', 'Ensemble']), key=f"model_{model_name}"):
                    selected_models.append(model_name)
            else:
                st.checkbox(f"{model_name} (not installed)", value=False, disabled=True, key=f"model_{model_name}")
        
        st.markdown("---")
        st.markdown("### About Models")
        st.markdown("""
        **Simple Trend**: Moving averages + momentum
        
        **ARIMA**: Statistical time series (AutoRegressive Integrated Moving Average)
        
        **Prophet**: Facebook's forecasting tool with seasonality
        
        **LSTM**: Deep learning neural network
        
        **Gann**: Geometric angles and cycles
        
        **Ensemble**: Weighted combination of all model
        st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
    if not selected_models:
        st.warning("Please select at least one forecasting model from the sidebar.")
        return
    
    # Generate forecasts for selected models
    forecasts = {}
    
    with st.spinner("Generating forecasts..."):
        if 'Simple Trend' in selected_models:
            forecasts['Simple Trend'] = simple_forecast(df, forecast_days)
        
        if 'ARIMA' in selected_models and ARIMA_AVAILABLE:
            forecasts['ARIMA'] = arima_forecast(df, forecast_days)
        
        if 'Prophet' in selected_models and PROPHET_AVAILABLE:
            forecasts['Prophet'] = prophet_forecast(df, forecast_days)
        
        if 'LSTM' in selected_models and LSTM_AVAILABLE:
            forecasts['LSTM'] = lstm_forecast(df, forecast_days)
        
        if 'Gann' in selected_models:
            forecasts['Gann'] = gann_forecast(df, forecast_days)
        
        # Ensemble must run after other models
        if 'Ensemble' in selected_models:
            # Only use non-ensemble forecasts for ensemble
            base_forecasts = [v for k, v in forecasts.items() if k != 'Ensemble' and v is not None]
            if len(base_forecasts) >= 2:
                forecasts['Ensemble'] = ensemble_forecast(base_forecasts)
            else:
                st.info("Ensemble requires at least 2 other models to be selected.")
    
    # Filter out None forecasts
    forecasts = {k: v for k, v in forecasts.items() if v is not None}
    
    if not forecasts:
        st.error("Could not generate any forecasts. Please try different settings or models.")
        return
    
    # Display forecast metrics for each model
    st.markdown("---")
    st.subheader(f"📊 {forecast_days}-Day Forecast Targets")
    
    cols = st.columns(len(forecasts))
    for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
        with cols[idx]:
            forecast_end_price = forecast_df['Forecast'].iloc[-1]
            forecast_change = forecast_end_price - current_price
            forecast_change_pct = (forecast_change / current_price) * 100
            
            st.metric(
                f"{model_name}",
                f"${forecast_end_price:.2f}",
                f"{forecast_change:.2f} ({forecast_change_pct:+.2f}%)"
            )
    
    # Create and display chart
    fig = create_forecast_chart(df, forecasts, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast comparison table
    with st.expander("📊 Compare All Models", expanded=False):
        # Create comparison dataframe
        comparison_data = []
        for model_name, forecast_df in forecasts.items():
            end_price = forecast_df['Forecast'].iloc[-1]
            change = end_price - current_price
            change_pct = (change / current_price) * 100
            upper = forecast_df['Upper_Bound'].iloc[-1]
            lower = forecast_df['Lower_Bound'].iloc[-1]
            
            comparison_data.append({
                'Model': model_name,
                'Target Price': f"${end_price:.2f}",
                'Change': f"${change:.2f}",
                'Change %': f"{change_pct:+.2f}%",
                'Upper Range': f"${upper:.2f}",
                'Lower Range': f"${lower:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # Individual forecast tables
    with st.expander("📈 View Detailed Forecast Data", expanded=False):
        tab_names = list(forecasts.keys())
        tabs = st.tabs(tab_names)
        
        for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
            with tabs[idx]:
                forecast_display = forecast_df.copy()
                forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
                forecast_display = forecast_display.drop('Model', axis=1).round(2)
                st.dataframe(forecast_display, use_container_width=True)
                
                # Download button
                csv = forecast_display.to_csv()
                st.download_button(
                    label=f"📥 Download {model_name} CSV",
                    data=csv,
                    file_name=f"{ticker}_{model_name}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key=f"download_{model_name}"
            
    # Create and display chart
    fig = create_forecast_chart(df, forecast_df, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    with st.expander("📊 View Forecast Data", expanded=False):
        forecast_display = forecast_df.copy()
        forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
        forecast_display = forecast_display.round(2)
        st.dataframe(forecast_display, use_container_width=True)
        
        # Download button
        csv = forecast_display.to_csv()
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This forecast is for educational purposes only. Not financial advice. Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()
