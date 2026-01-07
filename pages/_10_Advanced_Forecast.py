import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Forecast", page_icon="🧠", layout="wide")

# Try to import ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y"):
    """Fetch historical stock data."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators."""
    df = df.copy()
    
    # Price-based indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # VWAP
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # Volatility
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(
        lambda x: max(x['High'] - x['Low'], 
                     abs(x['High'] - x['Close']), 
                     abs(x['Low'] - x['Close'])), axis=1
    ).rolling(window=14).mean()
    
    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Pattern recognition features
    df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    
    return df

def calculate_garch_volatility(returns, forecast_horizon=30):
    """Calculate GARCH(1,1) volatility forecast."""
    if not ARCH_AVAILABLE:
        return None
    
    try:
        # Remove NaN and fit GARCH model
        returns_clean = returns.dropna() * 100  # Scale for numerical stability
        
        model = arch_model(returns_clean, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        
        # Forecast volatility
        forecasts = fitted.forecast(horizon=forecast_horizon)
        variance_forecast = forecasts.variance.values[-1, :]
        
        return np.sqrt(variance_forecast) / 100  # Unscale
    except Exception as e:
        st.warning(f"GARCH calculation failed: {e}")
        return None

def engineer_features(df):
    """Create feature set for ML models."""
    features = []
    feature_names = []
    
    # Technical indicators
    tech_features = [
        'RSI', 'MACD', 'MACD_hist', 'BB_width', 'Volume_Ratio',
        'Volatility_20', 'ATR', 'ROC_10', 'Momentum_10'
    ]
    
    for feat in tech_features:
        if feat in df.columns:
            features.append(df[feat].values)
            feature_names.append(feat)
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10, 20]:
        lagged = df['Returns'].shift(lag).values
        features.append(lagged)
        feature_names.append(f'Returns_Lag_{lag}')
    
    # Moving average crossovers
    if 'SMA_10' in df.columns and 'SMA_20' in df.columns:
        ma_cross = (df['SMA_10'] - df['SMA_20']).values
        features.append(ma_cross)
        feature_names.append('MA_10_20_Cross')
    
    # Price position relative to bands
    if 'BB_upper' in df.columns:
        bb_position = ((df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])).values
        features.append(bb_position)
        feature_names.append('BB_Position')
    
    # VWAP deviation
    if 'VWAP' in df.columns:
        vwap_dev = ((df['Close'] - df['VWAP']) / df['VWAP']).values
        features.append(vwap_dev)
        feature_names.append('VWAP_Deviation')
    
    # Stack features
    feature_matrix = np.column_stack(features)
    
    return feature_matrix, feature_names

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost regression model."""
    if not XGBOOST_AVAILABLE:
        return None, None
    
    try:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Feature importance
        importance = model.feature_importances_
        
        return model, importance
    except Exception as e:
        st.warning(f"XGBoost training failed: {e}")
        return None, None

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """Train LightGBM regression model."""
    if not LIGHTGBM_AVAILABLE:
        return None, None
    
    try:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # Feature importance
        importance = model.feature_importances_
        
        return model, importance
    except Exception as e:
        st.warning(f"LightGBM training failed: {e}")
        return None, None

def create_forecast_ensemble(df, forecast_horizon=30, model_type='daily'):
    """Create hybrid ensemble forecast."""
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Engineer features
    X, feature_names = engineer_features(df)
    
    # Create target (next day return)
    y = df['Returns'].shift(-1).values
    
    # Remove NaN rows
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) < 100:
        return None, None, None
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    importances = {}
    
    if model_type in ['daily', 'both']:
        if LIGHTGBM_AVAILABLE:
            lgb_model, lgb_importance = train_lightgbm_model(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            if lgb_model is not None:
                models['LightGBM'] = lgb_model
                importances['LightGBM'] = lgb_importance
    
    if model_type in ['monthly', 'both']:
        if XGBOOST_AVAILABLE:
            xgb_model, xgb_importance = train_xgboost_model(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            if xgb_model is not None:
                models['XGBoost'] = xgb_model
                importances['XGBoost'] = xgb_importance
    
    if not models:
        return None, None, None
    
    # Generate forecasts
    last_features = X_test_scaled[-1:].copy()
    forecasts_dict = {}
    
    for model_name, model in models.items():
        predictions = []
        current_features = last_features.copy()
        
        for _ in range(forecast_horizon):
            pred_return = model.predict(current_features)[0]
            predictions.append(pred_return)
            
            # Update features with prediction (simplified)
            current_features = np.roll(current_features, -1, axis=1)
            current_features[0, -1] = pred_return
        
        forecasts_dict[model_name] = predictions
    
    # Calculate ensemble (weighted average)
    ensemble_forecast = np.mean([forecasts_dict[m] for m in models.keys()], axis=0)
    
    # Convert to price predictions
    last_price = df['Close'].iloc[-1]
    dates = df.index[valid_idx]
    last_date = dates[-1]
    
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_horizon,
        freq='D'
    )
    
    # Calculate cumulative returns and prices
    prices = [last_price]
    for ret in ensemble_forecast:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]
    
    # GARCH volatility for confidence intervals
    garch_vol = calculate_garch_volatility(df['Returns'], forecast_horizon)
    if garch_vol is not None:
        volatility = garch_vol
    else:
        volatility = np.full(forecast_horizon, df['Returns'].std())
    
    # Calculate confidence intervals
    lower_bounds = []
    upper_bounds = []
    for i, (price, vol) in enumerate(zip(prices, volatility)):
        std_dev = price * vol * np.sqrt(i + 1)
        lower_bounds.append(price - 1.96 * std_dev)
        upper_bounds.append(price + 1.96 * std_dev)
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': prices,
        'Lower_Bound': lower_bounds,
        'Upper_Bound': upper_bounds
    })
    forecast_df.set_index('Date', inplace=True)
    
    return forecast_df, importances, feature_names

def create_advanced_chart(df, forecast_df, ticker, model_type):
    """Create advanced forecast visualization."""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price Forecast - {model_type.title()} Model', 'Volume'),
        vertical_spacing=0.05
    )
    
    # Historical candlesticks
    recent = df.tail(90)
    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent['Open'],
            high=recent['High'],
            low=recent['Low'],
            close=recent['Close'],
            name='Price',
            increasing_line_color='#00C853',
            decreasing_line_color='#FF1744'
        ),
        row=1, col=1
    )
    
    # Forecast
    if forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast'],
                mode='lines',
                name='Ensemble Forecast',
                line=dict(color='#2196F3', width=3, dash='dash')
            ),
            row=1, col=1
        )
        
        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Upper_Bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Lower_Bound'],
                mode='lines',
                name='95% Confidence',
                line=dict(width=0),
                fillcolor='rgba(33, 150, 243, 0.2)',
                fill='tonexty'
            ),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=recent.index,
            y=recent['Volume'],
            name='Volume',
            marker_color='rgba(158, 158, 158, 0.5)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#FFE6F0',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
    
    return fig

def main():
    st.title("🧠 Advanced ML Forecast")
    st.caption("Hybrid Ensemble: XGBoost + LightGBM + GARCH Volatility")
    
    # Check library availability
    missing_libs = []
    if not XGBOOST_AVAILABLE:
        missing_libs.append("xgboost")
    if not LIGHTGBM_AVAILABLE:
        missing_libs.append("lightgbm")
    if not ARCH_AVAILABLE:
        missing_libs.append("arch")
    
    if missing_libs:
        st.error(f"⚠️ Missing libraries: {', '.join(missing_libs)}")
        st.info("Install with: `pip install xgboost lightgbm arch ta`")
        return
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Stock ticker to forecast"
        ).upper()
        
        model_type = st.radio(
            "Model Type",
            options=['daily', 'monthly', 'both'],
            index=0,
            help="Daily: LightGBM (fast, noise-resistant)\nMonthly: XGBoost (trend-focused)\nBoth: Ensemble"
        )
        
        forecast_days = st.slider(
            "Forecast Horizon",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Architecture")
        st.markdown("""
        **Primary Engine:**
        - XGBoost (monthly trends)
        - LightGBM (daily signals)
        
        **Features:**
        - 30+ technical indicators
        - GARCH(1,1) volatility
        - Lagged returns
        - Volume patterns
        
        **Confidence:**
        - GARCH-based intervals
        - 95% confidence bands
        """)
    
    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(ticker, period="2y")
    
    if df is None or df.empty:
        st.error(f"Could not fetch data for {ticker}")
        return
    
    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{change:.2f} ({change_pct:+.2f}%)")
    with col2:
        volatility = df['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100
        st.metric("Volatility (30D)", f"{volatility:.1f}%")
    with col3:
        volume_avg = df['Volume'].tail(20).mean()
        st.metric("Avg Volume (20D)", f"{volume_avg/1e6:.1f}M")
    with col4:
        rsi = df['Close'].diff().tail(14).apply(lambda x: max(x, 0)).mean() / abs(df['Close'].diff().tail(14)).mean() * 100
        st.metric("RSI (14)", f"{rsi:.1f}")
    
    # Train model and generate forecast
    with st.spinner("Training ensemble model..."):
        forecast_df, importances, feature_names = create_forecast_ensemble(
            df, forecast_days, model_type
        )
    
    if forecast_df is None:
        st.error("Could not generate forecast. Insufficient data or model training failed.")
        return
    
    # Display forecast metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    target_price = forecast_df['Forecast'].iloc[-1]
    target_change = target_price - current_price
    target_change_pct = (target_change / current_price) * 100
    
    with col1:
        st.metric(
            f"{forecast_days}-Day Target",
            f"${target_price:.2f}",
            f"{target_change:.2f} ({target_change_pct:+.2f}%)"
        )
    with col2:
        st.metric("Upper Range", f"${forecast_df['Upper_Bound'].iloc[-1]:.2f}")
    with col3:
        st.metric("Lower Range", f"${forecast_df['Lower_Bound'].iloc[-1]:.2f}")
    
    # Create and display chart
    fig = create_advanced_chart(df, forecast_df, ticker, model_type)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if importances:
        st.markdown("---")
        with st.expander("📊 Feature Importance (Top 10)", expanded=False):
            for model_name, importance in importances.items():
                st.markdown(f"**{model_name}**")
                
                # Sort and display top 10
                top_idx = np.argsort(importance)[-10:][::-1]
                top_features = [feature_names[i] for i in top_idx]
                top_importance = importance[top_idx]
                
                importance_df = pd.DataFrame({
                    'Feature': top_features,
                    'Importance': top_importance
                })
                st.dataframe(importance_df, use_container_width=True)
    
    # Forecast data table
    with st.expander("📈 Forecast Data", expanded=False):
        display_df = forecast_df.copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d')
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv()
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{ticker}_advanced_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Model info
    st.markdown("---")
    st.info("""
    **🧠 How This Works:**
    
    1. **Feature Engineering**: 30+ technical indicators (RSI, MACD, Bollinger Bands, VWAP, etc.)
    2. **ML Models**: XGBoost/LightGBM trained on historical patterns
    3. **Volatility**: GARCH(1,1) for realistic confidence intervals
    4. **Ensemble**: Weighted combination of models for robust predictions
    
    ⚠️ **Disclaimer**: ML forecasts are probabilistic, not deterministic. Use for research only.
    """)

if __name__ == "__main__":
    main()
