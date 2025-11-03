"""
Visualization Module for Options Trading Platform
Creates charts, dashboards, and visual analysis tools
"""

import dash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..data.database import DatabaseManager
from ..analysis.market_dynamics import MarketDynamicsAnalyzer
from ..analysis.big_trades import BigTradesDetector
from ..api.schwab_client import SchwabClient
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

def format_analysis_result(result):
    """Convert MarketAnalysisResult to formatted display string for Dash"""
    if result is None:
        return "No analysis available"
    
    # Extract the key information for display
    formatted = []
    
    # Header
    formatted.append(f"📊 {result.analysis_type.upper()} ANALYSIS")
    formatted.append(f"⏰ Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    formatted.append("")
    
    # Market Sentiment
    sentiment = result.sentiment
    formatted.append("🎯 MARKET SENTIMENT")
    formatted.append(f"   • Put/Call Ratio: {sentiment.put_call_ratio:.3f}")
    formatted.append(f"   • VIX Level: {sentiment.vix_level}")
    formatted.append(f"   • Gamma Exposure: {sentiment.gamma_exposure:.3f}")
    formatted.append(f"   • Dealer Position: {sentiment.dealer_positioning}")
    formatted.append(f"   • Sentiment Score: {sentiment.sentiment_score:.3f}")
    formatted.append("")
    
    # Recommendations (this is the actionable data!)
    if result.recommendations:
        formatted.append("🎯 ACTIONABLE RECOMMENDATIONS")
        for rec in result.recommendations:
            formatted.append(f"   {rec}")
        formatted.append("")
    
    # Risk Factors
    if result.risk_factors:
        formatted.append("⚠️  RISK FACTORS")
        for risk in result.risk_factors:
            formatted.append(f"   {risk}")
        formatted.append("")
    
    # Key Levels (Top 5 most important)
    key_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
    formatted.append("📈 KEY LEVELS")
    for symbol in key_symbols:
        max_pain = result.key_levels.get(f'{symbol}_max_pain', 'N/A')
        put_wall = result.key_levels.get(f'{symbol}_put_wall', 'N/A')
        call_wall = result.key_levels.get(f'{symbol}_call_wall', 'N/A')
        if max_pain != 'N/A':
            formatted.append(f"   • {symbol}: Max Pain ${max_pain} | Put Wall ${put_wall} | Call Wall ${call_wall}")
    formatted.append("")
    
    # Confidence
    formatted.append(f"📊 Confidence Score: {result.confidence_score:.2f}")
    
    return "\n".join(formatted)

class OptionsVisualization:
    """
    Creates various visualizations for options trading data
    """
    
    def __init__(self, port: int = 8050):
        """Initialize the options dashboard"""
        self.app = Dash(__name__)
        self.port = port
        self.schwab_client = None
        self.analyzer = None
        self.big_trades_detector = None
        self.db_manager = None
        self.trader_dashboard = None
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _lazy_init_components(self):
        """Lazy initialization of components to avoid recursion"""
        if self.schwab_client is None:
            try:
                self.schwab_client = SchwabClient()
                self.analyzer = MarketDynamicsAnalyzer(self.schwab_client)
                self.big_trades_detector = BigTradesDetector(self.schwab_client)
                # Skip database for now to avoid recursion
                # self.db_manager = DatabaseManager()
                
                # Import here to avoid circular imports
                from .trader_dashboard import TraderDashboard
                self.trader_dashboard = TraderDashboard()
                logger.info("✅ Components initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing components: {e}")
                # Create mock components for basic functionality
                self.schwab_client = None
        
    def create_options_flow_chart(self, symbol: str, days: int = 5) -> go.Figure:
        """
        Create options flow visualization using live Schwab data
        """
        try:
            # First try to get data from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.db_manager.get_options_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # If no database data, fetch live data from Schwab
            if df.empty:
                logger.info(f"No database data found for {symbol}, fetching live data...")
                return self._create_live_options_flow_chart(symbol)
            
            # Process database data
            fig = go.Figure()
            
            # Group by contract type
            calls = df[df['contract_type'] == 'call']
            puts = df[df['contract_type'] == 'put']
            
            # Add volume traces
            if not calls.empty:
                fig.add_trace(go.Scatter(
                    x=calls['timestamp'],
                    y=calls['volume'],
                    mode='lines+markers',
                    name='Call Volume',
                    line=dict(color='green'),
                    marker=dict(size=4)
                ))
            
            if not puts.empty:
                fig.add_trace(go.Scatter(
                    x=puts['timestamp'],
                    y=puts['volume'],
                    mode='lines+markers',
                    name='Put Volume',
                    line=dict(color='red'),
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                title=f"{symbol} Options Flow - Last {days} Days",
                xaxis_title="Time",
                yaxis_title="Volume",
                hovermode='x unified',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating options flow chart: {e}")
            return self._create_empty_chart(f"Error loading data for {symbol}")

    def _create_live_options_flow_chart(self, symbol: str) -> go.Figure:
        """
        Create options flow chart using live Schwab API data
        """
        try:
            # Get current quote
            quote_data = self.schwab_client.get_quote(symbol)
            if not quote_data or symbol not in quote_data:
                return self._create_empty_chart(f"No quote data available for {symbol}")
            
            current_price = quote_data[symbol]['quote']['lastPrice']
            
            # Get options chain
            options_chain = self.schwab_client.get_options_chain(symbol, strike_count=20)
            if not options_chain:
                return self._create_empty_chart(f"No options data available for {symbol}")
            
            # Convert to DataFrame
            options_df, spot_price = self.schwab_client.options_chain_to_dataframe(options_chain)
            
            if options_df.empty:
                return self._create_empty_chart(f"No options chain data for {symbol}")
            
            # Create visualization
            fig = go.Figure()
            
            # Group by contract type
            calls = options_df[options_df['putCall'] == 'CALL']
            puts = options_df[options_df['putCall'] == 'PUT']
            
            if not calls.empty:
                # Plot call volume vs strike
                fig.add_trace(go.Scatter(
                    x=calls['strike'],
                    y=calls['totalVolume'],
                    mode='markers',
                    name='Call Volume',
                    marker=dict(
                        size=calls['totalVolume'] / calls['totalVolume'].max() * 30,
                        color='green',
                        opacity=0.7
                    ),
                    text=calls['strike'],
                    hovertemplate='Strike: %{text}<br>Volume: %{y}<extra></extra>'
                ))
            
            if not puts.empty:
                # Plot put volume vs strike
                fig.add_trace(go.Scatter(
                    x=puts['strike'],
                    y=puts['totalVolume'],
                    mode='markers',
                    name='Put Volume',
                    marker=dict(
                        size=puts['totalVolume'] / puts['totalVolume'].max() * 30,
                        color='red',
                        opacity=0.7
                    ),
                    text=puts['strike'],
                    hovertemplate='Strike: %{text}<br>Volume: %{y}<extra></extra>'
                ))
            
            # Add current price line
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Current: ${current_price:.2f}"
            )
            
            fig.update_layout(
                title=f"{symbol} Live Options Volume by Strike",
                xaxis_title="Strike Price",
                yaxis_title="Volume",
                hovermode='closest',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating live options flow chart: {e}")
            return self._create_empty_chart(f"Error fetching live data for {symbol}")
        
        # Aggregate by contract type and time
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily_flow = df.groupby(['date', 'contract_type']).agg({
            'volume': 'sum',
            'premium': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{symbol} Options Volume', f'{symbol} Average Premium'],
            vertical_spacing=0.15
        )
        
        # Volume chart
        for contract_type in ['call', 'put']:
            data = daily_flow[daily_flow['contract_type'] == contract_type]
            color = 'green' if contract_type == 'call' else 'red'
            
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data['volume'],
                    name=f'{contract_type.title()} Volume',
                    marker_color=color,
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Premium chart
        for contract_type in ['call', 'put']:
            data = daily_flow[daily_flow['contract_type'] == contract_type]
            color = 'darkgreen' if contract_type == 'call' else 'darkred'
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['premium'],
                    mode='lines+markers',
                    name=f'{contract_type.title()} Avg Premium',
                    line=dict(color=color, width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'Options Flow Analysis - {symbol}',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_put_call_ratio_chart(self, symbols: List[str], days: int = 30) -> go.Figure:
        """
        Create put/call ratio chart using live data with gauge visualization
        """
        try:
            fig = make_subplots(
                rows=1, cols=len(symbols),
                subplot_titles=[f"{symbol} P/C Ratio" for symbol in symbols],
                specs=[[{'type': 'indicator'}] * len(symbols)]
            )
            
            for idx, symbol in enumerate(symbols, 1):
                try:
                    # Get live market analysis
                    analysis = self.market_analyzer.analyze_short_term_dynamics([symbol])
                    if hasattr(analysis, 'sentiment') and hasattr(analysis.sentiment, 'put_call_ratio'):
                        pc_ratio = analysis.sentiment.put_call_ratio
                        
                        # Determine color based on ratio
                        if pc_ratio > 1.2:
                            color = "red"  # Bearish
                        elif pc_ratio < 0.8:
                            color = "green"  # Bullish
                        else:
                            color = "yellow"  # Neutral
                            
                        # Add gauge
                        fig.add_trace(
                            go.Indicator(
                                mode="gauge+number+delta",
                                value=pc_ratio,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': f"{symbol} P/C Ratio"},
                                delta={'reference': 1.0},
                                gauge={
                                    'axis': {'range': [None, 2.5]},
                                    'bar': {'color': color},
                                    'steps': [
                                        {'range': [0, 0.8], 'color': "lightgreen"},
                                        {'range': [0.8, 1.2], 'color': "lightyellow"},
                                        {'range': [1.2, 2.5], 'color': "lightcoral"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 1.5
                                    }
                                }
                            ),
                            row=1, col=idx
                        )
                        
                except Exception as e:
                    logger.warning(f"Could not get P/C ratio for {symbol}: {e}")
                    continue
            
            fig.update_layout(
                title='Put/Call Ratio Indicators',
                height=300
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating put/call ratio chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    def create_big_trades_chart(self, days: int = 7) -> go.Figure:
        """
        Create big trades visualization with live data
        """
        try:
            # Get live big trades using the detector
            big_trades = self.big_trades_detector.scan_for_big_trades(
                ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA', 'MSFT'],
                min_premium=50000  # $50k minimum
            )
            
            if not big_trades:
                return self._create_empty_chart("No big trades detected")
            
            # Create scatter plot
            fig = go.Figure()
            
            # Color by sentiment
            color_map = {'bullish': 'green', 'bearish': 'red', 'neutral': 'blue'}
            
            # Group by sentiment
            for sentiment in ['bullish', 'bearish', 'neutral']:
                sentiment_trades = [t for t in big_trades if t.sentiment == sentiment]
                if not sentiment_trades:
                    continue
                
                timestamps = [t.timestamp for t in sentiment_trades]
                notional_values = [t.notional_value for t in sentiment_trades]
                size_scores = [t.size_score for t in sentiment_trades]
                
                text_labels = []
                for trade in sentiment_trades:
                    text_labels.append(
                        f"{trade.symbol} {trade.contract_type} {trade.strike}<br>"
                        f"Volume: {trade.volume:,}<br>"
                        f"Notional: ${trade.notional_value:,.0f}"
                    )
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=notional_values,
                        mode='markers',
                        marker=dict(
                            size=[score * 3 for score in size_scores],  # Size based on trade size
                            color=color_map.get(sentiment, 'gray'),
                            opacity=0.7,
                            line=dict(width=1, color='black')
                        ),
                        text=text_labels,
                        hovertemplate='%{text}<extra></extra>',
                        name=f'{sentiment.title()} Trades'
                    )
                )
            
            fig.update_layout(
                title='Live Big Options Trades',
                xaxis_title='Time',
                yaxis_title='Notional Value ($)',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating big trades chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    def create_unusual_activity_chart(self, days: int = 7) -> go.Figure:
        """
        Create unusual activity chart with live data
        """
        try:
            # Get live market analysis for multiple symbols
            symbols = ['SPY', 'QQQ', 'TSLA', 'AAPL', 'NVDA']
            unusual_data = []
            
            for symbol in symbols:
                analysis = self.market_analyzer.analyze_short_term_dynamics([symbol])
                if hasattr(analysis, 'unusual_activity') and analysis.unusual_activity:
                    for activity in analysis.unusual_activity:
                        unusual_data.append({
                            'symbol': activity.symbol,
                            'timestamp': activity.timestamp,
                            'contract_type': activity.contract_type,
                            'volume': activity.volume,
                            'strike': activity.strike,
                            'implied_volatility': activity.implied_volatility
                        })
            
            if not unusual_data:
                return self._create_empty_chart("No unusual activity detected")
            
            df = pd.DataFrame(unusual_data)
            
            # Create bar chart by symbol
            fig = go.Figure()
            
            symbols_activity = df.groupby('symbol')['volume'].sum().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=symbols_activity.index,
                    y=symbols_activity.values,
                    name='Unusual Volume',
                    marker_color='orange'
                )
            )
            
            fig.update_layout(
                title='Unusual Options Activity',
                xaxis_title='Symbol',
                yaxis_title='Volume',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating unusual activity chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    def create_iv_surface_chart(self, symbol: str) -> go.Figure:
        """
        Create implied volatility surface
        """
        df = self.db_manager.get_options_data(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=1)
        )
        
        if df.empty:
            return self._create_empty_chart(f"No IV data available for {symbol}")
        
        # Filter for recent data and valid IV
        df = df[df['implied_volatility'] > 0]
        
        if df.empty:
            return self._create_empty_chart(f"No valid IV data for {symbol}")
        
        # Calculate time to expiration
        df['expiration_date'] = pd.to_datetime(df['expiration'])
        df['dte'] = (df['expiration_date'] - pd.Timestamp.now()).dt.days
        
        # Create pivot table for surface
        pivot_calls = df[df['contract_type'] == 'call'].pivot_table(
            values='implied_volatility',
            index='dte',
            columns='strike',
            aggfunc='mean'
        )
        
        pivot_puts = df[df['contract_type'] == 'put'].pivot_table(
            values='implied_volatility',
            index='dte',
            columns='strike',
            aggfunc='mean'
        )
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'{symbol} Call IV Surface', f'{symbol} Put IV Surface'],
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )
        
        # Calls surface
        if not pivot_calls.empty:
            fig.add_trace(
                go.Surface(
                    z=pivot_calls.values,
                    x=pivot_calls.columns,
                    y=pivot_calls.index,
                    colorscale='Viridis',
                    name='Calls'
                ),
                row=1, col=1
            )
        
        # Puts surface
        if not pivot_puts.empty:
            fig.add_trace(
                go.Surface(
                    z=pivot_puts.values,
                    x=pivot_puts.columns,
                    y=pivot_puts.index,
                    colorscale='Plasma',
                    name='Puts'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f'Implied Volatility Surface - {symbol}',
            height=600
        )
        
        return fig
    
    def create_gamma_exposure_chart(self, symbols: List[str]) -> go.Figure:
        """
        Create gamma exposure chart with live data - fetches next trading day if market closed
        """
        try:
            fig = go.Figure()
            
            for symbol in symbols:
                try:
                    # Get live options chain - this will automatically get next available expiration
                    chain_data = self.schwab_client.get_options_chain(
                        symbol=symbol,
                        contract_type="ALL",
                        strike_count=30  # More strikes for better gamma exposure view
                    )
                    
                    # Get current stock price (use close price if last price is 0 due to after hours)
                    quotes = self.schwab_client.get_quotes([symbol])
                    quote_data = quotes.get(symbol, {})
                    current_price = quote_data.get('lastPrice', 0) or quote_data.get('closePrice', 0)
                    
                    logger.info(f"Processing gamma for {symbol} at price ${current_price}")
                    
                    if not current_price:
                        logger.warning(f"No price data for {symbol}")
                        continue
                    
                    call_gamma_by_strike = {}
                    put_gamma_by_strike = {}
                    
                    # Process call options - get from multiple expirations for better gamma data
                    if 'callExpDateMap' in chain_data:
                        exp_count = 0
                        for exp_date, strikes_data in chain_data['callExpDateMap'].items():
                            exp_count += 1
                            if exp_count > 3:  # Limit to first 3 expirations
                                break
                                
                            logger.info(f"Processing calls for {symbol} expiration: {exp_date}")
                            for strike_price, options_data in strikes_data.items():
                                strike = float(strike_price)
                                
                                for option in options_data:
                                    gamma = option.get('gamma', 0)
                                    open_interest = option.get('openInterest', 0)
                                    days_to_exp = option.get('daysToExpiration', 0)
                                    
                                    # Log first few for debugging
                                    if len(call_gamma_by_strike) < 3:
                                        logger.info(f"Call option {strike}: gamma={gamma}, OI={open_interest}, DTE={days_to_exp}")
                                    
                                    # Only use options with positive DTE (not expired) and valid gamma
                                    if days_to_exp > 0 and abs(gamma) > 0.001 and open_interest > 0:
                                        # Gamma exposure = gamma * open_interest * 100 * underlying_price
                                        # Dealer GEX: Calls negative (dealers need to buy rallies)
                                        gamma_exposure = -gamma * open_interest * 100 * current_price
                                        
                                        if strike not in call_gamma_by_strike:
                                            call_gamma_by_strike[strike] = 0
                                        call_gamma_by_strike[strike] += gamma_exposure
                    
                    # Process put options - get from multiple expirations
                    if 'putExpDateMap' in chain_data:
                        exp_count = 0
                        for exp_date, strikes_data in chain_data['putExpDateMap'].items():
                            exp_count += 1
                            if exp_count > 3:  # Limit to first 3 expirations
                                break
                                
                            logger.info(f"Processing puts for {symbol} expiration: {exp_date}")
                            for strike_price, options_data in strikes_data.items():
                                strike = float(strike_price)
                                
                                for option in options_data:
                                    gamma = option.get('gamma', 0)
                                    open_interest = option.get('openInterest', 0)
                                    days_to_exp = option.get('daysToExpiration', 0)
                                    
                                    # Log first few for debugging
                                    if len(put_gamma_by_strike) < 3:
                                        logger.info(f"Put option {strike}: gamma={gamma}, OI={open_interest}, DTE={days_to_exp}")
                                    
                                    # Only use options with positive DTE (not expired) and valid gamma
                                    if days_to_exp > 0 and abs(gamma) > 0.001 and open_interest > 0:
                                        # Gamma exposure = gamma * open_interest * 100 * underlying_price
                                        # Dealer GEX: Puts positive (dealers need to sell dips)
                                        gamma_exposure = gamma * open_interest * 100 * current_price
                                        
                                        if strike not in put_gamma_by_strike:
                                            put_gamma_by_strike[strike] = 0
                                        put_gamma_by_strike[strike] += gamma_exposure
                    
                    # Add call gamma bars
                    if call_gamma_by_strike:
                        logger.info(f"Found {len(call_gamma_by_strike)} call strikes with gamma for {symbol}")
                        strikes = list(call_gamma_by_strike.keys())
                        gamma_values = list(call_gamma_by_strike.values())
                        
                        fig.add_trace(
                            go.Bar(
                                x=strikes,
                                y=gamma_values,
                                name=f'{symbol} Call Gamma',
                                marker_color='green',
                                opacity=0.7,
                                text=[f'${v/1e6:.1f}M' for v in gamma_values],
                                textposition='auto'
                            )
                        )
                    else:
                        logger.warning(f"No call gamma data for {symbol}")
                    
                    # Add put gamma bars
                    if put_gamma_by_strike:
                        logger.info(f"Found {len(put_gamma_by_strike)} put strikes with gamma for {symbol}")
                        strikes = list(put_gamma_by_strike.keys())
                        gamma_values = list(put_gamma_by_strike.values())
                        
                        fig.add_trace(
                            go.Bar(
                                x=strikes,
                                y=gamma_values,
                                name=f'{symbol} Put Gamma',
                                marker_color='red',
                                opacity=0.7,
                                text=[f'${v/1e6:.1f}M' for v in gamma_values],
                                textposition='auto'
                            )
                        )
                    else:
                        logger.warning(f"No put gamma data for {symbol}")
                    
                    # Add current price reference line
                    if call_gamma_by_strike or put_gamma_by_strike:
                        fig.add_vline(
                            x=current_price,
                            line_dash="dash",
                            line_color="black",
                            annotation_text=f"{symbol}: ${current_price:.2f}",
                            annotation_position="top"
                        )
                        
                except Exception as e:
                    logger.warning(f"Could not get gamma data for {symbol}: {e}")
                    continue
            
            # Check if we have any gamma data at all
            total_symbols_with_gamma = 0
            for symbol in symbols:
                if any(f'{symbol}' in trace.name for trace in fig.data):
                    total_symbols_with_gamma += 1
            
            if total_symbols_with_gamma == 0:
                # No gamma data available - market is closed, only expired options available
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    text="📊 Market Closed - No Active Gamma Data<br><br>" +
                         "🕐 Only expired options available (DTE ≤ 0)<br>" +
                         "⏰ Gamma exposure will show during trading hours<br>" +
                         "🎯 Live gamma data available: Mon-Fri 9:30 AM - 4:00 PM ET<br><br>" +
                         "💡 Gamma measures options price sensitivity to underlying moves",
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                    align="center"
                )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
            
            fig.update_layout(
                title='Gamma Exposure by Strike Price',
                xaxis_title='Strike Price ($)',
                yaxis_title='Gamma Exposure ($ Millions)',
                height=500,
                barmode='relative',
                showlegend=True,
                hovermode='x unified',
                yaxis=dict(tickformat='.1s')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gamma exposure chart: {e}")
            return self._create_empty_chart(f"Gamma Error: {str(e)}")
    
    def create_unusual_activity_chart(self, days: int = 7) -> go.Figure:
        """
        Create unusual activity timeline
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = self.db_manager.get_unusual_activity(
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return self._create_empty_chart("No unusual activity found")
        
        # Create timeline chart
        fig = go.Figure()
        
        # Color by activity type
        activity_colors = {
            'volume_spike': 'blue',
            'big_trade': 'red',
            'iv_surge': 'orange',
            'sweep': 'purple',
            'unusual_flow': 'green'
        }
        
        for activity_type in df['activity_type'].unique():
            type_data = df[df['activity_type'] == activity_type]
            
            fig.add_trace(
                go.Scatter(
                    x=type_data['timestamp'],
                    y=type_data['severity'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=activity_colors.get(activity_type, 'gray'),
                        opacity=0.7
                    ),
                    text=type_data.apply(
                        lambda row: f"{row['symbol']}: {row['description']}",
                        axis=1
                    ),
                    hovertemplate='%{text}<br>Severity: %{y}<extra></extra>',
                    name=activity_type.replace('_', ' ').title()
                )
            )
        
        fig.update_layout(
            title='Unusual Activity Timeline',
            xaxis_title='Time',
            yaxis_title='Severity Score',
            height=500,
            hovermode='closest'
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Create an empty chart with a message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 Professional Options Trading Intelligence", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Navigation tabs
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id="main-tabs", value='trader-intelligence', children=[
                        dcc.Tab(label='🎯 Trader Intelligence', value='trader-intelligence'),
                        dcc.Tab(label='📊 Market Analysis', value='market-analysis'),
                        dcc.Tab(label='📈 Options Flow', value='options-flow'),
                        dcc.Tab(label='💰 Big Trades', value='big-trades'),
                        dcc.Tab(label='📉 Gamma Exposure', value='gamma-exposure')
                    ])
                ])
            ], className="mb-4"),
            
            # Content area
            html.Div(id='tab-content'),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def render_tab_content(active_tab, n_intervals):
            if active_tab == 'trader-intelligence':
                return self._render_trader_intelligence_tab()
            elif active_tab == 'market-analysis':
                return self._render_market_analysis_tab()
            elif active_tab == 'options-flow':
                return self._render_options_flow_tab()
            elif active_tab == 'big-trades':
                return self._render_big_trades_tab()
            elif active_tab == 'gamma-exposure':
                return self._render_gamma_exposure_tab()
            else:
                return html.Div("Select a tab")
    
    def _render_trader_intelligence_tab(self):
        """Render professional trader intelligence tab"""
        try:
            self._lazy_init_components()
            if self.trader_dashboard is None:
                return html.Div([
                    html.H4("🎯 Professional Trader Intelligence"),
                    html.P("Initializing components..."),
                    html.P("This may take a moment on first load.")
                ])
            
            charts = self.trader_dashboard.create_trader_intelligence_view()
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['market_overview'])
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['smart_money_flow'])
                    ], width=8),
                    dbc.Col([
                        dcc.Graph(figure=charts['risk_monitor'])
                    ], width=4)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['trading_opportunities'])
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=charts['positioning_heatmap'])
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['sector_rotation'])
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=charts['gamma_wall_levels'])
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['overnight_setup'])
                    ], width=12)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering trader intelligence: {e}")
            return html.Div(f"Error loading trader intelligence: {str(e)}")
    
    def _render_market_analysis_tab(self):
        """Render market analysis tab"""
        try:
            self._lazy_init_components()
            if self.analyzer is None:
                return html.Div([
                    html.H4("📊 Market Analysis"),
                    html.P("Schwab API client not available. Please check configuration.")
                ])
            
            # Get market analysis
            short_term = self.analyzer.analyze_short_term_dynamics()
            mid_term = self.analyzer.analyze_mid_term_dynamics()
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H4("📈 Short-term Analysis (1-3 days)"),
                        html.Pre(short_term, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
                    ], width=6),
                    dbc.Col([
                        html.H4("📊 Mid-term Analysis (1-2 weeks)"),
                        html.Pre(mid_term, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
                    ], width=6)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering market analysis: {e}")
            return html.Div(f"Error loading market analysis: {str(e)}")
    
    def _render_options_flow_tab(self):
        """Render options flow tab"""
        try:
            symbols = ['SPY', 'QQQ', 'IWM']
            charts = []
            
            for symbol in symbols:
                flow_chart = self.create_options_flow_chart(symbol, 5)
                charts.append(dbc.Col([
                    dcc.Graph(figure=flow_chart)
                ], width=4))
            
            return dbc.Container([
                dbc.Row(charts)
            ])
            
        except Exception as e:
            logger.error(f"Error rendering options flow: {e}")
            return html.Div(f"Error loading options flow: {str(e)}")
    
    def _render_big_trades_tab(self):
        """Render big trades tab"""
        try:
            big_trades_chart = self.create_big_trades_chart(['SPY', 'QQQ', 'AAPL', 'TSLA'])
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=big_trades_chart)
                    ], width=12)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering big trades: {e}")
            return html.Div(f"Error loading big trades: {str(e)}")
    
    def _render_gamma_exposure_tab(self):
        """Render gamma exposure tab"""
        try:
            gamma_chart = self.create_gamma_exposure_chart(['SPY', 'QQQ'])
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=gamma_chart)
                    ], width=12)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering gamma exposure: {e}")
            return html.Div(f"Error loading gamma exposure: {str(e)}")
    
    def create_big_trades_chart(self, symbols: List[str]) -> go.Figure:
        """Create big trades detection chart"""
        return self._create_empty_chart("Big Trades Detection - Feature available with full API integration")
    
    def create_gamma_exposure_chart(self, symbols: List[str]) -> go.Figure:
        """Create gamma exposure chart"""
        return self._create_empty_chart("Gamma Exposure Analysis - Live during market hours")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=message,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig

    def run(self, debug: bool = False):
        """Run the dashboard"""
        logger.info(f"Starting professional trader dashboard at http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

class OptionsDashboard:
    """
    Dash-based dashboard for options trading analysis
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8050):
        self.host = host
        self.port = port
        self.viz = OptionsVisualization()
        self.schwab_client = SchwabClient()
        self.market_analyzer = MarketDynamicsAnalyzer(self.schwab_client)
        self.big_trades_detector = BigTradesDetector(self.schwab_client)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 Professional Options Trading Intelligence", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Navigation tabs
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id="main-tabs", value='trader-intelligence', children=[
                        dcc.Tab(label='🎯 Trader Intelligence', value='trader-intelligence'),
                        dcc.Tab(label='📊 Market Analysis', value='market-analysis'),
                        dcc.Tab(label='📈 Options Flow', value='options-flow'),
                        dcc.Tab(label='💰 Big Trades', value='big-trades'),
                        dcc.Tab(label='📉 Gamma Exposure', value='gamma-exposure')
                    ])
                ])
            ], className="mb-4"),
            
            # Content area
            html.Div(id='tab-content'),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def render_tab_content(active_tab, n_intervals):
            if active_tab == 'trader-intelligence':
                return self._render_trader_intelligence_tab()
            elif active_tab == 'market-analysis':
                return self._render_market_analysis_tab()
            elif active_tab == 'options-flow':
                return self._render_options_flow_tab()
            elif active_tab == 'big-trades':
                return self._render_big_trades_tab()
            elif active_tab == 'gamma-exposure':
                return self._render_gamma_exposure_tab()
            else:
                return html.Div("Select a tab")
    
    def _render_trader_intelligence_tab(self):
        """Render professional trader intelligence tab"""
        try:
            self._lazy_init_components()
            if self.trader_dashboard is None:
                return html.Div([
                    html.H4("🎯 Professional Trader Intelligence"),
                    html.P("Initializing components..."),
                    html.P("This may take a moment on first load.")
                ])
            
            charts = self.trader_dashboard.create_trader_intelligence_view()
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['market_overview'])
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['smart_money_flow'])
                    ], width=8),
                    dbc.Col([
                        dcc.Graph(figure=charts['risk_monitor'])
                    ], width=4)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['trading_opportunities'])
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=charts['positioning_heatmap'])
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['sector_rotation'])
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=charts['gamma_wall_levels'])
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=charts['overnight_setup'])
                    ], width=12)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering trader intelligence: {e}")
            return html.Div(f"Error loading trader intelligence: {str(e)}")
    
    def _render_market_analysis_tab(self):
        """Render market analysis tab"""
        try:
            self._lazy_init_components()
            if self.analyzer is None:
                return html.Div([
                    html.H4("📊 Market Analysis"),
                    html.P("Schwab API client not available. Please check configuration.")
                ])
            
            # Get market analysis
            short_term = self.analyzer.analyze_short_term_dynamics()
            mid_term = self.analyzer.analyze_mid_term_dynamics()
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H4("📈 Short-term Analysis (1-3 days)"),
                        html.Pre(short_term, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
                    ], width=6),
                    dbc.Col([
                        html.H4("📊 Mid-term Analysis (1-2 weeks)"),
                        html.Pre(mid_term, style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
                    ], width=6)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering market analysis: {e}")
            return html.Div(f"Error loading market analysis: {str(e)}")
    
    def _render_options_flow_tab(self):
        """Render options flow tab"""
        try:
            symbols = ['SPY', 'QQQ', 'IWM']
            charts = []
            
            for symbol in symbols:
                flow_chart = self.create_options_flow_chart(symbol, 5)
                charts.append(dbc.Col([
                    dcc.Graph(figure=flow_chart)
                ], width=4))
            
            return dbc.Container([
                dbc.Row(charts)
            ])
            
        except Exception as e:
            logger.error(f"Error rendering options flow: {e}")
            return html.Div(f"Error loading options flow: {str(e)}")
    
    def _render_big_trades_tab(self):
        """Render big trades tab"""
        try:
            big_trades_chart = self.create_big_trades_chart(['SPY', 'QQQ', 'AAPL', 'TSLA'])
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=big_trades_chart)
                    ], width=12)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering big trades: {e}")
            return html.Div(f"Error loading big trades: {str(e)}")
    
    def _render_gamma_exposure_tab(self):
        """Render gamma exposure tab"""
        try:
            gamma_chart = self.create_gamma_exposure_chart(['SPY', 'QQQ'])
            
            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=gamma_chart)
                    ], width=12)
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error rendering gamma exposure: {e}")
            return html.Div(f"Error loading gamma exposure: {str(e)}")
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        logger.info(f"Starting professional trader dashboard at http://localhost:{self.port}")
        self.app.run_server(host='0.0.0.0', port=self.port, debug=debug)

# Example usage
if __name__ == "__main__":
    # Create visualizations
    viz = OptionsVisualization()
    
    # Create individual charts
    flow_chart = viz.create_options_flow_chart('SPY')
    flow_chart.show()
    
    # Start dashboard
    dashboard = OptionsDashboard()
    dashboard.run(debug=True)