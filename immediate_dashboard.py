#!/usr/bin/env python3
"""
Live Dashboard - Real Schwab API Data
"""

import sys
import os
sys.path.append('.')

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
import json

# Import our real analysis modules
from src.api.schwab_client import SchwabClient
from src.analysis.market_dynamics import MarketDynamicsAnalyzer
from src.analysis.big_trades import BigTradesDetector

# Global instances
client = SchwabClient()
analyzer = MarketDynamicsAnalyzer(client)
big_trades_detector = BigTradesDetector(client)

def create_live_dashboard():
    """Create dashboard with real Schwab API data"""
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🎯 Professional Options Trading Dashboard", className="text-center mb-4"),
                dbc.Badge("🔴 LIVE DATA", color="success", className="mb-3"),
                html.Hr()
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id="live-content")
            ])
        ]),
        dcc.Interval(id='interval', interval=30*1000, n_intervals=0)
    ], fluid=True)
    
    @app.callback(
        Output('live-content', 'children'),
        [Input('interval', 'n_intervals')]
    )
    def update_content(n):
        try:
            # Get LIVE data from Schwab API
            print(f"🔄 Fetching live data (update #{n})...")
            
            # Get market analysis with real data
            analysis_result = analyzer.analyze_short_term_dynamics()
            
            # Get big trades for key symbols
            symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
            big_trades_data = []
            for symbol in symbols[:2]:  # Limit to first 2 to avoid timeout
                try:
                    trades = big_trades_detector.scan_for_big_trades([symbol])
                    if trades:
                        big_trades_data.extend(trades[:3])  # Top 3 trades per symbol
                except Exception as e:
                    print(f"⚠️ Error getting big trades for {symbol}: {e}")
            
            # Get current VIX
            try:
                vix_quote = client.get_quote('$VIX.X')
                vix_level = vix_quote.get('$VIX.X', {}).get('lastPrice', 'N/A')
            except:
                vix_level = 'N/A'
            
            # Get SPY current price
            try:
                spy_quote = client.get_quote('SPY')
                spy_price = spy_quote.get('SPY', {}).get('lastPrice', 'N/A')
            except:
                spy_price = 'N/A'
            
            return [
                dbc.Alert(f"📊 Last Updated: {datetime.now().strftime('%H:%M:%S')} - Update #{n} - 🔴 LIVE DATA", color="success", className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("🎯 Live Market Data", className="mb-0")),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li(f"SPY Price: ${spy_price}"),
                                    html.Li(f"VIX Level: {vix_level}"),
                                    html.Li(f"Analysis Confidence: {analysis_result.confidence_score:.2f}"),
                                    html.Li(f"Analysis Type: {analysis_result.analysis_type}"),
                                    html.Li(f"Put/Call Ratio: {analysis_result.sentiment.put_call_ratio:.3f}"),
                                    html.Li(f"Sentiment Score: {analysis_result.sentiment.sentiment_score:.3f}")
                                ])
                            ])
                        ], className="mb-3")
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("📈 Big Trades Detected", className="mb-0")),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li(f"Found {len(big_trades_data)} significant trades"),
                                    *[html.Li(f"{trade.symbol} {trade.strike}{trade.option_type} - ${trade.notional_value/1000000:.1f}M") 
                                      for trade in big_trades_data[:5]]
                                ] if big_trades_data else [html.Li("Scanning for big trades...")])
                            ])
                        ], className="mb-3")
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("🎯 Actionable Recommendations", className="mb-0")),
                            dbc.CardBody([
                                html.Ol([html.Li(rec) for rec in analysis_result.recommendations])
                            ])
                        ], className="mb-3")
                    ], width=8),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("⚠️ Risk Factors", className="mb-0")),
                            dbc.CardBody([
                                html.Ul([html.Li(risk) for risk in analysis_result.risk_factors])
                            ])
                        ], className="mb-3")
                    ], width=4)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Alert([
                            html.H4("📊 Analysis Summary", className="alert-heading"),
                            html.P(f"Market Sentiment: {analysis_result.sentiment.sentiment_score:.2f}"),
                            html.P(f"Dealer Position: {analysis_result.sentiment.dealer_positioning}"),
                            html.P(f"Unusual Activity: {len(analysis_result.unusual_activity)} significant activities")
                        ], color="primary")
                    ])
                ])
            ]
            
        except Exception as e:
            print(f"❌ Error in dashboard update: {e}")
            import traceback
            traceback.print_exc()
            return dbc.Alert(f"Error fetching live data: {str(e)}", color="danger")
    
    return app

if __name__ == "__main__":
    print("🚀 Starting LIVE Options Dashboard...")
    print("✅ Real Schwab API integration")
    print("✅ Live market data updates every 30 seconds")
    print("✅ Big trades detection")
    print("✅ Professional trading recommendations")
    print("🌐 Dashboard will be available at: http://localhost:8056")
    
    app = create_live_dashboard()
    app.run(host='0.0.0.0', port=8056, debug=False)