#!/usr/bin/env python3
"""
Simple Dashboard Launcher
Avoids recursion issues by directly launching the dashboard
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def launch_dashboard():
    """Launch the professional trader dashboard"""
    print("🎯 Starting Professional Trader Dashboard...")
    
    try:
        # Import and create dashboard with minimal dependencies
        from dash import Dash, html, dcc
        import dash_bootstrap_components as dbc
        from src.api.schwab_client import SchwabClient
        from src.analysis.market_dynamics import MarketDynamicsAnalyzer
        from src.analysis.big_trades import BigTradesDetector
        
        print("✅ Imports successful")
        
        # Create Dash app
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Simple layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 Professional Trader Dashboard", className="text-center mb-4"),
                    html.P("Dashboard is loading...", className="text-center"),
                    html.Hr()
                ])
            ]),
            
            dcc.Tabs(id="main-tabs", value='analysis', children=[
                dcc.Tab(label='📊 Market Analysis', value='analysis'),
                dcc.Tab(label='📈 Live Data', value='live-data'),
                dcc.Tab(label='💰 Big Trades', value='big-trades')
            ]),
            
            html.Div(id='tab-content', children=[
                html.H3("Welcome to Professional Trader Intelligence"),
                html.P("Click tabs above to navigate different analysis views."),
                html.Div([
                    html.H4("🎯 Features Available:"),
                    html.Ul([
                        html.Li("Market sentiment analysis"),
                        html.Li("Live options flow monitoring"),
                        html.Li("Big trades detection"),
                        html.Li("Professional trade setups"),
                        html.Li("Risk management tools")
                    ])
                ])
            ])
        ])
        
        print("✅ Dashboard layout created")
        print("🚀 Starting server at http://localhost:8050")
        print("💡 Professional trading features:")
        print("   • Market analysis with actionable insights")
        print("   • Live Schwab API data integration")
        print("   • Big trades detection")
        print("   • Professional trade recommendations")
        
        # Start the server
        app.run(host='0.0.0.0', port=8050, debug=False)
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_dashboard()