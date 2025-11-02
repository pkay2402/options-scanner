"""
Professional Trader Dashboard
Real-time actionable intelligence for professional trading
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from ..analysis.trader_intelligence import TraderIntelligenceEngine, MarketIntelligence
from ..visualization.dashboard import OptionsVisualization

logger = logging.getLogger(__name__)

class TraderDashboard(OptionsVisualization):
    """
    Professional trader's real-time intelligence dashboard
    """
    
    def __init__(self):
        super().__init__()
        try:
            self.intel_engine = TraderIntelligenceEngine(self.schwab_client)
        except Exception as e:
            logger.warning(f"Could not initialize trader intelligence engine: {e}")
            self.intel_engine = None
        
    def create_trader_intelligence_view(self) -> Dict[str, go.Figure]:
        """
        Create comprehensive trader intelligence view
        """
        logger.info("🎯 Generating trader intelligence dashboard...")
        
        # Check if intelligence engine is available
        if self.intel_engine is None:
            return self._create_fallback_charts()
        
        try:
            # Get market intelligence
            intel = self.intel_engine.generate_market_intelligence()
        except Exception as e:
            logger.error(f"Error generating intelligence: {e}")
            return self._create_fallback_charts()
        
        # Create individual components
        charts = {
            'market_overview': self._create_market_overview_chart(intel),
            'smart_money_flow': self._create_smart_money_flow_chart(intel.smart_money_flows),
            'positioning_heatmap': self._create_positioning_heatmap(intel.positioning_analysis),
            'trading_opportunities': self._create_opportunities_chart(intel.trading_opportunities),
            'sector_rotation': self._create_sector_rotation_chart(intel.sector_rotation),
            'risk_monitor': self._create_risk_monitor_chart(intel),
            'gamma_wall_levels': self._create_gamma_wall_chart(intel.positioning_analysis),
            'overnight_setup': self._create_overnight_setup_chart(intel)
        }
        
        return charts
    
    def _create_market_overview_chart(self, intel: MarketIntelligence) -> go.Figure:
        """Create market regime and VIX overview"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Market Regime", "VIX Analysis", 
                "Key Events", "Risk Factors"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "table"}, {"type": "table"}]
            ]
        )
        
        # Market regime indicator
        regime_colors = {
            'risk_on': 'green',
            'risk_off': 'red', 
            'rotation': 'orange',
            'consolidation': 'blue',
            'vol_expansion': 'purple',
            'vol_contraction': 'yellow'
        }
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=1,
                title={"text": f"Market Regime<br>{intel.market_regime.value.replace('_', ' ').title()}"},
                gauge={
                    'bar': {'color': regime_colors.get(intel.market_regime.value, 'gray')},
                    'bgcolor': 'lightgray',
                    'steps': [{'range': [0, 1], 'color': "lightgray"}]
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # VIX regime
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=1,
                title={"text": f"VIX Regime<br>{intel.vix_regime}"},
                number={'font': {'size': 12}}
            ),
            row=1, col=2
        )
        
        # Key events table
        if intel.key_events:
            events_df = pd.DataFrame({'Events': intel.key_events})
            fig.add_trace(
                go.Table(
                    header=dict(values=['🔥 Key Market Events']),
                    cells=dict(values=[intel.key_events])
                ),
                row=2, col=1
            )
        
        # Risk factors table  
        if intel.risk_factors:
            fig.add_trace(
                go.Table(
                    header=dict(values=['⚠️ Risk Factors']),
                    cells=dict(values=[intel.risk_factors])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="🎯 MARKET INTELLIGENCE OVERVIEW",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_smart_money_flow_chart(self, flows: List) -> go.Figure:
        """Create smart money flow analysis"""
        if not flows:
            return self._create_empty_chart("No smart money flow data available")
        
        # Create dataframe
        data = []
        for flow in flows:
            data.append({
                'Symbol': flow.symbol,
                'Net Flow ($M)': flow.net_flow / 1e6,
                'Call Flow ($M)': flow.call_flow / 1e6, 
                'Put Flow ($M)': flow.put_flow / 1e6,
                'Avg Trade Size ($K)': flow.avg_trade_size / 1000,
                'Institutional': flow.institutional_trades,
                'Retail': flow.retail_trades,
                'Confidence': f"{flow.confidence:.1%}"
            })
        
        df = pd.DataFrame(data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Net Money Flow", "Call vs Put Flow", 
                "Institutional vs Retail", "Trade Characteristics"
            ]
        )
        
        # Net flow waterfall
        colors = ['green' if x > 0 else 'red' for x in df['Net Flow ($M)']]
        
        fig.add_trace(
            go.Bar(
                x=df['Symbol'],
                y=df['Net Flow ($M)'],
                marker_color=colors,
                name='Net Flow',
                text=[f"${x:.1f}M" for x in df['Net Flow ($M)']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Call vs Put flows
        fig.add_trace(
            go.Bar(
                x=df['Symbol'],
                y=df['Call Flow ($M)'],
                name='Call Flow',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=df['Symbol'],
                y=-df['Put Flow ($M)'],  # Negative for visual separation
                name='Put Flow',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Institutional vs Retail
        fig.add_trace(
            go.Scatter(
                x=df['Institutional'],
                y=df['Retail'],
                mode='markers+text',
                text=df['Symbol'],
                textposition='top center',
                marker=dict(
                    size=df['Avg Trade Size ($K)'] / 10,  # Size based on avg trade
                    color=df['Net Flow ($M)'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Net Flow ($M)")
                ),
                name='Trade Distribution'
            ),
            row=2, col=1
        )
        
        # Trade characteristics table
        display_df = df[['Symbol', 'Net Flow ($M)', 'Avg Trade Size ($K)', 'Confidence']].head(5)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color='lightblue'
                ),
                cells=dict(
                    values=[display_df[col] for col in display_df.columns],
                    fill_color='white'
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="💰 SMART MONEY FLOW ANALYSIS",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_positioning_heatmap(self, positioning: List) -> go.Figure:
        """Create dealer positioning heatmap"""
        if not positioning:
            return self._create_empty_chart("No positioning data available")
        
        # Create positioning matrix
        symbols = [pos.symbol for pos in positioning]
        metrics = ['Gamma Exposure', 'Delta Exposure', 'Max Pain Distance', 'Gamma Wall Distance']
        
        z_values = []
        for pos in positioning:
            # Get current price for distance calculations
            try:
                quote = self.schwab_client.get_quote(pos.symbol)
                current_price = quote.get(pos.symbol, {}).get('lastPrice', 1)
                
                max_pain_dist = ((pos.max_pain - current_price) / current_price) * 100 if pos.max_pain else 0
                gamma_wall_dist = ((pos.gamma_wall - current_price) / current_price) * 100 if pos.gamma_wall else 0
                
                z_values.append([
                    pos.dealer_gamma_exposure / 1e9,  # In billions
                    pos.dealer_delta_exposure / 1e6,  # In millions  
                    max_pain_dist,  # Percentage
                    gamma_wall_dist  # Percentage
                ])
            except Exception:
                z_values.append([0, 0, 0, 0])
        
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=metrics,
            y=symbols,
            colorscale='RdYlGn',
            text=[[f"{val:.1f}" for val in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🎯 DEALER POSITIONING HEATMAP",
            xaxis_title="Metrics",
            yaxis_title="Symbols",
            height=400
        )
        
        return fig
    
    def _create_opportunities_chart(self, opportunities: List) -> go.Figure:
        """Create trading opportunities chart"""
        if not opportunities:
            return self._create_empty_chart("No trading opportunities identified")
        
        # Create opportunities dataframe
        data = []
        for opp in opportunities[:10]:  # Top 10
            data.append({
                'Symbol': opp.symbol,
                'Signal': opp.signal.value,
                'Strategy': opp.strategy,
                'R/R': opp.risk_reward,
                'Probability': opp.probability,
                'Time': opp.time_horizon,
                'Catalyst': opp.catalyst,
                'Size %': f"{opp.position_size*100:.1f}%",
                'Confidence': opp.confidence,
                'Score': opp.confidence * opp.risk_reward  # Combined score
            })
        
        df = pd.DataFrame(data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Risk/Reward vs Probability", "Strategy Distribution",
                "Top Opportunities", "Position Sizing"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "table"}, {"type": "bar"}]
            ]
        )
        
        # Risk/Reward vs Probability scatter
        colors = ['green' if 'BUY' in signal else 'red' if 'SELL' in signal else 'blue' 
                 for signal in df['Signal']]
        
        fig.add_trace(
            go.Scatter(
                x=df['Probability'],
                y=df['R/R'],
                mode='markers+text',
                text=df['Symbol'],
                textposition='top center',
                marker=dict(
                    size=df['Confidence'] * 30,  # Size by confidence
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                name='Opportunities'
            ),
            row=1, col=1
        )
        
        # Strategy distribution pie
        strategy_counts = df['Strategy'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=strategy_counts.index,
                values=strategy_counts.values,
                name="Strategies"
            ),
            row=1, col=2
        )
        
        # Top opportunities table
        display_df = df[['Symbol', 'Signal', 'Strategy', 'R/R', 'Catalyst']].head(5)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color='gold'
                ),
                cells=dict(
                    values=[display_df[col] for col in display_df.columns],
                    fill_color='lightyellow'
                )
            ),
            row=2, col=1
        )
        
        # Position sizing
        fig.add_trace(
            go.Bar(
                x=df['Symbol'][:5],
                y=[float(x.strip('%')) for x in df['Size %'][:5]],
                marker_color='blue',
                name='Position Size %'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="🎯 TRADING OPPORTUNITIES ANALYSIS",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_sector_rotation_chart(self, sector_rotation: Dict[str, float]) -> go.Figure:
        """Create sector rotation analysis"""
        if not sector_rotation:
            return self._create_empty_chart("No sector rotation data available")
        
        # Create sector mapping
        sector_names = {
            'XLF': 'Financials', 'XLK': 'Technology', 'XLE': 'Energy',
            'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLU': 'Utilities',
            'XLP': 'Staples', 'XLRE': 'Real Estate', 'XLB': 'Materials', 'XLY': 'Discretionary'
        }
        
        symbols = list(sector_rotation.keys())
        names = [sector_names.get(sym, sym) for sym in symbols]
        performance = list(sector_rotation.values())
        
        # Color based on performance
        colors = ['darkgreen' if p > 1 else 'green' if p > 0.5 else 'lightgreen' if p > 0 
                 else 'lightcoral' if p > -0.5 else 'red' if p > -1 else 'darkred' 
                 for p in performance]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=performance,
                marker_color=colors,
                text=[f"{p:+.1f}%" for p in performance],
                textposition='auto',
                name='Daily Performance'
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="🔄 SECTOR ROTATION HEATMAP",
            xaxis_title="Sector",
            yaxis_title="Daily Performance (%)",
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_risk_monitor_chart(self, intel: MarketIntelligence) -> go.Figure:
        """Create risk monitoring dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "VIX Term Structure", "Put/Call Skew",
                "Risk Factors", "Market Stress Indicators"
            ]
        )
        
        # VIX level indicator (simplified)
        try:
            vix_data = self.schwab_client.get_quote('VIX')
            vix_level = vix_data.get('VIX', {}).get('lastPrice', 20)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=vix_level,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "VIX Level"},
                    gauge={
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "red" if vix_level > 25 else "yellow" if vix_level > 20 else "green"},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgreen"},
                            {'range': [15, 25], 'color': "lightyellow"},
                            {'range': [25, 50], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                ),
                row=1, col=1
            )
        except Exception:
            pass
        
        # Risk factors
        if intel.risk_factors:
            fig.add_trace(
                go.Table(
                    header=dict(values=['⚠️ Risk Factors']),
                    cells=dict(values=[intel.risk_factors])
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="⚠️ RISK MONITORING DASHBOARD",
            height=600
        )
        
        return fig
    
    def _create_gamma_wall_chart(self, positioning: List) -> go.Figure:
        """Create gamma wall and key levels chart"""
        if not positioning:
            return self._create_empty_chart("No positioning data for gamma walls")
        
        fig = go.Figure()
        
        for pos in positioning[:5]:  # Top 5 symbols
            try:
                # Get current price
                quote = self.schwab_client.get_quote(pos.symbol)
                current_price = quote.get(pos.symbol, {}).get('lastPrice', 0)
                
                if not current_price:
                    continue
                
                # Create levels chart
                levels = []
                labels = []
                colors = []
                
                # Add current price
                levels.append(current_price)
                labels.append(f"{pos.symbol} Current")
                colors.append('blue')
                
                # Add gamma wall
                if pos.gamma_wall:
                    levels.append(pos.gamma_wall)
                    labels.append(f"{pos.symbol} Gamma Wall")
                    colors.append('purple')
                
                # Add max pain
                if pos.max_pain:
                    levels.append(pos.max_pain)
                    labels.append(f"{pos.symbol} Max Pain")
                    colors.append('orange')
                
                # Add support/resistance
                for level in pos.support_levels:
                    levels.append(level)
                    labels.append(f"{pos.symbol} Support")
                    colors.append('green')
                
                for level in pos.resistance_levels:
                    levels.append(level)
                    labels.append(f"{pos.symbol} Resistance")
                    colors.append('red')
                
                if levels:
                    fig.add_trace(
                        go.Scatter(
                            x=labels,
                            y=levels,
                            mode='markers',
                            marker=dict(
                                size=12,
                                color=colors,
                                symbol='diamond'
                            ),
                            name=pos.symbol,
                            text=[f"${level:.2f}" for level in levels],
                            textposition='top center'
                        )
                    )
                    
            except Exception as e:
                logger.warning(f"Error creating gamma wall chart for {pos.symbol}: {e}")
                continue
        
        fig.update_layout(
            title="🎯 GAMMA WALLS & KEY LEVELS",
            xaxis_title="Level Type",
            yaxis_title="Price Level ($)",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_overnight_setup_chart(self, intel: MarketIntelligence) -> go.Figure:
        """Create overnight setup guidance"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"<b>OVERNIGHT SETUP</b><br><br>{intel.overnight_setup}<br><br>" +
                 f"<b>Market Regime:</b> {intel.market_regime.value.replace('_', ' ').title()}<br>" +
                 f"<b>VIX Environment:</b> {intel.vix_regime}<br><br>" +
                 "<b>Key Levels to Watch:</b><br>" +
                 "• SPY: Monitor key option strikes<br>" +
                 "• VIX: Watch for overnight moves<br>" +
                 "• Futures: ES, NQ for direction",
            showarrow=False,
            font=dict(size=14),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title="🌙 OVERNIGHT POSITIONING GUIDE",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def _create_fallback_charts(self) -> Dict[str, go.Figure]:
        """Create fallback charts when intelligence engine is not available"""
        return {
            'market_overview': self._create_fallback_overview(),
            'smart_money_flow': self._create_empty_chart("Smart money flow analysis unavailable - Schwab API not connected"),
            'positioning_heatmap': self._create_empty_chart("Positioning data unavailable - Schwab API not connected"),
            'trading_opportunities': self._create_empty_chart("Trading opportunities unavailable - Analysis engine not initialized"),
            'sector_rotation': self._create_empty_chart("Sector rotation data unavailable - Schwab API not connected"),
            'risk_monitor': self._create_empty_chart("Risk monitoring unavailable - Market data not available"),
            'gamma_wall_levels': self._create_empty_chart("Gamma wall analysis unavailable - Options data not available"),
            'overnight_setup': self._create_fallback_overnight_setup()
        }
    
    def _create_fallback_overview(self) -> go.Figure:
        """Create fallback market overview"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="<b>🎯 PROFESSIONAL TRADER DASHBOARD</b><br><br>" +
                 "Dashboard is initializing...<br><br>" +
                 "<b>Features Available:</b><br>" +
                 "• Market regime analysis<br>" +
                 "• Smart money flow tracking<br>" +
                 "• Trading opportunity identification<br>" +
                 "• Risk monitoring<br>" +
                 "• Gamma wall levels<br>" +
                 "• Overnight positioning guides<br><br>" +
                 "<i>Please ensure Schwab API is configured</i>",
            showarrow=False,
            font=dict(size=14),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        fig.update_layout(
            title="🎯 MARKET INTELLIGENCE OVERVIEW",
            height=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def _create_fallback_overnight_setup(self) -> go.Figure:
        """Create fallback overnight setup"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="<b>🌙 OVERNIGHT POSITIONING GUIDE</b><br><br>" +
                 "Professional trader setup requires:<br><br>" +
                 "1. Schwab API configuration<br>" +
                 "2. Market data connection<br>" +
                 "3. Options chain access<br><br>" +
                 "<b>Next Steps:</b><br>" +
                 "• Configure Schwab API credentials<br>" +
                 "• Ensure market data access<br>" +
                 "• Restart dashboard for full features",
            showarrow=False,
            font=dict(size=14),
            align="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        fig.update_layout(
            title="🌙 OVERNIGHT POSITIONING GUIDE",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
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
            yaxis=dict(visible=False)
        )
        return fig