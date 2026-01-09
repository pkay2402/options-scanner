#!/usr/bin/env python3
"""
Options Opportunity Newsletter Generator
Creates beautiful Substack-style newsletters from scanner results
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append('.')

# Import the scanner logic
from opportunity_scanner import (
    scan_opportunities, 
    DEFAULT_WATCHLIST,
    get_options_data,
    analyze_gamma_squeeze_setup,
    analyze_momentum_flow,
    analyze_volatility_play,
    analyze_reversal_setup
)

# Configure Streamlit page
st.set_page_config(
    page_title="Options Newsletter Generator",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_newsletter_html(opportunities, scan_date):
    """Generate beautiful HTML newsletter"""
    
    # Count setup types
    gamma_setups = [o for o in opportunities if 'GAMMA' in o['type']]
    momentum_setups = [o for o in opportunities if 'MOMENTUM' in o['type']]
    volatility_setups = [o for o in opportunities if 'VOLATILITY' in o['type']]
    reversal_setups = [o for o in opportunities if 'REVERSAL' in o['type']]
    
    # Calculate summary stats
    total_opportunities = len(opportunities)
    bullish_count = len([o for o in opportunities if o.get('direction') == 'BULLISH' or 'GAMMA' in o['type']])
    bearish_count = len([o for o in opportunities if o.get('direction') == 'BEARISH'])
    
    # Get top symbols
    top_symbols = list(set([o['symbol'] for o in opportunities[:5]]))
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Options Trade Opportunities - {scan_date}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 680px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            color: #667eea;
            font-size: 2.5em;
            font-weight: 700;
        }}
        .header .subtitle {{
            color: #6c757d;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .header .date {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .intro {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .intro h2 {{
            margin-top: 0;
            font-size: 1.5em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0 30px 0;
        }}
        .stat-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .opportunity {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
        }}
        .opportunity-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f1f3f5;
        }}
        .opportunity-title {{
            font-size: 1.8em;
            font-weight: bold;
            color: #1a1a1a;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge-gamma {{
            background-color: #9b59b6;
            color: white;
        }}
        .badge-momentum {{
            background-color: #3498db;
            color: white;
        }}
        .badge-volatility {{
            background-color: #e74c3c;
            color: white;
        }}
        .badge-reversal {{
            background-color: #f39c12;
            color: white;
        }}
        .badge-bullish {{
            background-color: #2ecc71;
            color: white;
        }}
        .badge-bearish {{
            background-color: #e74c3c;
            color: white;
        }}
        .badge-high {{
            background-color: #2ecc71;
            color: white;
        }}
        .badge-medium {{
            background-color: #f39c12;
            color: white;
        }}
        .opportunity-content {{
            margin: 15px 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #1a1a1a;
        }}
        .rationale {{
            background: #fff9e6;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .trade-suggestion {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .trade-suggestion h4 {{
            margin-top: 0;
        }}
        .trade-suggestion ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .section-divider {{
            border: 0;
            height: 2px;
            background: linear-gradient(to right, transparent, #dee2e6, transparent);
            margin: 40px 0;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-top: 40px;
        }}
        .summary h2 {{
            margin-top: 0;
            font-size: 1.8em;
        }}
        .summary-section {{
            margin: 20px 0;
        }}
        .summary-section h3 {{
            font-size: 1.2em;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
            padding-bottom: 5px;
        }}
        .summary-list {{
            list-style: none;
            padding: 0;
        }}
        .summary-list li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        .summary-list li:last-child {{
            border-bottom: none;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
            color: #6c757d;
            font-size: 0.9em;
        }}
        .cta {{
            background: #667eea;
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            text-align: center;
            margin: 30px 0;
            font-weight: bold;
        }}
        .disclaimer {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
            font-size: 0.85em;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 Options Trade Opportunities</h1>
            <div class="subtitle">Smart Options Scanner Daily Digest</div>
            <div class="date">{scan_date}</div>
        </div>

        <!-- Introduction -->
        <div class="intro">
            <h2>Today's Market Scan Results</h2>
            <p>We've analyzed options flow, Greek positioning, and institutional activity across the most liquid stocks. Here are today's top opportunities ranked by probability and risk/reward.</p>
        </div>

        <!-- Stats Overview -->
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-number">{total_opportunities}</div>
                <div class="stat-label">Total Opportunities</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{bullish_count}</div>
                <div class="stat-label">Bullish Setups</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{bearish_count}</div>
                <div class="stat-label">Bearish Setups</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(top_symbols)}</div>
                <div class="stat-label">Top Symbols</div>
            </div>
        </div>

        <hr class="section-divider">

        <!-- Opportunities -->
"""

    # Add each opportunity
    for idx, opp in enumerate(opportunities, 1):
        symbol = opp['symbol']
        current_price = opp['current_price']
        confidence = opp['confidence']
        setup_type = opp['type']
        
        # Determine setup details
        if 'GAMMA' in setup_type:
            setup_name = 'Gamma Squeeze Setup'
            badge_class = 'badge-gamma'
            direction_badge = ''
        elif 'MOMENTUM' in setup_type:
            direction = opp.get('direction', 'BULLISH')
            setup_name = f'{direction.title()} Momentum Flow'
            badge_class = 'badge-momentum'
            direction_badge = f'<span class="badge badge-{direction.lower()}">{direction}</span>'
        elif 'VOLATILITY' in setup_type:
            setup_name = 'Volatility Expansion Play'
            badge_class = 'badge-volatility'
            direction_badge = ''
        else:
            direction = opp.get('direction', 'NEUTRAL')
            setup_name = f'{direction.title()} Reversal Setup'
            badge_class = 'badge-reversal'
            direction_badge = f'<span class="badge badge-{direction.lower()}">{direction}</span>'
        
        conf_badge = f'<span class="badge badge-{confidence.lower()}">{confidence} CONFIDENCE</span>'
        
        html += f"""
        <div class="opportunity">
            <div class="opportunity-header">
                <div>
                    <div class="opportunity-title">#{idx} - {symbol} @ ${current_price:.2f}</div>
                    <div style="margin-top: 10px;">
                        <span class="badge {badge_class}">{setup_name}</span>
                        {direction_badge}
                        {conf_badge}
                    </div>
                </div>
            </div>
            <div class="opportunity-content">
"""
        
        # Add setup-specific content
        if 'GAMMA' in setup_type:
            html += f"""
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Target Strike</div>
                        <div class="metric-value">${opp['target_strike']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Upside Potential</div>
                        <div class="metric-value">+{opp['upside_potential']:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Call OI Above</div>
                        <div class="metric-value">{opp['call_oi_above']:,.0f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Call/Put Ratio</div>
                        <div class="metric-value">{opp['ratio']:.2f}x</div>
                    </div>
                </div>
                <div class="rationale">
                    <strong>Setup Rationale:</strong> Heavy call open interest ({opp['call_oi_above']:,.0f} contracts) concentrated above current price at ${opp['target_strike']:.2f}. 
                    As price approaches this level, dealers must buy underlying to hedge, creating upward pressure. 
                    Call/Put ratio of {opp['ratio']:.2f}x indicates strong bullish positioning.
                </div>
                <div class="trade-suggestion">
                    <h4>💡 Suggested Trade</h4>
                    <ul>
                        <li>Buy {symbol} ${opp['target_strike']:.0f}C (near-term expiry)</li>
                        <li>Or use debit call spread: Long ${opp['target_strike']:.0f}C / Short ${opp['target_strike'] * 1.05:.0f}C</li>
                        <li>Target: ${opp['target_strike']:.2f} ({opp['upside_potential']:.1f}% gain)</li>
                        <li>Risk: Price fails to reach gamma wall, time decay</li>
                    </ul>
                </div>
"""
        
        elif 'MOMENTUM' in setup_type:
            direction = opp['direction']
            html += f"""
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Direction</div>
                        <div class="metric-value">{direction}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Dominant Premium</div>
                        <div class="metric-value">${opp['dominant_premium']/1e6:.1f}M</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">C/P Volume Ratio</div>
                        <div class="metric-value">{opp['cp_ratio']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Volume</div>
                        <div class="metric-value">{opp['call_volume'] + opp['put_volume']:,.0f}</div>
                    </div>
                </div>
                <div class="rationale">
                    <strong>Setup Rationale:</strong> Strong {direction.lower()} flow detected with ${opp['dominant_premium']/1e6:.1f}M in premium. 
                    Volume bias ({opp['call_volume']:,.0f} calls vs {opp['put_volume']:,.0f} puts) suggests directional conviction.
                    Institutional money is positioning for continued {direction.lower()} move.
                </div>
"""
            
            if direction == 'BULLISH':
                html += f"""
                <div class="trade-suggestion">
                    <h4>💡 Suggested Trade</h4>
                    <ul>
                        <li>Buy ATM/slightly OTM calls (1-2 weeks out)</li>
                        <li>Target strike: ${current_price * 1.05:.0f} - ${current_price * 1.10:.0f}</li>
                        <li>Or use bull call spread to reduce cost</li>
                        <li>Stop loss: ${current_price * 0.97:.2f} (3% below entry)</li>
                    </ul>
                </div>
"""
            else:
                html += f"""
                <div class="trade-suggestion">
                    <h4>💡 Suggested Trade</h4>
                    <ul>
                        <li>Buy ATM/slightly OTM puts (1-2 weeks out)</li>
                        <li>Target strike: ${current_price * 0.95:.0f} - ${current_price * 0.90:.0f}</li>
                        <li>Or use bear put spread to reduce cost</li>
                        <li>Stop loss: ${current_price * 1.03:.2f} (3% above entry)</li>
                    </ul>
                </div>
"""
        
        elif 'VOLATILITY' in setup_type:
            html += f"""
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Current IV</div>
                        <div class="metric-value">{opp['current_iv']:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">ATM Straddle</div>
                        <div class="metric-value">${opp['straddle_price']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Expected Move</div>
                        <div class="metric-value">±{opp['expected_move']:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">ATM Strike</div>
                        <div class="metric-value">${opp['atm_strike']:.2f}</div>
                    </div>
                </div>
                <div class="rationale">
                    <strong>Setup Rationale:</strong> Implied volatility at {opp['current_iv']:.1f}% is relatively low, suggesting options are underpriced. 
                    Expected move of ±{opp['expected_move']:.1f}% priced into ATM straddle at ${opp['straddle_price']:.2f}.
                    Potential for volatility expansion on upcoming catalyst or market event.
                </div>
                <div class="trade-suggestion">
                    <h4>💡 Suggested Trade</h4>
                    <ul>
                        <li>Buy ATM straddle: Long ${opp['atm_strike']:.0f}C + Long ${opp['atm_strike']:.0f}P</li>
                        <li>Cost: ${opp['straddle_price']:.2f} per share (${opp['straddle_price'] * 100:.0f} per contract)</li>
                        <li>Breakevens: ${opp['atm_strike'] - opp['straddle_price']:.2f} / ${opp['atm_strike'] + opp['straddle_price']:.2f}</li>
                        <li>Profit if price moves beyond ±{opp['expected_move']:.1f}%</li>
                    </ul>
                </div>
"""
        
        else:  # REVERSAL
            direction = opp['direction']
            html += f"""
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Direction</div>
                        <div class="metric-value">{direction}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Premium Flow</div>
                        <div class="metric-value">${opp['premium_flow']/1e6:.1f}M</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Key Strike</div>
                        <div class="metric-value">${opp['key_strike']:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Unusual Volume</div>
                        <div class="metric-value">{opp['unusual_volume']:,.0f}</div>
                    </div>
                </div>
                <div class="rationale">
                    <strong>Setup Rationale:</strong> Detected unusual institutional activity with ${opp['premium_flow']/1e6:.1f}M flowing into {opp['option_type']}s.
                    High volume relative to open interest ({opp['unusual_volume']:,.0f} contracts) suggests new positioning.
                    {opp['num_unusual']} strikes show abnormal activity, indicating potential {direction.lower()} reversal.
                </div>
                <div class="trade-suggestion">
                    <h4>💡 Suggested Trade</h4>
                    <ul>
                        <li>Follow the institutional flow: {opp['option_type']}s near ${opp['key_strike']:.0f}</li>
                        <li>Direction: {direction}</li>
                        <li>Watch for continued flow in same direction</li>
                        <li>Risk management: Exit if flow reverses or price breaks key levels</li>
                    </ul>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        if idx < len(opportunities):
            html += '<hr class="section-divider">'
    
    # Add summary section
    html += f"""
        <!-- Summary Section -->
        <div class="summary">
            <h2>📋 Today's Trading Summary</h2>
            
            <div class="summary-section">
                <h3>Market Bias</h3>
                <p>Today's scan revealed <strong>{bullish_count} bullish</strong> and <strong>{bearish_count} bearish</strong> setups. 
                The market is showing {'predominantly bullish' if bullish_count > bearish_count else 'predominantly bearish' if bearish_count > bullish_count else 'mixed'} sentiment in options positioning.</p>
            </div>

            <div class="summary-section">
                <h3>Top Symbols to Watch</h3>
                <ul class="summary-list">
"""
    
    # Add top symbols with their setups
    symbol_setups = {}
    for opp in opportunities[:10]:
        symbol = opp['symbol']
        if symbol not in symbol_setups:
            symbol_setups[symbol] = []
        
        if 'GAMMA' in opp['type']:
            symbol_setups[symbol].append('Gamma Squeeze')
        elif 'MOMENTUM' in opp['type']:
            symbol_setups[symbol].append(f"{opp.get('direction', 'Neutral')} Momentum")
        elif 'VOLATILITY' in opp['type']:
            symbol_setups[symbol].append('Vol Play')
        else:
            symbol_setups[symbol].append(f"{opp.get('direction', 'Neutral')} Reversal")
    
    for symbol, setups in list(symbol_setups.items())[:5]:
        setup_str = ', '.join(setups[:2])
        html += f"                    <li><strong>{symbol}</strong>: {setup_str}</li>\n"
    
    html += """
                </ul>
            </div>

            <div class="summary-section">
                <h3>Setup Type Breakdown</h3>
                <ul class="summary-list">
"""
    
    if gamma_setups:
        html += f"                    <li><strong>🚀 Gamma Squeeze:</strong> {len(gamma_setups)} opportunities - Strong upside potential from dealer hedging</li>\n"
    if momentum_setups:
        bullish_momentum = [o for o in momentum_setups if o.get('direction') == 'BULLISH']
        bearish_momentum = [o for o in momentum_setups if o.get('direction') == 'BEARISH']
        html += f"                    <li><strong>📈 Momentum Flow:</strong> {len(momentum_setups)} opportunities ({len(bullish_momentum)} bullish, {len(bearish_momentum)} bearish)</li>\n"
    if volatility_setups:
        html += f"                    <li><strong>⚡ Volatility Plays:</strong> {len(volatility_setups)} opportunities - Low IV expansion setups</li>\n"
    if reversal_setups:
        html += f"                    <li><strong>🔄 Reversals:</strong> {len(reversal_setups)} opportunities - Institutional positioning shifts</li>\n"
    
    html += """
                </ul>
            </div>

            <div class="summary-section">
                <h3>Key Takeaways</h3>
                <ul class="summary-list">
"""
    
    # Generate smart takeaways
    if bullish_count > bearish_count * 1.5:
        html += "                    <li>Strong bullish sentiment across multiple symbols - consider bullish strategies</li>\n"
    elif bearish_count > bullish_count * 1.5:
        html += "                    <li>Bearish positioning dominant - protective strategies recommended</li>\n"
    else:
        html += "                    <li>Mixed market sentiment - selective opportunities in both directions</li>\n"
    
    if gamma_setups:
        html += f"                    <li>Gamma walls identified in {len(gamma_setups)} stocks - watch for explosive moves</li>\n"
    
    if len([o for o in opportunities if o.get('confidence') == 'HIGH']) >= 3:
        html += f"                    <li>{len([o for o in opportunities if o.get('confidence') == 'HIGH'])} high-confidence setups available today</li>\n"
    
    html += f"""
                    <li>Most active symbols: {', '.join(top_symbols[:5])}</li>
                    <li>Scan completed at {scan_date}</li>
                </ul>
            </div>

            <div class="summary-section">
                <h3>Trading Plan</h3>
                <p><strong>Best Opportunities:</strong> Focus on the top {min(3, len(opportunities))} ranked setups with HIGH confidence ratings.</p>
                <p><strong>Risk Management:</strong> Size positions appropriately, use stop losses, and don't chase trades.</p>
                <p><strong>Monitor:</strong> Watch for flow confirmation and price action at key levels throughout the day.</p>
            </div>
        </div>

        <!-- CTA -->
        <div class="cta">
            These opportunities are time-sensitive. Review the setups and execute your trading plan accordingly.
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This analysis is for educational and informational purposes only. Options trading involves substantial risk and is not suitable for all investors. 
            Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. 
            The information provided does not constitute investment advice, financial advice, trading advice, or any other sort of advice.
        </div>

        <!-- Footer -->
        <div class="footer">
            <p><strong>Options Opportunity Scanner</strong></p>
            <p>Powered by Smart Options Analysis | Generated on {scan_date}</p>
            <p style="margin-top: 15px; font-size: 0.85em;">
                © {datetime.now().year} Options Scanner. All rights reserved.
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def main():
    st.title("📰 Options Newsletter Generator")
    st.markdown("Create beautiful Substack-style newsletters from your scanner results")
    
    # Sidebar
    st.sidebar.header("Newsletter Settings")
    
    # Watchlist
    use_default = st.sidebar.checkbox("Use Default Watchlist", value=True)
    
    if use_default:
        symbols = DEFAULT_WATCHLIST
    else:
        custom_symbols = st.sidebar.text_area(
            "Custom Watchlist (one per line)",
            value="\n".join(DEFAULT_WATCHLIST[:10]),
            height=150
        )
        symbols = [s.strip().upper() for s in custom_symbols.split('\n') if s.strip()]
    
    # Newsletter options
    max_opportunities = st.sidebar.slider("Max Opportunities", 3, 15, 10)
    
    newsletter_title = st.sidebar.text_input(
        "Newsletter Title",
        value="Options Trade Opportunities"
    )
    
    # Generate button
    if st.sidebar.button("🔍 Generate Newsletter", type="primary"):
        st.cache_data.clear()
        
        with st.spinner("Scanning markets and generating newsletter..."):
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, symbol):
                progress_bar.progress(current / total)
                status_text.text(f"Scanning {symbol}... ({current}/{total})")
            
            # Scan
            opportunities = scan_opportunities(symbols, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if not opportunities:
                st.error("No opportunities found. Try adjusting your watchlist or filters.")
                return
            
            # Limit opportunities
            opportunities = opportunities[:max_opportunities]
            
            # Generate newsletter
            scan_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
            newsletter_html = generate_newsletter_html(opportunities, scan_date)
            
            # Store in session state
            st.session_state['newsletter_html'] = newsletter_html
            st.session_state['opportunities'] = opportunities
            st.session_state['scan_date'] = scan_date
            
            st.success(f"✅ Newsletter generated with {len(opportunities)} opportunities!")
    
    # Display newsletter
    if 'newsletter_html' in st.session_state:
        newsletter_html = st.session_state['newsletter_html']
        
        # Preview
        st.header("📧 Newsletter Preview")
        
        # Show HTML preview
        st.components.v1.html(newsletter_html, height=800, scrolling=True)
        
        st.markdown("---")
        
        # Download options
        st.header("📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download HTML
            st.download_button(
                label="📄 Download HTML",
                data=newsletter_html,
                file_name=f"options_newsletter_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )
        
        with col2:
            # Copy to clipboard helper
            st.markdown("""
            **To send as email:**
            1. Download HTML file
            2. Open in browser
            3. Copy all (Cmd/Ctrl + A)
            4. Paste into email client
            """)
        
        # Stats
        st.markdown("---")
        st.header("📊 Newsletter Stats")
        
        opportunities = st.session_state.get('opportunities', [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Opportunities", len(opportunities))
        
        with col2:
            bullish = len([o for o in opportunities if o.get('direction') == 'BULLISH' or 'GAMMA' in o['type']])
            st.metric("Bullish Setups", bullish)
        
        with col3:
            bearish = len([o for o in opportunities if o.get('direction') == 'BEARISH'])
            st.metric("Bearish Setups", bearish)
        
        with col4:
            high_conf = len([o for o in opportunities if o.get('confidence') == 'HIGH'])
            st.metric("High Confidence", high_conf)
    
    else:
        # Initial state
        st.info("👈 Configure settings and click **Generate Newsletter** to create your daily digest")
        
        st.markdown("""
        ### Newsletter Features:
        
        - **Professional Substack-style design** 📧
        - **Comprehensive opportunity breakdown** with metrics and analysis
        - **Trade suggestions** for each setup
        - **Executive summary** at the end with key takeaways
        - **Mobile-responsive** HTML format
        - **Ready to email** - just copy and paste
        
        ### Perfect for:
        - Daily trading newsletters
        - Team distribution
        - Personal trade journal
        - Client reports
        - Social media content
        """)

if __name__ == "__main__":
    main()
