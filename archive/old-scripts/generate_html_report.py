#!/usr/bin/env python3
"""
Generate HTML Report for Options Trading Analysis
Simple solution to display analysis data in HTML format
"""

import sys
import os
sys.path.append('.')

from src.analysis.market_dynamics import MarketDynamicsAnalyzer
from src.api.schwab_client import SchwabClient
from datetime import datetime

def generate_html_report():
    """Generate a comprehensive HTML report with live analysis"""
    
    # Get live analysis
    client = SchwabClient()
    analyzer = MarketDynamicsAnalyzer(client)
    
    print("🔄 Fetching live market analysis...")
    short_term = analyzer.analyze_short_term_dynamics()
    mid_term = analyzer.analyze_mid_term_dynamics()
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Professional Options Trading Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .timestamp {{
            opacity: 0.8;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        .analysis-section {{
            padding: 30px;
            border-right: 1px solid #eee;
        }}
        .analysis-section:last-child {{
            border-right: none;
        }}
        .section-title {{
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
            display: flex;
            align-items: center;
        }}
        .analysis-content {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            overflow-x: auto;
            border-left: 5px solid #3498db;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }}
        .refresh-btn {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #3498db;
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 50px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
            transition: all 0.3s ease;
        }}
        .refresh-btn:hover {{
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(52, 152, 219, 0.4);
        }}
        @media (max-width: 768px) {{
            .content {{
                grid-template-columns: 1fr;
            }}
            .analysis-section {{
                border-right: none;
                border-bottom: 1px solid #eee;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Professional Options Trading Analysis</h1>
            <div class="timestamp">📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST</div>
        </div>
        
        <div class="content">
            <div class="analysis-section">
                <h2 class="section-title">📈 Short-term Analysis (1-3 days)</h2>
                <div class="analysis-content">{format_analysis_for_html(short_term)}</div>
            </div>
            
            <div class="analysis-section">
                <h2 class="section-title">📊 Mid-term Analysis (1-2 weeks)</h2>
                <div class="analysis-content">{format_analysis_for_html(mid_term)}</div>
            </div>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="window.location.reload()">🔄 Refresh Data</button>
    
    <script>
        // Auto-refresh every 60 seconds
        setTimeout(function() {{
            window.location.reload();
        }}, 60000);
    </script>
</body>
</html>
"""
    
    return html_content

def format_analysis_for_html(result):
    """Format MarketAnalysisResult for HTML display"""
    if result is None:
        return "No analysis available"
    
    formatted = []
    
    # Market Sentiment
    sentiment = result.sentiment
    formatted.append("🎯 MARKET SENTIMENT")
    formatted.append(f"   • Put/Call Ratio: {sentiment.put_call_ratio:.3f}")
    formatted.append(f"   • VIX Level: {sentiment.vix_level}")
    formatted.append(f"   • Gamma Exposure: {sentiment.gamma_exposure:.3f}")
    formatted.append(f"   • Dealer Position: {sentiment.dealer_positioning}")
    formatted.append(f"   • Sentiment Score: {sentiment.sentiment_score:.3f}")
    formatted.append("")
    
    # Actionable Recommendations
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
    
    # Key Levels
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
    
    return "\\n".join(formatted)

if __name__ == "__main__":
    print("🎯 Generating Professional Trading Analysis HTML Report...")
    
    try:
        html_content = generate_html_report()
        
        # Save to file
        output_file = "options_analysis_report.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated: {output_file}")
        print(f"🌐 Open in browser: file://{os.path.abspath(output_file)}")
        print("📊 Report includes:")
        print("   • Live market sentiment analysis")
        print("   • Actionable trading recommendations")
        print("   • Risk factors and position sizing")
        print("   • Key support/resistance levels")
        print("   • Professional trade setups")
        print("🔄 Auto-refreshes every 60 seconds")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()