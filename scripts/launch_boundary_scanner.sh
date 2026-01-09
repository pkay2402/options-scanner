#!/bin/bash
# Quick Start Script for Reflecting Boundaries Scanner

echo "🎯 Reflecting Boundaries Scanner - Quick Start"
echo "=============================================="
echo ""

# Check if running from correct directory
if [ ! -f "boundary_scanner.py" ]; then
    echo "❌ Error: boundary_scanner.py not found"
    echo "Please run this script from the /schwab/options directory"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check/install dependencies
echo "📦 Checking dependencies..."
pip3 install -q streamlit yfinance pandas numpy plotly requests 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed"
else
    echo "⚠️  Some dependencies may need manual installation"
fi
echo ""

# Check for alert config
if [ ! -f "alerts_config.json" ]; then
    if [ -f "alerts_config.json.template" ]; then
        echo "📋 Creating alerts_config.json from template..."
        cp alerts_config.json.template alerts_config.json
        echo "✅ Alert config created (disabled by default)"
        echo "   Edit alerts_config.json to enable notifications"
    fi
else
    echo "✅ Alert config found"
fi
echo ""

# Check for Schwab API
if [ -f "src/api/schwab_client.py" ]; then
    echo "🔌 Schwab API integration available"
    if [ -f "schwab_client.json" ]; then
        echo "✅ Schwab tokens found"
    else
        echo "⚠️  Schwab not authenticated (will use yfinance)"
        echo "   Run: python scripts/auth_setup.py to authenticate"
    fi
else
    echo "📊 Using yfinance for data (Schwab API not found)"
fi
echo ""

echo "=============================================="
echo "🚀 Starting Boundary Scanner..."
echo ""
echo "The scanner will open in your browser automatically."
echo "If not, navigate to: http://localhost:8501"
echo ""
echo "Quick Tips:"
echo "  • Start with SPY, QQQ, or IWM for market signals"
echo "  • Try 2y or 5y lookback for more data"
echo "  • Enable backtest to see historical performance"
echo "  • Signals are RARE - that's why they work!"
echo ""
echo "Press Ctrl+C to stop the scanner"
echo "=============================================="
echo ""

# Launch streamlit
streamlit run boundary_scanner.py
