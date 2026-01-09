#!/bin/bash
# Quick launch script for Options Scanner with network access

echo "🚀 Starting Options Scanner..."
echo ""

# Change to the project directory
cd "$(dirname "$0")"

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    echo "Please install Python 3"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed!"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

# Check token status
echo "📋 Checking API token status..."
python3 scripts/check_token.py
TOKEN_STATUS=$?

if [ $TOKEN_STATUS -ne 0 ]; then
    echo ""
    echo "⚠️  WARNING: Token issue detected!"
    echo "The app may not work correctly."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "✅ Starting Streamlit app..."
echo ""
echo "📱 Access from:"
echo "   • Your computer: http://localhost:8501"
echo "   • Same WiFi devices: http://$(ipconfig getifaddr en0 2>/dev/null || echo "YOUR-IP"):8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Streamlit with network access
streamlit run Main_Dashboard.py --server.address 0.0.0.0 --server.port 8501
