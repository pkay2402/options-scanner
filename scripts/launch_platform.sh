#!/bin/bash

# Launch Options Trading Platform
# Multi-page Streamlit application

echo "🚀 Launching Options Trading Platform..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Launch the main dashboard
echo "✅ Starting application..."
echo "📊 Navigate to http://localhost:8501 in your browser"
echo ""

streamlit run Main_Dashboard.py
