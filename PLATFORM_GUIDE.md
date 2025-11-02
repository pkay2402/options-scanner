# 📊 Options Trading Platform - Multi-Page Web Application

A professional, organized multi-page Streamlit application for options analysis and trading signals.

## 🎯 Quick Start

### Launch the Platform

```bash
./launch_platform.sh
```

Or manually:
```bash
streamlit run Main_Dashboard.py
```

Then navigate to: **http://localhost:8501**

## 🗂️ Platform Structure

```
📊 Options Trading Platform
│
├── 🏠 Main_Dashboard.py         # Landing page & overview
│
├── pages/                        # All tool pages
│   ├── 2_📈_Max_Gamma_Scanner.py
│   ├── 3_📍_Index_Positioning.py
│   ├── 4_🎯_Boundary_Scanner.py
│   ├── 5_🌊_Flow_Scanner.py
│   ├── 6_💎_Opportunity_Scanner.py
│   ├── 7_📊_Options_Flow_Monitor.py
│   ├── 8_⚡_Immediate_Dashboard.py
│   ├── 9_📰_Newsletter_Generator.py
│   ├── 10_📄_Report_Generator.py
│   └── 11_⚙️_Settings.py
│
├── components/                   # Shared UI components
│   └── ui_components.py
│
├── src/                         # Core functionality
│   ├── api/                     # Schwab API client
│   ├── analysis/                # Analysis modules
│   ├── monitoring/              # Flow monitoring
│   └── visualization/           # Charts & dashboards
│
└── assets/                      # Static assets
```

## 📈 Available Tools

### Market Analysis
- **Max Gamma Scanner** - Identify max gamma strikes and dealer positioning
- **Index Positioning** - SPY/SPX/QQQ key levels and gamma walls
- **Market Dynamics** - Short-term and mid-term market analysis

### Signal Detection
- **Boundary Scanner** - Berg methodology for turning points (6 buy + 3 sell signals)
- **Flow Scanner** - Unusual activity, block trades, sweeps
- **Opportunity Scanner** - Systematic trade setup identification

### Live Monitoring
- **Options Flow Monitor** - Real-time flow tracking
- **Immediate Dashboard** - Quick market snapshot

### Reports & Insights
- **Newsletter Generator** - Substack-style HTML newsletters
- **Report Generator** - Custom HTML report exports

### Configuration
- **Settings** - Schwab API, alerts, preferences, system info

## 🎨 Features

### ✅ Organized Navigation
- Single entry point (Main_Dashboard.py)
- Sidebar navigation to all tools
- Quick access buttons
- Breadcrumb navigation

### ✅ Consistent Design
- Custom CSS theme throughout
- Responsive layouts
- Dark mode optimized
- Professional styling

### ✅ Shared Components
- Reusable UI elements
- Consistent data tables
- Unified charts
- Common utilities

### ✅ Data Management
- Schwab API integration
- yfinance fallback
- Smart caching (60-300s)
- Auto-refresh options

### ✅ User Experience
- Fast page loads
- Intuitive navigation
- Mobile-friendly
- Export capabilities

## 🚀 Usage Workflow

### Morning Routine
1. Open **Main Dashboard** for overview
2. Check **Index Positioning** for key levels
3. Run **Opportunity Scanner** for trade setups
4. Monitor **Options Flow** for institutional activity

### Signal Detection
1. **Boundary Scanner** for turning points
2. **Flow Scanner** for unusual activity
3. **Max Gamma** for dealer positioning
4. Cross-reference signals

### End of Day
1. Generate **Newsletter** with daily summary
2. Export **Reports** for record keeping
3. Review performance metrics
4. Set up alerts for next session

## ⚙️ Configuration

### Schwab API Setup
1. Navigate to **Settings** page
2. Check API connection status
3. Re-authenticate if needed (every 7 days)
4. Test connection

### Alert Setup
1. Go to **Settings > Alerts**
2. Enable email/webhook notifications
3. Configure SMTP settings
4. Select trigger types
5. Save configuration

### Preferences
1. **Settings > Preferences**
2. Set default watchlist
3. Configure display options
4. Adjust cache settings
5. Save preferences

## 📊 Key Metrics

- **9 Analysis Tools** - Comprehensive coverage
- **6 Buy Signal Types** - Bottom boundary detection
- **3 Sell Signal Types** - Top boundary warnings
- **25+ Symbols** - Default watchlist
- **Real-time Data** - Live market feeds

## 🔧 Technical Details

### Requirements
```bash
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
yfinance>=0.2.28
```

### Data Sources
- **Primary**: Schwab API (requires authentication)
- **Fallback**: yfinance (free, no auth required)
- **Historical**: Up to 10 years available

### Performance
- **Caching**: 60-300 second TTL
- **Parallel Fetching**: Multi-symbol support
- **Background Processing**: Available for long-running tasks

## 📝 Notes

### Original Files Preserved
All original Python scripts remain in the root directory:
- `max_gamma_scanner.py`
- `boundary_scanner.py`
- `flow_scanner.py`
- `opportunity_scanner.py`
- etc.

These are **copied** (not moved) to the `pages/` directory, so you can still run them standalone if needed.

### Page Numbering
Pages are numbered (2_, 3_, 4_, etc.) to control display order in sidebar. Page 1 is reserved for the main dashboard.

### Navigation
Use the sidebar or Quick Access buttons to navigate between pages. Streamlit automatically handles multi-page routing.

## 🎯 Next Steps

1. **Launch the platform**: `./launch_platform.sh`
2. **Configure Schwab API**: Settings > Schwab API
3. **Set up alerts**: Settings > Alerts
4. **Customize watchlist**: Settings > Preferences
5. **Start scanning**: Navigate to any tool page

## 💡 Tips

- **Combine tools**: Use multiple scanners for signal confirmation
- **Focus on boundaries**: Best signals occur at extremes
- **Use backtesting**: Validate strategies before trading
- **Set up alerts**: Never miss important triggers
- **Export data**: Keep records of all analysis

## ⚠️ Disclaimer

This platform is for educational purposes only. Not financial advice. Always conduct your own research and consult with licensed professionals before making trading decisions.

---

**Version**: 1.0.0  
**Built with**: Streamlit Multi-Page Apps  
**Platform**: Python 3.11+
