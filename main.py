"""
Main Application Entry Point
Options Trading Platform - Comprehensive analysis and monitoring
"""

import asyncio
import threading
import time
import argparse
import logging
from datetime import datetime
from typing import List, Optional

from src.api.schwab_client import SchwabClient
from src.analysis.market_dynamics import MarketDynamicsAnalyzer, IndividualStockAnalyzer
from src.analysis.big_trades import BigTradesDetector
from src.monitoring.options_flow import OptionsFlowMonitor, console_alert_handler, log_alert_handler
from src.visualization.dashboard import OptionsVisualization
from src.data.database import DatabaseManager
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

class OptionsTrader:
    """
    Main application class for the Options Trading Platform
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.schwab_client = None
        self.market_analyzer = None
        self.stock_analyzer = None
        self.big_trades_detector = None
        self.monitor = None
        self.dashboard = None
        self.db_manager = DatabaseManager()
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=getattr(logging, self.settings.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.settings.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Options Trading Platform...")
        
        try:
            # Initialize Schwab client
            self.schwab_client = SchwabClient()
            logger.info("✓ Schwab client initialized")
            
            # Initialize analyzers
            self.market_analyzer = MarketDynamicsAnalyzer(self.schwab_client)
            self.stock_analyzer = IndividualStockAnalyzer(self.schwab_client)
            self.big_trades_detector = BigTradesDetector(self.schwab_client)
            logger.info("✓ Analysis modules initialized")
            
            # Initialize monitoring
            self.monitor = OptionsFlowMonitor()
            self.monitor.add_alert_handler(console_alert_handler)
            self.monitor.add_alert_handler(log_alert_handler)
            logger.info("✓ Monitoring system initialized")
            
            # Initialize dashboard
            self.dashboard = OptionsVisualization(
                port=self.settings.DASHBOARD_PORT
            )
            logger.info("✓ Dashboard initialized")
            
            logger.info("🚀 Platform initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def run_analysis(self, symbols: List[str] = None, analysis_type: str = "both"):
        """
        Run market analysis with actionable insights
        """
        if not self.market_analyzer:
            logger.error("Platform not initialized. Call initialize() first.")
            return
        
        logger.info(f"Running {analysis_type} analysis...")
        
        try:
            results = {}
            
            if analysis_type in ["short", "both"]:
                logger.info("Running short-term analysis...")
                short_term = self.market_analyzer.analyze_short_term_dynamics(symbols)
                results['short_term'] = short_term
                
                # Print comprehensive actionable analysis
                print(f"\n" + "="*80)
                print(f"📊 SHORT-TERM ANALYSIS")
                print(f"="*80)
                print(f"📈 Put/Call Ratio: {short_term.sentiment.put_call_ratio:.2f}")
                print(f"📊 VIX Level: {short_term.sentiment.vix_level:.1f}")
                print(f"🎯 Sentiment Score: {short_term.sentiment.sentiment_score:.2f}")
                print(f"✅ Confidence: {short_term.confidence_score:.1%}")
                
                print(f"\n� ACTIONABLE TRADE SETUPS:")
                print(f"-"*50)
                for rec in short_term.recommendations:
                    print(f"{rec}")
                
                if short_term.risk_factors:
                    print(f"\n⚠️  RISK MANAGEMENT:")
                    print(f"-"*50)
                    for risk in short_term.risk_factors:
                        print(f"{risk}")
            
            if analysis_type in ["mid", "both"]:
                logger.info("Running mid-term analysis...")
                mid_term = self.market_analyzer.analyze_mid_term_dynamics(symbols)
                results['mid_term'] = mid_term
                
                # Print comprehensive mid-term analysis
                print(f"\n" + "="*80)
                print(f"📈 MID-TERM ANALYSIS")
                print(f"="*80)
                print(f"📈 Put/Call Ratio: {mid_term.sentiment.put_call_ratio:.2f}")
                print(f"🎯 Gamma Exposure: {mid_term.sentiment.gamma_exposure:.2f}B")
                print(f"🏦 Dealer Positioning: {mid_term.sentiment.dealer_positioning}")
                print(f"✅ Confidence: {mid_term.confidence_score:.1%}")
                
                print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
                print(f"-"*50)
                for rec in mid_term.recommendations:
                    print(f"{rec}")
                
                if mid_term.risk_factors:
                    print(f"\n⚠️  RISK ASSESSMENT:")
                    print(f"-"*50)
                    for risk in mid_term.risk_factors:
                        print(f"{risk}")
            
            print(f"\n" + "="*80)
            print(f"📞 TRADING HOTLINE: Monitor these setups for execution")
            print(f"⏰ NEXT UPDATE: Check again in 15-30 minutes for changes")
            print(f"="*80)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return None
    
    def scan_big_trades(self, symbols: List[str] = None, min_premium: float = None):
        """
        Scan for big trades
        """
        if not self.big_trades_detector:
            logger.error("Platform not initialized. Call initialize() first.")
            return
        
        logger.info("Scanning for big trades...")
        
        try:
            big_trades = self.big_trades_detector.scan_for_big_trades(symbols, min_premium)
            
            if not big_trades:
                print("No big trades found.")
                return []
            
            print(f"\n💰 FOUND {len(big_trades)} BIG TRADES")
            print("=" * 80)
            
            for i, trade in enumerate(big_trades[:10], 1):  # Show top 10
                print(f"\n{i}. {trade.symbol} {trade.contract_type.upper()} ${trade.strike}")
                print(f"   Expiration: {trade.expiration}")
                print(f"   Volume: {trade.volume:,} contracts")
                print(f"   Premium: ${trade.premium * trade.volume * 100:,.0f}")
                print(f"   Notional: ${trade.notional_value:,.0f}")
                print(f"   Sentiment: {trade.sentiment}")
                print(f"   Size Score: {trade.size_score:.1f}/10")
                
                if trade.analysis_notes:
                    print(f"   Note: {trade.analysis_notes[0]}")
            
            return big_trades
            
        except Exception as e:
            logger.error(f"Error scanning big trades: {str(e)}")
            return []
    
    def analyze_stock(self, symbol: str):
        """
        Analyze individual stock
        """
        if not self.stock_analyzer:
            logger.error("Platform not initialized. Call initialize() first.")
            return
        
        logger.info(f"Analyzing {symbol}...")
        
        try:
            analysis = self.stock_analyzer.analyze_stock(symbol)
            
            if 'error' in analysis:
                print(f"❌ Error analyzing {symbol}: {analysis['error']}")
                return None
            
            print(f"\n📊 INDIVIDUAL STOCK ANALYSIS: {symbol}")
            print("=" * 50)
            
            # Options metrics
            metrics = analysis.get('options_metrics', {})
            if metrics:
                print(f"Total Volume: {metrics.get('total_volume', 0):,}")
                print(f"Total OI: {metrics.get('total_oi', 0):,}")
                print(f"Put/Call Volume Ratio: {metrics.get('put_call_volume_ratio', 0):.2f}")
                print(f"Average IV: {metrics.get('avg_iv', 0):.1%}")
                print(f"Max Pain: ${metrics.get('max_pain', 0):.0f}")
            
            # Sentiment
            sentiment = analysis.get('sentiment', {})
            if sentiment:
                print(f"\nSentiment: {sentiment.get('sentiment', 'neutral')}")
                print(f"Call Ratio: {sentiment.get('call_ratio', 0):.1%}")
                print(f"Confidence: {sentiment.get('confidence', 0):.1%}")
            
            # Unusual activity
            unusual = analysis.get('unusual_activity', [])
            if unusual:
                print(f"\n⚡ Top Unusual Activity:")
                for i, activity in enumerate(unusual[:3], 1):
                    print(f"  {i}. {activity.contract_type.upper()} ${activity.strike} "
                          f"Vol: {activity.volume:,} IV: {activity.implied_volatility:.1%}")
            
            # Recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                print(f"\n📋 Recommendations:")
                for rec in recommendations:
                    print(f"  • {rec}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            return None
    
    def start_monitoring(self, symbols: List[str] = None):
        """
        Start real-time monitoring
        """
        if not self.monitor:
            logger.error("Platform not initialized. Call initialize() first.")
            return
        
        logger.info("Starting real-time monitoring...")
        self.monitor.start_monitoring(symbols)
        
        try:
            print("\n🎯 REAL-TIME MONITORING ACTIVE")
            print("Press Ctrl+C to stop monitoring")
            print("=" * 50)
            
            while True:
                # Show performance metrics every 5 minutes
                time.sleep(300)
                metrics = self.monitor.get_performance_metrics()
                logger.info(f"Monitoring metrics: {metrics['success_rate']:.1f}% success rate, "
                           f"{metrics['total_alerts']} total alerts")
                
        except KeyboardInterrupt:
            print("\n\nStopping monitoring...")
            self.monitor.stop_monitoring()
            print("Monitoring stopped.")
    
    def start_dashboard(self, debug: bool = False):
        """
        Start professional trader web dashboard
        """
        if not self.dashboard:
            logger.error("Platform not initialized. Call initialize() first.")
            return
            
        print("\n🎯 Starting Professional Trader Dashboard...")
        print("📊 Dashboard starting at http://localhost:8050")
        print("💡 Professional trader intelligence with:")
        print("   • Market regime analysis")
        print("   • Smart money flow tracking") 
        print("   • Trading opportunity identification")
        print("   • Risk monitoring")
        print("   • Gamma wall levels")
        print("   • Overnight positioning guides")
        
        logger.info("Starting professional trader dashboard...")
        self.dashboard.run(debug=debug)
    
    def run_interactive(self):
        """
        Run interactive mode
        """
        print("\n🚀 OPTIONS TRADING PLATFORM - INTERACTIVE MODE")
        print("=" * 60)
        
        while True:
            print("\nAvailable commands:")
            print("1. analyze [symbol] - Analyze market or individual stock")
            print("2. big_trades - Scan for big trades")
            print("3. monitor - Start real-time monitoring")
            print("4. dashboard - Start web dashboard")
            print("5. status - Show system status")
            print("6. exit - Exit the application")
            
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == "exit":
                    break
                elif command.startswith("analyze"):
                    parts = command.split()
                    symbol = parts[1] if len(parts) > 1 else None
                    
                    if symbol:
                        self.analyze_stock(symbol.upper())
                    else:
                        self.run_analysis()
                        
                elif command == "big_trades":
                    self.scan_big_trades()
                    
                elif command == "monitor":
                    self.start_monitoring()
                    
                elif command == "dashboard":
                    self.start_dashboard()
                    
                elif command == "status":
                    self.show_status()
                    
                else:
                    print("Unknown command. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                logger.error(f"Error executing command: {str(e)}")
                print(f"Error: {str(e)}")
    
    def show_status(self):
        """
        Show system status
        """
        print("\n📊 SYSTEM STATUS")
        print("=" * 30)
        
        # Database stats
        try:
            db_stats = self.db_manager.get_database_stats()
            print(f"Options Data Records: {db_stats.get('options_data_count', 0):,}")
            print(f"Big Trades Records: {db_stats.get('big_trades_count', 0):,}")
            print(f"Unique Symbols: {db_stats.get('unique_symbols', 0):,}")
        except Exception as e:
            print(f"Database Status: Error - {str(e)}")
        
        # Monitoring status
        if self.monitor:
            if self.monitor.is_monitoring:
                metrics = self.monitor.get_performance_metrics()
                print(f"Monitoring: Active ({metrics['monitored_symbols']} symbols)")
                print(f"Success Rate: {metrics['success_rate']:.1f}%")
                print(f"Total Alerts: {metrics['total_alerts']}")
            else:
                print("Monitoring: Inactive")
        
        # Settings
        print(f"Update Interval: {self.settings.MARKET_DATA_UPDATE_INTERVAL}s")
        print(f"Real-time Data: {'Enabled' if self.settings.REAL_TIME_DATA_ENABLED else 'Disabled'}")
        print(f"Alerts: {'Enabled' if self.settings.ENABLE_ALERTS else 'Disabled'}")

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="Options Trading Platform")
    parser.add_argument("--mode", choices=["interactive", "analysis", "monitor", "dashboard"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--symbols", nargs="+", help="Symbols to analyze/monitor")
    parser.add_argument("--analysis-type", choices=["short", "mid", "both"], 
                       default="both", help="Analysis type")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and initialize application
    app = OptionsTrader()
    
    if not app.initialize():
        print("❌ Failed to initialize platform. Check logs for details.")
        return 1
    
    try:
        if args.mode == "interactive":
            app.run_interactive()
        elif args.mode == "analysis":
            app.run_analysis(args.symbols, args.analysis_type)
        elif args.mode == "monitor":
            app.start_monitoring(args.symbols)
        elif args.mode == "dashboard":
            app.start_dashboard(args.debug)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())