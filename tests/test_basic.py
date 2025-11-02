"""
Basic tests for the Options Trading Platform
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import get_settings
from src.analysis.market_dynamics import MarketDynamicsAnalyzer, OptionsFlow, MarketSentiment
from src.analysis.big_trades import BigTradesDetector, BigTrade
from src.data.database import DatabaseManager

class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_get_settings(self):
        """Test settings retrieval"""
        settings = get_settings()
        self.assertIsNotNone(settings)
        self.assertTrue(hasattr(settings, 'DATABASE_URL'))
        self.assertTrue(hasattr(settings, 'LOG_LEVEL'))

class TestMarketDynamicsAnalyzer(unittest.TestCase):
    """Test market dynamics analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.analyzer = MarketDynamicsAnalyzer(self.mock_client)
    
    def test_calculate_market_sentiment_empty_data(self):
        """Test sentiment calculation with empty data"""
        sentiment = self.analyzer._calculate_market_sentiment([])
        self.assertIsInstance(sentiment, MarketSentiment)
        self.assertEqual(sentiment.put_call_ratio, 0.0)
        self.assertEqual(sentiment.sentiment_score, 0.0)
    
    def test_calculate_market_sentiment_with_data(self):
        """Test sentiment calculation with sample data"""
        # Create sample options flow data
        options_data = [
            OptionsFlow(
                symbol='SPY',
                timestamp=datetime.now(),
                contract_type='call',
                expiration='2024-01-19',
                strike=450.0,
                volume=1000,
                open_interest=5000,
                premium=5.0,
                bid=4.9,
                ask=5.1,
                delta=0.5,
                gamma=0.01,
                theta=-0.05,
                vega=0.1,
                implied_volatility=0.25
            ),
            OptionsFlow(
                symbol='SPY',
                timestamp=datetime.now(),
                contract_type='put',
                expiration='2024-01-19',
                strike=440.0,
                volume=800,
                open_interest=3000,
                premium=3.0,
                bid=2.9,
                ask=3.1,
                delta=-0.3,
                gamma=0.008,
                theta=-0.04,
                vega=0.08,
                implied_volatility=0.28
            )
        ]
        
        sentiment = self.analyzer._calculate_market_sentiment(options_data)
        
        self.assertIsInstance(sentiment, MarketSentiment)
        self.assertGreater(sentiment.put_call_ratio, 0)
        self.assertGreater(sentiment.confidence_level, 0)
        self.assertLessEqual(sentiment.confidence_level, 1.0)
    
    def test_identify_key_levels(self):
        """Test key levels identification"""
        options_data = [
            OptionsFlow(
                symbol='SPY',
                timestamp=datetime.now(),
                contract_type='call',
                expiration='2024-01-19',
                strike=450.0,
                volume=1000,
                open_interest=10000,  # High OI
                premium=5.0,
                bid=4.9,
                ask=5.1,
                delta=0.5,
                gamma=0.01,
                theta=-0.05,
                vega=0.1,
                implied_volatility=0.25
            )
        ]
        
        key_levels = self.analyzer._identify_key_levels(options_data)
        
        self.assertIsInstance(key_levels, dict)
        # Should contain max pain point for SPY
        self.assertIn('SPY_max_pain', key_levels)

class TestBigTradesDetector(unittest.TestCase):
    """Test big trades detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_client = Mock()
        self.detector = BigTradesDetector(self.mock_client)
    
    def test_calculate_size_score(self):
        """Test size score calculation"""
        # Test small trade
        small_score = self.detector._calculate_size_score(100, 50000, 1000000)
        self.assertGreaterEqual(small_score, 1.0)
        self.assertLessEqual(small_score, 10.0)
        
        # Test large trade
        large_score = self.detector._calculate_size_score(10000, 10000000, 100000000)
        self.assertGreater(large_score, small_score)
        self.assertLessEqual(large_score, 10.0)
    
    def test_calculate_urgency_score(self):
        """Test urgency score calculation"""
        # Test same-day expiration
        same_day_score = self.detector._calculate_urgency_score(0, 1.0, 1000, 500)
        
        # Test long-term expiration
        long_term_score = self.detector._calculate_urgency_score(90, 1.0, 1000, 500)
        
        self.assertGreater(same_day_score, long_term_score)
    
    def test_determine_trade_sentiment(self):
        """Test trade sentiment determination"""
        # Test bullish call
        sentiment = self.detector._determine_trade_sentiment('call', 0.9, 30, 1000, 500)
        self.assertEqual(sentiment, 'bullish')
        
        # Test bearish put
        sentiment = self.detector._determine_trade_sentiment('put', 1.1, 30, 1000, 500)
        self.assertEqual(sentiment, 'bearish')
    
    def test_determine_trade_type(self):
        """Test trade type determination"""
        # Opening trade (volume > 50% of OI)
        trade_type = self.detector._determine_trade_type(1000, 1500)
        self.assertEqual(trade_type, 'opening')
        
        # Closing trade (volume < 20% of OI)
        trade_type = self.detector._determine_trade_type(100, 1000)
        self.assertEqual(trade_type, 'closing')

class TestDatabaseManager(unittest.TestCase):
    """Test database operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use in-memory SQLite for testing
        import tempfile
        self.test_db = tempfile.NamedTemporaryFile(delete=False)
        
        # Mock settings to use test database
        with patch('src.data.database.get_settings') as mock_settings:
            mock_settings.return_value.DATABASE_URL = f"sqlite:///{self.test_db.name}"
            mock_settings.return_value.DATA_RETENTION_DAYS = 365
            self.db_manager = DatabaseManager()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        os.unlink(self.test_db.name)
    
    def test_database_initialization(self):
        """Test database table creation"""
        stats = self.db_manager.get_database_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('options_data_count', stats)
        self.assertIn('big_trades_count', stats)
    
    def test_store_options_data(self):
        """Test storing options data"""
        options_data = [{
            'symbol': 'TEST',
            'timestamp': datetime.now(),
            'contract_type': 'call',
            'expiration': '2024-01-19',
            'strike': 100.0,
            'volume': 500,
            'open_interest': 1000,
            'premium': 2.5,
            'bid': 2.4,
            'ask': 2.6,
            'delta': 0.5,
            'gamma': 0.01,
            'theta': -0.05,
            'vega': 0.1,
            'implied_volatility': 0.25,
            'underlying_price': 100.0
        }]
        
        result = self.db_manager.store_options_data(options_data)
        self.assertTrue(result)
        
        # Verify data was stored
        df = self.db_manager.get_options_data(symbol='TEST')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['symbol'], 'TEST')
    
    def test_store_big_trade(self):
        """Test storing big trade data"""
        big_trade = {
            'symbol': 'TEST',
            'timestamp': datetime.now(),
            'contract_type': 'call',
            'expiration': '2024-01-19',
            'strike': 100.0,
            'volume': 1000,
            'open_interest': 2000,
            'premium': 5.0,
            'notional_value': 500000.0,
            'trade_type': 'opening',
            'sentiment': 'bullish',
            'size_score': 7.5,
            'urgency_score': 6.0,
            'confidence_level': 0.8,
            'underlying_price': 100.0,
            'time_to_expiration': 30,
            'moneyness': 1.0,
            'analysis_notes': ['Large bullish bet'],
            'related_trades': []
        }
        
        result = self.db_manager.store_big_trade(big_trade)
        self.assertTrue(result)
        
        # Verify data was stored
        df = self.db_manager.get_big_trades(symbol='TEST')
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['symbol'], 'TEST')

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @patch('src.api.schwab_client.SchwabClient')
    def test_full_analysis_workflow(self, mock_client_class):
        """Test complete analysis workflow"""
        # Mock the Schwab client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock options chain response
        mock_options_chain = {
            'callExpDateMap': {
                '2024-01-19:45': {
                    '450.0': [{
                        'totalVolume': 1000,
                        'openInterest': 5000,
                        'mark': 5.0,
                        'bid': 4.9,
                        'ask': 5.1,
                        'delta': 0.5,
                        'gamma': 0.01,
                        'theta': -0.05,
                        'vega': 0.1,
                        'volatility': 0.25
                    }]
                }
            },
            'putExpDateMap': {
                '2024-01-19:45': {
                    '440.0': [{
                        'totalVolume': 800,
                        'openInterest': 3000,
                        'mark': 3.0,
                        'bid': 2.9,
                        'ask': 3.1,
                        'delta': -0.3,
                        'gamma': 0.008,
                        'theta': -0.04,
                        'vega': 0.08,
                        'volatility': 0.28
                    }]
                }
            }
        }
        
        mock_client.get_options_chain.return_value = mock_options_chain
        
        # Test market analysis
        analyzer = MarketDynamicsAnalyzer(mock_client)
        result = analyzer.analyze_short_term_dynamics(['SPY'])
        
        self.assertIsNotNone(result)
        self.assertEqual(result.analysis_type, 'short_term')
        self.assertIsInstance(result.sentiment, MarketSentiment)
        self.assertIsInstance(result.recommendations, list)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestConfiguration))
    test_suite.addTest(unittest.makeSuite(TestMarketDynamicsAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestBigTradesDetector))
    test_suite.addTest(unittest.makeSuite(TestDatabaseManager))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)