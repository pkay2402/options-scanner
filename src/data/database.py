"""
Database Management Module
Handles data storage and retrieval for options trading data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.sqlite import insert
import json

from ..utils.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()

class OptionsData(Base):
    """Options data table"""
    __tablename__ = 'options_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    contract_type = Column(String(4), nullable=False)  # 'call' or 'put'
    expiration = Column(String(20), nullable=False)
    strike = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    open_interest = Column(Integer, default=0)
    premium = Column(Float, default=0.0)
    bid = Column(Float, default=0.0)
    ask = Column(Float, default=0.0)
    delta = Column(Float, default=0.0)
    gamma = Column(Float, default=0.0)
    theta = Column(Float, default=0.0)
    vega = Column(Float, default=0.0)
    implied_volatility = Column(Float, default=0.0)
    underlying_price = Column(Float, default=0.0)

class BigTrades(Base):
    """Big trades table"""
    __tablename__ = 'big_trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    contract_type = Column(String(4), nullable=False)
    expiration = Column(String(20), nullable=False)
    strike = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    open_interest = Column(Integer, default=0)
    premium = Column(Float, nullable=False)
    notional_value = Column(Float, nullable=False)
    trade_type = Column(String(20), default='unknown')
    sentiment = Column(String(20), default='neutral')
    size_score = Column(Float, default=0.0)
    urgency_score = Column(Float, default=0.0)
    confidence_level = Column(Float, default=0.0)
    underlying_price = Column(Float, default=0.0)
    time_to_expiration = Column(Integer, default=0)
    moneyness = Column(Float, default=1.0)
    analysis_notes = Column(Text)
    related_trades = Column(Text)

class MarketSentiment(Base):
    """Market sentiment table"""
    __tablename__ = 'market_sentiment'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    analysis_type = Column(String(20), nullable=False)  # 'short_term', 'mid_term'
    put_call_ratio = Column(Float, default=0.0)
    vix_level = Column(Float, default=0.0)
    gamma_exposure = Column(Float, default=0.0)
    dealer_positioning = Column(String(20), default='neutral')
    sentiment_score = Column(Float, default=0.0)
    confidence_level = Column(Float, default=0.0)
    key_levels = Column(Text)  # JSON string
    recommendations = Column(Text)  # JSON string
    risk_factors = Column(Text)  # JSON string

class UnusualActivity(Base):
    """Unusual activity table"""
    __tablename__ = 'unusual_activity'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    activity_type = Column(String(30), nullable=False)
    description = Column(Text)
    metrics = Column(Text)  # JSON string
    severity = Column(Float, default=0.0)
    market_impact = Column(String(20), default='low')

class DatabaseManager:
    """
    Manages database operations for options trading data
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.settings = get_settings()
            self.engine = create_engine(self.settings.DATABASE_URL, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self._create_tables()
            DatabaseManager._initialized = True
    
    def _create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def store_options_data(self, options_data: List[Dict[str, Any]]) -> bool:
        """
        Store options data in the database
        """
        try:
            session = self.get_session()
            
            for data in options_data:
                # Convert to database model
                db_option = OptionsData(
                    symbol=data.get('symbol'),
                    timestamp=data.get('timestamp', datetime.now()),
                    contract_type=data.get('contract_type'),
                    expiration=data.get('expiration'),
                    strike=data.get('strike'),
                    volume=data.get('volume', 0),
                    open_interest=data.get('open_interest', 0),
                    premium=data.get('premium', 0.0),
                    bid=data.get('bid', 0.0),
                    ask=data.get('ask', 0.0),
                    delta=data.get('delta', 0.0),
                    gamma=data.get('gamma', 0.0),
                    theta=data.get('theta', 0.0),
                    vega=data.get('vega', 0.0),
                    implied_volatility=data.get('implied_volatility', 0.0),
                    underlying_price=data.get('underlying_price', 0.0)
                )
                
                session.add(db_option)
            
            session.commit()
            session.close()
            
            logger.info(f"Stored {len(options_data)} options data records")
            return True
            
        except Exception as e:
            logger.error(f"Error storing options data: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def store_big_trade(self, big_trade: Dict[str, Any]) -> bool:
        """
        Store a big trade in the database
        """
        try:
            session = self.get_session()
            
            db_trade = BigTrades(
                symbol=big_trade.get('symbol'),
                timestamp=big_trade.get('timestamp', datetime.now()),
                contract_type=big_trade.get('contract_type'),
                expiration=big_trade.get('expiration'),
                strike=big_trade.get('strike'),
                volume=big_trade.get('volume'),
                open_interest=big_trade.get('open_interest', 0),
                premium=big_trade.get('premium'),
                notional_value=big_trade.get('notional_value'),
                trade_type=big_trade.get('trade_type', 'unknown'),
                sentiment=big_trade.get('sentiment', 'neutral'),
                size_score=big_trade.get('size_score', 0.0),
                urgency_score=big_trade.get('urgency_score', 0.0),
                confidence_level=big_trade.get('confidence_level', 0.0),
                underlying_price=big_trade.get('underlying_price', 0.0),
                time_to_expiration=big_trade.get('time_to_expiration', 0),
                moneyness=big_trade.get('moneyness', 1.0),
                analysis_notes=json.dumps(big_trade.get('analysis_notes', [])),
                related_trades=json.dumps(big_trade.get('related_trades', []))
            )
            
            session.add(db_trade)
            session.commit()
            session.close()
            
            logger.info(f"Stored big trade for {big_trade.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing big trade: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def store_market_sentiment(self, sentiment_data: Dict[str, Any]) -> bool:
        """
        Store market sentiment analysis
        """
        try:
            session = self.get_session()
            
            db_sentiment = MarketSentiment(
                timestamp=sentiment_data.get('timestamp', datetime.now()),
                analysis_type=sentiment_data.get('analysis_type', 'short_term'),
                put_call_ratio=sentiment_data.get('put_call_ratio', 0.0),
                vix_level=sentiment_data.get('vix_level', 0.0),
                gamma_exposure=sentiment_data.get('gamma_exposure', 0.0),
                dealer_positioning=sentiment_data.get('dealer_positioning', 'neutral'),
                sentiment_score=sentiment_data.get('sentiment_score', 0.0),
                confidence_level=sentiment_data.get('confidence_level', 0.0),
                key_levels=json.dumps(sentiment_data.get('key_levels', {})),
                recommendations=json.dumps(sentiment_data.get('recommendations', [])),
                risk_factors=json.dumps(sentiment_data.get('risk_factors', []))
            )
            
            session.add(db_sentiment)
            session.commit()
            session.close()
            
            logger.info(f"Stored market sentiment analysis")
            return True
            
        except Exception as e:
            logger.error(f"Error storing market sentiment: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def store_unusual_activity(self, activity_data: Dict[str, Any]) -> bool:
        """
        Store unusual activity detection
        """
        try:
            session = self.get_session()
            
            db_activity = UnusualActivity(
                symbol=activity_data.get('symbol'),
                timestamp=activity_data.get('timestamp', datetime.now()),
                activity_type=activity_data.get('activity_type'),
                description=activity_data.get('description'),
                metrics=json.dumps(activity_data.get('metrics', {})),
                severity=activity_data.get('severity', 0.0),
                market_impact=activity_data.get('market_impact', 'low')
            )
            
            session.add(db_activity)
            session.commit()
            session.close()
            
            logger.info(f"Stored unusual activity for {activity_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing unusual activity: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_options_data(self, symbol: str = None, start_date: datetime = None,
                        end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve options data from database
        """
        try:
            session = self.get_session()
            
            query = session.query(OptionsData)
            
            # Apply filters
            if symbol:
                query = query.filter(OptionsData.symbol == symbol)
            if start_date:
                query = query.filter(OptionsData.timestamp >= start_date)
            if end_date:
                query = query.filter(OptionsData.timestamp <= end_date)
            
            # Order by timestamp and limit
            query = query.order_by(OptionsData.timestamp.desc())
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            session.close()
            
            # Convert to DataFrame
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'symbol': result.symbol,
                    'timestamp': result.timestamp,
                    'contract_type': result.contract_type,
                    'expiration': result.expiration,
                    'strike': result.strike,
                    'volume': result.volume,
                    'open_interest': result.open_interest,
                    'premium': result.premium,
                    'bid': result.bid,
                    'ask': result.ask,
                    'delta': result.delta,
                    'gamma': result.gamma,
                    'theta': result.theta,
                    'vega': result.vega,
                    'implied_volatility': result.implied_volatility,
                    'underlying_price': result.underlying_price
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving options data: {str(e)}")
            return pd.DataFrame()
    
    def get_big_trades(self, symbol: str = None, start_date: datetime = None,
                      end_date: datetime = None, min_notional: float = None,
                      limit: int = 100) -> pd.DataFrame:
        """
        Retrieve big trades from database
        """
        try:
            session = self.get_session()
            
            query = session.query(BigTrades)
            
            # Apply filters
            if symbol:
                query = query.filter(BigTrades.symbol == symbol)
            if start_date:
                query = query.filter(BigTrades.timestamp >= start_date)
            if end_date:
                query = query.filter(BigTrades.timestamp <= end_date)
            if min_notional:
                query = query.filter(BigTrades.notional_value >= min_notional)
            
            # Order by notional value and limit
            query = query.order_by(BigTrades.notional_value.desc())
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            session.close()
            
            # Convert to DataFrame
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'symbol': result.symbol,
                    'timestamp': result.timestamp,
                    'contract_type': result.contract_type,
                    'expiration': result.expiration,
                    'strike': result.strike,
                    'volume': result.volume,
                    'open_interest': result.open_interest,
                    'premium': result.premium,
                    'notional_value': result.notional_value,
                    'trade_type': result.trade_type,
                    'sentiment': result.sentiment,
                    'size_score': result.size_score,
                    'urgency_score': result.urgency_score,
                    'confidence_level': result.confidence_level,
                    'underlying_price': result.underlying_price,
                    'time_to_expiration': result.time_to_expiration,
                    'moneyness': result.moneyness,
                    'analysis_notes': json.loads(result.analysis_notes) if result.analysis_notes else [],
                    'related_trades': json.loads(result.related_trades) if result.related_trades else []
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving big trades: {str(e)}")
            return pd.DataFrame()
    
    def get_market_sentiment_history(self, analysis_type: str = None,
                                   start_date: datetime = None,
                                   end_date: datetime = None,
                                   limit: int = 100) -> pd.DataFrame:
        """
        Retrieve market sentiment history
        """
        try:
            session = self.get_session()
            
            query = session.query(MarketSentiment)
            
            # Apply filters
            if analysis_type:
                query = query.filter(MarketSentiment.analysis_type == analysis_type)
            if start_date:
                query = query.filter(MarketSentiment.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketSentiment.timestamp <= end_date)
            
            # Order by timestamp and limit
            query = query.order_by(MarketSentiment.timestamp.desc())
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            session.close()
            
            # Convert to DataFrame
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'timestamp': result.timestamp,
                    'analysis_type': result.analysis_type,
                    'put_call_ratio': result.put_call_ratio,
                    'vix_level': result.vix_level,
                    'gamma_exposure': result.gamma_exposure,
                    'dealer_positioning': result.dealer_positioning,
                    'sentiment_score': result.sentiment_score,
                    'confidence_level': result.confidence_level,
                    'key_levels': json.loads(result.key_levels) if result.key_levels else {},
                    'recommendations': json.loads(result.recommendations) if result.recommendations else [],
                    'risk_factors': json.loads(result.risk_factors) if result.risk_factors else []
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving market sentiment: {str(e)}")
            return pd.DataFrame()
    
    def get_unusual_activity(self, symbol: str = None, activity_type: str = None,
                           start_date: datetime = None, end_date: datetime = None,
                           min_severity: float = None, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve unusual activity records
        """
        try:
            session = self.get_session()
            
            query = session.query(UnusualActivity)
            
            # Apply filters
            if symbol:
                query = query.filter(UnusualActivity.symbol == symbol)
            if activity_type:
                query = query.filter(UnusualActivity.activity_type == activity_type)
            if start_date:
                query = query.filter(UnusualActivity.timestamp >= start_date)
            if end_date:
                query = query.filter(UnusualActivity.timestamp <= end_date)
            if min_severity:
                query = query.filter(UnusualActivity.severity >= min_severity)
            
            # Order by severity and limit
            query = query.order_by(UnusualActivity.severity.desc())
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            session.close()
            
            # Convert to DataFrame
            data = []
            for result in results:
                data.append({
                    'id': result.id,
                    'symbol': result.symbol,
                    'timestamp': result.timestamp,
                    'activity_type': result.activity_type,
                    'description': result.description,
                    'metrics': json.loads(result.metrics) if result.metrics else {},
                    'severity': result.severity,
                    'market_impact': result.market_impact
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving unusual activity: {str(e)}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = None) -> bool:
        """
        Clean up old data beyond retention period
        """
        if days_to_keep is None:
            days_to_keep = self.settings.DATA_RETENTION_DAYS
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            session = self.get_session()
            
            # Clean up old options data
            deleted_options = session.query(OptionsData).filter(
                OptionsData.timestamp < cutoff_date
            ).delete()
            
            # Clean up old unusual activity
            deleted_activity = session.query(UnusualActivity).filter(
                UnusualActivity.timestamp < cutoff_date
            ).delete()
            
            # Keep market sentiment and big trades longer (they're more valuable)
            sentiment_cutoff = datetime.now() - timedelta(days=days_to_keep * 2)
            deleted_sentiment = session.query(MarketSentiment).filter(
                MarketSentiment.timestamp < sentiment_cutoff
            ).delete()
            
            session.commit()
            session.close()
            
            logger.info(f"Cleaned up old data: {deleted_options} options records, "
                       f"{deleted_activity} activity records, {deleted_sentiment} sentiment records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        """
        try:
            session = self.get_session()
            
            stats = {
                'options_data_count': session.query(OptionsData).count(),
                'big_trades_count': session.query(BigTrades).count(),
                'market_sentiment_count': session.query(MarketSentiment).count(),
                'unusual_activity_count': session.query(UnusualActivity).count(),
                'latest_options_data': session.query(OptionsData.timestamp).order_by(
                    OptionsData.timestamp.desc()
                ).first(),
                'oldest_options_data': session.query(OptionsData.timestamp).order_by(
                    OptionsData.timestamp.asc()
                ).first(),
                'unique_symbols': session.query(OptionsData.symbol).distinct().count()
            }
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def export_data_to_csv(self, table_name: str, filepath: str,
                          start_date: datetime = None, end_date: datetime = None) -> bool:
        """
        Export data to CSV file
        """
        try:
            if table_name == 'options_data':
                df = self.get_options_data(start_date=start_date, end_date=end_date, limit=None)
            elif table_name == 'big_trades':
                df = self.get_big_trades(start_date=start_date, end_date=end_date, limit=None)
            elif table_name == 'market_sentiment':
                df = self.get_market_sentiment_history(start_date=start_date, end_date=end_date, limit=None)
            elif table_name == 'unusual_activity':
                df = self.get_unusual_activity(start_date=start_date, end_date=end_date, limit=None)
            else:
                logger.error(f"Unknown table name: {table_name}")
                return False
            
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} records to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {str(e)}")
            return False