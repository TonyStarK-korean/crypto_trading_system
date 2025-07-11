"""
Database Management System for Crypto Trading
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB
from loguru import logger
import psycopg2

Base = declarative_base()

class OHLCVData(Base):
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )

class TradingSignals(Base):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    strategy_name = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    __table_args__ = (
        Index('idx_symbol_strategy_timestamp', 'symbol', 'strategy_name', 'timestamp'),
    )

class BacktestResults(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    config = Column(JSONB, nullable=True)

class TradingPositions(Base):
    __tablename__ = 'trading_positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # LONG, SHORT
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)

class DatabaseManager:
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Default to PostgreSQL
            user = os.getenv('DB_USER', 'postgres')
            password = os.getenv('DB_PASSWORD', 'password')
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '5432')
            database = os.getenv('DB_NAME', 'crypto_trading')
            
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def insert_ohlcv_data(self, data: List[Dict[str, Any]]) -> bool:
        """Insert OHLCV data into database"""
        session = self.get_session()
        try:
            for record in data:
                ohlcv = OHLCVData(
                    symbol=record['symbol'],
                    exchange=record['exchange'],
                    timeframe=record['timeframe'],
                    timestamp=record['timestamp'],
                    open=record['open'],
                    high=record['high'],
                    low=record['low'],
                    close=record['close'],
                    volume=record['volume']
                )
                session.merge(ohlcv)
            
            session.commit()
            logger.info(f"Inserted {len(data)} OHLCV records")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting OHLCV data: {e}")
            return False
        finally:
            session.close()
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, 
                       start_date: datetime = None, 
                       end_date: datetime = None) -> pd.DataFrame:
        """Retrieve OHLCV data as DataFrame"""
        session = self.get_session()
        try:
            query = session.query(OHLCVData).filter(
                OHLCVData.symbol == symbol,
                OHLCVData.timeframe == timeframe
            )
            
            if start_date:
                query = query.filter(OHLCVData.timestamp >= start_date)
            if end_date:
                query = query.filter(OHLCVData.timestamp <= end_date)
            
            query = query.order_by(OHLCVData.timestamp)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = []
            for record in results:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open,
                    'high': record.high,
                    'low': record.low,
                    'close': record.close,
                    'volume': record.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def insert_trading_signal(self, signal: Dict[str, Any]) -> bool:
        """Insert trading signal"""
        session = self.get_session()
        try:
            trading_signal = TradingSignals(
                symbol=signal['symbol'],
                strategy_name=signal['strategy_name'],
                signal_type=signal['signal_type'],
                timestamp=signal['timestamp'],
                price=signal['price'],
                confidence=signal.get('confidence'),
                metadata=signal.get('metadata')
            )
            session.add(trading_signal)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting trading signal: {e}")
            return False
        finally:
            session.close()
    
    def insert_backtest_result(self, result: Dict[str, Any]) -> bool:
        """Insert backtest result"""
        session = self.get_session()
        try:
            backtest_result = BacktestResults(
                strategy_name=result['strategy_name'],
                symbol=result['symbol'],
                start_date=result['start_date'],
                end_date=result['end_date'],
                initial_capital=result['initial_capital'],
                final_value=result['final_value'],
                total_return=result['total_return'],
                max_drawdown=result['max_drawdown'],
                sharpe_ratio=result.get('sharpe_ratio'),
                win_rate=result.get('win_rate'),
                total_trades=result['total_trades'],
                config=result.get('config')
            )
            session.add(backtest_result)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting backtest result: {e}")
            return False
        finally:
            session.close()
    
    def get_latest_price(self, symbol: str, timeframe: str = '1m') -> float:
        """Get latest price for a symbol"""
        session = self.get_session()
        try:
            latest_record = session.query(OHLCVData).filter(
                OHLCVData.symbol == symbol,
                OHLCVData.timeframe == timeframe
            ).order_by(OHLCVData.timestamp.desc()).first()
            
            if latest_record:
                return latest_record.close
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to save space"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old OHLCV data (keep only recent data)
            deleted_count = session.query(OHLCVData).filter(
                OHLCVData.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted_count} old OHLCV records")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old data: {e}")
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self.get_session()
        try:
            stats = {}
            
            # Count records in each table
            stats['ohlcv_records'] = session.query(OHLCVData).count()
            stats['trading_signals'] = session.query(TradingSignals).count()
            stats['backtest_results'] = session.query(BacktestResults).count()
            stats['trading_positions'] = session.query(TradingPositions).count()
            
            # Get date range of data
            oldest_data = session.query(OHLCVData.timestamp).order_by(OHLCVData.timestamp.asc()).first()
            newest_data = session.query(OHLCVData.timestamp).order_by(OHLCVData.timestamp.desc()).first()
            
            if oldest_data and newest_data:
                stats['data_range'] = {
                    'start': oldest_data[0],
                    'end': newest_data[0]
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            session.close()