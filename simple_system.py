"""
Simplified Crypto Trading System
간단한 버전의 암호화폐 거래 시스템 (SQLAlchemy 없이)
"""

import os
import sys
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import ccxt
import threading
from loguru import logger

# Simple Database Manager using SQLite
class SimpleDatabaseManager:
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                exchange TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                symbol TEXT,
                start_date DATETIME,
                end_date DATETIME,
                initial_capital REAL,
                final_value REAL,
                total_return REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("SQLite database initialized")
    
    def insert_ohlcv_data(self, data: List[Dict]):
        """Insert OHLCV data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for record in data:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO ohlcv_data 
                    (symbol, exchange, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record['symbol'],
                    record['exchange'],
                    record['timeframe'],
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ))
            except Exception as e:
                logger.error(f"Error inserting OHLCV data: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(data)} OHLCV records")
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Get OHLCV data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
        
        return df
    
    def insert_backtest_result(self, result: Dict):
        """Insert backtest result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_results 
            (strategy_name, symbol, start_date, end_date, initial_capital, 
             final_value, total_return, max_drawdown, total_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['strategy_name'],
            result['symbol'],
            result['start_date'],
            result['end_date'],
            result['initial_capital'],
            result['final_value'],
            result['total_return'],
            result['max_drawdown'],
            result['total_trades']
        ))
        
        conn.commit()
        conn.close()
        logger.info("Backtest result saved")

# Simple Data Collector
class SimpleDataCollector:
    def __init__(self, db_manager: SimpleDatabaseManager):
        self.db_manager = db_manager
        self.exchange = ccxt.binance()
        self.running = False
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv:
                data = []
                for candle in ohlcv:
                    data.append({
                        'symbol': symbol,
                        'exchange': 'binance',
                        'timeframe': timeframe,
                        'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                
                self.db_manager.insert_ohlcv_data(data)
                logger.info(f"Fetched {len(data)} records for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return False
    
    def start_data_collection(self, symbols: List[str], timeframes: List[str], interval: int = 300):
        """Start automated data collection"""
        self.running = True
        
        def collect_data():
            while self.running:
                for symbol in symbols:
                    for timeframe in timeframes:
                        self.fetch_ohlcv_data(symbol, timeframe)
                        time.sleep(1)  # Rate limiting
                
                logger.info(f"Data collection cycle completed. Sleeping for {interval} seconds...")
                time.sleep(interval)
        
        thread = threading.Thread(target=collect_data)
        thread.daemon = True
        thread.start()
        logger.info("Data collection started")
    
    def stop_data_collection(self):
        """Stop data collection"""
        self.running = False
        logger.info("Data collection stopped")

# Simple Backtest Engine
class SimpleBacktestEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
    
    def moving_average_strategy(self, data: pd.DataFrame, fast_period: int = 10, slow_period: int = 30):
        """Simple Moving Average Crossover Strategy"""
        if len(data) < slow_period:
            return {'error': 'Not enough data'}
        
        # Calculate moving averages
        data['MA_fast'] = data['close'].rolling(window=fast_period).mean()
        data['MA_slow'] = data['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        data['signal'][slow_period:] = np.where(
            data['MA_fast'][slow_period:] > data['MA_slow'][slow_period:], 1, 0
        )
        data['position'] = data['signal'].diff()
        
        # Simulate trading
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        
        for i in range(len(data)):
            if data.iloc[i]['position'] == 1:  # Buy signal
                if self.position == 0:
                    self.position = self.capital / data.iloc[i]['close']
                    self.entry_price = data.iloc[i]['close']
                    self.capital = 0
                    
            elif data.iloc[i]['position'] == -1:  # Sell signal
                if self.position > 0:
                    self.capital = self.position * data.iloc[i]['close']
                    pnl = self.capital - self.initial_capital
                    self.trades.append({
                        'entry_price': self.entry_price,
                        'exit_price': data.iloc[i]['close'],
                        'pnl': pnl,
                        'return': pnl / self.initial_capital * 100
                    })
                    self.position = 0
            
            # Calculate current equity
            if self.position > 0:
                current_value = self.position * data.iloc[i]['close']
            else:
                current_value = self.capital
            
            self.equity_curve.append(current_value)
        
        # Calculate final results
        final_value = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        if self.equity_curve:
            peak = np.maximum.accumulate(self.equity_curve)
            drawdown = (peak - self.equity_curve) / peak * 100
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in self.trades if t['pnl'] < 0]),
            'trades': self.trades
        }

# Simple Trading System
class SimpleTradingSystem:
    def __init__(self):
        self.db_manager = SimpleDatabaseManager()
        self.data_collector = SimpleDataCollector(self.db_manager)
        self.backtest_engine = SimpleBacktestEngine()
        self.running = False
        logger.info("Simple Trading System initialized")
    
    def start_system(self, symbols: List[str] = None, timeframes: List[str] = None):
        """Start the trading system"""
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT']
        if timeframes is None:
            timeframes = ['1h', '4h']
        
        self.running = True
        
        # Start data collection
        self.data_collector.start_data_collection(symbols, timeframes)
        
        logger.info("Trading system started")
    
    def stop_system(self):
        """Stop the trading system"""
        self.running = False
        self.data_collector.stop_data_collection()
        logger.info("Trading system stopped")
    
    def run_backtest(self, symbol: str, timeframe: str = '1h', strategy: str = 'MA_Cross'):
        """Run backtest"""
        try:
            # Get data
            data = self.db_manager.get_ohlcv_data(symbol, timeframe, limit=500)
            
            if data.empty:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return {'error': 'No data available'}
            
            # Run backtest
            if strategy == 'MA_Cross':
                results = self.backtest_engine.moving_average_strategy(data)
            else:
                return {'error': 'Unknown strategy'}
            
            # Save results
            backtest_result = {
                'strategy_name': strategy,
                'symbol': symbol,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'initial_capital': results['initial_capital'],
                'final_value': results['final_value'],
                'total_return': results['total_return'],
                'max_drawdown': results['max_drawdown'],
                'total_trades': results['total_trades']
            }
            
            self.db_manager.insert_backtest_result(backtest_result)
            
            logger.info(f"Backtest completed for {symbol}: {results['total_return']:.2f}% return")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def get_system_status(self):
        """Get system status"""
        return {
            'running': self.running,
            'timestamp': datetime.now().isoformat(),
            'database_file': self.db_manager.db_path,
            'data_collection_active': self.data_collector.running
        }

def main():
    """Main function"""
    try:
        # Initialize system
        system = SimpleTradingSystem()
        
        # Start system
        system.start_system(['BTC/USDT', 'ETH/USDT'], ['1h'])
        
        # Wait for some data collection
        logger.info("Waiting for data collection...")
        time.sleep(30)
        
        # Run backtest
        logger.info("Running backtest...")
        results = system.run_backtest('BTC/USDT', '1h', 'MA_Cross')
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Strategy: MA_Cross")
        print(f"Symbol: BTC/USDT")
        print(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0):.2f}%")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Winning Trades: {results.get('winning_trades', 0)}")
        print(f"Losing Trades: {results.get('losing_trades', 0)}")
        print("="*50)
        
        # Get system status
        status = system.get_system_status()
        print(f"\nSystem Status: {json.dumps(status, indent=2)}")
        
        # Keep running
        logger.info("System running. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(60)
                logger.info("System heartbeat...")
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            system.stop_system()
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()