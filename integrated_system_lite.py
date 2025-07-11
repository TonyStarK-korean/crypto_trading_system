"""
Integrated Crypto Trading System - Lite Version
통합된 암호화폐 거래 시스템 - 가벼운 버전 (SQLite 사용)
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
import asyncio
import websockets
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """시스템 설정"""
    database_path: str = "data/integrated_trading.db"
    symbols: List[str] = None
    timeframes: List[str] = None
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02
    update_interval: int = 300  # 5 minutes
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        if self.timeframes is None:
            self.timeframes = ['1h', '4h', '1d']

class DatabaseManager:
    """데이터베이스 관리자 - SQLite 사용"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV 데이터 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # 거래 신호 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL NOT NULL,
                confidence REAL,
                metadata TEXT
            )
        ''')
        
        # 백테스트 결과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_date DATETIME NOT NULL,
                end_date DATETIME NOT NULL,
                initial_capital REAL NOT NULL,
                final_value REAL NOT NULL,
                total_return REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL,
                total_trades INTEGER NOT NULL,
                win_rate REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 시스템 메트릭 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def insert_ohlcv_data(self, data: List[Dict]):
        """OHLCV 데이터 삽입"""
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
        """OHLCV 데이터 조회"""
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
    
    def insert_trading_signal(self, signal: Dict):
        """거래 신호 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_signals 
            (symbol, strategy_name, signal_type, timestamp, price, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'],
            signal['strategy_name'],
            signal['signal_type'],
            signal['timestamp'],
            signal['price'],
            signal.get('confidence'),
            json.dumps(signal.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def insert_backtest_result(self, result: Dict):
        """백테스트 결과 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_results 
            (strategy_name, symbol, start_date, end_date, initial_capital, 
             final_value, total_return, max_drawdown, sharpe_ratio, total_trades, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['strategy_name'],
            result['symbol'],
            result['start_date'],
            result['end_date'],
            result['initial_capital'],
            result['final_value'],
            result['total_return'],
            result['max_drawdown'],
            result.get('sharpe_ratio'),
            result['total_trades'],
            result.get('win_rate')
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_price(self, symbol: str, timeframe: str = '1h') -> Optional[float]:
        """최신 가격 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT close FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (symbol, timeframe))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None

class DataCollector:
    """데이터 수집기"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.exchange = ccxt.binance()
        self.running = False
        self.collection_thread = None
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """OHLCV 데이터 수집"""
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
                logger.info(f"Fetched {len(data)} records for {symbol} {timeframe}")
                return True
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return False
    
    def start_collection(self, symbols: List[str], timeframes: List[str], interval: int = 300):
        """데이터 수집 시작"""
        self.running = True
        
        def collect_data():
            while self.running:
                for symbol in symbols:
                    for timeframe in timeframes:
                        if self.running:
                            self.fetch_ohlcv_data(symbol, timeframe)
                            time.sleep(1)  # Rate limiting
                
                if self.running:
                    logger.info(f"Data collection cycle completed. Sleeping for {interval} seconds...")
                    time.sleep(interval)
        
        self.collection_thread = threading.Thread(target=collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Data collection started")
    
    def stop_collection(self):
        """데이터 수집 중지"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Data collection stopped")

class RealTimeDataManager:
    """실시간 데이터 관리자"""
    
    def __init__(self, symbols: List[str], db_manager: DatabaseManager):
        self.symbols = symbols
        self.db_manager = db_manager
        self.exchange = ccxt.binance()
        self.latest_prices = {}
        self.running = False
        self.price_thread = None
    
    def start_price_monitoring(self, interval: int = 30):
        """가격 모니터링 시작"""
        self.running = True
        
        def monitor_prices():
            while self.running:
                for symbol in self.symbols:
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        self.latest_prices[symbol] = {
                            'price': price,
                            'timestamp': datetime.now(),
                            'volume': ticker.get('quoteVolume', 0)
                        }
                        
                        # 가격 신호 체크
                        self.check_price_signals(symbol, price)
                        
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error fetching price for {symbol}: {e}")
                
                if self.running:
                    time.sleep(interval)
        
        self.price_thread = threading.Thread(target=monitor_prices)
        self.price_thread.daemon = True
        self.price_thread.start()
        logger.info("Price monitoring started")
    
    def check_price_signals(self, symbol: str, current_price: float):
        """가격 신호 체크"""
        try:
            # 간단한 가격 변화 알림
            if symbol in self.latest_prices:
                prev_price = self.latest_prices[symbol]['price']
                change_pct = (current_price - prev_price) / prev_price * 100
                
                if abs(change_pct) > 2.0:  # 2% 이상 변화
                    signal = {
                        'symbol': symbol,
                        'strategy_name': 'price_alert',
                        'signal_type': 'ALERT',
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'confidence': abs(change_pct),
                        'metadata': {'change_pct': change_pct}
                    }
                    
                    self.db_manager.insert_trading_signal(signal)
                    logger.info(f"Price alert for {symbol}: {change_pct:.2f}% change")
        except Exception as e:
            logger.error(f"Error checking price signals: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """최신 가격 조회"""
        if symbol in self.latest_prices:
            return self.latest_prices[symbol]['price']
        return None
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.running = False
        if self.price_thread:
            self.price_thread.join(timeout=5)
        logger.info("Price monitoring stopped")

class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self, db_manager: DatabaseManager, initial_capital: float = 10000):
        self.db_manager = db_manager
        self.initial_capital = initial_capital
    
    def moving_average_strategy(self, symbol: str, timeframe: str = '1h', 
                               fast_period: int = 10, slow_period: int = 30):
        """이동평균 교차 전략"""
        # 데이터 가져오기
        df = self.db_manager.get_ohlcv_data(symbol, timeframe, limit=500)
        
        if df.empty or len(df) < slow_period:
            return None
        
        # 이동평균 계산
        df['MA_fast'] = df['close'].rolling(window=fast_period).mean()
        df['MA_slow'] = df['close'].rolling(window=slow_period).mean()
        
        # 신호 생성
        df['signal'] = 0
        df.iloc[slow_period:, df.columns.get_loc('signal')] = np.where(
            df['MA_fast'].iloc[slow_period:] > df['MA_slow'].iloc[slow_period:], 1, 0
        )
        df['position'] = df['signal'].diff()
        
        # 수익률 계산
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            if df.iloc[i]['position'] == 1:  # Buy signal
                if position == 0:
                    position = capital / df.iloc[i]['close']
                    entry_price = df.iloc[i]['close']
                    capital = 0
                    
            elif df.iloc[i]['position'] == -1:  # Sell signal
                if position > 0:
                    capital = position * df.iloc[i]['close']
                    pnl = capital - self.initial_capital
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': df.iloc[i]['close'],
                        'pnl': pnl,
                        'return_pct': (df.iloc[i]['close'] / entry_price - 1) * 100
                    })
                    position = 0
            
            # 현재 자산 가치
            if position > 0:
                current_value = position * df.iloc[i]['close']
            else:
                current_value = capital
            
            equity_curve.append(current_value)
        
        # 최종 결과 계산
        if position > 0:
            final_value = position * df.iloc[-1]['close']
        else:
            final_value = capital
        
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # 최대 드로우다운 계산
        if equity_curve:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak * 100
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        # 샤프 비율 계산
        if trades:
            returns = [t['return_pct'] for t in trades]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # 승률 계산
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        return {
            'strategy_name': 'MA_Cross',
            'symbol': symbol,
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trades': trades,
            'current_signal': df.iloc[-1]['signal']
        }

class IntegratedTradingSystem:
    """통합 거래 시스템"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # 구성 요소 초기화
        self.db_manager = DatabaseManager(self.config.database_path)
        self.data_collector = DataCollector(self.db_manager)
        self.realtime_manager = RealTimeDataManager(self.config.symbols, self.db_manager)
        self.backtest_engine = BacktestEngine(self.db_manager, self.config.initial_capital)
        
        self.running = False
        logger.info("Integrated Trading System initialized")
    
    def start_system(self):
        """시스템 시작"""
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        
        # 데이터 수집 시작
        self.data_collector.start_collection(
            self.config.symbols, 
            self.config.timeframes, 
            self.config.update_interval
        )
        
        # 실시간 가격 모니터링 시작
        self.realtime_manager.start_price_monitoring(30)
        
        logger.info("Integrated Trading System started")
    
    def stop_system(self):
        """시스템 중지"""
        if not self.running:
            logger.warning("System is not running")
            return
        
        self.running = False
        
        # 모든 구성 요소 중지
        self.data_collector.stop_collection()
        self.realtime_manager.stop_monitoring()
        
        logger.info("Integrated Trading System stopped")
    
    def run_backtest(self, symbol: str, timeframe: str = '1h', strategy: str = 'MA_Cross'):
        """백테스트 실행"""
        try:
            if strategy == 'MA_Cross':
                results = self.backtest_engine.moving_average_strategy(symbol, timeframe)
            else:
                logger.error(f"Unknown strategy: {strategy}")
                return None
            
            if results:
                # 결과 저장
                self.db_manager.insert_backtest_result(results)
                logger.info(f"Backtest completed for {symbol}: {results['total_return']:.2f}% return")
                
                return results
            else:
                logger.warning(f"No backtest results for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        status = {
            'running': self.running,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'symbols': self.config.symbols,
                'timeframes': self.config.timeframes,
                'initial_capital': self.config.initial_capital,
                'update_interval': self.config.update_interval
            },
            'components': {
                'data_collector': self.data_collector.running,
                'realtime_manager': self.realtime_manager.running,
                'database': os.path.exists(self.config.database_path)
            },
            'latest_prices': self.realtime_manager.latest_prices
        }
        
        return status
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """최신 가격 조회"""
        return self.realtime_manager.get_latest_price(symbol)
    
    def run_multiple_backtests(self, symbols: List[str] = None, timeframes: List[str] = None):
        """다중 백테스트 실행"""
        if symbols is None:
            symbols = self.config.symbols
        if timeframes is None:
            timeframes = ['1h']
        
        results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                result = self.run_backtest(symbol, timeframe)
                if result:
                    results.append(result)
                time.sleep(1)  # Rate limiting
        
        return results

def main():
    """메인 함수"""
    logger.info("Starting Integrated Crypto Trading System...")
    
    try:
        # 시스템 설정
        config = SystemConfig(
            symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            timeframes=['1h', '4h'],
            initial_capital=10000.0,
            update_interval=300
        )
        
        # 시스템 초기화
        system = IntegratedTradingSystem(config)
        
        # 시스템 시작
        system.start_system()
        
        # 초기 데이터 수집 대기
        logger.info("Waiting for initial data collection...")
        time.sleep(30)
        
        # 백테스트 실행
        logger.info("Running backtests...")
        results = system.run_multiple_backtests()
        
        # 결과 출력
        if results:
            print("\n" + "="*60)
            print("BACKTEST RESULTS SUMMARY")
            print("="*60)
            
            for result in results:
                print(f"Strategy: {result['strategy_name']}")
                print(f"Symbol: {result['symbol']}")
                print(f"Total Return: {result['total_return']:.2f}%")
                print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Win Rate: {result['win_rate']:.2f}%")
                print(f"Current Signal: {'BUY' if result['current_signal'] == 1 else 'SELL'}")
                print("-" * 60)
        
        # 시스템 상태 출력
        status = system.get_system_status()
        print("\nSYSTEM STATUS:")
        print(json.dumps(status, indent=2, default=str))
        
        # 시스템 실행 유지
        logger.info("System running. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(60)
                
                # 주기적 상태 체크
                current_prices = []
                for symbol in config.symbols:
                    price = system.get_latest_price(symbol)
                    if price:
                        current_prices.append(f"{symbol}: ${price:,.2f}")
                
                if current_prices:
                    logger.info("Current prices: " + ", ".join(current_prices))
                
        except KeyboardInterrupt:
            logger.info("Shutting down system...")
            system.stop_system()
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()