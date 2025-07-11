"""
Working Crypto Trading System
실제 작동하는 암호화폐 거래 시스템
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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTradingSystem:
    """간단한 거래 시스템"""
    
    def __init__(self, symbols: List[str] = None, initial_capital: float = 10000):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        self.initial_capital = initial_capital
        self.exchange = ccxt.binance()
        self.db_path = "data/working_system.db"
        self.running = False
        self.latest_prices = {}
        
        # 데이터베이스 초기화
        self.init_database()
        
        # 스레드
        self.data_thread = None
        self.price_thread = None
        
        logger.info("Trading System initialized")
    
    def init_database(self):
        """데이터베이스 초기화"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                strategy TEXT,
                return_pct REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                win_rate REAL,
                current_signal TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                price REAL,
                change_pct REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def collect_data(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        """데이터 수집"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                for candle in ohlcv:
                    cursor.execute('''
                        INSERT OR REPLACE INTO ohlcv_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        timeframe,
                        datetime.fromtimestamp(candle[0] / 1000),
                        candle[1], candle[2], candle[3], candle[4], candle[5]
                    ))
                
                conn.commit()
                conn.close()
                logger.info(f"Collected {len(ohlcv)} records for {symbol} {timeframe}")
                return True
                
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return False
    
    def get_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """데이터 조회"""
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
    
    def simple_ma_backtest(self, symbol: str, timeframe: str = '1h', fast_period: int = 10, slow_period: int = 30):
        """간단한 이동평균 백테스트"""
        # 데이터 가져오기
        df = self.get_data(symbol, timeframe, limit=200)
        
        if df.empty or len(df) < slow_period:
            logger.warning(f"Not enough data for {symbol}")
            return None
        
        # 이동평균 계산
        df['MA_fast'] = df['close'].rolling(window=fast_period).mean()
        df['MA_slow'] = df['close'].rolling(window=slow_period).mean()
        
        # 신호 생성 (간단한 방법)
        signals = []
        for i in range(len(df)):
            if i >= slow_period:
                if df.iloc[i]['MA_fast'] > df.iloc[i]['MA_slow']:
                    signals.append(1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
        
        df['signal'] = signals
        
        # 백테스트 계산
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(1, len(df)):
            prev_signal = df.iloc[i-1]['signal']
            curr_signal = df.iloc[i]['signal']
            
            # 매수 신호
            if prev_signal == 0 and curr_signal == 1 and position == 0:
                position = capital / df.iloc[i]['close']
                entry_price = df.iloc[i]['close']
                capital = 0
                
            # 매도 신호
            elif prev_signal == 1 and curr_signal == 0 and position > 0:
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
        
        # 최종 계산
        if position > 0:
            final_value = position * df.iloc[-1]['close']
        else:
            final_value = capital
        
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # 최대 드로우다운
        if equity_curve:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak * 100
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        # 승률
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # 현재 신호
        current_signal = 'BUY' if df.iloc[-1]['signal'] == 1 else 'SELL'
        
        result = {
            'symbol': symbol,
            'strategy': 'MA_Cross',
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'current_signal': current_signal,
            'trades': trades
        }
        
        # 결과 저장
        self.save_backtest_result(result)
        
        return result
    
    def save_backtest_result(self, result: Dict):
        """백테스트 결과 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backtest_results 
            (symbol, strategy, return_pct, max_drawdown, total_trades, win_rate, current_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['symbol'],
            result['strategy'],
            result['total_return'],
            result['max_drawdown'],
            result['total_trades'],
            result['win_rate'],
            result['current_signal']
        ))
        
        conn.commit()
        conn.close()
    
    def monitor_prices(self):
        """가격 모니터링"""
        while self.running:
            for symbol in self.symbols:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # 가격 변화 체크
                    if symbol in self.latest_prices:
                        prev_price = self.latest_prices[symbol]
                        change_pct = (current_price - prev_price) / prev_price * 100
                        
                        if abs(change_pct) > 2.0:  # 2% 이상 변화
                            self.save_price_alert(symbol, current_price, change_pct)
                            logger.info(f"Price alert: {symbol} {change_pct:.2f}% change")
                    
                    self.latest_prices[symbol] = current_price
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
            
            time.sleep(30)  # 30초마다 체크
    
    def save_price_alert(self, symbol: str, price: float, change_pct: float):
        """가격 알림 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO price_alerts (symbol, price, change_pct)
            VALUES (?, ?, ?)
        ''', (symbol, price, change_pct))
        
        conn.commit()
        conn.close()
    
    def collect_all_data(self):
        """모든 데이터 수집"""
        while self.running:
            for symbol in self.symbols:
                for timeframe in ['1h', '4h']:
                    if self.running:
                        self.collect_data(symbol, timeframe)
                        time.sleep(2)
            
            if self.running:
                logger.info("Data collection cycle completed")
                time.sleep(300)  # 5분 대기
    
    def start_system(self):
        """시스템 시작"""
        if self.running:
            logger.warning("System already running")
            return
        
        self.running = True
        
        # 데이터 수집 스레드
        self.data_thread = threading.Thread(target=self.collect_all_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # 가격 모니터링 스레드
        self.price_thread = threading.Thread(target=self.monitor_prices)
        self.price_thread.daemon = True
        self.price_thread.start()
        
        logger.info("Trading system started")
    
    def stop_system(self):
        """시스템 중지"""
        self.running = False
        
        if self.data_thread:
            self.data_thread.join(timeout=5)
        if self.price_thread:
            self.price_thread.join(timeout=5)
        
        logger.info("Trading system stopped")
    
    def run_backtests(self):
        """모든 심볼에 대한 백테스트 실행"""
        results = []
        
        for symbol in self.symbols:
            logger.info(f"Running backtest for {symbol}")
            result = self.simple_ma_backtest(symbol)
            if result:
                results.append(result)
            time.sleep(1)
        
        return results
    
    def get_system_status(self):
        """시스템 상태 조회"""
        return {
            'running': self.running,
            'symbols': self.symbols,
            'latest_prices': self.latest_prices,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_backtest_history(self, limit: int = 10):
        """백테스트 기록 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, strategy, return_pct, max_drawdown, total_trades, 
                   win_rate, current_signal, created_at
            FROM backtest_results
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results

def main():
    """메인 함수"""
    print("="*60)
    print("CRYPTO TRADING SYSTEM - WORKING VERSION")
    print("="*60)
    
    try:
        # 시스템 초기화
        system = SimpleTradingSystem(
            symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
            initial_capital=10000
        )
        
        # 시스템 시작
        system.start_system()
        
        # 초기 데이터 수집
        logger.info("Collecting initial data...")
        for symbol in system.symbols:
            system.collect_data(symbol, '1h', 100)
            time.sleep(2)
        
        # 백테스트 실행
        logger.info("Running backtests...")
        results = system.run_backtests()
        
        # 결과 출력
        if results:
            print("\n" + "="*60)
            print("BACKTEST RESULTS")
            print("="*60)
            
            for result in results:
                print(f"Symbol: {result['symbol']}")
                print(f"Strategy: {result['strategy']}")
                print(f"Initial Capital: ${result['initial_capital']:,.2f}")
                print(f"Final Value: ${result['final_value']:,.2f}")
                print(f"Total Return: {result['total_return']:.2f}%")
                print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Win Rate: {result['win_rate']:.2f}%")
                print(f"Current Signal: {result['current_signal']}")
                print("-" * 60)
        
        # 시스템 상태
        status = system.get_system_status()
        print(f"\nSYSTEM STATUS: {json.dumps(status, indent=2, default=str)}")
        
        # 시스템 실행 유지
        logger.info("System running. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(60)
                
                # 현재 가격 출력
                if system.latest_prices:
                    prices = [f"{k}: ${v:,.2f}" for k, v in system.latest_prices.items()]
                    logger.info("Current prices: " + ", ".join(prices))
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            system.stop_system()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()