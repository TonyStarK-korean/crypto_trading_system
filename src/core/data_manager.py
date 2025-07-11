"""
데이터 관리 및 수집 시스템
"""

import pandas as pd
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import yaml
import os


class DataManager:
    """실시간 및 히스토리 데이터 관리"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.exchange = self._initialize_exchange()
        self.data_cache = {}
        self.latest_data = {}
        
    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            'exchange': {
                'name': 'binance',
                'api_key': '',
                'secret': '',
                'sandbox': True
            },
            'data': {
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
                'limit': 1000
            }
        }
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """거래소 초기화"""
        exchange_config = self.config['exchange']
        exchange_class = getattr(ccxt, exchange_config['name'])
        
        exchange = exchange_class({
            'apiKey': exchange_config['api_key'],
            'secret': exchange_config['secret'],
            'sandbox': exchange_config['sandbox'],
            'enableRateLimit': True
        })
        
        return exchange
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """OHLCV 데이터 수집"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"데이터 수집 오류: {symbol} {timeframe} - {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """히스토리 데이터 수집"""
        try:
            # 실제 구현에서는 거래소 API를 통해 히스토리 데이터 수집
            # 여기서는 예시 데이터 생성
            date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
            np.random.seed(42)
            
            data = {
                'open': np.random.normal(50000, 1000, len(date_range)),
                'high': np.random.normal(50500, 1000, len(date_range)),
                'low': np.random.normal(49500, 1000, len(date_range)),
                'close': np.random.normal(50000, 1000, len(date_range)),
                'volume': np.random.normal(1000, 200, len(date_range))
            }
            
            df = pd.DataFrame(data, index=date_range)
            return df
            
        except Exception as e:
            logger.error(f"히스토리 데이터 수집 오류: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 이동평균
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # 거래량 지표
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 계산 오류: {e}")
            return df
    
    def get_latest_price(self, symbol: str) -> float:
        """최신 가격 조회"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"가격 조회 오류: {symbol} - {e}")
            return 0.0
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """데이터 저장"""
        try:
            filename = f"data/{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename)
            logger.info(f"데이터 저장 완료: {filename}")
        except Exception as e:
            logger.error(f"데이터 저장 오류: {e}")
    
    def load_data(self, symbol: str, timeframe: str, date: str) -> pd.DataFrame:
        """저장된 데이터 로드"""
        try:
            filename = f"data/{symbol.replace('/', '_')}_{timeframe}_{date}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                return df
            else:
                logger.warning(f"파일을 찾을 수 없습니다: {filename}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            return pd.DataFrame() 