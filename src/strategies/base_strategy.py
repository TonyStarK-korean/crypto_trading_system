"""
기본 전략 클래스
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod
from loguru import logger


class BaseStrategy(ABC):
    """전략 기본 클래스"""
    
    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.position = None
        self.last_signal = None
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, ml_signal: Optional[float] = None) -> Dict:
        """매매 신호 생성 (추상 메서드)"""
        pass
    
    def calculate_position_size(self, balance: float, price: float, risk_pct: float = 0.02) -> float:
        """포지션 크기 계산"""
        try:
            risk_amount = balance * risk_pct
            position_size = risk_amount / price
            return position_size
        except Exception as e:
            logger.error(f"포지션 크기 계산 오류: {e}")
            return 0.0
    
    def calculate_stop_loss(self, entry_price: float, side: str, atr: float, multiplier: float = 2.0) -> float:
        """스탑로스 계산"""
        try:
            if side == 'long':
                stop_loss = entry_price - (atr * multiplier)
            else:
                stop_loss = entry_price + (atr * multiplier)
            return stop_loss
        except Exception as e:
            logger.error(f"스탑로스 계산 오류: {e}")
            return None
    
    def calculate_take_profit(self, entry_price: float, side: str, atr: float, multiplier: float = 3.0) -> float:
        """익절 계산"""
        try:
            if side == 'long':
                take_profit = entry_price + (atr * multiplier)
            else:
                take_profit = entry_price - (atr * multiplier)
            return take_profit
        except Exception as e:
            logger.error(f"익절 계산 오류: {e}")
            return None
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """ATR (Average True Range) 계산"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            logger.error(f"ATR 계산 오류: {e}")
            return 0.0
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """데이터 유효성 검사"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            if data.empty:
                logger.warning("데이터가 비어있습니다")
                return False
            
            if not all(col in data.columns for col in required_columns):
                logger.warning("필수 컬럼이 누락되었습니다")
                return False
            
            if data.isnull().any().any():
                logger.warning("데이터에 NaN 값이 있습니다")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"데이터 검증 오류: {e}")
            return False
    
    def get_strategy_info(self) -> Dict:
        """전략 정보 반환"""
        return {
            'name': self.name,
            'params': self.params,
            'description': self.__doc__ or "No description available"
        }
    
    def update_params(self, new_params: Dict):
        """전략 파라미터 업데이트"""
        self.params.update(new_params)
        logger.info(f"전략 파라미터 업데이트: {new_params}")
    
    def reset(self):
        """전략 상태 초기화"""
        self.position = None
        self.last_signal = None
        logger.info("전략 상태 초기화 완료") 