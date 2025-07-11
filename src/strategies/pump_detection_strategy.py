"""
급등 초입 진입전략 (1H봉 기준)
급등 패턴을 감지하고 초기에 진입하는 전략
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class PumpDetectionStrategy:
    """급등 초입 진입전략"""
    
    def __init__(self, 
                 volume_threshold: float = 2.0,  # 거래량 증가 임계값
                 price_threshold: float = 0.03,  # 가격 상승 임계값 (3%)
                 momentum_period: int = 4,       # 모멘텀 계산 기간
                 rsi_oversold: float = 30,      # RSI 과매도 기준
                 rsi_overbought: float = 70):   # RSI 과매수 기준
        
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
        self.momentum_period = momentum_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        # 이동평균
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 거래량 이동평균
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # 가격 변화율
        df['price_change'] = df['close'].pct_change()
        df['price_change_4h'] = df['close'].pct_change(periods=4)
        
        # 거래량 변화율
        df['volume_change'] = df['volume'].pct_change()
        
        # 모멘텀 지표
        df['momentum'] = df['close'] - df['close'].shift(self.momentum_period)
        df['momentum_ratio'] = df['momentum'] / df['close'].shift(self.momentum_period)
        
        # 볼린저 밴드
        df['bb_upper'] = df['ma_20'] + (df['close'].rolling(window=20).std() * 2)
        df['bb_lower'] = df['ma_20'] - (df['close'].rolling(window=20).std() * 2)
        
        return df
    
    def detect_pump_pattern(self, df: pd.DataFrame, current_idx: int) -> bool:
        """급등 패턴 감지"""
        if current_idx < 50:  # 충분한 데이터 필요
            return False
        
        current_data = df.iloc[current_idx]
        prev_data = df.iloc[current_idx - 1]
        
        # 1. 거래량 급증 확인
        volume_surge = (current_data['volume'] > prev_data['volume'] * self.volume_threshold and
                       current_data['volume'] > current_data['volume_ma'] * 1.5)
        
        # 2. 가격 급등 확인
        price_surge = (current_data['price_change'] > self.price_threshold and
                      current_data['price_change_4h'] > self.price_threshold * 1.5)
        
        # 3. 모멘텀 확인
        momentum_positive = (current_data['momentum_ratio'] > 0.01 and
                           current_data['momentum_ratio'] < 0.2)
        
        # 4. RSI 조건
        rsi_ok = (current_data['rsi'] < self.rsi_overbought and
                 current_data['rsi'] > 20)
        
        # 5. 볼린저 밴드 조건
        bb_condition = (current_data['close'] > current_data['ma_20'] and
                       current_data['close'] < current_data['bb_upper'] * 1.05)
        
        # 6. 이전 패턴 확인
        recent_surge = False
        for i in range(1, 3):
            if current_idx - i >= 0:
                past_data = df.iloc[current_idx - i]
                if (past_data['price_change'] > self.price_threshold * 2 or
                    past_data['volume_change'] > self.volume_threshold * 2):
                    recent_surge = True
                    break
        
        # 급등 패턴 조건
        pump_pattern = (volume_surge and 
                       price_surge and 
                       momentum_positive and 
                       rsi_ok and 
                       bb_condition and 
                       not recent_surge)
        
        return pump_pattern
    
    def calculate_position_size(self, balance: float, price: float, risk_per_trade: float = 0.02) -> float:
        """포지션 크기 계산 (리스크 2% 기준)"""
        risk_amount = balance * risk_per_trade
        position_size = risk_amount / price
        return position_size
    
    def check_exit_conditions(self, position: Dict, current_price: float, current_data: pd.Series) -> Tuple[Optional[str], Dict]:
        """청산 조건 확인 (단순화된 로직)"""
        entry_price = position['entry_price']
        current_profit_rate = (current_price - entry_price) / entry_price * 100
        
        # 고점 추적
        if 'highest_price' not in position:
            position['highest_price'] = entry_price
        
        if current_price > position['highest_price']:
            position['highest_price'] = current_price
        
        # 시간 계산
        hold_time = current_data.name - position['entry_time']
        hold_hours = hold_time.total_seconds() / 3600
        
        # 1. 손절 조건: -2% 손실
        if current_profit_rate <= -2:
            return 'sell', position
        
        # 2. 익절 조건: 5% 이상 수익
        if current_profit_rate >= 5:
            return 'sell', position
        
        # 3. 트레일링 스탑: 고점 대비 3% 하락
        peak_price = position['highest_price']
        drawdown = (peak_price - current_price) / peak_price * 100
        if drawdown >= 3 and current_profit_rate > 0:
            return 'sell', position
        
        # 4. 시간 기반 청산: 6시간 이상 보유
        if hold_hours >= 6:
            return 'sell', position
        
        return None, position
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int, position: Optional[Dict] = None, balance: float = 0) -> Tuple[Optional[str], Optional[Dict]]:
        """매매 신호 생성"""
        try:
            # 기술적 지표 계산
            df = self.calculate_technical_indicators(df)
            
            # 롱 포지션이 없을 때만 진입 신호 확인
            if position is None:
                if self.detect_pump_pattern(df, current_idx):
                    current_data = df.iloc[current_idx]
                    entry_price = current_data['close']
                    position_size = self.calculate_position_size(balance, entry_price)
                    
                    # 새로운 포지션 객체 생성
                    new_position = {
                        'entry_price': entry_price,
                        'entry_time': current_data.name,
                        'size': position_size,
                        'highest_price': entry_price
                    }
                    
                    return 'buy', new_position
            
            # 포지션이 있을 때 청산 조건 확인
            elif position is not None:
                current_data = df.iloc[current_idx]
                current_price = current_data['close']
                
                # 청산 조건 확인
                signal, position = self.check_exit_conditions(position, current_price, current_data)
                if signal == 'sell':
                    return 'sell', position
            
            return None, position
            
        except Exception as e:
            print(f"신호 생성 오류: {e}")
            return None, position 