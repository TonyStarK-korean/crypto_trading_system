"""
볼린저 밴드 브레이크아웃 전략
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class BollingerBreakoutStrategy:
    """볼린저 밴드 브레이크아웃 전략"""
    
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, float, float]:
        """볼린저 밴드 계산"""
        if current_idx < self.window:
            return None, None, None
        
        prices = df['close'].iloc[current_idx-self.window:current_idx]
        sma = prices.mean()
        std = prices.std()
        
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        return upper_band, sma, lower_band
    
    def check_breakout_conditions(self, df: pd.DataFrame, current_idx: int) -> bool:
        """브레이크아웃 조건 확인"""
        if current_idx < self.window:
            return False
        
        current_data = df.iloc[current_idx]
        prev_data = df.iloc[current_idx-1]
        
        # 볼린저 밴드 계산
        upper_band, sma, lower_band = self.calculate_bollinger_bands(df, current_idx)
        
        if upper_band is None:
            return False
        
        # 브레이크아웃 조건:
        # 1. 현재 가격이 볼린저 밴드 상단을 돌파
        # 2. 이전 가격은 상단 밴드 아래
        # 3. 거래량 증가
        price_breakout = (current_data['close'] > upper_band and 
                         prev_data['close'] <= upper_band)
        
        volume_increase = current_data['volume'] > prev_data['volume'] * 1.2
        
        return price_breakout and volume_increase
    
    def calculate_position_size(self, balance: float, price: float, risk_per_trade: float = 0.02) -> float:
        """포지션 크기 계산 (리스크 2% 기준)"""
        risk_amount = balance * risk_per_trade
        position_size = risk_amount / price
        return position_size
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int, position: Optional[Dict] = None, balance: float = 0) -> Tuple[Optional[str], Optional[Dict]]:
        """매매 신호 생성"""
        try:
            # 롱 포지션이 없을 때만 진입 신호 확인
            if position is None:
                if self.check_breakout_conditions(df, current_idx):
                    current_data = df.iloc[current_idx]
                    entry_price = current_data['close']
                    position_size = self.calculate_position_size(balance, entry_price)
                    
                    print(f"🔵 진입: {current_data.name} | 가격: {entry_price:,.2f} | 비중: {position_size:.4f} | 자산: {balance:,.2f}")
                    return 'buy', None
            
            # 포지션이 있을 때 청산 조건 확인
            elif position is not None:
                current_data = df.iloc[current_idx]
                entry_price = position['entry_price']
                current_price = current_data['close']
                
                # 손절: 진입가 대비 -2%
                stop_loss = entry_price * 0.98
                
                # 트레일링 익절 시스템
                if 'highest_price' not in position:
                    position['highest_price'] = entry_price
                
                # 고점 업데이트
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                
                # 트레일링 익절: 고점 대비 -1.5% 하락 시 청산
                trailing_stop = position['highest_price'] * 0.985
                
                # 청산 조건 확인
                if current_price <= stop_loss:
                    pnl = (current_price - entry_price) * position['size']
                    new_balance = balance + pnl
                    loss_rate = (pnl / balance) * 100
                    print(f"🔴 손절: {current_data.name} | 가격: {current_price:,.2f} | 손실: {pnl:,.2f} | 손실률: {loss_rate:.2f}% | 자산: {new_balance:,.2f}")
                    return 'sell', position
                
                elif current_price <= trailing_stop and current_price > entry_price:
                    pnl = (current_price - entry_price) * position['size']
                    new_balance = balance + pnl
                    profit_rate = (pnl / balance) * 100
                    print(f"🟢 트레일링 익절: {current_data.name} | 가격: {current_price:,.2f} | 수익: {pnl:,.2f} | 수익률: {profit_rate:.2f}% | 자산: {new_balance:,.2f}")
                    return 'sell', position
            
            return None, position
            
        except Exception as e:
            print(f"신호 생성 오류: {e}")
            return None, position 