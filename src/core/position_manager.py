"""
포지션 관리 시스템
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from enum import Enum


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_id: Optional[str] = None


class PositionManager:
    """포지션 관리 시스템"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
    def open_position(self, symbol: str, side: PositionSide, size: float, 
                     entry_price: float, leverage: float = 1.0,
                     stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None) -> bool:
        """포지션 오픈"""
        try:
            # 기존 포지션이 있는지 확인
            if symbol in self.positions:
                logger.warning(f"이미 {symbol} 포지션이 존재합니다")
                return False
            
            # 잔고 확인
            required_margin = (size * entry_price) / leverage
            if required_margin > self.current_balance:
                logger.error(f"잔고 부족: 필요 {required_margin}, 보유 {self.current_balance}")
                return False
            
            # 포지션 생성
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                entry_time=datetime.now(),
                current_price=entry_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_id=f"{symbol}_{side.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            self.positions[symbol] = position
            self.current_balance -= required_margin
            
            logger.info(f"포지션 오픈: {symbol} {side.value} {size}@{entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"포지션 오픈 오류: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "manual") -> bool:
        """포지션 클로즈"""
        try:
            if symbol not in self.positions:
                logger.warning(f"{symbol} 포지션이 존재하지 않습니다")
                return False
            
            position = self.positions[symbol]
            
            # PnL 계산
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * position.size * position.leverage
            else:
                pnl = (position.entry_price - exit_price) * position.size * position.leverage
            
            # 잔고 업데이트
            margin_return = (position.size * position.entry_price) / position.leverage
            self.current_balance += margin_return + pnl
            
            # 포지션 정보 업데이트
            position.current_price = exit_price
            position.realized_pnl = pnl
            position.unrealized_pnl = 0.0
            
            # 거래 기록
            trade_record = {
                'symbol': symbol,
                'side': position.side.value,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'size': position.size,
                'leverage': position.leverage,
                'pnl': pnl,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(),
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # 통계 업데이트
            self.total_pnl += pnl
            self._update_drawdown()
            
            logger.info(f"포지션 클로즈: {symbol} PnL: {pnl:.2f} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"포지션 클로즈 오류: {e}")
            return False
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """포지션 가격 업데이트"""
        try:
            for symbol, price in price_updates.items():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.current_price = price
                    
                    # 미실현 PnL 계산
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = (price - position.entry_price) * position.size * position.leverage
                    else:
                        position.unrealized_pnl = (position.entry_price - price) * position.size * position.leverage
                    
                    # 스탑로스/익절 체크
                    self._check_stop_conditions(symbol, price)
                    
        except Exception as e:
            logger.error(f"포지션 가격 업데이트 오류: {e}")
    
    def _check_stop_conditions(self, symbol: str, current_price: float):
        """스탑로스/익절 조건 체크"""
        position = self.positions[symbol]
        
        # 스탑로스 체크
        if position.stop_loss:
            if (position.side == PositionSide.LONG and current_price <= position.stop_loss) or \
               (position.side == PositionSide.SHORT and current_price >= position.stop_loss):
                self.close_position(symbol, current_price, "stop_loss")
                return
        
        # 익절 체크
        if position.take_profit:
            if (position.side == PositionSide.LONG and current_price >= position.take_profit) or \
               (position.side == PositionSide.SHORT and current_price <= position.take_profit):
                self.close_position(symbol, current_price, "take_profit")
                return
    
    def _update_drawdown(self):
        """최대 낙폭 업데이트"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_position_summary(self) -> Dict:
        """포지션 요약 정보"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        
        return {
            'current_balance': self.current_balance,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """승률 계산"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return (winning_trades / len(self.trade_history)) * 100
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """거래 히스토리를 DataFrame으로 반환"""
        if not self.trade_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_history)
        df['duration'] = df['exit_time'] - df['entry_time']
        df['return_pct'] = (df['pnl'] / (df['entry_price'] * df['size'] / df['leverage'])) * 100
        
        return df
    
    def calculate_compound_return(self) -> float:
        """복리 수익률 계산"""
        if self.initial_balance <= 0:
            return 0.0
        
        return ((self.current_balance / self.initial_balance) - 1) * 100
    
    def reset(self):
        """포지션 매니저 초기화"""
        self.positions.clear()
        self.closed_positions.clear()
        self.trade_history.clear()
        self.current_balance = self.initial_balance
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance 