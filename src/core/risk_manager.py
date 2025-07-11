"""
리스크 관리 시스템
적응형 리스크 관리 및 포트폴리오 관리
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """리스크 관리자"""
    
    def __init__(self,
                 max_portfolio_risk: float = 0.02,  # 포트폴리오 최대 리스크
                 max_position_size: float = 0.1,    # 단일 포지션 최대 크기
                 max_correlation: float = 0.7,      # 최대 상관관계
                 volatility_lookback: int = 20,     # 변동성 계산 기간
                 var_confidence: float = 0.05):     # VaR 신뢰도
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.volatility_lookback = volatility_lookback
        self.var_confidence = var_confidence
        
        self.portfolio_positions = {}
        self.correlation_matrix = None
        self.volatility_estimates = {}
        
    def calculate_portfolio_risk(self, positions: Dict, price_data: Dict) -> float:
        """포트폴리오 리스크 계산"""
        if not positions:
            return 0.0
        
        # 각 포지션의 리스크 계산
        position_risks = {}
        for symbol, position in positions.items():
            if symbol in price_data:
                volatility = self.calculate_volatility(price_data[symbol])
                position_value = position['size'] * position['current_price']
                position_risks[symbol] = position_value * volatility
        
        # 포트폴리오 전체 리스크 (상관관계 고려)
        if len(position_risks) == 1:
            return list(position_risks.values())[0]
        
        # 상관관계 행렬을 이용한 포트폴리오 리스크 계산
        symbols = list(position_risks.keys())
        risks = np.array(list(position_risks.values()))
        
        if self.correlation_matrix is not None:
            corr_subset = self.get_correlation_subset(symbols)
            portfolio_variance = np.dot(risks, np.dot(corr_subset, risks))
            portfolio_risk = np.sqrt(portfolio_variance)
        else:
            # 상관관계 정보가 없으면 단순 합계
            portfolio_risk = np.sum(risks)
        
        return portfolio_risk
    
    def calculate_volatility(self, price_series: pd.Series) -> float:
        """변동성 계산"""
        if len(price_series) < self.volatility_lookback:
            return 0.02  # 기본값
        
        returns = price_series.pct_change().dropna()
        volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]
        
        # 연간화 (1시간 데이터 기준)
        return volatility * np.sqrt(24 * 365)
    
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """Value at Risk 계산"""
        if confidence is None:
            confidence = self.var_confidence
        
        if len(returns) < 30:
            return 0.0
        
        return np.percentile(returns.dropna(), confidence * 100)
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               balance: float, volatility: float) -> float:
        """포지션 크기 계산"""
        # Kelly Criterion 기반 계산
        win_rate = self.get_historical_win_rate(symbol)
        avg_win = self.get_average_win(symbol)
        avg_loss = self.get_average_loss(symbol)
        
        if avg_loss == 0:
            kelly_fraction = 0.02
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 0-25% 제한
        
        # 변동성 조정
        volatility_adjustment = min(1.0, 0.15 / max(volatility, 0.01))
        
        # 기본 포지션 크기
        base_position = balance * self.max_portfolio_risk * kelly_fraction * volatility_adjustment
        
        # 최대 포지션 크기 제한
        max_position = balance * self.max_position_size
        
        return min(base_position, max_position)
    
    def get_historical_win_rate(self, symbol: str) -> float:
        """과거 승률 계산"""
        # 실제 구현에서는 과거 거래 데이터를 사용
        return 0.55  # 기본값
    
    def get_average_win(self, symbol: str) -> float:
        """평균 수익 계산"""
        # 실제 구현에서는 과거 거래 데이터를 사용
        return 0.04  # 기본값
    
    def get_average_loss(self, symbol: str) -> float:
        """평균 손실 계산"""
        # 실제 구현에서는 과거 거래 데이터를 사용
        return 0.02  # 기본값
    
    def update_correlation_matrix(self, price_data: Dict):
        """상관관계 행렬 업데이트"""
        if len(price_data) < 2:
            return
        
        # 수익률 계산
        returns_data = {}
        for symbol, prices in price_data.items():
            if len(prices) > 1:
                returns_data[symbol] = prices.pct_change().dropna()
        
        if len(returns_data) < 2:
            return
        
        # 상관관계 행렬 계산
        returns_df = pd.DataFrame(returns_data)
        self.correlation_matrix = returns_df.corr()
    
    def get_correlation_subset(self, symbols: List[str]) -> np.ndarray:
        """특정 심볼들의 상관관계 부분행렬 추출"""
        if self.correlation_matrix is None:
            return np.eye(len(symbols))
        
        try:
            subset = self.correlation_matrix.loc[symbols, symbols]
            return subset.values
        except:
            return np.eye(len(symbols))
    
    def check_correlation_limits(self, new_symbol: str, existing_positions: Dict) -> bool:
        """상관관계 제한 확인"""
        if not existing_positions or self.correlation_matrix is None:
            return True
        
        existing_symbols = list(existing_positions.keys())
        
        for symbol in existing_symbols:
            try:
                correlation = self.correlation_matrix.loc[new_symbol, symbol]
                if abs(correlation) > self.max_correlation:
                    logger.warning(f"높은 상관관계 감지: {new_symbol} - {symbol} ({correlation:.2f})")
                    return False
            except:
                continue
        
        return True
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> Dict:
        """낙폭 계산"""
        if len(equity_curve) < 2:
            return {'current_dd': 0, 'max_dd': 0, 'dd_duration': 0}
        
        # 누적 최대값 계산
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # 현재 낙폭
        current_dd = drawdown.iloc[-1]
        
        # 최대 낙폭
        max_dd = drawdown.min()
        
        # 낙폭 지속 기간
        dd_duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown.iloc[i] < 0:
                dd_duration += 1
            else:
                break
        
        return {
            'current_dd': current_dd,
            'max_dd': max_dd,
            'dd_duration': dd_duration
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """샤프 비율 계산"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """소르티노 비율 계산"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        
        return excess_returns.mean() / downside_std
    
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """칼마 비율 계산"""
        if max_drawdown == 0:
            return 0.0
        
        annualized_return = returns.mean() * 252  # 일간 수익률 기준
        return annualized_return / abs(max_drawdown)
    
    def should_reduce_position(self, symbol: str, current_loss: float) -> bool:
        """포지션 축소 여부 결정"""
        # 연속 손실 확인
        consecutive_losses = self.get_consecutive_losses(symbol)
        
        # 큰 손실 발생 시 포지션 축소
        if current_loss > 0.05:  # 5% 이상 손실
            return True
        
        # 연속 손실이 3회 이상
        if consecutive_losses >= 3:
            return True
        
        return False
    
    def get_consecutive_losses(self, symbol: str) -> int:
        """연속 손실 횟수 계산"""
        # 실제 구현에서는 거래 히스토리를 사용
        return 0  # 기본값
    
    def calculate_portfolio_metrics(self, positions: Dict, price_data: Dict) -> Dict:
        """포트폴리오 메트릭 계산"""
        if not positions:
            return {}
        
        total_value = 0
        total_pnl = 0
        position_count = len(positions)
        
        for symbol, position in positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]['close'].iloc[-1]
                position_value = position['size'] * current_price
                pnl = (current_price - position['entry_price']) * position['size']
                
                total_value += position_value
                total_pnl += pnl
        
        portfolio_return = total_pnl / total_value if total_value > 0 else 0
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'portfolio_return': portfolio_return,
            'position_count': position_count,
            'avg_position_size': total_value / position_count if position_count > 0 else 0
        }
    
    def generate_risk_report(self, positions: Dict, price_data: Dict, 
                           equity_curve: pd.Series) -> Dict:
        """리스크 보고서 생성"""
        portfolio_risk = self.calculate_portfolio_risk(positions, price_data)
        portfolio_metrics = self.calculate_portfolio_metrics(positions, price_data)
        
        # 수익률 계산
        returns = equity_curve.pct_change().dropna()
        
        # 낙폭 계산
        drawdown_info = self.calculate_drawdown(equity_curve)
        
        # 리스크 메트릭 계산
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, drawdown_info['max_dd'])
        var = self.calculate_var(returns)
        
        return {
            'portfolio_risk': portfolio_risk,
            'portfolio_metrics': portfolio_metrics,
            'drawdown_info': drawdown_info,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'var_5': var,
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        }

class PortfolioManager:
    """포트폴리오 관리자"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.positions = {}
        self.cash_balance = 0.0
        self.equity_curve = []
        
    def add_position(self, symbol: str, position: Dict) -> bool:
        """포지션 추가"""
        # 상관관계 확인
        if not self.risk_manager.check_correlation_limits(symbol, self.positions):
            return False
        
        # 포트폴리오 리스크 확인
        test_positions = self.positions.copy()
        test_positions[symbol] = position
        
        # 리스크 한도 확인 (실제 구현에서는 price_data 필요)
        # portfolio_risk = self.risk_manager.calculate_portfolio_risk(test_positions, price_data)
        # if portfolio_risk > self.risk_manager.max_portfolio_risk:
        #     return False
        
        self.positions[symbol] = position
        return True
    
    def remove_position(self, symbol: str):
        """포지션 제거"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def update_positions(self, price_data: Dict):
        """포지션 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]['close'].iloc[-1]
                position['current_price'] = current_price
                position['current_pnl'] = (current_price - position['entry_price']) * position['size']
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 가치 계산"""
        total_value = self.cash_balance
        for position in self.positions.values():
            if 'current_price' in position:
                total_value += position['size'] * position['current_price']
        return total_value
    
    def rebalance_portfolio(self, target_weights: Dict, price_data: Dict):
        """포트폴리오 리밸런싱"""
        total_value = self.get_portfolio_value()
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            
            if symbol in self.positions:
                current_value = self.positions[symbol]['size'] * self.positions[symbol]['current_price']
                rebalance_amount = target_value - current_value
                
                if abs(rebalance_amount) > total_value * 0.01:  # 1% 이상 차이
                    # 리밸런싱 실행
                    self.execute_rebalance(symbol, rebalance_amount, price_data[symbol])
    
    def execute_rebalance(self, symbol: str, amount: float, price_data: pd.DataFrame):
        """리밸런싱 실행"""
        current_price = price_data['close'].iloc[-1]
        
        if amount > 0:  # 매수
            additional_shares = amount / current_price
            if symbol in self.positions:
                self.positions[symbol]['size'] += additional_shares
            else:
                self.add_position(symbol, {
                    'entry_price': current_price,
                    'size': additional_shares,
                    'current_price': current_price
                })
        else:  # 매도
            if symbol in self.positions:
                reduce_shares = abs(amount) / current_price
                self.positions[symbol]['size'] = max(0, self.positions[symbol]['size'] - reduce_shares)
                
                if self.positions[symbol]['size'] == 0:
                    self.remove_position(symbol)
        
        self.cash_balance -= amount