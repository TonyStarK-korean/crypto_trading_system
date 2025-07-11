"""
성과 분석 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


class PerformanceAnalyzer:
    """성과 분석 시스템"""
    
    def __init__(self):
        pass
    
    def analyze_performance(self, equity_curve: List[Dict], trade_log: List[Dict], 
                          daily_returns: List[float]) -> Dict:
        """성과 분석"""
        try:
            if not equity_curve:
                return {}
            
            # 기본 지표 계산
            total_return = self._calculate_total_return(equity_curve)
            annual_return = self._calculate_annual_return(equity_curve)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            win_rate = self._calculate_win_rate(trade_log)
            profit_factor = self._calculate_profit_factor(trade_log)
            
            # 추가 지표
            avg_trade = self._calculate_avg_trade(trade_log)
            max_consecutive_losses = self._calculate_max_consecutive_losses(trade_log)
            calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trade_log),
                'avg_trade': avg_trade,
                'max_consecutive_losses': max_consecutive_losses,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            logger.error(f"성과 분석 오류: {e}")
            return {}
    
    def _calculate_total_return(self, equity_curve: List[Dict]) -> float:
        """총 수익률 계산"""
        try:
            if len(equity_curve) < 2:
                return 0.0
            
            initial_value = equity_curve[0]['balance']
            final_value = equity_curve[-1]['balance'] + equity_curve[-1]['unrealized_pnl']
            
            return ((final_value / initial_value) - 1) * 100
            
        except Exception as e:
            logger.error(f"총 수익률 계산 오류: {e}")
            return 0.0
    
    def _calculate_annual_return(self, equity_curve: List[Dict]) -> float:
        """연간 수익률 계산"""
        try:
            if len(equity_curve) < 2:
                return 0.0
            
            start_date = equity_curve[0]['timestamp']
            end_date = equity_curve[-1]['timestamp']
            
            # 기간 계산 (일 단위)
            days = (end_date - start_date).days
            if days == 0:
                return 0.0
            
            total_return = self._calculate_total_return(equity_curve) / 100
            annual_return = ((1 + total_return) ** (365 / days) - 1) * 100
            
            return annual_return
            
        except Exception as e:
            logger.error(f"연간 수익률 계산 오류: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """최대 낙폭 계산"""
        try:
            if not equity_curve:
                return 0.0
            
            # 총 자산 곡선 생성
            total_values = []
            for point in equity_curve:
                total_value = point['balance'] + point['unrealized_pnl']
                total_values.append(total_value)
            
            # 최대 낙폭 계산
            peak = total_values[0]
            max_dd = 0.0
            
            for value in total_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            
            return max_dd * 100
            
        except Exception as e:
            logger.error(f"최대 낙폭 계산 오류: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """샤프 비율 계산"""
        try:
            if not daily_returns:
                return 0.0
            
            returns = np.array(daily_returns)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # 연간화 (일간 수익률을 연간으로 변환)
            sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252))
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"샤프 비율 계산 오류: {e}")
            return 0.0
    
    def _calculate_win_rate(self, trade_log: List[Dict]) -> float:
        """승률 계산"""
        try:
            if not trade_log:
                return 0.0
            
            winning_trades = sum(1 for trade in trade_log if trade.get('pnl', 0) > 0)
            total_trades = len(trade_log)
            
            return (winning_trades / total_trades) * 100
            
        except Exception as e:
            logger.error(f"승률 계산 오류: {e}")
            return 0.0
    
    def _calculate_profit_factor(self, trade_log: List[Dict]) -> float:
        """손익비 계산"""
        try:
            if not trade_log:
                return 0.0
            
            gross_profit = sum(trade.get('pnl', 0) for trade in trade_log if trade.get('pnl', 0) > 0)
            gross_loss = abs(sum(trade.get('pnl', 0) for trade in trade_log if trade.get('pnl', 0) < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
            
        except Exception as e:
            logger.error(f"손익비 계산 오류: {e}")
            return 0.0
    
    def _calculate_avg_trade(self, trade_log: List[Dict]) -> float:
        """평균 거래 수익 계산"""
        try:
            if not trade_log:
                return 0.0
            
            total_pnl = sum(trade.get('pnl', 0) for trade in trade_log)
            return total_pnl / len(trade_log)
            
        except Exception as e:
            logger.error(f"평균 거래 수익 계산 오류: {e}")
            return 0.0
    
    def _calculate_max_consecutive_losses(self, trade_log: List[Dict]) -> int:
        """최대 연속 손실 계산"""
        try:
            if not trade_log:
                return 0
            
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in trade_log:
                if trade.get('pnl', 0) < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            logger.error(f"최대 연속 손실 계산 오류: {e}")
            return 0
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """칼마 비율 계산"""
        try:
            if max_drawdown == 0:
                return 0.0
            
            return annual_return / max_drawdown
            
        except Exception as e:
            logger.error(f"칼마 비율 계산 오류: {e}")
            return 0.0
    
    def generate_report(self, performance: Dict) -> str:
        """성과 보고서 생성"""
        try:
            report = """
=== 성과 분석 보고서 ===

📊 기본 지표:
• 총 수익률: {total_return:.2f}%
• 연간 수익률: {annual_return:.2f}%
• 최대 낙폭: {max_drawdown:.2f}%
• 샤프 비율: {sharpe_ratio:.2f}

📈 거래 지표:
• 총 거래 수: {total_trades}
• 승률: {win_rate:.2f}%
• 손익비: {profit_factor:.2f}
• 평균 거래 수익: {avg_trade:.2f}

⚠️ 리스크 지표:
• 최대 연속 손실: {max_consecutive_losses}
• 칼마 비율: {calmar_ratio:.2f}

""".format(**performance)
            
            return report
            
        except Exception as e:
            logger.error(f"보고서 생성 오류: {e}")
            return "보고서 생성 실패" 