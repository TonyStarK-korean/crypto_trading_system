"""
백테스트 엔진
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..core.data_manager import DataManager
from ..core.position_manager import PositionManager, PositionSide
from ..ml.price_predictor import PricePredictor
from .performance_analyzer import PerformanceAnalyzer


class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self, initial_balance: float = 10000.0, commission: float = 0.001):
        self.initial_balance = initial_balance
        self.commission = commission
        self.data_manager = DataManager()
        self.position_manager = PositionManager(initial_balance)
        self.performance_analyzer = PerformanceAnalyzer()
        self.ml_predictor = PricePredictor()
        
        # 백테스트 결과
        self.equity_curve = []
        self.trade_log = []
        self.daily_returns = []
        
    def run_backtest(self, strategy_func: Callable, data: pd.DataFrame, 
                    strategy_params: Dict = None, use_ml: bool = True) -> Dict:
        """백테스트 실행"""
        try:
            logger.info("백테스트 시작...")
            
            # 데이터 준비
            data = self.data_manager.calculate_technical_indicators(data)
            
            # ML 모델 훈련 (선택사항)
            if use_ml:
                logger.info("ML 모델 훈련 중...")
                self.ml_predictor.train_model(data)
            
            # 백테스트 실행
            results = self._execute_backtest(strategy_func, data, strategy_params, use_ml)
            
            # 성과 분석
            performance = self.performance_analyzer.analyze_performance(
                self.equity_curve, self.trade_log, self.daily_returns
            )
            
            logger.info("백테스트 완료")
            return {
                'performance': performance,
                'equity_curve': self.equity_curve,
                'trade_log': self.trade_log,
                'daily_returns': self.daily_returns
            }
            
        except Exception as e:
            logger.error(f"백테스트 실행 오류: {e}")
            return {}
    
    def _execute_backtest(self, strategy_func: Callable, data: pd.DataFrame,
                         strategy_params: Dict, use_ml: bool):
        """백테스트 실행 로직"""
        try:
            # 초기화
            self.position_manager.reset()
            self.equity_curve = []
            self.trade_log = []
            self.daily_returns = []
            
            current_balance = self.initial_balance
            last_balance = current_balance
            
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                current_time = current_data.index[-1]
                
                # ML 예측 (선택사항)
                ml_signal = None
                if use_ml and i > 100:  # 충분한 데이터가 있을 때
                    ml_prob = self.ml_predictor.predict_probability(current_data)
                    if ml_prob is not None:
                        ml_signal = 1 if ml_prob > 0.6 else (-1 if ml_prob < 0.4 else 0)
                
                # 전략 신호 생성
                strategy_signal = strategy_func(current_data, strategy_params, ml_signal)
                
                # 거래 실행
                self._execute_trades(strategy_signal, current_price, current_time)
                
                # 포지션 가격 업데이트
                if self.position_manager.positions:
                    price_updates = {symbol: current_price 
                                   for symbol in self.position_manager.positions.keys()}
                    self.position_manager.update_position_prices(price_updates)
                
                # 잔고 업데이트
                current_balance = self.position_manager.current_balance
                
                # 수익률 계산
                if i > 0:
                    daily_return = (current_balance - last_balance) / last_balance
                    self.daily_returns.append(daily_return)
                
                # 자산 곡선 기록
                self.equity_curve.append({
                    'timestamp': current_time,
                    'balance': current_balance,
                    'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.position_manager.positions.values())
                })
                
                last_balance = current_balance
                
                # 진행상황 로그
                if i % 1000 == 0:
                    logger.info(f"백테스트 진행률: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"백테스트 실행 오류: {e}")
            return False
    
    def _execute_trades(self, signal: Dict, current_price: float, current_time: datetime):
        """거래 실행"""
        try:
            if not signal:
                return
            
            symbol = signal.get('symbol', 'BTC/USDT')
            action = signal.get('action')  # 'buy', 'sell', 'hold'
            size = signal.get('size', 0.1)
            leverage = signal.get('leverage', 1.0)
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            # 기존 포지션 확인
            existing_position = self.position_manager.positions.get(symbol)
            
            if action == 'buy' and not existing_position:
                # 롱 포지션 오픈
                success = self.position_manager.open_position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    size=size,
                    entry_price=current_price,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if success:
                    self.trade_log.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'action': 'buy',
                        'price': current_price,
                        'size': size,
                        'leverage': leverage
                    })
            
            elif action == 'sell' and not existing_position:
                # 숏 포지션 오픈
                success = self.position_manager.open_position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    size=size,
                    entry_price=current_price,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if success:
                    self.trade_log.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'action': 'sell',
                        'price': current_price,
                        'size': size,
                        'leverage': leverage
                    })
            
            elif action == 'close' and existing_position:
                # 포지션 클로즈
                success = self.position_manager.close_position(
                    symbol=symbol,
                    exit_price=current_price,
                    reason='strategy_signal'
                )
                
                if success:
                    self.trade_log.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'action': 'close',
                        'price': current_price,
                        'pnl': existing_position.realized_pnl
                    })
                    
        except Exception as e:
            logger.error(f"거래 실행 오류: {e}")
    
    def plot_results(self, save_path: str = None):
        """백테스트 결과 시각화"""
        try:
            if not self.equity_curve:
                logger.warning("시각화할 데이터가 없습니다")
                return
            
            # 자산 곡선
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['total_value'] = equity_df['balance'] + equity_df['unrealized_pnl']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 자산 곡선
            axes[0, 0].plot(equity_df['timestamp'], equity_df['total_value'], label='Total Value')
            axes[0, 0].plot(equity_df['timestamp'], equity_df['balance'], label='Balance', alpha=0.7)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 수익률 분포
            if self.daily_returns:
                axes[0, 1].hist(self.daily_returns, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Daily Returns Distribution')
                axes[0, 1].set_xlabel('Daily Return')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True)
            
            # 누적 수익률
            if self.daily_returns:
                cumulative_returns = np.cumprod(1 + np.array(self.daily_returns)) - 1
                axes[1, 0].plot(cumulative_returns)
                axes[1, 0].set_title('Cumulative Returns')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Cumulative Return')
                axes[1, 0].grid(True)
            
            # 거래 히스토리
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                if 'pnl' in trade_df.columns:
                    axes[1, 1].bar(range(len(trade_df)), trade_df['pnl'])
                    axes[1, 1].set_title('Trade PnL')
                    axes[1, 1].set_xlabel('Trade Number')
                    axes[1, 1].set_ylabel('PnL')
                    axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"차트 저장 완료: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"시각화 오류: {e}")
    
    def save_results(self, filepath: str):
        """백테스트 결과 저장"""
        try:
            results = {
                'equity_curve': self.equity_curve,
                'trade_log': self.trade_log,
                'daily_returns': self.daily_returns,
                'performance': self.performance_analyzer.analyze_performance(
                    self.equity_curve, self.trade_log, self.daily_returns
                )
            }
            
            # CSV로 저장
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(f"{filepath}_equity.csv", index=False)
            
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                trade_df.to_csv(f"{filepath}_trades.csv", index=False)
            
            logger.info(f"백테스트 결과 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"결과 저장 오류: {e}")
    
    def compare_strategies(self, strategies: Dict[str, Callable], data: pd.DataFrame) -> Dict:
        """여러 전략 비교"""
        try:
            results = {}
            
            for strategy_name, strategy_func in strategies.items():
                logger.info(f"전략 실행 중: {strategy_name}")
                
                # 백테스트 실행
                result = self.run_backtest(strategy_func, data)
                
                if result:
                    results[strategy_name] = result['performance']
            
            # 결과 비교
            comparison_df = pd.DataFrame(results).T
            
            logger.info("전략 비교 결과:")
            logger.info(comparison_df)
            
            return results
            
        except Exception as e:
            logger.error(f"전략 비교 오류: {e}")
            return {} 