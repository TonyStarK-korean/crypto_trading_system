#!/usr/bin/env python3
"""
고급 백테스트 엔진
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import os
import json
from datetime import datetime
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.bollinger_breakout_strategy import BollingerBreakoutStrategy


class AdvancedBacktestEngine:
    """고급 백테스트 엔진"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.strategies = {
            'Bollinger Breakout': BollingerBreakoutStrategy()
        }
    
    def get_strategy(self, strategy_name: str):
        """전략 가져오기"""
        return self.strategies.get(strategy_name)
    
    def generate_sample_data(self, days: int = 365) -> pd.DataFrame:
        """샘플 데이터 생성"""
        date_range = pd.date_range(start="2023-01-01", periods=days*24, freq='1H')
        np.random.seed(42)
        
        # 상승 추세가 있는 데이터 생성
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, len(date_range)):
            # 상승 추세 + 변동성
            trend = 0.0001  # 0.01% 상승 추세
            volatility = np.random.normal(0, 0.02)
            change = trend + volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = {
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.normal(1000, 200, len(date_range))
        }
        
        df = pd.DataFrame(data, index=date_range)
        
        # high, low 조정
        for i in range(len(df)):
            df.loc[df.index[i], 'high'] = max(df.loc[df.index[i], 'high'], df.loc[df.index[i], 'close'])
            df.loc[df.index[i], 'low'] = min(df.loc[df.index[i], 'low'], df.loc[df.index[i], 'close'])
            df.loc[df.index[i], 'open'] = df.loc[df.index[i], 'close'] * (1 + np.random.normal(0, 0.005))
        
        return df
    
    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """저장된 데이터 로드"""
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                return None
            
            # 가장 최근 파일 찾기
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if not files:
                return None
            
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
            file_path = os.path.join(data_dir, latest_file)
            
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # BTC/USDT 컬럼 찾기
            btc_columns = [col for col in df.columns if 'BTCUSDT' in col]
            if btc_columns:
                # BTC 데이터만 추출
                btc_data = df[btc_columns].copy()
                btc_data.columns = ['open', 'high', 'low', 'close', 'volume']
                return btc_data
            
            return None
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """데이터 저장"""
        try:
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            
            filename = f"{data_dir}/{symbol.replace('/', '')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename)
            print(f"데이터 저장 완료: {filename}")
            
        except Exception as e:
            print(f"데이터 저장 오류: {e}")
    
    def run_backtest(self, df: pd.DataFrame, strategy_name: str) -> Optional[Dict]:
        """백테스트 실행"""
        try:
            # 전략 초기화
            strategy = self.get_strategy(strategy_name)
            if strategy is None:
                print(f"전략을 찾을 수 없습니다: {strategy_name}")
                return None
            
            print(f"\n=== {strategy_name} 백테스트 시작 ===")
            print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"총 캔들 수: {len(df)}")
            print(f"초기 자본: {self.initial_balance:,.2f}")
            print("="*50)
            
            # 백테스트 변수 초기화
            balance = self.initial_balance
            position = None
            trades = []
            equity_curve = []
            
            # 백테스트 실행
            for i in range(200, len(df)):  # 200개 캔들 후부터 시작
                current_data = df.iloc[i]
                current_time = current_data.name
                
                # 전략 신호 생성 (자산 정보 전달)
                signal, position = strategy.generate_signal(df, i, position, balance)
                
                # 거래 실행
                if signal == 'buy' and position is None:
                    # 롱 포지션 오픈
                    entry_price = current_data['close']
                    position_size = strategy.calculate_position_size(balance, entry_price)
                    
                    position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'size': position_size,
                        'highest_price': entry_price  # 트레일링 익절용 고점 초기화
                    }
                    
                    trades.append({
                        'time': str(current_time),
                        'action': 'buy',
                        'price': entry_price,
                        'balance': balance,
                        'size': position_size
                    })
                    
                elif signal == 'sell' and position is not None:
                    # 포지션 클로즈
                    exit_price = current_data['close']
                    pnl = (exit_price - position['entry_price']) * position['size']
                    balance += pnl
                    
                    trades.append({
                        'time': str(current_time),
                        'action': 'sell',
                        'price': exit_price,
                        'pnl': pnl,
                        'balance': balance,
                        'size': position['size'],
                        'highest_price': position.get('highest_price', 0)
                    })
                    position = None
                
                # 트레일링 익절 로직 (포지션이 있을 때)
                elif position is not None:
                    current_price = current_data['close']
                    entry_price = position['entry_price']
                    
                    # 고점 업데이트
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                    
                    # 손절: 진입가 대비 -2%
                    stop_loss = entry_price * 0.98
                    
                    # 트레일링 익절: 고점 대비 -1.5% 하락 시 청산
                    trailing_stop = position['highest_price'] * 0.985
                    
                    if current_price <= stop_loss:
                        pnl = (current_price - entry_price) * position['size']
                        balance += pnl
                        
                        trades.append({
                            'time': str(current_time),
                            'action': 'sell',
                            'price': current_price,
                            'pnl': pnl,
                            'balance': balance,
                            'size': position['size'],
                            'highest_price': position['highest_price'],
                            'reason': 'stop_loss'
                        })
                        position = None
                        
                    elif current_price <= trailing_stop and current_price > entry_price:
                        pnl = (current_price - entry_price) * position['size']
                        balance += pnl
                        
                        trades.append({
                            'time': str(current_time),
                            'action': 'sell',
                            'price': current_price,
                            'pnl': pnl,
                            'balance': balance,
                            'size': position['size'],
                            'highest_price': position['highest_price'],
                            'reason': 'trailing_stop'
                        })
                        position = None
                
                # 자산 곡선 업데이트
                current_equity = balance
                if position is not None:
                    current_price = current_data['close']
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_curve.append({
                    'time': str(current_time),
                    'equity': current_equity,
                    'balance': balance,
                    'position_value': current_equity - balance if position is not None else 0
                })
            
            # 최종 포지션 정리
            if position is not None:
                final_price = df.iloc[-1]['close']
                final_pnl = (final_price - position['entry_price']) * position['size']
                balance += final_pnl
                
                trades.append({
                    'time': str(df.index[-1]),
                    'action': 'sell',
                    'price': final_price,
                    'pnl': final_pnl,
                    'balance': balance,
                    'size': position['size'],
                    'highest_price': position.get('highest_price', 0),
                    'reason': 'final'
                })
            
            # 백테스트 결과
            total_return = ((balance / self.initial_balance) - 1) * 100
            completed_trades = [t for t in trades if 'pnl' in t]
            
            print(f"\n=== 백테스트 완료 ===")
            print(f"최종 자산: {balance:,.2f}")
            print(f"총 수익률: {total_return:.2f}%")
            print(f"총 거래 수: {len(completed_trades)}")
            
            if completed_trades:
                winning_trades = [t for t in completed_trades if t['pnl'] > 0]
                losing_trades = [t for t in completed_trades if t['pnl'] < 0]
                
                print(f"승률: {len(winning_trades)/len(completed_trades)*100:.1f}%")
                print(f"평균 수익: {np.mean([t['pnl'] for t in winning_trades]):.2f}" if winning_trades else "평균 수익: 0")
                print(f"평균 손실: {np.mean([t['pnl'] for t in losing_trades]):.2f}" if losing_trades else "평균 손실: 0")
            
            return {
                'equity_curve': equity_curve,
                'trades': trades,
                'final_balance': balance,
                'total_return': total_return
            }
            
        except Exception as e:
            print(f"백테스트 실행 오류: {e}")
            return None
    
    def analyze_performance(self, equity_curve: List[Dict], trades: List[Dict]) -> Dict:
        """성과 분석"""
        try:
            if not equity_curve:
                return {}
            
            # 기본 통계
            initial_equity = equity_curve[0]['equity']
            final_equity = equity_curve[-1]['equity']
            total_return = ((final_equity / initial_equity) - 1) * 100
            
            # 최대 낙폭 계산
            peak = initial_equity
            max_drawdown = 0
            
            for point in equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']
                drawdown = (peak - point['equity']) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # 거래 통계
            completed_trades = [t for t in trades if 'pnl' in t]
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            return {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(completed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            }
            
        except Exception as e:
            print(f"성과 분석 오류: {e}")
            return {}
    
    def generate_report(self, performance: Dict, trades: List[Dict]) -> str:
        """보고서 생성"""
        try:
            report = "\n" + "="*60 + "\n"
            report += "백테스트 결과 보고서\n"
            report += "="*60 + "\n"
            
            if not performance:
                report += "성과 데이터가 없습니다.\n"
                return report
            
            report += f"총 수익률: {performance.get('total_return', 0):.2f}%\n"
            report += f"최대 낙폭: {performance.get('max_drawdown', 0):.2f}%\n"
            report += f"승률: {performance.get('win_rate', 0):.1f}%\n"
            report += f"총 거래 수: {performance.get('total_trades', 0)}\n"
            report += f"승리 거래: {performance.get('winning_trades', 0)}\n"
            report += f"패배 거래: {performance.get('losing_trades', 0)}\n"
            report += f"평균 수익: {performance.get('avg_win', 0):.2f}\n"
            report += f"평균 손실: {performance.get('avg_loss', 0):.2f}\n"
            report += f"수익 팩터: {performance.get('profit_factor', 0):.2f}\n"
            
            return report
            
        except Exception as e:
            return f"보고서 생성 오류: {e}"
    
    def save_results(self, results: Dict, strategy_name: str):
        """결과 저장"""
        try:
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{results_dir}/{strategy_name.replace(' ', '_')}_{timestamp}.json"
            
            # JSON 직렬화 가능한 형태로 변환
            serializable_results = {
                'strategy': strategy_name,
                'timestamp': timestamp,
                'performance': results.get('performance', {}),
                'final_balance': results.get('final_balance', 0),
                'total_return': results.get('total_return', 0),
                'trades_count': len(results.get('trades', [])),
                'equity_curve_count': len(results.get('equity_curve', []))
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"결과 저장 완료: {filename}")
            
        except Exception as e:
            print(f"결과 저장 오류: {e}") 