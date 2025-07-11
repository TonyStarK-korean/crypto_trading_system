#!/usr/bin/env python3
"""
트레일링 익절 시스템 테스트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.bollinger_breakout_strategy import BollingerBreakoutStrategy


def generate_test_data():
    """테스트용 데이터 생성"""
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq='1H')
    np.random.seed(42)
    
    # 상승 추세가 있는 데이터 생성
    base_price = 50000
    prices = [base_price]
    
    for i in range(1, len(date_range)):
        # 상승 추세 + 변동성
        trend = 0.001  # 0.1% 상승 추세
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


def test_trailing_stop():
    """트레일링 익절 테스트"""
    print("=== 트레일링 익절 시스템 테스트 ===")
    
    # 테스트 데이터 생성
    df = generate_test_data()
    print(f"테스트 데이터 생성 완료: {len(df)}개 캔들")
    
    # 전략 초기화
    strategy = BollingerBreakoutStrategy()
    
    # 백테스트 실행
    balance = 10000
    position = None
    trades = []
    
    print("\n백테스트 시작...")
    
    for i in range(200, len(df)):
        current_data = df.iloc[i]
        current_time = current_data.name
        
        # 전략 신호 생성
        signal, position = strategy.generate_signal(df, i, position)
        
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
            
            print(f"진입: {current_time} - 가격: {entry_price:,.2f}")
            
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
            
            print(f"청산: {current_time} - 가격: {exit_price:,.2f}, PnL: {pnl:,.2f}, 고점: {position.get('highest_price', 0):,.2f}")
            position = None
        
        # 트레일링 익절 로직 (포지션이 있을 때)
        elif position is not None:
            current_price = current_data['close']
            entry_price = position['entry_price']
            
            # 고점 업데이트
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                print(f"고점 업데이트: {current_time} - {current_price:,.2f}")
            
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
                
                print(f"손절: {current_time} - 가격: {current_price:,.2f}, PnL: {pnl:,.2f}")
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
                
                print(f"트레일링 익절: {current_time} - 가격: {current_price:,.2f}, PnL: {pnl:,.2f}, 고점: {position['highest_price']:,.2f}")
                position = None
    
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
        
        print(f"최종 청산: {df.index[-1]} - 가격: {final_price:,.2f}, PnL: {final_pnl:,.2f}")
    
    # 결과 분석
    print("\n=== 백테스트 결과 ===")
    print(f"초기 자본: 10,000")
    print(f"최종 자본: {balance:,.2f}")
    print(f"총 수익률: {((balance/10000)-1)*100:.2f}%")
    print(f"총 거래 수: {len([t for t in trades if 'pnl' in t])}")
    
    # 거래 상세 분석
    completed_trades = [t for t in trades if 'pnl' in t]
    if completed_trades:
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] < 0]
        
        print(f"승률: {len(winning_trades)/len(completed_trades)*100:.1f}%")
        print(f"평균 수익: {np.mean([t['pnl'] for t in winning_trades]):.2f}" if winning_trades else "평균 수익: 0")
        print(f"평균 손실: {np.mean([t['pnl'] for t in losing_trades]):.2f}" if losing_trades else "평균 손실: 0")
        
        # 청산 이유 분석
        reasons = {}
        for trade in completed_trades:
            reason = trade.get('reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        print("\n청산 이유:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count}회")


if __name__ == "__main__":
    test_trailing_stop() 