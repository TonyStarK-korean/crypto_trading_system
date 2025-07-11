#!/usr/bin/env python3
"""
간단한 백테스트 실행 파일 (matplotlib 없이 작동)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class SimpleBacktestEngine:
    """간단한 백테스트 엔진"""
    
    def __init__(self, initial_balance: float = 10000.0, commission: float = 0.001):
        self.initial_balance = initial_balance
        self.commission = commission
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def generate_sample_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """샘플 데이터 생성"""
        logger.info("샘플 데이터 생성 중...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)
        
        # 기본 가격 생성 (랜덤 워크)
        base_price = 50000
        returns = np.random.normal(0, 0.02, len(date_range))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
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
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        # 이동평균
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['ma_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def simple_ma_strategy(self, data: pd.DataFrame, current_index: int) -> dict:
        """간단한 이동평균 교차 전략"""
        if current_index < 50:
            return {}
        
        current_data = data.iloc[current_index]
        prev_data = data.iloc[current_index - 1]
        
        current_price = current_data['close']
        current_ma_20 = current_data['ma_20']
        current_ma_50 = current_data['ma_50']
        prev_ma_20 = prev_data['ma_20']
        prev_ma_50 = prev_data['ma_50']
        current_rsi = current_data['rsi'] if not pd.isna(current_data['rsi']) else 50
        
        # 골든 크로스 (단기선이 장기선을 상향 돌파)
        if (prev_ma_20 <= prev_ma_50 and current_ma_20 > current_ma_50 and 
            current_rsi < 70):
            return {
                'action': 'buy',
                'price': current_price,
                'size': 0.1
            }
        
        # 데드 크로스 (단기선이 장기선을 하향 돌파)
        elif (prev_ma_20 >= prev_ma_50 and current_ma_20 < current_ma_50 and 
              current_rsi > 30):
            return {
                'action': 'sell',
                'price': current_price,
                'size': 0.1
            }
        
        return {}
    
    def run_backtest(self, data: pd.DataFrame, symbol: str = "BTC/USDT", strategy_name: str = "이동평균 교차") -> dict:
        """백테스트 실행"""
        return self.run_backtest_with_strategy(data, symbol, strategy_name, None)
    
    def run_backtest_with_strategy(self, data: pd.DataFrame, symbol: str = "BTC/USDT", strategy_name: str = "이동평균 교차", external_strategy=None) -> dict:
        """백테스트 실행"""
        print("🔄 백테스트 실행 중...")
        
        # 전략 초기화
        if external_strategy is not None:
            strategy = external_strategy
        elif strategy_name == "급등 초입 진입전략":
            from src.strategies.pump_detection_strategy import PumpDetectionStrategy
            strategy = PumpDetectionStrategy()
        else:
            strategy = None
        
        position = None
        
        for i in range(50, len(data)):
            current_data = data.iloc[i]
            current_time = current_data.name
            
            # 전략 신호 생성
            if strategy:
                signal, position = strategy.generate_signal(data, i, position, self.balance)
            else:
                signal = self.simple_ma_strategy(data, i)
                if signal.get('action') == 'buy':
                    signal = 'buy'
                elif signal.get('action') == 'sell':
                    signal = 'sell'
                else:
                    signal = None
            
            
            # 거래 실행
            if signal == 'buy' and self.position is None:
                # 첫 진입 (비중의 30%만)
                entry_price = current_data['close']
                position_usdt = self.balance * 0.0333  # 비중 10%의 30% (3.33%)
                position_size = position_usdt / entry_price  # 실제 코인 수량
                cost = position_usdt * (1 + self.commission)
                
                if cost <= self.balance:
                    self.position = {
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'size': position_size,
                        'usdt': position_usdt,
                        'entry_type': 'first'  # 첫 진입 표시
                    }
                    self.balance -= cost
                    
                    self.trades.append({
                        'time': str(current_time),
                        'action': 'buy',
                        'price': entry_price,
                        'size': position_size,
                        'usdt': position_usdt,
                        'balance': self.balance
                    })
                    
                    # 진입 로그 (고도화된 정보 포함)
                    leverage = position.get('leverage', 1.0) if position else 1.0
                    market_regime = position.get('market_regime', 'unknown') if position else 'unknown'
                    total_equity = self.balance + (entry_price * position_size)
                    
                    print(f"🚀 진입: {current_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   심볼: {symbol} | 비중: {(position_usdt/self.balance)*100:.2f}% | 레버리지: {leverage:.1f}x")
                    print(f"   진입가: {entry_price:,.4f} | 진입금액: {position_usdt:,.2f} USDT | 수량: {position_size:,.4f}")
                    print(f"   전략: {strategy_name} | 시장상태: {market_regime} | 현재 총 자산: {total_equity:,.2f} USDT")
                    
                    # 고급 전략 추가 정보
                    if position and strategy_name == "고급 통합전략":
                        print(f"   ML신뢰도: {position.get('ml_confidence', 0):.3f} | MTF신호: {position.get('mtf_signal', 0):.3f}")
                        print(f"   진입임계값: {position.get('entry_threshold', 0):.3f} | 조정신호: {position.get('adjusted_signal', 0):.3f}")
                    print()
            
            elif signal == 'buy' and self.position is not None:
                # 이미 포지션이 있으면 추가매수하지 않음
                pass
            
            elif signal == 'sell' and self.position is not None:
                # 포지션 클로즈
                exit_price = current_data['close']
                position_size = self.position['size']
                entry_value = self.position['entry_price'] * position_size
                exit_value = exit_price * position_size
                entry_fee = entry_value * self.commission
                exit_fee = exit_value * self.commission
                pnl = exit_value - entry_value - entry_fee - exit_fee
                self.balance += entry_value + pnl  # 진입금액+수익금(손실금)
                hold_time = current_time - self.position['entry_time']
                hold_hours = hold_time.total_seconds() / 3600
                hold_minutes = (hold_time.total_seconds() % 3600) / 60
                profit_rate = (pnl / entry_value) * 100 if entry_value != 0 else 0
                
                # 실제 투입된 총 비중 계산
                total_invested_ratio = (self.position['usdt'] / self.initial_balance) * 100
                
                self.trades.append({
                    'time': str(current_time),
                    'action': 'sell',
                    'price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'balance': self.balance
                })
                
                # 청산 로그
                leverage = 1.0
                status = "🟢 수익" if pnl > 0 else "🔴 손실"
                
                # 청산 타입에 따른 메시지
                if 'entry_type' in self.position:
                    if self.position['entry_type'] == 'first':
                        exit_type = "손절/익절"
                    elif self.position['entry_type'] == 'additional':
                        exit_type = "추가매수 후 청산"
                    else:
                        exit_type = "트레일링 매도"
                else:
                    exit_type = "청산"
                
                print(f"{status} {exit_type}: {current_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"   심볼: {symbol} | 비중: {total_invested_ratio:.2f}% | 레버리지: {leverage}x")
                print(f"   청산가: {exit_price:,.4f} | 청산금액: {exit_value:,.2f} USDT | 수량: {position_size:,.4f}")
                print(f"   전략: {strategy_name} | 수익률: {profit_rate:+.2f}% | 수익금: {pnl:+.2f} USDT")
                print(f"   보유시간: {hold_hours:.0f}시간 {hold_minutes:.0f}분 | 현재 총 자산: {self.balance:,.2f} USDT")
                print()
                self.position = None
                position = None  # 전략의 position도 초기화
            
            # 자산 곡선 기록
            current_equity = self.balance
            if self.position is not None:
                unrealized_pnl = (current_data['close'] - self.position['entry_price']) * self.position['size']
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'time': str(current_time),
                'equity': current_equity,
                'balance': self.balance
            })
        
        # 최종 포지션 정리 (로그 삭제)
        if self.position is not None:
            final_price = data.iloc[-1]['close']
            final_pnl = (final_price - self.position['entry_price']) * self.position['size']
            self.balance += final_price * self.position['size'] * (1 - self.commission)
            
            self.trades.append({
                'time': str(data.index[-1]),
                'action': 'sell',
                'price': final_price,
                'size': self.position['size'],
                'pnl': final_pnl,
                'balance': self.balance
            })
        
        return self.calculate_performance()
    
    def calculate_performance(self) -> dict:
        """성과 계산"""
        # 기본 성과 정보
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        final_balance = self.balance
        
        if not self.trades:
            # 거래가 없는 경우 기본 정보만 반환
            return {
                'performance': {
                    'total_return': total_return,
                    'final_balance': final_balance,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'profit_factor': 0,
                    'avg_profit': 0,
                    'avg_loss': 0
                },
                'trades': [],
                'equity_curve': self.equity_curve,
                'weekly_performance': {},
                'monthly_performance': {}
            }
        
        # 거래 분석
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        
        total_trades = len(sell_trades)
        profitable_trades = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 수익/손실 분석
        profits = [t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0]
        losses = [t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        # 최대 낙폭 계산
        equity_values = [e['equity'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # 주별/월별 성과 계산
        weekly_performance = self.calculate_weekly_performance()
        monthly_performance = self.calculate_monthly_performance()
        
        return {
            'performance': {
                'total_return': total_return,
                'final_balance': final_balance,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss
            },
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'weekly_performance': weekly_performance,
            'monthly_performance': monthly_performance
        }
    
    def calculate_weekly_performance(self) -> dict:
        """주별 성과 계산"""
        if not self.trades:
            return {}
        
        # 거래 데이터를 주별로 그룹화
        weekly_data = {}
        for trade in self.trades:
            if trade['action'] == 'sell' and 'pnl' in trade:
                trade_date = pd.to_datetime(trade['time'])
                week_key = trade_date.strftime('%Y-W%U')
                
                if week_key not in weekly_data:
                    weekly_data[week_key] = {
                        'trades': 0,
                        'profit': 0,
                        'loss': 0,
                        'pnl': 0
                    }
                
                weekly_data[week_key]['trades'] += 1
                weekly_data[week_key]['pnl'] += trade['pnl']
                
                if trade['pnl'] > 0:
                    weekly_data[week_key]['profit'] += trade['pnl']
                else:
                    weekly_data[week_key]['loss'] += abs(trade['pnl'])
        
        return weekly_data
    
    def calculate_monthly_performance(self) -> dict:
        """월별 성과 계산"""
        if not self.trades:
            return {}
        
        # 거래 데이터를 월별로 그룹화
        monthly_data = {}
        for trade in self.trades:
            if trade['action'] == 'sell' and 'pnl' in trade:
                trade_date = pd.to_datetime(trade['time'])
                month_key = trade_date.strftime('%Y-%m')
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        'trades': 0,
                        'profit': 0,
                        'loss': 0,
                        'pnl': 0
                    }
                
                monthly_data[month_key]['trades'] += 1
                monthly_data[month_key]['pnl'] += trade['pnl']
                
                if trade['pnl'] > 0:
                    monthly_data[month_key]['profit'] += trade['pnl']
                else:
                    monthly_data[month_key]['loss'] += abs(trade['pnl'])
        
        return monthly_data


def main():
    """메인 함수"""
    try:
        logger.info("=== 간단한 백테스트 시작 ===")
        
        # 기본 설정
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        initial_balance = 10000.0
        commission = 0.001
        
        logger.info(f"기간: {start_date} ~ {end_date}")
        logger.info(f"초기 자본: {initial_balance:,.0f}")
        logger.info(f"수수료: {commission*100:.1f}%")
        
        # 백테스트 엔진 초기화
        engine = SimpleBacktestEngine(initial_balance=initial_balance, commission=commission)
        
        # 데이터 생성
        data = engine.generate_sample_data(start_date, end_date)
        data = engine.calculate_technical_indicators(data)
        
        logger.info(f"데이터 생성 완료: {len(data)} 개의 캔들")
        
        # 백테스트 실행
        results = engine.run_backtest(data)
        
        if results:
            performance = results['performance']
            
            print("\n" + "="*60)
            print("📊 백테스트 결과")
            print("="*60)
            print(f"총 수익률: {performance.get('total_return', 0):.2f}%")
            print(f"최종 자본: {performance.get('final_balance', 0):,.0f} USDT")
            print(f"최대 낙폭: {performance.get('max_drawdown', 0):.2f}%")
            print(f"승률: {performance.get('win_rate', 0):.2f}%")
            print(f"총 거래 수: {performance.get('total_trades', 0)}")
            print(f"손익비: {performance.get('profit_factor', 0):.2f}")
            print(f"평균 수익: {performance.get('avg_profit', 0):,.2f} USDT")
            print(f"평균 손실: {performance.get('avg_loss', 0):,.2f} USDT")
            
            # 주별 성과 보고서
            if results.get('weekly_performance'):
                print("\n" + "="*60)
                print("📅 주별 성과 보고서")
                print("="*60)
                for week, data in results['weekly_performance'].items():
                    if data['trades'] > 0:
                        win_rate = (data['profit'] / (data['profit'] + data['loss']) * 100) if (data['profit'] + data['loss']) > 0 else 0
                        print(f"{week}: 거래 {data['trades']}회 | PnL: {data['pnl']:+,.0f} | 승률: {win_rate:.1f}%")
            
            # 월별 성과 보고서
            if results.get('monthly_performance'):
                print("\n" + "="*60)
                print("📅 월별 성과 보고서")
                print("="*60)
                for month, data in results['monthly_performance'].items():
                    if data['trades'] > 0:
                        win_rate = (data['profit'] / (data['profit'] + data['loss']) * 100) if (data['profit'] + data['loss']) > 0 else 0
                        print(f"{month}: 거래 {data['trades']}회 | PnL: {data['pnl']:+,.0f} | 승률: {win_rate:.1f}%")
            
            # 최종 성과 보고서
            print("\n" + "="*60)
            print("📈 최종 성과 보고서")
            print("="*60)
            print(f"초기 자본: {initial_balance:,.0f} USDT")
            print(f"최종 자본: {performance.get('final_balance', 0):,.0f} USDT")
            print(f"총 수익: {performance.get('final_balance', 0) - initial_balance:+,.0f} USDT")
            print(f"총 수익률: {performance.get('total_return', 0):.2f}%")
            print(f"연간 수익률: {performance.get('total_return', 0) * (365/30):.2f}%")  # 30일 기준으로 연간화
            print(f"샤프 비율: {performance.get('profit_factor', 0):.2f}")
            print(f"최대 낙폭: {performance.get('max_drawdown', 0):.2f}%")
            print(f"총 거래 수: {performance.get('total_trades', 0)}")
            print(f"승률: {performance.get('win_rate', 0):.2f}%")
            print("="*60)
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"results/simple_backtest_{timestamp}.json"
            os.makedirs("results", exist_ok=True)
            
            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"결과 저장 완료: {results_filename}")
            logger.info("백테스트가 성공적으로 완료되었습니다!")
            
        else:
            logger.error("백테스트 실행에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"백테스트 실행 오류: {e}")


if __name__ == "__main__":
    # 로깅 설정
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    main() 