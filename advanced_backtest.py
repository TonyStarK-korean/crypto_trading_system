#!/usr/bin/env python3
"""
향상된 백테스트 시스템
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import sys
import ccxt
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.bollinger_breakout_strategy import BollingerBreakoutStrategy


class AdvancedBacktestEngine:
    """향상된 백테스트 엔진"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.strategies = {
            'bollinger_breakout': BollingerBreakoutStrategy()
        }
        
    def generate_sample_data(self, symbol: str = "BTC/USDT", start_date: str = "2023-01-01", 
                           end_date: str = "2023-12-31") -> pd.DataFrame:
        """샘플 데이터 생성"""
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
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str = "1h"):
        """데이터 저장"""
        try:
            filename = f"data/{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename)
            logger.info(f"데이터 저장 완료: {filename}")
            return filename
        except Exception as e:
            logger.error(f"데이터 저장 오류: {e}")
            return None
    
    def load_data(self, symbol: str, timeframe: str = "1h") -> Optional[pd.DataFrame]:
        """저장된 데이터 로드"""
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                return None
            
            # 데이터 파일 검색
            pattern = f"{symbol.replace('/', '_')}_{timeframe}_*.csv"
            files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol.replace('/', '_')}_{timeframe}_")]
            
            if not files:
                return None
            
            # 가장 최근 파일 로드
            latest_file = sorted(files)[-1]
            filepath = os.path.join(data_dir, latest_file)
            
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"데이터 로드 완료: {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            return None
    
    def run_backtest(self, df: pd.DataFrame, strategy_name: str) -> Dict:
        """백테스트 실행"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"전략을 찾을 수 없습니다: {strategy_name}")
                return {}
            
            strategy = self.strategies[strategy_name]
            balance = self.initial_balance
            position = None
            trades = []
            equity_curve = []
            
            logger.info(f"백테스트 시작: {strategy_name}")
            logger.info(f"초기 자본: {balance:,.0f}")
            
            for i in range(200, len(df)):  # 200개 캔들 이후부터 시작
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
                        'size': position_size
                    }
                    
                    trades.append({
                        'time': str(current_time),
                        'action': 'buy',
                        'price': entry_price,
                        'balance': balance,
                        'size': position_size
                    })
                    
                    logger.info(f"진입: {current_time} - 가격: {entry_price:,.2f}, 크기: {position_size:.4f}")
                    
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
                        'size': position['size']
                    })
                    
                    logger.info(f"청산: {current_time} - 가격: {exit_price:,.2f}, PnL: {pnl:,.2f}")
                    position = None
                
                # 자산 곡선 기록
                current_equity = balance
                if position is not None:
                    unrealized_pnl = (current_data['close'] - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_curve.append({
                    'time': str(current_time),
                    'equity': current_equity,
                    'balance': balance
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
                    'size': position['size']
                })
                
                logger.info(f"최종 청산: {df.index[-1]} - 가격: {final_price:,.2f}, PnL: {final_pnl:,.2f}")
            
            return {
                'equity_curve': equity_curve,
                'trades': trades,
                'final_balance': balance
            }
            
        except Exception as e:
            logger.error(f"백테스트 실행 오류: {e}")
            return {}
    
    def analyze_performance(self, equity_curve: List[Dict], trades: List[Dict]) -> Dict:
        """성과 분석"""
        try:
            if not equity_curve:
                return {}
            
            final_equity = equity_curve[-1]['equity']
            total_return = ((final_equity / self.initial_balance) - 1) * 100
            
            # 최대 낙폭 계산
            peak = self.initial_balance
            max_dd = 0
            for point in equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']
                dd = (peak - point['equity']) / peak
                if dd > max_dd:
                    max_dd = dd
            
            # 거래 통계
            completed_trades = [t for t in trades if 'pnl' in t]
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] < 0]
            
            total_trades = len(completed_trades)
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            
            avg_profit = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            # 주별/월별 성과
            weekly_performance = self._calculate_periodic_performance(equity_curve, 'W')
            monthly_performance = self._calculate_periodic_performance(equity_curve, 'M')
            
            return {
                'total_return': total_return,
                'max_drawdown': max_dd * 100,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'final_balance': final_equity,
                'weekly_performance': weekly_performance,
                'monthly_performance': monthly_performance
            }
            
        except Exception as e:
            logger.error(f"성과 분석 오류: {e}")
            return {}
    
    def _calculate_periodic_performance(self, equity_curve: List[Dict], period: str) -> Dict:
        """주별/월별 성과 계산"""
        try:
            df = pd.DataFrame(equity_curve)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # 기간별 그룹화
            if period == 'W':
                grouped = df.resample('W').last()
            else:  # 'M'
                grouped = df.resample('M').last()
            
            performance = {}
            for date, row in grouped.iterrows():
                if pd.notna(row['equity']):
                    period_return = ((row['equity'] / self.initial_balance) - 1) * 100
                    performance[date.strftime('%Y-%m-%d')] = {
                        'equity': row['equity'],
                        'return': period_return
                    }
            
            return performance
            
        except Exception as e:
            logger.error(f"기간별 성과 계산 오류: {e}")
            return {}
    
    def generate_report(self, performance: Dict, trades: List[Dict]) -> str:
        """성과 보고서 생성"""
        try:
            report = f"""
{'='*60}
백테스트 성과 보고서
{'='*60}

📊 기본 지표:
• 총 수익률: {performance.get('total_return', 0):.2f}%
• 최대 낙폭: {performance.get('max_drawdown', 0):.2f}%
• 승률: {performance.get('win_rate', 0):.2f}%
• 총 거래 수: {performance.get('total_trades', 0)}
• 손익비: {performance.get('profit_factor', 0):.2f}
• 최종 자산: {performance.get('final_balance', 0):,.0f}

📈 거래 통계:
• 평균 수익: {performance.get('avg_profit', 0):.2f}
• 평균 손실: {performance.get('avg_loss', 0):.2f}

📅 주별 성과:
"""
            
            weekly_perf = performance.get('weekly_performance', {})
            for week, data in list(weekly_perf.items())[-4:]:  # 최근 4주
                report += f"• {week}: {data['return']:.2f}% (자산: {data['equity']:,.0f})\n"
            
            report += f"""
📅 월별 성과:
"""
            
            monthly_perf = performance.get('monthly_performance', {})
            for month, data in list(monthly_perf.items())[-3:]:  # 최근 3개월
                report += f"• {month}: {data['return']:.2f}% (자산: {data['equity']:,.0f})\n"
            
            report += f"""
{'='*60}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"보고서 생성 오류: {e}")
            return "보고서 생성 실패"
    
    def save_results(self, results: Dict, strategy_name: str):
        """결과 저장"""
        try:
            filename = f"results/backtest_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"결과 저장 완료: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"결과 저장 오류: {e}")
            return None


def show_menu():
    """메뉴 표시"""
    print("\n" + "="*50)
    print("코인선물 자동매매 백테스트 시스템")
    print("="*50)
    print("1. 백테스트 실행")
    print("2. 백테스트 데이터 다운로드")
    print("3. 종료")
    print("="*50)


def show_strategies():
    """사용 가능한 전략 표시"""
    strategies = {
        '1': 'bollinger_breakout'
    }
    
    print("\n사용 가능한 전략:")
    for key, strategy in strategies.items():
        print(f"{key}. {strategy}")
    
    return strategies


def show_download_menu():
    """다운로드 메뉴 표시"""
    print("\n" + "="*50)
    print("데이터 다운로드 설정")
    print("="*50)
    print("1. 최근 1개월")
    print("2. 최근 3개월")
    print("3. 최근 6개월")
    print("4. 최근 1년")
    print("5. 사용자 정의 기간 (시작일-종료일)")
    print("6. 돌아가기")
    print("="*50)


def get_custom_date_range():
    """사용자 정의 날짜 범위 입력"""
    print("\n사용자 정의 날짜 범위 설정")
    print("="*30)
    
    while True:
        try:
            start_date_str = input("시작일을 입력하세요 (YYYY-MM-DD): ")
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            
            end_date_str = input("종료일을 입력하세요 (YYYY-MM-DD): ")
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            if start_date >= end_date:
                print("시작일은 종료일보다 이전이어야 합니다.")
                continue
                
            if end_date > datetime.now():
                print("종료일은 오늘 이전이어야 합니다.")
                continue
                
            return start_date, end_date
            
        except ValueError:
            print("올바른 날짜 형식(YYYY-MM-DD)으로 입력하세요.")


def download_binance_usdtp_ohlcv_by_period(months=1, start_date=None, end_date=None, timeframe='1h', save_dir='data'):
    """바이낸스 USDT.P 전체 심볼 특정 기간 1시간봉 OHLCV 통합 다운로드 (개선된 버전)"""
    try:
        # 바이낸스 선물 시장 설정
        binance = ccxt.binance({
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            },
            'rateLimit': 1200,
            'enableRateLimit': True
        })
        
        print("바이낸스 USDT.P 심볼 목록 조회 중...")
        markets = binance.load_markets()
        
        # USDT 선물 심볼만 필터링 (개선된 필터링)
        usdtp_symbols = []
        for symbol, market in markets.items():
            if (market.get('future', False) and 
                market.get('quote', '') == 'USDT' and 
                market.get('active', False) and 
                '/USDT' in symbol and
                'USDT:USDT' in symbol):  # USDT 선물만 필터링
                usdtp_symbols.append(symbol)
        
        print(f"총 {len(usdtp_symbols)}개 USDT.P 심볼 발견!")
        
        if len(usdtp_symbols) == 0:
            print("USDT.P 심볼을 찾을 수 없습니다. 바이낸스 API 상태를 확인하세요.")
            return None

        # 기간 계산
        if start_date is None or end_date is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=months*30)
        else:
            start_time = start_date
            end_time = end_date
        
        print(f"다운로드 기간: {start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}")
        print(f"예상 캔들 수: 약 {int((end_time - start_time).total_seconds() / 3600)}개 (1시간봉 기준)")
        
        all_dfs = []
        successful_downloads = 0
        failed_downloads = 0
        
        for idx, symbol in enumerate(usdtp_symbols):
            try:
                print(f"[{idx+1}/{len(usdtp_symbols)}] {symbol} 데이터 다운로드 중...")
                
                # 기간별 데이터 다운로드
                ohlcv_data = []
                current_start = start_time
                
                while current_start < end_time:
                    try:
                        # 한 번에 최대 1000개 캔들씩 다운로드
                        ohlcv = binance.fetch_ohlcv(
                            symbol, 
                            timeframe=timeframe, 
                            since=int(current_start.timestamp() * 1000),
                            limit=1000
                        )
                        
                        if len(ohlcv) == 0:
                            break
                            
                        ohlcv_data.extend(ohlcv)
                        
                        # 다음 시작 시간 계산
                        last_timestamp = ohlcv[-1][0]
                        current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)
                        
                        # 진행상황 표시
                        progress = min(100, (current_start - start_time) / (end_time - start_time) * 100)
                        print(f"  진행률: {progress:.1f}% ({len(ohlcv_data)}개 캔들)")
                        
                        # API 호출 제한 방지 (더 안전한 딜레이)
                        time.sleep(0.2)
                        
                    except Exception as e:
                        print(f"  부분 다운로드 실패: {e}")
                        break
                
                if len(ohlcv_data) > 0:
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp','open','high','low','close','volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # 중복 제거 및 정렬
                    df = df[~df.index.duplicated(keep='first')]
                    df = df.sort_index()
                    
                    # 기간 필터링
                    df = df[(df.index >= start_time) & (df.index <= end_time)]
                    
                    if len(df) > 0:
                        # 컬럼명 정리 (특수문자 제거)
                        clean_symbol = symbol.replace('/', '').replace(':', '_')
                        df = df.add_prefix(clean_symbol + '_')
                        all_dfs.append(df)
                        successful_downloads += 1
                        print(f"  ✓ {len(df)}개 캔들 다운로드 완료")
                    else:
                        print(f"  ✗ {symbol}: 지정 기간 내 데이터 없음")
                        failed_downloads += 1
                else:
                    print(f"  ✗ {symbol}: 데이터 없음")
                    failed_downloads += 1
                    
            except Exception as e:
                print(f"  ✗ {symbol} 다운로드 실패: {e}")
                failed_downloads += 1
                continue
        
        print(f"\n다운로드 완료: 성공 {successful_downloads}개, 실패 {failed_downloads}개")
        
        if not all_dfs:
            print("다운로드된 데이터가 없습니다.")
            return None
            
        # 타임스탬프 기준 병합
        print("데이터 병합 중...")
        merged = pd.concat(all_dfs, axis=1, join='outer')
        merged.reset_index(inplace=True)
        merged = merged.sort_values('timestamp')
        
        # 파일 저장
        os.makedirs(save_dir, exist_ok=True)
        start_date_str = start_time.strftime('%Y%m%d')
        end_date_str = end_time.strftime('%Y%m%d')
        filename = f"{save_dir}/binance_usdtp_1h_{start_date_str}_to_{end_date_str}.csv"
        merged.to_csv(filename, index=False)
        print(f"통합 데이터 저장 완료: {filename}")
        print(f"총 {len(merged)}개 타임스탬프, {len(merged.columns)-1}개 심볼")
        return filename
        
    except Exception as e:
        print(f"다운로드 중 오류 발생: {e}")
        return None


def main():
    """메인 함수"""
    engine = AdvancedBacktestEngine(initial_balance=10000.0)
    
    while True:
        show_menu()
        choice = input("선택하세요: ").strip()
        
        if choice == '1':
            # 백테스트 실행
            strategies = show_strategies()
            strategy_choice = input("전략을 선택하세요: ").strip()
            
            if strategy_choice in strategies:
                strategy_name = strategies[strategy_choice]
                
                # 데이터 로드 시도
                df = engine.load_data("BTC/USDT", "1h")
                
                if df is None:
                    print("저장된 데이터가 없습니다. 샘플 데이터를 생성합니다...")
                    df = engine.generate_sample_data()
                    engine.save_data(df, "BTC/USDT", "1h")
                
                # 백테스트 실행
                results = engine.run_backtest(df, strategy_name)
                
                if results:
                    # 성과 분석
                    performance = engine.analyze_performance(results['equity_curve'], results['trades'])
                    
                    # 보고서 생성 및 출력
                    report = engine.generate_report(performance, results['trades'])
                    print(report)
                    
                    # 결과 저장
                    all_results = {
                        'performance': performance,
                        'equity_curve': results['equity_curve'],
                        'trades': results['trades']
                    }
                    engine.save_results(all_results, strategy_name)
                
            else:
                print("잘못된 선택입니다.")
        
        elif choice == '2':
            # 데이터 다운로드 메뉴
            while True:
                show_download_menu()
                download_choice = input("선택하세요: ").strip()
                
                if download_choice == '1':
                    print("최근 1개월 데이터 다운로드를 시작합니다...")
                    filename = download_binance_usdtp_ohlcv_by_period(months=1)
                    if filename:
                        print(f"통합 데이터가 저장되었습니다: {filename}")
                    break
                    
                elif download_choice == '2':
                    print("최근 3개월 데이터 다운로드를 시작합니다...")
                    filename = download_binance_usdtp_ohlcv_by_period(months=3)
                    if filename:
                        print(f"통합 데이터가 저장되었습니다: {filename}")
                    break
                    
                elif download_choice == '3':
                    print("최근 6개월 데이터 다운로드를 시작합니다...")
                    filename = download_binance_usdtp_ohlcv_by_period(months=6)
                    if filename:
                        print(f"통합 데이터가 저장되었습니다: {filename}")
                    break
                    
                elif download_choice == '4':
                    print("최근 1년 데이터 다운로드를 시작합니다...")
                    filename = download_binance_usdtp_ohlcv_by_period(months=12)
                    if filename:
                        print(f"통합 데이터가 저장되었습니다: {filename}")
                    break
                    
                elif download_choice == '5':
                    start_date, end_date = get_custom_date_range()
                    print(f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} 데이터 다운로드를 시작합니다...")
                    filename = download_binance_usdtp_ohlcv_by_period(start_date=start_date, end_date=end_date)
                    if filename:
                        print(f"통합 데이터가 저장되었습니다: {filename}")
                    break
                    
                elif download_choice == '6':
                    break
                    
                else:
                    print("잘못된 선택입니다.")
        
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
        
        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main() 