#!/usr/bin/env python3
"""
암호화폐 백테스트 실행 파일
사용법: python run_backtest.py [옵션]
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.data_manager import DataManager
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.bollinger_breakout_strategy import BollingerBreakoutStrategy
from src.strategies.base_strategy import BaseStrategy


class BacktestRunner:
    """백테스트 실행기"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.data_manager = DataManager()
        self.strategies = {
            'bollinger_breakout': BollingerBreakoutStrategy(),
            # 다른 전략들 추가 가능
        }
        
    def load_config(self) -> dict:
        """설정 파일 로드"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 오류: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            'backtest': {
                'initial_balance': 10000.0,
                'commission': 0.001,
                'slippage': 0.0005
            },
            'strategy': {
                'params': {
                    'short_period': 20,
                    'long_period': 50,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70
                }
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def get_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1h") -> pd.DataFrame:
        """데이터 수집"""
        try:
            # 먼저 저장된 데이터 확인
            data_file = f"data/{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            if os.path.exists(data_file):
                logger.info(f"저장된 데이터 로드: {data_file}")
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                return df
            
            # 데이터 수집
            logger.info(f"데이터 수집 중: {symbol} ({start_date} ~ {end_date})")
            data = self.data_manager.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                logger.warning("실제 데이터를 수집할 수 없어 샘플 데이터를 생성합니다")
                data = self.generate_sample_data(symbol, start_date, end_date)
            
            # 데이터 저장
            os.makedirs("data", exist_ok=True)
            data.to_csv(data_file)
            logger.info(f"데이터 저장 완료: {data_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            return self.generate_sample_data(symbol, start_date, end_date)
    
    def generate_sample_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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
    
    def run_simple_strategy(self, data: pd.DataFrame, params: dict) -> dict:
        """간단한 이동평균 교차 전략"""
        def strategy_func(data, params, ml_signal=None):
            if len(data) < 50:
                return {}
            
            # 이동평균 계산
            short_ma = data['close'].rolling(window=params.get('short_period', 20)).mean()
            long_ma = data['close'].rolling(window=params.get('long_period', 50)).mean()
            
            current_price = data['close'].iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            prev_short_ma = short_ma.iloc[-2]
            prev_long_ma = long_ma.iloc[-2]
            
            # RSI 계산
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=params.get('rsi_period', 14)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=params.get('rsi_period', 14)).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # 골든 크로스 (단기선이 장기선을 상향 돌파)
            if (prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma and 
                current_rsi < params.get('rsi_overbought', 70)):
                
                if ml_signal is None or ml_signal >= 0:
                    return {
                        'action': 'buy',
                        'symbol': 'BTC/USDT',
                        'size': 0.1,
                        'leverage': 1.0
                    }
            
            # 데드 크로스 (단기선이 장기선을 하향 돌파)
            elif (prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma and 
                  current_rsi > params.get('rsi_oversold', 30)):
                return {
                    'action': 'sell',
                    'symbol': 'BTC/USDT',
                    'size': 0.1,
                    'leverage': 1.0
                }
            
            return {}
        
        return strategy_func
    
    def run_backtest(self, symbol: str, strategy_name: str, start_date: str, end_date: str, 
                    timeframe: str = "1h", initial_balance: float = 10000.0, 
                    commission: float = 0.001) -> dict:
        """백테스트 실행"""
        try:
            logger.info("=== 백테스트 시작 ===")
            logger.info(f"심볼: {symbol}")
            logger.info(f"전략: {strategy_name}")
            logger.info(f"기간: {start_date} ~ {end_date}")
            logger.info(f"초기 자본: {initial_balance:,.0f}")
            
            # 데이터 수집
            data = self.get_data(symbol, start_date, end_date, timeframe)
            if data.empty:
                logger.error("데이터를 수집할 수 없습니다")
                return {}
            
            logger.info(f"데이터 수집 완료: {len(data)} 개의 캔들")
            
            # 기술적 지표 계산
            data = self.data_manager.calculate_technical_indicators(data)
            
            # 백테스트 엔진 초기화
            engine = BacktestEngine(initial_balance=initial_balance, commission=commission)
            
            # 전략 선택
            if strategy_name == 'bollinger_breakout':
                strategy_func = self.strategies['bollinger_breakout'].generate_signal
                strategy_params = {}
            else:
                # 기본 이동평균 전략
                config = self.load_config()
                strategy_params = config.get('strategy', {}).get('params', {})
                strategy_func = self.run_simple_strategy(data, strategy_params)
            
            # 백테스트 실행
            results = engine.run_backtest(
                strategy_func=strategy_func,
                data=data,
                strategy_params=strategy_params,
                use_ml=False
            )
            
            if results:
                # 결과 출력
                performance = results['performance']
                logger.info("\n=== 백테스트 결과 ===")
                logger.info(f"총 수익률: {performance.get('total_return', 0):.2f}%")
                logger.info(f"연간 수익률: {performance.get('annual_return', 0):.2f}%")
                logger.info(f"최대 낙폭: {performance.get('max_drawdown', 0):.2f}%")
                logger.info(f"샤프 비율: {performance.get('sharpe_ratio', 0):.2f}")
                logger.info(f"승률: {performance.get('win_rate', 0):.2f}%")
                logger.info(f"총 거래 수: {performance.get('total_trades', 0)}")
                logger.info(f"손익비: {performance.get('profit_factor', 0):.2f}")
                
                # 결과 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_filename = f"results/backtest_{strategy_name}_{timestamp}.json"
                os.makedirs("results", exist_ok=True)
                
                with open(results_filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"결과 저장 완료: {results_filename}")
                
                # 차트 생성
                chart_filename = f"results/backtest_{strategy_name}_{timestamp}.png"
                engine.plot_results(chart_filename)
                logger.info(f"차트 생성 완료: {chart_filename}")
                
                return results
            
            logger.error("백테스트 실행 실패")
            return {}
            
        except Exception as e:
            logger.error(f"백테스트 실행 오류: {e}")
            return {}


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="암호화폐 백테스트 실행기")
    parser.add_argument("--symbol", "-s", default="BTC/USDT", help="거래 심볼")
    parser.add_argument("--strategy", "-st", default="simple", choices=["simple", "bollinger_breakout"], 
                       help="백테스트 전략")
    parser.add_argument("--start-date", "-sd", default="2023-01-01", help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", "-ed", default="2023-12-31", help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--timeframe", "-t", default="1h", help="시간프레임")
    parser.add_argument("--initial-balance", "-ib", type=float, default=10000.0, help="초기 자본")
    parser.add_argument("--commission", "-c", type=float, default=0.001, help="수수료")
    parser.add_argument("--config", "-cf", default="config/config.yaml", help="설정 파일 경로")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 백테스트 실행
    runner = BacktestRunner(args.config)
    results = runner.run_backtest(
        symbol=args.symbol,
        strategy_name=args.strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        initial_balance=args.initial_balance,
        commission=args.commission
    )
    
    if results:
        logger.info("백테스트가 성공적으로 완료되었습니다!")
    else:
        logger.error("백테스트 실행에 실패했습니다.")


if __name__ == "__main__":
    main() 