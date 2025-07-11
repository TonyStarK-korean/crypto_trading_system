#!/usr/bin/env python3
"""
Crypto Trading System - Main Entry Point
"""

import sys
import os
import argparse
from pathlib import Path
from loguru import logger
import yaml

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.data_manager import DataManager
from src.backtest.backtest_engine import BacktestEngine
from src.strategies.ma_cross_strategy import MACrossStrategy


def load_config(config_path: str = "config/config.yaml") -> dict:
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"설정 파일 로드 오류: {e}")
        return {}


def setup_logging(config: dict):
    """로깅 설정"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'logs/trading.log')
    
    # 로그 디렉토리 생성
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 로거 설정
    logger.remove()  # 기본 핸들러 제거
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=log_config.get('max_size', '10MB'),
        retention=log_config.get('backup_count', 5)
    )


def run_backtest(config: dict):
    """백테스트 실행"""
    try:
        logger.info("백테스트 시작...")
        
        # 설정 추출
        backtest_config = config.get('backtest', {})
        initial_balance = backtest_config.get('initial_balance', 10000.0)
        commission = backtest_config.get('commission', 0.001)
        
        # 데이터 매니저 초기화
        data_manager = DataManager()
        
        # 샘플 데이터 생성 (실제로는 거래소에서 데이터 수집)
        logger.info("데이터 수집 중...")
        data = data_manager.get_historical_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        if data.empty:
            logger.error("데이터를 수집할 수 없습니다")
            return
        
        # 기술적 지표 계산
        data = data_manager.calculate_technical_indicators(data)
        
        # 백테스트 엔진 초기화
        engine = BacktestEngine(initial_balance=initial_balance, commission=commission)
        
        # 전략 초기화
        strategy_config = config.get('strategy', {})
        strategy_params = strategy_config.get('params', {})
        
        def ma_cross_strategy(data, params, ml_signal=None):
            """이동평균 교차 전략"""
            if len(data) < 50:
                return {}
            
            short_ma = data['close'].rolling(window=params.get('short_period', 20)).mean()
            long_ma = data['close'].rolling(window=params.get('long_period', 50)).mean()
            
            current_price = data['close'].iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            prev_short_ma = short_ma.iloc[-2]
            prev_long_ma = long_ma.iloc[-2]
            
            # 골든 크로스 (단기선이 장기선을 상향 돌파)
            if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
                if ml_signal is None or ml_signal >= 0:
                    return {
                        'action': 'buy',
                        'symbol': 'BTC/USDT',
                        'size': 0.1,
                        'leverage': 1.0
                    }
            
            # 데드 크로스 (단기선이 장기선을 하향 돌파)
            elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
                return {
                    'action': 'sell',
                    'symbol': 'BTC/USDT',
                    'size': 0.1,
                    'leverage': 1.0
                }
            
            return {}
        
        # 백테스트 실행
        results = engine.run_backtest(
            strategy_func=ma_cross_strategy,
            data=data,
            strategy_params=strategy_params,
            use_ml=config.get('ml', {}).get('use_ml', False)
        )
        
        if results:
            # 결과 출력
            performance = results['performance']
            logger.info("=== 백테스트 결과 ===")
            logger.info(f"총 수익률: {performance.get('total_return', 0):.2f}%")
            logger.info(f"연간 수익률: {performance.get('annual_return', 0):.2f}%")
            logger.info(f"최대 낙폭: {performance.get('max_drawdown', 0):.2f}%")
            logger.info(f"샤프 비율: {performance.get('sharpe_ratio', 0):.2f}")
            logger.info(f"승률: {performance.get('win_rate', 0):.2f}%")
            logger.info(f"총 거래 수: {performance.get('total_trades', 0)}")
            
            # 결과 저장
            engine.save_results("backtest_results")
            
            # 차트 생성
            engine.plot_results("backtest_chart.png")
        
        logger.info("백테스트 완료")
        
    except Exception as e:
        logger.error(f"백테스트 실행 오류: {e}")


def run_live_trading(config: dict):
    """실시간 거래 실행"""
    try:
        logger.info("실시간 거래 시작...")
        # TODO: 실시간 거래 로직 구현
        logger.warning("실시간 거래 기능은 아직 구현되지 않았습니다")
        
    except Exception as e:
        logger.error(f"실시간 거래 오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Crypto Trading System")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--mode", "-m", choices=["backtest", "live"], default="backtest", help="실행 모드")
    parser.add_argument("--symbol", "-s", default="BTC/USDT", help="거래 심볼")
    parser.add_argument("--timeframe", "-t", default="1h", help="시간프레임")
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    if not config:
        logger.error("설정을 로드할 수 없습니다")
        return
    
    # 로깅 설정
    setup_logging(config)
    
    logger.info("Crypto Trading System 시작")
    logger.info(f"모드: {args.mode}")
    logger.info(f"심볼: {args.symbol}")
    logger.info(f"시간프레임: {args.timeframe}")
    
    try:
        if args.mode == "backtest":
            run_backtest(config)
        elif args.mode == "live":
            run_live_trading(config)
        else:
            logger.error(f"지원하지 않는 모드입니다: {args.mode}")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
    
    logger.info("Crypto Trading System 종료")


if __name__ == "__main__":
    main() 