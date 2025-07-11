#!/usr/bin/env python3
"""
백테스트 실행 예제
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.data_manager import DataManager
from src.backtest.backtest_engine import BacktestEngine
from loguru import logger


def main():
    """백테스트 예제 실행"""
    try:
        logger.info("=== 백테스트 예제 시작 ===")
        
        # 1. 데이터 매니저 초기화
        data_manager = DataManager()
        
        # 2. 샘플 데이터 생성 (실제로는 거래소 API 사용)
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
        
        logger.info(f"수집된 데이터: {len(data)} 개의 캔들")
        
        # 3. 기술적 지표 계산
        logger.info("기술적 지표 계산 중...")
        data = data_manager.calculate_technical_indicators(data)
        
        # 4. 백테스트 엔진 초기화
        engine = BacktestEngine(initial_balance=10000.0, commission=0.001)
        
        # 5. 간단한 이동평균 교차 전략 정의
        def simple_ma_strategy(data, params, ml_signal=None):
            """간단한 이동평균 교차 전략"""
            if len(data) < 50:
                return {}
            
            # 이동평균 계산
            short_ma = data['close'].rolling(window=20).mean()
            long_ma = data['close'].rolling(window=50).mean()
            
            current_price = data['close'].iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            prev_short_ma = short_ma.iloc[-2]
            prev_long_ma = long_ma.iloc[-2]
            
            # RSI 확인
            rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else 50
            
            # 골든 크로스 (단기선이 장기선을 상향 돌파)
            if (prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma and 
                rsi < 70):  # 과매수 상태가 아닐 때
                
                # ML 신호가 있으면 확인
                if ml_signal is None or ml_signal >= 0:
                    return {
                        'action': 'buy',
                        'symbol': 'BTC/USDT',
                        'size': 0.1,
                        'leverage': 1.0
                    }
            
            # 데드 크로스 (단기선이 장기선을 하향 돌파)
            elif (prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma and 
                  rsi > 30):  # 과매도 상태가 아닐 때
                return {
                    'action': 'sell',
                    'symbol': 'BTC/USDT',
                    'size': 0.1,
                    'leverage': 1.0
                }
            
            return {}
        
        # 6. 백테스트 실행
        logger.info("백테스트 실행 중...")
        results = engine.run_backtest(
            strategy_func=simple_ma_strategy,
            data=data,
            strategy_params={},
            use_ml=False  # ML 비활성화 (예제용)
        )
        
        # 7. 결과 출력
        if results:
            performance = results['performance']
            
            logger.info("\n=== 백테스트 결과 ===")
            logger.info(f"총 수익률: {performance.get('total_return', 0):.2f}%")
            logger.info(f"연간 수익률: {performance.get('annual_return', 0):.2f}%")
            logger.info(f"최대 낙폭: {performance.get('max_drawdown', 0):.2f}%")
            logger.info(f"샤프 비율: {performance.get('sharpe_ratio', 0):.2f}")
            logger.info(f"승률: {performance.get('win_rate', 0):.2f}%")
            logger.info(f"총 거래 수: {performance.get('total_trades', 0)}")
            logger.info(f"손익비: {performance.get('profit_factor', 0):.2f}")
            
            # 8. 결과 저장
            engine.save_results("example_backtest_results")
            logger.info("결과가 저장되었습니다: example_backtest_results_*.csv")
            
            # 9. 차트 생성
            engine.plot_results("example_backtest_chart.png")
            logger.info("차트가 생성되었습니다: example_backtest_chart.png")
        
        logger.info("=== 백테스트 예제 완료 ===")
        
    except Exception as e:
        logger.error(f"백테스트 예제 실행 오류: {e}")


if __name__ == "__main__":
    main() 