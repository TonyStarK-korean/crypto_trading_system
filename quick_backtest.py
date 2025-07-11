#!/usr/bin/env python3
"""
간단한 백테스트 실행 스크립트
사용법: python quick_backtest.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_backtest import BacktestRunner


def main():
    """간단한 백테스트 실행"""
    try:
        logger.info("=== 간단한 백테스트 시작 ===")
        
        # 기본 설정
        symbol = "BTC/USDT"
        strategy = "simple"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        timeframe = "1h"
        initial_balance = 10000.0
        commission = 0.001
        
        logger.info(f"심볼: {symbol}")
        logger.info(f"전략: {strategy}")
        logger.info(f"기간: {start_date} ~ {end_date}")
        logger.info(f"초기 자본: {initial_balance:,.0f}")
        
        # 백테스트 실행
        runner = BacktestRunner()
        results = runner.run_backtest(
            symbol=symbol,
            strategy_name=strategy,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            initial_balance=initial_balance,
            commission=commission
        )
        
        if results:
            logger.info("백테스트가 성공적으로 완료되었습니다!")
            logger.info("결과는 results/ 폴더에 저장되었습니다.")
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