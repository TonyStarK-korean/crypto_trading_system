"""
Integrated Crypto Trading System
통합된 암호화폐 거래 시스템 - 모든 구성 요소를 하나로 연결
"""

import os
import sys
import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import json
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all components
from database.database_manager import DatabaseManager
from realtime.websocket_client import RealTimeDataManager
from backtest.backtrader_engine import BacktraderEngine, BacktestConfig
from backtest.backtrader_engine import MovingAverageCrossStrategy, RSIMeanReversionStrategy
from data_collection.data_collector import DataCollectionManager, DataCollectionConfig
from monitoring.performance_monitor import PerformanceMonitor
from core.data_manager import DataManager
from strategies.strategy_manager import StrategyManager

class IntegratedTradingSystem:
    """통합 거래 시스템 메인 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.database_manager = None
        self.realtime_manager = None
        self.backtest_engine = None
        self.data_collector = None
        self.performance_monitor = None
        self.strategy_manager = None
        
        # System state
        self.running = False
        self.last_update = None
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("Integrated Trading System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'database': {
                'url': 'postgresql://postgres:password@localhost:5432/crypto_trading',
                'backup_interval': 3600  # 1 hour
            },
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
                'timeframes': ['1m', '5m', '15m', '1h'],
                'initial_capital': 10000,
                'risk_per_trade': 0.02,
                'max_positions': 5
            },
            'data_collection': {
                'exchanges': ['binance'],
                'update_interval': 300,  # 5 minutes
                'historical_days': 30
            },
            'monitoring': {
                'enabled': True,
                'alert_cpu_threshold': 80,
                'alert_memory_threshold': 85,
                'alert_drawdown_threshold': 10
            },
            'realtime': {
                'enabled': True,
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            }
        }
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        # Console handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # File handler
        logger.add(
            f"{log_dir}/trading_system.log",
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
    
    def initialize_components(self):
        """모든 구성 요소 초기화"""
        try:
            # Database Manager
            self.database_manager = DatabaseManager(self.config['database']['url'])
            logger.info("Database manager initialized")
            
            # Performance Monitor
            self.performance_monitor = PerformanceMonitor()
            logger.info("Performance monitor initialized")
            
            # Data Collection Manager
            data_config = DataCollectionConfig(
                exchanges=self.config['data_collection']['exchanges'],
                symbols=self.config['trading']['symbols'],
                timeframes=self.config['trading']['timeframes'],
                update_interval=self.config['data_collection']['update_interval'],
                lookback_days=self.config['data_collection']['historical_days']
            )
            self.data_collector = DataCollectionManager(data_config, self.database_manager)
            logger.info("Data collection manager initialized")
            
            # Real-time Data Manager
            if self.config['realtime']['enabled']:
                self.realtime_manager = RealTimeDataManager(
                    self.config['realtime']['symbols'],
                    self.database_manager
                )
                logger.info("Real-time data manager initialized")
            
            # Backtest Engine
            backtest_config = BacktestConfig(
                initial_cash=self.config['trading']['initial_capital'],
                risk_per_trade=self.config['trading']['risk_per_trade'],
                max_positions=self.config['trading']['max_positions']
            )
            self.backtest_engine = BacktraderEngine(backtest_config)
            logger.info("Backtest engine initialized")
            
            # Strategy Manager
            self.strategy_manager = StrategyManager()
            logger.info("Strategy manager initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def start_system(self):
        """시스템 시작"""
        if self.running:
            logger.warning("System is already running")
            return
        
        try:
            # Initialize components if not already done
            if self.database_manager is None:
                self.initialize_components()
            
            self.running = True
            self.last_update = datetime.now()
            
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            # Start data collection
            if self.data_collector:
                self.data_collector.start_automated_collection()
            
            # Start real-time data streams
            if self.realtime_manager:
                self.realtime_manager.start_ticker_stream()
                self.realtime_manager.start_kline_stream('1m')
            
            logger.info("Integrated Trading System started successfully")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.running = False
            raise
    
    def stop_system(self):
        """시스템 정지"""
        if not self.running:
            logger.warning("System is not running")
            return
        
        try:
            self.running = False
            
            # Stop real-time data streams
            if self.realtime_manager:
                self.realtime_manager.stop_all_streams()
            
            # Stop data collection
            if self.data_collector:
                self.data_collector.stop_automated_collection()
            
            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            logger.info("Integrated Trading System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def run_backtest(self, strategy_name: str, symbol: str, timeframe: str = '1h', days: int = 30) -> Dict[str, Any]:
        """백테스트 실행"""
        try:
            # Get data from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.database_manager.get_ohlcv_data(
                symbol=symbol.replace('/', ''),
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return {'error': 'No data available'}
            
            # Setup backtest engine
            self.backtest_engine.setup_cerebro()
            self.backtest_engine.add_data(df, symbol)
            
            # Select strategy
            if strategy_name == 'MA_Cross':
                strategy_class = MovingAverageCrossStrategy
                strategy_params = {'fast_period': 10, 'slow_period': 30}
            elif strategy_name == 'RSI_MeanReversion':
                strategy_class = RSIMeanReversionStrategy
                strategy_params = {'rsi_period': 14}
            else:
                logger.error(f"Unknown strategy: {strategy_name}")
                return {'error': 'Unknown strategy'}
            
            # Add strategy
            self.backtest_engine.add_strategy(strategy_class, **strategy_params)
            
            # Run backtest
            results = self.backtest_engine.run_backtest()
            
            # Store results in database
            backtest_result = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': results['initial_cash'],
                'final_value': results['final_value'],
                'total_return': results['total_return'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio'],
                'total_trades': results['total_trades'],
                'config': strategy_params
            }
            
            self.database_manager.insert_backtest_result(backtest_result)
            
            # Log performance metrics
            if self.performance_monitor:
                sample_trades = [{'pnl': results['total_return'] / results['total_trades']}] * results['total_trades']
                self.performance_monitor.log_trading_metrics(
                    strategy_name, symbol, sample_trades, []
                )
            
            logger.info(f"Backtest completed for {strategy_name} on {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        status = {
            'running': self.running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'components': {
                'database': self.database_manager is not None,
                'realtime': self.realtime_manager is not None,
                'backtest': self.backtest_engine is not None,
                'data_collector': self.data_collector is not None,
                'performance_monitor': self.performance_monitor is not None
            }
        }
        
        # Add database stats
        if self.database_manager:
            status['database_stats'] = self.database_manager.get_database_stats()
        
        # Add data collection status
        if self.data_collector:
            status['data_collection'] = self.data_collector.get_collection_status()
        
        # Add performance report
        if self.performance_monitor:
            status['performance'] = self.performance_monitor.get_performance_report(hours=1)
        
        return status
    
    def collect_historical_data(self, symbol: str, timeframe: str = '1h', days: int = 30):
        """과거 데이터 수집"""
        if self.data_collector:
            self.data_collector.collect_historical_data(symbol, timeframe, days)
            logger.info(f"Historical data collection started for {symbol}")
        else:
            logger.error("Data collector not initialized")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """최신 가격 조회"""
        if self.realtime_manager:
            return self.realtime_manager.get_latest_price(symbol)
        elif self.database_manager:
            return self.database_manager.get_latest_price(symbol)
        return None
    
    def run_multiple_backtests(self, strategies: List[str], symbols: List[str], timeframe: str = '1h', days: int = 30) -> List[Dict]:
        """여러 전략/종목에 대한 백테스트 실행"""
        results = []
        
        for strategy in strategies:
            for symbol in symbols:
                result = self.run_backtest(strategy, symbol, timeframe, days)
                result['strategy'] = strategy
                result['symbol'] = symbol
                results.append(result)
                
                # Wait between backtests to avoid overwhelming the system
                time.sleep(1)
        
        return results
    
    def generate_trading_report(self, hours: int = 24) -> Dict[str, Any]:
        """거래 보고서 생성"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'system_status': self.get_system_status(),
                'backtest_results': [],
                'performance_metrics': {}
            }
            
            # Get recent backtest results
            # This would typically query the database for recent results
            
            # Add performance metrics
            if self.performance_monitor:
                report['performance_metrics'] = self.performance_monitor.get_performance_report(hours)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trading report: {e}")
            return {'error': str(e)}

def main():
    """메인 함수"""
    try:
        # Initialize system
        system = IntegratedTradingSystem()
        
        # Start system
        system.start_system()
        
        # Collect some historical data
        system.collect_historical_data('BTC/USDT', '1h', 7)
        
        # Wait for data collection
        time.sleep(10)
        
        # Run a sample backtest
        results = system.run_backtest('MA_Cross', 'BTC/USDT', '1h', 7)
        print(f"Backtest results: {results}")
        
        # Get system status
        status = system.get_system_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")
        
        # Keep system running
        try:
            while True:
                time.sleep(60)
                logger.info("System running...")
                
                # Generate periodic report
                report = system.generate_trading_report(hours=1)
                logger.info(f"Generated report with {len(report)} sections")
                
        except KeyboardInterrupt:
            logger.info("Shutting down system...")
            system.stop_system()
            
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()