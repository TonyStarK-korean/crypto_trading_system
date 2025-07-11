"""
Performance Monitoring and Logging System
"""

import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import os
from loguru import logger
from dataclasses import dataclass, asdict
import sqlite3
import pandas as pd
from collections import defaultdict, deque
import traceback
import sys

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    
@dataclass
class TradingMetrics:
    timestamp: datetime
    strategy_name: str
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    open_positions: int
    
@dataclass
class AlertConfig:
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '=='
    alert_type: str  # 'email', 'log', 'webhook'
    cooldown_minutes: int = 30
    message: str = ""

class MetricsCollector:
    """Collects system and trading metrics"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.network_stats_start = psutil.net_io_counters()
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=process_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def collect_trading_metrics(self, strategy_name: str, symbol: str, 
                              trades_data: List[Dict], positions_data: List[Dict]) -> TradingMetrics:
        """Collect trading performance metrics"""
        try:
            total_trades = len(trades_data)
            winning_trades = sum(1 for trade in trades_data if trade.get('pnl', 0) > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / max(total_trades, 1) * 100
            
            total_pnl = sum(trade.get('pnl', 0) for trade in trades_data)
            
            # Calculate drawdown
            cumulative_pnl = 0
            peak = 0
            max_drawdown = 0
            current_drawdown = 0
            
            for trade in trades_data:
                cumulative_pnl += trade.get('pnl', 0)
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                drawdown = (peak - cumulative_pnl) / max(peak, 1) * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                current_drawdown = drawdown
            
            # Calculate Sharpe ratio (simplified)
            if trades_data:
                returns = [trade.get('pnl', 0) for trade in trades_data]
                if len(returns) > 1:
                    avg_return = sum(returns) / len(returns)
                    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                    sharpe_ratio = avg_return / max(std_return, 0.001)
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            open_positions = len([p for p in positions_data if p.get('status') == 'OPEN'])
            
            metrics = TradingMetrics(
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                symbol=symbol,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                open_positions=open_positions
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return None

class AlertManager:
    """Manages alerts based on metrics"""
    
    def __init__(self):
        self.alerts: List[AlertConfig] = []
        self.alert_history: Dict[str, datetime] = {}
        self.callbacks: Dict[str, Callable] = {}
        
    def add_alert(self, alert_config: AlertConfig):
        """Add an alert configuration"""
        self.alerts.append(alert_config)
        logger.info(f"Added alert for {alert_config.metric_name}")
    
    def add_callback(self, alert_type: str, callback: Callable):
        """Add callback for alert type"""
        self.callbacks[alert_type] = callback
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check if any alerts should be triggered"""
        current_time = datetime.now()
        
        for alert in self.alerts:
            try:
                # Check if we're in cooldown period
                last_alert = self.alert_history.get(alert.metric_name)
                if last_alert:
                    if current_time - last_alert < timedelta(minutes=alert.cooldown_minutes):
                        continue
                
                # Get metric value
                metric_value = metrics.get(alert.metric_name)
                if metric_value is None:
                    continue
                
                # Check threshold
                should_alert = False
                if alert.operator == '>':
                    should_alert = metric_value > alert.threshold
                elif alert.operator == '<':
                    should_alert = metric_value < alert.threshold
                elif alert.operator == '>=':
                    should_alert = metric_value >= alert.threshold
                elif alert.operator == '<=':
                    should_alert = metric_value <= alert.threshold
                elif alert.operator == '==':
                    should_alert = metric_value == alert.threshold
                
                if should_alert:
                    self._trigger_alert(alert, metric_value)
                    self.alert_history[alert.metric_name] = current_time
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert.metric_name}: {e}")
    
    def _trigger_alert(self, alert: AlertConfig, value: Any):
        """Trigger an alert"""
        message = alert.message or f"Alert: {alert.metric_name} {alert.operator} {alert.threshold} (current: {value})"
        
        logger.warning(f"ALERT TRIGGERED: {message}")
        
        # Call registered callback
        if alert.alert_type in self.callbacks:
            try:
                self.callbacks[alert.alert_type](alert, value, message)
            except Exception as e:
                logger.error(f"Error calling alert callback: {e}")

class MetricsStorage:
    """Stores metrics in SQLite database"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_used_gb REAL,
                        memory_available_gb REAL,
                        disk_usage_percent REAL,
                        network_bytes_sent INTEGER,
                        network_bytes_recv INTEGER,
                        process_count INTEGER
                    )
                ''')
                
                # Trading metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trading_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        strategy_name TEXT,
                        symbol TEXT,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        win_rate REAL,
                        total_pnl REAL,
                        current_drawdown REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        open_positions INTEGER
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_timestamp ON trading_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_strategy ON trading_metrics(strategy_name)')
                
                conn.commit()
                logger.info("Metrics database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing metrics database: {e}")
    
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_used_gb, 
                     memory_available_gb, disk_usage_percent, network_bytes_sent, 
                     network_bytes_recv, process_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp,
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.memory_used_gb,
                    metrics.memory_available_gb,
                    metrics.disk_usage_percent,
                    metrics.network_bytes_sent,
                    metrics.network_bytes_recv,
                    metrics.process_count
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
    
    def store_trading_metrics(self, metrics: TradingMetrics):
        """Store trading metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_metrics 
                    (timestamp, strategy_name, symbol, total_trades, winning_trades, 
                     losing_trades, win_rate, total_pnl, current_drawdown, 
                     max_drawdown, sharpe_ratio, open_positions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp,
                    metrics.strategy_name,
                    metrics.symbol,
                    metrics.total_trades,
                    metrics.winning_trades,
                    metrics.losing_trades,
                    metrics.win_rate,
                    metrics.total_pnl,
                    metrics.current_drawdown,
                    metrics.max_drawdown,
                    metrics.sharpe_ratio,
                    metrics.open_positions
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing trading metrics: {e}")
    
    def get_system_metrics(self, hours: int = 24) -> pd.DataFrame:
        """Get system metrics for last N hours"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM system_metrics 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                '''
                df = pd.read_sql_query(query, conn, params=(start_time,))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
                
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return pd.DataFrame()
    
    def get_trading_metrics(self, strategy_name: str = None, hours: int = 24) -> pd.DataFrame:
        """Get trading metrics for last N hours"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                if strategy_name:
                    query = '''
                        SELECT * FROM trading_metrics 
                        WHERE timestamp >= ? AND strategy_name = ?
                        ORDER BY timestamp DESC
                    '''
                    df = pd.read_sql_query(query, conn, params=(start_time, strategy_name))
                else:
                    query = '''
                        SELECT * FROM trading_metrics 
                        WHERE timestamp >= ? 
                        ORDER BY timestamp DESC
                    '''
                    df = pd.read_sql_query(query, conn, params=(start_time,))
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
                
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old metrics data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old system metrics
                cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_date,))
                system_deleted = cursor.rowcount
                
                # Delete old trading metrics
                cursor.execute('DELETE FROM trading_metrics WHERE timestamp < ?', (cutoff_date,))
                trading_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up {system_deleted} system metrics and {trading_deleted} trading metrics")
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, db_path: str = "C:/projects/crypto_trading_system/data/metrics.db"):
        self.collector = MetricsCollector()
        self.storage = MetricsStorage(db_path)
        self.alert_manager = AlertManager()
        self.running = False
        self.monitor_thread = None
        self.collection_interval = 60  # seconds
        
        # Setup default alerts
        self.setup_default_alerts()
    
    def setup_default_alerts(self):
        """Setup default system alerts"""
        # High CPU usage
        self.alert_manager.add_alert(AlertConfig(
            metric_name='cpu_percent',
            threshold=80.0,
            operator='>',
            alert_type='log',
            cooldown_minutes=10,
            message='High CPU usage detected'
        ))
        
        # High memory usage
        self.alert_manager.add_alert(AlertConfig(
            metric_name='memory_percent',
            threshold=85.0,
            operator='>',
            alert_type='log',
            cooldown_minutes=10,
            message='High memory usage detected'
        ))
        
        # High drawdown
        self.alert_manager.add_alert(AlertConfig(
            metric_name='current_drawdown',
            threshold=10.0,
            operator='>',
            alert_type='log',
            cooldown_minutes=30,
            message='High drawdown detected'
        ))
        
        # Low win rate
        self.alert_manager.add_alert(AlertConfig(
            metric_name='win_rate',
            threshold=30.0,
            operator='<',
            alert_type='log',
            cooldown_minutes=60,
            message='Low win rate detected'
        ))
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.running:
            logger.warning("Performance monitoring is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.collector.collect_system_metrics()
                if system_metrics:
                    self.storage.store_system_metrics(system_metrics)
                    
                    # Check system alerts
                    system_dict = asdict(system_metrics)
                    self.alert_manager.check_alerts(system_dict)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                traceback.print_exc()
                time.sleep(self.collection_interval)
    
    def log_trading_metrics(self, strategy_name: str, symbol: str, 
                          trades_data: List[Dict], positions_data: List[Dict]):
        """Log trading metrics"""
        try:
            trading_metrics = self.collector.collect_trading_metrics(
                strategy_name, symbol, trades_data, positions_data
            )
            
            if trading_metrics:
                self.storage.store_trading_metrics(trading_metrics)
                
                # Check trading alerts
                trading_dict = asdict(trading_metrics)
                self.alert_manager.check_alerts(trading_dict)
                
                logger.info(f"Logged trading metrics for {strategy_name}:{symbol}")
                
        except Exception as e:
            logger.error(f"Error logging trading metrics: {e}")
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            # Get system metrics
            system_df = self.storage.get_system_metrics(hours)
            
            # Get trading metrics
            trading_df = self.storage.get_trading_metrics(hours=hours)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours,
                'system_metrics': {},
                'trading_metrics': {},
                'alerts_triggered': len(self.alert_manager.alert_history)
            }
            
            # System metrics summary
            if not system_df.empty:
                report['system_metrics'] = {
                    'avg_cpu_percent': system_df['cpu_percent'].mean(),
                    'max_cpu_percent': system_df['cpu_percent'].max(),
                    'avg_memory_percent': system_df['memory_percent'].mean(),
                    'max_memory_percent': system_df['memory_percent'].max(),
                    'avg_disk_usage': system_df['disk_usage_percent'].mean(),
                    'samples_collected': len(system_df)
                }
            
            # Trading metrics summary
            if not trading_df.empty:
                report['trading_metrics'] = {
                    'strategies': trading_df['strategy_name'].unique().tolist(),
                    'symbols': trading_df['symbol'].unique().tolist(),
                    'total_trades': trading_df['total_trades'].sum(),
                    'avg_win_rate': trading_df['win_rate'].mean(),
                    'total_pnl': trading_df['total_pnl'].sum(),
                    'max_drawdown': trading_df['max_drawdown'].max(),
                    'avg_sharpe_ratio': trading_df['sharpe_ratio'].mean()
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def export_metrics(self, file_path: str, hours: int = 24):
        """Export metrics to JSON file"""
        try:
            report = self.get_performance_report(hours)
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some trading metrics
    sample_trades = [
        {'pnl': 100.0, 'symbol': 'BTC/USDT'},
        {'pnl': -50.0, 'symbol': 'BTC/USDT'},
        {'pnl': 75.0, 'symbol': 'BTC/USDT'},
    ]
    
    sample_positions = [
        {'status': 'OPEN', 'symbol': 'BTC/USDT'},
        {'status': 'CLOSED', 'symbol': 'ETH/USDT'},
    ]
    
    # Log trading metrics
    monitor.log_trading_metrics('MA_Cross', 'BTC/USDT', sample_trades, sample_positions)
    
    # Generate report
    report = monitor.get_performance_report(hours=1)
    print(json.dumps(report, indent=2, default=str))
    
    # Keep monitoring for a while
    try:
        time.sleep(120)  # Monitor for 2 minutes
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()