"""
Automated Data Collection System for Crypto Trading
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import time
import asyncio
import threading
from loguru import logger
import schedule
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class DataCollectionConfig:
    exchanges: List[str] = None
    symbols: List[str] = None
    timeframes: List[str] = None
    lookback_days: int = 30
    update_interval: int = 60  # seconds
    batch_size: int = 100
    rate_limit: float = 1.0  # requests per second
    retry_attempts: int = 3
    save_to_file: bool = True
    save_to_database: bool = True
    
    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ['binance', 'coinbasepro', 'kraken']
        if self.symbols is None:
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'SOL/USDT']
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

class ExchangeDataCollector:
    """Collect data from a single exchange"""
    
    def __init__(self, exchange_name: str, config: DataCollectionConfig):
        self.exchange_name = exchange_name
        self.config = config
        self.exchange = None
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': os.getenv(f'{self.exchange_name.upper()}_API_KEY'),
                'secret': os.getenv(f'{self.exchange_name.upper()}_SECRET'),
                'sandbox': os.getenv(f'{self.exchange_name.upper()}_SANDBOX', 'True').lower() == 'true',
                'enableRateLimit': True,
                'rateLimit': 1000 / self.config.rate_limit,
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"Successfully connected to {self.exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            self.exchange = None
    
    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols"""
        if not self.exchange:
            return []
        
        try:
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol in markets.keys() 
                      if markets[symbol]['active'] and 
                      markets[symbol]['type'] == 'spot']
            return symbols
        except Exception as e:
            logger.error(f"Error getting symbols from {self.exchange_name}: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        if not self.exchange:
            return pd.DataFrame()
        
        try:
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['exchange'] = self.exchange_name
            df['timeframe'] = timeframe
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe} from {self.exchange_name}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], timeframe: str) -> List[pd.DataFrame]:
        """Fetch data for multiple symbols"""
        dataframes = []
        
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe)
                if not df.empty:
                    dataframes.append(df)
                
                # Rate limiting
                time.sleep(1.0 / self.config.rate_limit)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        return dataframes
    
    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Fetch historical data for a specific date range"""
        if not self.exchange:
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.now()
        
        all_data = []
        current_date = start_date
        
        # Calculate timeframe in milliseconds
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        
        while current_date < end_date:
            try:
                since = int(current_date.timestamp() * 1000)
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                
                if not ohlcv:
                    break
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['exchange'] = self.exchange_name
                df['timeframe'] = timeframe
                
                all_data.append(df)
                
                # Update current date
                if len(ohlcv) > 0:
                    current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                else:
                    break
                
                # Rate limiting
                time.sleep(1.0 / self.config.rate_limit)
                
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                break
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Remove duplicates
            result = result.drop_duplicates(subset=['timestamp', 'symbol', 'timeframe'])
            result = result.sort_values('timestamp')
            return result
        
        return pd.DataFrame()

class DataCollectionManager:
    """Manages data collection across multiple exchanges"""
    
    def __init__(self, config: DataCollectionConfig = None, database_manager = None):
        self.config = config or DataCollectionConfig()
        self.database_manager = database_manager
        self.collectors = {}
        self.running = False
        self.scheduler_thread = None
        self.initialize_collectors()
    
    def initialize_collectors(self):
        """Initialize exchange collectors"""
        for exchange_name in self.config.exchanges:
            try:
                collector = ExchangeDataCollector(exchange_name, self.config)
                if collector.exchange:
                    self.collectors[exchange_name] = collector
                    logger.info(f"Initialized collector for {exchange_name}")
            except Exception as e:
                logger.error(f"Failed to initialize collector for {exchange_name}: {e}")
    
    def collect_current_data(self):
        """Collect current data from all exchanges"""
        logger.info("Starting data collection cycle")
        
        with ThreadPoolExecutor(max_workers=len(self.collectors)) as executor:
            futures = []
            
            for exchange_name, collector in self.collectors.items():
                for timeframe in self.config.timeframes:
                    future = executor.submit(
                        self._collect_exchange_data,
                        collector,
                        exchange_name,
                        timeframe
                    )
                    futures.append(future)
            
            # Process results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self._process_collected_data(result)
                except Exception as e:
                    logger.error(f"Error in data collection: {e}")
    
    def _collect_exchange_data(self, collector: ExchangeDataCollector, exchange_name: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Collect data from a specific exchange and timeframe"""
        try:
            # Get available symbols
            available_symbols = collector.get_available_symbols()
            
            # Filter symbols that are in our config
            symbols_to_collect = [s for s in self.config.symbols if s in available_symbols]
            
            if not symbols_to_collect:
                logger.warning(f"No matching symbols found for {exchange_name}")
                return None
            
            # Collect data
            dataframes = collector.fetch_multiple_symbols(symbols_to_collect[:5], timeframe)  # Limit to 5 symbols
            
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                logger.info(f"Collected {len(combined_df)} records from {exchange_name} for {timeframe}")
                return combined_df
            
        except Exception as e:
            logger.error(f"Error collecting data from {exchange_name}: {e}")
        
        return None
    
    def _process_collected_data(self, data: pd.DataFrame):
        """Process and store collected data"""
        if data.empty:
            return
        
        # Save to database
        if self.config.save_to_database and self.database_manager:
            try:
                # Convert DataFrame to database format
                db_records = []
                for _, row in data.iterrows():
                    db_records.append({
                        'symbol': row['symbol'],
                        'exchange': row['exchange'],
                        'timeframe': row['timeframe'],
                        'timestamp': row['timestamp'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                
                self.database_manager.insert_ohlcv_data(db_records)
                logger.info(f"Saved {len(db_records)} records to database")
                
            except Exception as e:
                logger.error(f"Error saving to database: {e}")
        
        # Save to file
        if self.config.save_to_file:
            try:
                data_dir = "C:/projects/crypto_trading_system/data"
                os.makedirs(data_dir, exist_ok=True)
                
                # Group by exchange and timeframe
                for (exchange, timeframe), group in data.groupby(['exchange', 'timeframe']):
                    filename = f"{exchange}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = os.path.join(data_dir, filename)
                    group.to_csv(filepath, index=False)
                    logger.info(f"Saved data to {filepath}")
                    
            except Exception as e:
                logger.error(f"Error saving to file: {e}")
    
    def collect_historical_data(self, symbol: str, timeframe: str, days_back: int = 30):
        """Collect historical data for a specific symbol"""
        start_date = datetime.now() - timedelta(days=days_back)
        
        for exchange_name, collector in self.collectors.items():
            try:
                logger.info(f"Collecting historical data for {symbol} from {exchange_name}")
                
                df = collector.fetch_historical_data(symbol, timeframe, start_date)
                
                if not df.empty:
                    self._process_collected_data(df)
                    logger.info(f"Collected {len(df)} historical records for {symbol}")
                else:
                    logger.warning(f"No historical data found for {symbol} on {exchange_name}")
                    
            except Exception as e:
                logger.error(f"Error collecting historical data for {symbol}: {e}")
    
    def start_automated_collection(self):
        """Start automated data collection"""
        if self.running:
            logger.warning("Data collection is already running")
            return
        
        self.running = True
        
        # Schedule regular data collection
        schedule.every(self.config.update_interval).seconds.do(self.collect_current_data)
        
        # Schedule daily cleanup
        schedule.every().day.at("02:00").do(self._daily_cleanup)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info(f"Started automated data collection (update interval: {self.config.update_interval}s)")
    
    def stop_automated_collection(self):
        """Stop automated data collection"""
        self.running = False
        schedule.clear()
        logger.info("Stopped automated data collection")
    
    def _run_scheduler(self):
        """Run the scheduler"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def _daily_cleanup(self):
        """Daily cleanup tasks"""
        if self.database_manager:
            try:
                self.database_manager.cleanup_old_data(days_to_keep=30)
                logger.info("Completed daily cleanup")
            except Exception as e:
                logger.error(f"Error during daily cleanup: {e}")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        status = {
            'running': self.running,
            'collectors': list(self.collectors.keys()),
            'symbols': self.config.symbols,
            'timeframes': self.config.timeframes,
            'update_interval': self.config.update_interval,
            'last_collection': datetime.now().isoformat()
        }
        
        if self.database_manager:
            status['database_stats'] = self.database_manager.get_database_stats()
        
        return status

# Example usage
if __name__ == "__main__":
    # Configuration
    config = DataCollectionConfig(
        exchanges=['binance'],
        symbols=['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        timeframes=['1m', '5m', '1h'],
        update_interval=300,  # 5 minutes
        lookback_days=7
    )
    
    # Initialize data collection manager
    manager = DataCollectionManager(config)
    
    # Collect some historical data
    manager.collect_historical_data('BTC/USDT', '1h', days_back=7)
    
    # Start automated collection
    manager.start_automated_collection()
    
    # Keep running
    try:
        while True:
            time.sleep(60)
            status = manager.get_collection_status()
            logger.info(f"Collection status: {status}")
    except KeyboardInterrupt:
        manager.stop_automated_collection()
        logger.info("Data collection stopped")