"""
Real-time WebSocket Data Client for Crypto Trading
"""

import asyncio
import json
import websockets
import ccxt
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from loguru import logger
from dataclasses import dataclass
import threading
import time

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    exchange: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None

class BinanceWebSocketClient:
    def __init__(self, symbols: List[str], callbacks: Dict[str, Callable] = None):
        self.symbols = [s.lower() for s in symbols]
        self.callbacks = callbacks or {}
        self.running = False
        self.connection = None
        self.base_url = "wss://stream.binance.com:9443/ws/"
        
    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            # Create stream names for ticker data
            streams = [f"{symbol}@ticker" for symbol in self.symbols]
            stream_url = self.base_url + "/".join(streams)
            
            logger.info(f"Connecting to Binance WebSocket: {stream_url}")
            
            async with websockets.connect(stream_url) as websocket:
                self.connection = websocket
                self.running = True
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.running = False
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            if 'stream' in data:
                stream_data = data['data']
                symbol = stream_data['s']
                
                market_data = MarketData(
                    symbol=symbol,
                    price=float(stream_data['c']),
                    volume=float(stream_data['v']),
                    timestamp=datetime.fromtimestamp(stream_data['E'] / 1000),
                    exchange='binance',
                    bid=float(stream_data['b']),
                    ask=float(stream_data['a']),
                    high_24h=float(stream_data['h']),
                    low_24h=float(stream_data['l']),
                    change_24h=float(stream_data['P'])
                )
                
                # Call registered callbacks
                if 'on_ticker' in self.callbacks:
                    await self.callbacks['on_ticker'](market_data)
                    
            else:
                # Handle single stream data
                symbol = data['s']
                market_data = MarketData(
                    symbol=symbol,
                    price=float(data['c']),
                    volume=float(data['v']),
                    timestamp=datetime.fromtimestamp(data['E'] / 1000),
                    exchange='binance',
                    bid=float(data['b']),
                    ask=float(data['a']),
                    high_24h=float(data['h']),
                    low_24h=float(data['l']),
                    change_24h=float(data['P'])
                )
                
                if 'on_ticker' in self.callbacks:
                    await self.callbacks['on_ticker'](market_data)
                    
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def start(self):
        """Start WebSocket connection in background"""
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.connect())
            
        self.thread = threading.Thread(target=run_websocket)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.connection:
            asyncio.create_task(self.connection.close())

class KlineWebSocketClient:
    def __init__(self, symbols: List[str], interval: str = '1m', callbacks: Dict[str, Callable] = None):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.callbacks = callbacks or {}
        self.running = False
        self.connection = None
        self.base_url = "wss://stream.binance.com:9443/ws/"
        
    async def connect(self):
        """Connect to Binance Kline WebSocket"""
        try:
            # Create stream names for kline data
            streams = [f"{symbol}@kline_{self.interval}" for symbol in self.symbols]
            stream_url = self.base_url + "/".join(streams)
            
            logger.info(f"Connecting to Binance Kline WebSocket: {stream_url}")
            
            async with websockets.connect(stream_url) as websocket:
                self.connection = websocket
                self.running = True
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self._handle_kline_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing kline message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling kline message: {e}")
                        
        except Exception as e:
            logger.error(f"Kline WebSocket connection error: {e}")
        finally:
            self.running = False
    
    async def _handle_kline_message(self, data: Dict[str, Any]):
        """Handle incoming Kline WebSocket message"""
        try:
            if 'stream' in data:
                kline_data = data['data']['k']
            else:
                kline_data = data['k']
            
            # Only process completed klines
            if kline_data['x']:  # kline is closed
                ohlcv_data = {
                    'symbol': kline_data['s'],
                    'timestamp': datetime.fromtimestamp(kline_data['t'] / 1000),
                    'open': float(kline_data['o']),
                    'high': float(kline_data['h']),
                    'low': float(kline_data['l']),
                    'close': float(kline_data['c']),
                    'volume': float(kline_data['v']),
                    'interval': self.interval
                }
                
                if 'on_kline' in self.callbacks:
                    await self.callbacks['on_kline'](ohlcv_data)
                    
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    def start(self):
        """Start Kline WebSocket connection in background"""
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.connect())
            
        self.thread = threading.Thread(target=run_websocket)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop Kline WebSocket connection"""
        self.running = False
        if self.connection:
            asyncio.create_task(self.connection.close())

class RealTimeDataManager:
    def __init__(self, symbols: List[str], database_manager=None):
        self.symbols = symbols
        self.database_manager = database_manager
        self.ticker_client = None
        self.kline_client = None
        self.latest_data = {}
        self.callbacks = {
            'on_ticker': self._on_ticker_update,
            'on_kline': self._on_kline_update
        }
        
    async def _on_ticker_update(self, market_data: MarketData):
        """Handle ticker updates"""
        self.latest_data[market_data.symbol] = {
            'price': market_data.price,
            'volume': market_data.volume,
            'timestamp': market_data.timestamp,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'high_24h': market_data.high_24h,
            'low_24h': market_data.low_24h,
            'change_24h': market_data.change_24h
        }
        
        logger.debug(f"Ticker update for {market_data.symbol}: ${market_data.price}")
    
    async def _on_kline_update(self, kline_data: Dict[str, Any]):
        """Handle kline updates and store in database"""
        if self.database_manager:
            # Prepare data for database insertion
            db_data = [{
                'symbol': kline_data['symbol'],
                'exchange': 'binance',
                'timeframe': kline_data['interval'],
                'timestamp': kline_data['timestamp'],
                'open': kline_data['open'],
                'high': kline_data['high'],
                'low': kline_data['low'],
                'close': kline_data['close'],
                'volume': kline_data['volume']
            }]
            
            self.database_manager.insert_ohlcv_data(db_data)
            
        logger.debug(f"Kline update for {kline_data['symbol']}: {kline_data['close']}")
    
    def start_ticker_stream(self):
        """Start real-time ticker stream"""
        self.ticker_client = BinanceWebSocketClient(
            symbols=self.symbols,
            callbacks=self.callbacks
        )
        self.ticker_client.start()
        logger.info("Ticker stream started")
    
    def start_kline_stream(self, interval: str = '1m'):
        """Start real-time kline stream"""
        self.kline_client = KlineWebSocketClient(
            symbols=self.symbols,
            interval=interval,
            callbacks=self.callbacks
        )
        self.kline_client.start()
        logger.info(f"Kline stream started for {interval} interval")
    
    def stop_all_streams(self):
        """Stop all WebSocket streams"""
        if self.ticker_client:
            self.ticker_client.stop()
        if self.kline_client:
            self.kline_client.stop()
        logger.info("All streams stopped")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        if symbol in self.latest_data:
            return self.latest_data[symbol]['price']
        return None
    
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for a symbol"""
        return self.latest_data.get(symbol)
    
    def add_callback(self, callback_name: str, callback_func: Callable):
        """Add custom callback function"""
        self.callbacks[callback_name] = callback_func

# Example usage
if __name__ == "__main__":
    async def custom_ticker_callback(market_data: MarketData):
        print(f"Custom callback: {market_data.symbol} - ${market_data.price}")
    
    # Initialize real-time data manager
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    rt_manager = RealTimeDataManager(symbols)
    
    # Add custom callback
    rt_manager.add_callback('on_ticker', custom_ticker_callback)
    
    # Start streams
    rt_manager.start_ticker_stream()
    rt_manager.start_kline_stream('1m')
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Get latest data
            for symbol in symbols:
                price = rt_manager.get_latest_price(symbol)
                if price:
                    print(f"{symbol}: ${price}")
    except KeyboardInterrupt:
        rt_manager.stop_all_streams()
        print("Streams stopped")