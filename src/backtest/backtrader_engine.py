"""
Advanced Backtest Engine using Backtrader Framework
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_cash: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    position_sizing: str = 'fixed'  # 'fixed', 'percent', 'kelly'
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 5
    
class TechnicalIndicators:
    """Technical indicators for strategies"""
    
    @staticmethod
    def sma(data, period):
        return bt.indicators.SimpleMovingAverage(data, period=period)
    
    @staticmethod
    def ema(data, period):
        return bt.indicators.ExponentialMovingAverage(data, period=period)
    
    @staticmethod
    def rsi(data, period=14):
        return bt.indicators.RSI(data, period=period)
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        return bt.indicators.MACD(data, period_me1=fast, period_me2=slow, period_signal=signal)
    
    @staticmethod
    def bollinger_bands(data, period=20, devfactor=2.0):
        return bt.indicators.BollingerBands(data, period=period, devfactor=devfactor)
    
    @staticmethod
    def atr(data, period=14):
        return bt.indicators.ATR(data, period=period)
    
    @staticmethod
    def stochastic(data, period=14, period_dfast=3):
        return bt.indicators.Stochastic(data, period=period, period_dfast=period_dfast)

class BaseStrategy(bt.Strategy):
    """Base strategy class with common functionality"""
    
    params = (
        ('stop_loss', 0.02),        # 2% stop loss
        ('take_profit', 0.06),      # 6% take profit
        ('position_size', 0.1),     # 10% of portfolio per trade
        ('risk_per_trade', 0.02),   # 2% risk per trade
        ('trailing_stop', False),   # Enable trailing stop
        ('trailing_percent', 0.02), # 2% trailing stop
    )
    
    def __init__(self):
        self.order = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        
    def notify_order(self, order):
        """Called when order status changes"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}')
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.stop_loss_price = self.entry_price * (1 - self.params.stop_loss)
                if self.params.take_profit > 0:
                    self.take_profit_price = self.entry_price * (1 + self.params.take_profit)
                    
            elif order.issell():
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}')
                
                # Calculate P&L
                if self.entry_price:
                    pnl = (order.executed.price - self.entry_price) / self.entry_price * 100
                    self.log(f'P&L: {pnl:.2f}%')
                    
                    if pnl > 0:
                        self.winning_trades += 1
                        self.current_consecutive_losses = 0
                    else:
                        self.losing_trades += 1
                        self.current_consecutive_losses += 1
                        if self.current_consecutive_losses > self.max_consecutive_losses:
                            self.max_consecutive_losses = self.current_consecutive_losses
                
                self.trade_count += 1
                self.entry_price = None
                self.stop_loss_price = None
                self.take_profit_price = None
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.status.name}')
            
        self.order = None
    
    def notify_trade(self, trade):
        """Called when a trade is closed"""
        if not trade.isclosed:
            return
            
        self.log(f'TRADE CLOSED - PnL: {trade.pnl:.2f}, PnL Net: {trade.pnlcomm:.2f}')
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()}: {txt}')
    
    def get_position_size(self):
        """Calculate position size based on risk management"""
        if self.params.position_size <= 0:
            return 0
            
        cash = self.broker.get_cash()
        current_price = self.data.close[0]
        
        if self.params.risk_per_trade > 0 and self.params.stop_loss > 0:
            # Risk-based position sizing
            risk_amount = cash * self.params.risk_per_trade
            stop_loss_amount = current_price * self.params.stop_loss
            position_size = risk_amount / stop_loss_amount
            max_position_value = cash * self.params.position_size
            position_size = min(position_size, max_position_value / current_price)
        else:
            # Fixed percentage position sizing
            position_size = (cash * self.params.position_size) / current_price
        
        return int(position_size)
    
    def check_exit_conditions(self):
        """Check if we should exit the position"""
        if not self.position:
            return False
            
        current_price = self.data.close[0]
        
        # Stop loss
        if self.stop_loss_price and current_price <= self.stop_loss_price:
            self.log(f'Stop Loss triggered at {current_price:.2f}')
            return True
        
        # Take profit
        if self.take_profit_price and current_price >= self.take_profit_price:
            self.log(f'Take Profit triggered at {current_price:.2f}')
            return True
            
        # Trailing stop
        if self.params.trailing_stop and self.entry_price:
            high_since_entry = max(self.data.high.get(ago=i) for i in range(len(self.data.high)))
            trailing_stop_price = high_since_entry * (1 - self.params.trailing_percent)
            if current_price <= trailing_stop_price:
                self.log(f'Trailing Stop triggered at {current_price:.2f}')
                return True
        
        return False

class MovingAverageCrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('volume_filter', True),
    )
    
    def __init__(self):
        super().__init__()
        self.fast_ma = TechnicalIndicators.sma(self.data.close, self.params.fast_period)
        self.slow_ma = TechnicalIndicators.sma(self.data.close, self.params.slow_period)
        self.volume_ma = TechnicalIndicators.sma(self.data.volume, 20) if self.params.volume_filter else None
        
        # Signal when fast MA crosses above slow MA
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        """Main strategy logic"""
        if self.order:
            return
            
        # Check exit conditions
        if self.position and self.check_exit_conditions():
            self.order = self.sell()
            return
            
        # Entry conditions
        if not self.position:
            # Buy signal: fast MA crosses above slow MA
            if self.crossover > 0:
                # Volume filter
                if self.volume_ma is None or self.data.volume[0] > self.volume_ma[0]:
                    size = self.get_position_size()
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.log(f'BUY ORDER: Size: {size}, Price: {self.data.close[0]:.2f}')

class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('rsi_exit_high', 60),
        ('rsi_exit_low', 40),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = TechnicalIndicators.rsi(self.data.close, self.params.rsi_period)
        self.bb = TechnicalIndicators.bollinger_bands(self.data.close)
    
    def next(self):
        """Main strategy logic"""
        if self.order:
            return
            
        # Check exit conditions
        if self.position and self.check_exit_conditions():
            self.order = self.sell()
            return
            
        # Custom exit conditions for RSI strategy
        if self.position:
            if self.rsi[0] >= self.params.rsi_exit_high:
                self.order = self.sell()
                self.log(f'RSI EXIT: RSI: {self.rsi[0]:.2f}')
                return
        
        # Entry conditions
        if not self.position:
            # Buy signal: RSI oversold and price near lower Bollinger Band
            if (self.rsi[0] <= self.params.rsi_oversold and 
                self.data.close[0] <= self.bb.lines.bot[0]):
                size = self.get_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f'RSI BUY: RSI: {self.rsi[0]:.2f}, Price: {self.data.close[0]:.2f}')

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Breakout Strategy"""
    
    params = (
        ('bb_period', 20),
        ('bb_devfactor', 2.0),
        ('volume_threshold', 1.5),
    )
    
    def __init__(self):
        super().__init__()
        self.bb = TechnicalIndicators.bollinger_bands(
            self.data.close, 
            self.params.bb_period, 
            self.params.bb_devfactor
        )
        self.volume_ma = TechnicalIndicators.sma(self.data.volume, 20)
    
    def next(self):
        """Main strategy logic"""
        if self.order:
            return
            
        # Check exit conditions
        if self.position and self.check_exit_conditions():
            self.order = self.sell()
            return
            
        # Entry conditions
        if not self.position:
            # Buy signal: price breaks above upper Bollinger Band with volume
            if (self.data.close[0] > self.bb.lines.top[0] and 
                self.data.volume[0] > self.volume_ma[0] * self.params.volume_threshold):
                size = self.get_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f'BB BREAKOUT BUY: Price: {self.data.close[0]:.2f}, BB Top: {self.bb.lines.top[0]:.2f}')

class BacktraderEngine:
    """Advanced Backtest Engine using Backtrader"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.cerebro = None
        self.results = None
        
    def setup_cerebro(self):
        """Setup Backtrader Cerebro engine"""
        self.cerebro = bt.Cerebro()
        
        # Set broker settings
        self.cerebro.broker.set_cash(self.config.initial_cash)
        self.cerebro.broker.setcommission(commission=self.config.commission)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # Add observers
        self.cerebro.addobserver(bt.observers.Trades)
        self.cerebro.addobserver(bt.observers.BuySell)
        self.cerebro.addobserver(bt.observers.Value)
        
        logger.info(f"Cerebro setup completed with initial cash: ${self.config.initial_cash}")
    
    def add_data(self, data: pd.DataFrame, name: str = 'data'):
        """Add data to backtest"""
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create Backtrader data feed
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        self.cerebro.adddata(bt_data, name=name)
        logger.info(f"Added data feed '{name}' with {len(data)} records")
    
    def add_strategy(self, strategy_class, **kwargs):
        """Add strategy to backtest"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
        logger.info(f"Added strategy: {strategy_class.__name__}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the backtest"""
        if not self.cerebro:
            self.setup_cerebro()
        
        logger.info("Starting backtest...")
        start_time = datetime.now()
        
        # Run backtest
        self.results = self.cerebro.run()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Extract results
        strat = self.results[0]
        
        # Get analyzer results
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis()
        
        # Calculate final results
        final_value = self.cerebro.broker.get_value()
        total_return = (final_value - self.config.initial_cash) / self.config.initial_cash * 100
        
        results = {
            'initial_cash': self.config.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe.get('sharperatio', 0),
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
            'max_drawdown_period': drawdown.get('max', {}).get('len', 0),
            'total_trades': trades.get('total', {}).get('total', 0),
            'winning_trades': trades.get('won', {}).get('total', 0),
            'losing_trades': trades.get('lost', {}).get('total', 0),
            'win_rate': trades.get('won', {}).get('total', 0) / max(trades.get('total', {}).get('total', 1), 1) * 100,
            'avg_win': trades.get('won', {}).get('pnl', {}).get('average', 0),
            'avg_loss': trades.get('lost', {}).get('pnl', {}).get('average', 0),
            'profit_factor': abs(trades.get('won', {}).get('pnl', {}).get('total', 0)) / max(abs(trades.get('lost', {}).get('pnl', {}).get('total', 0)), 1),
            'sqn': sqn.get('sqn', 0),
            'duration': duration.total_seconds(),
            'trades_per_day': trades.get('total', {}).get('total', 0) / max(duration.days, 1),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Backtest completed in {duration.total_seconds():.2f} seconds")
        logger.info(f"Final Value: ${final_value:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        
        return results
    
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        if not self.results:
            logger.error("No backtest results to plot")
            return
        
        # Create the plot
        figs = self.cerebro.plot(style='candlestick', barup='green', bardown='red')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, results: Dict[str, Any], file_path: str):
        """Save backtest results to file"""
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {file_path}")

# Example usage
if __name__ == "__main__":
    # Load sample data
    data_path = "C:/projects/crypto_trading_system/data/BTC_USDT_1h_20250709.csv"
    
    if os.path.exists(data_path):
        # Load data
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Setup backtest
        config = BacktestConfig(
            initial_cash=10000,
            commission=0.001,
            risk_per_trade=0.02
        )
        
        engine = BacktraderEngine(config)
        engine.setup_cerebro()
        
        # Add data
        engine.add_data(df, 'BTCUSDT')
        
        # Add strategy
        engine.add_strategy(
            MovingAverageCrossStrategy,
            fast_period=10,
            slow_period=30,
            stop_loss=0.02,
            take_profit=0.06
        )
        
        # Run backtest
        results = engine.run_backtest()
        
        # Save results
        results_dir = "C:/projects/crypto_trading_system/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{results_dir}/backtrader_results_{timestamp}.json"
        engine.save_results(results, results_file)
        
        # Plot results
        plot_file = f"{results_dir}/backtrader_plot_{timestamp}.png"
        engine.plot_results(plot_file)
    else:
        logger.error(f"Data file not found: {data_path}")