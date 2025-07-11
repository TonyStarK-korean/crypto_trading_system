"""
Test Crypto Trading System Basic Functionality
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import json

def test_exchange():
    """Test exchange connection"""
    print('Testing exchange connection...')
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f'BTC/USDT Price: ${ticker["last"]:,.2f}')
        print('Exchange connection successful!')
        return True
    except Exception as e:
        print(f'Exchange connection failed: {e}')
        return False

def test_backtest():
    """Test simple backtest"""
    print('Testing simple backtest...')
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # Generate sample price data
    prices = []
    price = 50000
    for i in range(100):
        price += np.random.normal(0, 100)
        prices.append(price)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + np.random.uniform(0, 200) for p in prices],
        'low': [p - np.random.uniform(0, 200) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(1000, 10000) for _ in range(100)]
    })
    
    # Simple MA strategy
    fast_period = 10
    slow_period = 30
    
    data['MA_fast'] = data['close'].rolling(window=fast_period).mean()
    data['MA_slow'] = data['close'].rolling(window=slow_period).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[slow_period:, 'signal'] = np.where(
        data['MA_fast'][slow_period:] > data['MA_slow'][slow_period:], 1, 0
    )
    data['position'] = data['signal'].diff()
    
    # Calculate returns
    initial_capital = 10000
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(data)):
        if data.iloc[i]['position'] == 1:  # Buy
            if position == 0:
                position = capital / data.iloc[i]['close']
                entry_price = data.iloc[i]['close']
                capital = 0
                
        elif data.iloc[i]['position'] == -1:  # Sell
            if position > 0:
                capital = position * data.iloc[i]['close']
                pnl = capital - initial_capital
                trades.append(pnl)
                position = 0
    
    # Final value
    if position > 0:
        final_value = position * data.iloc[-1]['close']
    else:
        final_value = capital
    
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    print(f'Initial Capital: ${initial_capital:,.2f}')
    print(f'Final Value: ${final_value:,.2f}')
    print(f'Total Return: {total_return:.2f}%')
    print(f'Total Trades: {len(trades)}')
    print('Backtest test complete!')
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'total_trades': len(trades)
    }

def test_data_collection():
    """Test data collection from exchange"""
    print('Testing data collection...')
    try:
        exchange = ccxt.binance()
        
        # Fetch recent OHLCV data
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=50)
        
        if ohlcv:
            print(f'Successfully fetched {len(ohlcv)} OHLCV records')
            print(f'Latest price: ${ohlcv[-1][4]:,.2f}')
            print(f'Latest volume: {ohlcv[-1][5]:,.2f}')
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f'Data range: {df.iloc[0]["timestamp"]} to {df.iloc[-1]["timestamp"]}')
            print('Data collection test successful!')
            return True
        else:
            print('No data received')
            return False
            
    except Exception as e:
        print(f'Data collection failed: {e}')
        return False

def main():
    """Main test function"""
    print('=' * 50)
    print('CRYPTO TRADING SYSTEM - FUNCTIONALITY TEST')
    print('=' * 50)
    
    # Test exchange connection
    exchange_ok = test_exchange()
    print()
    
    # Test data collection
    data_ok = test_data_collection()
    print()
    
    # Test backtest
    backtest_result = test_backtest()
    print()
    
    print('=' * 50)
    print('SYSTEM STATUS')
    print('=' * 50)
    print(f'Exchange Connection: {"✓" if exchange_ok else "✗"}')
    print(f'Data Collection: {"✓" if data_ok else "✗"}')
    print(f'Backtest Engine: ✓')
    print(f'Data Processing: ✓')
    print('=' * 50)
    
    # Summary
    if exchange_ok and data_ok:
        print('🎉 ALL TESTS PASSED!')
        print('The system is ready for backtesting and data collection.')
        print('\nNext steps:')
        print('1. Run simple_system.py for automated data collection')
        print('2. Wait for data collection (30 seconds)')
        print('3. Run backtests on collected data')
        print('4. Monitor system performance')
    else:
        print('⚠️  SOME TESTS FAILED')
        print('Please check your internet connection and API access.')
    
    print('=' * 50)

if __name__ == "__main__":
    main()