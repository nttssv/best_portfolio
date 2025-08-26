"""
Bitcoin vs Portfolio Performance Comparison
Analyze Bitcoin performance and compare with portfolio strategies
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_bitcoin_performance():
    """Analyze Bitcoin performance and create comparison."""
    
    print('📊 Fetching Bitcoin data...')
    
    try:
        # Fetch Bitcoin data (5 years)
        btc = yf.download('BTC-USD', period='5y', progress=False)
        btc_prices = btc['Close']
        btc_returns = btc_prices.pct_change().dropna()
        
        # Calculate Bitcoin metrics
        annual_return = (1 + btc_returns.mean()) ** 252 - 1
        volatility = btc_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + btc_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 25-year projection (conservative adjustment for crypto)
        btc_adj_return = annual_return * 0.6  # 40% haircut for crypto volatility
        final_value_btc = 100000 * (1 + btc_adj_return) ** 25
        
        print(f'✅ Bitcoin 5-Year Performance Analysis:')
        print(f'   Annual Return: {annual_return:.2%}')
        print(f'   Volatility: {volatility:.2%}')
        print(f'   Sharpe Ratio: {sharpe:.3f}')
        print(f'   Max Drawdown: {max_drawdown:.2%}')
        print(f'   Adjusted Future Return: {btc_adj_return:.2%}')
        print(f'   25Y Projection: ${final_value_btc:,.0f}')
        print(f'   Wealth Multiple: {final_value_btc/100000:.1f}x')
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'adj_return': btc_adj_return,
            'final_value': final_value_btc,
            'returns': btc_returns,
            'prices': btc_prices
        }
        
    except Exception as e:
        print(f'❌ Error fetching Bitcoin data: {e}')
        print('📊 Using historical Bitcoin estimates...')
        # Return estimated values based on historical performance
        annual_return = 0.45  # ~45% historical
        volatility = 0.75     # ~75% volatility
        sharpe = 0.60         # ~0.6 Sharpe
        max_drawdown = -0.85  # ~85% max drawdown
        adj_return = 0.27     # 45% * 0.6 adjustment
        final_value = 100000 * (1 + 0.27) ** 25
        
        print(f'✅ Bitcoin Estimated Performance:')
        print(f'   Annual Return: {annual_return:.2%}')
        print(f'   Volatility: {volatility:.2%}')
        print(f'   Sharpe Ratio: {sharpe:.3f}')
        print(f'   Max Drawdown: {max_drawdown:.2%}')
        print(f'   Adjusted Future Return: {adj_return:.2%}')
        print(f'   25Y Projection: ${final_value:,.0f}')
        print(f'   Wealth Multiple: {final_value/100000:.1f}x')
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'adj_return': adj_return,
            'final_value': final_value,
            'returns': None,
            'prices': None
        }

if __name__ == "__main__":
    btc_data = analyze_bitcoin_performance()
