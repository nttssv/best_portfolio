"""
Comprehensive Correlation Analysis - Find the best uncorrelated stocks
Analyze correlation matrix across all 96 tickers to identify optimal diversification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from data_fetcher import MarketDataFetcher
from portfolio_metrics import PortfolioMetrics
import warnings
warnings.filterwarnings('ignore')

def find_least_correlated_stocks(returns_data, num_stocks=5):
    """Find the combination of stocks with lowest average correlation."""
    
    print(f"🔍 Analyzing {len(returns_data.columns)} stocks for optimal diversification...")
    
    # Calculate correlation matrix
    corr_matrix = returns_data.corr()
    
    # Get all possible combinations of num_stocks
    symbols = returns_data.columns.tolist()
    best_combination = None
    lowest_avg_correlation = float('inf')
    
    num_combinations = len(list(combinations(symbols, num_stocks)))
    print(f"📊 Testing {num_combinations:,} combinations...")
    
    # For computational efficiency, sample combinations if too many
    all_combinations = list(combinations(symbols, num_stocks))
    if len(all_combinations) > 100000:
        print(f"⚡ Sampling 100,000 combinations for efficiency...")
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, 100000)
    
    for i, combo in enumerate(all_combinations):
        if i % 10000 == 0:
            print(f"   Progress: {i:,}/{len(all_combinations):,} combinations tested")
        
        # Get correlation submatrix for this combination
        combo_corr = corr_matrix.loc[list(combo), list(combo)]
        
        # Calculate average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(combo_corr, dtype=bool), k=1)
        avg_correlation = combo_corr.where(mask).stack().mean()
        
        if avg_correlation < lowest_avg_correlation:
            lowest_avg_correlation = avg_correlation
            best_combination = combo
    
    return best_combination, lowest_avg_correlation, corr_matrix

def analyze_portfolio_performance(returns_data, stocks):
    """Analyze performance metrics for a given stock combination."""
    
    portfolio_returns = returns_data[list(stocks)]
    
    # Equal weight portfolio
    equal_weight_returns = portfolio_returns.mean(axis=1)
    
    # Calculate metrics
    metrics = {}
    annual_return = (1 + equal_weight_returns.mean()) ** 252 - 1
    volatility = equal_weight_returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cumulative = (1 + equal_weight_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_returns': equal_weight_returns
    }
    
    return metrics

def main():
    """Run comprehensive correlation analysis."""
    
    print("🔬 COMPREHENSIVE CORRELATION ANALYSIS")
    print("="*70)
    print("Finding the best uncorrelated stocks for optimal diversification")
    print("="*70)
    
    # Get all available data
    print("\n📊 Fetching market data for all assets...")
    data_fetcher = MarketDataFetcher(period_years=5)
    symbols = data_fetcher.get_top_assets_by_volume(num_assets=96)
    prices_data = data_fetcher.fetch_historical_data(symbols)
    returns_data = data_fetcher.calculate_returns()
    
    print(f"✅ Loaded {len(returns_data.columns)} assets with {len(returns_data)} trading days")
    
    # Find least correlated 5 stocks
    print(f"\n🎯 Finding 5 least correlated stocks...")
    best_combo, avg_corr, corr_matrix = find_least_correlated_stocks(returns_data, 5)
    
    print(f"\n🏆 BEST 5-STOCK COMBINATION (Lowest Correlation):")
    print("="*60)
    print(f"Stocks: {', '.join(best_combo)}")
    print(f"Average Correlation: {avg_corr:.4f}")
    
    # Analyze performance of uncorrelated portfolio
    uncorr_metrics = analyze_portfolio_performance(returns_data, best_combo)
    
    # Compare with current recommended portfolio
    current_portfolio = ['GE', 'XOM', 'NVDA', 'ORCL', 'WMT']
    current_available = [s for s in current_portfolio if s in returns_data.columns]
    
    if len(current_available) >= 5:
        current_metrics = analyze_portfolio_performance(returns_data, current_available[:5])
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print("="*60)
        print(f"{'Metric':<20} {'Uncorrelated':<15} {'Current':<15} {'Winner'}")
        print("-"*60)
        print(f"{'Annual Return':<20} {uncorr_metrics['annual_return']:>13.2%} {current_metrics['annual_return']:>13.2%} {'Uncorr' if uncorr_metrics['annual_return'] > current_metrics['annual_return'] else 'Current'}")
        print(f"{'Volatility':<20} {uncorr_metrics['volatility']:>13.2%} {current_metrics['volatility']:>13.2%} {'Uncorr' if uncorr_metrics['volatility'] < current_metrics['volatility'] else 'Current'}")
        print(f"{'Sharpe Ratio':<20} {uncorr_metrics['sharpe']:>13.3f} {current_metrics['sharpe']:>13.3f} {'Uncorr' if uncorr_metrics['sharpe'] > current_metrics['sharpe'] else 'Current'}")
        print(f"{'Max Drawdown':<20} {uncorr_metrics['max_drawdown']:>13.2%} {current_metrics['max_drawdown']:>13.2%} {'Uncorr' if uncorr_metrics['max_drawdown'] > current_metrics['max_drawdown'] else 'Current'}")
        
        # Calculate correlations for current portfolio
        current_corr_matrix = corr_matrix.loc[current_available[:5], current_available[:5]]
        mask = np.triu(np.ones_like(current_corr_matrix, dtype=bool), k=1)
        current_avg_corr = current_corr_matrix.where(mask).stack().mean()
        
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print("-"*40)
        print(f"Uncorrelated Portfolio Avg Correlation: {avg_corr:.4f}")
        print(f"Current Portfolio Avg Correlation: {current_avg_corr:.4f}")
        print(f"Diversification Improvement: {((current_avg_corr - avg_corr) / current_avg_corr * 100):+.1f}%")
    
    # Find top 10 least correlated combinations
    print(f"\n🔍 TOP 10 LEAST CORRELATED 5-STOCK COMBINATIONS:")
    print("="*60)
    
    # Quick analysis of top combinations
    all_combinations = list(combinations(returns_data.columns, 5))
    if len(all_combinations) > 50000:
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, 50000)
    
    combination_scores = []
    for combo in all_combinations:
        combo_corr = corr_matrix.loc[list(combo), list(combo)]
        mask = np.triu(np.ones_like(combo_corr, dtype=bool), k=1)
        avg_correlation = combo_corr.where(mask).stack().mean()
        combination_scores.append((combo, avg_correlation))
    
    # Sort by correlation
    combination_scores.sort(key=lambda x: x[1])
    
    print(f"{'Rank':<4} {'Stocks':<30} {'Avg Corr':<10} {'Performance'}")
    print("-"*70)
    
    for i, (combo, corr) in enumerate(combination_scores[:10]):
        metrics = analyze_portfolio_performance(returns_data, combo)
        print(f"{i+1:<4} {', '.join(combo):<30} {corr:>8.4f} {metrics['annual_return']:>8.1%} / {metrics['sharpe']:>5.2f}")
    
    # Sector analysis
    print(f"\n🏭 SECTOR ANALYSIS OF BEST COMBINATION:")
    print("="*50)
    
    # Simple sector classification (you could enhance this)
    sector_map = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
        'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Technology', 'ORCL': 'Technology',
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
        'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer',
        'GE': 'Industrial', 'CAT': 'Industrial', 'BA': 'Industrial'
    }
    
    for stock in best_combo:
        sector = sector_map.get(stock, 'Other')
        print(f"  {stock}: {sector}")
    
    # 25-year projection for uncorrelated portfolio
    print(f"\n💰 25-YEAR PROJECTION - UNCORRELATED PORTFOLIO:")
    print("="*50)
    
    initial_investment = 100000
    annual_return = uncorr_metrics['annual_return']
    
    # Apply conservative adjustment
    if annual_return > 0.25:
        adjusted_return = annual_return * 0.75
    elif annual_return > 0.15:
        adjusted_return = annual_return * 0.85
    else:
        adjusted_return = annual_return * 0.95
    
    final_value = initial_investment * (1 + adjusted_return) ** 25
    total_return = (final_value / initial_investment - 1) * 100
    
    print(f"Historical Return: {annual_return:.2%}")
    print(f"Adjusted Return: {adjusted_return:.2%}")
    print(f"Initial Investment: ${initial_investment:,}")
    print(f"25-Year Final Value: ${final_value:,.0f}")
    print(f"Total Return: {total_return:.1f}%")
    print(f"Wealth Multiple: {final_value/initial_investment:.1f}x")
    
    return {
        'best_combination': best_combo,
        'avg_correlation': avg_corr,
        'metrics': uncorr_metrics,
        'final_value': final_value
    }

if __name__ == "__main__":
    results = main()
    
    print(f"\n🎉 CORRELATION ANALYSIS COMPLETED!")
    print("="*50)
    print(f"✅ Best 5-stock combination: {', '.join(results['best_combination'])}")
    print(f"✅ Average correlation: {results['avg_correlation']:.4f}")
    print(f"✅ 25-year projection: ${results['final_value']:,.0f}")
