"""
Quarterly Growth Strategy - Dynamic rebalancing to top 5 growth performers
Every quarter, select top 5 growth stocks/ETFs and optimize portfolio using MPT
Backtest over 20 years to analyze performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from portfolio_optimizer import EfficientFrontierOptimizer, OptimizationConstraints
from portfolio_metrics import PortfolioMetrics

class QuarterlyGrowthStrategy:
    """Quarterly rebalancing strategy based on top growth performers."""
    
    def __init__(self, universe_size=96, top_growth=5):
        self.universe_size = universe_size
        self.top_growth = top_growth
        self.quarterly_portfolios = []
        self.performance_history = []
        
    def get_quarterly_dates(self, start_date, end_date):
        """Generate quarterly rebalancing dates."""
        dates = []
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        while current <= end:
            dates.append(current)
            # Move to next quarter
            if current.month in [1, 2, 3]:
                current = current.replace(month=4, day=1)
            elif current.month in [4, 5, 6]:
                current = current.replace(month=7, day=1)
            elif current.month in [7, 8, 9]:
                current = current.replace(month=10, day=1)
            else:
                current = current.replace(year=current.year + 1, month=1, day=1)
        
        return dates
    
    def calculate_growth_performance(self, prices_data, lookback_days=252):
        """Calculate growth performance for each asset over lookback period."""
        growth_metrics = {}
        
        for symbol in prices_data.columns:
            try:
                price_series = prices_data[symbol].dropna()
                if len(price_series) >= lookback_days:
                    # Calculate total return over lookback period
                    start_price = price_series.iloc[-lookback_days]
                    end_price = price_series.iloc[-1]
                    total_return = (end_price / start_price) - 1
                    
                    # Calculate volatility-adjusted return (Sharpe-like metric)
                    returns = price_series.pct_change().dropna()
                    if len(returns) >= lookback_days - 1:
                        recent_returns = returns.tail(lookback_days - 1)
                        volatility = recent_returns.std() * np.sqrt(252)
                        risk_adj_return = total_return / volatility if volatility > 0 else 0
                        
                        growth_metrics[symbol] = {
                            'total_return': total_return,
                            'volatility': volatility,
                            'risk_adj_return': risk_adj_return,
                            'sharpe_proxy': risk_adj_return
                        }
            except Exception as e:
                continue
        
        return growth_metrics
    
    def select_top_growth_stocks(self, growth_metrics, top_n=5):
        """Select top N growth stocks based on risk-adjusted returns."""
        # Sort by risk-adjusted return (Sharpe proxy)
        sorted_stocks = sorted(growth_metrics.items(), 
                             key=lambda x: x[1]['risk_adj_return'], 
                             reverse=True)
        
        top_stocks = [stock[0] for stock in sorted_stocks[:top_n]]
        return top_stocks
    
    def optimize_portfolio_mpt(self, returns_data, selected_stocks):
        """Apply MPT optimization to selected stocks."""
        try:
            # Filter returns for selected stocks
            portfolio_returns = returns_data[selected_stocks].dropna()
            
            if len(portfolio_returns.columns) < 3:
                # If less than 3 stocks, use equal weights
                weights = np.array([1/len(selected_stocks)] * len(selected_stocks))
                return dict(zip(selected_stocks, weights))
            
            # Initialize optimizer
            optimizer = EfficientFrontierOptimizer(portfolio_returns)
            
            # Set constraints for growth strategy
            constraints = OptimizationConstraints(
                min_weight=0.05,  # Minimum 5% allocation
                max_weight=0.50,  # Maximum 50% in any single stock
                min_return=0.15   # Target 15%+ returns
            )
            
            # Optimize for maximum Sharpe ratio
            result = optimizer.optimize_portfolio(constraints=constraints)
            
            if result['optimization_status'] == 'optimal':
                weights_dict = dict(zip(selected_stocks, result['weights']))
                return weights_dict
            else:
                # Fallback to equal weights
                weights = np.array([1/len(selected_stocks)] * len(selected_stocks))
                return dict(zip(selected_stocks, weights))
                
        except Exception as e:
            print(f"Optimization failed: {e}, using equal weights")
            weights = np.array([1/len(selected_stocks)] * len(selected_stocks))
            return dict(zip(selected_stocks, weights))
    
    def backtest_strategy(self, start_date='2005-01-01', end_date='2025-01-01'):
        """Backtest the quarterly growth strategy over 20 years."""
        
        print("🚀 QUARTERLY GROWTH STRATEGY BACKTEST (20 Years)")
        print("="*70)
        print("Strategy: Every quarter, select top 5 growth performers and optimize with MPT")
        print("="*70)
        
        # Get market data
        print("\n📊 Fetching 20-year market data...")
        data_fetcher = MarketDataFetcher(period_years=20)
        
        symbols = data_fetcher.get_top_assets_by_volume(num_assets=self.universe_size)
        prices_data = data_fetcher.fetch_historical_data(symbols)
        returns_data = data_fetcher.calculate_returns()
        
        print(f"✅ Loaded {len(prices_data.columns)} assets from {start_date} to {end_date}")
        
        # Generate quarterly rebalancing dates using actual data dates
        start_actual = prices_data.index[0]
        end_actual = prices_data.index[-1]
        quarterly_dates = self.get_quarterly_dates(start_actual, end_actual)
        print(f"📅 Generated {len(quarterly_dates)} quarterly rebalancing periods")
        
        # Initialize portfolio tracking
        portfolio_value = 100000  # Start with $100k
        portfolio_history = []
        holdings_history = []
        
        print(f"\n🔄 Running quarterly rebalancing strategy...")
        
        for i, rebal_date in enumerate(tqdm(quarterly_dates[:-1])):
            try:
                # Get data up to rebalancing date
                available_data = prices_data[prices_data.index <= rebal_date]
                available_returns = returns_data[returns_data.index <= rebal_date]
                
                if len(available_data) < 252:  # Need at least 1 year of data
                    continue
                
                # Calculate growth metrics for stock selection
                growth_metrics = self.calculate_growth_performance(available_data)
                
                if len(growth_metrics) < self.top_growth:
                    continue
                
                # Select top growth stocks
                top_stocks = self.select_top_growth_stocks(growth_metrics, self.top_growth)
                
                # Optimize portfolio using MPT
                weights = self.optimize_portfolio_mpt(available_returns, top_stocks)
                
                # Calculate portfolio performance until next rebalancing
                next_date = quarterly_dates[i + 1] if i + 1 < len(quarterly_dates) else pd.to_datetime(end_date)
                
                # Get returns for the quarter
                quarter_returns = returns_data[
                    (returns_data.index > rebal_date) & 
                    (returns_data.index <= next_date)
                ]
                
                if len(quarter_returns) > 0:
                    # Calculate portfolio returns for the quarter
                    for date, daily_returns in quarter_returns.iterrows():
                        daily_portfolio_return = 0
                        for stock, weight in weights.items():
                            if stock in daily_returns.index:
                                daily_portfolio_return += weight * daily_returns[stock]
                        
                        # Update portfolio value and track history
                        portfolio_value *= (1 + daily_portfolio_return)
                        portfolio_history.append({
                            'date': date,
                            'value': portfolio_value,
                            'holdings': top_stocks.copy(),
                            'weights': weights.copy()
                        })
                
                # Store quarterly portfolio info
                self.quarterly_portfolios.append({
                    'date': rebal_date,
                    'holdings': top_stocks,
                    'weights': weights,
                    'growth_metrics': growth_metrics
                })
                
            except Exception as e:
                print(f"Error processing {rebal_date}: {e}")
                continue
        
        # Calculate final performance metrics
        if portfolio_history:
            final_value = portfolio_history[-1]['value']
            total_return = (final_value / 100000) - 1
            years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
            annualized_return = (final_value / 100000) ** (1/years) - 1
            
            print(f"\n🏆 QUARTERLY GROWTH STRATEGY RESULTS:")
            print("="*50)
            print(f"Initial Investment: $100,000")
            print(f"Final Value: ${final_value:,.0f}")
            print(f"Total Return: {total_return:.1%}")
            print(f"Annualized Return: {annualized_return:.2%}")
            print(f"Wealth Multiple: {final_value/100000:.1f}x")
            print(f"Number of Rebalances: {len(self.quarterly_portfolios)}")
            
            return {
                'final_value': final_value,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'portfolio_history': portfolio_history,
                'quarterly_portfolios': self.quarterly_portfolios
            }
        
        return None
    
    def create_performance_visualization(self, results):
        """Create comprehensive performance visualization."""
        
        if not results:
            print("No results to visualize")
            return
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Portfolio Value Growth
        ax1 = plt.subplot(3, 2, 1)
        portfolio_history = results['portfolio_history']
        dates = [entry['date'] for entry in portfolio_history]
        values = [entry['value'] for entry in portfolio_history]
        
        ax1.plot(dates, values, linewidth=2, color='#2E86AB', label='Quarterly Growth Strategy')
        ax1.set_title('Portfolio Value Growth (20 Years)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
        
        # Add benchmark comparison (simple buy & hold S&P 500 proxy)
        spy_growth = 100000 * (1.10) ** 20  # Assume 10% annual S&P return
        ax1.axhline(y=spy_growth, color='red', linestyle='--', alpha=0.7, label='S&P 500 Benchmark')
        ax1.legend()
        
        # 2. Holdings Evolution
        ax2 = plt.subplot(3, 2, 2)
        
        # Count frequency of holdings
        all_holdings = {}
        for quarter in self.quarterly_portfolios:
            for holding in quarter['holdings']:
                all_holdings[holding] = all_holdings.get(holding, 0) + 1
        
        # Top 10 most frequent holdings
        top_holdings = sorted(all_holdings.items(), key=lambda x: x[1], reverse=True)[:10]
        holdings_names = [h[0] for h in top_holdings]
        holdings_counts = [h[1] for h in top_holdings]
        
        bars = ax2.bar(holdings_names, holdings_counts, color='#A23B72', alpha=0.8)
        ax2.set_title('Most Frequently Selected Stocks', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Quarters Selected')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, holdings_counts):
            ax2.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, count),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # 3. Quarterly Returns Distribution
        ax3 = plt.subplot(3, 2, 3)
        
        # Calculate quarterly returns
        quarterly_returns = []
        for i in range(1, len(values)):
            if i % 63 == 0:  # Approximately quarterly (63 trading days)
                qtr_return = (values[i] / values[i-63]) - 1
                quarterly_returns.append(qtr_return)
        
        ax3.hist(quarterly_returns, bins=20, alpha=0.7, color='#F18F01', edgecolor='black')
        ax3.set_title('Quarterly Returns Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Quarterly Return')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=np.mean(quarterly_returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(quarterly_returns):.1%}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe Ratio
        ax4 = plt.subplot(3, 2, 4)
        
        # Calculate rolling annual Sharpe ratio
        daily_returns = []
        for i in range(1, len(values)):
            daily_ret = (values[i] / values[i-1]) - 1
            daily_returns.append(daily_ret)
        
        daily_returns = pd.Series(daily_returns)
        rolling_sharpe = []
        window = 252  # 1 year
        
        for i in range(window, len(daily_returns)):
            returns_window = daily_returns.iloc[i-window:i]
            annual_return = returns_window.mean() * 252
            annual_vol = returns_window.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            rolling_sharpe.append(sharpe)
        
        sharpe_dates = dates[window:window+len(rolling_sharpe)]
        ax4.plot(sharpe_dates, rolling_sharpe, color='#C73E1D', linewidth=2)
        ax4.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        ax4.legend()
        
        # 5. Strategy Performance Summary
        ax5 = plt.subplot(3, 1, 3)
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create performance summary table
        summary_data = [
            ['Metric', 'Quarterly Growth Strategy', 'Benchmark (10% annual)'],
            ['Final Value', f"${results['final_value']:,.0f}", f"${spy_growth:,.0f}"],
            ['Total Return', f"{results['total_return']:.1%}", "572.7%"],
            ['Annualized Return', f"{results['annualized_return']:.2%}", "10.00%"],
            ['Wealth Multiple', f"{results['final_value']/100000:.1f}x", "6.7x"],
            ['Avg Quarterly Return', f"{np.mean(quarterly_returns):.1%}" if quarterly_returns else "N/A", "2.41%"],
            ['Strategy Advantage', f"{((results['final_value']/spy_growth)-1)*100:+.1f}%", "0.0%"]
        ]
        
        table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if j == 1 and i == 6:  # Highlight strategy advantage
                    table[(i, j)].set_facecolor('#E8F5E8' if '+' in summary_data[i][j] else '#FFE8E8')
                elif i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
        
        plt.suptitle('Quarterly Growth Strategy - 20 Year Backtest Results', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save the visualization
        plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/quarterly_growth_strategy.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        
        print("✅ Saved: quarterly_growth_strategy.png")
        
        return results

def main():
    """Run the quarterly growth strategy analysis."""
    
    strategy = QuarterlyGrowthStrategy(universe_size=96, top_growth=5)
    results = strategy.backtest_strategy(start_date='2005-01-01', end_date='2025-01-01')
    
    if results:
        strategy.create_performance_visualization(results)
        
        print(f"\n🎯 STRATEGY COMPARISON:")
        print("="*40)
        print(f"Quarterly Growth Strategy: ${results['final_value']:,.0f}")
        print(f"Static Portfolio (from memory): ~$4.4M")
        print(f"Performance Advantage: {((results['final_value']/4400000)-1)*100:+.1f}%")
        
        return results
    else:
        print("❌ Strategy backtest failed")
        return None

if __name__ == "__main__":
    results = main()
