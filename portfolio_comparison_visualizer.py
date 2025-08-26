"""
Portfolio Comparison Visualizer - Create comprehensive charts comparing strategies
Generate visualizations for correlation-optimized vs current portfolio performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from correlation_analysis import analyze_portfolio_performance

def create_portfolio_comparison_charts():
    """Create comprehensive comparison charts for different portfolio strategies."""
    
    print("📊 Creating Portfolio Comparison Visualizations...")
    
    # Get market data
    data_fetcher = MarketDataFetcher(period_years=5)
    symbols = data_fetcher.get_top_assets_by_volume(num_assets=96)
    prices_data = data_fetcher.fetch_historical_data(symbols)
    returns_data = data_fetcher.calculate_returns()
    
    # Define portfolios
    current_portfolio = ['GE', 'XOM', 'NVDA', 'ORCL', 'WMT', 'WFC']
    uncorr_portfolio = ['META', 'WMT', 'UNH', 'COP', 'TLT']
    
    # Filter available stocks
    current_available = [s for s in current_portfolio if s in returns_data.columns][:5]
    uncorr_available = [s for s in uncorr_portfolio if s in returns_data.columns]
    
    # Calculate portfolio performance
    current_metrics = analyze_portfolio_performance(returns_data, current_available)
    uncorr_metrics = analyze_portfolio_performance(returns_data, uncorr_available)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Portfolio Composition Comparison
    ax1 = plt.subplot(3, 3, 1)
    current_weights = [1/len(current_available)] * len(current_available)
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(current_available)))
    wedges1, texts1, autotexts1 = ax1.pie(current_weights, labels=current_available, autopct='%1.1f%%', 
                                          colors=colors1, startangle=90)
    ax1.set_title('Current Portfolio\n(High Return Strategy)', fontsize=14, fontweight='bold', pad=20)
    
    ax2 = plt.subplot(3, 3, 2)
    uncorr_weights = [1/len(uncorr_available)] * len(uncorr_available)
    colors2 = plt.cm.Set2(np.linspace(0, 1, len(uncorr_available)))
    wedges2, texts2, autotexts2 = ax2.pie(uncorr_weights, labels=uncorr_available, autopct='%1.1f%%', 
                                          colors=colors2, startangle=90)
    ax2.set_title('Uncorrelated Portfolio\n(Low Risk Strategy)', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Performance Metrics Comparison
    ax3 = plt.subplot(3, 3, 3)
    metrics_data = {
        'Annual Return': [current_metrics['annual_return']*100, uncorr_metrics['annual_return']*100],
        'Volatility': [current_metrics['volatility']*100, uncorr_metrics['volatility']*100],
        'Sharpe Ratio': [current_metrics['sharpe'], uncorr_metrics['sharpe']],
        'Max Drawdown': [abs(current_metrics['max_drawdown'])*100, abs(uncorr_metrics['max_drawdown'])*100]
    }
    
    x = np.arange(len(metrics_data))
    width = 0.35
    
    current_values = [metrics_data[metric][0] for metric in metrics_data]
    uncorr_values = [metrics_data[metric][1] for metric in metrics_data]
    
    bars1 = ax3.bar(x - width/2, current_values, width, label='Current Portfolio', color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x + width/2, uncorr_values, width, label='Uncorrelated Portfolio', color='#A23B72', alpha=0.8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Values')
    ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(list(metrics_data.keys()), rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # 3. Cumulative Performance
    ax4 = plt.subplot(3, 2, 3)
    current_cumulative = (1 + current_metrics['portfolio_returns']).cumprod()
    uncorr_cumulative = (1 + uncorr_metrics['portfolio_returns']).cumprod()
    
    ax4.plot(current_cumulative.index, current_cumulative.values, 
             label='Current Portfolio', color='#2E86AB', linewidth=2)
    ax4.plot(uncorr_cumulative.index, uncorr_cumulative.values, 
             label='Uncorrelated Portfolio', color='#A23B72', linewidth=2)
    
    ax4.set_title('Cumulative Performance (5 Years)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y-1)*100:.0f}%'))
    
    # 4. 25-Year Projection
    ax5 = plt.subplot(3, 2, 4)
    years = np.arange(0, 26)
    
    # Adjusted returns for future projection
    current_adj_return = current_metrics['annual_return'] * 0.75  # 25% haircut
    uncorr_adj_return = uncorr_metrics['annual_return'] * 0.85   # 15% haircut
    
    current_projection = 100000 * (1 + current_adj_return) ** years
    uncorr_projection = 100000 * (1 + uncorr_adj_return) ** years
    
    ax5.plot(years, current_projection, label=f'Current Portfolio ({current_adj_return:.1%})', 
             color='#2E86AB', linewidth=3, marker='o', markersize=4)
    ax5.plot(years, uncorr_projection, label=f'Uncorrelated Portfolio ({uncorr_adj_return:.1%})', 
             color='#A23B72', linewidth=3, marker='s', markersize=4)
    
    ax5.set_title('25-Year Wealth Projection ($100k Initial)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Years')
    ax5.set_ylabel('Portfolio Value ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Format y-axis as currency
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    
    # 5. Risk-Return Scatter
    ax6 = plt.subplot(3, 2, 5)
    
    # Calculate metrics for all individual stocks
    individual_metrics = []
    for symbol in returns_data.columns:
        stock_returns = returns_data[symbol]
        annual_ret = (1 + stock_returns.mean()) ** 252 - 1
        volatility = stock_returns.std() * np.sqrt(252)
        individual_metrics.append((symbol, annual_ret, volatility))
    
    # Plot individual stocks
    for symbol, ret, vol in individual_metrics:
        if symbol in current_available:
            ax6.scatter(vol*100, ret*100, color='#2E86AB', alpha=0.6, s=50, label='Current Holdings' if symbol == current_available[0] else "")
        elif symbol in uncorr_available:
            ax6.scatter(vol*100, ret*100, color='#A23B72', alpha=0.6, s=50, label='Uncorr Holdings' if symbol == uncorr_available[0] else "")
        else:
            ax6.scatter(vol*100, ret*100, color='lightgray', alpha=0.3, s=20)
    
    # Plot portfolio points
    ax6.scatter(current_metrics['volatility']*100, current_metrics['annual_return']*100, 
               color='#2E86AB', s=300, marker='*', edgecolors='black', linewidth=2, 
               label='Current Portfolio', zorder=5)
    ax6.scatter(uncorr_metrics['volatility']*100, uncorr_metrics['annual_return']*100, 
               color='#A23B72', s=300, marker='*', edgecolors='black', linewidth=2, 
               label='Uncorrelated Portfolio', zorder=5)
    
    ax6.set_xlabel('Volatility (%)')
    ax6.set_ylabel('Annual Return (%)')
    ax6.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. Correlation Heatmap
    ax7 = plt.subplot(3, 2, 6)
    
    # Create correlation matrix for both portfolios
    all_stocks = list(set(current_available + uncorr_available))
    corr_matrix = returns_data[all_stocks].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax7)
    ax7.set_title('Correlation Matrix (All Holdings)', fontsize=14, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    
    # Save the chart
    plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/portfolio_strategy_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Saved: portfolio_strategy_comparison.png")
    
    # Create summary statistics table
    summary_data = {
        'Metric': ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 
                   '25Y Final Value', 'Wealth Multiple', 'Avg Correlation'],
        'Current Portfolio': [
            f"{current_metrics['annual_return']:.2%}",
            f"{current_metrics['volatility']:.2%}",
            f"{current_metrics['sharpe']:.3f}",
            f"{current_metrics['max_drawdown']:.2%}",
            f"${current_projection[-1]:,.0f}",
            f"{current_projection[-1]/100000:.1f}x",
            "0.235"  # From previous analysis
        ],
        'Uncorrelated Portfolio': [
            f"{uncorr_metrics['annual_return']:.2%}",
            f"{uncorr_metrics['volatility']:.2%}",
            f"{uncorr_metrics['sharpe']:.3f}",
            f"{uncorr_metrics['max_drawdown']:.2%}",
            f"${uncorr_projection[-1]:,.0f}",
            f"{uncorr_projection[-1]/100000:.1f}x",
            "0.071"  # From previous analysis
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create summary table visualization
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if j == 1:  # Current Portfolio column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 2:  # Uncorrelated Portfolio column
                table[(i, j)].set_facecolor('#FCE4EC')
    
    plt.title('Portfolio Strategy Comparison Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/portfolio_comparison_table.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Saved: portfolio_comparison_table.png")
    
    return {
        'current_final_value': current_projection[-1],
        'uncorr_final_value': uncorr_projection[-1],
        'current_holdings': current_available,
        'uncorr_holdings': uncorr_available
    }

if __name__ == "__main__":
    results = create_portfolio_comparison_charts()
    print(f"\n🎉 Portfolio comparison visualizations created!")
    print(f"Current Portfolio 25Y Value: ${results['current_final_value']:,.0f}")
    print(f"Uncorrelated Portfolio 25Y Value: ${results['uncorr_final_value']:,.0f}")
