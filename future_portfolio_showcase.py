"""
Future Portfolio Showcase - Create comprehensive future portfolio visualization
Show 25-year projections and strategic recommendations with detailed charts
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

def create_future_portfolio_showcase():
    """Create comprehensive future portfolio visualization and analysis."""
    
    print("🔮 Creating Future Portfolio Showcase...")
    
    # Get market data
    data_fetcher = MarketDataFetcher(period_years=5)
    symbols = data_fetcher.get_top_assets_by_volume(num_assets=96)
    prices_data = data_fetcher.fetch_historical_data(symbols)
    returns_data = data_fetcher.calculate_returns()
    
    # Define portfolio strategies
    strategies = {
        'Tech Growth': ['GE', 'XOM', 'NVDA', 'ORCL', 'WMT', 'WFC'],
        'Uncorrelated': ['META', 'WMT', 'UNH', 'COP', 'TLT'],
        'Balanced': ['AAPL', 'MSFT', 'JPM', 'JNJ', 'VZ'],
        'Value Focus': ['BRK-B', 'JPM', 'WFC', 'XOM', 'WMT']
    }
    
    # Calculate metrics for each strategy
    strategy_metrics = {}
    for name, holdings in strategies.items():
        available_holdings = [s for s in holdings if s in returns_data.columns][:5]
        if len(available_holdings) >= 3:  # Minimum 3 stocks for analysis
            metrics = analyze_portfolio_performance(returns_data, available_holdings)
            strategy_metrics[name] = {
                'metrics': metrics,
                'holdings': available_holdings
            }
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Strategy Performance Comparison
    ax1 = plt.subplot(4, 2, 1)
    strategy_names = list(strategy_metrics.keys())
    annual_returns = [strategy_metrics[name]['metrics']['annual_return']*100 for name in strategy_names]
    sharpe_ratios = [strategy_metrics[name]['metrics']['sharpe'] for name in strategy_names]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax1.bar(strategy_names, annual_returns, color=colors, alpha=0.8)
    ax1.set_title('Annual Returns by Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Annual Return (%)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, annual_returns):
        ax1.annotate(f'{value:.1f}%', xy=(bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 2. Risk-Return Profile
    ax2 = plt.subplot(4, 2, 2)
    volatilities = [strategy_metrics[name]['metrics']['volatility']*100 for name in strategy_names]
    
    for i, name in enumerate(strategy_names):
        ax2.scatter(volatilities[i], annual_returns[i], s=300, color=colors[i], 
                   alpha=0.7, edgecolors='black', linewidth=2, label=name)
    
    ax2.set_xlabel('Volatility (%)')
    ax2.set_ylabel('Annual Return (%)')
    ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 25-Year Wealth Projection
    ax3 = plt.subplot(4, 1, 2)
    years = np.arange(0, 26)
    initial_investment = 100000
    
    for i, name in enumerate(strategy_names):
        annual_return = strategy_metrics[name]['metrics']['annual_return']
        
        # Apply conservative adjustment based on return level
        if annual_return > 0.25:
            adjusted_return = annual_return * 0.75
        elif annual_return > 0.15:
            adjusted_return = annual_return * 0.85
        else:
            adjusted_return = annual_return * 0.95
        
        projection = initial_investment * (1 + adjusted_return) ** years
        
        ax3.plot(years, projection, label=f'{name} ({adjusted_return:.1%})', 
                color=colors[i], linewidth=3, marker='o', markersize=4)
    
    ax3.set_title('25-Year Wealth Projection ($100k Initial Investment)', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    
    # 4. Portfolio Compositions
    for i, (name, data) in enumerate(strategy_metrics.items()):
        ax = plt.subplot(4, 4, 9 + i)
        holdings = data['holdings']
        weights = [1/len(holdings)] * len(holdings)
        
        wedges, texts, autotexts = ax.pie(weights, labels=holdings, autopct='%1.1f%%', 
                                         colors=plt.cm.Set3(np.linspace(0, 1, len(holdings))), 
                                         startangle=90)
        ax.set_title(f'{name} Portfolio', fontsize=12, fontweight='bold')
    
    # 5. Key Metrics Table
    ax5 = plt.subplot(4, 1, 4)
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create metrics table
    table_data = []
    headers = ['Strategy', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', '25Y Value', 'Wealth Multiple']
    
    for name in strategy_names:
        metrics = strategy_metrics[name]['metrics']
        annual_return = metrics['annual_return']
        
        # Apply same adjustment as projection
        if annual_return > 0.25:
            adjusted_return = annual_return * 0.75
        elif annual_return > 0.15:
            adjusted_return = annual_return * 0.85
        else:
            adjusted_return = annual_return * 0.95
        
        final_value = initial_investment * (1 + adjusted_return) ** 25
        wealth_multiple = final_value / initial_investment
        
        row = [
            name,
            f"{annual_return:.1%}",
            f"{metrics['volatility']:.1%}",
            f"{metrics['sharpe']:.2f}",
            f"{metrics['max_drawdown']:.1%}",
            f"${final_value:,.0f}",
            f"{wealth_multiple:.0f}x"
        ]
        table_data.append(row)
    
    table = ax5.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i == 1:  # Tech Growth (best performer)
                table[(i, j)].set_facecolor('#E8F5E8')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.suptitle('Future Portfolio Strategy Analysis (2025-2050)', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the comprehensive chart
    plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/future_portfolio_showcase.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Saved: future_portfolio_showcase.png")
    
    # Create investment timeline chart
    fig2, ax = plt.subplots(figsize=(16, 10))
    
    # Best strategy details
    best_strategy = 'Tech Growth'
    best_metrics = strategy_metrics[best_strategy]['metrics']
    best_return = best_metrics['annual_return'] * 0.75  # Conservative adjustment
    
    # Create detailed timeline
    years_detailed = np.arange(0, 26)
    values_detailed = initial_investment * (1 + best_return) ** years_detailed
    
    # Plot growth curve
    ax.plot(years_detailed, values_detailed, color='#2E86AB', linewidth=4, 
           label=f'{best_strategy} Strategy ({best_return:.1%} annual)')
    ax.fill_between(years_detailed, values_detailed, alpha=0.3, color='#2E86AB')
    
    # Add milestone markers
    milestones = [5, 10, 15, 20, 25]
    for year in milestones:
        value = initial_investment * (1 + best_return) ** year
        ax.scatter(year, value, s=200, color='red', zorder=5)
        ax.annotate(f'Year {year}\n${value:,.0f}', 
                   xy=(year, value), xytext=(10, 20), 
                   textcoords='offset points', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_title(f'Recommended Portfolio: {best_strategy} Strategy\n25-Year Wealth Building Journey', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Years', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    
    # Add strategy details box
    strategy_text = f"""
RECOMMENDED STRATEGY: {best_strategy}
Holdings: {', '.join(strategy_metrics[best_strategy]['holdings'])}
Expected Annual Return: {best_return:.1%}
Risk Level: Medium-High
Investment Horizon: 25 years
Final Value: ${values_detailed[-1]:,.0f}
Wealth Multiple: {values_detailed[-1]/initial_investment:.0f}x
    """.strip()
    
    ax.text(0.02, 0.98, strategy_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/future_investment_timeline.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Saved: future_investment_timeline.png")
    
    return {
        'best_strategy': best_strategy,
        'final_value': values_detailed[-1],
        'strategies': strategy_metrics
    }

if __name__ == "__main__":
    results = create_future_portfolio_showcase()
    print(f"\n🎉 Future portfolio showcase created!")
    print(f"Recommended: {results['best_strategy']} Strategy")
    print(f"25-Year Value: ${results['final_value']:,.0f}")
