"""
Final Analysis - Find best portfolio and create comprehensive visualizations for README
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from advanced_portfolio_optimizer import AdvancedPortfolioAnalysis, MultiObjectiveOptimizer
from portfolio_evolution_tracker import PortfolioEvolutionTracker

def find_best_portfolio():
    """Find the best performing portfolio from all analyses."""
    print("🔍 Finding Best Portfolio from All Analyses")
    print("="*60)
    
    # Load data
    data_fetcher = MarketDataFetcher(period_years=20)
    symbols = data_fetcher.get_top_assets_by_volume(num_assets=80)
    prices_data = data_fetcher.fetch_historical_data(symbols)
    returns_data = data_fetcher.calculate_returns()
    
    print(f"✅ Loaded {len(returns_data.columns)} assets, {len(returns_data)} days")
    
    # Run advanced optimization
    advanced_analysis = AdvancedPortfolioAnalysis()
    results = advanced_analysis.run_comprehensive_analysis(period_years=20)
    
    # Find best portfolio from optimized portfolios
    best_portfolio = None
    best_metrics = None
    best_scenario = None
    best_return = 0
    
    for scenario, result in results['optimized_portfolios'].items():
        if result['metrics'] and result['metrics']['annualized_return'] > best_return:
            best_return = result['metrics']['annualized_return']
            best_portfolio = result['weights']
            best_metrics = result['metrics']
            best_scenario = scenario
    
    # Also check Pareto frontier
    for portfolio in results['pareto_portfolios']:
        if portfolio['metrics']['annualized_return'] > best_return:
            best_return = portfolio['metrics']['annualized_return']
            best_portfolio = portfolio['weights']
            best_metrics = portfolio['metrics']
            best_scenario = 'pareto_optimal'
    
    return {
        'weights': best_portfolio,
        'metrics': best_metrics,
        'scenario': best_scenario,
        'returns_data': returns_data,
        'prices_data': prices_data,
        'symbols': returns_data.columns.tolist()
    }

def calculate_25_year_projection(portfolio_data):
    """Calculate 25-year projection from $100k."""
    annual_return = portfolio_data['metrics']['annualized_return']
    initial_investment = 100000
    years = 25
    
    # Compound growth calculation
    final_value = initial_investment * (1 + annual_return) ** years
    total_return = (final_value / initial_investment - 1) * 100
    
    return {
        'initial_investment': initial_investment,
        'annual_return': annual_return,
        'final_value': final_value,
        'total_return': total_return,
        'years': years
    }

def create_portfolio_composition_chart(portfolio_data, save_path="results/plots/best_portfolio_composition.png"):
    """Create portfolio composition pie chart."""
    weights = portfolio_data['weights']
    symbols = portfolio_data['symbols']
    
    # Get top 10 holdings
    weight_series = pd.Series(weights, index=symbols).sort_values(ascending=False)
    top_10 = weight_series.head(10)
    others = weight_series.tail(-10).sum()
    
    if others > 0:
        top_10['Others'] = others
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_10)))
    
    wedges, texts, autotexts = plt.pie(top_10.values, labels=top_10.index, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    
    plt.title('Best Portfolio Composition - Top Holdings', fontsize=16, fontweight='bold', pad=20)
    
    # Add metrics text
    metrics = portfolio_data['metrics']
    metrics_text = f"""Portfolio Metrics:
Annual Return: {metrics['annualized_return']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Volatility: {metrics['annualized_volatility']:.2%}"""
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_growth_projection_chart(projection_data, save_path="results/plots/growth_projection_25yr.png"):
    """Create 25-year growth projection chart."""
    years = np.arange(0, projection_data['years'] + 1)
    values = projection_data['initial_investment'] * (1 + projection_data['annual_return']) ** years
    
    plt.figure(figsize=(12, 8))
    plt.plot(years, values, linewidth=3, color='green', marker='o', markersize=4)
    plt.fill_between(years, values, alpha=0.3, color='green')
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.title('Portfolio Growth Projection - 25 Years', fontsize=16, fontweight='bold')
    plt.xlabel('Years', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate(f'Start: ${projection_data["initial_investment"]:,}', 
                xy=(0, projection_data['initial_investment']), 
                xytext=(2, projection_data['initial_investment'] * 1.5),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, fontweight='bold')
    
    plt.annotate(f'End: ${projection_data["final_value"]:,.0f}', 
                xy=(projection_data['years'], projection_data['final_value']), 
                xytext=(projection_data['years']-3, projection_data['final_value'] * 0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold')
    
    # Add summary text
    summary_text = f"""Investment Summary:
Initial: ${projection_data['initial_investment']:,}
Annual Return: {projection_data['annual_return']:.2%}
Final Value: ${projection_data['final_value']:,.0f}
Total Return: {projection_data['total_return']:.1f}%"""
    
    plt.figtext(0.02, 0.98, summary_text, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_performance_comparison_chart(portfolio_data, save_path="results/plots/performance_comparison.png"):
    """Create performance comparison chart."""
    metrics = portfolio_data['metrics']
    
    # Benchmark comparisons
    benchmarks = {
        'S&P 500': {'return': 0.10, 'sharpe': 0.7, 'drawdown': 0.20},
        'Balanced Fund': {'return': 0.08, 'sharpe': 0.6, 'drawdown': 0.15},
        'Our Portfolio': {
            'return': metrics['annualized_return'], 
            'sharpe': metrics['sharpe_ratio'], 
            'drawdown': metrics['max_drawdown']
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Returns comparison
    names = list(benchmarks.keys())
    returns = [benchmarks[name]['return'] * 100 for name in names]
    colors = ['gray', 'lightblue', 'green']
    
    bars1 = axes[0].bar(names, returns, color=colors)
    axes[0].set_title('Annual Returns (%)', fontweight='bold')
    axes[0].set_ylabel('Return (%)')
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Sharpe ratio comparison
    sharpes = [benchmarks[name]['sharpe'] for name in names]
    bars2 = axes[1].bar(names, sharpes, color=colors)
    axes[1].set_title('Sharpe Ratio', fontweight='bold')
    axes[1].set_ylabel('Sharpe Ratio')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Max drawdown comparison
    drawdowns = [benchmarks[name]['drawdown'] * 100 for name in names]
    bars3 = axes[2].bar(names, drawdowns, color=colors)
    axes[2].set_title('Maximum Drawdown (%)', fontweight='bold')
    axes[2].set_ylabel('Drawdown (%)')
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Portfolio Performance vs Benchmarks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def print_portfolio_summary(portfolio_data, projection_data):
    """Print comprehensive portfolio summary."""
    print("\n" + "="*80)
    print("🏆 BEST PORTFOLIO ANALYSIS - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    metrics = portfolio_data['metrics']
    weights = portfolio_data['weights']
    symbols = portfolio_data['symbols']
    
    print(f"\n📊 PORTFOLIO PERFORMANCE METRICS")
    print("-" * 50)
    print(f"Annual Return:        {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.3f}")
    print(f"Volatility:           {metrics['annualized_volatility']:.2%}")
    print(f"Maximum Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"Sortino Ratio:        {metrics['sortino_ratio']:.3f}")
    print(f"Calmar Ratio:         {metrics['calmar_ratio']:.3f}")
    print(f"VaR (5%):            {metrics['var_5']:.2%}")
    print(f"CVaR (5%):           {metrics['cvar_5']:.2%}")
    
    print(f"\n💰 25-YEAR INVESTMENT PROJECTION")
    print("-" * 50)
    print(f"Initial Investment:   ${projection_data['initial_investment']:,}")
    print(f"Annual Return:        {projection_data['annual_return']:.2%}")
    print(f"Investment Period:    {projection_data['years']} years")
    print(f"Final Value:          ${projection_data['final_value']:,.0f}")
    print(f"Total Return:         {projection_data['total_return']:.1f}%")
    print(f"Wealth Multiple:      {projection_data['final_value']/projection_data['initial_investment']:.1f}x")
    
    print(f"\n🎯 TOP 15 PORTFOLIO HOLDINGS")
    print("-" * 50)
    weight_series = pd.Series(weights, index=symbols).sort_values(ascending=False)
    top_15 = weight_series.head(15)
    
    print(f"{'Asset':<8} {'Weight':<8} {'Value ($100k)':<12}")
    print("-" * 30)
    for asset, weight in top_15.items():
        value = weight * 100000
        print(f"{asset:<8} {weight:.2%}    ${value:>8,.0f}")
    
    others_weight = weight_series.tail(-15).sum()
    if others_weight > 0:
        print(f"{'Others':<8} {others_weight:.2%}    ${others_weight*100000:>8,.0f}")
    
    print(f"\n📈 PORTFOLIO STATISTICS")
    print("-" * 50)
    print(f"Total Assets:         {len([w for w in weights if w > 0.001])}")
    print(f"Concentration (HHI):  {np.sum(weights**2):.3f}")
    print(f"Max Single Weight:    {np.max(weights):.2%}")
    print(f"Min Single Weight:    {np.min(weights[weights > 0]):.2%}")
    print(f"Effective Assets:     {1/np.sum(weights**2):.1f}")
    
    return {
        'top_holdings': top_15.to_dict(),
        'summary_stats': {
            'annual_return': metrics['annualized_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'final_value_25yr': projection_data['final_value']
        }
    }

def main():
    """Run final analysis and create all visualizations."""
    print("🚀 FINAL PORTFOLIO ANALYSIS")
    print("="*60)
    
    # Create plots directory
    os.makedirs("results/plots", exist_ok=True)
    
    # Find best portfolio
    portfolio_data = find_best_portfolio()
    
    # Calculate 25-year projection
    projection_data = calculate_25_year_projection(portfolio_data)
    
    # Create visualizations
    print("\n🎨 Creating Visualization Charts...")
    composition_chart = create_portfolio_composition_chart(portfolio_data)
    growth_chart = create_growth_projection_chart(projection_data)
    comparison_chart = create_performance_comparison_chart(portfolio_data)
    
    print(f"✅ Charts saved:")
    print(f"   - {composition_chart}")
    print(f"   - {growth_chart}")
    print(f"   - {comparison_chart}")
    
    # Print comprehensive summary
    summary = print_portfolio_summary(portfolio_data, projection_data)
    
    print(f"\n🎉 FINAL ANALYSIS COMPLETED!")
    print("="*80)
    
    return {
        'portfolio_data': portfolio_data,
        'projection_data': projection_data,
        'summary': summary,
        'charts': [composition_chart, growth_chart, comparison_chart]
    }

if __name__ == "__main__":
    results = main()
