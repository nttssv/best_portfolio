"""
Portfolio Allocation Timeline - Show stock allocation percentages by year
Create detailed visualization of portfolio composition evolution over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quarterly_growth_strategy import QuarterlyGrowthStrategy

def create_allocation_timeline():
    """Create comprehensive allocation timeline visualization."""
    
    print("📊 Creating Portfolio Allocation Timeline...")
    
    # Run the quarterly growth strategy to get historical data
    strategy = QuarterlyGrowthStrategy(universe_size=96, top_growth=5)
    results = strategy.backtest_strategy(start_date='2005-01-01', end_date='2025-01-01')
    
    if not results or not strategy.quarterly_portfolios:
        print("❌ No data available for allocation timeline")
        return
    
    # Extract allocation data by year
    allocation_data = []
    for quarter_info in strategy.quarterly_portfolios:
        year = quarter_info['date'].year
        quarter = f"Q{((quarter_info['date'].month-1)//3)+1}"
        
        for stock, weight in quarter_info['weights'].items():
            allocation_data.append({
                'Year': year,
                'Quarter': quarter,
                'Date': quarter_info['date'],
                'Stock': stock,
                'Weight': weight * 100  # Convert to percentage
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(allocation_data)
    
    if df.empty:
        print("❌ No allocation data found")
        return
    
    print(f"✅ Processing {len(df)} allocation records from {df['Year'].min()} to {df['Year'].max()}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Allocation Heatmap by Year
    ax1 = plt.subplot(4, 1, 1)
    
    # Create pivot table for heatmap
    yearly_allocations = df.groupby(['Year', 'Stock'])['Weight'].mean().unstack(fill_value=0)
    
    # Get top 15 most frequently held stocks for better visualization
    stock_frequency = df['Stock'].value_counts()
    top_stocks = stock_frequency.head(15).index.tolist()
    yearly_allocations_top = yearly_allocations[top_stocks]
    
    sns.heatmap(yearly_allocations_top.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Allocation %'}, ax=ax1)
    ax1.set_title('Portfolio Allocation Heatmap by Year (Top 15 Stocks)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Stock Symbol')
    
    # 2. Stacked Area Chart - Portfolio Composition Over Time
    ax2 = plt.subplot(4, 1, 2)
    
    # Prepare data for stacked area chart
    quarterly_data = df.groupby(['Date', 'Stock'])['Weight'].first().unstack(fill_value=0)
    
    # Select top 10 stocks by average allocation
    avg_allocations = quarterly_data.mean().sort_values(ascending=False)
    top_10_stocks = avg_allocations.head(10).index.tolist()
    
    # Create stacked area chart
    quarterly_top10 = quarterly_data[top_10_stocks]
    
    # Use a color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_10_stocks)))
    
    ax2.stackplot(quarterly_top10.index, *[quarterly_top10[stock] for stock in top_10_stocks], 
                 labels=top_10_stocks, colors=colors, alpha=0.8)
    
    ax2.set_title('Portfolio Composition Evolution (Top 10 Holdings)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Allocation %')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Individual Stock Allocation Timeline
    ax3 = plt.subplot(4, 1, 3)
    
    # Show top 8 stocks with individual lines
    top_8_stocks = avg_allocations.head(8).index.tolist()
    
    for i, stock in enumerate(top_8_stocks):
        stock_data = quarterly_data[stock]
        ax3.plot(stock_data.index, stock_data.values, marker='o', linewidth=2, 
                label=stock, markersize=4)
    
    ax3.set_title('Individual Stock Allocation Trends (Top 8 Holdings)', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Allocation %')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Allocation Statistics Table
    ax4 = plt.subplot(4, 1, 4)
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate statistics for top holdings
    stats_data = []
    headers = ['Stock', 'Avg Allocation %', 'Max Allocation %', 'Quarters Held', 'First Appearance', 'Last Appearance']
    
    for stock in top_10_stocks:
        stock_df = df[df['Stock'] == stock]
        avg_alloc = stock_df['Weight'].mean()
        max_alloc = stock_df['Weight'].max()
        quarters_held = len(stock_df)
        first_appear = stock_df['Date'].min().strftime('%Y-Q%m')
        last_appear = stock_df['Date'].max().strftime('%Y-Q%m')
        
        stats_data.append([
            stock,
            f"{avg_alloc:.1f}%",
            f"{max_alloc:.1f}%",
            quarters_held,
            first_appear,
            last_appear
        ])
    
    table = ax4.table(cellText=stats_data, colLabels=headers,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    plt.suptitle('Quarterly Growth Strategy - Portfolio Allocation Timeline (2005-2025)', 
                fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 0.85, 0.96])
    
    # Save the visualization
    plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/portfolio_allocation_timeline.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Saved: portfolio_allocation_timeline.png")
    
    # Create a separate yearly summary chart
    fig2, ax = plt.subplots(figsize=(16, 10))
    
    # Yearly allocation summary - show top 5 stocks per year
    yearly_summary = df.groupby(['Year', 'Stock'])['Weight'].mean().reset_index()
    
    years = sorted(df['Year'].unique())
    x_pos = np.arange(len(years))
    
    # Get top 5 stocks overall
    top_5_overall = df['Stock'].value_counts().head(5).index.tolist()
    colors_5 = plt.cm.Set2(np.linspace(0, 1, 5))
    
    bottom = np.zeros(len(years))
    
    for i, stock in enumerate(top_5_overall):
        yearly_weights = []
        for year in years:
            year_data = yearly_summary[(yearly_summary['Year'] == year) & (yearly_summary['Stock'] == stock)]
            weight = year_data['Weight'].iloc[0] if len(year_data) > 0 else 0
            yearly_weights.append(weight)
        
        ax.bar(x_pos, yearly_weights, bottom=bottom, label=stock, 
               color=colors_5[i], alpha=0.8, width=0.6)
        bottom += yearly_weights
    
    ax.set_title('Top 5 Holdings - Yearly Allocation Breakdown', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Allocation %')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, year in enumerate(years):
        cumulative = 0
        for j, stock in enumerate(top_5_overall):
            year_data = yearly_summary[(yearly_summary['Year'] == year) & (yearly_summary['Stock'] == stock)]
            weight = year_data['Weight'].iloc[0] if len(year_data) > 0 else 0
            if weight > 5:  # Only label if > 5%
                ax.text(i, cumulative + weight/2, f'{weight:.0f}%', 
                       ha='center', va='center', fontweight='bold', fontsize=9)
            cumulative += weight
    
    plt.tight_layout()
    plt.savefig('/Users/nttssv/Documents/efficient_frontier/results/plots/yearly_allocation_breakdown.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Saved: yearly_allocation_breakdown.png")
    
    # Print summary statistics
    print(f"\n📈 ALLOCATION TIMELINE SUMMARY:")
    print("="*50)
    print(f"Total Quarters Analyzed: {len(strategy.quarterly_portfolios)}")
    print(f"Unique Stocks Held: {df['Stock'].nunique()}")
    print(f"Years Covered: {df['Year'].min()} - {df['Year'].max()}")
    
    print(f"\n🏆 TOP 10 MOST HELD STOCKS:")
    print("-"*40)
    for i, (stock, count) in enumerate(stock_frequency.head(10).items(), 1):
        avg_weight = df[df['Stock'] == stock]['Weight'].mean()
        print(f"{i:2d}. {stock:<6} - {count:2d} quarters ({avg_weight:.1f}% avg)")
    
    return {
        'allocation_data': df,
        'top_stocks': top_stocks,
        'yearly_allocations': yearly_allocations
    }

if __name__ == "__main__":
    results = create_allocation_timeline()
