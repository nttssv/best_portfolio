#!/usr/bin/env python3
"""
Drawdown Analysis for Quarterly Growth Strategy
Analyzes and visualizes maximum drawdown periods and recovery times
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from quarterly_growth_strategy import QuarterlyGrowthStrategy
from datetime import datetime
import seaborn as sns

def analyze_drawdowns():
    """Analyze drawdown periods and create comprehensive visualization"""
    
    print("📉 DRAWDOWN ANALYSIS FOR QUARTERLY GROWTH STRATEGY")
    print("="*60)
    
    # Run strategy to get portfolio history
    strategy = QuarterlyGrowthStrategy(universe_size=96, top_growth=5)
    results = strategy.backtest_strategy(start_date='2005-01-01', end_date='2025-01-01')
    
    if not results or not results['portfolio_history']:
        print("❌ No portfolio history available")
        return
    
    portfolio_history = results['portfolio_history']
    dates = [pd.to_datetime(entry['date']) for entry in portfolio_history]
    values = np.array([entry['value'] for entry in portfolio_history])
    
    # Calculate cumulative returns and drawdowns
    cumulative = values
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    
    # Find maximum drawdown details
    max_dd_idx = np.argmin(drawdown)
    max_dd_value = drawdown[max_dd_idx]
    max_dd_date = dates[max_dd_idx]
    
    # Find peak before max drawdown
    peak_idx = np.argmax(rolling_max[:max_dd_idx+1])
    peak_date = dates[peak_idx]
    peak_value = values[peak_idx]
    
    # Find recovery point
    recovery_idx = None
    for i in range(max_dd_idx, len(values)):
        if values[i] >= peak_value:
            recovery_idx = i
            break
    
    recovery_date = dates[recovery_idx] if recovery_idx else None
    
    # Calculate duration
    if recovery_date:
        duration_days = (recovery_date - peak_date).days
        duration_months = duration_days / 30.44
    else:
        duration_days = (dates[-1] - peak_date).days
        duration_months = duration_days / 30.44
        recovery_date = 'Not recovered'
    
    # Print summary
    print(f"📊 MAXIMUM DRAWDOWN SUMMARY:")
    print(f"Maximum Drawdown: {max_dd_value:.2%}")
    print(f"Peak Date: {peak_date.strftime('%Y-%m-%d')}")
    print(f"Peak Value: ${peak_value:,.0f}")
    print(f"Trough Date: {max_dd_date.strftime('%Y-%m-%d')}")
    print(f"Trough Value: ${values[max_dd_idx]:,.0f}")
    if recovery_date != 'Not recovered':
        print(f"Recovery Date: {recovery_date.strftime('%Y-%m-%d')}")
        print(f"Duration: {duration_days} days ({duration_months:.1f} months)")
    else:
        print(f"Recovery: {recovery_date}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Portfolio Value with Drawdown Periods
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(dates, values, linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.plot(dates, rolling_max, linewidth=1, color='red', linestyle='--', alpha=0.7, label='Peak Value')
    
    # Highlight maximum drawdown period
    if recovery_date != 'Not recovered':
        mask = (np.array(dates) >= peak_date) & (np.array(dates) <= recovery_date)
        ax1.fill_between(dates, 0, values, where=mask, alpha=0.3, color='red', 
                        label=f'Max DD Period ({duration_months:.1f} months)')
    
    ax1.set_title('Portfolio Value with Maximum Drawdown Period', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Chart
    ax2 = plt.subplot(3, 2, 2)
    ax2.fill_between(dates, drawdown * 100, 0, alpha=0.7, color='red', label='Drawdown %')
    ax2.axhline(y=max_dd_value * 100, color='darkred', linestyle='--', 
               label=f'Max DD: {max_dd_value:.1%}')
    ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown Duration Analysis
    ax3 = plt.subplot(3, 2, 3)
    
    # Find all significant drawdown periods (>10%)
    significant_dds = []
    in_drawdown = False
    dd_start = None
    
    for i, dd in enumerate(drawdown):
        if dd < -0.10 and not in_drawdown:
            in_drawdown = True
            dd_start = i
        elif dd >= -0.05 and in_drawdown:
            in_drawdown = False
            if dd_start is not None:
                significant_dds.append({
                    'start_date': dates[dd_start],
                    'end_date': dates[i],
                    'duration_days': (dates[i] - dates[dd_start]).days,
                    'max_dd': np.min(drawdown[dd_start:i+1]) * 100
                })
    
    # Plot duration vs magnitude
    if significant_dds:
        durations = [dd['duration_days'] for dd in significant_dds]
        magnitudes = [abs(dd['max_dd']) for dd in significant_dds]
        
        scatter = ax3.scatter(durations, magnitudes, s=100, alpha=0.7, c=magnitudes, 
                             cmap='Reds', edgecolors='black')
        
        # Highlight maximum drawdown
        max_dd_duration = duration_days if recovery_date != 'Not recovered' else duration_days
        ax3.scatter([max_dd_duration], [abs(max_dd_value * 100)], s=200, color='darkred', 
                   marker='*', label='Maximum Drawdown', edgecolors='black', linewidth=2)
        
        ax3.set_title('Drawdown Duration vs Magnitude', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Duration (Days)')
        ax3.set_ylabel('Maximum Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Drawdown Magnitude (%)')
    
    # 4. Recovery Time Analysis
    ax4 = plt.subplot(3, 2, 4)
    
    # Calculate recovery times for significant drawdowns
    recovery_times = []
    for dd in significant_dds:
        recovery_times.append(dd['duration_days'] / 30.44)  # Convert to months
    
    if recovery_times:
        ax4.hist(recovery_times, bins=15, alpha=0.7, color='#F18F01', edgecolor='black')
        ax4.axvline(x=duration_months, color='red', linestyle='--', linewidth=2,
                   label=f'Max DD Recovery: {duration_months:.1f} months')
        ax4.set_title('Distribution of Recovery Times', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Recovery Time (Months)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Underwater Chart (Time Below Peak)
    ax5 = plt.subplot(3, 1, 3)
    underwater = (values / rolling_max - 1) * 100
    ax5.fill_between(dates, underwater, 0, alpha=0.7, color='lightcoral', label='Underwater %')
    ax5.axhline(y=max_dd_value * 100, color='darkred', linestyle='--', 
               label=f'Max Drawdown: {max_dd_value:.1%}')
    ax5.set_title('Underwater Chart - Portfolio Below Peak Value', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Underwater (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis
    ax5.xaxis.set_major_locator(mdates.YearLocator(2))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig('results/plots/drawdown_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: drawdown_analysis.png")
    
    # Return summary data
    return {
        'max_drawdown': max_dd_value,
        'max_dd_duration_days': duration_days,
        'max_dd_duration_months': duration_months,
        'peak_date': peak_date,
        'trough_date': max_dd_date,
        'recovery_date': recovery_date,
        'significant_drawdowns': len(significant_dds),
        'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0
    }

if __name__ == "__main__":
    summary = analyze_drawdowns()
    
    print(f"\n📈 DRAWDOWN SUMMARY STATISTICS:")
    print("="*40)
    print(f"Maximum Drawdown: {summary['max_drawdown']:.2%}")
    print(f"Duration: {summary['max_dd_duration_days']} days ({summary['max_dd_duration_months']:.1f} months)")
    print(f"Significant Drawdowns (>10%): {summary['significant_drawdowns']}")
    print(f"Average Recovery Time: {summary['avg_recovery_time']:.1f} months")
