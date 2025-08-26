"""
Main execution script for extended 20-year portfolio analysis.
Combines original efficient frontier analysis with comprehensive 20-year performance evaluation.
"""

import sys
import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extended_analysis import ExtendedPortfolioAnalysis
from export_results import ResultsExporter

warnings.filterwarnings('ignore')


def run_extended_portfolio_analysis():
    """Run complete 20-year extended portfolio analysis."""
    
    print("🚀 EXTENDED 20-YEAR PORTFOLIO ANALYSIS")
    print("="*70)
    print("Features:")
    print("• 20 years of historical data")
    print("• Monthly and annual returns analysis")
    print("• Portfolio growth simulation ($100,000 initial)")
    print("• Comprehensive performance metrics")
    print("• Interactive visualizations")
    print("="*70)
    
    try:
        start_time = datetime.now()
        
        # Initialize extended analysis
        analysis = ExtendedPortfolioAnalysis(initial_investment=100000)
        
        # Run complete analysis
        results = analysis.run_complete_extended_analysis(
            num_assets=50,  # Analyze top 50 assets
            save_plots=True,
            plots_dir="results/plots"
        )
        
        # Export additional results
        print("\n💾 Exporting Extended Analysis Results...")
        exporter = ResultsExporter("results")
        
        # Export performance summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"results/performance_summary_20yr_{timestamp}.csv"
        results['summary_table'].to_csv(summary_file, index=False)
        print(f"   Performance summary exported to: {summary_file}")
        
        # Export portfolio composition
        if results['portfolio_weights'] is not None:
            composition_file = exporter.export_portfolio_composition(
                results['portfolio_weights'],
                results['symbols'],
                portfolio_name="20-Year Optimized Portfolio",
                filename=f"portfolio_composition_20yr_{timestamp}.csv"
            )
        
        # Create comprehensive report
        extended_results = {
            'data_summary': {
                'num_assets': len(results['symbols']),
                'date_range': results['data_period'],
                'trading_days': len(analysis.portfolio_returns),
                'data_completeness': 1.0
            },
            'optimal_portfolios': results['optimal_portfolios'],
            'frontier_df': pd.DataFrame(),  # Not applicable for extended analysis
            'symbols': results['symbols'],
            'performance_data': results['performance_data']
        }
        
        summary_report = exporter.create_summary_report(
            extended_results,
            filename=f"extended_analysis_summary_{timestamp}.txt"
        )
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n🎉 EXTENDED ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Total Runtime: {duration}")
        print(f"Analysis Period: {results['data_period'][0].date()} to {results['data_period'][1].date()}")
        print(f"Assets Analyzed: {len(results['symbols'])}")
        print(f"Initial Investment: ${analysis.initial_investment:,.0f}")
        print(f"Final Portfolio Value: ${results['performance_data']['final_value']:,.0f}")
        print(f"Total Return: {results['performance_data']['total_return']:.2%}")
        print(f"CAGR: {results['performance_data']['cagr']:.2%}")
        print(f"Sharpe Ratio: {results['performance_data']['metrics']['sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {results['performance_data']['metrics']['max_drawdown']:.2%}")
        
        print(f"\n📁 Results Directory: results/")
        print("📊 Generated Visualizations:")
        print("   • monthly_returns_20yr.html - Monthly returns time series")
        print("   • annual_returns_20yr.html - Annual returns bar chart")
        print("   • portfolio_growth_20yr.html - Portfolio value growth curve")
        
        print("\n📈 Key Insights:")
        performance = results['performance_data']
        if performance['cagr'] > 0.10:
            print("   ✅ Strong long-term performance with >10% CAGR")
        elif performance['cagr'] > 0.07:
            print("   ✅ Solid long-term performance with >7% CAGR")
        else:
            print("   ⚠️  Modest long-term performance")
            
        if performance['metrics']['sharpe_ratio'] > 1.0:
            print("   ✅ Good risk-adjusted returns (Sharpe > 1.0)")
        else:
            print("   ⚠️  Risk-adjusted returns could be improved")
            
        if performance['metrics']['max_drawdown'] < 0.20:
            print("   ✅ Reasonable drawdown control (<20%)")
        else:
            print("   ⚠️  Significant drawdowns experienced (>20%)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: Extended analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main execution function."""
    results = run_extended_portfolio_analysis()
    
    print("\n🔍 Next Steps:")
    print("1. Review the interactive HTML charts in results/plots/")
    print("2. Examine the performance summary CSV for detailed metrics")
    print("3. Consider the portfolio composition for implementation")
    print("4. Use insights for portfolio strategy refinement")


if __name__ == "__main__":
    main()
