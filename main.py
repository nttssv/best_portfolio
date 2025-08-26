"""
Main execution script for efficient frontier portfolio optimization.
Orchestrates the complete optimization workflow from data fetching to results export.
"""

import sys
import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import MarketDataFetcher
from portfolio_optimizer import EfficientFrontierOptimizer, OptimizationConstraints
from portfolio_metrics import PortfolioMetrics
from visualization import PortfolioVisualizer
from export_results import ResultsExporter

warnings.filterwarnings('ignore')


class EfficientFrontierPipeline:
    """Complete pipeline for efficient frontier portfolio optimization."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the optimization pipeline.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or self._default_config()
        self.results = {}
        
        # Initialize components
        self.data_fetcher = None
        self.optimizer = None
        self.visualizer = PortfolioVisualizer()
        self.exporter = ResultsExporter(self.config['output_dir'])
        
        print("Efficient Frontier Portfolio Optimization Pipeline")
        print("=" * 55)
        print(f"Target Assets: {self.config['num_assets']}")
        print(f"Historical Period: {self.config['period_years']} years")
        print(f"Min Return: {self.config['min_return']:.1%}")
        print(f"Min Sharpe: {self.config['min_sharpe']:.1f}")
        print(f"Max Drawdown: {self.config['max_drawdown']:.1%}")
        print("=" * 55)
    
    def _default_config(self) -> dict:
        """Default configuration parameters."""
        return {
            'num_assets': 100,
            'period_years': 5,
            'min_return': 0.20,
            'min_sharpe': 2.0,
            'max_drawdown': 0.03,
            'risk_free_rate': 0.02,
            'n_frontier_portfolios': 100,
            'output_dir': 'results',
            'save_plots': True,
            'create_interactive_plots': True
        }
    
    def step1_fetch_data(self):
        """Step 1: Fetch and validate market data."""
        print("\n🔄 STEP 1: Fetching Market Data")
        print("-" * 40)
        
        # Initialize data fetcher
        self.data_fetcher = MarketDataFetcher(
            period_years=self.config['period_years']
        )
        
        # Get top assets by volume
        print(f"Identifying top {self.config['num_assets']} assets by trading volume...")
        symbols = self.data_fetcher.get_top_assets_by_volume(
            num_assets=self.config['num_assets']
        )
        
        # Fetch historical data
        print("Downloading historical price data...")
        prices = self.data_fetcher.fetch_historical_data(symbols)
        returns = self.data_fetcher.calculate_returns()
        
        # Validate data quality
        validation = self.data_fetcher.validate_data_quality()
        
        # Store results
        self.results['symbols'] = self.data_fetcher.symbols
        self.results['prices'] = prices
        self.results['returns'] = returns
        self.results['data_summary'] = self.data_fetcher.get_data_summary()
        self.results['data_validation'] = validation
        
        print(f"✅ Data fetching completed")
        print(f"   Assets: {len(self.results['symbols'])}")
        print(f"   Date Range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"   Trading Days: {len(returns)}")
        
        if validation.get('warnings'):
            print("⚠️  Data Quality Warnings:")
            for warning in validation['warnings'][:3]:  # Show first 3 warnings
                print(f"   - {warning}")
    
    def step2_optimize_portfolios(self):
        """Step 2: Run portfolio optimization."""
        print("\n🔄 STEP 2: Portfolio Optimization")
        print("-" * 40)
        
        if 'returns' not in self.results:
            raise ValueError("No returns data available. Run step1_fetch_data first.")
        
        # Initialize optimizer
        self.optimizer = EfficientFrontierOptimizer(
            returns=self.results['returns'],
            risk_free_rate=self.config['risk_free_rate']
        )
        
        # Set optimization constraints
        constraints = OptimizationConstraints(
            min_return=self.config['min_return'],
            min_sharpe=self.config['min_sharpe'],
            max_drawdown=self.config['max_drawdown']
        )
        
        # Find optimal portfolios
        print("Finding optimal portfolios...")
        optimal_portfolios = self.optimizer.find_optimal_portfolios(constraints)
        
        # Generate efficient frontier
        print(f"Generating efficient frontier with {self.config['n_frontier_portfolios']} portfolios...")
        frontier_df = self.optimizer.generate_efficient_frontier(
            n_portfolios=self.config['n_frontier_portfolios'],
            constraints=constraints
        )
        
        # Store results
        self.results['optimal_portfolios'] = optimal_portfolios
        self.results['frontier_df'] = frontier_df
        self.results['constraints'] = constraints
        
        # Summary statistics
        feasible_count = len(self.optimizer.feasible_portfolios)
        total_count = len(self.optimizer.efficient_portfolios)
        
        print(f"✅ Optimization completed")
        print(f"   Efficient portfolios: {total_count}")
        print(f"   Feasible portfolios: {feasible_count}")
        print(f"   Feasibility rate: {feasible_count/max(total_count,1):.1%}")
        
        # Show optimal portfolio results
        for name, portfolio in optimal_portfolios.items():
            if portfolio['weights'] is not None:
                metrics = portfolio['metrics']
                constraints_met = portfolio['constraints_met']['all_constraints']
                print(f"   {name.replace('_', ' ').title()}: "
                      f"Return={metrics['annualized_return']:.1%}, "
                      f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                      f"Constraints={'✅' if constraints_met else '❌'}")
    
    def step3_create_visualizations(self):
        """Step 3: Create visualizations."""
        print("\n🔄 STEP 3: Creating Visualizations")
        print("-" * 40)
        
        if 'frontier_df' not in self.results:
            raise ValueError("No optimization results available. Run step2_optimize_portfolios first.")
        
        # Create output directory for plots
        plots_dir = os.path.join(self.config['output_dir'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Efficient Frontier Plot
        print("Creating efficient frontier visualization...")
        if self.config['save_plots']:
            save_path = os.path.join(plots_dir, 'efficient_frontier.png')
        else:
            save_path = None
        
        self.visualizer.plot_efficient_frontier(
            self.results['frontier_df'],
            self.results['optimal_portfolios'],
            save_path=save_path,
            interactive=self.config['create_interactive_plots']
        )
        
        # 2. Portfolio Composition for best portfolio
        best_portfolio = self._get_best_portfolio()
        if best_portfolio:
            print("Creating portfolio composition chart...")
            if self.config['save_plots']:
                save_path = os.path.join(plots_dir, 'portfolio_composition.png')
            else:
                save_path = None
            
            self.visualizer.plot_portfolio_composition(
                best_portfolio['weights'],
                self.results['symbols'],
                title="Optimal Portfolio Composition",
                save_path=save_path
            )
        
        # 3. Performance Comparison
        print("Creating performance comparison chart...")
        if self.config['save_plots']:
            save_path = os.path.join(plots_dir, 'performance_comparison.png')
        else:
            save_path = None
        
        self.visualizer.plot_performance_comparison(
            self.results['optimal_portfolios'],
            save_path=save_path
        )
        
        # 4. Returns Distribution Analysis
        if best_portfolio:
            print("Creating returns distribution analysis...")
            portfolio_returns = self.results['returns'].dot(best_portfolio['weights'])
            market_returns = self.results['returns'].mean(axis=1)  # Equal-weighted market
            
            returns_data = {
                'optimal_portfolio': portfolio_returns,
                'equal_weighted_market': market_returns
            }
            
            if self.config['save_plots']:
                save_path = os.path.join(plots_dir, 'returns_distribution.png')
            else:
                save_path = None
            
            self.visualizer.plot_returns_distribution(
                returns_data,
                save_path=save_path
            )
        
        print("✅ Visualizations completed")
    
    def step4_export_results(self):
        """Step 4: Export results to files."""
        print("\n🔄 STEP 4: Exporting Results")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Export efficient frontier data
        print("Exporting efficient frontier data...")
        frontier_file = self.exporter.export_efficient_frontier(
            self.results['frontier_df'],
            filename=f"efficient_frontier_{timestamp}.csv"
        )
        
        # 2. Export optimal portfolios
        print("Exporting optimal portfolio weights and metrics...")
        portfolios_file = self.exporter.export_optimal_portfolios(
            self.results['optimal_portfolios'],
            self.results['symbols'],
            filename=f"optimal_portfolios_{timestamp}.xlsx"
        )
        
        # 3. Export detailed composition for best portfolio
        best_portfolio = self._get_best_portfolio()
        if best_portfolio:
            print("Exporting detailed portfolio composition...")
            composition_file = self.exporter.export_portfolio_composition(
                best_portfolio['weights'],
                self.results['symbols'],
                portfolio_name="Optimal Portfolio",
                filename=f"portfolio_composition_{timestamp}.csv"
            )
        
        # 4. Create comprehensive summary report
        print("Creating summary report...")
        summary_file = self.exporter.create_summary_report(
            self.results,
            filename=f"optimization_summary_{timestamp}.txt"
        )
        
        # 5. Backtest best portfolio
        if best_portfolio:
            print("Running backtest analysis...")
            backtest_results = self.optimizer.backtest_portfolio(best_portfolio['weights'])
            backtest_file = self.exporter.export_backtest_results(
                backtest_results,
                filename=f"backtest_results_{timestamp}.csv"
            )
        
        print("✅ Results export completed")
        print(f"   All files saved to: {self.config['output_dir']}/")
    
    def _get_best_portfolio(self):
        """Get the best portfolio based on constraints satisfaction and Sharpe ratio."""
        best_portfolio = None
        best_score = -np.inf
        
        for portfolio in self.results['optimal_portfolios'].values():
            if portfolio['weights'] is not None:
                # Prioritize constraint satisfaction, then Sharpe ratio
                constraints_met = portfolio['constraints_met']['all_constraints']
                sharpe_ratio = portfolio['metrics']['sharpe_ratio']
                
                score = (10 if constraints_met else 0) + sharpe_ratio
                
                if score > best_score:
                    best_score = score
                    best_portfolio = portfolio
        
        return best_portfolio
    
    def run_complete_analysis(self):
        """Run the complete optimization pipeline."""
        try:
            start_time = datetime.now()
            
            # Execute all steps
            self.step1_fetch_data()
            self.step2_optimize_portfolios()
            self.step3_create_visualizations()
            self.step4_export_results()
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 55)
            print(f"Total Runtime: {duration}")
            print(f"Results Directory: {self.config['output_dir']}/")
            
            # Key findings summary
            feasible_portfolios = [p for p in self.results['optimal_portfolios'].values() 
                                 if p['weights'] is not None and p['constraints_met']['all_constraints']]
            
            if feasible_portfolios:
                print(f"✅ Found {len(feasible_portfolios)} portfolio(s) meeting all constraints")
                best_portfolio = self._get_best_portfolio()
                if best_portfolio:
                    metrics = best_portfolio['metrics']
                    print(f"🏆 Best Portfolio Performance:")
                    print(f"   • Return: {metrics['annualized_return']:.2%}")
                    print(f"   • Volatility: {metrics['annualized_volatility']:.2%}")
                    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                    print(f"   • Max Drawdown: {metrics['max_drawdown']:.2%}")
            else:
                print("⚠️  No portfolios found meeting all constraints")
                print("   Consider relaxing constraints or expanding asset universe")
            
            print("\n📊 Next Steps:")
            print("   1. Review the optimization_summary.txt for detailed analysis")
            print("   2. Examine optimal_portfolios.xlsx for portfolio weights")
            print("   3. Check plots/ directory for visualizations")
            print("   4. Consider implementing the recommended portfolio")
            
        except Exception as e:
            print(f"\n❌ ERROR: Analysis failed with exception: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main execution function."""
    # Configuration
    config = {
        'num_assets': 100,
        'period_years': 5,
        'min_return': 0.20,
        'min_sharpe': 2.0,
        'max_drawdown': 0.03,
        'risk_free_rate': 0.02,
        'n_frontier_portfolios': 100,
        'output_dir': 'results',
        'save_plots': True,
        'create_interactive_plots': True
    }
    
    # Create and run pipeline
    pipeline = EfficientFrontierPipeline(config)
    pipeline.run_complete_analysis()


if __name__ == "__main__":
    main()
