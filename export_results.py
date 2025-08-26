"""
Export module for portfolio optimization results.
Handles CSV/Excel export of portfolios, metrics, and analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ResultsExporter:
    """Export portfolio optimization results to various formats."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def export_efficient_frontier(self, frontier_df: pd.DataFrame, 
                                 filename: Optional[str] = None) -> str:
        """
        Export efficient frontier data to CSV.
        
        Args:
            frontier_df: Efficient frontier DataFrame
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"efficient_frontier_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data for export
        export_df = frontier_df.copy()
        
        # Format percentage columns
        pct_columns = ['annualized_return', 'annualized_volatility', 'max_drawdown', 'var_5', 'cvar_5']
        for col in pct_columns:
            if col in export_df.columns:
                export_df[f'{col}_pct'] = export_df[col] * 100
        
        # Round numerical columns
        numerical_cols = export_df.select_dtypes(include=[np.number]).columns
        export_df[numerical_cols] = export_df[numerical_cols].round(6)
        
        # Export to CSV
        export_df.to_csv(filepath, index=False)
        print(f"Efficient frontier data exported to: {filepath}")
        
        return filepath
    
    def export_optimal_portfolios(self, optimal_portfolios: Dict, 
                                 symbols: List[str],
                                 filename: Optional[str] = None) -> str:
        """
        Export optimal portfolio weights and metrics.
        
        Args:
            optimal_portfolios: Dictionary of optimal portfolios
            symbols: Asset symbols
            filename: Custom filename (optional)
            
        Returns:
            Path to exported Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimal_portfolios_{timestamp}.xlsx"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet with all portfolio metrics
            summary_data = []
            
            for name, portfolio in optimal_portfolios.items():
                if portfolio['weights'] is not None:
                    metrics = portfolio['metrics']
                    constraints = portfolio['constraints_met']
                    
                    summary_data.append({
                        'Portfolio': name.replace('_', ' ').title(),
                        'Annualized Return (%)': metrics['annualized_return'] * 100,
                        'Annualized Volatility (%)': metrics['annualized_volatility'] * 100,
                        'Sharpe Ratio': metrics['sharpe_ratio'],
                        'Sortino Ratio': metrics['sortino_ratio'],
                        'Calmar Ratio': metrics['calmar_ratio'],
                        'Maximum Drawdown (%)': metrics['max_drawdown'] * 100,
                        'VaR 5% (%)': metrics['var_5'] * 100,
                        'CVaR 5% (%)': metrics['cvar_5'] * 100,
                        'Skewness': metrics['skewness'],
                        'Kurtosis': metrics['kurtosis'],
                        'Return Constraint Met': constraints['return_constraint'],
                        'Sharpe Constraint Met': constraints['sharpe_constraint'],
                        'Drawdown Constraint Met': constraints['drawdown_constraint'],
                        'All Constraints Met': constraints['all_constraints']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
            
            # Individual portfolio weights sheets
            for name, portfolio in optimal_portfolios.items():
                if portfolio['weights'] is not None:
                    weights = portfolio['weights']
                    
                    # Create weights DataFrame
                    weights_df = pd.DataFrame({
                        'Symbol': symbols,
                        'Weight': weights,
                        'Weight (%)': weights * 100
                    })
                    
                    # Filter out zero weights and sort by weight
                    weights_df = weights_df[weights_df['Weight'] > 0.0001]
                    weights_df = weights_df.sort_values('Weight', ascending=False)
                    
                    # Add summary statistics
                    summary_stats = pd.DataFrame({
                        'Metric': ['Number of Assets', 'Largest Weight (%)', 'Smallest Weight (%)', 
                                  'Weight Concentration (Top 5)', 'Effective Number of Assets'],
                        'Value': [
                            len(weights_df),
                            weights_df['Weight (%)'].max(),
                            weights_df['Weight (%)'].min(),
                            weights_df.head(5)['Weight (%)'].sum(),
                            1 / np.sum(weights**2)  # Inverse of Herfindahl index
                        ]
                    })
                    
                    # Export weights
                    sheet_name = f"{name[:25]}_Weights"  # Excel sheet name limit
                    weights_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add summary stats below weights
                    start_row = len(weights_df) + 3
                    summary_stats.to_excel(writer, sheet_name=sheet_name, 
                                         startrow=start_row, index=False)
        
        print(f"Optimal portfolios exported to: {filepath}")
        return filepath
    
    def export_portfolio_composition(self, weights: np.ndarray, symbols: List[str],
                                   portfolio_name: str = "Portfolio",
                                   filename: Optional[str] = None) -> str:
        """
        Export detailed portfolio composition.
        
        Args:
            weights: Portfolio weights
            symbols: Asset symbols
            portfolio_name: Name of the portfolio
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_composition_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create composition DataFrame
        composition_df = pd.DataFrame({
            'Symbol': symbols,
            'Weight': weights,
            'Weight_Percent': weights * 100
        })
        
        # Add sector classification (simplified example)
        composition_df['Sector'] = self._classify_sectors(symbols)
        
        # Filter significant positions
        composition_df = composition_df[composition_df['Weight'] > 0.0001]
        composition_df = composition_df.sort_values('Weight', ascending=False)
        
        # Add cumulative weights
        composition_df['Cumulative_Weight'] = composition_df['Weight'].cumsum()
        composition_df['Cumulative_Weight_Percent'] = composition_df['Cumulative_Weight'] * 100
        
        # Add portfolio statistics
        stats_df = pd.DataFrame({
            'Metric': [
                'Total Assets',
                'Effective Number of Assets',
                'Largest Position (%)',
                'Top 5 Concentration (%)',
                'Top 10 Concentration (%)',
                'Herfindahl Index'
            ],
            'Value': [
                len(composition_df),
                1 / np.sum(weights**2),
                composition_df['Weight_Percent'].iloc[0] if len(composition_df) > 0 else 0,
                composition_df.head(5)['Weight_Percent'].sum(),
                composition_df.head(10)['Weight_Percent'].sum(),
                np.sum(weights**2)
            ]
        })
        
        # Export with both composition and statistics
        with open(filepath, 'w') as f:
            f.write(f"Portfolio Composition Report - {portfolio_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PORTFOLIO STATISTICS\n")
            f.write("=" * 50 + "\n")
            stats_df.to_csv(f, index=False)
            
            f.write(f"\n\nPORTFOLIO HOLDINGS\n")
            f.write("=" * 50 + "\n")
            composition_df.to_csv(f, index=False)
        
        print(f"Portfolio composition exported to: {filepath}")
        return filepath
    
    def _classify_sectors(self, symbols: List[str]) -> List[str]:
        """
        Classify symbols into sectors (simplified mapping).
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            List of sector classifications
        """
        # Simplified sector mapping
        sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
            'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
            'NFLX': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
            'CRM': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology',
            
            # Financial
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'GS': 'Financial', 'MS': 'Financial', 'C': 'Financial',
            'V': 'Financial', 'MA': 'Financial', 'AXP': 'Financial',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            
            # Consumer
            'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            'HD': 'Consumer', 'MCD': 'Consumer', 'NKE': 'Consumer',
            
            # ETFs
            'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'VTI': 'ETF',
            'ARKK': 'ETF', 'XLF': 'ETF', 'XLK': 'ETF', 'XLE': 'ETF'
        }
        
        return [sector_mapping.get(symbol, 'Other') for symbol in symbols]
    
    def export_backtest_results(self, backtest_data: Dict, 
                               filename: Optional[str] = None) -> str:
        """
        Export backtest results.
        
        Args:
            backtest_data: Dictionary with backtest results
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Extract time series data
        portfolio_returns = backtest_data['portfolio_returns']
        cumulative_returns = backtest_data['cumulative_returns']
        
        # Create time series DataFrame
        results_df = pd.DataFrame({
            'Date': portfolio_returns.index,
            'Daily_Return': portfolio_returns.values,
            'Daily_Return_Percent': portfolio_returns.values * 100,
            'Cumulative_Return': cumulative_returns.values,
            'Cumulative_Return_Percent': (cumulative_returns.values - 1) * 100
        })
        
        # Add rolling metrics
        results_df['Rolling_Volatility_30D'] = portfolio_returns.rolling(30).std() * np.sqrt(252)
        results_df['Rolling_Sharpe_30D'] = (
            portfolio_returns.rolling(30).mean() * 252 - 0.02
        ) / (portfolio_returns.rolling(30).std() * np.sqrt(252))
        
        # Calculate drawdowns
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        results_df['Drawdown'] = drawdown.values
        results_df['Drawdown_Percent'] = drawdown.values * 100
        
        # Export
        results_df.to_csv(filepath, index=False)
        print(f"Backtest results exported to: {filepath}")
        
        return filepath
    
    def create_summary_report(self, optimization_results: Dict,
                             filename: Optional[str] = None) -> str:
        """
        Create comprehensive summary report.
        
        Args:
            optimization_results: Complete optimization results
            filename: Custom filename (optional)
            
        Returns:
            Path to exported report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_summary_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("EFFICIENT FRONTIER PORTFOLIO OPTIMIZATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            if 'data_summary' in optimization_results:
                data_summary = optimization_results['data_summary']
                f.write("DATA SUMMARY\n")
                f.write("-" * 30 + "\n")
                f.write(f"Number of Assets: {data_summary.get('num_assets', 'N/A')}\n")
                f.write(f"Date Range: {data_summary.get('date_range', 'N/A')}\n")
                f.write(f"Trading Days: {data_summary.get('trading_days', 'N/A')}\n")
                f.write(f"Data Completeness: {data_summary.get('data_completeness', 0):.2%}\n\n")
            
            # Optimization constraints
            f.write("OPTIMIZATION CONSTRAINTS\n")
            f.write("-" * 30 + "\n")
            f.write("• Minimum Annualized Return: 20%\n")
            f.write("• Minimum Sharpe Ratio: 2.0\n")
            f.write("• Maximum Drawdown: 3%\n\n")
            
            # Results summary
            if 'optimal_portfolios' in optimization_results:
                optimal_portfolios = optimization_results['optimal_portfolios']
                f.write("OPTIMAL PORTFOLIOS FOUND\n")
                f.write("-" * 30 + "\n")
                
                for name, portfolio in optimal_portfolios.items():
                    if portfolio['weights'] is not None:
                        metrics = portfolio['metrics']
                        constraints = portfolio['constraints_met']
                        
                        f.write(f"\n{name.upper().replace('_', ' ')} PORTFOLIO:\n")
                        f.write(f"  Return: {metrics['annualized_return']:.2%}\n")
                        f.write(f"  Volatility: {metrics['annualized_volatility']:.2%}\n")
                        f.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n")
                        f.write(f"  Max Drawdown: {metrics['max_drawdown']:.2%}\n")
                        f.write(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}\n")
                        f.write(f"  Constraints Met: {'✓' if constraints['all_constraints'] else '✗'}\n")
                        
                        # Top holdings
                        weights = portfolio['weights']
                        top_indices = np.argsort(weights)[-5:][::-1]
                        f.write("  Top 5 Holdings:\n")
                        for i, idx in enumerate(top_indices):
                            if weights[idx] > 0.001:
                                symbol = optimization_results.get('symbols', [f'Asset_{idx}'])[idx]
                                f.write(f"    {i+1}. {symbol}: {weights[idx]:.1%}\n")
            
            # Feasibility analysis
            if 'frontier_df' in optimization_results:
                frontier_df = optimization_results['frontier_df']
                if not frontier_df.empty:
                    feasible_count = frontier_df['constraints_met'].sum()
                    total_count = len(frontier_df)
                    
                    f.write(f"\nFEASIBILITY ANALYSIS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Total Portfolios Generated: {total_count}\n")
                    f.write(f"Feasible Portfolios: {feasible_count}\n")
                    f.write(f"Feasibility Rate: {feasible_count/total_count:.1%}\n")
                    
                    if feasible_count > 0:
                        feasible = frontier_df[frontier_df['constraints_met'] == True]
                        f.write(f"\nFeasible Portfolio Ranges:\n")
                        f.write(f"  Return: {feasible['annualized_return'].min():.2%} - {feasible['annualized_return'].max():.2%}\n")
                        f.write(f"  Volatility: {feasible['annualized_volatility'].min():.2%} - {feasible['annualized_volatility'].max():.2%}\n")
                        f.write(f"  Sharpe: {feasible['sharpe_ratio'].min():.3f} - {feasible['sharpe_ratio'].max():.3f}\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            feasible_found = False
            if 'optimal_portfolios' in optimization_results:
                for portfolio in optimization_results['optimal_portfolios'].values():
                    if (portfolio['weights'] is not None and 
                        portfolio['constraints_met']['all_constraints']):
                        feasible_found = True
                        break
            
            if feasible_found:
                f.write("✓ Feasible portfolios meeting all constraints were found.\n")
                f.write("✓ Consider the Max Sharpe portfolio for optimal risk-adjusted returns.\n")
                f.write("✓ Review portfolio composition for concentration risk.\n")
                f.write("✓ Implement regular rebalancing to maintain target weights.\n")
            else:
                f.write("⚠ No portfolios meeting all constraints were found.\n")
                f.write("⚠ Consider relaxing constraints:\n")
                f.write("  - Reduce minimum return requirement\n")
                f.write("  - Lower minimum Sharpe ratio\n")
                f.write("  - Increase maximum drawdown tolerance\n")
                f.write("⚠ Expand asset universe for better diversification.\n")
        
        print(f"Summary report exported to: {filepath}")
        return filepath


def main():
    """Example usage of ResultsExporter."""
    # Create sample data
    exporter = ResultsExporter()
    
    # Sample efficient frontier data
    np.random.seed(42)
    n_portfolios = 50
    frontier_df = pd.DataFrame({
        'annualized_return': np.random.uniform(0.05, 0.25, n_portfolios),
        'annualized_volatility': np.random.uniform(0.10, 0.30, n_portfolios),
        'sharpe_ratio': np.random.uniform(0.5, 3.0, n_portfolios),
        'max_drawdown': np.random.uniform(0.02, 0.15, n_portfolios),
        'constraints_met': np.random.choice([True, False], n_portfolios, p=[0.3, 0.7])
    })
    
    # Export frontier
    exporter.export_efficient_frontier(frontier_df)
    
    print("Example export completed!")


if __name__ == "__main__":
    main()
