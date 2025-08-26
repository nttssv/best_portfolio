"""
Large-scale portfolio analysis with full asset universe and 1000+ portfolio combinations.
Enhanced version for comprehensive portfolio weight analysis across 96 assets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from portfolio_optimizer import EfficientFrontierOptimizer, OptimizationConstraints
from portfolio_metrics import PortfolioMetrics
from extended_analysis import ExtendedPortfolioAnalysis


class LargeScalePortfolioAnalysis:
    """Large-scale portfolio analysis with full asset universe and extensive portfolio generation."""
    
    def __init__(self, initial_investment: float = 100000.0):
        """
        Initialize large-scale analysis.
        
        Args:
            initial_investment: Initial portfolio value in dollars
        """
        self.initial_investment = initial_investment
        self.data_fetcher = None
        self.returns_data = None
        self.prices_data = None
        self.metrics_calc = PortfolioMetrics()
        self.all_portfolios = []
        self.portfolio_results = []
        
    def fetch_full_universe_data(self, period_years: int = 20) -> pd.DataFrame:
        """
        Fetch data for the full asset universe (96 assets).
        
        Args:
            period_years: Years of historical data
            
        Returns:
            Returns DataFrame
        """
        print(f"🌐 Fetching {period_years} years of data for FULL asset universe...")
        
        # Initialize data fetcher
        self.data_fetcher = MarketDataFetcher(period_years=period_years)
        
        # Get all available assets (full universe)
        symbols = self.data_fetcher.get_top_assets_by_volume(num_assets=100)  # Get max available
        
        # Fetch historical data
        self.prices_data = self.data_fetcher.fetch_historical_data(symbols)
        self.returns_data = self.data_fetcher.calculate_returns()
        
        print(f"✅ Full universe data fetching completed:")
        print(f"   Total Assets: {len(self.returns_data.columns)}")
        print(f"   Date Range: {self.prices_data.index[0].date()} to {self.prices_data.index[-1].date()}")
        print(f"   Trading Days: {len(self.returns_data)}")
        
        return self.returns_data
    
    def generate_random_portfolios(self, n_portfolios: int = 1000, 
                                 max_assets_per_portfolio: int = 30,
                                 min_weight: float = 0.01,
                                 max_weight: float = 0.15) -> List[np.ndarray]:
        """
        Generate random portfolio weight combinations.
        
        Args:
            n_portfolios: Number of portfolios to generate
            max_assets_per_portfolio: Maximum number of assets per portfolio
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            
        Returns:
            List of portfolio weight arrays
        """
        print(f"🎲 Generating {n_portfolios} random portfolio combinations...")
        
        portfolios = []
        n_assets = len(self.returns_data.columns)
        
        for i in tqdm(range(n_portfolios), desc="Generating portfolios"):
            # Randomly select number of assets for this portfolio
            n_selected = np.random.randint(5, min(max_assets_per_portfolio + 1, n_assets + 1))
            
            # Randomly select which assets to include
            selected_indices = np.random.choice(n_assets, size=n_selected, replace=False)
            
            # Generate random weights for selected assets
            raw_weights = np.random.exponential(scale=1.0, size=n_selected)
            
            # Normalize weights to sum to 1
            normalized_weights = raw_weights / raw_weights.sum()
            
            # Apply weight constraints
            normalized_weights = np.clip(normalized_weights, min_weight, max_weight)
            normalized_weights = normalized_weights / normalized_weights.sum()
            
            # Create full weight vector
            full_weights = np.zeros(n_assets)
            full_weights[selected_indices] = normalized_weights
            
            portfolios.append(full_weights)
        
        self.all_portfolios = portfolios
        print(f"✅ Generated {len(portfolios)} portfolio combinations")
        
        return portfolios
    
    def generate_systematic_portfolios(self, n_portfolios: int = 1000) -> List[np.ndarray]:
        """
        Generate systematic portfolio combinations using different strategies.
        
        Args:
            n_portfolios: Number of portfolios to generate
            
        Returns:
            List of portfolio weight arrays
        """
        print(f"⚙️ Generating {n_portfolios} systematic portfolio combinations...")
        
        portfolios = []
        n_assets = len(self.returns_data.columns)
        
        # Strategy 1: Equal weight portfolios with different asset counts
        equal_weight_portfolios = int(n_portfolios * 0.2)
        for i in range(equal_weight_portfolios):
            n_selected = np.random.randint(5, min(51, n_assets + 1))
            selected_indices = np.random.choice(n_assets, size=n_selected, replace=False)
            
            weights = np.zeros(n_assets)
            weights[selected_indices] = 1.0 / n_selected
            portfolios.append(weights)
        
        # Strategy 2: Market cap weighted (using recent returns as proxy)
        market_weight_portfolios = int(n_portfolios * 0.2)
        recent_returns = self.returns_data.tail(252).mean()  # Last year average
        market_weights = np.maximum(recent_returns, 0)
        market_weights = market_weights / market_weights.sum()
        
        for i in range(market_weight_portfolios):
            # Add noise to market weights
            noise = np.random.normal(0, 0.1, n_assets)
            noisy_weights = market_weights + noise
            noisy_weights = np.maximum(noisy_weights, 0)
            
            # Select top assets
            n_selected = np.random.randint(10, min(41, n_assets + 1))
            top_indices = np.argsort(noisy_weights)[-n_selected:]
            
            weights = np.zeros(n_assets)
            weights[top_indices] = noisy_weights[top_indices]
            weights = weights / weights.sum()
            portfolios.append(weights)
        
        # Strategy 3: Momentum-based portfolios
        momentum_portfolios = int(n_portfolios * 0.2)
        momentum_scores = self.returns_data.tail(63).mean()  # 3-month momentum
        
        for i in range(momentum_portfolios):
            # Select top momentum assets
            n_selected = np.random.randint(8, min(31, n_assets + 1))
            top_momentum = np.argsort(momentum_scores)[-n_selected:]
            
            # Generate weights inversely related to volatility
            volatilities = self.returns_data.iloc[:, top_momentum].std()
            inv_vol_weights = 1 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            
            weights = np.zeros(n_assets)
            weights[top_momentum] = inv_vol_weights
            portfolios.append(weights)
        
        # Strategy 4: Low volatility portfolios
        low_vol_portfolios = int(n_portfolios * 0.2)
        volatilities = self.returns_data.std()
        
        for i in range(low_vol_portfolios):
            # Select low volatility assets
            n_selected = np.random.randint(10, min(41, n_assets + 1))
            low_vol_indices = np.argsort(volatilities)[:n_selected]
            
            weights = np.zeros(n_assets)
            weights[low_vol_indices] = 1.0 / n_selected
            portfolios.append(weights)
        
        # Strategy 5: Random combinations (fill remainder)
        remaining = n_portfolios - len(portfolios)
        random_portfolios = self.generate_random_portfolios(remaining)
        portfolios.extend(random_portfolios)
        
        self.all_portfolios = portfolios[:n_portfolios]
        print(f"✅ Generated {len(self.all_portfolios)} systematic portfolio combinations")
        
        return self.all_portfolios
    
    def calculate_portfolio_metrics_batch(self, portfolios: List[np.ndarray], 
                                        batch_size: int = 100) -> List[Dict]:
        """
        Calculate metrics for a batch of portfolios efficiently.
        
        Args:
            portfolios: List of portfolio weight arrays
            batch_size: Number of portfolios to process in each batch
            
        Returns:
            List of portfolio metrics dictionaries
        """
        print(f"📊 Calculating metrics for {len(portfolios)} portfolios...")
        
        results = []
        
        for i in tqdm(range(0, len(portfolios), batch_size), desc="Processing batches"):
            batch = portfolios[i:i + batch_size]
            batch_results = []
            
            for weights in batch:
                try:
                    # Calculate portfolio returns
                    portfolio_returns = self.returns_data.dot(weights)
                    
                    # Calculate metrics
                    metrics = self.metrics_calc.calculate_all_metrics(weights, self.returns_data)
                    
                    # Add portfolio-specific data
                    portfolio_data = {
                        'weights': weights,
                        'portfolio_returns': portfolio_returns,
                        'n_assets': np.sum(weights > 0.001),  # Number of assets with meaningful weight
                        'concentration': np.sum(weights**2),  # Herfindahl index
                        'max_weight': np.max(weights),
                        **metrics
                    }
                    
                    batch_results.append(portfolio_data)
                    
                except Exception as e:
                    # Skip problematic portfolios
                    continue
            
            results.extend(batch_results)
        
        self.portfolio_results = results
        print(f"✅ Calculated metrics for {len(results)} portfolios")
        
        return results
    
    def analyze_portfolio_universe(self) -> pd.DataFrame:
        """
        Analyze the entire portfolio universe and create summary statistics.
        
        Returns:
            DataFrame with portfolio analysis results
        """
        print("🔍 Analyzing portfolio universe...")
        
        if not self.portfolio_results:
            raise ValueError("No portfolio results available. Run calculate_portfolio_metrics_batch first.")
        
        # Convert results to DataFrame
        data_for_df = []
        for i, result in enumerate(self.portfolio_results):
            row = {
                'portfolio_id': i,
                'annualized_return': result['annualized_return'],
                'annualized_volatility': result['annualized_volatility'],
                'sharpe_ratio': result['sharpe_ratio'],
                'sortino_ratio': result['sortino_ratio'],
                'max_drawdown': result['max_drawdown'],
                'calmar_ratio': result['calmar_ratio'],
                'var_5': result['var_5'],
                'cvar_5': result['cvar_5'],
                'skewness': result['skewness'],
                'kurtosis': result['kurtosis'],
                'n_assets': result['n_assets'],
                'concentration': result['concentration'],
                'max_weight': result['max_weight']
            }
            data_for_df.append(row)
        
        portfolio_df = pd.DataFrame(data_for_df)
        
        # Calculate summary statistics
        print(f"\n📈 Portfolio Universe Analysis Summary:")
        print(f"   Total Portfolios Analyzed: {len(portfolio_df)}")
        print(f"   Return Range: {portfolio_df['annualized_return'].min():.2%} to {portfolio_df['annualized_return'].max():.2%}")
        print(f"   Volatility Range: {portfolio_df['annualized_volatility'].min():.2%} to {portfolio_df['annualized_volatility'].max():.2%}")
        print(f"   Sharpe Range: {portfolio_df['sharpe_ratio'].min():.3f} to {portfolio_df['sharpe_ratio'].max():.3f}")
        print(f"   Max Drawdown Range: {portfolio_df['max_drawdown'].min():.2%} to {portfolio_df['max_drawdown'].max():.2%}")
        print(f"   Average Assets per Portfolio: {portfolio_df['n_assets'].mean():.1f}")
        
        # Find top performers
        top_sharpe = portfolio_df.loc[portfolio_df['sharpe_ratio'].idxmax()]
        top_return = portfolio_df.loc[portfolio_df['annualized_return'].idxmax()]
        min_drawdown = portfolio_df.loc[portfolio_df['max_drawdown'].idxmin()]
        
        print(f"\n🏆 Top Performing Portfolios:")
        print(f"   Best Sharpe ({top_sharpe['sharpe_ratio']:.3f}): Return={top_sharpe['annualized_return']:.2%}, Vol={top_sharpe['annualized_volatility']:.2%}")
        print(f"   Highest Return ({top_return['annualized_return']:.2%}): Sharpe={top_return['sharpe_ratio']:.3f}, Vol={top_return['annualized_volatility']:.2%}")
        print(f"   Lowest Drawdown ({min_drawdown['max_drawdown']:.2%}): Return={min_drawdown['annualized_return']:.2%}, Sharpe={min_drawdown['sharpe_ratio']:.3f}")
        
        return portfolio_df
    
    def create_enhanced_visualizations(self, portfolio_df: pd.DataFrame, 
                                     save_plots: bool = True,
                                     plots_dir: str = "results/plots") -> None:
        """
        Create enhanced visualizations for large-scale portfolio analysis.
        
        Args:
            portfolio_df: DataFrame with portfolio analysis results
            save_plots: Whether to save plots
            plots_dir: Directory to save plots
        """
        print("📊 Creating enhanced visualizations for large-scale analysis...")
        
        import os
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 3D Risk-Return-Sharpe Scatter Plot
        fig = go.Figure(data=go.Scatter3d(
            x=portfolio_df['annualized_volatility'],
            y=portfolio_df['annualized_return'],
            z=portfolio_df['sharpe_ratio'],
            mode='markers',
            marker=dict(
                size=3,
                color=portfolio_df['sharpe_ratio'],
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.7
            ),
            text=[f"Portfolio {i}<br>Return: {r:.2%}<br>Vol: {v:.2%}<br>Sharpe: {s:.3f}" 
                  for i, r, v, s in zip(portfolio_df['portfolio_id'], 
                                       portfolio_df['annualized_return'],
                                       portfolio_df['annualized_volatility'],
                                       portfolio_df['sharpe_ratio'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Portfolio Analysis - {len(portfolio_df)} Portfolios',
            scene=dict(
                xaxis_title='Annualized Volatility',
                yaxis_title='Annualized Return',
                zaxis_title='Sharpe Ratio'
            ),
            width=1000,
            height=700
        )
        
        if save_plots:
            fig.write_html(f"{plots_dir}/portfolio_universe_3d.html")
            print(f"   3D portfolio universe saved to {plots_dir}/portfolio_universe_3d.html")
        
        fig.show()
        
        # 2. Enhanced Efficient Frontier with Density
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Risk-Return Scatter', 'Sharpe Ratio Distribution',
                                         'Drawdown vs Return', 'Portfolio Concentration'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"secondary_y": False}]])
        
        # Risk-Return scatter
        fig.add_trace(go.Scatter(
            x=portfolio_df['annualized_volatility'],
            y=portfolio_df['annualized_return'],
            mode='markers',
            marker=dict(
                color=portfolio_df['sharpe_ratio'],
                colorscale='Viridis',
                size=4,
                opacity=0.6
            ),
            name='Portfolios'
        ), row=1, col=1)
        
        # Sharpe ratio histogram
        fig.add_trace(go.Histogram(
            x=portfolio_df['sharpe_ratio'],
            nbinsx=50,
            name='Sharpe Distribution'
        ), row=1, col=2)
        
        # Drawdown vs Return
        fig.add_trace(go.Scatter(
            x=portfolio_df['max_drawdown'],
            y=portfolio_df['annualized_return'],
            mode='markers',
            marker=dict(
                color=portfolio_df['sharpe_ratio'],
                colorscale='Viridis',
                size=4,
                opacity=0.6
            ),
            name='DD vs Return'
        ), row=2, col=1)
        
        # Portfolio concentration
        fig.add_trace(go.Scatter(
            x=portfolio_df['n_assets'],
            y=portfolio_df['concentration'],
            mode='markers',
            marker=dict(
                color=portfolio_df['sharpe_ratio'],
                colorscale='Viridis',
                size=4,
                opacity=0.6
            ),
            name='Concentration'
        ), row=2, col=2)
        
        fig.update_layout(
            title=f'Portfolio Universe Analysis - {len(portfolio_df)} Portfolios',
            height=800,
            width=1200,
            showlegend=False
        )
        
        if save_plots:
            fig.write_html(f"{plots_dir}/portfolio_analysis_dashboard.html")
            print(f"   Analysis dashboard saved to {plots_dir}/portfolio_analysis_dashboard.html")
        
        fig.show()
    
    def run_large_scale_analysis(self, n_portfolios: int = 1000,
                                period_years: int = 20,
                                save_results: bool = True) -> Dict:
        """
        Run complete large-scale portfolio analysis.
        
        Args:
            n_portfolios: Number of portfolios to analyze
            period_years: Years of historical data
            save_results: Whether to save results
            
        Returns:
            Complete analysis results
        """
        print("🚀 LARGE-SCALE PORTFOLIO ANALYSIS")
        print("="*70)
        print(f"Target Portfolios: {n_portfolios}")
        print(f"Asset Universe: Full available assets")
        print(f"Historical Period: {period_years} years")
        print("="*70)
        
        start_time = datetime.now()
        
        # Step 1: Fetch full universe data
        self.fetch_full_universe_data(period_years=period_years)
        
        # Step 2: Generate portfolio combinations
        portfolios = self.generate_systematic_portfolios(n_portfolios=n_portfolios)
        
        # Step 3: Calculate metrics for all portfolios
        portfolio_results = self.calculate_portfolio_metrics_batch(portfolios)
        
        # Step 4: Analyze portfolio universe
        portfolio_df = self.analyze_portfolio_universe()
        
        # Step 5: Create visualizations
        self.create_enhanced_visualizations(portfolio_df, save_plots=save_results)
        
        # Step 6: Export results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export portfolio analysis
            portfolio_file = f"results/large_scale_analysis_{timestamp}.csv"
            portfolio_df.to_csv(portfolio_file, index=False)
            print(f"   Large-scale analysis exported to: {portfolio_file}")
            
            # Export top performers
            top_performers = portfolio_df.nlargest(20, 'sharpe_ratio')
            top_file = f"results/top_performers_{timestamp}.csv"
            top_performers.to_csv(top_file, index=False)
            print(f"   Top performers exported to: {top_file}")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        results = {
            'portfolio_df': portfolio_df,
            'portfolio_results': portfolio_results,
            'n_assets': len(self.returns_data.columns),
            'n_portfolios': len(portfolio_results),
            'analysis_period': (self.prices_data.index[0], self.prices_data.index[-1]),
            'runtime': duration
        }
        
        print(f"\n🎉 LARGE-SCALE ANALYSIS COMPLETED")
        print("="*70)
        print(f"Runtime: {duration}")
        print(f"Assets Analyzed: {results['n_assets']}")
        print(f"Portfolios Generated: {results['n_portfolios']}")
        print(f"Best Sharpe Ratio: {portfolio_df['sharpe_ratio'].max():.3f}")
        print(f"Best Return: {portfolio_df['annualized_return'].max():.2%}")
        print(f"Lowest Drawdown: {portfolio_df['max_drawdown'].min():.2%}")
        
        return results


def main():
    """Example usage of LargeScalePortfolioAnalysis."""
    # Initialize large-scale analysis
    analysis = LargeScalePortfolioAnalysis(initial_investment=100000)
    
    # Run large-scale analysis
    results = analysis.run_large_scale_analysis(
        n_portfolios=1000,
        period_years=20,
        save_results=True
    )
    
    print(f"\n📊 Large-Scale Analysis Complete!")
    print("Check the results/ directory for detailed analysis files.")


if __name__ == "__main__":
    main()
