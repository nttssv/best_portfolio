"""
Extended portfolio analysis module for 20-year historical performance evaluation.
Includes monthly returns, annual returns, portfolio growth simulation, and comprehensive metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from portfolio_optimizer import EfficientFrontierOptimizer, OptimizationConstraints
from portfolio_metrics import PortfolioMetrics
from visualization import PortfolioVisualizer


class ExtendedPortfolioAnalysis:
    """Extended portfolio analysis with 20-year historical performance evaluation."""
    
    def __init__(self, initial_investment: float = 100000.0):
        """
        Initialize extended analysis.
        
        Args:
            initial_investment: Initial portfolio value in dollars
        """
        self.initial_investment = initial_investment
        self.data_fetcher = None
        self.optimizer = None
        self.portfolio_weights = None
        self.returns_data = None
        self.prices_data = None
        self.portfolio_returns = None
        self.monthly_returns = None
        self.annual_returns = None
        self.portfolio_values = None
        self.metrics_calc = PortfolioMetrics()
        
    def fetch_extended_data(self, num_assets: int = 50, period_years: int = 20) -> pd.DataFrame:
        """
        Fetch 20 years of historical data for analysis.
        
        Args:
            num_assets: Number of assets to analyze
            period_years: Years of historical data
            
        Returns:
            Returns DataFrame
        """
        print(f"🔄 Fetching {period_years} years of historical data for {num_assets} assets...")
        
        # Initialize data fetcher with extended period
        self.data_fetcher = MarketDataFetcher(period_years=period_years)
        
        # Get top assets by volume
        symbols = self.data_fetcher.get_top_assets_by_volume(num_assets=num_assets)
        
        # Fetch historical data
        self.prices_data = self.data_fetcher.fetch_historical_data(symbols)
        self.returns_data = self.data_fetcher.calculate_returns()
        
        print(f"✅ Extended data fetching completed:")
        print(f"   Assets: {len(self.returns_data.columns)}")
        print(f"   Date Range: {self.prices_data.index[0].date()} to {self.prices_data.index[-1].date()}")
        print(f"   Trading Days: {len(self.returns_data)}")
        
        return self.returns_data
    
    def optimize_portfolio_extended(self, relaxed_constraints: bool = True) -> Dict:
        """
        Optimize portfolio with potentially relaxed constraints for 20-year analysis.
        
        Args:
            relaxed_constraints: Whether to use more realistic constraints
            
        Returns:
            Optimization results
        """
        print("🎯 Optimizing portfolio for extended analysis...")
        
        # Initialize optimizer
        self.optimizer = EfficientFrontierOptimizer(self.returns_data, risk_free_rate=0.02)
        
        # Set constraints - relaxed for realistic 20-year analysis
        if relaxed_constraints:
            constraints = OptimizationConstraints(
                min_return=0.12,    # 12% minimum return (more realistic)
                min_sharpe=1.0,     # 1.0 minimum Sharpe (more realistic)
                max_drawdown=0.15,  # 15% max drawdown (more realistic)
                max_weight=0.15,    # Max 15% in any asset
                max_assets=30       # Max 30 assets
            )
            print("   Using relaxed constraints for realistic 20-year analysis")
        else:
            constraints = OptimizationConstraints(
                min_return=0.20,
                min_sharpe=2.0,
                max_drawdown=0.03
            )
            print("   Using original strict constraints")
        
        # Find optimal portfolios
        optimal_portfolios = self.optimizer.find_optimal_portfolios(constraints)
        
        # Select best feasible portfolio
        best_portfolio = None
        best_score = -np.inf
        
        for name, portfolio in optimal_portfolios.items():
            if portfolio['weights'] is not None:
                constraints_met = portfolio['constraints_met']['all_constraints']
                sharpe_ratio = portfolio['metrics']['sharpe_ratio']
                score = (10 if constraints_met else 0) + sharpe_ratio
                
                if score > best_score:
                    best_score = score
                    best_portfolio = portfolio
                    best_name = name
        
        if best_portfolio:
            self.portfolio_weights = best_portfolio['weights']
            print(f"✅ Selected {best_name} portfolio:")
            print(f"   Return: {best_portfolio['metrics']['annualized_return']:.2%}")
            print(f"   Sharpe: {best_portfolio['metrics']['sharpe_ratio']:.3f}")
            print(f"   Max DD: {best_portfolio['metrics']['max_drawdown']:.2%}")
            print(f"   Constraints Met: {'✅' if best_portfolio['constraints_met']['all_constraints'] else '❌'}")
        else:
            # Fallback to equal-weighted portfolio
            print("⚠️  No optimal portfolio found, using equal-weighted portfolio")
            self.portfolio_weights = np.ones(len(self.returns_data.columns)) / len(self.returns_data.columns)
        
        return optimal_portfolios
    
    def calculate_portfolio_performance(self) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics over 20 years.
        
        Returns:
            Dictionary with performance data
        """
        print("📊 Calculating portfolio performance over 20-year period...")
        
        if self.portfolio_weights is None:
            raise ValueError("Portfolio weights not set. Run optimize_portfolio_extended first.")
        
        # Calculate daily portfolio returns
        self.portfolio_returns = self.returns_data.dot(self.portfolio_weights)
        
        # Calculate monthly returns
        self.monthly_returns = self.portfolio_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Calculate annual returns
        self.annual_returns = self.portfolio_returns.resample('Y').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Calculate portfolio values starting with initial investment
        self.portfolio_values = self.initial_investment * (1 + self.portfolio_returns).cumprod()
        
        # Calculate comprehensive metrics
        performance_metrics = self.metrics_calc.calculate_all_metrics(
            self.portfolio_weights, self.returns_data
        )
        
        # Calculate additional 20-year specific metrics
        total_return = (self.portfolio_values.iloc[-1] / self.initial_investment) - 1
        cagr = (self.portfolio_values.iloc[-1] / self.initial_investment) ** (1/20) - 1
        
        # Calculate rolling metrics
        rolling_sharpe = self.portfolio_returns.rolling(252).apply(
            lambda x: (x.mean() * 252 - 0.02) / (x.std() * np.sqrt(252))
        )
        
        # Calculate drawdowns
        running_max = self.portfolio_values.expanding().max()
        drawdowns = (self.portfolio_values - running_max) / running_max
        
        performance_data = {
            'portfolio_returns': self.portfolio_returns,
            'monthly_returns': self.monthly_returns,
            'annual_returns': self.annual_returns,
            'portfolio_values': self.portfolio_values,
            'drawdowns': drawdowns,
            'rolling_sharpe': rolling_sharpe,
            'metrics': performance_metrics,
            'total_return': total_return,
            'cagr': cagr,
            'initial_investment': self.initial_investment,
            'final_value': self.portfolio_values.iloc[-1]
        }
        
        print(f"✅ Performance calculation completed:")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   CAGR: {cagr:.2%}")
        print(f"   Final Value: ${self.portfolio_values.iloc[-1]:,.0f}")
        print(f"   Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
        
        return performance_data
    
    def create_monthly_returns_chart(self, save_path: Optional[str] = None) -> None:
        """
        Create monthly returns time series chart.
        
        Args:
            save_path: Path to save the chart
        """
        print("📈 Creating monthly returns time series chart...")
        
        fig = go.Figure()
        
        # Add monthly returns line
        fig.add_trace(go.Scatter(
            x=self.monthly_returns.index,
            y=self.monthly_returns.values * 100,
            mode='lines+markers',
            name='Monthly Returns',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{x|%Y-%m}</b><br>' +
                         'Monthly Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Add recession shading (approximate major recessions)
        recessions = [
            ('2008-01-01', '2009-06-30', 'Financial Crisis'),
            ('2020-02-01', '2020-04-30', 'COVID-19 Pandemic')
        ]
        
        for start, end, name in recessions:
            if pd.Timestamp(start) >= self.monthly_returns.index[0] and pd.Timestamp(end) <= self.monthly_returns.index[-1]:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="red", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text=name,
                    annotation_position="top left"
                )
        
        fig.update_layout(
            title='Portfolio Monthly Returns Over 20 Years',
            xaxis_title='Date',
            yaxis_title='Monthly Return (%)',
            hovermode='x unified',
            showlegend=True,
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   Monthly returns chart saved to {save_path}")
        
        fig.show()
    
    def create_annual_returns_chart(self, save_path: Optional[str] = None) -> None:
        """
        Create annual returns bar chart.
        
        Args:
            save_path: Path to save the chart
        """
        print("📊 Creating annual returns bar chart...")
        
        # Prepare data
        years = [d.year for d in self.annual_returns.index]
        returns_pct = self.annual_returns.values * 100
        
        # Color bars based on performance
        colors = ['green' if r > 0 else 'red' for r in returns_pct]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years,
            y=returns_pct,
            marker_color=colors,
            name='Annual Returns',
            hovertemplate='<b>%{x}</b><br>' +
                         'Annual Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add average return line
        avg_return = np.mean(returns_pct)
        fig.add_hline(y=avg_return, line_dash="dash", line_color="blue", 
                     annotation_text=f"Average: {avg_return:.1f}%")
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        
        fig.update_layout(
            title='Portfolio Annual Returns Over 20 Years',
            xaxis_title='Year',
            yaxis_title='Annual Return (%)',
            showlegend=False,
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"   Annual returns chart saved to {save_path}")
        
        fig.show()
    
    def create_portfolio_growth_chart(self, save_path: Optional[str] = None) -> None:
        """
        Create portfolio value growth curve with drawdown overlay.
        
        Args:
            save_path: Path to save the chart
        """
        print("📈 Creating portfolio growth curve with drawdowns...")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Portfolio Value Growth', 'Drawdowns'),
            row_heights=[0.7, 0.3]
        )
        
        # Portfolio value growth (top chart)
        fig.add_trace(
            go.Scatter(
                x=self.portfolio_values.index,
                y=self.portfolio_values.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                             'Portfolio Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add initial investment line
        fig.add_hline(
            y=self.initial_investment,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${self.initial_investment:,.0f}",
            row=1, col=1
        )
        
        # Drawdowns (bottom chart)
        drawdowns_pct = ((self.portfolio_values - self.portfolio_values.expanding().max()) / 
                        self.portfolio_values.expanding().max()) * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdowns_pct.index,
                y=drawdowns_pct.values,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                             'Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line for drawdowns
        fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Portfolio Growth Simulation - ${self.initial_investment:,.0f} Initial Investment',
            height=800,
            width=1200,
            showlegend=True
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"   Portfolio growth chart saved to {save_path}")
        
        fig.show()
    
    def create_performance_summary_table(self) -> pd.DataFrame:
        """
        Create comprehensive performance metrics summary table.
        
        Returns:
            DataFrame with performance metrics
        """
        print("📋 Creating performance summary table...")
        
        if self.portfolio_returns is None:
            raise ValueError("Portfolio performance not calculated. Run calculate_portfolio_performance first.")
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(self.portfolio_weights, self.returns_data)
        
        # Additional calculations
        total_return = (self.portfolio_values.iloc[-1] / self.initial_investment) - 1
        cagr = (self.portfolio_values.iloc[-1] / self.initial_investment) ** (1/20) - 1
        
        # Best and worst years
        best_year = self.annual_returns.max()
        worst_year = self.annual_returns.min()
        positive_years = (self.annual_returns > 0).sum()
        
        # Volatility of annual returns
        annual_volatility = self.annual_returns.std()
        
        # Create summary table
        summary_data = [
            ['Investment Period', f"{self.portfolio_values.index[0].date()} to {self.portfolio_values.index[-1].date()}"],
            ['Initial Investment', f"${self.initial_investment:,.0f}"],
            ['Final Value', f"${self.portfolio_values.iloc[-1]:,.0f}"],
            ['Total Return', f"{total_return:.2%}"],
            ['Compound Annual Growth Rate (CAGR)', f"{cagr:.2%}"],
            ['Annualized Return', f"{metrics['annualized_return']:.2%}"],
            ['Annualized Volatility', f"{metrics['annualized_volatility']:.2%}"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
            ['Sortino Ratio', f"{metrics['sortino_ratio']:.3f}"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.3f}"],
            ['Maximum Drawdown', f"{metrics['max_drawdown']:.2%}"],
            ['Value at Risk (5%)', f"{metrics['var_5']:.2%}"],
            ['Conditional VaR (5%)', f"{metrics['cvar_5']:.2%}"],
            ['Best Annual Return', f"{best_year:.2%}"],
            ['Worst Annual Return', f"{worst_year:.2%}"],
            ['Positive Years', f"{positive_years} out of {len(self.annual_returns)}"],
            ['Annual Return Volatility', f"{annual_volatility:.2%}"],
            ['Skewness', f"{metrics['skewness']:.3f}"],
            ['Kurtosis', f"{metrics['kurtosis']:.3f}"]
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        
        print("✅ Performance summary table created")
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE SUMMARY (20 YEARS)")
        print("="*60)
        for _, row in summary_df.iterrows():
            print(f"{row['Metric']:<35}: {row['Value']}")
        print("="*60)
        
        return summary_df
    
    def run_complete_extended_analysis(self, num_assets: int = 50, 
                                     save_plots: bool = True,
                                     plots_dir: str = "results/plots") -> Dict:
        """
        Run complete 20-year extended portfolio analysis.
        
        Args:
            num_assets: Number of assets to analyze
            save_plots: Whether to save plots
            plots_dir: Directory to save plots
            
        Returns:
            Complete analysis results
        """
        print("🚀 Starting Complete 20-Year Portfolio Analysis")
        print("="*60)
        
        # Step 1: Fetch extended data
        self.fetch_extended_data(num_assets=num_assets, period_years=20)
        
        # Step 2: Optimize portfolio
        optimal_portfolios = self.optimize_portfolio_extended(relaxed_constraints=True)
        
        # Step 3: Calculate performance
        performance_data = self.calculate_portfolio_performance()
        
        # Step 4: Create visualizations
        import os
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
        
        # Monthly returns chart
        monthly_path = f"{plots_dir}/monthly_returns_20yr.html" if save_plots else None
        self.create_monthly_returns_chart(save_path=monthly_path)
        
        # Annual returns chart
        annual_path = f"{plots_dir}/annual_returns_20yr.html" if save_plots else None
        self.create_annual_returns_chart(save_path=annual_path)
        
        # Portfolio growth chart
        growth_path = f"{plots_dir}/portfolio_growth_20yr.html" if save_plots else None
        self.create_portfolio_growth_chart(save_path=growth_path)
        
        # Step 5: Create summary table
        summary_table = self.create_performance_summary_table()
        
        # Compile results
        results = {
            'performance_data': performance_data,
            'optimal_portfolios': optimal_portfolios,
            'summary_table': summary_table,
            'portfolio_weights': self.portfolio_weights,
            'symbols': self.data_fetcher.symbols,
            'data_period': (self.prices_data.index[0], self.prices_data.index[-1])
        }
        
        print("\n🎉 Complete 20-Year Analysis Finished!")
        print(f"   Analysis Period: {results['data_period'][0].date()} to {results['data_period'][1].date()}")
        print(f"   Assets Analyzed: {len(results['symbols'])}")
        print(f"   Final Portfolio Value: ${performance_data['final_value']:,.0f}")
        print(f"   Total Return: {performance_data['total_return']:.2%}")
        print(f"   CAGR: {performance_data['cagr']:.2%}")
        
        return results


def main():
    """Example usage of ExtendedPortfolioAnalysis."""
    # Initialize extended analysis
    analysis = ExtendedPortfolioAnalysis(initial_investment=100000)
    
    # Run complete analysis
    results = analysis.run_complete_extended_analysis(
        num_assets=30,  # Start with 30 assets for testing
        save_plots=True
    )
    
    print("\n📊 Extended Analysis Complete!")
    print("Check the results/plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
