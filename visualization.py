"""
Visualization module for efficient frontier and portfolio analysis.
Creates comprehensive charts and plots for portfolio optimization results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PortfolioVisualizer:
    """Create comprehensive visualizations for portfolio analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = {
            'efficient': '#1f77b4',
            'feasible': '#ff7f0e',
            'optimal': '#2ca02c',
            'infeasible': '#d62728',
            'benchmark': '#9467bd'
        }
    
    def plot_efficient_frontier(self, frontier_df: pd.DataFrame, 
                               optimal_portfolios: Dict = None,
                               save_path: Optional[str] = None,
                               interactive: bool = True) -> None:
        """
        Plot the efficient frontier with constraint regions.
        
        Args:
            frontier_df: DataFrame with efficient frontier portfolios
            optimal_portfolios: Dictionary of optimal portfolios
            save_path: Path to save the plot
            interactive: Whether to create interactive plotly chart
        """
        if frontier_df.empty:
            print("No data to plot - frontier_df is empty")
            return
        
        if interactive:
            self._plot_interactive_frontier(frontier_df, optimal_portfolios, save_path)
        else:
            self._plot_static_frontier(frontier_df, optimal_portfolios, save_path)
    
    def _plot_static_frontier(self, frontier_df: pd.DataFrame, 
                             optimal_portfolios: Dict = None,
                             save_path: Optional[str] = None) -> None:
        """Create static matplotlib efficient frontier plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Separate feasible and infeasible portfolios
        feasible = frontier_df[frontier_df['constraints_met'] == True]
        infeasible = frontier_df[frontier_df['constraints_met'] == False]
        
        # Plot infeasible portfolios
        if not infeasible.empty:
            ax.scatter(infeasible['annualized_volatility'], infeasible['annualized_return'],
                      c=self.colors['infeasible'], alpha=0.6, s=30, 
                      label='Infeasible (Constraints Not Met)')
        
        # Plot feasible portfolios
        if not feasible.empty:
            ax.scatter(feasible['annualized_volatility'], feasible['annualized_return'],
                      c=self.colors['feasible'], alpha=0.8, s=40,
                      label='Feasible (All Constraints Met)')
        
        # Plot efficient frontier line
        if len(frontier_df) > 1:
            sorted_df = frontier_df.sort_values('annualized_volatility')
            ax.plot(sorted_df['annualized_volatility'], sorted_df['annualized_return'],
                   color=self.colors['efficient'], linewidth=2, alpha=0.7,
                   label='Efficient Frontier')
        
        # Plot optimal portfolios
        if optimal_portfolios:
            for name, portfolio in optimal_portfolios.items():
                if portfolio['weights'] is not None:
                    metrics = portfolio['metrics']
                    ax.scatter(metrics['annualized_volatility'], metrics['annualized_return'],
                              c=self.colors['optimal'], s=100, marker='*',
                              label=f'Optimal: {name.replace("_", " ").title()}')
        
        # Add constraint lines
        ax.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='Min Return (20%)')
        
        # Formatting
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        ax.set_title('Efficient Frontier with Constraints', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format axes as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Efficient frontier plot saved to {save_path}")
        
        plt.show()
    
    def _plot_interactive_frontier(self, frontier_df: pd.DataFrame,
                                  optimal_portfolios: Dict = None,
                                  save_path: Optional[str] = None) -> None:
        """Create interactive plotly efficient frontier plot."""
        fig = go.Figure()
        
        # Separate feasible and infeasible portfolios
        feasible = frontier_df[frontier_df['constraints_met'] == True]
        infeasible = frontier_df[frontier_df['constraints_met'] == False]
        
        # Plot infeasible portfolios
        if not infeasible.empty:
            fig.add_trace(go.Scatter(
                x=infeasible['annualized_volatility'],
                y=infeasible['annualized_return'],
                mode='markers',
                name='Infeasible',
                marker=dict(color=self.colors['infeasible'], size=8, opacity=0.6),
                hovertemplate='<b>Infeasible Portfolio</b><br>' +
                             'Return: %{y:.2%}<br>' +
                             'Volatility: %{x:.2%}<br>' +
                             'Sharpe: %{customdata[0]:.3f}<br>' +
                             'Max DD: %{customdata[1]:.2%}<extra></extra>',
                customdata=np.column_stack((infeasible['sharpe_ratio'], infeasible['max_drawdown']))
            ))
        
        # Plot feasible portfolios
        if not feasible.empty:
            fig.add_trace(go.Scatter(
                x=feasible['annualized_volatility'],
                y=feasible['annualized_return'],
                mode='markers',
                name='Feasible',
                marker=dict(color=self.colors['feasible'], size=10, opacity=0.8),
                hovertemplate='<b>Feasible Portfolio</b><br>' +
                             'Return: %{y:.2%}<br>' +
                             'Volatility: %{x:.2%}<br>' +
                             'Sharpe: %{customdata[0]:.3f}<br>' +
                             'Max DD: %{customdata[1]:.2%}<extra></extra>',
                customdata=np.column_stack((feasible['sharpe_ratio'], feasible['max_drawdown']))
            ))
        
        # Plot efficient frontier line
        if len(frontier_df) > 1:
            sorted_df = frontier_df.sort_values('annualized_volatility')
            fig.add_trace(go.Scatter(
                x=sorted_df['annualized_volatility'],
                y=sorted_df['annualized_return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color=self.colors['efficient'], width=3),
                hoverinfo='skip'
            ))
        
        # Plot optimal portfolios
        if optimal_portfolios:
            for name, portfolio in optimal_portfolios.items():
                if portfolio['weights'] is not None:
                    metrics = portfolio['metrics']
                    fig.add_trace(go.Scatter(
                        x=[metrics['annualized_volatility']],
                        y=[metrics['annualized_return']],
                        mode='markers',
                        name=f'Optimal: {name.replace("_", " ").title()}',
                        marker=dict(color=self.colors['optimal'], size=15, symbol='star'),
                        hovertemplate=f'<b>{name.replace("_", " ").title()}</b><br>' +
                                     'Return: %{y:.2%}<br>' +
                                     'Volatility: %{x:.2%}<br>' +
                                     f'Sharpe: {metrics["sharpe_ratio"]:.3f}<br>' +
                                     f'Max DD: {metrics["max_drawdown"]:.2%}<extra></extra>'
                    ))
        
        # Add constraint lines
        fig.add_hline(y=0.20, line_dash="dash", line_color="red", 
                     annotation_text="Min Return (20%)")
        
        # Update layout
        fig.update_layout(
            title='Interactive Efficient Frontier with Constraints',
            xaxis_title='Annualized Volatility',
            yaxis_title='Annualized Return',
            xaxis=dict(tickformat='.0%'),
            yaxis=dict(tickformat='.0%'),
            hovermode='closest',
            width=1000,
            height=700,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
            print(f"Interactive plot saved to {save_path.replace('.png', '.html')}")
        
        fig.show()
    
    def plot_portfolio_composition(self, weights: np.ndarray, symbols: List[str],
                                  title: str = "Portfolio Composition",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot portfolio composition as pie chart and bar chart.
        
        Args:
            weights: Portfolio weights
            symbols: Asset symbols
            title: Plot title
            save_path: Path to save the plot
        """
        # Filter out very small weights
        significant_weights = weights > 0.01
        plot_weights = weights[significant_weights]
        plot_symbols = [symbols[i] for i in range(len(symbols)) if significant_weights[i]]
        
        # Group small weights as "Others"
        if len(plot_weights) < len(weights):
            other_weight = weights[~significant_weights].sum()
            plot_weights = np.append(plot_weights, other_weight)
            plot_symbols.append('Others')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_weights)))
        wedges, texts, autotexts = ax1.pie(plot_weights, labels=plot_symbols, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax1.set_title(f'{title} - Pie Chart')
        
        # Bar chart
        bars = ax2.bar(range(len(plot_weights)), plot_weights * 100, color=colors)
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Weight (%)')
        ax2.set_title(f'{title} - Bar Chart')
        ax2.set_xticks(range(len(plot_symbols)))
        ax2.set_xticklabels(plot_symbols, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, weight in zip(bars, plot_weights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{weight*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio composition plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, portfolios: Dict[str, Dict],
                                   save_path: Optional[str] = None) -> None:
        """
        Compare performance metrics across multiple portfolios.
        
        Args:
            portfolios: Dictionary of portfolio results
            save_path: Path to save the plot
        """
        # Extract metrics for comparison
        metrics_data = []
        for name, portfolio in portfolios.items():
            if portfolio['weights'] is not None:
                metrics = portfolio['metrics']
                metrics_data.append({
                    'Portfolio': name.replace('_', ' ').title(),
                    'Return': metrics['annualized_return'],
                    'Volatility': metrics['annualized_volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Sortino Ratio': metrics['sortino_ratio'],
                    'Calmar Ratio': metrics['calmar_ratio']
                })
        
        if not metrics_data:
            print("No portfolio data to compare")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_to_plot = ['Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 
                          'Sortino Ratio', 'Calmar Ratio']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            bars = ax.bar(df['Portfolio'], df[metric], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(df))))
            
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            
            # Format y-axis for percentage metrics
            if metric in ['Return', 'Volatility', 'Max Drawdown']:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                if metric in ['Return', 'Volatility', 'Max Drawdown']:
                    label = f'{value:.1%}'
                else:
                    label = f'{value:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       label, ha='center', va='bottom', fontsize=10)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, returns_data: Dict[str, pd.Series],
                                 save_path: Optional[str] = None) -> None:
        """
        Plot return distributions for multiple portfolios.
        
        Args:
            returns_data: Dictionary of portfolio returns
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        ax1 = axes[0, 0]
        for name, returns in returns_data.items():
            ax1.hist(returns, bins=50, alpha=0.7, label=name.replace('_', ' ').title())
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Daily Returns')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax2 = axes[0, 1]
        for name, returns in returns_data.items():
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)')
        ax2.grid(True, alpha=0.3)
        
        # Box plot
        ax3 = axes[1, 0]
        returns_list = [returns.values for returns in returns_data.values()]
        labels = [name.replace('_', ' ').title() for name in returns_data.keys()]
        ax3.boxplot(returns_list, labels=labels)
        ax3.set_title('Returns Box Plot')
        ax3.set_ylabel('Daily Returns')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax4 = axes[1, 1]
        for name, returns in returns_data.items():
            cumulative = (1 + returns).cumprod()
            ax4.plot(cumulative.index, cumulative.values, 
                    label=name.replace('_', ' ').title(), linewidth=2)
        ax4.set_title('Cumulative Returns')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Returns distribution plot saved to {save_path}")
        
        plt.show()
    
    def create_dashboard(self, frontier_df: pd.DataFrame, 
                        optimal_portfolios: Dict,
                        returns_data: Dict[str, pd.Series],
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            frontier_df: Efficient frontier data
            optimal_portfolios: Optimal portfolio results
            returns_data: Portfolio returns data
            save_path: Path to save dashboard
        """
        # Create interactive dashboard using plotly
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Efficient Frontier', 'Performance Metrics', 
                           'Portfolio Composition', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "pie"}, {"secondary_y": False}]]
        )
        
        # Efficient frontier (top-left)
        if not frontier_df.empty:
            feasible = frontier_df[frontier_df['constraints_met'] == True]
            infeasible = frontier_df[frontier_df['constraints_met'] == False]
            
            if not infeasible.empty:
                fig.add_trace(go.Scatter(
                    x=infeasible['annualized_volatility'],
                    y=infeasible['annualized_return'],
                    mode='markers',
                    name='Infeasible',
                    marker=dict(color='red', size=6)
                ), row=1, col=1)
            
            if not feasible.empty:
                fig.add_trace(go.Scatter(
                    x=feasible['annualized_volatility'],
                    y=feasible['annualized_return'],
                    mode='markers',
                    name='Feasible',
                    marker=dict(color='blue', size=8)
                ), row=1, col=1)
        
        # Performance metrics (top-right)
        if optimal_portfolios:
            portfolio_names = []
            sharpe_ratios = []
            returns = []
            
            for name, portfolio in optimal_portfolios.items():
                if portfolio['weights'] is not None:
                    portfolio_names.append(name.replace('_', ' ').title())
                    sharpe_ratios.append(portfolio['metrics']['sharpe_ratio'])
                    returns.append(portfolio['metrics']['annualized_return'])
            
            fig.add_trace(go.Bar(
                x=portfolio_names,
                y=sharpe_ratios,
                name='Sharpe Ratio',
                marker_color='green'
            ), row=1, col=2)
        
        # Portfolio composition (bottom-left) - show best portfolio
        if optimal_portfolios:
            best_portfolio = None
            best_sharpe = -np.inf
            
            for portfolio in optimal_portfolios.values():
                if (portfolio['weights'] is not None and 
                    portfolio['metrics']['sharpe_ratio'] > best_sharpe):
                    best_sharpe = portfolio['metrics']['sharpe_ratio']
                    best_portfolio = portfolio
            
            if best_portfolio:
                weights = best_portfolio['weights']
                # Show top 10 holdings
                top_indices = np.argsort(weights)[-10:]
                top_weights = weights[top_indices]
                top_symbols = [f'Asset_{i}' for i in top_indices]  # Placeholder symbols
                
                fig.add_trace(go.Pie(
                    labels=top_symbols,
                    values=top_weights,
                    name="Portfolio Composition"
                ), row=2, col=1)
        
        # Cumulative returns (bottom-right)
        for name, returns in returns_data.items():
            cumulative = (1 + returns).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name=f'{name.replace("_", " ").title()} Returns'
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Portfolio Optimization Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")
        
        fig.show()


def main():
    """Example usage of PortfolioVisualizer."""
    # Generate sample data
    np.random.seed(42)
    n_portfolios = 50
    
    # Sample efficient frontier data
    returns = np.random.uniform(0.05, 0.25, n_portfolios)
    volatilities = np.random.uniform(0.10, 0.30, n_portfolios)
    sharpe_ratios = (returns - 0.02) / volatilities
    max_drawdowns = np.random.uniform(0.02, 0.15, n_portfolios)
    constraints_met = (returns >= 0.20) & (sharpe_ratios >= 2.0) & (max_drawdowns <= 0.03)
    
    frontier_df = pd.DataFrame({
        'annualized_return': returns,
        'annualized_volatility': volatilities,
        'sharpe_ratio': sharpe_ratios,
        'max_drawdown': max_drawdowns,
        'constraints_met': constraints_met
    })
    
    # Sample optimal portfolios
    optimal_portfolios = {
        'max_sharpe': {
            'weights': np.random.dirichlet(np.ones(10)),
            'metrics': {
                'annualized_return': 0.22,
                'annualized_volatility': 0.15,
                'sharpe_ratio': 1.33,
                'max_drawdown': 0.08
            }
        }
    }
    
    # Create visualizer and plot
    visualizer = PortfolioVisualizer()
    visualizer.plot_efficient_frontier(frontier_df, optimal_portfolios)


if __name__ == "__main__":
    main()
