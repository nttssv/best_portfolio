"""
Portfolio Evolution Tracker - Visualize portfolio composition changes over time
including dynamic rebalancing strategies and regime-based adjustments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import seaborn as sns
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from advanced_portfolio_optimizer import AdvancedPortfolioAnalysis, DynamicRebalancer, MultiObjectiveOptimizer


class PortfolioEvolutionTracker:
    """Track and visualize portfolio composition evolution over time."""
    
    def __init__(self, returns_data: pd.DataFrame, prices_data: pd.DataFrame):
        """
        Initialize portfolio evolution tracker.
        
        Args:
            returns_data: Returns DataFrame
            prices_data: Prices DataFrame
        """
        self.returns_data = returns_data
        self.prices_data = prices_data
        self.evolution_history = []
        self.rebalancing_events = []
        
    def simulate_portfolio_evolution(self, 
                                   initial_weights: np.ndarray,
                                   rebalance_frequency: int = 21,
                                   strategy: str = 'momentum') -> pd.DataFrame:
        """
        Simulate portfolio evolution with dynamic rebalancing.
        
        Args:
            initial_weights: Initial portfolio weights
            rebalance_frequency: Days between rebalances
            strategy: Rebalancing strategy ('momentum', 'volatility', 'mean_reversion')
            
        Returns:
            DataFrame with portfolio evolution history
        """
        print(f"📈 Simulating portfolio evolution with {strategy} strategy...")
        
        # Initialize tracking variables
        current_weights = initial_weights.copy()
        portfolio_value = 100000  # Start with $100k
        evolution_data = []
        
        # Get rebalancing dates
        rebalance_dates = self.returns_data.index[::rebalance_frequency]
        
        for i, date in enumerate(tqdm(self.returns_data.index, desc="Simulating evolution")):
            # Calculate daily portfolio return
            daily_returns = self.returns_data.loc[date]
            portfolio_return = np.dot(current_weights, daily_returns)
            portfolio_value *= (1 + portfolio_return)
            
            # Natural weight drift due to price changes
            asset_returns = daily_returns.values
            current_weights = current_weights * (1 + asset_returns)
            current_weights = current_weights / current_weights.sum()  # Renormalize
            
            # Check if rebalancing date
            is_rebalance_date = date in rebalance_dates
            rebalance_reason = None
            
            if is_rebalance_date and i > 63:  # Need history for calculations
                new_weights, reason = self._calculate_rebalancing_weights(
                    date, current_weights, strategy
                )
                
                if new_weights is not None:
                    # Calculate turnover
                    turnover = np.sum(np.abs(new_weights - current_weights)) / 2
                    
                    # Apply transaction costs (0.1% per trade)
                    transaction_cost = turnover * 0.001
                    portfolio_value *= (1 - transaction_cost)
                    
                    current_weights = new_weights
                    rebalance_reason = reason
                    
                    # Record rebalancing event
                    self.rebalancing_events.append({
                        'date': date,
                        'turnover': turnover,
                        'transaction_cost': transaction_cost,
                        'reason': reason,
                        'strategy': strategy
                    })
            
            # Record daily state
            evolution_data.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'portfolio_return': portfolio_return,
                'is_rebalance': is_rebalance_date,
                'rebalance_reason': rebalance_reason,
                'strategy': strategy,
                **{f'weight_{asset}': weight for asset, weight in zip(self.returns_data.columns, current_weights)}
            })
        
        evolution_df = pd.DataFrame(evolution_data)
        evolution_df.set_index('date', inplace=True)
        
        print(f"✅ Portfolio evolution simulation completed:")
        print(f"   Final Value: ${evolution_df['portfolio_value'].iloc[-1]:,.0f}")
        print(f"   Total Return: {(evolution_df['portfolio_value'].iloc[-1] / 100000 - 1):.2%}")
        print(f"   Rebalancing Events: {len(self.rebalancing_events)}")
        
        return evolution_df
    
    def _calculate_rebalancing_weights(self, date: pd.Timestamp, 
                                     current_weights: np.ndarray,
                                     strategy: str) -> Tuple[Optional[np.ndarray], str]:
        """Calculate new weights based on rebalancing strategy."""
        
        date_pos = self.returns_data.index.get_loc(date)
        lookback_period = 63  # 3 months
        
        if date_pos < lookback_period:
            return None, "insufficient_history"
        
        lookback_returns = self.returns_data.iloc[date_pos-lookback_period:date_pos]
        
        if strategy == 'momentum':
            # Momentum strategy: overweight recent winners
            momentum_scores = lookback_returns.mean() * 252  # Annualized
            momentum_ranks = momentum_scores.rank(pct=True)
            
            # Adjust weights based on momentum (±20% adjustment)
            momentum_adjustment = (momentum_ranks - 0.5) * 0.4
            new_weights = current_weights * (1 + momentum_adjustment)
            
        elif strategy == 'volatility':
            # Volatility strategy: overweight low volatility assets
            volatilities = lookback_returns.std() * np.sqrt(252)
            vol_ranks = volatilities.rank(pct=True, ascending=True)  # Low vol = high rank
            
            vol_adjustment = (vol_ranks - 0.5) * 0.3
            new_weights = current_weights * (1 + vol_adjustment)
            
        elif strategy == 'mean_reversion':
            # Mean reversion: underweight recent winners
            recent_performance = lookback_returns.mean() * 252
            performance_ranks = recent_performance.rank(pct=True, ascending=False)  # Reverse
            
            reversion_adjustment = (performance_ranks - 0.5) * 0.25
            new_weights = current_weights * (1 + reversion_adjustment)
            
        else:
            return None, "unknown_strategy"
        
        # Apply constraints
        new_weights = np.maximum(new_weights, 0.005)  # Min 0.5%
        new_weights = np.minimum(new_weights, 0.20)   # Max 20%
        new_weights = new_weights / new_weights.sum()  # Normalize
        
        # Only rebalance if significant change (>5% turnover)
        turnover = np.sum(np.abs(new_weights - current_weights)) / 2
        if turnover > 0.05:
            return new_weights, f"{strategy}_rebalance"
        else:
            return None, "minimal_change"
    
    def create_composition_heatmap(self, evolution_df: pd.DataFrame, 
                                 top_n_assets: int = 15) -> go.Figure:
        """
        Create heatmap showing portfolio composition over time.
        
        Args:
            evolution_df: Portfolio evolution DataFrame
            top_n_assets: Number of top assets to show
            
        Returns:
            Plotly figure
        """
        print(f"🎨 Creating portfolio composition heatmap...")
        
        # Get weight columns
        weight_cols = [col for col in evolution_df.columns if col.startswith('weight_')]
        weights_df = evolution_df[weight_cols]
        
        # Remove 'weight_' prefix from column names
        weights_df.columns = [col.replace('weight_', '') for col in weights_df.columns]
        
        # Select top assets by average weight
        avg_weights = weights_df.mean().sort_values(ascending=False)
        top_assets = avg_weights.head(top_n_assets).index
        
        # Subsample data for visualization (every 10th day)
        sample_dates = weights_df.index[::10]
        heatmap_data = weights_df.loc[sample_dates, top_assets]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.T.values,
            x=[d.strftime('%Y-%m') for d in heatmap_data.index],
            y=heatmap_data.columns,
            colorscale='Viridis',
            colorbar=dict(title="Portfolio Weight"),
            hoverongaps=False,
            hovertemplate='Date: %{x}<br>Asset: %{y}<br>Weight: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Portfolio Composition Evolution - Top {top_n_assets} Assets',
            xaxis_title='Date',
            yaxis_title='Assets',
            height=600,
            width=1200
        )
        
        return fig
    
    def create_weight_evolution_chart(self, evolution_df: pd.DataFrame,
                                    assets_to_track: Optional[List[str]] = None) -> go.Figure:
        """
        Create line chart showing weight evolution for specific assets.
        
        Args:
            evolution_df: Portfolio evolution DataFrame
            assets_to_track: List of assets to track (if None, uses top 10)
            
        Returns:
            Plotly figure
        """
        print(f"📊 Creating weight evolution chart...")
        
        # Get weight columns
        weight_cols = [col for col in evolution_df.columns if col.startswith('weight_')]
        weights_df = evolution_df[weight_cols]
        weights_df.columns = [col.replace('weight_', '') for col in weights_df.columns]
        
        # Select assets to track
        if assets_to_track is None:
            avg_weights = weights_df.mean().sort_values(ascending=False)
            assets_to_track = avg_weights.head(10).index.tolist()
        
        # Create subplot
        fig = go.Figure()
        
        # Color palette
        colors = px.colors.qualitative.Set3
        
        for i, asset in enumerate(assets_to_track):
            if asset in weights_df.columns:
                fig.add_trace(go.Scatter(
                    x=weights_df.index,
                    y=weights_df[asset] * 100,  # Convert to percentage
                    mode='lines',
                    name=asset,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{asset}<br>Date: %{{x}}<br>Weight: %{{y:.2f}}%<extra></extra>'
                ))
        
        # Add rebalancing events
        rebalance_dates = [event['date'] for event in self.rebalancing_events]
        if rebalance_dates:
            fig.add_trace(go.Scatter(
                x=rebalance_dates,
                y=[0] * len(rebalance_dates),
                mode='markers',
                marker=dict(symbol='triangle-up', size=8, color='red'),
                name='Rebalancing Events',
                hovertemplate='Rebalancing Event<br>Date: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Portfolio Weight Evolution Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Weight (%)',
            height=600,
            width=1200,
            hovermode='x unified'
        )
        
        return fig
    
    def create_turnover_analysis(self) -> go.Figure:
        """Create analysis of portfolio turnover over time."""
        
        if not self.rebalancing_events:
            print("⚠️ No rebalancing events to analyze")
            return None
        
        print(f"📈 Creating turnover analysis...")
        
        rebalance_df = pd.DataFrame(self.rebalancing_events)
        rebalance_df['date'] = pd.to_datetime(rebalance_df['date'])
        rebalance_df.set_index('date', inplace=True)
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Turnover Over Time', 'Cumulative Transaction Costs'),
            vertical_spacing=0.1
        )
        
        # Turnover chart
        fig.add_trace(go.Scatter(
            x=rebalance_df.index,
            y=rebalance_df['turnover'] * 100,
            mode='lines+markers',
            name='Turnover %',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ), row=1, col=1)
        
        # Transaction costs
        rebalance_df['cumulative_cost'] = rebalance_df['transaction_cost'].cumsum() * 100
        fig.add_trace(go.Scatter(
            x=rebalance_df.index,
            y=rebalance_df['cumulative_cost'],
            mode='lines',
            name='Cumulative Costs %',
            line=dict(color='red', width=2),
            fill='tonexty'
        ), row=2, col=1)
        
        fig.update_layout(
            title='Portfolio Rebalancing Analysis',
            height=600,
            width=1200
        )
        
        fig.update_yaxes(title_text="Turnover (%)", row=1, col=1)
        fig.update_yaxes(title_text="Transaction Costs (%)", row=2, col=1)
        
        return fig
    
    def create_comprehensive_dashboard(self, evolution_df: pd.DataFrame) -> go.Figure:
        """Create comprehensive portfolio evolution dashboard."""
        
        print(f"🎛️ Creating comprehensive portfolio evolution dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value Growth', 'Top 10 Asset Weights',
                'Monthly Rebalancing Activity', 'Weight Concentration Over Time',
                'Return Attribution', 'Risk Metrics Evolution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Portfolio value growth
        fig.add_trace(go.Scatter(
            x=evolution_df.index,
            y=evolution_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ), row=1, col=1)
        
        # 2. Top asset weights
        weight_cols = [col for col in evolution_df.columns if col.startswith('weight_')]
        weights_df = evolution_df[weight_cols]
        weights_df.columns = [col.replace('weight_', '') for col in weights_df.columns]
        
        top_assets = weights_df.mean().nlargest(5).index
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, asset in enumerate(top_assets):
            fig.add_trace(go.Scatter(
                x=evolution_df.index,
                y=weights_df[asset] * 100,
                mode='lines',
                name=asset,
                line=dict(color=colors[i], width=1.5)
            ), row=1, col=2)
        
        # 3. Rebalancing activity
        if self.rebalancing_events:
            rebalance_df = pd.DataFrame(self.rebalancing_events)
            rebalance_df['date'] = pd.to_datetime(rebalance_df['date'])
            monthly_rebalances = rebalance_df.groupby(rebalance_df['date'].dt.to_period('M')).size()
            
            fig.add_trace(go.Bar(
                x=[str(period) for period in monthly_rebalances.index],
                y=monthly_rebalances.values,
                name='Rebalances/Month',
                marker_color='lightblue'
            ), row=2, col=1)
        
        # 4. Weight concentration (Herfindahl index)
        weights_array = weights_df.values
        concentration = np.sum(weights_array**2, axis=1)
        
        fig.add_trace(go.Scatter(
            x=evolution_df.index,
            y=concentration,
            mode='lines',
            name='Concentration Index',
            line=dict(color='red', width=2)
        ), row=2, col=2)
        
        # 5. Rolling returns
        rolling_returns = evolution_df['portfolio_return'].rolling(252).mean() * 252 * 100
        fig.add_trace(go.Scatter(
            x=evolution_df.index,
            y=rolling_returns,
            mode='lines',
            name='Rolling Annual Return %',
            line=dict(color='darkgreen', width=2)
        ), row=3, col=1)
        
        # 6. Rolling volatility
        rolling_vol = evolution_df['portfolio_return'].rolling(252).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(
            x=evolution_df.index,
            y=rolling_vol,
            mode='lines',
            name='Rolling Volatility %',
            line=dict(color='orange', width=2)
        ), row=3, col=2)
        
        fig.update_layout(
            title='Comprehensive Portfolio Evolution Dashboard',
            height=1000,
            width=1400,
            showlegend=True
        )
        
        return fig


def main():
    """Run portfolio evolution analysis."""
    print("🚀 PORTFOLIO EVOLUTION ANALYSIS")
    print("="*60)
    
    # Load data
    data_fetcher = MarketDataFetcher(period_years=20)
    symbols = data_fetcher.get_top_assets_by_volume(num_assets=50)
    prices_data = data_fetcher.fetch_historical_data(symbols)
    returns_data = data_fetcher.calculate_returns()
    
    print(f"✅ Loaded data: {len(returns_data.columns)} assets, {len(returns_data)} days")
    
    # Initialize tracker
    tracker = PortfolioEvolutionTracker(returns_data, prices_data)
    
    # Create initial equal-weight portfolio
    initial_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
    
    # Simulate different strategies
    strategies = ['momentum', 'volatility', 'mean_reversion']
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 Analyzing {strategy} strategy...")
        evolution_df = tracker.simulate_portfolio_evolution(
            initial_weights, 
            rebalance_frequency=21,
            strategy=strategy
        )
        results[strategy] = evolution_df
        
        # Create visualizations
        print(f"🎨 Creating visualizations for {strategy} strategy...")
        
        # Composition heatmap
        heatmap_fig = tracker.create_composition_heatmap(evolution_df)
        heatmap_fig.write_html(f"results/plots/composition_heatmap_{strategy}.html")
        
        # Weight evolution
        weight_fig = tracker.create_weight_evolution_chart(evolution_df)
        weight_fig.write_html(f"results/plots/weight_evolution_{strategy}.html")
        
        # Comprehensive dashboard
        dashboard_fig = tracker.create_comprehensive_dashboard(evolution_df)
        dashboard_fig.write_html(f"results/plots/portfolio_dashboard_{strategy}.html")
        
        # Turnover analysis
        turnover_fig = tracker.create_turnover_analysis()
        if turnover_fig:
            turnover_fig.write_html(f"results/plots/turnover_analysis_{strategy}.html")
        
        print(f"✅ {strategy.title()} strategy analysis completed")
    
    # Strategy comparison
    print(f"\n📈 STRATEGY COMPARISON")
    print("="*60)
    
    for strategy, df in results.items():
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value / 100000 - 1) * 100
        annual_return = df['portfolio_return'].mean() * 252 * 100
        volatility = df['portfolio_return'].std() * np.sqrt(252) * 100
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        print(f"{strategy.title()} Strategy:")
        print(f"  Final Value: ${final_value:,.0f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Annual Return: {annual_return:.2f}%")
        print(f"  Volatility: {volatility:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print()
    
    print("📁 All visualizations saved to results/plots/")
    print("🎉 Portfolio evolution analysis completed!")


if __name__ == "__main__":
    main()
