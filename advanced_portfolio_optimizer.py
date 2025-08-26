"""
Advanced Portfolio Optimization System with Multi-Objective Optimization,
Regime Detection, Dynamic Rebalancing, and Enhanced Risk Management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from tqdm import tqdm
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import seaborn as sns
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from portfolio_metrics import PortfolioMetrics


class RegimeDetector:
    """Market regime detection using Gaussian Mixture Models."""
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
        """
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        self.regime_probs = None
        
    def detect_regimes(self, returns: pd.DataFrame, 
                      features: Optional[List[str]] = None) -> pd.Series:
        """
        Detect market regimes based on return characteristics.
        
        Args:
            returns: Returns DataFrame
            features: List of features to use for regime detection
            
        Returns:
            Series with regime labels
        """
        print("🔍 Detecting market regimes...")
        
        # Calculate features for regime detection
        window = 63  # 3-month rolling window
        
        feature_data = pd.DataFrame(index=returns.index)
        
        # Market-level features
        market_returns = returns.mean(axis=1)
        feature_data['market_return'] = market_returns.rolling(window).mean()
        feature_data['market_volatility'] = market_returns.rolling(window).std()
        feature_data['market_skewness'] = market_returns.rolling(window).skew()
        feature_data['market_kurtosis'] = market_returns.rolling(window).kurt()
        
        # Cross-sectional features
        feature_data['cross_sectional_vol'] = returns.std(axis=1).rolling(window).mean()
        feature_data['correlation_mean'] = returns.rolling(window).corr().mean().mean()
        
        # VIX proxy (market stress indicator)
        feature_data['stress_indicator'] = market_returns.rolling(window).std() * np.sqrt(252)
        
        # Momentum features
        feature_data['momentum_1m'] = market_returns.rolling(21).sum()
        feature_data['momentum_3m'] = market_returns.rolling(63).sum()
        
        # Drop NaN values
        feature_data = feature_data.dropna()
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(feature_data)
        
        # Fit Gaussian Mixture Model
        self.model = GaussianMixture(n_components=self.n_regimes, 
                                   covariance_type='full',
                                   random_state=42)
        
        regime_labels = self.model.fit_predict(features_scaled)
        regime_probs = self.model.predict_proba(features_scaled)
        
        # Create regime series
        self.regime_labels = pd.Series(regime_labels, 
                                     index=feature_data.index, 
                                     name='regime')
        
        self.regime_probs = pd.DataFrame(regime_probs,
                                       index=feature_data.index,
                                       columns=[f'regime_{i}' for i in range(self.n_regimes)])
        
        # Label regimes based on characteristics
        regime_stats = []
        for regime in range(self.n_regimes):
            mask = self.regime_labels == regime
            regime_data = feature_data[mask]
            
            stats = {
                'regime': regime,
                'avg_return': regime_data['market_return'].mean(),
                'avg_volatility': regime_data['market_volatility'].mean(),
                'avg_stress': regime_data['stress_indicator'].mean(),
                'periods': mask.sum()
            }
            regime_stats.append(stats)
        
        regime_df = pd.DataFrame(regime_stats)
        
        print(f"✅ Detected {self.n_regimes} market regimes:")
        for _, row in regime_df.iterrows():
            regime_type = self._classify_regime(row)
            print(f"   Regime {int(row['regime'])}: {regime_type}")
            print(f"     Return: {row['avg_return']:.4f}, Vol: {row['avg_volatility']:.4f}")
            print(f"     Periods: {int(row['periods'])}")
        
        return self.regime_labels
    
    def _classify_regime(self, regime_stats: pd.Series) -> str:
        """Classify regime based on statistics."""
        if regime_stats['avg_return'] > 0.001 and regime_stats['avg_volatility'] < 0.015:
            return "Bull Market (Low Vol)"
        elif regime_stats['avg_return'] > 0.0005 and regime_stats['avg_volatility'] > 0.02:
            return "Bull Market (High Vol)"
        elif regime_stats['avg_return'] < -0.0005:
            return "Bear Market"
        else:
            return "Neutral/Sideways"


class MultiObjectiveOptimizer:
    """Multi-objective portfolio optimization with advanced constraints."""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize multi-objective optimizer.
        
        Args:
            returns: Returns DataFrame
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.metrics_calc = PortfolioMetrics(risk_free_rate=risk_free_rate)
        
        # Pre-calculate statistics
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
    def optimize_multi_objective(self, 
                                objectives: Dict[str, float],
                                constraints: Dict[str, float],
                                method: str = 'weighted_sum') -> Dict:
        """
        Multi-objective portfolio optimization.
        
        Args:
            objectives: Dictionary of objective weights
            constraints: Dictionary of constraint values
            method: Optimization method ('weighted_sum', 'epsilon_constraint')
            
        Returns:
            Optimization results
        """
        if method == 'weighted_sum':
            return self._weighted_sum_optimization(objectives, constraints)
        elif method == 'epsilon_constraint':
            return self._epsilon_constraint_optimization(objectives, constraints)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_sum_optimization(self, 
                                 objectives: Dict[str, float],
                                 constraints: Dict[str, float]) -> Dict:
        """Weighted sum multi-objective optimization."""
        
        def objective_function(weights):
            """Combined objective function."""
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            
            # Calculate portfolio returns for drawdown
            portfolio_returns = self.returns.dot(weights)
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdowns.min())
            
            # Calculate CVaR
            var_5 = np.percentile(portfolio_returns, 5)
            cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean()
            
            # Sharpe ratio
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Multi-objective function (minimize negative of weighted sum)
            objective = 0
            if 'return' in objectives:
                objective += objectives['return'] * portfolio_return
            if 'sharpe' in objectives:
                objective += objectives['sharpe'] * sharpe
            if 'volatility' in objectives:
                objective -= objectives['volatility'] * portfolio_vol  # Minimize volatility
            if 'drawdown' in objectives:
                objective -= objectives['drawdown'] * max_drawdown  # Minimize drawdown
            if 'cvar' in objectives:
                objective -= objectives['cvar'] * abs(cvar_5)  # Minimize CVaR
            
            return -objective  # Minimize negative
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Add specific constraints
        if 'min_return' in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: np.dot(x, self.mean_returns) - constraints['min_return']
            })
        
        if 'max_volatility' in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: constraints['max_volatility'] - np.sqrt(np.dot(x, np.dot(self.cov_matrix, x)))
            })
        
        # Bounds
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective_function, x0, method='SLSQP', 
                         bounds=bounds, constraints=cons,
                         options={'maxiter': 1000})
        
        if result.success:
            weights = result.x
            metrics = self.metrics_calc.calculate_all_metrics(weights, self.returns)
            
            return {
                'weights': weights,
                'metrics': metrics,
                'optimization_status': 'optimal',
                'objective_value': -result.fun
            }
        else:
            return {
                'weights': None,
                'metrics': None,
                'optimization_status': 'failed',
                'message': result.message
            }
    
    def generate_pareto_frontier(self, 
                               n_points: int = 50,
                               constraints: Dict[str, float] = None) -> List[Dict]:
        """
        Generate Pareto frontier for return vs risk trade-off.
        
        Args:
            n_points: Number of points on frontier
            constraints: Additional constraints
            
        Returns:
            List of portfolio results on Pareto frontier
        """
        print(f"🎯 Generating Pareto frontier with {n_points} points...")
        
        if constraints is None:
            constraints = {'min_weight': 0.0, 'max_weight': 0.2}
        
        # Generate target returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max() * 0.8  # Conservative upper bound
        target_returns = np.linspace(min_return, max_return, n_points)
        
        pareto_portfolios = []
        
        for target_return in tqdm(target_returns, desc="Generating Pareto frontier"):
            # Minimize risk for given return
            objectives = {'volatility': 1.0, 'drawdown': 0.5}
            current_constraints = constraints.copy()
            current_constraints['min_return'] = target_return
            
            result = self.optimize_multi_objective(objectives, current_constraints)
            
            if result['optimization_status'] == 'optimal':
                result['target_return'] = target_return
                pareto_portfolios.append(result)
        
        print(f"✅ Generated {len(pareto_portfolios)} efficient portfolios")
        return pareto_portfolios


class DynamicRebalancer:
    """Dynamic portfolio rebalancing strategies."""
    
    def __init__(self, returns: pd.DataFrame, initial_weights: np.ndarray):
        """
        Initialize dynamic rebalancer.
        
        Args:
            returns: Returns DataFrame
            initial_weights: Initial portfolio weights
        """
        self.returns = returns
        self.initial_weights = initial_weights
        self.rebalancing_history = []
        
    def momentum_rebalancing(self, 
                           lookback_period: int = 63,
                           rebalance_frequency: int = 21,
                           momentum_threshold: float = 0.1) -> pd.DataFrame:
        """
        Momentum-based rebalancing strategy.
        
        Args:
            lookback_period: Period for momentum calculation
            rebalance_frequency: Rebalancing frequency in days
            momentum_threshold: Threshold for momentum signal
            
        Returns:
            DataFrame with rebalancing history
        """
        print(f"📈 Implementing momentum rebalancing strategy...")
        
        rebalance_dates = self.returns.index[::rebalance_frequency]
        weights_history = []
        
        current_weights = self.initial_weights.copy()
        
        for date in rebalance_dates:
            if date not in self.returns.index:
                continue
                
            # Get position in returns
            date_pos = self.returns.index.get_loc(date)
            
            if date_pos < lookback_period:
                weights_history.append({
                    'date': date,
                    'weights': current_weights.copy(),
                    'reason': 'insufficient_history'
                })
                continue
            
            # Calculate momentum scores
            lookback_returns = self.returns.iloc[date_pos-lookback_period:date_pos]
            momentum_scores = lookback_returns.mean() * 252  # Annualized
            
            # Rank assets by momentum
            momentum_ranks = momentum_scores.rank(ascending=False, pct=True)
            
            # Adjust weights based on momentum
            momentum_adjustment = (momentum_ranks - 0.5) * momentum_threshold
            new_weights = current_weights * (1 + momentum_adjustment)
            
            # Normalize weights
            new_weights = np.maximum(new_weights, 0.01)  # Minimum weight
            new_weights = np.minimum(new_weights, 0.25)  # Maximum weight
            new_weights = new_weights / new_weights.sum()
            
            weights_history.append({
                'date': date,
                'weights': new_weights.copy(),
                'momentum_scores': momentum_scores.copy(),
                'reason': 'momentum_rebalance'
            })
            
            current_weights = new_weights
        
        print(f"✅ Completed momentum rebalancing with {len(weights_history)} rebalances")
        return pd.DataFrame(weights_history)
    
    def volatility_targeting(self, 
                           target_volatility: float = 0.15,
                           rebalance_frequency: int = 21,
                           lookback_period: int = 63) -> pd.DataFrame:
        """
        Volatility targeting rebalancing strategy.
        
        Args:
            target_volatility: Target portfolio volatility
            rebalance_frequency: Rebalancing frequency in days
            lookback_period: Period for volatility estimation
            
        Returns:
            DataFrame with rebalancing history
        """
        print(f"🎯 Implementing volatility targeting strategy...")
        
        rebalance_dates = self.returns.index[::rebalance_frequency]
        weights_history = []
        
        current_weights = self.initial_weights.copy()
        
        for date in rebalance_dates:
            if date not in self.returns.index:
                continue
                
            date_pos = self.returns.index.get_loc(date)
            
            if date_pos < lookback_period:
                weights_history.append({
                    'date': date,
                    'weights': current_weights.copy(),
                    'portfolio_vol': np.nan,
                    'vol_scalar': 1.0,
                    'reason': 'insufficient_history'
                })
                continue
            
            # Calculate current portfolio volatility
            lookback_returns = self.returns.iloc[date_pos-lookback_period:date_pos]
            portfolio_returns = lookback_returns.dot(current_weights)
            current_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Calculate volatility scalar
            vol_scalar = target_volatility / current_vol if current_vol > 0 else 1.0
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit scaling
            
            # Scale weights (this is simplified - in practice you'd adjust position sizes)
            # Here we adjust by changing concentration
            if vol_scalar < 1.0:  # Reduce risk - diversify more
                new_weights = current_weights * 0.8 + np.ones(len(current_weights)) / len(current_weights) * 0.2
            elif vol_scalar > 1.0:  # Increase risk - concentrate more
                top_performers = current_weights.argsort()[-10:]  # Top 10 assets
                new_weights = current_weights.copy()
                new_weights[top_performers] *= 1.2
            else:
                new_weights = current_weights.copy()
            
            # Normalize weights
            new_weights = np.maximum(new_weights, 0.005)
            new_weights = new_weights / new_weights.sum()
            
            weights_history.append({
                'date': date,
                'weights': new_weights.copy(),
                'portfolio_vol': current_vol,
                'vol_scalar': vol_scalar,
                'reason': 'volatility_targeting'
            })
            
            current_weights = new_weights
        
        print(f"✅ Completed volatility targeting with {len(weights_history)} rebalances")
        return pd.DataFrame(weights_history)


class AdvancedPortfolioAnalysis:
    """Advanced portfolio analysis combining all optimization techniques."""
    
    def __init__(self, initial_investment: float = 100000.0):
        """
        Initialize advanced portfolio analysis.
        
        Args:
            initial_investment: Initial portfolio value
        """
        self.initial_investment = initial_investment
        self.data_fetcher = None
        self.returns_data = None
        self.prices_data = None
        self.regime_detector = RegimeDetector()
        self.multi_optimizer = None
        self.results = {}
        
    def run_comprehensive_analysis(self, period_years: int = 20) -> Dict:
        """
        Run comprehensive advanced portfolio analysis.
        
        Args:
            period_years: Years of historical data
            
        Returns:
            Complete analysis results
        """
        print("🚀 ADVANCED PORTFOLIO OPTIMIZATION ANALYSIS")
        print("="*80)
        print("Features: Multi-Objective | Regime Detection | Dynamic Rebalancing")
        print("="*80)
        
        start_time = datetime.now()
        
        # Step 1: Fetch data
        print("\n📊 Step 1: Data Collection")
        self.data_fetcher = MarketDataFetcher(period_years=period_years)
        symbols = self.data_fetcher.get_top_assets_by_volume(num_assets=80)
        self.prices_data = self.data_fetcher.fetch_historical_data(symbols)
        self.returns_data = self.data_fetcher.calculate_returns()
        
        print(f"✅ Loaded {len(self.returns_data.columns)} assets, {len(self.returns_data)} days")
        
        # Step 2: Regime Detection
        print("\n🔍 Step 2: Market Regime Detection")
        regimes = self.regime_detector.detect_regimes(self.returns_data)
        
        # Step 3: Multi-Objective Optimization
        print("\n🎯 Step 3: Multi-Objective Portfolio Optimization")
        self.multi_optimizer = MultiObjectiveOptimizer(self.returns_data)
        
        # Define multiple optimization scenarios
        optimization_scenarios = {
            'balanced': {
                'objectives': {'return': 0.4, 'sharpe': 0.3, 'volatility': 0.2, 'drawdown': 0.1},
                'constraints': {'min_weight': 0.0, 'max_weight': 0.15, 'min_return': 0.10}
            },
            'growth': {
                'objectives': {'return': 0.6, 'sharpe': 0.2, 'volatility': 0.1, 'drawdown': 0.1},
                'constraints': {'min_weight': 0.0, 'max_weight': 0.20, 'min_return': 0.15}
            },
            'conservative': {
                'objectives': {'return': 0.2, 'sharpe': 0.3, 'volatility': 0.3, 'drawdown': 0.2},
                'constraints': {'min_weight': 0.0, 'max_weight': 0.10, 'max_volatility': 0.12}
            },
            'risk_parity': {
                'objectives': {'sharpe': 0.5, 'volatility': 0.3, 'drawdown': 0.2},
                'constraints': {'min_weight': 0.005, 'max_weight': 0.08}
            }
        }
        
        optimized_portfolios = {}
        for scenario_name, scenario_config in optimization_scenarios.items():
            print(f"   Optimizing {scenario_name} portfolio...")
            result = self.multi_optimizer.optimize_multi_objective(
                scenario_config['objectives'],
                scenario_config['constraints']
            )
            optimized_portfolios[scenario_name] = result
        
        # Step 4: Generate Pareto Frontier
        print("\n📈 Step 4: Generating Pareto Frontier")
        pareto_portfolios = self.multi_optimizer.generate_pareto_frontier(
            n_points=30,
            constraints={'min_weight': 0.0, 'max_weight': 0.15}
        )
        
        # Step 5: Dynamic Rebalancing Analysis
        print("\n🔄 Step 5: Dynamic Rebalancing Strategies")
        best_portfolio = max(optimized_portfolios.values(), 
                           key=lambda x: x['metrics']['sharpe_ratio'] if x['metrics'] else 0)
        
        if best_portfolio['weights'] is not None:
            rebalancer = DynamicRebalancer(self.returns_data, best_portfolio['weights'])
            
            momentum_strategy = rebalancer.momentum_rebalancing()
            volatility_strategy = rebalancer.volatility_targeting()
        else:
            momentum_strategy = None
            volatility_strategy = None
        
        # Step 6: Performance Analysis
        print("\n📊 Step 6: Performance Analysis & Visualization")
        self._create_advanced_visualizations(
            optimized_portfolios, pareto_portfolios, regimes,
            momentum_strategy, volatility_strategy
        )
        
        # Compile results
        end_time = datetime.now()
        
        self.results = {
            'optimized_portfolios': optimized_portfolios,
            'pareto_portfolios': pareto_portfolios,
            'regimes': regimes,
            'momentum_strategy': momentum_strategy,
            'volatility_strategy': volatility_strategy,
            'n_assets': len(self.returns_data.columns),
            'analysis_period': (self.prices_data.index[0], self.prices_data.index[-1]),
            'runtime': end_time - start_time
        }
        
        # Summary
        print(f"\n🎉 ADVANCED ANALYSIS COMPLETED")
        print("="*80)
        print(f"Runtime: {self.results['runtime']}")
        print(f"Assets: {self.results['n_assets']}")
        print(f"Regimes Detected: {len(regimes.unique())}")
        print(f"Pareto Portfolios: {len(pareto_portfolios)}")
        
        # Best portfolio summary
        best_scenario = max(optimized_portfolios.keys(), 
                          key=lambda x: optimized_portfolios[x]['metrics']['sharpe_ratio'] 
                          if optimized_portfolios[x]['metrics'] else 0)
        
        best_metrics = optimized_portfolios[best_scenario]['metrics']
        if best_metrics:
            print(f"\n🏆 Best Portfolio ({best_scenario}):")
            print(f"   Return: {best_metrics['annualized_return']:.2%}")
            print(f"   Sharpe: {best_metrics['sharpe_ratio']:.3f}")
            print(f"   Max DD: {best_metrics['max_drawdown']:.2%}")
        
        return self.results
    
    def _create_advanced_visualizations(self, optimized_portfolios, pareto_portfolios, 
                                      regimes, momentum_strategy, volatility_strategy):
        """Create advanced visualizations for all analysis components."""
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Multi-Objective Portfolios', 'Pareto Frontier',
                'Market Regimes Over Time', 'Dynamic Rebalancing Performance',
                'Portfolio Composition Comparison', 'Risk-Return Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Multi-objective portfolios
        for scenario, result in optimized_portfolios.items():
            if result['metrics']:
                fig.add_trace(go.Scatter(
                    x=[result['metrics']['annualized_volatility']],
                    y=[result['metrics']['annualized_return']],
                    mode='markers',
                    marker=dict(size=15, symbol='star'),
                    name=f'{scenario.title()}',
                    text=f"Sharpe: {result['metrics']['sharpe_ratio']:.3f}"
                ), row=1, col=1)
        
        # Plot 2: Pareto frontier
        if pareto_portfolios:
            pareto_vols = [p['metrics']['annualized_volatility'] for p in pareto_portfolios]
            pareto_rets = [p['metrics']['annualized_return'] for p in pareto_portfolios]
            pareto_sharpes = [p['metrics']['sharpe_ratio'] for p in pareto_portfolios]
            
            fig.add_trace(go.Scatter(
                x=pareto_vols,
                y=pareto_rets,
                mode='markers+lines',
                marker=dict(color=pareto_sharpes, colorscale='Viridis', size=8),
                name='Pareto Frontier',
                line=dict(width=2)
            ), row=1, col=2)
        
        # Plot 3: Market regimes
        regime_colors = ['blue', 'red', 'green', 'orange', 'purple']
        market_returns = self.returns_data.mean(axis=1).cumsum()
        
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_dates = regimes[regime_mask].index
            regime_values = market_returns[regime_dates]
            
            fig.add_trace(go.Scatter(
                x=regime_dates,
                y=regime_values,
                mode='markers',
                marker=dict(color=regime_colors[regime % len(regime_colors)], size=4),
                name=f'Regime {regime}',
                showlegend=True
            ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Advanced Portfolio Optimization Dashboard',
            height=1200,
            width=1400,
            showlegend=True
        )
        
        # Save visualization
        import os
        os.makedirs("results/plots", exist_ok=True)
        fig.write_html("results/plots/advanced_portfolio_analysis.html")
        print("   Advanced analysis dashboard saved to results/plots/advanced_portfolio_analysis.html")
        
        fig.show()


def main():
    """Run advanced portfolio analysis."""
    analysis = AdvancedPortfolioAnalysis(initial_investment=100000)
    results = analysis.run_comprehensive_analysis(period_years=20)
    
    print("\n📁 Analysis complete! Check results/plots/ for visualizations.")


if __name__ == "__main__":
    main()
