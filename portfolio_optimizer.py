"""
Portfolio optimization engine using Modern Portfolio Theory (MPT).
Implements efficient frontier construction with strict performance constraints.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from portfolio_metrics import PortfolioMetrics
warnings.filterwarnings('ignore')


@dataclass
class OptimizationConstraints:
    """Container for portfolio optimization constraints."""
    min_return: float = 0.20  # Minimum 20% annualized return
    min_sharpe: float = 2.0   # Minimum Sharpe ratio of 2
    max_drawdown: float = 0.03  # Maximum 3% drawdown
    min_weight: float = 0.0   # Minimum asset weight
    max_weight: float = 1.0   # Maximum asset weight
    max_assets: Optional[int] = None  # Maximum number of assets in portfolio


class EfficientFrontierOptimizer:
    """
    Efficient frontier portfolio optimizer with advanced constraints.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the optimizer.
        
        Args:
            returns: Daily returns DataFrame
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
        
        # Calculate expected returns and covariance matrix
        self.expected_returns = returns.mean() * self.trading_days
        self.cov_matrix = returns.cov() * self.trading_days
        self.n_assets = len(returns.columns)
        
        # Initialize metrics calculator
        self.metrics_calc = PortfolioMetrics(risk_free_rate)
        
        # Storage for optimization results
        self.efficient_portfolios = []
        self.feasible_portfolios = []
        
    def _estimate_expected_returns(self, method: str = 'historical') -> np.ndarray:
        """
        Estimate expected returns using different methods.
        
        Args:
            method: 'historical', 'capm', or 'shrinkage'
            
        Returns:
            Expected returns array
        """
        if method == 'historical':
            return self.expected_returns.values
        
        elif method == 'shrinkage':
            # James-Stein shrinkage estimator
            market_return = self.expected_returns.mean()
            shrinkage_factor = 0.2
            shrunk_returns = (1 - shrinkage_factor) * self.expected_returns + shrinkage_factor * market_return
            return shrunk_returns.values
        
        else:
            return self.expected_returns.values
    
    def optimize_portfolio(self, target_return: Optional[float] = None, 
                          target_risk: Optional[float] = None,
                          constraints: OptimizationConstraints = None) -> Dict:
        """
        Optimize portfolio for given target return or risk.
        
        Args:
            target_return: Target annualized return
            target_risk: Target annualized volatility
            constraints: Optimization constraints
            
        Returns:
            Dictionary with optimization results
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Set up optimization variables
        w = cp.Variable(self.n_assets)
        
        # Expected returns and covariance
        mu = self._estimate_expected_returns()
        Sigma = self.cov_matrix.values
        
        # Portfolio return and risk
        portfolio_return = mu.T @ w
        portfolio_risk = cp.quad_form(w, Sigma)
        
        # Base constraints
        constraints_list = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= constraints.min_weight,  # Minimum weight
            w <= constraints.max_weight   # Maximum weight
        ]
        
        # Target constraints
        if target_return is not None:
            constraints_list.append(portfolio_return >= target_return)
        
        if target_risk is not None:
            constraints_list.append(portfolio_risk <= target_risk**2)
        
        # Cardinality constraint (maximum number of assets)
        if constraints.max_assets is not None:
            # Use binary variables for asset selection
            z = cp.Variable(self.n_assets, boolean=True)
            constraints_list.extend([
                w <= constraints.max_weight * z,
                w >= constraints.min_weight * z,
                cp.sum(z) <= constraints.max_assets
            ])
        
        # Objective function
        if target_return is not None:
            # Minimize risk for given return
            objective = cp.Minimize(portfolio_risk)
        elif target_risk is not None:
            # Maximize return for given risk
            objective = cp.Maximize(portfolio_return)
        else:
            # Maximize Sharpe ratio (approximate)
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights = w.value
                
                # Calculate portfolio metrics
                portfolio_metrics = self.metrics_calc.calculate_all_metrics(weights, self.returns)
                
                # Check if portfolio meets all constraints
                constraint_check = self.metrics_calc.check_constraints(
                    portfolio_metrics, 
                    constraints.min_return,
                    constraints.min_sharpe,
                    constraints.max_drawdown
                )
                
                return {
                    'weights': weights,
                    'metrics': portfolio_metrics,
                    'constraints_met': constraint_check,
                    'optimization_status': problem.status,
                    'objective_value': problem.value
                }
            else:
                return {
                    'weights': None,
                    'metrics': None,
                    'constraints_met': None,
                    'optimization_status': problem.status,
                    'objective_value': None
                }
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {
                'weights': None,
                'metrics': None,
                'constraints_met': None,
                'optimization_status': 'error',
                'objective_value': None,
                'error': str(e)
            }
    
    def generate_efficient_frontier(self, n_portfolios: int = 100, 
                                   constraints: OptimizationConstraints = None) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Args:
            n_portfolios: Number of portfolios to generate
            constraints: Optimization constraints
            
        Returns:
            DataFrame with efficient frontier portfolios
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Define return range
        min_return = max(self.expected_returns.min(), 0.05)  # At least 5%
        max_return = min(self.expected_returns.max() * 0.8, 0.50)  # Cap at 50%
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        efficient_portfolios = []
        
        print(f"Generating efficient frontier with {n_portfolios} portfolios...")
        
        for i, target_return in enumerate(target_returns):
            if i % 20 == 0:
                print(f"Progress: {i}/{n_portfolios} portfolios")
            
            result = self.optimize_portfolio(target_return=target_return, constraints=constraints)
            
            if result['weights'] is not None:
                portfolio_data = {
                    'target_return': target_return,
                    'weights': result['weights'],
                    **result['metrics'],
                    'constraints_met': result['constraints_met']['all_constraints']
                }
                efficient_portfolios.append(portfolio_data)
        
        if not efficient_portfolios:
            print("Warning: No feasible portfolios found!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        frontier_df = pd.DataFrame(efficient_portfolios)
        
        # Store results
        self.efficient_portfolios = efficient_portfolios
        self.feasible_portfolios = [p for p in efficient_portfolios if p['constraints_met']]
        
        print(f"Generated {len(efficient_portfolios)} efficient portfolios")
        print(f"Found {len(self.feasible_portfolios)} portfolios meeting all constraints")
        
        return frontier_df
    
    def find_optimal_portfolios(self, constraints: OptimizationConstraints = None) -> Dict[str, Dict]:
        """
        Find key optimal portfolios (max Sharpe, min volatility, etc.).
        
        Args:
            constraints: Optimization constraints
            
        Returns:
            Dictionary of optimal portfolios
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        optimal_portfolios = {}
        
        # Maximum Sharpe ratio portfolio
        print("Finding maximum Sharpe ratio portfolio...")
        max_sharpe_result = self._optimize_max_sharpe(constraints)
        if max_sharpe_result['weights'] is not None:
            optimal_portfolios['max_sharpe'] = max_sharpe_result
        
        # Minimum volatility portfolio
        print("Finding minimum volatility portfolio...")
        min_vol_result = self.optimize_portfolio(constraints=constraints)
        if min_vol_result['weights'] is not None:
            optimal_portfolios['min_volatility'] = min_vol_result
        
        # Maximum return portfolio (subject to constraints)
        print("Finding maximum return portfolio...")
        max_return_result = self._optimize_max_return(constraints)
        if max_return_result['weights'] is not None:
            optimal_portfolios['max_return'] = max_return_result
        
        # Portfolio meeting exact constraints
        print("Finding portfolio meeting exact constraints...")
        constrained_result = self._optimize_constrained_portfolio(constraints)
        if constrained_result['weights'] is not None:
            optimal_portfolios['constrained'] = constrained_result
        
        return optimal_portfolios
    
    def _optimize_max_sharpe(self, constraints: OptimizationConstraints) -> Dict:
        """Optimize for maximum Sharpe ratio."""
        # Use scipy optimization for better Sharpe ratio optimization
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((constraints.min_weight, constraints.max_weight) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            weights = result.x
            metrics = self.metrics_calc.calculate_all_metrics(weights, self.returns)
            constraint_check = self.metrics_calc.check_constraints(
                metrics, constraints.min_return, constraints.min_sharpe, constraints.max_drawdown
            )
            
            return {
                'weights': weights,
                'metrics': metrics,
                'constraints_met': constraint_check,
                'optimization_status': 'optimal',
                'objective_value': -result.fun
            }
        else:
            return {
                'weights': None,
                'metrics': None,
                'constraints_met': None,
                'optimization_status': 'failed',
                'objective_value': None
            }
    
    def _optimize_max_return(self, constraints: OptimizationConstraints) -> Dict:
        """Optimize for maximum return subject to constraints."""
        # Set up optimization
        w = cp.Variable(self.n_assets)
        mu = self._estimate_expected_returns()
        
        # Objective: maximize return
        objective = cp.Maximize(mu.T @ w)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight,
        ]
        
        # Add risk constraint if needed (to prevent extreme portfolios)
        max_vol = 0.4  # Maximum 40% volatility
        Sigma = self.cov_matrix.values
        constraints_list.append(cp.quad_form(w, Sigma) <= max_vol**2)
        
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                weights = w.value
                metrics = self.metrics_calc.calculate_all_metrics(weights, self.returns)
                constraint_check = self.metrics_calc.check_constraints(
                    metrics, constraints.min_return, constraints.min_sharpe, constraints.max_drawdown
                )
                
                return {
                    'weights': weights,
                    'metrics': metrics,
                    'constraints_met': constraint_check,
                    'optimization_status': problem.status,
                    'objective_value': problem.value
                }
            else:
                return {'weights': None, 'metrics': None, 'constraints_met': None, 
                       'optimization_status': problem.status, 'objective_value': None}
        except:
            return {'weights': None, 'metrics': None, 'constraints_met': None, 
                   'optimization_status': 'error', 'objective_value': None}
    
    def _optimize_constrained_portfolio(self, constraints: OptimizationConstraints) -> Dict:
        """Find portfolio that exactly meets the specified constraints."""
        # This is a challenging problem - we'll use a multi-objective approach
        
        # Try different target returns around the minimum required
        target_returns = np.linspace(constraints.min_return, constraints.min_return * 1.5, 20)
        
        best_portfolio = None
        best_score = -np.inf
        
        for target_return in target_returns:
            result = self.optimize_portfolio(target_return=target_return, constraints=constraints)
            
            if result['weights'] is not None and result['constraints_met']['all_constraints']:
                # Score based on how well it meets all constraints
                metrics = result['metrics']
                score = (
                    min(metrics['sharpe_ratio'] / constraints.min_sharpe, 2.0) +
                    min(metrics['annualized_return'] / constraints.min_return, 2.0) +
                    min(constraints.max_drawdown / max(metrics['max_drawdown'], 0.001), 2.0)
                )
                
                if score > best_score:
                    best_score = score
                    best_portfolio = result
        
        return best_portfolio if best_portfolio else {
            'weights': None, 'metrics': None, 'constraints_met': None,
            'optimization_status': 'no_feasible_solution', 'objective_value': None
        }
    
    def get_portfolio_composition(self, weights: np.ndarray, top_n: int = 10) -> pd.DataFrame:
        """
        Get portfolio composition with asset weights.
        
        Args:
            weights: Portfolio weights
            top_n: Number of top holdings to show
            
        Returns:
            DataFrame with portfolio composition
        """
        composition = pd.DataFrame({
            'Symbol': self.returns.columns,
            'Weight': weights,
            'Weight_Pct': weights * 100
        })
        
        composition = composition[composition['Weight'] > 0.001]  # Filter out tiny weights
        composition = composition.sort_values('Weight', ascending=False)
        
        return composition.head(top_n)
    
    def backtest_portfolio(self, weights: np.ndarray, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict:
        """
        Backtest portfolio performance.
        
        Args:
            weights: Portfolio weights
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results
        """
        # Use subset of data for backtesting if dates specified
        returns_data = self.returns
        if start_date or end_date:
            returns_data = self.returns.loc[start_date:end_date]
        
        # Calculate portfolio returns
        portfolio_returns = returns_data.dot(weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(weights, returns_data)
        
        return {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'metrics': metrics,
            'start_date': returns_data.index[0],
            'end_date': returns_data.index[-1]
        }


def main():
    """Example usage of EfficientFrontierOptimizer."""
    from data_fetcher import MarketDataFetcher
    
    # Fetch data
    print("Fetching market data...")
    fetcher = MarketDataFetcher(period_years=3)
    symbols = fetcher.get_top_assets_by_volume(num_assets=20)  # Start with 20 assets
    prices = fetcher.fetch_historical_data(symbols)
    returns = fetcher.calculate_returns()
    
    # Initialize optimizer
    optimizer = EfficientFrontierOptimizer(returns)
    
    # Set constraints
    constraints = OptimizationConstraints(
        min_return=0.15,  # Slightly lower for testing
        min_sharpe=1.5,   # Slightly lower for testing
        max_drawdown=0.05  # Slightly higher for testing
    )
    
    # Find optimal portfolios
    print("\nFinding optimal portfolios...")
    optimal_portfolios = optimizer.find_optimal_portfolios(constraints)
    
    # Display results
    for name, portfolio in optimal_portfolios.items():
        if portfolio['weights'] is not None:
            print(f"\n{name.upper()} PORTFOLIO:")
            print(f"Return: {portfolio['metrics']['annualized_return']:.2%}")
            print(f"Volatility: {portfolio['metrics']['annualized_volatility']:.2%}")
            print(f"Sharpe Ratio: {portfolio['metrics']['sharpe_ratio']:.3f}")
            print(f"Max Drawdown: {portfolio['metrics']['max_drawdown']:.2%}")
            print(f"Constraints Met: {portfolio['constraints_met']['all_constraints']}")
            
            # Show top holdings
            composition = optimizer.get_portfolio_composition(portfolio['weights'], top_n=5)
            print("Top Holdings:")
            for _, row in composition.iterrows():
                print(f"  {row['Symbol']}: {row['Weight_Pct']:.1f}%")


if __name__ == "__main__":
    main()
