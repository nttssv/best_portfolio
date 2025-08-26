"""
Portfolio performance metrics calculation module.
Implements comprehensive risk and return metrics for portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PortfolioMetrics:
    """Calculate comprehensive portfolio performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
    
    def calculate_returns(self, weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns given weights and asset returns.
        
        Args:
            weights: Portfolio weights array
            returns: Asset returns DataFrame
            
        Returns:
            Portfolio returns time series
        """
        return returns.dot(weights)
    
    def annualized_return(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate annualized return."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        total_return = np.prod(1 + returns)
        n_periods = len(returns)
        annualized = (total_return ** (self.trading_days / n_periods)) - 1
        return annualized
    
    def annualized_volatility(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate annualized volatility (standard deviation)."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        return np.std(returns) * np.sqrt(self.trading_days)
    
    def sharpe_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Sharpe ratio."""
        ann_return = self.annualized_return(returns)
        ann_vol = self.annualized_volatility(returns)
        
        if ann_vol == 0:
            return 0
        
        return (ann_return - self.risk_free_rate) / ann_vol
    
    def sortino_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        ann_return = self.annualized_return(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_vol = np.std(downside_returns) * np.sqrt(self.trading_days)
        
        if downside_vol == 0:
            return np.inf
        
        return (ann_return - self.risk_free_rate) / downside_vol
    
    def maximum_drawdown(self, returns: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Returns:
            Dictionary with max_drawdown, drawdown_duration, recovery_time
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = np.min(drawdown)
        
        # Find drawdown periods
        dd_start = None
        dd_end = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:  # In drawdown
                if dd_start is None:
                    dd_start = i
                current_dd_duration += 1
                if dd == max_dd:
                    dd_end = i
            else:  # Out of drawdown
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
                current_dd_duration = 0
                dd_start = None
        
        # Recovery time (days to recover from max drawdown)
        recovery_time = 0
        if dd_end is not None:
            for i in range(dd_end + 1, len(drawdown)):
                if drawdown[i] >= 0:
                    recovery_time = i - dd_end
                    break
            else:
                recovery_time = len(drawdown) - dd_end  # Still in drawdown
        
        return {
            'max_drawdown': abs(max_dd),
            'drawdown_duration': max_dd_duration,
            'recovery_time': recovery_time
        }
    
    def value_at_risk(self, returns: Union[pd.Series, np.ndarray], confidence: float = 0.05) -> float:
        """Calculate Value at Risk (VaR) at given confidence level."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        return np.percentile(returns, confidence * 100)
    
    def conditional_var(self, returns: Union[pd.Series, np.ndarray], confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        var = self.value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def calmar_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        ann_return = self.annualized_return(returns)
        max_dd = self.maximum_drawdown(returns)['max_drawdown']
        
        if max_dd == 0:
            return np.inf
        
        return ann_return / max_dd
    
    def information_ratio(self, portfolio_returns: Union[pd.Series, np.ndarray], 
                         benchmark_returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Information Ratio vs benchmark."""
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days)
        
        if tracking_error == 0:
            return 0
        
        return np.mean(excess_returns) * self.trading_days / tracking_error
    
    def beta(self, portfolio_returns: Union[pd.Series, np.ndarray], 
             market_returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate portfolio beta vs market."""
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.values
        if isinstance(market_returns, pd.Series):
            market_returns = market_returns.values
        
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 0
        
        return covariance / market_variance
    
    def alpha(self, portfolio_returns: Union[pd.Series, np.ndarray], 
              market_returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate portfolio alpha (Jensen's alpha)."""
        portfolio_return = self.annualized_return(portfolio_returns)
        market_return = self.annualized_return(market_returns)
        portfolio_beta = self.beta(portfolio_returns, market_returns)
        
        return portfolio_return - (self.risk_free_rate + portfolio_beta * (market_return - self.risk_free_rate))
    
    def skewness(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate return distribution skewness."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        return stats.skew(returns)
    
    def kurtosis(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate return distribution kurtosis."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        return stats.kurtosis(returns)
    
    def calculate_all_metrics(self, weights: np.ndarray, returns: pd.DataFrame, 
                             benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns DataFrame
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            Dictionary of all calculated metrics
        """
        # Calculate portfolio returns
        portfolio_returns = self.calculate_returns(weights, returns)
        
        # Basic metrics
        metrics = {
            'annualized_return': self.annualized_return(portfolio_returns),
            'annualized_volatility': self.annualized_volatility(portfolio_returns),
            'sharpe_ratio': self.sharpe_ratio(portfolio_returns),
            'sortino_ratio': self.sortino_ratio(portfolio_returns),
            'calmar_ratio': self.calmar_ratio(portfolio_returns),
            'skewness': self.skewness(portfolio_returns),
            'kurtosis': self.kurtosis(portfolio_returns),
            'var_5': self.value_at_risk(portfolio_returns, 0.05),
            'cvar_5': self.conditional_var(portfolio_returns, 0.05),
        }
        
        # Drawdown metrics
        dd_metrics = self.maximum_drawdown(portfolio_returns)
        metrics.update(dd_metrics)
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            metrics.update({
                'information_ratio': self.information_ratio(portfolio_returns, benchmark_returns),
                'beta': self.beta(portfolio_returns, benchmark_returns),
                'alpha': self.alpha(portfolio_returns, benchmark_returns)
            })
        
        return metrics
    
    def check_constraints(self, metrics: Dict[str, float], 
                         min_return: float = 0.20, 
                         min_sharpe: float = 2.0, 
                         max_drawdown: float = 0.03) -> Dict[str, bool]:
        """
        Check if portfolio meets specified constraints.
        
        Args:
            metrics: Portfolio metrics dictionary
            min_return: Minimum required annualized return
            min_sharpe: Minimum required Sharpe ratio
            max_drawdown: Maximum allowed drawdown
            
        Returns:
            Dictionary of constraint satisfaction results
        """
        constraints = {
            'return_constraint': metrics['annualized_return'] >= min_return,
            'sharpe_constraint': metrics['sharpe_ratio'] >= min_sharpe,
            'drawdown_constraint': metrics['max_drawdown'] <= max_drawdown,
        }
        
        constraints['all_constraints'] = all(constraints.values())
        
        return constraints
    
    def portfolio_summary(self, weights: np.ndarray, returns: pd.DataFrame, 
                         symbols: list, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate a comprehensive portfolio summary.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns DataFrame
            symbols: Asset symbols
            benchmark_returns: Optional benchmark returns
            
        Returns:
            DataFrame with portfolio summary
        """
        # Calculate metrics
        metrics = self.calculate_all_metrics(weights, returns, benchmark_returns)
        
        # Check constraints
        constraints = self.check_constraints(metrics)
        
        # Create summary DataFrame
        summary_data = []
        
        # Performance metrics
        summary_data.extend([
            ['Annualized Return', f"{metrics['annualized_return']:.2%}"],
            ['Annualized Volatility', f"{metrics['annualized_volatility']:.2%}"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
            ['Sortino Ratio', f"{metrics['sortino_ratio']:.3f}"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.3f}"],
        ])
        
        # Risk metrics
        summary_data.extend([
            ['Maximum Drawdown', f"{metrics['max_drawdown']:.2%}"],
            ['VaR (5%)', f"{metrics['var_5']:.2%}"],
            ['CVaR (5%)', f"{metrics['cvar_5']:.2%}"],
            ['Skewness', f"{metrics['skewness']:.3f}"],
            ['Kurtosis', f"{metrics['kurtosis']:.3f}"],
        ])
        
        # Constraint satisfaction
        summary_data.extend([
            ['Return Constraint (≥20%)', '✓' if constraints['return_constraint'] else '✗'],
            ['Sharpe Constraint (≥2.0)', '✓' if constraints['sharpe_constraint'] else '✗'],
            ['Drawdown Constraint (≤3%)', '✓' if constraints['drawdown_constraint'] else '✗'],
            ['All Constraints Met', '✓' if constraints['all_constraints'] else '✗'],
        ])
        
        # Benchmark metrics if available
        if benchmark_returns is not None:
            summary_data.extend([
                ['Information Ratio', f"{metrics['information_ratio']:.3f}"],
                ['Beta', f"{metrics['beta']:.3f}"],
                ['Alpha', f"{metrics['alpha']:.2%}"],
            ])
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        
        return summary_df


def main():
    """Example usage of PortfolioMetrics."""
    # Generate sample data
    np.random.seed(42)
    n_assets = 10
    n_days = 1000
    
    # Create sample returns
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (n_days, n_assets)),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Equal weight portfolio
    weights = np.ones(n_assets) / n_assets
    
    # Initialize metrics calculator
    metrics_calc = PortfolioMetrics()
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_all_metrics(weights, returns)
    
    print("Portfolio Metrics Example:")
    print("-" * 40)
    for metric, value in metrics.items():
        if 'ratio' in metric or 'return' in metric:
            print(f"{metric}: {value:.3f}")
        elif 'drawdown' in metric or 'var' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.3f}")
    
    # Check constraints
    constraints = metrics_calc.check_constraints(metrics)
    print(f"\nConstraints Met: {constraints['all_constraints']}")


if __name__ == "__main__":
    main()
