"""
Market data fetching module for efficient frontier portfolio optimization.
Handles downloading and preprocessing of top assets by trading volume.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MarketDataFetcher:
    """Fetches and preprocesses market data for portfolio optimization."""
    
    def __init__(self, period_years: int = 5):
        """
        Initialize the data fetcher.
        
        Args:
            period_years: Number of years of historical data to fetch
        """
        self.period_years = period_years
        self.data = None
        self.returns = None
        self.symbols = []
        
    def get_top_assets_by_volume(self, num_assets: int = 100) -> List[str]:
        """
        Get top assets by trading volume from major indices.
        
        Args:
            num_assets: Number of top assets to return
            
        Returns:
            List of ticker symbols
        """
        # Get major index components
        indices = {
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'IWM': 'iShares Russell 2000 ETF',
            'VTI': 'Vanguard Total Stock Market ETF',
            'ARKK': 'ARK Innovation ETF',
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund',
            'XLV': 'Health Care Select Sector SPDR Fund',
            'XLI': 'Industrial Select Sector SPDR Fund'
        }
        
        # Popular large-cap stocks with high volume
        high_volume_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT',
            'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MPC',
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'RTX',
            'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'NFLX', 'ROKU',
            'BRK-B', 'SPY', 'QQQ', 'IWM', 'VTI', 'GLD', 'SLV', 'TLT',
            'ARKK', 'ARKQ', 'ARKG', 'ARKW', 'ARKF', 'XLF', 'XLK', 'XLE',
            'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'XME',
            'SMH', 'IBB', 'XBI', 'SOXX', 'FINX', 'HACK', 'ROBO', 'ICLN'
        ]
        
        # Combine all symbols and remove duplicates
        all_symbols = list(set(list(indices.keys()) + high_volume_stocks))
        
        print(f"Fetching volume data for {len(all_symbols)} symbols...")
        
        # Get recent volume data to rank assets
        volume_data = {}
        for symbol in tqdm(all_symbols, desc="Fetching volume data"):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                if not hist.empty and len(hist) > 10:
                    avg_volume = hist['Volume'].mean()
                    avg_price = hist['Close'].mean()
                    dollar_volume = avg_volume * avg_price
                    volume_data[symbol] = dollar_volume
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        # Sort by dollar volume and return top assets
        sorted_symbols = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:num_assets]]
        
        print(f"Selected top {len(top_symbols)} assets by trading volume")
        return top_symbols
    
    def fetch_historical_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch historical price data for given symbols.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            DataFrame with adjusted close prices
        """
        print(f"Fetching {self.period_years} years of historical data...")
        
        # Calculate start date
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=self.period_years)
        
        # Fetch data
        data = yf.download(
            symbols, 
            start=start_date, 
            end=end_date, 
            progress=True,
            group_by='ticker'
        )
        
        # Extract adjusted close prices
        if len(symbols) == 1:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].to_frame()
                prices.columns = symbols
            else:
                prices = data['Close'].to_frame()  # Fallback to Close
                prices.columns = symbols
        else:
            # Handle multi-level columns from yfinance
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Adj Close' in data.columns.get_level_values(1):
                        prices = data.xs('Adj Close', level=1, axis=1)
                    else:
                        prices = data.xs('Close', level=1, axis=1)
                else:
                    # Single level columns - extract Close prices
                    close_cols = [col for col in data.columns if col.endswith('Close')]
                    if close_cols:
                        prices = data[close_cols]
                        # Clean column names to get symbols
                        prices.columns = [col.split()[0] if ' ' in col else col for col in prices.columns]
                    else:
                        # Last resort - use all data
                        prices = data
            except Exception as e:
                print(f"Warning: Error extracting prices, using Close data: {e}")
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data.xs('Close', level=1, axis=1)
                else:
                    prices = data
        
        # Clean data
        prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))  # Remove assets with >20% missing data
        prices = prices.ffill().bfill()  # Forward/backward fill
        
        self.data = prices
        self.symbols = list(prices.columns)
        
        print(f"Successfully fetched data for {len(self.symbols)} assets")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        return prices
    
    def calculate_returns(self, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame with daily returns
        """
        if self.data is None:
            raise ValueError("No price data available. Call fetch_historical_data first.")
        
        if method == 'log':
            returns = np.log(self.data / self.data.shift(1))
        else:
            returns = self.data.pct_change()
        
        returns = returns.dropna()
        self.returns = returns
        
        print(f"Calculated {method} returns for {len(returns.columns)} assets")
        print(f"Returns date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        
        return returns
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the fetched data."""
        if self.data is None or self.returns is None:
            return {}
        
        summary = {
            'num_assets': len(self.symbols),
            'date_range': (self.data.index[0].date(), self.data.index[-1].date()),
            'total_days': len(self.data),
            'trading_days': len(self.returns),
            'symbols': self.symbols[:10],  # Show first 10 symbols
            'data_completeness': (self.data.notna().sum() / len(self.data)).mean()
        }
        
        return summary
    
    def validate_data_quality(self) -> Dict:
        """
        Validate the quality of fetched data.
        
        Returns:
            Dictionary with validation results
        """
        if self.returns is None:
            return {'valid': False, 'reason': 'No returns data available'}
        
        validation = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for sufficient data
        min_days = 252 * 2  # At least 2 years of trading days
        if len(self.returns) < min_days:
            validation['warnings'].append(f"Limited data: {len(self.returns)} days (recommended: >{min_days})")
        
        # Check for extreme returns (potential data errors)
        extreme_returns = (self.returns.abs() > 0.5).any()
        if extreme_returns.any():
            extreme_assets = extreme_returns[extreme_returns].index.tolist()
            validation['warnings'].append(f"Extreme returns detected in: {extreme_assets[:5]}")
        
        # Check for assets with too many zero returns (potential issues)
        zero_returns_pct = (self.returns == 0).sum() / len(self.returns)
        problematic_assets = zero_returns_pct[zero_returns_pct > 0.1].index.tolist()
        if problematic_assets:
            validation['warnings'].append(f"High zero returns in: {problematic_assets[:5]}")
        
        # Check correlation matrix condition
        corr_matrix = self.returns.corr()
        eigenvals = np.linalg.eigvals(corr_matrix.values)
        condition_number = np.max(eigenvals) / np.min(eigenvals)
        if condition_number > 1000:
            validation['warnings'].append(f"High correlation matrix condition number: {condition_number:.0f}")
        
        return validation


def main():
    """Example usage of MarketDataFetcher."""
    # Initialize fetcher
    fetcher = MarketDataFetcher(period_years=5)
    
    # Get top assets
    symbols = fetcher.get_top_assets_by_volume(num_assets=50)  # Start with 50 for testing
    
    # Fetch data
    prices = fetcher.fetch_historical_data(symbols)
    returns = fetcher.calculate_returns()
    
    # Validate data
    validation = fetcher.validate_data_quality()
    print("\nData Validation Results:")
    print(f"Valid: {validation['valid']}")
    if validation.get('warnings'):
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Print summary
    summary = fetcher.get_data_summary()
    print(f"\nData Summary:")
    print(f"Assets: {summary['num_assets']}")
    print(f"Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Trading Days: {summary['trading_days']}")
    print(f"Data Completeness: {summary['data_completeness']:.2%}")


if __name__ == "__main__":
    main()
