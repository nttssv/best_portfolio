"""
Future Portfolio Analysis - Forward-looking portfolio optimization for next 25 years
Based on current market conditions, valuations, and emerging trends (2025-2050)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from tqdm import tqdm
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from advanced_portfolio_optimizer import MultiObjectiveOptimizer
from portfolio_metrics import PortfolioMetrics


class FuturePortfolioAnalyzer:
    """Analyze and recommend portfolios for future 25-year performance."""
    
    def __init__(self):
        """Initialize future portfolio analyzer."""
        self.current_date = datetime.now()
        self.target_date = self.current_date + timedelta(days=25*365)
        self.metrics_calc = PortfolioMetrics()
        
    def get_current_market_data(self) -> pd.DataFrame:
        """Get current market data and recent performance."""
        print("📊 Fetching Current Market Data (2025)...")
        
        # Define future-focused asset universe
        future_assets = {
            # Technology & AI
            'NVDA': 'NVIDIA Corp',
            'MSFT': 'Microsoft Corp', 
            'GOOGL': 'Alphabet Inc',
            'AAPL': 'Apple Inc',
            'TSLA': 'Tesla Inc',
            'AMD': 'Advanced Micro Devices',
            'PLTR': 'Palantir Technologies',
            'SMCI': 'Super Micro Computer',
            
            # Clean Energy & Infrastructure
            'ENPH': 'Enphase Energy',
            'NEE': 'NextEra Energy',
            'ICLN': 'iShares Clean Energy ETF',
            'LIT': 'Global X Lithium ETF',
            'ARKK': 'ARK Innovation ETF',
            
            # Emerging Markets & Demographics
            'VWO': 'Vanguard Emerging Markets',
            'INDA': 'iShares MSCI India ETF',
            'FXI': 'iShares China Large-Cap ETF',
            'EWZ': 'iShares MSCI Brazil ETF',
            
            # Healthcare & Biotech
            'UNH': 'UnitedHealth Group',
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer Inc',
            'MRNA': 'Moderna Inc',
            'GILD': 'Gilead Sciences',
            
            # Commodities & Inflation Hedges
            'GLD': 'SPDR Gold Trust',
            'SLV': 'iShares Silver Trust',
            'DBC': 'Invesco DB Commodity Index',
            'PDBC': 'Invesco Optimum Yield Diversified Commodity',
            
            # REITs & Real Assets
            'VNQ': 'Vanguard Real Estate ETF',
            'O': 'Realty Income Corp',
            'PLD': 'Prologis Inc',
            
            # Broad Market & Defensive
            'QQQ': 'Invesco QQQ Trust',
            'SPY': 'SPDR S&P 500 ETF',
            'VTI': 'Vanguard Total Stock Market',
            'BND': 'Vanguard Total Bond Market',
            'TIPS': 'iShares TIPS Bond ETF'
        }
        
        symbols = list(future_assets.keys())
        
        # Fetch recent data (3 years for trend analysis)
        print(f"Downloading data for {len(symbols)} future-focused assets...")
        
        data = yf.download(symbols, period="3y", progress=True)
        
        # Handle different data structures
        if isinstance(data.columns, pd.MultiIndex):
            # Check what columns are available
            available_cols = data.columns.get_level_values(1).unique()
            if 'Adj Close' in available_cols:
                prices = data.xs('Adj Close', level=1, axis=1)
            elif 'Close' in available_cols:
                prices = data.xs('Close', level=1, axis=1)
            else:
                # Use the first available price column
                price_col = available_cols[0]
                prices = data.xs(price_col, level=1, axis=1)
        else:
            if len(symbols) == 1:
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close'].to_frame()
                    prices.columns = symbols
                else:
                    prices = data.iloc[:, -1].to_frame()  # Use last column
                    prices.columns = symbols
            else:
                prices = data
            
        # Clean data
        prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))
        prices = prices.ffill().bfill()
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        print(f"✅ Loaded {len(prices.columns)} assets with {len(returns)} trading days")
        
        return returns, prices, future_assets
    
    def analyze_current_valuations(self, symbols: list) -> pd.DataFrame:
        """Analyze current valuations and market conditions."""
        print("💰 Analyzing Current Market Valuations...")
        
        valuation_data = []
        
        for symbol in tqdm(symbols, desc="Fetching valuation metrics"):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get key valuation metrics
                pe_ratio = info.get('trailingPE', np.nan)
                peg_ratio = info.get('pegRatio', np.nan)
                price_to_book = info.get('priceToBook', np.nan)
                market_cap = info.get('marketCap', np.nan)
                revenue_growth = info.get('revenueGrowth', np.nan)
                profit_margins = info.get('profitMargins', np.nan)
                
                # Calculate recent momentum
                hist = ticker.history(period="1y")
                if len(hist) > 0:
                    ytd_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                else:
                    ytd_return = np.nan
                    volatility = np.nan
                
                valuation_data.append({
                    'symbol': symbol,
                    'pe_ratio': pe_ratio,
                    'peg_ratio': peg_ratio,
                    'price_to_book': price_to_book,
                    'market_cap': market_cap,
                    'revenue_growth': revenue_growth,
                    'profit_margins': profit_margins,
                    'ytd_return': ytd_return,
                    'volatility': volatility
                })
                
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}")
                continue
        
        valuation_df = pd.DataFrame(valuation_data)
        
        # Calculate valuation scores (lower is better for value)
        valuation_df['value_score'] = (
            valuation_df['pe_ratio'].rank(ascending=True, pct=True).fillna(0.5) * 0.3 +
            valuation_df['peg_ratio'].rank(ascending=True, pct=True).fillna(0.5) * 0.3 +
            valuation_df['price_to_book'].rank(ascending=True, pct=True).fillna(0.5) * 0.2 +
            (1 - valuation_df['ytd_return'].rank(ascending=True, pct=True).fillna(0.5)) * 0.2
        )
        
        # Calculate growth score (higher is better)
        valuation_df['growth_score'] = (
            valuation_df['revenue_growth'].rank(ascending=True, pct=True).fillna(0.5) * 0.4 +
            valuation_df['profit_margins'].rank(ascending=True, pct=True).fillna(0.5) * 0.3 +
            valuation_df['ytd_return'].rank(ascending=True, pct=True).fillna(0.5) * 0.3
        )
        
        # Calculate quality score
        valuation_df['quality_score'] = (
            valuation_df['profit_margins'].rank(ascending=True, pct=True).fillna(0.5) * 0.5 +
            (1 - valuation_df['volatility'].rank(ascending=True, pct=True).fillna(0.5)) * 0.5
        )
        
        return valuation_df
    
    def create_future_portfolio_strategies(self, returns: pd.DataFrame, 
                                         valuation_df: pd.DataFrame) -> dict:
        """Create multiple forward-looking portfolio strategies."""
        print("🚀 Creating Future Portfolio Strategies...")
        
        # Align data
        common_symbols = list(set(returns.columns) & set(valuation_df['symbol']))
        returns_aligned = returns[common_symbols]
        valuation_aligned = valuation_df[valuation_df['symbol'].isin(common_symbols)].set_index('symbol')
        
        strategies = {}
        
        # Strategy 1: AI & Technology Focus (High Growth)
        tech_symbols = ['NVDA', 'MSFT', 'GOOGL', 'AAPL', 'AMD', 'PLTR', 'TSLA']
        tech_weights = self._create_sector_portfolio(returns_aligned, tech_symbols, 'equal_weight')
        if tech_weights is not None:
            strategies['ai_tech_growth'] = {
                'weights': tech_weights,
                'description': 'AI & Technology Growth Portfolio',
                'target_return': 0.25,  # 25% annual target
                'risk_level': 'High'
            }
        
        # Strategy 2: Clean Energy & Sustainability
        clean_symbols = ['ENPH', 'NEE', 'ICLN', 'LIT', 'TSLA']
        clean_weights = self._create_sector_portfolio(returns_aligned, clean_symbols, 'equal_weight')
        if clean_weights is not None:
            strategies['clean_energy'] = {
                'weights': clean_weights,
                'description': 'Clean Energy & Sustainability Portfolio',
                'target_return': 0.20,
                'risk_level': 'Medium-High'
            }
        
        # Strategy 3: Emerging Markets & Demographics
        em_symbols = ['VWO', 'INDA', 'FXI', 'EWZ']
        em_weights = self._create_sector_portfolio(returns_aligned, em_symbols, 'equal_weight')
        if em_weights is not None:
            strategies['emerging_markets'] = {
                'weights': em_weights,
                'description': 'Emerging Markets & Demographics Portfolio',
                'target_return': 0.18,
                'risk_level': 'High'
            }
        
        # Strategy 4: Value + Quality (Conservative Growth)
        value_symbols = valuation_aligned.nlargest(10, 'value_score').index.tolist()
        quality_symbols = valuation_aligned.nlargest(10, 'quality_score').index.tolist()
        value_quality_symbols = list(set(value_symbols + quality_symbols))
        vq_weights = self._create_sector_portfolio(returns_aligned, value_quality_symbols, 'quality_weighted')
        if vq_weights is not None:
            strategies['value_quality'] = {
                'weights': vq_weights,
                'description': 'Value + Quality Portfolio',
                'target_return': 0.15,
                'risk_level': 'Medium'
            }
        
        # Strategy 5: Balanced Future Portfolio (Recommended)
        balanced_allocation = {
            'tech': 0.40,      # Technology & AI
            'clean': 0.20,     # Clean energy
            'healthcare': 0.15, # Healthcare & biotech
            'emerging': 0.10,   # Emerging markets
            'commodities': 0.10, # Inflation hedges
            'reits': 0.05      # Real estate
        }
        
        balanced_weights = self._create_balanced_future_portfolio(
            returns_aligned, valuation_aligned, balanced_allocation
        )
        if balanced_weights is not None:
            strategies['balanced_future'] = {
                'weights': balanced_weights,
                'description': 'Balanced Future Portfolio (Recommended)',
                'target_return': 0.22,
                'risk_level': 'Medium-High'
            }
        
        return strategies
    
    def _create_sector_portfolio(self, returns: pd.DataFrame, 
                               target_symbols: list, 
                               weighting: str = 'equal_weight') -> np.ndarray:
        """Create portfolio weights for specific sector."""
        available_symbols = [s for s in target_symbols if s in returns.columns]
        
        if len(available_symbols) == 0:
            return None
        
        weights = np.zeros(len(returns.columns))
        
        if weighting == 'equal_weight':
            weight_per_asset = 1.0 / len(available_symbols)
            for symbol in available_symbols:
                idx = returns.columns.get_loc(symbol)
                weights[idx] = weight_per_asset
        
        return weights
    
    def _create_balanced_future_portfolio(self, returns: pd.DataFrame,
                                        valuation_df: pd.DataFrame,
                                        allocation: dict) -> np.ndarray:
        """Create balanced portfolio for future performance."""
        
        sector_mapping = {
            'tech': ['NVDA', 'MSFT', 'GOOGL', 'AAPL', 'AMD', 'PLTR', 'TSLA', 'QQQ'],
            'clean': ['ENPH', 'NEE', 'ICLN', 'LIT'],
            'healthcare': ['UNH', 'JNJ', 'PFE', 'MRNA', 'GILD'],
            'emerging': ['VWO', 'INDA', 'FXI', 'EWZ'],
            'commodities': ['GLD', 'SLV', 'DBC', 'PDBC'],
            'reits': ['VNQ', 'O', 'PLD']
        }
        
        weights = np.zeros(len(returns.columns))
        
        for sector, target_allocation in allocation.items():
            sector_symbols = [s for s in sector_mapping.get(sector, []) if s in returns.columns]
            
            if sector_symbols:
                # Use quality scores for weighting within sector
                sector_weights = []
                for symbol in sector_symbols:
                    if symbol in valuation_df.index:
                        quality = valuation_df.loc[symbol, 'quality_score']
                        growth = valuation_df.loc[symbol, 'growth_score']
                        combined_score = quality * 0.6 + growth * 0.4
                        sector_weights.append(combined_score)
                    else:
                        sector_weights.append(0.5)  # Neutral weight
                
                # Normalize sector weights
                sector_weights = np.array(sector_weights)
                sector_weights = sector_weights / sector_weights.sum()
                
                # Apply to portfolio
                for i, symbol in enumerate(sector_symbols):
                    idx = returns.columns.get_loc(symbol)
                    weights[idx] = target_allocation * sector_weights[i]
        
        return weights
    
    def project_25_year_performance(self, strategies: dict, returns: pd.DataFrame) -> dict:
        """Project 25-year performance for each strategy."""
        print("📈 Projecting 25-Year Performance...")
        
        projections = {}
        
        for strategy_name, strategy_data in strategies.items():
            weights = strategy_data['weights']
            target_return = strategy_data['target_return']
            
            # Calculate historical metrics
            portfolio_returns = returns.dot(weights)
            historical_return = portfolio_returns.mean() * 252
            historical_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Adjust for future expectations (more conservative)
            # Assume returns will be 70% of historical due to higher valuations
            expected_return = min(historical_return * 0.7, target_return)
            expected_vol = historical_vol * 1.1  # Slightly higher volatility
            
            # 25-year projection
            initial_investment = 100000
            final_value = initial_investment * (1 + expected_return) ** 25
            total_return = (final_value / initial_investment - 1) * 100
            
            # Calculate metrics
            sharpe_ratio = expected_return / expected_vol if expected_vol > 0 else 0
            
            projections[strategy_name] = {
                'strategy_data': strategy_data,
                'historical_return': historical_return,
                'expected_return': expected_return,
                'expected_volatility': expected_vol,
                'sharpe_ratio': sharpe_ratio,
                'initial_investment': initial_investment,
                'final_value': final_value,
                'total_return': total_return,
                'wealth_multiple': final_value / initial_investment
            }
        
        return projections
    
    def create_recommendation_report(self, projections: dict) -> None:
        """Create comprehensive recommendation report."""
        print("\n" + "="*80)
        print("🔮 FUTURE PORTFOLIO RECOMMENDATIONS (2025-2050)")
        print("="*80)
        
        # Sort by expected final value
        sorted_strategies = sorted(projections.items(), 
                                 key=lambda x: x[1]['final_value'], 
                                 reverse=True)
        
        print(f"\n🏆 TOP PORTFOLIO STRATEGIES FOR NEXT 25 YEARS")
        print("-" * 70)
        
        for i, (strategy_name, data) in enumerate(sorted_strategies, 1):
            strategy_info = data['strategy_data']
            
            print(f"\n{i}. {strategy_info['description']}")
            print(f"   Risk Level: {strategy_info['risk_level']}")
            print(f"   Expected Annual Return: {data['expected_return']:.2%}")
            print(f"   Expected Volatility: {data['expected_volatility']:.2%}")
            print(f"   Sharpe Ratio: {data['sharpe_ratio']:.3f}")
            print(f"   25-Year Final Value: ${data['final_value']:,.0f}")
            print(f"   Total Return: {data['total_return']:.1f}%")
            print(f"   Wealth Multiple: {data['wealth_multiple']:.1f}x")
        
        # Detailed recommendation
        best_strategy = sorted_strategies[0]
        best_name, best_data = best_strategy
        
        print(f"\n🎯 RECOMMENDED PORTFOLIO: {best_data['strategy_data']['description']}")
        print("="*70)
        print(f"💰 Investment Projection:")
        print(f"   Initial Investment: ${best_data['initial_investment']:,}")
        print(f"   Expected Final Value: ${best_data['final_value']:,.0f}")
        print(f"   Annual Return Needed: {best_data['expected_return']:.2%}")
        print(f"   Risk Level: {best_data['strategy_data']['risk_level']}")
        
        print(f"\n📊 Key Assumptions for 2025-2050:")
        print(f"   • Technology continues to drive growth")
        print(f"   • Clean energy transition accelerates")
        print(f"   • Emerging markets benefit from demographics")
        print(f"   • Inflation remains elevated (commodities hedge)")
        print(f"   • AI/automation creates new opportunities")
        
        print(f"\n⚠️  IMPORTANT DISCLAIMERS:")
        print(f"   • Future performance may differ significantly from projections")
        print(f"   • Market conditions and valuations will change")
        print(f"   • Regular rebalancing and monitoring recommended")
        print(f"   • Consider your risk tolerance and investment timeline")
        print(f"   • This is for educational purposes only")


def main():
    """Run future portfolio analysis."""
    print("🔮 FUTURE PORTFOLIO ANALYSIS (2025-2050)")
    print("="*60)
    
    analyzer = FuturePortfolioAnalyzer()
    
    # Get current market data
    returns, prices, asset_info = analyzer.get_current_market_data()
    
    # Analyze valuations
    valuation_df = analyzer.analyze_current_valuations(list(returns.columns))
    
    # Create strategies
    strategies = analyzer.create_future_portfolio_strategies(returns, valuation_df)
    
    # Project performance
    projections = analyzer.project_25_year_performance(strategies, returns)
    
    # Create recommendation
    analyzer.create_recommendation_report(projections)
    
    print(f"\n🎉 Future portfolio analysis completed!")
    return projections


if __name__ == "__main__":
    results = main()
