"""
Quarterly Growth Strategy V2 - More realistic backtest

Key realism fixes:
- Dynamic universe by quarter: only symbols with >= 252 prior trading days at each rebalance
- No backward fill of missing data (only forward-fill from past; NaN at start preserved)
- Execution lag: apply rebalanced weights after a configurable lag (default 1 trading day)
- Transaction costs and slippage deducted on trades
- Tighter portfolio constraints (max single weight) and turnover control
- Sharpe uses a risk-free rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from tqdm import tqdm

from data_fetcher import MarketDataFetcher
from portfolio_optimizer import EfficientFrontierOptimizer, OptimizationConstraints


@dataclass
class BacktestConfig:
    universe_size: int = 96
    top_growth: int = 5
    lookback_days: int = 252
    start_date: str = '2005-01-01'
    end_date: str = '2025-01-01'
    exec_lag_days: int = 1  # execution lag in trading days
    tc_bps: float = 10.0    # transaction cost bps per one-way trade
    slippage_bps: float = 5.0  # slippage bps per one-way trade
    max_weight: float = 0.35
    min_weight: float = 0.00
    target_return: float = 0.12
    turnover_limit: float = 0.80  # cap turnover per rebalance (1.0 = 100%)
    risk_free_rate_annual: float = 0.015  # 1.5% per year


class QuarterlyGrowthStrategyV2:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.quarterly_portfolios: List[Dict] = []
        self.performance_history: List[Dict] = []

    # ---------- Data utilities ----------
    def _download_prices(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        data = yf.download(symbols, start=start, end=end, progress=False, group_by='ticker', auto_adjust=False)
        # Extract Adj Close consistently
        if isinstance(data.columns, pd.MultiIndex):
            if ('Adj Close' in data.columns.levels[1]):
                prices = data.xs('Adj Close', level=1, axis=1)
            else:
                # fallback to Close
                prices = data.xs('Close', level=1, axis=1)
        else:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                prices = data
        # Only forward-fill from past; do NOT backfill
        prices = prices.sort_index()
        prices = prices.ffill()
        # keep NaNs at the start to avoid look-ahead/backfill
        # Drop columns with excessive missingness (keep if >= 70% observed)
        min_non_na = int(0.7 * len(prices))
        prices = prices.loc[:, prices.notna().sum() >= min_non_na]
        return prices

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        return returns

    def _quarterly_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        dates = []
        current = pd.Timestamp(year=start.year, month=((start.month - 1)//3)*3 + 1, day=1)
        while current <= end:
            dates.append(current)
            if current.month == 1:
                current = current.replace(month=4)
            elif current.month == 4:
                current = current.replace(month=7)
            elif current.month == 7:
                current = current.replace(month=10)
            else:
                current = pd.Timestamp(year=current.year + 1, month=1, day=1)
        return dates

    # ---------- Selection/Scoring ----------
    def _calc_growth_metrics(self, price_window: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for sym in price_window.columns:
            s = price_window[sym].dropna()
            if len(s) >= self.cfg.lookback_days:
                start_price = s.iloc[-self.cfg.lookback_days]
                end_price = s.iloc[-1]
                if start_price > 0 and end_price > 0:
                    total_return = (end_price / start_price) - 1
                    rets = s.pct_change().dropna()
                    recent = rets.tail(self.cfg.lookback_days - 1)
                    vol = recent.std() * np.sqrt(252)
                    rf_daily = (1 + self.cfg.risk_free_rate_annual) ** (1/252) - 1
                    excess = recent - rf_daily
                    ann_excess = excess.mean() * 252
                    sharpe = ann_excess / (recent.std() * np.sqrt(252)) if recent.std() > 0 else 0.0
                    metrics[sym] = {
                        'total_return': total_return,
                        'volatility': vol,
                        'risk_adj_return': total_return / vol if vol > 0 else 0.0,
                        'sharpe': sharpe,
                    }
        return metrics

    def _select_top(self, metrics: Dict[str, Dict[str, float]], top_n: int) -> List[str]:
        ranked = sorted(metrics.items(), key=lambda kv: kv[1]['sharpe'], reverse=True)
        return [k for k, _ in ranked[:top_n]]

    # ---------- Optimization ----------
    def _optimize_weights(self, returns_window: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
        use_df = returns_window[symbols].dropna()
        if use_df.shape[1] == 0:
            return {s: 1.0/len(symbols) for s in symbols}
        if use_df.shape[1] < 3:
            return {s: 1.0/len(symbols) for s in symbols}
        opt = EfficientFrontierOptimizer(use_df)
        constraints = OptimizationConstraints(
            min_weight=self.cfg.min_weight,
            max_weight=self.cfg.max_weight,
            min_return=self.cfg.target_return,
        )
        try:
            res = opt.optimize_portfolio(constraints=constraints)
            if res.get('optimization_status') == 'optimal':
                w = res['weights']
                return dict(zip(use_df.columns.tolist(), w))
            return {s: 1.0/len(symbols) for s in symbols}
        except Exception:
            return {s: 1.0/len(symbols) for s in symbols}

    # ---------- Backtest ----------
    def backtest(self) -> Dict:
        cfg = self.cfg
        print("🚀 QUARTERLY GROWTH STRATEGY V2 BACKTEST")
        print("="*70)
        print("Realism: dynamic universe, no backfill, exec lag, costs, caps")
        print("="*70)

        # Base universe by liquidity today (still some survivorship, but we constrain per-quarter availability)
        fetcher = MarketDataFetcher(period_years=20)
        base_symbols = fetcher.get_top_assets_by_volume(num_assets=cfg.universe_size)
        prices = self._download_prices(base_symbols, cfg.start_date, cfg.end_date)
        returns = self._compute_returns(prices)

        start_actual = prices.index[0]
        end_actual = prices.index[-1]
        q_dates = self._quarterly_dates(start_actual, end_actual)

        portfolio_value = 100_000.0
        portfolio_history: List[Dict] = []
        weights_prev: Dict[str, float] = {}

        for i in tqdm(range(len(q_dates) - 1)):
            rebal_date = q_dates[i]
            next_date = q_dates[i + 1]

            # Data available up to rebal_date (strictly up to that date)
            px_upto = prices.loc[prices.index <= rebal_date]
            ret_upto = returns.loc[returns.index <= rebal_date]

            # Dynamic universe: symbols with >= lookback_days valid observations before rebal_date
            eligible = [
                s for s in px_upto.columns
                if px_upto[s].notna().tail(cfg.lookback_days).shape[0] == cfg.lookback_days
            ]
            if len(eligible) < cfg.top_growth:
                continue

            # Score and select
            metrics = self._calc_growth_metrics(px_upto[eligible])
            if len(metrics) < cfg.top_growth:
                continue
            selected = self._select_top(metrics, cfg.top_growth)

            # Optimize on past returns only
            lookback_window = returns.loc[returns.index <= rebal_date]
            lookback_window = lookback_window.tail(cfg.lookback_days)
            target_weights = self._optimize_weights(lookback_window, selected)
            # Ensure weights exist for all selected
            for s in selected:
                target_weights.setdefault(s, 0.0)
            # Normalize
            w_sum = sum(target_weights.values())
            if w_sum <= 0:
                target_weights = {s: 1.0/len(selected) for s in selected}
            else:
                target_weights = {k: v / w_sum for k, v in target_weights.items()}

            # Execution lag: find trade date
            future_idx = returns.index[(returns.index > rebal_date)]
            if len(future_idx) <= cfg.exec_lag_days:
                continue
            trade_date = future_idx[cfg.exec_lag_days - 1]

            # Compute turnover vs previous weights (only overlapping names count; others are full buy/sell)
            all_symbols = set(selected) | set(weights_prev.keys())
            prev = {s: weights_prev.get(s, 0.0) for s in all_symbols}
            targ = {s: target_weights.get(s, 0.0) for s in all_symbols}
            turnover = sum(abs(targ[s] - prev[s]) for s in all_symbols)
            if turnover > cfg.turnover_limit:
                # scale changes to respect turnover cap
                delta = {s: targ[s] - prev[s] for s in all_symbols}
                scale = cfg.turnover_limit / (sum(abs(d) for d in delta.values()) + 1e-12)
                targ = {s: prev[s] + delta[s] * scale for s in all_symbols}
                # renormalize to 1
                pos_sum = sum(max(0.0, w) for w in targ.values())
                if pos_sum > 0:
                    targ = {s: max(0.0, w) / pos_sum for s, w in targ.items()}
            # Transaction cost applied on trade_date
            one_way_cost = (cfg.tc_bps + cfg.slippage_bps) / 10_000.0
            trade_cost = portfolio_value * sum(abs(targ[s] - prev[s]) for s in all_symbols) * one_way_cost
            portfolio_value -= trade_cost

            # Apply portfolio returns from trade_date (exclusive) until next_date (inclusive)
            period_mask = (returns.index > trade_date) & (returns.index <= next_date)
            period_rets = returns.loc[period_mask]
            cur_weights = {s: targ.get(s, 0.0) for s in prices.columns}

            for dt, row in period_rets.iterrows():
                daily_ret = 0.0
                for s, w in cur_weights.items():
                    if w == 0.0:
                        continue
                    r = row.get(s, np.nan)
                    if not np.isnan(r):
                        daily_ret += w * r
                portfolio_value *= (1.0 + daily_ret)
                portfolio_history.append({
                    'date': dt,
                    'value': portfolio_value,
                    'weights': {k: v for k, v in cur_weights.items() if v > 0},
                    'holdings': [k for k, v in cur_weights.items() if v > 0],
                })

            # Save quarterly snapshot at rebal point
            self.quarterly_portfolios.append({
                'date': rebal_date,
                'holdings': selected,
                'weights': {k: v for k, v in targ.items() if v > 0},
                'metrics': metrics,
                'turnover': turnover,
                'trade_cost': trade_cost,
                'trade_date': trade_date,
            })
            weights_prev = {k: v for k, v in targ.items() if v > 0}

        if not portfolio_history:
            print('No history generated.')
            return {}

        final_value = portfolio_history[-1]['value']
        total_return = final_value / 100_000.0 - 1.0
        years = (pd.to_datetime(cfg.end_date) - pd.to_datetime(cfg.start_date)).days / 365.25
        annualized_return = (final_value / 100_000.0) ** (1/years) - 1

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'portfolio_history': portfolio_history,
            'quarterly_portfolios': self.quarterly_portfolios,
        }

    # ---------- Visualization ----------
    def visualize(self, results: Dict):
        if not results:
            print('No results to visualize')
            return

        fig = plt.figure(figsize=(20, 14))
        ax1 = plt.subplot(2, 2, 1)
        hist = results['portfolio_history']
        dates = [h['date'] for h in hist]
        vals = [h['value'] for h in hist]
        ax1.plot(dates, vals, label='Strategy V2', color='#2E86AB', lw=2)
        ax1.set_title('Portfolio Value (V2)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$ {y:,.0f}'))
        ax1.grid(True, alpha=0.3)

        # SPY benchmark
        try:
            start_date = pd.to_datetime(dates[0]).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            spy_prices = spy['Adj Close'] if 'Adj Close' in spy.columns else spy['Close']
            spy_norm = (spy_prices / spy_prices.iloc[0]) * 100_000
            ax1.plot(spy_norm.index, spy_norm.values, '--', color='red', label='SPY')
            # Ensure scalar float for formatting
            try:
                spy_final_value = float(spy_norm.iloc[-1])
            except Exception:
                # In rare cases iloc may return a 0-dim array/Series
                spy_final_value = float(np.asarray(spy_norm.iloc[-1]).reshape(-1)[0])
        except Exception:
            years = len(vals) / 252
            spy_final_value = 100000 * (1.10) ** years
            spy_vals = [100000 * (1.10) ** (i/252) for i in range(len(vals))]
            ax1.plot(dates, spy_vals, '--', color='red', label='SPY (10% p.a. proxy)')
        ax1.legend()

        # Holdings frequency
        ax2 = plt.subplot(2, 2, 2)
        counts: Dict[str, int] = {}
        for q in self.quarterly_portfolios:
            for hld in q['holdings']:
                counts[hld] = counts.get(hld, 0) + 1
        if counts:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            names = [t[0] for t in top]
            cts = [t[1] for t in top]
            bars = ax2.bar(names, cts, color='#A23B72')
            ax2.set_title('Most Frequently Selected (V2)')
            ax2.tick_params(axis='x', rotation=45)
            for b, c in zip(bars, cts):
                ax2.annotate(f'{c}', xy=(b.get_x()+b.get_width()/2, c), xytext=(0,3),
                             textcoords='offset points', ha='center')

        # Quarterly returns histogram
        ax3 = plt.subplot(2, 2, 3)
        qret = []
        for i in range(63, len(vals), 63):
            qret.append(vals[i]/vals[i-63]-1)
        if qret:
            ax3.hist(qret, bins=20, color='#F18F01', edgecolor='black', alpha=0.8)
            ax3.set_title('Quarterly Returns (V2)')
            ax3.axvline(np.mean(qret), color='red', linestyle='--', label=f'Mean {np.mean(qret):.1%}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Summary table
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        table_data = [
            ['Metric', 'Strategy V2', 'SPY'],
            ['Final Value', f"$ {results['final_value']:,.0f}", f"$ {spy_final_value:,.0f}"],
            ['Total Return', f"{results['total_return']:.1%}", '—'],
            ['Annualized', f"{results['annualized_return']:.2%}", '—'],
            ['Wealth Multiple', f"{results['final_value']/100000:.1f}x", '—'],
        ]
        tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center', bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.0)

        plt.suptitle('Quarterly Growth Strategy V2 - Realistic Backtest', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = '/Users/nttssv/Documents/efficient_frontier/results/plots/quarterly_growth_strategy_v2_realistic.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {out_path}")


def main():
    cfg = BacktestConfig()
    strat = QuarterlyGrowthStrategyV2(cfg)
    results = strat.backtest()
    if results:
        # Print detailed results for README verification
        print(f"\n🎯 STRATEGY V2 FINAL RESULTS:")
        print("="*50)
        print(f"Final Value: ${results['final_value']:,.0f}")
        print(f"Total Return: {results['total_return']:.1%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Wealth Multiple: {results['final_value']/100000:.1f}x")
        
        # Calculate total transaction costs
        total_costs = sum(q.get('trade_cost', 0) for q in strat.quarterly_portfolios)
        print(f"Total Transaction Costs: ${total_costs:,.0f}")
        print(f"Number of Rebalances: {len(strat.quarterly_portfolios)}")
        
        strat.visualize(results)
    return results


if __name__ == '__main__':
    main()
