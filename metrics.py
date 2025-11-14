"""
Performance Metrics Calculation
Computes comprehensive trading metrics for strategy evaluation.

Primary metric: Profit Factor
Secondary: Sharpe, Sortino, MDD, hit rate, payoff ratio, trades/week

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculates comprehensive performance metrics for backtests.
    """

    def __init__(self, trades: pd.DataFrame, equity_curve: pd.DataFrame):
        """
        Initialize metrics calculator.

        Args:
            trades: DataFrame with trade log
            equity_curve: DataFrame with equity curve (timestamp, equity)
        """
        self.trades = trades
        self.equity_curve = equity_curve

        if len(trades) == 0:
            logger.warning("No trades to analyze")

        logger.info(f"PerformanceMetrics initialized with {len(trades)} trades")

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all performance metrics.

        Returns:
            Dictionary with all metrics
        """
        if len(self.trades) == 0:
            return self._empty_metrics()

        metrics = {
            # Basic metrics
            'total_trades': self.total_trades(),
            'winning_trades': self.winning_trades(),
            'losing_trades': self.losing_trades(),

            # Primary metric
            'profit_factor': self.profit_factor(),

            # Returns
            'total_pnl': self.total_pnl(),
            'total_return_pct': self.total_return_pct(),
            'avg_trade_pnl': self.avg_trade_pnl(),

            # Risk-adjusted returns
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),

            # Drawdown
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_pct': self.max_drawdown_pct(),
            'avg_drawdown': self.avg_drawdown(),

            # Win/loss stats
            'win_rate': self.win_rate(),
            'loss_rate': self.loss_rate(),
            'payoff_ratio': self.payoff_ratio(),
            'expectancy': self.expectancy(),

            # Trade duration
            'avg_bars_in_trade': self.avg_bars_in_trade(),
            'avg_winning_bars': self.avg_winning_bars(),
            'avg_losing_bars': self.avg_losing_bars(),

            # Frequency
            'trades_per_week': self.trades_per_week(),
            'trades_per_month': self.trades_per_month(),

            # Consecutive stats
            'max_consecutive_wins': self.max_consecutive_wins(),
            'max_consecutive_losses': self.max_consecutive_losses(),

            # Statistical significance
            't_statistic': self.t_statistic(),
            'p_value': self.p_value(),
        }

        return metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict when no trades."""
        return {
            'total_trades': 0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0,
            'total_return_pct': 0.0,
        }

    # =========================================================================
    # Basic Metrics
    # =========================================================================

    def total_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)

    def winning_trades(self) -> int:
        """Number of winning trades."""
        return (self.trades['pnl_net'] > 0).sum()

    def losing_trades(self) -> int:
        """Number of losing trades."""
        return (self.trades['pnl_net'] < 0).sum()

    def total_pnl(self) -> float:
        """Total profit/loss."""
        return self.trades['pnl_net'].sum()

    def total_return_pct(self) -> float:
        """Total return as percentage."""
        if len(self.equity_curve) == 0:
            return 0.0

        initial_equity = self.equity_curve['equity'].iloc[0]
        final_equity = self.equity_curve['equity'].iloc[-1]

        return ((final_equity / initial_equity) - 1) * 100

    def avg_trade_pnl(self) -> float:
        """Average PnL per trade."""
        return self.trades['pnl_net'].mean()

    # =========================================================================
    # Primary Metric: Profit Factor
    # =========================================================================

    def profit_factor(self) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss

        Primary metric for edge discovery.
        PF > 1.5 indicates viable strategy.
        """
        gross_profit = self.trades[self.trades['pnl_net'] > 0]['pnl_net'].sum()
        gross_loss = abs(self.trades[self.trades['pnl_net'] < 0]['pnl_net'].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    # =========================================================================
    # Risk-Adjusted Returns
    # =========================================================================

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Annualized for crypto (365 days).
        """
        if len(self.trades) < 2:
            return 0.0

        returns = self.trades['pnl_net'] / self.trades['size']  # Returns as % of position
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Annualize (assuming ~20 trades/week, 52 weeks/year)
        trades_per_year = 1040
        sharpe = (mean_return - risk_free_rate) / std_return
        sharpe_annual = sharpe * np.sqrt(trades_per_year)

        return sharpe_annual

    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation

        Similar to Sharpe but only penalizes downside volatility.
        """
        if len(self.trades) < 2:
            return 0.0

        returns = self.trades['pnl_net'] / self.trades['size']
        mean_return = returns.mean()

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return np.inf if mean_return > risk_free_rate else 0.0

        downside_std = negative_returns.std()

        if downside_std == 0:
            return 0.0

        # Annualize
        trades_per_year = 1040
        sortino = (mean_return - risk_free_rate) / downside_std
        sortino_annual = sortino * np.sqrt(trades_per_year)

        return sortino_annual

    def calmar_ratio(self) -> float:
        """
        Calmar Ratio = Annualized Return / Max Drawdown

        Measures return relative to worst drawdown.
        """
        mdd = self.max_drawdown_pct()

        if mdd == 0:
            return 0.0

        # Annualized return (assuming backtest period length)
        if len(self.equity_curve) > 0:
            days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            if days > 0:
                annual_return = (self.total_return_pct() / 100) * (365 / days)
                return annual_return / (abs(mdd) / 100)

        return 0.0

    # =========================================================================
    # Drawdown Metrics
    # =========================================================================

    def max_drawdown(self) -> float:
        """Maximum drawdown in absolute terms."""
        if len(self.equity_curve) == 0:
            return 0.0

        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = equity - running_max

        return drawdown.min()

    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as percentage."""
        if len(self.equity_curve) == 0:
            return 0.0

        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown_pct = ((equity - running_max) / running_max) * 100

        return drawdown_pct.min()

    def avg_drawdown(self) -> float:
        """Average drawdown during drawdown periods."""
        if len(self.equity_curve) == 0:
            return 0.0

        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = ((equity - running_max) / running_max) * 100

        # Only consider periods in drawdown
        drawdown_periods = drawdown[drawdown < 0]

        if len(drawdown_periods) == 0:
            return 0.0

        return drawdown_periods.mean()

    # =========================================================================
    # Win/Loss Statistics
    # =========================================================================

    def win_rate(self) -> float:
        """Win rate as percentage."""
        if len(self.trades) == 0:
            return 0.0

        return (self.winning_trades() / self.total_trades()) * 100

    def loss_rate(self) -> float:
        """Loss rate as percentage."""
        return 100 - self.win_rate()

    def payoff_ratio(self) -> float:
        """
        Payoff Ratio = Average Win / Average Loss

        Also known as Profit/Loss ratio.
        """
        wins = self.trades[self.trades['pnl_net'] > 0]['pnl_net']
        losses = self.trades[self.trades['pnl_net'] < 0]['pnl_net']

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return 0.0

        return avg_win / avg_loss

    def expectancy(self) -> float:
        """
        Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

        Expected value per trade.
        """
        wins = self.trades[self.trades['pnl_net'] > 0]['pnl_net']
        losses = self.trades[self.trades['pnl_net'] < 0]['pnl_net']

        if len(self.trades) == 0:
            return 0.0

        win_rate = self.win_rate() / 100
        loss_rate = self.loss_rate() / 100

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        return (win_rate * avg_win) - (loss_rate * avg_loss)

    # =========================================================================
    # Trade Duration
    # =========================================================================

    def avg_bars_in_trade(self) -> float:
        """Average number of bars held per trade."""
        if 'bars_in_trade' in self.trades.columns:
            return self.trades['bars_in_trade'].mean()
        return 0.0

    def avg_winning_bars(self) -> float:
        """Average bars held for winning trades."""
        if 'bars_in_trade' not in self.trades.columns:
            return 0.0

        winning_trades = self.trades[self.trades['pnl_net'] > 0]
        if len(winning_trades) == 0:
            return 0.0

        return winning_trades['bars_in_trade'].mean()

    def avg_losing_bars(self) -> float:
        """Average bars held for losing trades."""
        if 'bars_in_trade' not in self.trades.columns:
            return 0.0

        losing_trades = self.trades[self.trades['pnl_net'] < 0]
        if len(losing_trades) == 0:
            return 0.0

        return losing_trades['bars_in_trade'].mean()

    # =========================================================================
    # Trade Frequency
    # =========================================================================

    def trades_per_week(self) -> float:
        """Average trades per week."""
        if len(self.trades) == 0:
            return 0.0

        days = (self.trades['exit_time'].max() - self.trades['entry_time'].min()).days
        weeks = max(days / 7, 1)

        return len(self.trades) / weeks

    def trades_per_month(self) -> float:
        """Average trades per month."""
        return self.trades_per_week() * 4.33

    # =========================================================================
    # Consecutive Stats
    # =========================================================================

    def max_consecutive_wins(self) -> int:
        """Maximum consecutive winning trades."""
        if len(self.trades) == 0:
            return 0

        is_win = (self.trades['pnl_net'] > 0).astype(int)
        groups = (is_win != is_win.shift()).cumsum()
        consecutive = is_win.groupby(groups).cumsum()

        return consecutive.max()

    def max_consecutive_losses(self) -> int:
        """Maximum consecutive losing trades."""
        if len(self.trades) == 0:
            return 0

        is_loss = (self.trades['pnl_net'] < 0).astype(int)
        groups = (is_loss != is_loss.shift()).cumsum()
        consecutive = is_loss.groupby(groups).cumsum()

        return consecutive.max()

    # =========================================================================
    # Statistical Significance
    # =========================================================================

    def t_statistic(self) -> float:
        """T-statistic for trade PnL."""
        if len(self.trades) < 2:
            return 0.0

        pnl = self.trades['pnl_net']
        t_stat, _ = stats.ttest_1samp(pnl, 0)

        return t_stat

    def p_value(self) -> float:
        """P-value for statistical significance of edge."""
        if len(self.trades) < 2:
            return 1.0

        pnl = self.trades['pnl_net']
        _, p_val = stats.ttest_1samp(pnl, 0)

        return p_val

    # =========================================================================
    # Summary Report
    # =========================================================================

    def generate_report(self) -> str:
        """Generate formatted performance report."""
        metrics = self.calculate_all_metrics()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              BACKTEST PERFORMANCE REPORT                      ║
╠══════════════════════════════════════════════════════════════╣

OVERVIEW
  Total Trades:            {metrics['total_trades']}
  Winning Trades:          {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)
  Losing Trades:           {metrics['losing_trades']} ({metrics['loss_rate']:.1f}%)

PRIMARY METRIC
  Profit Factor:           {metrics['profit_factor']:.2f}

RETURNS
  Total PnL:               ${metrics['total_pnl']:.2f}
  Total Return:            {metrics['total_return_pct']:.2f}%
  Avg Trade PnL:           ${metrics['avg_trade_pnl']:.2f}
  Expectancy:              ${metrics['expectancy']:.2f}

RISK-ADJUSTED
  Sharpe Ratio:            {metrics['sharpe_ratio']:.2f}
  Sortino Ratio:           {metrics['sortino_ratio']:.2f}
  Calmar Ratio:            {metrics['calmar_ratio']:.2f}

DRAWDOWN
  Max Drawdown:            {metrics['max_drawdown_pct']:.2f}%
  Avg Drawdown:            {metrics['avg_drawdown']:.2f}%

WIN/LOSS STATS
  Payoff Ratio:            {metrics['payoff_ratio']:.2f}
  Max Consecutive Wins:    {metrics['max_consecutive_wins']}
  Max Consecutive Losses:  {metrics['max_consecutive_losses']}

FREQUENCY
  Trades per Week:         {metrics['trades_per_week']:.1f}
  Trades per Month:        {metrics['trades_per_month']:.1f}

STATISTICAL SIGNIFICANCE
  T-Statistic:             {metrics['t_statistic']:.2f}
  P-Value:                 {metrics['p_value']:.4f}

╚══════════════════════════════════════════════════════════════╝
        """

        return report


if __name__ == "__main__":
    # Example usage with dummy data
    trades = pd.DataFrame({
        'entry_time': pd.date_range('2024-01-01', periods=50, freq='D'),
        'exit_time': pd.date_range('2024-01-02', periods=50, freq='D'),
        'pnl_net': np.random.normal(10, 50, 50),  # Random PnL
        'size': [1000] * 50,
        'bars_in_trade': np.random.randint(5, 30, 50)
    })

    equity_curve = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='12H'),
        'equity': 10000 + np.cumsum(np.random.normal(5, 20, 100))
    }).set_index('timestamp')

    metrics = PerformanceMetrics(trades, equity_curve)
    print(metrics.generate_report())
