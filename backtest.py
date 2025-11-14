"""
Backtesting Engine for BTCUSDT Perpetual Strategies
Simulates trades with realistic costs: fees, funding, slippage.

Produces detailed trade logs and equity curves.

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    entry_price: float
    side: str  # 'long' or 'short'
    size: float  # Position size in contracts/USD
    stop_loss: float
    take_profit: float

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp', 'sl', 'time', 'manual'

    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    slippage: float = 0.0

    bars_in_trade: int = 0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    def to_dict(self) -> Dict:
        """Convert trade to dictionary"""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'side': self.side,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'fees': self.fees,
            'funding': self.funding,
            'slippage': self.slippage,
            'bars_in_trade': self.bars_in_trade,
            'mae': self.mae,
            'mfe': self.mfe,
        }


class BacktestEngine:
    """
    Event-driven backtest engine for perpetual futures.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize backtest engine.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.initial_capital = self.config['risk']['initial_capital_usd']
        self.maker_fee = self.config['costs']['maker_fee']
        self.taker_fee = self.config['costs']['taker_fee']
        self.slippage_bps = self.config['costs']['slippage_bps']

        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        self.current_position: Optional[Trade] = None

        logger.info(f"BacktestEngine initialized with ${self.initial_capital:,.0f} capital")

    def run(
        self,
        df: pd.DataFrame,
        signals_col: str = 'signal',
        max_bars_in_trade: Optional[int] = None
    ) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Run backtest on DataFrame with signals.

        Args:
            df: DataFrame with signals, entry_price, stop_loss, take_profit columns
            signals_col: Column name containing signals (1=long, -1=short, 0=nothing)
            max_bars_in_trade: Maximum bars to hold a position

        Returns:
            Tuple of (trade_list, equity_curve_df)
        """
        logger.info(f"Running backtest on {len(df)} bars...")

        # Reset state
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.current_position = None

        # Iterate through bars
        for idx in range(len(df)):
            bar = df.iloc[idx]
            timestamp = df.index[idx]

            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'peak_equity': self.peak_equity,
                'drawdown': (self.equity - self.peak_equity) / self.peak_equity * 100,
                'position': 1 if self.current_position else 0
            })

            # Check exit for existing position
            if self.current_position:
                exit_signal = self._check_exit(bar, self.current_position, max_bars_in_trade)

                if exit_signal:
                    exit_reason, exit_price = exit_signal
                    self._close_position(timestamp, exit_price, exit_reason)

            # Check for new entry signal
            if not self.current_position:
                signal = bar[signals_col]

                if signal != 0:
                    # Open new position
                    side = 'long' if signal == 1 else 'short'
                    entry_price = bar['entry_price']
                    stop_loss = bar['stop_loss']
                    take_profit = bar['take_profit']

                    # Calculate position size
                    size = self._calculate_position_size(entry_price, stop_loss, side)

                    self._open_position(
                        timestamp,
                        entry_price,
                        side,
                        size,
                        stop_loss,
                        take_profit
                    )

        # Close any open position at end
        if self.current_position:
            final_bar = df.iloc[-1]
            self._close_position(df.index[-1], final_bar['close'], 'end_of_data')

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

        # Build equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)

        return self.trades, equity_df

    def _open_position(
        self,
        timestamp: datetime,
        entry_price: float,
        side: str,
        size: float,
        stop_loss: float,
        take_profit: float
    ):
        """Open a new position."""
        # Apply slippage
        slippage_amount = entry_price * (self.slippage_bps / 10000)
        if side == 'long':
            actual_entry_price = entry_price + slippage_amount
        else:
            actual_entry_price = entry_price - slippage_amount

        # Calculate entry fee (taker fee for market orders)
        entry_fee = size * self.taker_fee

        # Deduct fee from equity
        self.equity -= entry_fee

        # Create trade object
        trade = Trade(
            entry_time=timestamp,
            entry_price=actual_entry_price,
            side=side,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees=entry_fee,
            slippage=slippage_amount * size
        )

        self.current_position = trade

        logger.debug(f"Opened {side} position: {size:.2f} @ {actual_entry_price:.2f}")

    def _close_position(self, timestamp: datetime, exit_price: float, reason: str):
        """Close current position."""
        if not self.current_position:
            return

        trade = self.current_position

        # Apply slippage on exit
        slippage_amount = exit_price * (self.slippage_bps / 10000)
        if trade.side == 'long':
            actual_exit_price = exit_price - slippage_amount
        else:
            actual_exit_price = exit_price + slippage_amount

        # Calculate gross PnL
        if trade.side == 'long':
            pnl_gross = (actual_exit_price - trade.entry_price) * trade.size / trade.entry_price
        else:
            pnl_gross = (trade.entry_price - actual_exit_price) * trade.size / trade.entry_price

        # Calculate exit fee
        exit_fee = trade.size * self.taker_fee

        # Total fees
        total_fees = trade.fees + exit_fee

        # Total slippage
        total_slippage = trade.slippage + (slippage_amount * trade.size)

        # Net PnL
        pnl_net = pnl_gross - total_fees - total_slippage

        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = actual_exit_price
        trade.exit_reason = reason
        trade.pnl_gross = pnl_gross
        trade.pnl_net = pnl_net
        trade.fees = total_fees
        trade.slippage = total_slippage
        trade.bars_in_trade = (timestamp - trade.entry_time).total_seconds() / 60 / 5  # Assuming 5m bars

        # Update equity
        self.equity += pnl_net

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Store trade
        self.trades.append(trade)

        logger.debug(f"Closed {trade.side} position: PnL = ${pnl_net:.2f}, Reason = {reason}")

        # Clear current position
        self.current_position = None

    def _check_exit(
        self,
        bar: pd.Series,
        trade: Trade,
        max_bars: Optional[int]
    ) -> Optional[Tuple[str, float]]:
        """
        Check if position should be exited.

        Returns:
            Tuple of (exit_reason, exit_price) if exit triggered, None otherwise
        """
        # Check stop loss
        if trade.side == 'long':
            if bar['low'] <= trade.stop_loss:
                return ('sl', trade.stop_loss)
        else:
            if bar['high'] >= trade.stop_loss:
                return ('sl', trade.stop_loss)

        # Check take profit
        if trade.side == 'long':
            if bar['high'] >= trade.take_profit:
                return ('tp', trade.take_profit)
        else:
            if bar['low'] <= trade.take_profit:
                return ('tp', trade.take_profit)

        # Check time-based exit
        if max_bars is not None:
            bars_in_trade = (bar.name - trade.entry_time).total_seconds() / 60 / 5
            if bars_in_trade >= max_bars:
                return ('time', bar['close'])

        return None

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        side: str
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: 'long' or 'short'

        Returns:
            Position size in USD
        """
        risk_per_trade_pct = self.config['risk']['default_risk_per_trade_pct']
        max_leverage = self.config['risk']['max_leverage']

        # Calculate risk amount
        risk_amount = self.equity * (risk_per_trade_pct / 100)

        # Calculate distance to stop as percentage
        if side == 'long':
            stop_distance_pct = abs((entry_price - stop_loss) / entry_price)
        else:
            stop_distance_pct = abs((stop_loss - entry_price) / entry_price)

        # Position size = risk amount / stop distance
        if stop_distance_pct > 0:
            position_size = risk_amount / stop_distance_pct
        else:
            position_size = self.equity * 0.1  # Fallback: 10% of equity

        # Apply leverage limit
        max_position_size = self.equity * max_leverage
        position_size = min(position_size, max_position_size)

        # Apply min/max limits
        min_size = self.config['risk']['min_position_size_usd']
        position_size = max(position_size, min_size)

        return position_size

    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame.

        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()

        trade_dicts = [t.to_dict() for t in self.trades]
        df = pd.DataFrame(trade_dicts)

        return df


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline
    from indicators import add_all_indicators
    from playbook_specs import get_small_ib_momentum_spec
    from rules_engine import RulesEngine

    # Load data
    pipeline = DataPipeline("config.yaml")
    df = pipeline.get_full_dataset(timeframe="5m", start_date="2024-01-01", end_date="2024-02-29")

    # Add indicators
    df = add_all_indicators(df)

    # Generate signals
    spec = get_small_ib_momentum_spec()
    engine = RulesEngine(spec)
    df_with_signals = engine.generate_signals(df)

    # Run backtest
    bt = BacktestEngine("config.yaml")
    trades, equity_curve = bt.run(df_with_signals, max_bars_in_trade=60)

    print(f"\n=== Backtest Results ===")
    print(f"Total trades: {len(trades)}")
    if len(trades) > 0:
        trade_log = bt.get_trade_log()
        print(f"\nWinning trades: {(trade_log['pnl_net'] > 0).sum()}")
        print(f"Losing trades: {(trade_log['pnl_net'] < 0).sum()}")
        print(f"Total PnL: ${trade_log['pnl_net'].sum():.2f}")
        print(f"Final equity: ${bt.equity:.2f}")
        print(f"Return: {((bt.equity / bt.initial_capital) - 1) * 100:.2f}%")

        print(f"\nSample trades:")
        print(trade_log[['entry_time', 'side', 'entry_price', 'exit_price', 'pnl_net', 'exit_reason']].head())
