"""
Rules Engine for Trade Signal Generation
Evaluates TradeSpec conditions and generates entry/exit signals.

Takes a TradeSpec and feature DataFrame, outputs trade signals with
entry/exit levels.

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from playbook_specs import TradeSpec, EntryCondition

logger = logging.getLogger(__name__)

# Safe built-in functions for eval expressions
SAFE_BUILTINS = {
    'abs': abs,
    'min': min,
    'max': max,
    'round': round,
    'len': len,
    'sum': sum,
    'any': any,
    'all': all,
}


class RulesEngine:
    """
    Evaluates trade rules and generates signals from TradeSpec.
    """

    def __init__(self, spec: TradeSpec, parameters: Optional[Dict] = None):
        """
        Initialize rules engine with a TradeSpec.

        Args:
            spec: TradeSpec defining the strategy
            parameters: Optional dict of parameter overrides
        """
        self.spec = spec
        self.parameters = self._build_parameters(parameters)

        logger.info(f"RulesEngine initialized for strategy: {spec.meta.id}")

    def _build_parameters(self, overrides: Optional[Dict] = None) -> Dict:
        """
        Build parameter dictionary from TradeSpec defaults and overrides.

        Args:
            overrides: Dictionary of parameter overrides

        Returns:
            Complete parameter dictionary
        """
        params = {}

        # Start with defaults from spec
        for param in self.spec.parameters:
            params[param.name] = param.default_value

        # Apply overrides if provided
        if overrides:
            params.update(overrides)

        return params

    def evaluate_condition(
        self,
        condition: EntryCondition,
        df: pd.DataFrame,
        idx: int,
        side: str,
        context: Dict
    ) -> bool:
        """
        Evaluate a single entry condition.

        Args:
            condition: EntryCondition to evaluate
            df: DataFrame with features
            idx: Current bar index
            side: 'long' or 'short'
            context: Additional context variables

        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Build evaluation context with current bar data
            eval_context = {
                'close': df['close'].iloc[idx],
                'open': df['open'].iloc[idx],
                'high': df['high'].iloc[idx],
                'low': df['low'].iloc[idx],
                'volume': df['volume'].iloc[idx],
                'side': side,
                **self.parameters,  # Include all parameters
                **context,  # Include any custom context
            }

            # Add all DataFrame columns available at this point in time
            for col in df.columns:
                if col in df:
                    eval_context[col] = df[col].iloc[idx]

            # Evaluate expression with safe builtins
            result = eval(condition.expression, {"__builtins__": SAFE_BUILTINS}, eval_context)

            return bool(result)

        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition.description}': {e}")
            return False

    def generate_signals(
        self,
        df: pd.DataFrame,
        allow_long: bool = True,
        allow_short: bool = True
    ) -> pd.DataFrame:
        """
        Generate entry signals for entire DataFrame.

        Args:
            df: DataFrame with features
            allow_long: Whether to generate long signals
            allow_short: Whether to generate short signals

        Returns:
            DataFrame with signal columns added:
            - 'signal': 1 = long, -1 = short, 0 = no signal
            - 'entry_price': Entry price for signal
            - 'stop_loss': Stop loss level
            - 'take_profit': Take profit level
        """
        logger.info(f"Generating signals for {len(df)} bars...")

        df = df.copy()

        # Initialize signal columns
        df['signal'] = 0
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        # Check if all required data fields are present
        missing_fields = [f for f in self.spec.required_data_fields if f not in df.columns]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return df

        # Iterate through bars
        for idx in range(len(df)):
            # Build context for this bar
            context = self._build_bar_context(df, idx)

            # Check filters
            if not self._check_filters(df, idx, context):
                continue

            # Check long conditions
            if allow_long and self.spec.entry_logic.allow_long:
                if self._check_entry_conditions(df, idx, 'long', context):
                    df.loc[df.index[idx], 'signal'] = 1
                    df.loc[df.index[idx], 'entry_price'] = df['close'].iloc[idx]

                    # Calculate stop and target
                    sl, tp = self._calculate_exit_levels(df, idx, 'long', context)
                    df.loc[df.index[idx], 'stop_loss'] = sl
                    df.loc[df.index[idx], 'take_profit'] = tp

            # Check short conditions
            if allow_short and self.spec.entry_logic.allow_short:
                if self._check_entry_conditions(df, idx, 'short', context):
                    df.loc[df.index[idx], 'signal'] = -1
                    df.loc[df.index[idx], 'entry_price'] = df['close'].iloc[idx]

                    # Calculate stop and target
                    sl, tp = self._calculate_exit_levels(df, idx, 'short', context)
                    df.loc[df.index[idx], 'stop_loss'] = sl
                    df.loc[df.index[idx], 'take_profit'] = tp

        num_signals = (df['signal'] != 0).sum()
        logger.info(f"Generated {num_signals} signals ({(df['signal'] == 1).sum()} long, "
                   f"{(df['signal'] == -1).sum()} short)")

        return df

    def _build_bar_context(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Build context dictionary for current bar evaluation.

        Args:
            df: DataFrame
            idx: Current bar index

        Returns:
            Dict with context variables
        """
        context = {}

        # Add recent values for pattern matching
        if idx > 0:
            context['prev_close'] = df['close'].iloc[idx - 1]
            context['prev_high'] = df['high'].iloc[idx - 1]
            context['prev_low'] = df['low'].iloc[idx - 1]
            context['prev_volume'] = df['volume'].iloc[idx - 1]

        # Add indicator-specific context
        # (Can be extended based on strategy needs)

        return context

    def _check_filters(self, df: pd.DataFrame, idx: int, context: Dict) -> bool:
        """
        Check if entry filters pass.

        Args:
            df: DataFrame
            idx: Current bar index
            context: Bar context

        Returns:
            True if all filters pass
        """
        filters = self.spec.filters

        # Volume filter
        if filters.min_volume_vs_avg is not None:
            if 'volume_vs_avg' in df.columns:
                if df['volume_vs_avg'].iloc[idx] < filters.min_volume_vs_avg:
                    return False

        if filters.max_volume_vs_avg is not None:
            if 'volume_vs_avg' in df.columns:
                if df['volume_vs_avg'].iloc[idx] > filters.max_volume_vs_avg:
                    return False

        # Session filters (low liquidity hours)
        if filters.avoid_low_liquidity_hours:
            hour = df.index[idx].hour
            # Avoid typically low-liquidity hours (customize as needed)
            if hour in [22, 23, 0, 1, 2]:  # Late night UTC
                return False

        # Custom filters
        for custom_filter in filters.custom_filters:
            try:
                eval_context = {
                    **self.parameters,
                    **context,
                }
                # Add current bar data
                for col in df.columns:
                    eval_context[col] = df[col].iloc[idx]

                if not eval(custom_filter, {"__builtins__": SAFE_BUILTINS}, eval_context):
                    return False

            except Exception as e:
                logger.warning(f"Error evaluating custom filter '{custom_filter}': {e}")
                return False

        # Min bars between entries
        if filters.min_bars_between_entries > 0:
            # Check if there was a recent signal
            lookback = min(idx, filters.min_bars_between_entries)
            if lookback > 0:
                recent_signals = df['signal'].iloc[idx - lookback:idx]
                if (recent_signals != 0).any():
                    return False

        return True

    def _check_entry_conditions(
        self,
        df: pd.DataFrame,
        idx: int,
        side: str,
        context: Dict
    ) -> bool:
        """
        Check if all entry conditions are met.

        Args:
            df: DataFrame
            idx: Current bar index
            side: 'long' or 'short'
            context: Bar context

        Returns:
            True if all required conditions are met
        """
        for condition in self.spec.entry_logic.conditions:
            if condition.required:
                if not self.evaluate_condition(condition, df, idx, side, context):
                    return False
            else:
                # Optional condition - just evaluate but don't fail if false
                self.evaluate_condition(condition, df, idx, side, context)

        return True

    def _calculate_exit_levels(
        self,
        df: pd.DataFrame,
        idx: int,
        side: str,
        context: Dict
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.

        Args:
            df: DataFrame
            idx: Current bar index
            side: 'long' or 'short'
            context: Bar context

        Returns:
            Tuple of (stop_loss, take_profit)
        """
        entry_price = df['close'].iloc[idx]

        # Get first stop loss rule (highest priority)
        sl_rules = sorted(self.spec.exit_logic.stop_loss_rules, key=lambda x: x.priority)
        tp_rules = sorted(self.spec.exit_logic.take_profit_rules, key=lambda x: x.priority)

        # Calculate stop loss
        stop_loss = entry_price * 0.95 if side == 'long' else entry_price * 1.05  # Default 5%

        if sl_rules:
            try:
                sl_rule = sl_rules[0].rule
                eval_context = {
                    'entry_price': entry_price,
                    'side': side,
                    **self.parameters,
                    **context,
                }
                for col in df.columns:
                    eval_context[col] = df[col].iloc[idx]

                stop_loss = eval(sl_rule, {"__builtins__": SAFE_BUILTINS}, eval_context)
            except Exception as e:
                logger.warning(f"Error calculating stop loss: {e}, using default")

        # Calculate take profit
        take_profit = entry_price * 1.02 if side == 'long' else entry_price * 0.98  # Default 2%

        if tp_rules:
            try:
                tp_rule = tp_rules[0].rule
                eval_context = {
                    'entry_price': entry_price,
                    'side': side,
                    **self.parameters,
                    **context,
                }
                for col in df.columns:
                    eval_context[col] = df[col].iloc[idx]

                take_profit = eval(tp_rule, {"__builtins__": SAFE_BUILTINS}, eval_context)
            except Exception as e:
                logger.warning(f"Error calculating take profit: {e}, using default")

        return stop_loss, take_profit


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline
    from indicators import add_all_indicators
    from playbook_specs import get_small_ib_momentum_spec

    # Load data
    pipeline = DataPipeline("config.yaml")
    df = pipeline.get_full_dataset(timeframe="5m", start_date="2024-01-01", end_date="2024-01-31")

    # Add indicators
    df = add_all_indicators(df)

    # Get strategy spec
    spec = get_small_ib_momentum_spec()

    # Initialize rules engine
    engine = RulesEngine(spec)

    # Generate signals
    df_with_signals = engine.generate_signals(df)

    print(f"\nSignals generated:")
    print(f"Long signals: {(df_with_signals['signal'] == 1).sum()}")
    print(f"Short signals: {(df_with_signals['signal'] == -1).sum()}")

    # Show sample signals
    signals = df_with_signals[df_with_signals['signal'] != 0]
    if len(signals) > 0:
        print(f"\nSample signals:")
        print(signals[['signal', 'entry_price', 'stop_loss', 'take_profit']].head())
