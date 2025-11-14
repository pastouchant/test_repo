"""
Feature Engineering and Label Generation for ML Models
Creates time-safe feature matrices and labels for strategy optimization.

Key responsibilities:
1. Build feature matrices from indicators (no lookahead bias)
2. Generate labels (TP hit vs SL hit, net of costs)
3. Session/regime features
4. Ensure all features are point-in-time valid

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Label Generation (for ML training)
# =============================================================================

def generate_trade_labels(
    df: pd.DataFrame,
    tp_distance_pct: float = 1.0,
    sl_distance_pct: float = 0.5,
    max_bars_in_trade: int = 60,
    fees_pct: float = 0.0006,  # Taker fee
    side: str = 'long'
) -> pd.Series:
    """
    Generate labels for each bar: did a hypothetical trade hit TP or SL first?

    Label values:
    - 1: TP hit first (good trade)
    - 0: SL hit first (bad trade)
    - -1: Neither hit within max_bars (unclear/timeout)

    Args:
        df: DataFrame with OHLC data
        tp_distance_pct: Take profit distance as % of entry price
        sl_distance_pct: Stop loss distance as % of entry price
        max_bars_in_trade: Maximum holding period
        fees_pct: Trading fees to account for
        side: 'long' or 'short'

    Returns:
        Series with labels
    """
    labels = pd.Series(-1, index=df.index)  # Default: unclear

    for i in range(len(df) - max_bars_in_trade):
        entry_price = df['close'].iloc[i]

        if side == 'long':
            tp_price = entry_price * (1 + tp_distance_pct / 100)
            sl_price = entry_price * (1 - sl_distance_pct / 100)

            # Look ahead to see which is hit first
            future_highs = df['high'].iloc[i+1:i+1+max_bars_in_trade]
            future_lows = df['low'].iloc[i+1:i+1+max_bars_in_trade]

            tp_hit_idx = (future_highs >= tp_price).idxmax() if (future_highs >= tp_price).any() else None
            sl_hit_idx = (future_lows <= sl_price).idxmax() if (future_lows <= sl_price).any() else None

            if tp_hit_idx is not None and sl_hit_idx is not None:
                # Both hit, check which came first
                if df.index.get_loc(tp_hit_idx) < df.index.get_loc(sl_hit_idx):
                    labels.iloc[i] = 1  # TP first
                else:
                    labels.iloc[i] = 0  # SL first
            elif tp_hit_idx is not None:
                labels.iloc[i] = 1  # Only TP hit
            elif sl_hit_idx is not None:
                labels.iloc[i] = 0  # Only SL hit
            # else: neither hit, label remains -1

        elif side == 'short':
            tp_price = entry_price * (1 - tp_distance_pct / 100)
            sl_price = entry_price * (1 + sl_distance_pct / 100)

            future_lows = df['low'].iloc[i+1:i+1+max_bars_in_trade]
            future_highs = df['high'].iloc[i+1:i+1+max_bars_in_trade]

            tp_hit_idx = (future_lows <= tp_price).idxmax() if (future_lows <= tp_price).any() else None
            sl_hit_idx = (future_highs >= sl_price).idxmax() if (future_highs >= sl_price).any() else None

            if tp_hit_idx is not None and sl_hit_idx is not None:
                if df.index.get_loc(tp_hit_idx) < df.index.get_loc(sl_hit_idx):
                    labels.iloc[i] = 1  # TP first
                else:
                    labels.iloc[i] = 0  # SL first
            elif tp_hit_idx is not None:
                labels.iloc[i] = 1
            elif sl_hit_idx is not None:
                labels.iloc[i] = 0

    logger.info(f"Generated labels: {(labels == 1).sum()} TP hits, "
                f"{(labels == 0).sum()} SL hits, {(labels == -1).sum()} unclear")

    return labels


# =============================================================================
# Session and Regime Features
# =============================================================================

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session-based features (Asia, London, NY).

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with session features added
    """
    df = df.copy()

    hour = df.index.hour

    # Session indicators (UTC hours)
    df['is_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['is_london'] = ((hour >= 7) & (hour < 16)).astype(int)
    df['is_ny'] = ((hour >= 12) & (hour < 21)).astype(int)
    df['is_ny_eth'] = ((hour >= 21) | (hour < 0)).astype(int)  # NY extended hours

    # Session overlaps
    df['is_london_ny_overlap'] = ((hour >= 12) & (hour < 16)).astype(int)

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime features (trend, volatility).

    Args:
        df: DataFrame with indicators

    Returns:
        DataFrame with regime features added
    """
    df = df.copy()

    # Trend regime (using EMAs if available)
    if 'ema_50' in df.columns and 'ema_200' in df.columns:
        df['bullish_regime'] = (df['ema_50'] > df['ema_200']).astype(int)
        df['bearish_regime'] = (df['ema_50'] < df['ema_200']).astype(int)

    # Price vs VWAP
    if 'vwap' in df.columns:
        df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
        df['vwap_distance_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100

    # Volatility regime
    if 'atr_14' in df.columns:
        atr_ma = df['atr_14'].rolling(50).mean()
        df['high_volatility'] = (df['atr_14'] > atr_ma * 1.5).astype(int)
        df['low_volatility'] = (df['atr_14'] < atr_ma * 0.7).astype(int)

    # IB compression (if IB available)
    if 'ib_range' in df.columns and 'asr_20' in df.columns:
        df['ib_compressed'] = (df['ib_range'] < df['asr_20']).astype(int)
        df['ib_vs_asr'] = df['ib_range'] / df['asr_20'].replace(0, np.nan)

    # Inside day detection
    if all(col in df.columns for col in ['pdh', 'pdl']):
        df['is_inside_day'] = ((df['high'] < df['pdh']) & (df['low'] > df['pdl'])).astype(int)

    return df


# =============================================================================
# Orderflow Features
# =============================================================================

def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add orderflow-specific features (CVD, delta, OI).

    Args:
        df: DataFrame with orderflow data

    Returns:
        DataFrame with orderflow features
    """
    df = df.copy()

    # Delta features
    if 'delta' in df.columns:
        # Rolling delta sum
        df['delta_sum_5'] = df['delta'].rolling(5).sum()
        df['delta_sum_10'] = df['delta'].rolling(10).sum()

        # Delta vs volume
        df['delta_ratio'] = df['delta'] / df['volume'].replace(0, np.nan)

        # Delta momentum
        df['delta_acceleration'] = df['delta'].diff()

    # CVD features
    if 'cvd' in df.columns:
        # CVD slope (rate of change)
        df['cvd_slope_5'] = df['cvd'].diff(5)
        df['cvd_slope_10'] = df['cvd'].diff(10)

        # CVD vs price correlation (simplified)
        df['cvd_price_corr'] = df['cvd'].rolling(20).corr(df['close'])

    # OI features
    if 'oi' in df.columns:
        # OI momentum
        df['oi_momentum'] = df['oi'].pct_change(5)

        # OI vs volume
        df['oi_volume_ratio'] = df['oi'] / df['volume'].replace(0, np.nan)

    return df


# =============================================================================
# Price Action Features
# =============================================================================

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price action and pattern features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with price action features
    """
    df = df.copy()

    # Returns
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_10'] = df['close'].pct_change(10)

    # Momentum
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)

    # Candle characteristics
    df['body_size'] = abs(df['close'] - df['open'])
    df['range_size'] = df['high'] - df['low']
    df['body_to_range'] = df['body_size'] / df['range_size'].replace(0, np.nan)

    # Is bullish/bearish candle
    df['is_bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['is_bearish_candle'] = (df['close'] < df['open']).astype(int)

    # Range position (where did candle close within its range?)
    df['close_position_in_range'] = (df['close'] - df['low']) / df['range_size'].replace(0, np.nan)

    # Distance from key levels
    if 'pdh' in df.columns:
        df['distance_from_pdh'] = (df['close'] - df['pdh']) / df['pdh']
    if 'pdl' in df.columns:
        df['distance_from_pdl'] = (df['close'] - df['pdl']) / df['pdl']

    # Consecutive candles in same direction
    df['consecutive_up'] = (df['close'] > df['open']).astype(int).groupby(
        (df['close'] <= df['open']).cumsum()).cumsum()
    df['consecutive_down'] = (df['close'] < df['open']).astype(int).groupby(
        (df['close'] >= df['open']).cumsum()).cumsum()

    return df


# =============================================================================
# Volume Profile Features
# =============================================================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features.

    Args:
        df: DataFrame with volume data

    Returns:
        DataFrame with volume features
    """
    df = df.copy()

    # Volume moving averages
    df['volume_ma_10'] = df['volume'].rolling(10).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()

    # Volume ratio vs MA
    if 'volume_vs_avg' not in df.columns:
        df['volume_vs_avg'] = df['volume'] / df['volume_ma_20']

    # Volume acceleration
    df['volume_change'] = df['volume'].diff()
    df['volume_acceleration'] = df['volume_change'].diff()

    # High/low volume flags
    df['is_high_volume'] = (df['volume'] > df['volume_ma_20'] * 1.5).astype(int)
    df['is_low_volume'] = (df['volume'] < df['volume_ma_20'] * 0.5).astype(int)

    return df


# =============================================================================
# Master Feature Builder
# =============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    include_labels: bool = False,
    label_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Build complete feature matrix for ML.

    Args:
        df: DataFrame with OHLCV and indicators
        include_labels: Whether to generate labels
        label_params: Parameters for label generation

    Returns:
        DataFrame with all features (and optionally labels)
    """
    logger.info("Building feature matrix...")

    df = df.copy()

    # Add all feature groups
    df = add_session_features(df)
    df = add_regime_features(df)
    df = add_orderflow_features(df)
    df = add_price_action_features(df)
    df = add_volume_features(df)

    # Generate labels if requested
    if include_labels:
        if label_params is None:
            label_params = {
                'tp_distance_pct': 1.0,
                'sl_distance_pct': 0.5,
                'max_bars_in_trade': 60,
                'fees_pct': 0.0006,
                'side': 'long'
            }

        df['label'] = generate_trade_labels(df, **label_params)

    # Drop rows with NaN in critical features (from indicator warm-up period)
    initial_rows = len(df)
    df.dropna(subset=['close', 'volume'], inplace=True)
    logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values (indicator warm-up)")

    logger.info(f"Feature matrix built: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def get_feature_columns(df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
    """
    Get list of feature columns (excluding OHLCV, labels, metadata).

    Args:
        df: DataFrame with features
        exclude_targets: Whether to exclude label/target columns

    Returns:
        List of feature column names
    """
    # Columns to exclude from features
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume',  # Raw OHLCV
        'timestamp', 'date', 'hour', 'day_of_week',  # Metadata
    ]

    if exclude_targets:
        exclude_cols.extend(['label', 'target', 'future_return'])

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def train_test_split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    gap: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets chronologically.

    Args:
        df: DataFrame to split
        train_ratio: Fraction of data to use for training
        gap: Number of rows to skip between train and test (prevent leakage)

    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx + gap:]

    logger.info(f"Train/test split: {len(train_df)} train / {len(test_df)} test")

    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline
    from indicators import add_all_indicators

    pipeline = DataPipeline("config.yaml")
    df = pipeline.get_full_dataset(timeframe="5m", start_date="2024-01-01", end_date="2024-02-29")

    # Add indicators
    df = add_all_indicators(df)

    # Build feature matrix
    df_features = build_feature_matrix(df, include_labels=True)

    print(f"\nFeature matrix shape: {df_features.shape}")
    print(f"\nFeature columns:")
    feature_cols = get_feature_columns(df_features)
    for col in feature_cols[:20]:  # Print first 20
        print(f"  - {col}")

    print(f"\nLabel distribution:")
    print(df_features['label'].value_counts())
