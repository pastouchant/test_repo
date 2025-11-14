"""
Technical Indicators and Orderflow Metrics for BTCUSDT Trading
Implements all indicators required by Flow Playbook strategies.

Includes:
- VWAP (session, daily, weekly) with standard deviation bands
- Range calculations (IB, Monday, Asia, London, PDH/PDL)
- CVD and delta metrics
- OI and funding metrics
- Wick measurements
- Volume features
- Standard technical indicators (EMA, ATR, etc.)

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# VWAP Calculations
# =============================================================================

def calculate_vwap(df: pd.DataFrame, session: str = "daily") -> pd.Series:
    """
    Calculate Volume-Weighted Average Price.

    Args:
        df: DataFrame with OHLCV data
        session: 'daily', 'weekly', or 'session' (resets at session start)

    Returns:
        Series with VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    if session == "daily":
        # Reset VWAP daily
        vwap = (typical_price * df['volume']).groupby(df.index.date).cumsum() / \
               df['volume'].groupby(df.index.date).cumsum()
    elif session == "weekly":
        # Reset VWAP weekly
        vwap = (typical_price * df['volume']).groupby(df.index.isocalendar().week).cumsum() / \
               df['volume'].groupby(df.index.isocalendar().week).cumsum()

    return vwap


def calculate_vwap_bands(df: pd.DataFrame, num_std: float = 1.0) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate VWAP standard deviation bands.

    Args:
        df: DataFrame with OHLCV and VWAP
        num_std: Number of standard deviations

    Returns:
        Tuple of (upper_band, lower_band)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    if 'vwap' not in df.columns:
        vwap = calculate_vwap(df)
    else:
        vwap = df['vwap']

    # Calculate variance
    variance = ((typical_price - vwap) ** 2 * df['volume']).cumsum() / df['volume'].cumsum()
    std = np.sqrt(variance)

    upper_band = vwap + (num_std * std)
    lower_band = vwap - (num_std * std)

    return upper_band, lower_band


# =============================================================================
# Range Calculations
# =============================================================================

def calculate_initial_balance(
    df: pd.DataFrame,
    session_start_hour: int = 0,
    ib_duration_hours: int = 1
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Initial Balance (first hour of session).

    Args:
        df: DataFrame with OHLCV data
        session_start_hour: Hour when session starts (UTC)
        ib_duration_hours: Duration of IB period in hours

    Returns:
        Tuple of (ib_high, ib_low, ib_range)
    """
    df = df.copy()

    # Identify IB periods (first hour of each day/session)
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    ib_high = pd.Series(index=df.index, dtype=float)
    ib_low = pd.Series(index=df.index, dtype=float)

    for date in df['date'].unique():
        day_data = df[df['date'] == date]

        # Find IB period (first hour of trading)
        ib_mask = (day_data['hour'] >= session_start_hour) & \
                  (day_data['hour'] < session_start_hour + ib_duration_hours)

        if ib_mask.any():
            ib_data = day_data[ib_mask]
            day_ib_high = ib_data['high'].max()
            day_ib_low = ib_data['low'].min()

            # Forward fill IB levels for the rest of the day
            ib_high[df['date'] == date] = day_ib_high
            ib_low[df['date'] == date] = day_ib_low

    ib_range = ib_high - ib_low

    return ib_high, ib_low, ib_range


def calculate_monday_range(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Monday's high and low for the week.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Tuple of (monday_high, monday_low)
    """
    df = df.copy()
    df['weekday'] = df.index.dayofweek
    df['week'] = df.index.isocalendar().week

    monday_high = pd.Series(index=df.index, dtype=float)
    monday_low = pd.Series(index=df.index, dtype=float)

    for week in df['week'].unique():
        week_data = df[df['week'] == week]
        monday_data = week_data[week_data['weekday'] == 0]  # Monday = 0

        if not monday_data.empty:
            mon_high = monday_data['high'].max()
            mon_low = monday_data['low'].min()

            # Apply to entire week
            monday_high[df['week'] == week] = mon_high
            monday_low[df['week'] == week] = mon_low

    return monday_high, monday_low


def calculate_session_range(
    df: pd.DataFrame,
    session: str = "asia"
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate session-specific high/low ranges.

    Sessions (UTC times):
    - Asia: 00:00 - 08:00
    - London: 07:00 - 16:00
    - NY: 12:00 - 21:00

    Args:
        df: DataFrame with OHLCV data
        session: 'asia', 'london', or 'ny'

    Returns:
        Tuple of (session_high, session_low)
    """
    # Define session hours (UTC)
    session_hours = {
        'asia': (0, 8),
        'london': (7, 16),
        'ny': (12, 21)
    }

    if session not in session_hours:
        raise ValueError(f"Unknown session: {session}")

    start_hour, end_hour = session_hours[session]

    df = df.copy()
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    session_high = pd.Series(index=df.index, dtype=float)
    session_low = pd.Series(index=df.index, dtype=float)

    for date in df['date'].unique():
        day_data = df[df['date'] == date]

        # Find session period
        session_mask = (day_data['hour'] >= start_hour) & (day_data['hour'] < end_hour)

        if session_mask.any():
            session_data = day_data[session_mask]
            sess_high = session_data['high'].max()
            sess_low = session_data['low'].min()

            # Apply to entire day (for strategies that use session range later in day)
            session_high[df['date'] == date] = sess_high
            session_low[df['date'] == date] = sess_low

    return session_high, session_low


def calculate_prior_day_levels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate previous day high, low, and close.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Tuple of (pdh, pdl, pdc)
    """
    df = df.copy()
    df['date'] = df.index.date

    daily_high = df.groupby('date')['high'].max()
    daily_low = df.groupby('date')['low'].min()
    daily_close = df.groupby('date')['close'].last()

    pdh = df['date'].map(daily_high.shift(1))
    pdl = df['date'].map(daily_low.shift(1))
    pdc = df['date'].map(daily_close.shift(1))

    return pdh, pdl, pdc


def average_session_range(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate average session range (ASR) over lookback period.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of sessions to average

    Returns:
        Series with ASR values
    """
    df = df.copy()
    df['date'] = df.index.date

    # Calculate daily range
    daily_range = df.groupby('date').apply(lambda x: x['high'].max() - x['low'].min())

    # Rolling average
    asr = daily_range.rolling(window=lookback).mean()

    # Map back to original index
    asr_series = df['date'].map(asr)

    return asr_series


# =============================================================================
# CVD and Delta Metrics
# =============================================================================

def cumulative_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Cumulative Volume Delta (CVD).

    Note: Requires 'delta' column (buy volume - sell volume per bar).
    If delta is not available, this is a placeholder.

    Args:
        df: DataFrame with 'delta' column

    Returns:
        Series with CVD values
    """
    if 'delta' not in df.columns:
        logger.warning("Delta column not found, CVD will be zero")
        return pd.Series(0, index=df.index)

    cvd = df['delta'].cumsum()
    return cvd


def delta_change(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """
    Calculate change in delta over N periods.

    Args:
        df: DataFrame with 'delta' column
        periods: Number of periods for change calculation

    Returns:
        Series with delta change
    """
    if 'delta' not in df.columns:
        return pd.Series(0, index=df.index)

    return df['delta'].diff(periods)


def cvd_divergence(df: pd.DataFrame, price_col: str = 'close', window: int = 14) -> pd.Series:
    """
    Detect CVD divergence from price.

    Simplified divergence: when price makes new high but CVD doesn't (bearish divergence)
    or price makes new low but CVD doesn't (bullish divergence).

    Args:
        df: DataFrame with price and CVD data
        price_col: Column to use for price
        window: Lookback window for high/low detection

    Returns:
        Series with divergence signal (-1 = bearish, +1 = bullish, 0 = none)
    """
    if 'cvd' not in df.columns:
        return pd.Series(0, index=df.index)

    price = df[price_col]
    cvd = df['cvd']

    # Rolling highs and lows
    price_high = price.rolling(window).max()
    price_low = price.rolling(window).min()
    cvd_high = cvd.rolling(window).max()
    cvd_low = cvd.rolling(window).min()

    divergence = pd.Series(0, index=df.index)

    # Bearish divergence: new price high but CVD doesn't confirm
    bearish_div = (price == price_high) & (cvd < cvd_high)
    divergence[bearish_div] = -1

    # Bullish divergence: new price low but CVD doesn't confirm
    bullish_div = (price == price_low) & (cvd > cvd_low)
    divergence[bullish_div] = 1

    return divergence


# =============================================================================
# Open Interest Metrics
# =============================================================================

def oi_change(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """
    Calculate change in Open Interest.

    Args:
        df: DataFrame with 'oi' column
        periods: Number of periods for change

    Returns:
        Series with OI change
    """
    if 'oi' not in df.columns:
        return pd.Series(0, index=df.index)

    return df['oi'].diff(periods)


def oi_change_pct(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """
    Calculate percentage change in Open Interest.

    Args:
        df: DataFrame with 'oi' column
        periods: Number of periods for change

    Returns:
        Series with OI % change
    """
    if 'oi' not in df.columns:
        return pd.Series(0, index=df.index)

    return df['oi'].pct_change(periods) * 100


# =============================================================================
# Wick and Candle Measurements
# =============================================================================

def calculate_wick_ratios(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate wick ratios for candle analysis.

    Args:
        df: DataFrame with OHLC data

    Returns:
        Tuple of (upper_wick_ratio, lower_wick_ratio, total_wick_ratio)
    """
    candle_range = df['high'] - df['low']
    candle_range = candle_range.replace(0, np.nan)  # Avoid division by zero

    body_top = df[['open', 'close']].max(axis=1)
    body_bottom = df[['open', 'close']].min(axis=1)

    upper_wick = df['high'] - body_top
    lower_wick = body_bottom - df['low']

    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range
    total_wick_ratio = (upper_wick + lower_wick) / candle_range

    return upper_wick_ratio, lower_wick_ratio, total_wick_ratio


def is_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect engulfing candle patterns.

    Returns:
        Series with values: 1 = bullish engulfing, -1 = bearish engulfing, 0 = none
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)

    prev_high = df[['open', 'close']].shift(1).max(axis=1)
    prev_low = df[['open', 'close']].shift(1).min(axis=1)

    curr_high = df[['open', 'close']].max(axis=1)
    curr_low = df[['open', 'close']].min(axis=1)

    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulf = (prev_close < prev_open) & \
                     (df['close'] > df['open']) & \
                     (curr_low < prev_low) & \
                     (curr_high > prev_high)

    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulf = (prev_close > prev_open) & \
                     (df['close'] < df['open']) & \
                     (curr_high > prev_high) & \
                     (curr_low < prev_low)

    engulfing = pd.Series(0, index=df.index)
    engulfing[bullish_engulf] = 1
    engulfing[bearish_engulf] = -1

    return engulfing


# =============================================================================
# Volume Features
# =============================================================================

def volume_vs_average(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate volume relative to moving average.

    Args:
        df: DataFrame with 'volume' column
        window: Window for moving average

    Returns:
        Series with volume ratio (current / average)
    """
    vol_ma = df['volume'].rolling(window=window).mean()
    return df['volume'] / vol_ma


def volume_spike(df: pd.DataFrame, threshold: float = 2.0, window: int = 20) -> pd.Series:
    """
    Detect volume spikes.

    Args:
        df: DataFrame with 'volume' column
        threshold: Multiplier for spike detection (e.g., 2.0 = 2x average)
        window: Window for average calculation

    Returns:
        Series with boolean spike indicators
    """
    vol_ratio = volume_vs_average(df, window)
    return vol_ratio > threshold


# =============================================================================
# Standard Technical Indicators
# =============================================================================

def calculate_ema(df: pd.DataFrame, column: str = 'close', period: int = 50) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        df: DataFrame
        column: Column to calculate EMA on
        period: EMA period

    Returns:
        Series with EMA values
    """
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_sma(df: pd.DataFrame, column: str = 'close', period: int = 50) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        df: DataFrame
        column: Column to calculate SMA on
        period: SMA period

    Returns:
        Series with SMA values
    """
    return df[column].rolling(window=period).mean()


def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with OHLC data
        period: ATR period

    Returns:
        Series with ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_z_score(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Calculate z-score (standardized deviation from mean).

    Args:
        df: DataFrame
        column: Column to calculate z-score on
        window: Window for mean/std calculation

    Returns:
        Series with z-scores
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    z_score = (df[column] - rolling_mean) / rolling_std

    return z_score


# =============================================================================
# Comprehensive Indicator Builder
# =============================================================================

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicators to DataFrame.

    This is a convenience function that adds all commonly used indicators.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all indicators added
    """
    logger.info("Adding all indicators...")

    df = df.copy()

    # VWAP
    df['vwap'] = calculate_vwap(df, session='daily')
    df['vwap_upper'], df['vwap_lower'] = calculate_vwap_bands(df, num_std=1.0)

    # Ranges
    df['ib_high'], df['ib_low'], df['ib_range'] = calculate_initial_balance(df)
    df['monday_high'], df['monday_low'] = calculate_monday_range(df)
    df['asia_high'], df['asia_low'] = calculate_session_range(df, 'asia')
    df['london_high'], df['london_low'] = calculate_session_range(df, 'london')
    df['ny_high'], df['ny_low'] = calculate_session_range(df, 'ny')
    df['pdh'], df['pdl'], df['pdc'] = calculate_prior_day_levels(df)
    df['asr_20'] = average_session_range(df, lookback=20)

    # CVD and Delta
    if 'delta' in df.columns:
        df['cvd'] = cumulative_volume_delta(df)
        df['delta_change_1'] = delta_change(df, periods=1)
        df['delta_change_5'] = delta_change(df, periods=5)
        df['cvd_divergence'] = cvd_divergence(df)

    # OI
    if 'oi' in df.columns:
        df['oi_change'] = oi_change(df, periods=1)
        df['oi_change_pct'] = oi_change_pct(df, periods=1)

    # Wicks
    df['upper_wick_ratio'], df['lower_wick_ratio'], df['total_wick_ratio'] = calculate_wick_ratios(df)
    df['is_engulfing'] = is_engulfing(df)

    # Volume
    df['volume_vs_avg'] = volume_vs_average(df, window=20)
    df['volume_spike'] = volume_spike(df, threshold=2.0, window=20)

    # Technical indicators
    df['ema_50'] = calculate_ema(df, 'close', 50)
    df['ema_200'] = calculate_ema(df, 'close', 200)
    df['sma_20'] = calculate_sma(df, 'close', 20)
    df['atr_14'] = average_true_range(df, 14)
    df['z_score'] = calculate_z_score(df, 'close', 20)

    # Session and day tags
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
    df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)

    logger.info(f"Added indicators. DataFrame now has {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline

    pipeline = DataPipeline("config.yaml")
    df = pipeline.get_full_dataset(timeframe="5m", start_date="2024-01-01", end_date="2024-01-31")

    df_with_indicators = add_all_indicators(df)

    print(f"\nDataFrame shape: {df_with_indicators.shape}")
    print(f"\nColumns: {list(df_with_indicators.columns)}")
    print(f"\nSample data:")
    print(df_with_indicators.tail())
