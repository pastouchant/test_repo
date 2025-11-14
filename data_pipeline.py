"""
Data Pipeline for BTCUSDT Perpetual
Fetches, caches, and preprocesses market data from multiple sources.

Data Sources:
1. CCXT/Bybit API: Free OHLCV data
2. Tardis/Databento (optional): Orderflow data (CVD, OI, funding)

Author: Phase-1 Edge Discovery System
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Handles all data fetching, caching, and preprocessing for BTCUSDT perpetual.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize data pipeline with configuration.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.symbol = self.config['market']['symbol']
        self.venue = self.config['market']['venue']

        # Setup cache directory
        self.cache_dir = Path(self.config['data']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange connection
        self.exchange = self._init_exchange()

        logger.info(f"DataPipeline initialized for {self.symbol} on {self.venue}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _init_exchange(self) -> ccxt.Exchange:
        """
        Initialize CCXT exchange connection.

        Returns:
            Exchange object
        """
        exchange_class = getattr(ccxt, self.venue)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # For perpetual futures
            }
        })
        logger.info(f"Initialized {self.venue} exchange connection")
        return exchange

    def fetch_ohlcv(
        self,
        timeframe: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for specified timeframe and date range.

        Args:
            timeframe: Candlestick timeframe (e.g., '1m', '5m', '15m')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{self.symbol}_{timeframe}_{start_date}_{end_date}.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            cache_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if cache_age_days < self.config['data'].get('cache_expiry_days', 7):
                logger.info(f"Loading OHLCV from cache: {cache_file}")
                return pd.read_parquet(cache_file)

        # Fetch fresh data
        logger.info(f"Fetching OHLCV for {self.symbol} {timeframe} from {start_date} to {end_date}")

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_candles = []
        current_ts = start_ts

        # Fetch in batches (CCXT limit is typically 1000 candles per request)
        while current_ts < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=f"{self.symbol}",
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)
                current_ts = candles[-1][0] + 1  # Move to next timestamp

                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)

                logger.debug(f"Fetched {len(candles)} candles, total: {len(all_candles)}")

            except Exception as e:
                logger.error(f"Error fetching OHLCV: {e}")
                # Implement exponential backoff retry
                for retry in range(3):
                    wait_time = 2 ** retry
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    try:
                        candles = self.exchange.fetch_ohlcv(
                            symbol=f"{self.symbol}",
                            timeframe=timeframe,
                            since=current_ts,
                            limit=1000
                        )
                        all_candles.extend(candles)
                        current_ts = candles[-1][0] + 1
                        break
                    except Exception as retry_error:
                        logger.error(f"Retry {retry + 1} failed: {retry_error}")
                        if retry == 2:
                            raise

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Filter to exact date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        logger.info(f"Fetched {len(df)} {timeframe} candles from {df.index[0]} to {df.index[-1]}")

        # Save to cache
        if use_cache:
            df.to_parquet(cache_file)
            logger.info(f"Saved to cache: {cache_file}")

        return df

    def fetch_orderflow_data(
        self,
        start_date: str,
        end_date: str,
        source: str = "tardis"
    ) -> pd.DataFrame:
        """
        Fetch orderflow data (CVD, delta, OI, funding) from paid sources.

        Args:
            start_date: Start date
            end_date: End date
            source: Data source ('tardis', 'databento', or 'none')

        Returns:
            DataFrame with orderflow data
        """
        if source == "none":
            logger.warning("Orderflow source set to 'none', generating synthetic placeholders")
            return self._generate_synthetic_orderflow(start_date, end_date)

        # TODO: Implement actual Tardis/Databento API calls
        # For now, generate synthetic data as placeholder
        logger.warning(f"Orderflow source '{source}' not yet implemented, using synthetic data")
        return self._generate_synthetic_orderflow(start_date, end_date)

    def _generate_synthetic_orderflow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate synthetic orderflow data for testing purposes.

        In production, replace this with actual orderflow data from Tardis/Databento.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with synthetic orderflow metrics
        """
        logger.warning("Generating SYNTHETIC orderflow data - replace with real data in production")

        # Fetch 1m OHLCV as base
        df = self.fetch_ohlcv('1m', start_date, end_date)

        # Generate synthetic CVD and delta
        # In reality, these would come from trade-by-trade data
        df['delta'] = np.random.normal(0, df['volume'] * 0.1)  # Randomdelta
        df['cvd'] = df['delta'].cumsum()  # Cumulative volume delta

        # Synthetic OI (tends to increase in trends)
        df['oi'] = 10000 + np.cumsum(np.random.normal(0, 100, len(df)))
        df['oi'] = df['oi'].abs()  # OI must be positive

        # Synthetic funding rate (mean-reverting around 0.01%)
        df['funding_rate'] = 0.0001 + np.random.normal(0, 0.00005, len(df))

        return df[['cvd', 'delta', 'oi', 'funding_rate']]

    def resample_timeframe(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe.

        Args:
            df: Input DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., '5m', '15m', '1h')

        Returns:
            Resampled DataFrame
        """
        # OHLCV resampling rules (only include if columns exist)
        base_resample_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Only use rules for columns that actually exist in the DataFrame
        resample_rules = {col: rule for col, rule in base_resample_rules.items() if col in df.columns}

        # Additional fields (sum for delta/volume, last for others)
        additional_rules = {}
        for col in df.columns:
            if col not in resample_rules:
                if col in ['delta', 'volume']:
                    additional_rules[col] = 'sum'
                else:
                    additional_rules[col] = 'last'

        all_rules = {**resample_rules, **additional_rules}

        # Resample
        df_resampled = df.resample(target_timeframe).agg(all_rules).dropna()

        logger.info(f"Resampled {len(df)} bars to {len(df_resampled)} bars at {target_timeframe}")

        return df_resampled

    def merge_data_sources(
        self,
        ohlcv: pd.DataFrame,
        orderflow: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge OHLCV and orderflow data into single DataFrame.

        Args:
            ohlcv: OHLCV DataFrame
            orderflow: Orderflow DataFrame (CVD, delta, OI, funding)

        Returns:
            Merged DataFrame
        """
        # Align indices (forward-fill orderflow data to match OHLCV timestamps)
        merged = ohlcv.join(orderflow, how='left')
        merged.fillna(method='ffill', inplace=True)

        logger.info(f"Merged data: {len(merged)} bars with {len(merged.columns)} columns")

        return merged

    def get_full_dataset(
        self,
        timeframe: str = "5m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_orderflow: bool = True
    ) -> pd.DataFrame:
        """
        Main method to get complete dataset for backtesting.

        Args:
            timeframe: Desired timeframe
            start_date: Start date (defaults to config)
            end_date: End date (defaults to config)
            include_orderflow: Whether to include orderflow data

        Returns:
            Complete DataFrame ready for indicator calculation
        """
        # Use config dates if not specified
        if start_date is None:
            start_date = self.config['backtest']['start_date']
        if end_date is None:
            end_date = self.config['backtest']['end_date']

        logger.info(f"Building full dataset for {timeframe} from {start_date} to {end_date}")

        # Fetch OHLCV
        ohlcv = self.fetch_ohlcv(timeframe, start_date, end_date)

        if include_orderflow:
            # Fetch orderflow data
            orderflow_source = self.config['data'].get('orderflow_source', 'none')
            orderflow = self.fetch_orderflow_data(start_date, end_date, source=orderflow_source)

            # Resample orderflow to match target timeframe if needed
            if timeframe != '1m':
                orderflow = self.resample_timeframe(orderflow, timeframe)

            # Merge
            df = self.merge_data_sources(ohlcv, orderflow)
        else:
            df = ohlcv

        # Add basic derived fields
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        logger.info(f"Dataset ready: {len(df)} bars, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.

        Args:
            df: DataFrame to validate

        Returns:
            True if data passes validation, False otherwise
        """
        required_fields = self.config['data']['required_fields']

        # Check for required columns
        missing_cols = [col for col in required_fields if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        # Check for NaN values
        nan_counts = df[required_fields].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
            # Allow small amount of NaNs at the beginning (indicator warmup)
            if nan_counts.max() > len(df) * 0.05:  # More than 5% NaNs
                logger.error("Too many NaN values in data")
                return False

        # Check for sufficient data
        min_bars = self.config['backtest'].get('min_bars_required', 1000)
        if len(df) < min_bars:
            logger.error(f"Insufficient data: {len(df)} bars (minimum: {min_bars})")
            return False

        logger.info("Data validation passed")
        return True


def main():
    """
    Example usage of DataPipeline.
    """
    # Initialize pipeline
    pipeline = DataPipeline("config.yaml")

    # Fetch complete dataset
    df = pipeline.get_full_dataset(
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-03-31",
        include_orderflow=True
    )

    # Validate
    is_valid = pipeline.validate_data(df)

    if is_valid:
        print(f"\n✓ Data fetched successfully: {len(df)} bars")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nData info:")
        print(df.info())
    else:
        print("\n✗ Data validation failed")


if __name__ == "__main__":
    main()
