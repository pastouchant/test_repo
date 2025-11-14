"""
Parameter Optimization for Trading Strategies
Implements grid search, random search, and Bayesian optimization.

Optimizes parameters to maximize Profit Factor subject to constraints
(min trades/week, max drawdown, min Sharpe).

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from itertools import product
import yaml
import logging
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from playbook_specs import TradeSpec
from rules_engine import RulesEngine
from backtest import BacktestEngine
from metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Optimizes strategy parameters using various search methods.
    """

    def __init__(
        self,
        spec: TradeSpec,
        df: pd.DataFrame,
        config_path: str = "config.yaml"
    ):
        """
        Initialize optimizer.

        Args:
            spec: TradeSpec to optimize
            df: DataFrame with features and indicators
            config_path: Path to configuration file
        """
        self.spec = spec
        self.df = df

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.opt_config = self.config['optimization']

        # Constraints
        self.min_pf = self.opt_config['min_profit_factor']
        self.min_trades_week = self.opt_config['min_trades_per_week']
        self.max_trades_week = self.opt_config['max_trades_per_week']
        self.min_sharpe = self.opt_config['min_sharpe']
        self.max_dd = self.opt_config['max_drawdown_pct']

        self.results = []

        logger.info(f"StrategyOptimizer initialized for {spec.meta.id}")

    def _objective_function(self, params: Dict) -> float:
        """
        Objective function to minimize (negative Profit Factor).

        Args:
            params: Parameter dictionary

        Returns:
            Negative PF (for minimization) or large penalty if constraints violated
        """
        try:
            # Generate signals with these parameters
            engine = RulesEngine(self.spec, parameters=params)
            df_signals = engine.generate_signals(self.df)

            # Run backtest
            bt = BacktestEngine(config_path="config.yaml")
            trades, equity_curve = bt.run(
                df_signals,
                max_bars_in_trade=self.spec.exit_logic.max_bars_in_trade
            )

            # Calculate metrics
            if len(trades) == 0:
                return 1000.0  # Large penalty for no trades

            trade_df = pd.DataFrame([t.to_dict() for t in trades])
            metrics_calc = PerformanceMetrics(trade_df, equity_curve)
            metrics = metrics_calc.calculate_all_metrics()

            # Check constraints
            pf = metrics['profit_factor']
            trades_week = metrics['trades_per_week']
            sharpe = metrics['sharpe_ratio']
            dd = abs(metrics['max_drawdown_pct'])

            # Penalty for constraint violations
            penalty = 0

            if trades_week < self.min_trades_week:
                penalty += 100 * (self.min_trades_week - trades_week)

            if trades_week > self.max_trades_week:
                penalty += 50 * (trades_week - self.max_trades_week)

            if sharpe < self.min_sharpe:
                penalty += 50 * (self.min_sharpe - sharpe)

            if dd > self.max_dd:
                penalty += 100 * (dd - self.max_dd)

            # Objective: maximize PF (minimize negative PF)
            objective = -pf + penalty

            # Store result
            result = {
                'params': params.copy(),
                'pf': pf,
                'sharpe': sharpe,
                'trades_week': trades_week,
                'max_dd': dd,
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'objective': objective
            }
            self.results.append(result)

            logger.debug(f"Evaluated params: PF={pf:.2f}, Sharpe={sharpe:.2f}, Trades/week={trades_week:.1f}")

            return objective

        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1000.0  # Large penalty on error

    def grid_search(self) -> Dict:
        """
        Grid search over parameter space.

        Returns:
            Best parameters found
        """
        logger.info("Starting grid search...")

        # Build parameter grid
        param_grid = {}
        for param in self.spec.parameters:
            if param.step is not None:
                param_grid[param.name] = np.arange(
                    param.min_value,
                    param.max_value + param.step,
                    param.step
                )
            else:
                # Default: 5 values across range
                param_grid[param.name] = np.linspace(
                    param.min_value,
                    param.max_value,
                    5
                )

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))

        logger.info(f"Grid search: {len(combinations)} combinations to evaluate")

        # Evaluate each combination
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            self._objective_function(params)

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(combinations)} combinations evaluated")

        # Find best
        best_result = min(self.results, key=lambda x: x['objective'])

        logger.info(f"Grid search complete. Best PF: {best_result['pf']:.2f}")

        return best_result

    def random_search(self, n_iter: int = 100) -> Dict:
        """
        Random search over parameter space.

        Args:
            n_iter: Number of random samples to evaluate

        Returns:
            Best parameters found
        """
        logger.info(f"Starting random search with {n_iter} iterations...")

        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for param in self.spec.parameters:
                params[param.name] = np.random.uniform(param.min_value, param.max_value)

            self._objective_function(params)

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{n_iter} iterations")

        # Find best
        best_result = min(self.results, key=lambda x: x['objective'])

        logger.info(f"Random search complete. Best PF: {best_result['pf']:.2f}")

        return best_result

    def bayesian_optimization(self, n_calls: int = 50) -> Dict:
        """
        Bayesian optimization using Gaussian Processes.

        Args:
            n_calls: Number of function evaluations

        Returns:
            Best parameters found
        """
        logger.info(f"Starting Bayesian optimization with {n_calls} calls...")

        # Define search space
        space = []
        param_names = []

        for param in self.spec.parameters:
            space.append(Real(param.min_value, param.max_value, name=param.name))
            param_names.append(param.name)

        # Objective function for skopt
        @use_named_args(space)
        def objective(**params):
            return self._objective_function(params)

        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=False
        )

        # Best parameters
        best_params = dict(zip(param_names, result.x))

        # Find corresponding result
        best_result = None
        for res in self.results:
            if all(abs(res['params'][k] - v) < 1e-6 for k, v in best_params.items()):
                best_result = res
                break

        if best_result is None:
            # Fallback: find minimum objective
            best_result = min(self.results, key=lambda x: x['objective'])

        logger.info(f"Bayesian optimization complete. Best PF: {best_result['pf']:.2f}")

        return best_result

    def optimize(self, method: str = "bayesian", **kwargs) -> Dict:
        """
        Run optimization using specified method.

        Args:
            method: 'grid', 'random', or 'bayesian'
            **kwargs: Additional arguments for specific methods

        Returns:
            Best parameters and metrics
        """
        self.results = []  # Reset results

        if method == "grid":
            return self.grid_search()
        elif method == "random":
            n_iter = kwargs.get('n_iter', 100)
            return self.random_search(n_iter)
        elif method == "bayesian":
            n_calls = kwargs.get('n_calls', 50)
            return self.bayesian_optimization(n_calls)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.

        Returns:
            DataFrame with all evaluated parameter sets and metrics
        """
        return pd.DataFrame(self.results)


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline
    from indicators import add_all_indicators
    from playbook_specs import get_small_ib_momentum_spec

    # Load data
    pipeline = DataPipeline("config.yaml")
    df = pipeline.get_full_dataset(timeframe="5m", start_date="2024-01-01", end_date="2024-02-29")

    # Add indicators
    df = add_all_indicators(df)

    # Get strategy spec
    spec = get_small_ib_momentum_spec()

    # Optimize
    optimizer = StrategyOptimizer(spec, df)
    best = optimizer.optimize(method="random", n_iter=20)  # Quick test with 20 iterations

    print("\n=== Optimization Results ===")
    print(f"Best Profit Factor: {best['pf']:.2f}")
    print(f"Sharpe Ratio: {best['sharpe']:.2f}")
    print(f"Trades/Week: {best['trades_week']:.1f}")
    print(f"Max Drawdown: {best['max_dd']:.2f}%")
    print(f"\nBest Parameters:")
    for param, value in best['params'].items():
        print(f"  {param}: {value:.3f}")
