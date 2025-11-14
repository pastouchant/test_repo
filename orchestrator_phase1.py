"""
Phase-1 Edge Discovery Orchestrator
Master coordinator for the entire edge discovery pipeline.

End-to-end workflow:
1. Load config
2. Fetch and prepare data
3. For each strategy in playbook:
   a. Backtest with default parameters
   b. If edge detected: optimize parameters
   c. If still good: train ML filter
   d. Validate out-of-sample
4. Prune edgeless strategies
5. Save approved strategies

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from data_pipeline import DataPipeline
from indicators import add_all_indicators
from features import build_feature_matrix, train_test_split_time_series
from playbook_specs import get_all_strategies, TradeSpec
from rules_engine import RulesEngine
from backtest import BacktestEngine
from metrics import PerformanceMetrics
from optimizer import StrategyOptimizer
from ml_models import TradeFilterModel

# Setup logging (create logs directory first)
Path('logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase1Orchestrator:
    """
    Orchestrates the complete Phase-1 edge discovery process.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize orchestrator.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path

        # Create output directories
        self._create_directories()

        # Results storage
        self.strategy_results = {}
        self.approved_strategies = []
        self.edgeless_strategies = []

        logger.info("=" * 80)
        logger.info("Phase-1 Edge Discovery Orchestrator Initialized")
        logger.info("=" * 80)

    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.config['output']['results_dir'],
            self.config['output']['trade_logs_dir'],
            self.config['output']['equity_curves_dir'],
            self.config['output']['reports_dir'],
            self.config['ml']['model_dir'],
            'logs'
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info("Output directories created")

    def run_full_pipeline(self):
        """
        Run complete Phase-1 edge discovery pipeline.
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING PHASE-1 EDGE DISCOVERY PIPELINE")
        logger.info("=" * 80 + "\n")

        # Step 1: Load and prepare data
        logger.info("STEP 1: Loading and preparing data...")
        df_train, df_test = self._load_and_prepare_data()

        # Step 2: Get all strategies from playbook
        strategies = get_all_strategies()
        logger.info(f"\nSTEP 2: Loaded {len(strategies)} strategies from playbook")

        # Step 3: Process each strategy
        for strategy_id, spec in strategies.items():
            logger.info("\n" + "-" * 80)
            logger.info(f"Processing Strategy: {spec.meta.name} ({strategy_id})")
            logger.info("-" * 80)

            result = self._process_strategy(spec, df_train, df_test)
            self.strategy_results[strategy_id] = result

        # Step 4: Prune and approve strategies
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Pruning and approving strategies...")
        logger.info("=" * 80)

        self._prune_and_approve_strategies()

        # Step 5: Generate reports
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Generating reports...")
        logger.info("=" * 80)

        self._generate_reports()

        # Step 6: Save results
        self._save_results()

        logger.info("\n" + "=" * 80)
        logger.info("PHASE-1 EDGE DISCOVERY COMPLETE")
        logger.info("=" * 80)

        self._print_final_summary()

    def _load_and_prepare_data(self) -> tuple:
        """
        Load and prepare data for backtesting.

        Returns:
            Tuple of (train_df, test_df)
        """
        # Initialize data pipeline
        pipeline = DataPipeline(self.config_path)

        # Get date range from config
        start_date = self.config['backtest']['start_date']
        end_date = self.config['backtest']['end_date']

        # Fetch data
        df = pipeline.get_full_dataset(
            timeframe=self.config['timeframes']['primary'],
            start_date=start_date,
            end_date=end_date,
            include_orderflow=True
        )

        logger.info(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        # Add all indicators
        df = add_all_indicators(df)

        # Build feature matrix
        df = build_feature_matrix(df, include_labels=False)

        # Train/test split
        train_ratio = self.config['backtest']['train_test_split']
        df_train, df_test = train_test_split_time_series(df, train_ratio=train_ratio)

        logger.info(f"Data split: {len(df_train)} train / {len(df_test)} test bars")

        return df_train, df_test

    def _process_strategy(
        self,
        spec: TradeSpec,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Dict:
        """
        Process a single strategy through the full pipeline.

        Args:
            spec: TradeSpec to process
            df_train: Training data
            df_test: Test data

        Returns:
            Dict with results
        """
        result = {
            'strategy_id': spec.meta.id,
            'strategy_name': spec.meta.name,
            'stage_reached': 'initial',
            'has_edge': False,
            'approved': False
        }

        # Stage 1: Initial backtest with default parameters
        logger.info("\n  Stage 1: Initial backtest with default parameters")
        initial_metrics = self._run_backtest(spec, df_train, params=None)

        result['initial_metrics'] = initial_metrics
        result['stage_reached'] = 'initial_backtest'

        # Check if has initial edge
        pf = initial_metrics.get('profit_factor', 0)
        trades = initial_metrics.get('total_trades', 0)

        logger.info(f"    Initial PF: {pf:.2f}, Trades: {trades}")

        if not self._has_edge(initial_metrics):
            logger.warning(f"  ✗ No initial edge detected. Skipping further optimization.")
            result['reason'] = "No initial edge"
            return result

        logger.info(f"  ✓ Initial edge detected!")
        result['has_edge'] = True

        # Stage 2: Parameter optimization
        logger.info("\n  Stage 2: Parameter optimization")
        optimized_params, optimized_metrics = self._optimize_parameters(spec, df_train)

        result['optimized_params'] = optimized_params
        result['optimized_metrics'] = optimized_metrics
        result['stage_reached'] = 'optimized'

        logger.info(f"    Optimized PF: {optimized_metrics['pf']:.2f}")

        # Check if still has edge after optimization
        if not self._has_edge(optimized_metrics, is_optimized=True):
            logger.warning(f"  ✗ Lost edge after optimization. Discarding.")
            result['reason'] = "Lost edge after optimization"
            result['has_edge'] = False
            return result

        # Stage 3: ML filtering (optional but recommended)
        logger.info("\n  Stage 3: ML signal filtering")
        ml_metrics = self._train_ml_filter(spec, df_train, optimized_params)

        if ml_metrics:
            result['ml_metrics'] = ml_metrics
            result['stage_reached'] = 'ml_filtered'
            logger.info(f"    ML-filtered PF: {ml_metrics.get('profit_factor', 0):.2f}")

        # Stage 4: Out-of-sample validation
        logger.info("\n  Stage 4: Out-of-sample validation")
        test_metrics = self._validate_out_of_sample(spec, df_test, optimized_params)

        result['test_metrics'] = test_metrics
        result['stage_reached'] = 'validated'

        logger.info(f"    Test PF: {test_metrics.get('profit_factor', 0):.2f}")

        # Check if passes validation
        if self._passes_validation(test_metrics):
            logger.info(f"  ✓ Strategy APPROVED!")
            result['approved'] = True
        else:
            logger.warning(f"  ✗ Failed out-of-sample validation")
            result['reason'] = "Failed out-of-sample validation"

        return result

    def _run_backtest(
        self,
        spec: TradeSpec,
        df: pd.DataFrame,
        params: Dict = None
    ) -> Dict:
        """
        Run backtest for a strategy.

        Args:
            spec: TradeSpec
            df: Data
            params: Optional parameter overrides

        Returns:
            Dict with metrics
        """
        # Generate signals
        engine = RulesEngine(spec, parameters=params)
        df_signals = engine.generate_signals(df)

        # Run backtest
        bt = BacktestEngine(self.config_path)
        trades, equity_curve = bt.run(
            df_signals,
            max_bars_in_trade=spec.exit_logic.max_bars_in_trade
        )

        # Calculate metrics
        if len(trades) == 0:
            return {'total_trades': 0, 'profit_factor': 0}

        trade_df = pd.DataFrame([t.to_dict() for t in trades])
        metrics_calc = PerformanceMetrics(trade_df, equity_curve)
        metrics = metrics_calc.calculate_all_metrics()

        return metrics

    def _optimize_parameters(
        self,
        spec: TradeSpec,
        df: pd.DataFrame
    ) -> tuple:
        """
        Optimize strategy parameters.

        Returns:
            Tuple of (best_params, metrics)
        """
        optimizer = StrategyOptimizer(spec, df, self.config_path)

        method = self.config['optimization']['default_method']
        max_iter = self.config['optimization']['max_iterations']

        if method == "bayesian":
            result = optimizer.optimize(method="bayesian", n_calls=max_iter)
        elif method == "random_search":
            result = optimizer.optimize(method="random", n_iter=max_iter)
        else:
            result = optimizer.optimize(method="grid")

        return result['params'], result

    def _train_ml_filter(
        self,
        spec: TradeSpec,
        df: pd.DataFrame,
        params: Dict
    ) -> Dict:
        """
        Train ML model to filter signals.

        Returns:
            Dict with ML-filtered metrics
        """
        try:
            # Build features with labels
            df_features = build_feature_matrix(df, include_labels=True)

            # Train model
            model = TradeFilterModel(model_type="logistic", config_path=self.config_path)
            train_metrics = model.train(df_features, cv_folds=3)

            # Save model
            model_path = Path(self.config['ml']['model_dir']) / f"{spec.meta.id}_model.pkl"
            model.save(str(model_path))

            return train_metrics

        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return {}

    def _validate_out_of_sample(
        self,
        spec: TradeSpec,
        df_test: pd.DataFrame,
        params: Dict
    ) -> Dict:
        """
        Validate strategy on out-of-sample test data.

        Returns:
            Dict with test metrics
        """
        return self._run_backtest(spec, df_test, params)

    def _has_edge(self, metrics: Dict, is_optimized: bool = False) -> bool:
        """
        Check if metrics indicate presence of edge.

        Args:
            metrics: Metrics dict
            is_optimized: Whether these are optimized metrics

        Returns:
            True if has edge
        """
        prune_criteria = self.config['edge_discovery']['prune_if']

        pf = metrics.get('profit_factor', 0)
        trades = metrics.get('total_trades', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0) / 100  # Convert to decimal

        # Check pruning criteria
        if pf < prune_criteria['profit_factor_below']:
            return False

        if sharpe < prune_criteria['sharpe_below']:
            return False

        if trades < prune_criteria['trades_below']:
            return False

        if win_rate < prune_criteria['win_rate_below']:
            return False

        return True

    def _passes_validation(self, test_metrics: Dict) -> bool:
        """
        Check if strategy passes validation criteria.

        Args:
            test_metrics: Out-of-sample test metrics

        Returns:
            True if passes validation
        """
        approve_criteria = self.config['edge_discovery']['approve_if']

        pf = test_metrics.get('profit_factor', 0)
        sharpe = test_metrics.get('sharpe_ratio', 0)
        trades = test_metrics.get('total_trades', 0)
        mdd = abs(test_metrics.get('max_drawdown_pct', 100))

        # Check approval criteria
        if pf < approve_criteria['profit_factor_above']:
            return False

        if sharpe < approve_criteria['sharpe_above']:
            return False

        if trades < approve_criteria['trades_above']:
            return False

        if mdd > approve_criteria['max_drawdown_below']:
            return False

        return True

    def _prune_and_approve_strategies(self):
        """Classify strategies as approved or edgeless."""
        for strategy_id, result in self.strategy_results.items():
            if result.get('approved', False):
                self.approved_strategies.append(result)
            else:
                self.edgeless_strategies.append(result)

        logger.info(f"\n  Approved: {len(self.approved_strategies)} strategies")
        logger.info(f"  Edgeless: {len(self.edgeless_strategies)} strategies")

    def _generate_reports(self):
        """Generate performance reports."""
        # Generate summary report
        report_lines = [
            "=" * 80,
            "PHASE-1 EDGE DISCOVERY SUMMARY REPORT",
            "=" * 80,
            "",
            f"Total Strategies Evaluated: {len(self.strategy_results)}",
            f"Approved Strategies: {len(self.approved_strategies)}",
            f"Edgeless Strategies: {len(self.edgeless_strategies)}",
            "",
            "=" * 80,
            "APPROVED STRATEGIES",
            "=" * 80,
        ]

        for strat in self.approved_strategies:
            report_lines.extend([
                "",
                f"Strategy: {strat['strategy_name']} ({strat['strategy_id']})",
                f"  Train PF: {strat['optimized_metrics']['pf']:.2f}",
                f"  Test PF: {strat['test_metrics'].get('profit_factor', 0):.2f}",
                f"  Sharpe: {strat['test_metrics'].get('sharpe_ratio', 0):.2f}",
                f"  Trades/Week: {strat['test_metrics'].get('trades_per_week', 0):.1f}",
            ])

        report_text = "\n".join(report_lines)

        # Save to file
        report_path = Path(self.config['output']['reports_dir']) / "phase1_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        logger.info(f"\n  Report saved to: {report_path}")
        print("\n" + report_text)

    def _save_results(self):
        """Save results to JSON files."""
        # Save approved strategies
        approved_path = self.config['output']['approved_strategies_file']
        with open(approved_path, 'w') as f:
            json.dump(self.approved_strategies, f, indent=2, default=str)

        logger.info(f"\n  Approved strategies saved to: {approved_path}")

        # Save edgeless strategies
        edgeless_path = self.config['output']['edgeless_strategies_file']
        with open(edgeless_path, 'w') as f:
            json.dump(self.edgeless_strategies, f, indent=2, default=str)

        logger.info(f"  Edgeless strategies saved to: {edgeless_path}")

    def _print_final_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"\n✓ Strategies Approved: {len(self.approved_strategies)}")
        print(f"✗ Strategies Pruned: {len(self.edgeless_strategies)}")

        if self.approved_strategies:
            print("\n🎯 APPROVED STRATEGIES:")
            for strat in self.approved_strategies:
                test_pf = strat['test_metrics'].get('profit_factor', 0)
                print(f"  • {strat['strategy_name']}: PF={test_pf:.2f}")

        print("\n" + "=" * 80)
        print("Next Steps:")
        print("  1. Review approved_strategies.json")
        print("  2. Analyze performance reports in ./results/reports/")
        print("  3. Proceed to Phase-2 (live bot) with approved strategies")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    orchestrator = Phase1Orchestrator("config.yaml")
    orchestrator.run_full_pipeline()


if __name__ == "__main__":
    main()
