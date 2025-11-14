"""
Machine Learning Models for Trade Signal Filtering
Trains classifiers to predict "good trades" vs "bad trades".

Uses time-series cross-validation to prevent lookahead bias.
Models: Logistic Regression, XGBoost, LightGBM.

Author: Phase-1 Edge Discovery System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
import logging
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

from features import get_feature_columns

logger = logging.getLogger(__name__)


class TradeFilterModel:
    """
    ML model to filter rule-based trade signals.

    Predicts probability of a "good trade" (TP hit before SL).
    """

    def __init__(self, model_type: str = "logistic", config_path: str = "config.yaml"):
        """
        Initialize ML model.

        Args:
            model_type: 'logistic', 'xgboost', 'lightgbm', or 'random_forest'
            config_path: Path to configuration file
        """
        self.model_type = model_type

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.ml_config = self.config['ml']

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None

        logger.info(f"TradeFilterModel initialized: {model_type}")

    def _init_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=self.ml_config['random_state'],
                max_iter=1000
            )

        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.ml_config['random_state'],
                n_jobs=-1
            )

        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed")

            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.ml_config['random_state'],
                n_jobs=-1
            )

        elif self.model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed")

            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.ml_config['random_state'],
                n_jobs=-1,
                verbose=-1
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        cv_folds: int = 5
    ) -> Dict:
        """
        Train model using time-series cross-validation.

        Args:
            df: DataFrame with features and labels
            label_col: Name of label column
            cv_folds: Number of CV folds

        Returns:
            Dict with training metrics
        """
        logger.info(f"Training {self.model_type} model with {cv_folds}-fold CV...")

        # Get feature columns
        self.feature_columns = get_feature_columns(df, exclude_targets=True)

        # Filter to valid labels (exclude -1 = unclear)
        df_valid = df[df[label_col] != -1].copy()

        logger.info(f"Training samples: {len(df_valid)} (filtered from {len(df)} total)")

        # Prepare features and labels
        X = df_valid[self.feature_columns].fillna(0)  # Fill NaN with 0
        y = df_valid[label_col]

        # Initialize model
        self._init_model()

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        cv_scores = []
        cv_auc = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_val_scaled)
            y_proba = self.model.predict_proba(X_val_scaled)[:, 1]

            score = self.model.score(X_val_scaled, y_val)
            auc = roc_auc_score(y_val, y_proba)

            cv_scores.append(score)
            cv_auc.append(auc)

            logger.info(f"Fold {fold + 1}/{cv_folds}: Accuracy={score:.3f}, AUC={auc:.3f}")

        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        logger.info(f"Training complete. Avg CV Accuracy: {np.mean(cv_scores):.3f}, AUC: {np.mean(cv_auc):.3f}")

        metrics = {
            'cv_accuracy': np.mean(cv_scores),
            'cv_auc': np.mean(cv_auc),
            'cv_std': np.std(cv_scores),
            'n_features': len(self.feature_columns),
            'n_samples': len(df_valid)
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict labels for new data.

        Args:
            df: DataFrame with features

        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for new data.

        Args:
            df: DataFrame with features

        Returns:
            Array of probabilities (for class 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return probabilities

    def optimize_threshold(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        metric: str = 'f1'
    ) -> float:
        """
        Find optimal probability threshold using precision-recall curve.

        Args:
            df: DataFrame with features and labels
            label_col: Label column name
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Optimal threshold
        """
        df_valid = df[df[label_col] != -1].copy()
        X = df_valid[self.feature_columns].fillna(0)
        y = df_valid[label_col]

        X_scaled = self.scaler.transform(X)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_proba)

        if metric == 'f1':
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            optimal_idx = np.argmax(precision)
        elif metric == 'recall':
            optimal_idx = np.argmax(recall)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        logger.info(f"Optimal threshold ({metric}): {optimal_threshold:.3f}")

        return optimal_threshold

    def filter_signals(
        self,
        df_with_signals: pd.DataFrame,
        threshold: float = 0.5,
        signal_col: str = 'signal'
    ) -> pd.DataFrame:
        """
        Filter rule-based signals using ML predictions.

        Args:
            df_with_signals: DataFrame with rule-based signals
            threshold: Probability threshold for filtering
            signal_col: Column with signals

        Returns:
            DataFrame with filtered signals
        """
        df = df_with_signals.copy()

        # Get predictions
        probabilities = self.predict_proba(df)

        # Filter: keep signal only if ML model says it's a good trade
        df[f'{signal_col}_filtered'] = df[signal_col].where(
            probabilities >= threshold,
            0  # Set to 0 (no signal) if below threshold
        )

        # Count filtered signals
        original_signals = (df[signal_col] != 0).sum()
        filtered_signals = (df[f'{signal_col}_filtered'] != 0).sum()

        logger.info(f"Filtered {original_signals} signals -> {filtered_signals} "
                   f"({filtered_signals / max(original_signals, 1) * 100:.1f}% kept)")

        return df

    def save(self, path: str):
        """Save model to disk."""
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }, path)

        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)

        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.feature_importance = data.get('feature_importance')
        self.model_type = data['model_type']

        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    from data_pipeline import DataPipeline
    from indicators import add_all_indicators
    from features import build_feature_matrix

    # Load data
    pipeline = DataPipeline("config.yaml")
    df = pipeline.get_full_dataset(timeframe="5m", start_date="2024-01-01", end_date="2024-02-29")

    # Add indicators
    df = add_all_indicators(df)

    # Build features with labels
    df_features = build_feature_matrix(df, include_labels=True)

    # Train model
    model = TradeFilterModel(model_type="logistic")
    metrics = model.train(df_features, cv_folds=3)

    print("\n=== ML Model Training Results ===")
    print(f"CV Accuracy: {metrics['cv_accuracy']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"CV AUC: {metrics['cv_auc']:.3f}")
    print(f"Features: {metrics['n_features']}")

    if model.feature_importance is not None:
        print("\nTop 10 Features:")
        print(model.feature_importance.head(10))

    # Optimize threshold
    optimal_threshold = model.optimize_threshold(df_features, metric='f1')
    print(f"\nOptimal F1 threshold: {optimal_threshold:.3f}")
