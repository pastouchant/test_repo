# Professional ML Trading Research System

Advanced machine learning trading system targeting high annual returns using neural networks, ensemble learning, and sophisticated feature engineering.

## Overview

This system implements a professional-grade ML trading framework with:

- **Advanced ML Models**: Neural Networks, Random Forest, Gradient Boosting
- **Comprehensive Feature Engineering**: 100+ technical indicators and market features
- **Multiple Trading Strategies**: 5 different ML-based strategies
- **Risk Management**: Dynamic position sizing, confidence-based allocation
- **Realistic Backtesting**: Simulated market data with regime changes and patterns

## Features

### Machine Learning Models

1. **Neural Network (MLP)**: Deep learning with 3 hidden layers
2. **Random Forest**: Ensemble of 200 decision trees
3. **Gradient Boosting**: Advanced boosting with feature importance
4. **Voting Ensemble**: Combines best performing models

### Feature Engineering (104 features)

- Basic price features (returns, log returns, price changes)
- Moving averages (SMA, EMA) across multiple timeframes
- Bollinger Bands and volatility indicators
- RSI and stochastic oscillators
- Volume-based features (OBV, volume ratios)
- Market microstructure indicators
- Pattern recognition (doji, hammer, engulfing patterns)
- Market regime detection
- Time-based session features

### Trading Strategies

1. **ML Momentum Fusion**: 30% allocation, targets 4.5% returns
2. **Neural Pattern Hunter**: 25% allocation, targets 5.5% returns
3. **Ensemble Signal Fusion**: 20% allocation, targets 3.5% returns
4. **Volatility Regime Trader**: 15% allocation, targets 2.8% returns
5. **High Frequency Scalper**: 10% allocation, targets 1.5% returns

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- requests >= 2.31.0

## Usage

Run the full ML trading system:

```bash
python3 ml_trading_system.py
```

The system will:
1. Generate realistic market data (365 days of 5-minute bars)
2. Create comprehensive ML features
3. Train multiple ML models with cross-validation
4. Run trading simulation
5. Generate performance report
6. Save results to JSON file

## Output

The system provides detailed performance metrics:

- Annual return percentage
- Win rate and profit factor
- Sharpe ratio
- Model performance breakdown
- Strategy-level statistics
- Feature importance analysis
- High-confidence trade analysis

Results are saved to `professional_ml_results_[timestamp].json`

## System Architecture

### Data Generation
- Realistic market simulation with multiple regime types
- Session-based patterns (Asia, London, NY)
- News event simulation
- Momentum and mean reversion effects

### Model Training
- Time series cross-validation (3 folds)
- Feature selection (top 100 features)
- Robust scaling for normalization
- Early stopping for neural networks

### Risk Management
- Maximum position size: 5%
- Maximum daily risk: 8%
- Maximum drawdown: 15%
- Confidence-based position scaling
- Dynamic stop loss (0.6x target)

## Performance Targets

- **Target Annual Return**: 100%+
- **Expected Win Rate**: 55-75%
- **Target Profit Factor**: 1.5-2.5
- **Target Sharpe Ratio**: 1.5-3.0

## Disclaimer

This is a research system for educational and backtesting purposes. The performance metrics are based on simulated data and do not represent real market conditions. Past performance does not guarantee future results.

**Important Notes:**
- Do not use this system for real trading without thorough validation
- Real market conditions differ significantly from simulations
- Always conduct proper risk assessment before live trading
- Consult with financial professionals before making trading decisions

## Technical Details

### Model Hyperparameters

**Neural Network:**
- Hidden layers: (100, 50, 25)
- Activation: ReLU
- Solver: Adam
- Alpha: 0.001
- Max iterations: 500
- Early stopping enabled

**Random Forest:**
- Estimators: 200
- Max depth: 15
- Min samples split: 10
- Min samples leaf: 5
- Max features: sqrt

**Gradient Boosting:**
- Estimators: 100
- Learning rate: 0.1
- Max depth: 6
- Subsample: 0.8

### Target Labels

Multi-class classification (5 classes):
- 0: Strong sell signal
- 1: Sell signal
- 2: Hold/neutral
- 3: Buy signal
- 4: Strong buy signal

### Success Probability Calculation

Base success rate enhanced by:
- Model confidence (up to 20% boost)
- Model agreement (up to 10% boost)
- Maximum probability: 90%

## License

Educational and research use only.
