# BTCUSDT Perpetual Edge Discovery System - Phase 1

A comprehensive, code-first research laboratory for discovering, validating, and optimizing profitable trading strategies on BTCUSDT perpetual futures (Bybit).

## 🎯 Project Overview

This is **Phase 1** of a two-phase project focused on **rigorous edge discovery** through:

1. **Strict Specification**: Convert narrative trading concepts into machine-testable rules
2. **Long-Window Backtesting**: Test strategies against historical data with realistic costs
3. **Edge Measurement**: Use Profit Factor as primary metric with robust statistical validation
4. **Pruning**: Automatically discard edgeless strategies
5. **ML Optimization**: Train classifiers to filter and enhance rule-based signals

**Phase 2** (future) will build a self-optimizing trading bot using Phase-1 outputs.

## 📚 Strategy Source

All strategies are derived from the **Flow Playbook Intraday** document, which defines 12 core setups based on:
- Market structure (IB, ranges, sessions)
- Orderflow concepts (CVD, delta, OI, funding)
- Trap and reversal mechanics
- Session-specific behaviors (Asia, London, NY)

## 🏗️ System Architecture

```
config.yaml                    # System configuration
data_pipeline.py              # Data fetching, caching, preprocessing
indicators.py                  # Technical indicators & orderflow metrics
features.py                    # ML feature engineering & labels
playbook_specs.py             # TradeSpec templates & 5 example strategies
rules_engine.py               # Signal generation from TradeSpecs
backtest.py                   # Event-driven backtest with costs
metrics.py                    # Performance metrics (PF, Sharpe, MDD, etc.)
optimizer.py                  # Parameter optimization (Grid/Random/Bayesian)
ml_models.py                  # ML signal filtering & training
orchestrator_phase1.py        # Master pipeline coordinator
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data/cache results/trade_logs results/equity_curves results/reports models
```

### 2. Configuration

Edit `config.yaml` to customize:
- Data sources and date ranges
- Fee structure (Bybit Europe defaults)
- Risk parameters
- Optimization settings

### 3. Run Phase-1 Pipeline

```bash
python orchestrator_phase1.py
```

## 📊 Implemented Strategies

1. **Small IB Momentum** - Tight initial balance signals trend day
2. **Monday Range Sweep** - Mean reversion from Monday range traps
3. **Asian Liquidity Trap** - Post-NY expansion reversals
4. **London Range Trap** - NY sweeps London range
5. **Fast Spike Trap** - Sharp spikes trap aggressive traders

## 📈 Performance Metrics

- **Primary**: Profit Factor (>1.5 required)
- **Secondary**: Sharpe (>0.5), Sortino, Max Drawdown (<15%)
- **Trade Frequency**: Target 15-20 trades/week

## 🤖 ML Integration

ML models filter rule-based signals using:
- Time-series cross-validation
- Feature engineering from indicators + orderflow
- Threshold optimization
- Models: Logistic, XGBoost, LightGBM

## 📁 Outputs

- `results/approved_strategies.json` - Validated strategies
- `results/edgeless_strategies.json` - Pruned strategies
- `results/reports/phase1_summary.txt` - Summary report
- `models/*.pkl` - Trained ML models

## ⚠️ Disclaimer

Educational and research purposes only. Trading involves risk of loss.

---

**Phase-1 Status**: ✅ Complete
**Last Updated**: 2025-01-14
