"""
Flow Playbook Trade Specifications
Contains TradeSpec template definitions and all instantiated strategies.

This module is the canonical source of all trading strategies derived from
the Flow Playbook Intraday PDF.

Author: Phase-1 Edge Discovery System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum

# =============================================================================
# Enums and Base Classes
# =============================================================================

class StrategyType(Enum):
    """Strategy classification"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    TRAP = "trap"
    BREAKOUT = "breakout"
    EXHAUSTION = "exhaustion"
    RECLAIM = "reclaim"


class SessionType(Enum):
    """Trading session types"""
    ASIA = "asia"
    LONDON = "london"
    NY = "ny"
    NY_ETH = "ny_eth"
    ALL = "all"


class DayType(Enum):
    """Day of week constraints"""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    WEEKEND = "weekend"
    ANY = "any"


# =============================================================================
# TradeSpec Template Components
# =============================================================================

@dataclass
class Parameter:
    """Tunable parameter for optimization"""
    name: str
    default_value: float
    min_value: float
    max_value: float
    step: Optional[float] = None
    description: str = ""


@dataclass
class Meta:
    """Strategy metadata"""
    name: str
    id: str
    type: StrategyType
    thesis: str
    version: str = "1.0"


@dataclass
class Context:
    """Market context and regime filters"""
    primary_timeframe: Literal["1m", "5m", "15m"]
    secondary_timeframes: List[Literal["1m", "5m", "15m", "1h", "4h"]] = field(default_factory=list)
    sessions: List[SessionType] = field(default_factory=lambda: [SessionType.ALL])
    day_types: List[DayType] = field(default_factory=lambda: [DayType.ANY])
    allowed_hours_utc: Optional[List[tuple]] = None
    require_htf_trend: bool = False
    htf_timeframe: Optional[str] = None
    htf_trend_indicator: Optional[str] = None
    require_volatility_regime: bool = False
    volatility_measure: Optional[str] = None


@dataclass
class IndicatorSpec:
    """Specification for a required indicator"""
    name: str
    func_name: str
    params: Dict = field(default_factory=dict)
    timeframe: str = "primary"


@dataclass
class EntryCondition:
    """Single boolean condition for entry"""
    description: str
    expression: str
    required: bool = True


@dataclass
class EntryLogic:
    """Entry rules and conditions"""
    trigger_description: str
    conditions: List[EntryCondition]
    require_cvd_confirmation: bool = False
    require_oi_confirmation: bool = False
    require_candle_pattern: Optional[str] = None
    entry_on_close: bool = True
    allow_long: bool = True
    allow_short: bool = True


@dataclass
class ExitRule:
    """Single exit rule specification"""
    name: str
    type: Literal["tp", "sl", "time", "trailing"]
    rule: str
    priority: int = 1


@dataclass
class ExitLogic:
    """Exit rules (TP, SL, time-based)"""
    stop_loss_rules: List[ExitRule]
    take_profit_rules: List[ExitRule]
    max_bars_in_trade: Optional[int] = None
    exit_at_session_close: bool = False
    use_partial_exits: bool = False
    partial_exit_rules: List[ExitRule] = field(default_factory=list)


@dataclass
class RiskSizing:
    """Risk and position sizing rules"""
    risk_per_trade_pct: float = 1.0
    max_leverage: float = 5.0
    use_atr_sizing: bool = False
    atr_multiplier: Optional[float] = None
    max_position_size_usd: Optional[float] = None
    min_position_size_usd: float = 10.0


@dataclass
class Filters:
    """Entry filters to avoid bad setups"""
    min_volume_vs_avg: Optional[float] = None
    max_volume_vs_avg: Optional[float] = None
    avoid_low_liquidity_hours: bool = True
    avoid_news_events: bool = False
    news_buffer_minutes: int = 30
    min_bars_between_entries: int = 5
    max_concurrent_positions: int = 1
    custom_filters: List[str] = field(default_factory=list)


@dataclass
class TradeSpec:
    """Complete specification for a single trading strategy"""
    meta: Meta
    context: Context
    indicators: List[IndicatorSpec]
    entry_logic: EntryLogic
    exit_logic: ExitLogic
    risk_sizing: RiskSizing
    filters: Filters
    parameters: List[Parameter]
    required_data_fields: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume"
    ])
    expected_trades_per_week: Optional[float] = None
    expected_win_rate: Optional[float] = None
    expected_payoff_ratio: Optional[float] = None
    notes: str = ""
    created_date: Optional[str] = None
    last_modified: Optional[str] = None


# =============================================================================
# Instantiated Strategies from Flow Playbook
# =============================================================================

def get_small_ib_momentum_spec() -> TradeSpec:
    """Strategy 1: Small Initial Balance Momentum Breakout"""
    return TradeSpec(
        meta=Meta(
            name="Small Initial Balance Momentum Breakout",
            id="IB_MOM",
            type=StrategyType.MOMENTUM,
            thesis="When IB (first hour) is tight vs recent volatility (ASR), trend day probability increases. "
                   "Breakout of IB with strong delta/OI signals directional conviction."
        ),
        context=Context(
            primary_timeframe="5m",
            secondary_timeframes=["1m", "15m", "1h"],
            sessions=[SessionType.ASIA, SessionType.LONDON],
            day_types=[DayType.ANY],
            allowed_hours_utc=[(0, 12)],
            require_volatility_regime=True,
            volatility_measure="ib_vs_asr"
        ),
        indicators=[
            IndicatorSpec("ib_range", "calculate_initial_balance",
                         {"session_start_hour": 0, "ib_duration_hours": 1}),
            IndicatorSpec("asr_20", "average_session_range", {"lookback": 20}),
            IndicatorSpec("vwap", "calculate_vwap", {}),
            IndicatorSpec("cvd", "cumulative_volume_delta", {}),
            IndicatorSpec("oi", "open_interest", {}),
        ],
        entry_logic=EntryLogic(
            trigger_description="Breakout of IB high/low with delta and OI confirmation",
            conditions=[
                EntryCondition("IB size < threshold * ASR",
                              "ib_range < ib_threshold_pct * asr_20", required=True),
                EntryCondition("Price breaks IB high (long) or low (short)",
                              "(close > ib_high and side == 'long') or (close < ib_low and side == 'short')",
                              required=True),
                EntryCondition("Strong delta in breakout direction",
                              "abs(delta_change_5) > delta_threshold", required=True),
                EntryCondition("OI increasing",
                              "oi_change_pct > oi_threshold", required=True),
            ],
            require_cvd_confirmation=True,
            require_oi_confirmation=True,
            entry_on_close=True
        ),
        exit_logic=ExitLogic(
            stop_loss_rules=[
                ExitRule("IB midpoint stop", "sl",
                        "ib_high if side == 'short' else ib_low", priority=1)
            ],
            take_profit_rules=[
                ExitRule("2x IB range", "tp",
                        "entry_price + (2 * ib_range if side == 'long' else -2 * ib_range)",
                        priority=1),
            ],
            max_bars_in_trade=60
        ),
        risk_sizing=RiskSizing(risk_per_trade_pct=1.0, max_leverage=3.0),
        filters=Filters(
            min_volume_vs_avg=0.8,
            min_bars_between_entries=12,
            custom_filters=["ib_range < 1.5 * asr_20"]
        ),
        parameters=[
            Parameter("ib_threshold_pct", 1.0, 0.5, 1.5, 0.1,
                     "IB size as % of ASR for setup validity"),
            Parameter("delta_threshold", 100, 50, 500, 50,
                     "Minimum CVD delta for breakout confirmation"),
            Parameter("oi_threshold", 0.5, 0.1, 2.0, 0.1,
                     "Minimum OI % change for confirmation"),
        ],
        required_data_fields=["open", "high", "low", "close", "volume", "cvd", "delta", "oi"],
        expected_trades_per_week=3.0,
        notes="Most effective during Asia session post NYSE close"
    )


def get_monday_range_sweep_spec() -> TradeSpec:
    """Strategy 2: Monday Range Sweep Mean Reversion"""
    return TradeSpec(
        meta=Meta(
            name="Monday Range Sweep Mean Reversion",
            id="MON_SWEEP",
            type=StrategyType.TRAP,
            thesis="Monday often sets weekly high/low. Tuesday/Wednesday sweep of Monday range traps "
                   "breakout traders, creating mean reversion opportunity back to range."
        ),
        context=Context(
            primary_timeframe="5m",
            secondary_timeframes=["15m", "1h"],
            sessions=[SessionType.NY],
            day_types=[DayType.TUESDAY, DayType.WEDNESDAY],
            allowed_hours_utc=[(12, 20)]
        ),
        indicators=[
            IndicatorSpec("monday_range", "monday_high_low", {}),
            IndicatorSpec("vwap", "calculate_vwap", {}),
            IndicatorSpec("cvd", "cumulative_volume_delta", {}),
            IndicatorSpec("oi", "open_interest", {}),
        ],
        entry_logic=EntryLogic(
            trigger_description="Sweep Monday high/low, then reclaim back inside with rejection pattern",
            conditions=[
                EntryCondition("Price sweeps Monday level",
                              "(high > monday_high and side == 'short') or (low < monday_low and side == 'long')",
                              required=True),
                EntryCondition("Wick rejection or SFP",
                              "total_wick_ratio > wick_threshold", required=True),
                EntryCondition("Reclaim inside range",
                              "(close < monday_high and side == 'short') or (close > monday_low and side == 'long')",
                              required=True),
            ],
            require_candle_pattern="engulfing",
            entry_on_close=True
        ),
        exit_logic=ExitLogic(
            stop_loss_rules=[
                ExitRule("Beyond sweep wick", "sl",
                        "high if side == 'long' else low", priority=1)
            ],
            take_profit_rules=[
                ExitRule("Opposite end of Monday range", "tp",
                        "monday_low if side == 'short' else monday_high", priority=1),
            ],
            max_bars_in_trade=120
        ),
        risk_sizing=RiskSizing(risk_per_trade_pct=1.5, max_leverage=5.0),
        filters=Filters(
            custom_filters=["is_tuesday or is_wednesday", "(monday_high - monday_low) > min_range_size"]
        ),
        parameters=[
            Parameter("wick_threshold", 0.6, 0.4, 0.8, 0.05, "Wick size as ratio of candle range"),
            Parameter("min_range_size", 200, 100, 500, 50, "Minimum Monday range size in $"),
        ],
        required_data_fields=["open", "high", "low", "close", "volume", "cvd", "delta", "oi"],
        expected_trades_per_week=2.0
    )


def get_asia_liquidity_trap_spec() -> TradeSpec:
    """Strategy 3: Asian Session Liquidity Trap Reversal"""
    return TradeSpec(
        meta=Meta(
            name="Asian Session Liquidity Trap Reversal",
            id="ASIA_TRAP",
            type=StrategyType.TRAP,
            thesis="Post-expansive NY session, price sweeps liquidity during Asia open but lacks "
                   "follow-through, trapping late entries and reversing."
        ),
        context=Context(
            primary_timeframe="5m",
            secondary_timeframes=["1m", "15m"],
            sessions=[SessionType.ASIA, SessionType.NY_ETH],
            allowed_hours_utc=[(20, 8)]
        ),
        indicators=[
            IndicatorSpec("ny_session_range", "session_range", {"session": "ny"}),
            IndicatorSpec("vwap", "calculate_vwap", {}),
            IndicatorSpec("cvd", "cumulative_volume_delta", {}),
            IndicatorSpec("oi", "open_interest", {}),
        ],
        entry_logic=EntryLogic(
            trigger_description="False breakout beyond key level with exhaustion signs, then reclaim",
            conditions=[
                EntryCondition("Price breaks key level",
                              "(high > ny_high and side == 'short') or (low < ny_low and side == 'long')",
                              required=True),
                EntryCondition("Large wick rejection",
                              "total_wick_ratio > wick_threshold", required=True),
                EntryCondition("Reclaim of broken level",
                              "(close < ny_high and side == 'short') or (close > ny_low and side == 'long')",
                              required=True),
            ],
            require_cvd_confirmation=True,
            require_oi_confirmation=True,
            entry_on_close=True
        ),
        exit_logic=ExitLogic(
            stop_loss_rules=[
                ExitRule("Beyond wick high", "sl",
                        "high + buffer if side == 'long' else low - buffer", priority=1)
            ],
            take_profit_rules=[
                ExitRule("Mid-range", "tp", "(ny_high + ny_low) / 2", priority=1),
            ],
            max_bars_in_trade=72
        ),
        risk_sizing=RiskSizing(risk_per_trade_pct=1.0, max_leverage=5.0),
        filters=Filters(
            min_volume_vs_avg=1.5,
            custom_filters=["prev_session_was_expansive"]
        ),
        parameters=[
            Parameter("wick_threshold", 0.65, 0.5, 0.85, 0.05, "Minimum wick ratio for rejection"),
            Parameter("buffer", 10, 5, 30, 5, "Buffer in $ beyond wick for stop"),
        ],
        required_data_fields=["open", "high", "low", "close", "volume", "cvd", "delta", "oi"],
        expected_trades_per_week=2.5
    )


def get_london_range_trap_spec() -> TradeSpec:
    """Strategy 4: London Range Trap and Reversal"""
    return TradeSpec(
        meta=Meta(
            name="London Range Trap and Reversal",
            id="LON_TRAP",
            type=StrategyType.TRAP,
            thesis="London session (3-8am ET) least likely to define daily high/low. NY session sweeps "
                   "this range, trapping traders before reversing."
        ),
        context=Context(
            primary_timeframe="5m",
            secondary_timeframes=["15m"],
            sessions=[SessionType.NY],
            allowed_hours_utc=[(12, 16)]
        ),
        indicators=[
            IndicatorSpec("london_range", "session_range", {"session": "london"}),
            IndicatorSpec("vwap", "calculate_vwap", {}),
            IndicatorSpec("cvd", "cumulative_volume_delta", {}),
        ],
        entry_logic=EntryLogic(
            trigger_description="NY sweeps London high/low, gets rejected, reclaims range",
            conditions=[
                EntryCondition("Sweep of London level",
                              "(high > london_high and side == 'short') or (low < london_low and side == 'long')",
                              required=True),
                EntryCondition("Wick rejection",
                              "total_wick_ratio > wick_threshold", required=True),
                EntryCondition("Reclaim of range",
                              "(close < london_high and side == 'short') or (close > london_low and side == 'long')",
                              required=True),
            ],
            entry_on_close=True
        ),
        exit_logic=ExitLogic(
            stop_loss_rules=[
                ExitRule("Beyond sweep", "sl",
                        "high + buffer if side == 'long' else low - buffer", priority=1)
            ],
            take_profit_rules=[
                ExitRule("Midpoint", "tp", "(london_high + london_low) / 2", priority=1),
            ],
            max_bars_in_trade=48
        ),
        risk_sizing=RiskSizing(risk_per_trade_pct=1.2, max_leverage=5.0),
        filters=Filters(
            custom_filters=["(london_high - london_low) > min_range_size"]
        ),
        parameters=[
            Parameter("wick_threshold", 0.6, 0.4, 0.8, 0.05, "Minimum wick ratio"),
            Parameter("min_range_size", 150, 50, 400, 25, "Minimum London range size in $"),
            Parameter("buffer", 15, 5, 40, 5, "Stop buffer in $"),
        ],
        required_data_fields=["open", "high", "low", "close", "volume", "cvd", "delta"],
        expected_trades_per_week=3.0
    )


def get_fast_spike_trap_spec() -> TradeSpec:
    """Strategy 5: Fast Spike Trapped Delta Reversal"""
    return TradeSpec(
        meta=Meta(
            name="Fast Spike Trapped Delta Reversal",
            id="SPIKE_TRAP",
            type=StrategyType.EXHAUSTION,
            thesis="Sharp spikes into key levels trap aggressive participants. Second push runs stops, "
                   "then reverses as trapped traders are forced out."
        ),
        context=Context(
            primary_timeframe="1m",
            secondary_timeframes=["5m"],
            sessions=[SessionType.ALL]
        ),
        indicators=[
            IndicatorSpec("vwap", "calculate_vwap", {}),
            IndicatorSpec("cvd", "cumulative_volume_delta", {}),
            IndicatorSpec("oi", "open_interest", {}),
            IndicatorSpec("atr", "average_true_range", {"period": 14}),
        ],
        entry_logic=EntryLogic(
            trigger_description="Fast spike into level, trapped delta visible, reversal confirmed",
            conditions=[
                EntryCondition("Price spike exceeds threshold",
                              "price_change_pct > spike_threshold", required=True),
                EntryCondition("Spike is fast",
                              "bars_to_spike <= max_bars_for_spike", required=True),
                EntryCondition("Delta surge",
                              "abs(delta_spike) > delta_threshold", required=True),
                EntryCondition("OI jump",
                              "oi_change_pct > oi_threshold", required=True),
                EntryCondition("Rejection",
                              "wick_ratio > wick_threshold or is_engulfing", required=True),
            ],
            require_cvd_confirmation=True,
            require_oi_confirmation=True,
            entry_on_close=True
        ),
        exit_logic=ExitLogic(
            stop_loss_rules=[
                ExitRule("Outside spike", "sl",
                        "high + buffer if side == 'long' else low - buffer", priority=1)
            ],
            take_profit_rules=[
                ExitRule("Return to VWAP", "tp", "vwap", priority=1),
            ],
            max_bars_in_trade=60
        ),
        risk_sizing=RiskSizing(
            risk_per_trade_pct=1.0,
            max_leverage=5.0,
            use_atr_sizing=True,
            atr_multiplier=1.5
        ),
        filters=Filters(
            min_volume_vs_avg=2.0,
            custom_filters=["not slow_grind_into_level", "level_is_key_structure"]
        ),
        parameters=[
            Parameter("spike_threshold", 0.5, 0.2, 1.5, 0.1, "Minimum price change % for spike"),
            Parameter("max_bars_for_spike", 3, 1, 5, 1, "Max bars for spike (must be fast)"),
            Parameter("delta_threshold", 150, 50, 500, 50, "Minimum delta spike magnitude"),
            Parameter("oi_threshold", 1.0, 0.3, 3.0, 0.2, "Minimum OI % increase"),
            Parameter("wick_threshold", 0.6, 0.4, 0.8, 0.05, "Minimum wick ratio"),
            Parameter("buffer", 10, 5, 25, 5, "Stop buffer in $"),
        ],
        required_data_fields=["open", "high", "low", "close", "volume", "cvd", "delta", "oi"],
        expected_trades_per_week=4.0,
        notes="Avoid slow grinds. Clean acceptance invalidates setup."
    )


# =============================================================================
# Strategy Registry
# =============================================================================

def get_all_strategies() -> Dict[str, TradeSpec]:
    """
    Get dictionary of all available strategies.

    Returns:
        Dict mapping strategy ID to TradeSpec
    """
    strategies = {
        "IB_MOM": get_small_ib_momentum_spec(),
        "MON_SWEEP": get_monday_range_sweep_spec(),
        "ASIA_TRAP": get_asia_liquidity_trap_spec(),
        "LON_TRAP": get_london_range_trap_spec(),
        "SPIKE_TRAP": get_fast_spike_trap_spec(),
    }

    return strategies


def get_strategy_by_id(strategy_id: str) -> TradeSpec:
    """
    Get a specific strategy by ID.

    Args:
        strategy_id: Strategy identifier (e.g., 'IB_MOM')

    Returns:
        TradeSpec for the requested strategy

    Raises:
        KeyError if strategy not found
    """
    strategies = get_all_strategies()

    if strategy_id not in strategies:
        raise KeyError(f"Strategy '{strategy_id}' not found. Available: {list(strategies.keys())}")

    return strategies[strategy_id]


if __name__ == "__main__":
    # Example usage
    print("Flow Playbook Strategies")
    print("=" * 80)

    strategies = get_all_strategies()

    for strat_id, spec in strategies.items():
        print(f"\nStrategy: {spec.meta.name} ({spec.meta.id})")
        print(f"Type: {spec.meta.type.value}")
        print(f"Thesis: {spec.meta.thesis[:100]}...")
        print(f"Timeframe: {spec.context.primary_timeframe}")
        print(f"Expected trades/week: {spec.expected_trades_per_week}")
        print(f"Parameters: {len(spec.parameters)}")

    print(f"\n\nTotal strategies: {len(strategies)}")
