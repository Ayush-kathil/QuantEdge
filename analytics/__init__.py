"""Analytics package: returns, volatility, indicators, statistics, and risk modeling."""
from analytics.returns import (
    compute_daily_returns,
    compute_cumulative_returns,
    compute_rolling_returns,
    compute_excess_returns
)
from analytics.volatility import (
    compute_realized_volatility,
    compute_rolling_volatility,
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_ewma_volatility
)
from analytics.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_bollinger_bands
)
from analytics.statistics import (
    DistributionStats,
    compute_distribution_statistics,
    compute_correlation_matrix
)
from analytics.risk import (
    DrawdownAnalysis,
    compute_drawdowns,
    compute_downside_deviation,
    compute_parametric_var,
    compute_historical_var,
    compute_conditional_var
)

__all__ = [
    "compute_daily_returns",
    "compute_cumulative_returns",
    "compute_rolling_returns",
    "compute_excess_returns",
    "compute_realized_volatility",
    "compute_rolling_volatility",
    "compute_parkinson_volatility",
    "compute_garman_klass_volatility",
    "compute_ewma_volatility",
    "compute_sma",
    "compute_ema",
    "compute_rsi",
    "compute_macd",
    "compute_atr",
    "compute_bollinger_bands",
    "DistributionStats",
    "compute_distribution_statistics",
    "compute_correlation_matrix",
    "DrawdownAnalysis",
    "compute_drawdowns",
    "compute_downside_deviation",
    "compute_parametric_var",
    "compute_historical_var",
    "compute_conditional_var"
]
