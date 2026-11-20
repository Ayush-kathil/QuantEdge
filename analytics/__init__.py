"""Analytics package: returns, volatility, indicators, statistics, risk, regimes, monte carlo, and reporting."""
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
from analytics.monte_carlo import (
    MonteCarloReport,
    run_trade_monte_carlo,
    run_daily_return_block_bootstrap
)
from analytics.regimes import (
    RegimeClassification,
    RegimePerformanceAttribution,
    classify_market_regimes,
    attribute_performance_by_regime
)
from analytics.reproducibility import (
    ExperimentMetadata,
    generate_experiment_manifest
)
from analytics.reporting import (
    generate_pdf_research_report
)
from analytics.futures_accounting import (
    FuturesContractSpec,
    FuturesTradeRecord,
    FuturesSimulationResult,
    get_futures_spec,
    simulate_futures_account,
    FUTURES_SPECS_REGISTRY
)
from analytics.portfolio import (
    PortfolioOptimizationResult,
    compute_rolling_beta,
    compute_rolling_correlation,
    compute_risk_contribution,
    optimize_portfolio
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
    "compute_conditional_var",
    "MonteCarloReport",
    "run_trade_monte_carlo",
    "run_daily_return_block_bootstrap",
    "RegimeClassification",
    "RegimePerformanceAttribution",
    "classify_market_regimes",
    "attribute_performance_by_regime",
    "ExperimentMetadata",
    "generate_experiment_manifest",
    "generate_pdf_research_report",
]
