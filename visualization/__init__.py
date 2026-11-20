"""Visualization and interactive plotting package."""
from visualization.charts import (
    apply_quant_layout,
    plot_market_explorer_candlestick,
    plot_returns_distribution,
    plot_rolling_volatility_comparison,
    plot_technical_indicators_suite,
    plot_equity_curves,
    plot_drawdown_underwater,
    plot_trade_markers_on_price,
    plot_autocorrelation_bar,
    QUANT_THEME
)
from visualization.dashboard import (
    render_metric_card,
    render_performance_summary_grid,
    format_dataframe_for_display
)

__all__ = [
    "apply_quant_layout",
    "plot_market_explorer_candlestick",
    "plot_returns_distribution",
    "plot_rolling_volatility_comparison",
    "plot_technical_indicators_suite",
    "plot_equity_curves",
    "plot_drawdown_underwater",
    "plot_trade_markers_on_price",
    "plot_autocorrelation_bar",
    "QUANT_THEME",
    "render_metric_card",
    "render_performance_summary_grid",
    "format_dataframe_for_display"
]
