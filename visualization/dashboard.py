"""Dashboard helper utilities and reusable UI components for QuantEdge."""
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
from backtesting.metrics import PerformanceMetrics


def render_metric_card(label: str, value: str, delta: Optional[str] = None, delta_color: str = "normal") -> None:
    """Renders a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_performance_summary_grid(metrics: PerformanceMetrics, benchmark_metrics: Optional[PerformanceMetrics] = None) -> None:
    """Renders a coordinated 6-card performance summary grid."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    bm_ret = benchmark_metrics.total_return if benchmark_metrics else None
    ret_delta = f"{(metrics.total_return - bm_ret)*100:+.2f}% vs B&H" if bm_ret is not None else None

    col1.metric("Total Return", f"{metrics.total_return * 100:+.2f}%", delta=ret_delta)
    col2.metric("CAGR", f"{metrics.cagr * 100:.2f}%" if not pd.isna(metrics.cagr) else "N/A")
    col3.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    col4.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    col5.metric("Max Drawdown", f"-{metrics.max_drawdown * 100:.2f}%")
    col6.metric("Win Rate", f"{metrics.win_rate * 100:.1f}%", f"{metrics.winning_trades}W / {metrics.losing_trades}L")


def format_dataframe_for_display(df: pd.DataFrame, precision: int = 2) -> pd.DataFrame:
    """Formats numeric columns of DataFrame for presentation."""
    formatted = df.copy()
    for col in formatted.select_dtypes(include=["float", "float64"]).columns:
        formatted[col] = formatted[col].round(precision)
    return formatted
