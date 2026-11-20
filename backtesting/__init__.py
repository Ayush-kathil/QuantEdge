"""Backtesting engine, execution modeling, metrics, and out-of-sample validation."""
from backtesting.execution import Trade, ExecutionModel
from backtesting.metrics import PerformanceMetrics, calculate_performance_metrics
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.validation import (
    ValidationReport,
    WalkForwardFoldResult,
    WalkForwardReport,
    chronological_split,
    run_out_of_sample_validation,
    run_walk_forward_validation,
    run_parameter_sensitivity_scan,
    run_friction_sensitivity_analysis
)

__all__ = [
    "Trade",
    "ExecutionModel",
    "PerformanceMetrics",
    "calculate_performance_metrics",
    "BacktestEngine",
    "BacktestResult",
    "ValidationReport",
    "WalkForwardFoldResult",
    "WalkForwardReport",
    "chronological_split",
    "run_out_of_sample_validation",
    "run_walk_forward_validation",
    "run_parameter_sensitivity_scan",
    "run_friction_sensitivity_analysis",
]
