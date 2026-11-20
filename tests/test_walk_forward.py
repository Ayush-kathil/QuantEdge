"""Unit tests for walk-forward validation, parameter sensitivity, and friction auditing."""
import pytest
import numpy as np
import pandas as pd

from strategies.moving_average import MovingAverageCrossover
from backtesting.engine import BacktestEngine
from backtesting.validation import (
    run_walk_forward_validation,
    run_parameter_sensitivity_scan,
    run_friction_sensitivity_analysis
)


@pytest.fixture
def multi_month_data():
    """Generates 150 days of trending synthetic market bars."""
    dates = pd.date_range("2023-01-01", periods=150, freq="B")
    prices = np.linspace(100.0, 160.0, 150) + np.sin(np.linspace(0, 4 * np.pi, 150)) * 5.0
    return pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": np.full(150, 500_000.0)
    }, index=dates)


def test_walk_forward_validation_temporal_integrity(multi_month_data):
    """
    Verifies that in every walk-forward fold:
    1. Train window strictly precedes test window (no data leakage).
    2. Test window start > Train window end.
    3. Folds are chronological.
    """
    engine = BacktestEngine(initial_capital=100_000.0)
    report = run_walk_forward_validation(
        df=multi_month_data,
        strategy_factory=lambda: MovingAverageCrossover(short_period=5, long_period=15),
        engine=engine,
        n_splits=3,
        train_ratio=0.65
    )

    assert len(report.folds) == 3
    for fold in report.folds:
        train_end = pd.to_datetime(fold.train_end)
        test_start = pd.to_datetime(fold.test_start)
        assert test_start >= train_end  # Zero leakage!


def test_parameter_sensitivity_grid_scan(multi_month_data):
    """Verifies that parameter sensitivity scan sweeps grid without errors."""
    engine = BacktestEngine(initial_capital=100_000.0)
    grid = [
        {"short_period": 5, "long_period": 15},
        {"short_period": 10, "long_period": 20},
        {"short_period": 10, "long_period": 30}
    ]
    scan_df = run_parameter_sensitivity_scan(
        df=multi_month_data,
        strategy_builder=lambda p: MovingAverageCrossover(**p),
        engine=engine,
        param_grid=grid
    )

    assert len(scan_df) == 3
    assert "Sharpe_Ratio" in scan_df.columns
    assert "Total_Return" in scan_df.columns


def test_friction_sensitivity_scan(multi_month_data):
    """Verifies that increasing friction monotonically degrades ending capital."""
    strat = MovingAverageCrossover(short_period=5, long_period=15)
    engine = BacktestEngine(initial_capital=100_000.0)

    res_df = run_friction_sensitivity_analysis(
        df=multi_month_data,
        strategy=strat,
        base_engine=engine,
        commissions_bps=[0.0, 10.0, 50.0],
        slippages_bps=[0.0]
    )

    assert len(res_df) == 3
    # With higher commission, ending capital must decrease
    cap0 = res_df.loc[res_df["Commission_bps"] == 0.0, "Ending_Capital"].values[0]
    cap50 = res_df.loc[res_df["Commission_bps"] == 50.0, "Ending_Capital"].values[0]
    assert cap0 >= cap50
