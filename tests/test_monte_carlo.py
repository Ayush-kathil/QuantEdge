"""Unit tests for Monte Carlo simulation engine and ruin probability estimation."""
import pytest
import numpy as np
import pandas as pd

from backtesting.execution import Trade
from analytics.monte_carlo import run_trade_monte_carlo, run_daily_return_block_bootstrap


@pytest.fixture
def sample_trades_list():
    """Deterministic trade log fixture for Monte Carlo testing."""
    return [
        Trade(1, "TEST", "LONG", "2024-01-01", "2024-01-05", 100, 105, 10, 1, 1, 50, 48, 0.048, 4),
        Trade(2, "TEST", "LONG", "2024-01-06", "2024-01-10", 105, 102, 10, 1, 1, -30, -32, -0.030, 4),
        Trade(3, "TEST", "LONG", "2024-01-11", "2024-01-15", 102, 108, 10, 1, 1, 60, 58, 0.057, 4),
        Trade(4, "TEST", "LONG", "2024-01-16", "2024-01-20", 108, 104, 10, 1, 1, -40, -42, -0.039, 4),
        Trade(5, "TEST", "LONG", "2024-01-21", "2024-01-25", 104, 112, 10, 1, 1, 80, 78, 0.075, 4),
    ]


def test_monte_carlo_dimensions_and_percentiles(sample_trades_list):
    """Verifies Monte Carlo output dimensions and strict percentile monotonicity."""
    report = run_trade_monte_carlo(
        trades=sample_trades_list,
        initial_capital=100_000.0,
        num_simulations=200,
        horizon_trades=5,
        seed=42
    )

    assert report.num_simulations == 200
    assert report.num_trades_sampled == 5
    assert report.simulated_equity_paths.shape == (200, 6)

    # Monotonicity check across trajectory percentiles at terminal step
    trajs = report.percentile_equity_trajectories
    p05 = trajs["5th"][-1]
    p25 = trajs["25th"][-1]
    p50 = trajs["50th"][-1]
    p75 = trajs["75th"][-1]
    p95 = trajs["95th"][-1]

    assert p05 <= p25 <= p50 <= p75 <= p95


def test_probability_of_ruin_bounds(sample_trades_list):
    """Verifies that ruin probabilities lie within valid [0.0, 1.0] interval."""
    report = run_trade_monte_carlo(
        trades=sample_trades_list,
        initial_capital=100_000.0,
        num_simulations=300,
        seed=123
    )

    assert 0.0 <= report.probability_of_ruin_20pct <= 1.0
    assert 0.0 <= report.probability_of_ruin_30pct <= 1.0
    assert 0.0 <= report.probability_of_ruin_50pct <= 1.0
    # Strict monotonic relationship: P(DD >= 50%) <= P(DD >= 30%) <= P(DD >= 20%)
    assert report.probability_of_ruin_50pct <= report.probability_of_ruin_30pct <= report.probability_of_ruin_20pct


def test_insufficient_trades_monte_carlo_fallback():
    """When fewer than 3 trades are provided, Monte Carlo handles gracefully without crashing."""
    report = run_trade_monte_carlo(trades=[], initial_capital=50_000.0, num_simulations=50)
    assert report.terminal_equity_mean == 50_000.0
    assert report.probability_of_ruin_50pct == 0.0


def test_block_bootstrap_shape():
    """Tests daily return stationary block bootstrap."""
    rets = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005, 0.008, -0.012, 0.018])
    paths = run_daily_return_block_bootstrap(rets, initial_capital=100_000.0, num_simulations=50, block_size=3)
    assert paths.shape == (50, len(rets) + 1)
    assert (paths[:, 0] == 100_000.0).all()
