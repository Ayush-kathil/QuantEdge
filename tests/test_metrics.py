"""Unit tests validating quantitative performance and risk metrics."""
import pytest
import numpy as np
import pandas as pd

from analytics.risk import compute_drawdowns, compute_downside_deviation
from analytics.returns import compute_daily_returns
from backtesting.metrics import calculate_performance_metrics
from backtesting.execution import Trade


def test_drawdown_calculation():
    """Validates maximum drawdown and duration calculation on known equity path."""
    # Path: 100 -> 120 (Peak) -> 90 (Trough, 25% drop) -> 120 (Recovery) -> 130
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.Series([100.0, 120.0, 90.0, 120.0, 130.0], index=dates)

    dd = compute_drawdowns(equity)
    # Peak is 120, trough is 90 -> drawdown is (90 - 120) / 120 = -30 / 120 = -0.25 (25%)
    assert pytest.approx(dd.max_drawdown) == 0.25
    assert pytest.approx(dd.max_drawdown_dollars) == 30.0
    assert dd.max_drawdown_peak_date == "2024-01-02"
    assert dd.max_drawdown_trough_date == "2024-01-03"
    assert dd.max_drawdown_recovery_date == "2024-01-04"


def test_sharpe_and_sortino_ratio():
    """Tests annualized Sharpe and Sortino ratio calculation."""
    # Synthetic constant daily return series with known mean and std
    np.random.seed(42)
    n = 252
    daily_rets = pd.Series(np.random.normal(loc=0.001, scale=0.01, size=n))
    equity = (1.0 + daily_rets).cumprod() * 100_000.0

    metrics = calculate_performance_metrics(
        equity_curve=equity,
        daily_returns=daily_rets,
        trades=[],
        risk_free_rate=0.0,
        trading_days=252
    )

    # Expected Sharpe = (mean_daily * 252) / (std_daily * sqrt(252))
    expected_vol = float(daily_rets.std(ddof=1) * np.sqrt(252))
    expected_sharpe = float(daily_rets.mean() * 252) / expected_vol

    assert pytest.approx(metrics.annualized_volatility, rel=1e-3) == expected_vol
    assert pytest.approx(metrics.sharpe_ratio, rel=1e-3) == expected_sharpe
    assert metrics.sortino_ratio is not None


def test_profit_factor_and_win_rate():
    """Verifies profit factor and win rate calculations with trades."""
    trades = [
        Trade(1, "TEST", "LONG", "2024-01-01", "2024-01-05", 100, 110, 10, 2, 2, 100, 96, 0.096, 4),
        Trade(2, "TEST", "LONG", "2024-01-06", "2024-01-10", 100, 95, 10, 2, 2, -50, -54, -0.054, 4),
        Trade(3, "TEST", "LONG", "2024-01-11", "2024-01-15", 100, 105, 10, 2, 2, 50, 46, 0.046, 4)
    ]
    # Wins: +96, +46 (Gross profits: 142)
    # Losses: -54 (Gross losses: 54)
    # Win rate: 2 / 3 = 66.67%
    # Profit factor: 142 / 54 = 2.6296
    dates = pd.date_range("2024-01-01", periods=10)
    equity = pd.Series(np.linspace(100_000, 105_000, 10), index=dates)
    daily_rets = equity.pct_change().fillna(0)

    metrics = calculate_performance_metrics(
        equity_curve=equity,
        daily_returns=daily_rets,
        trades=trades
    )

    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert pytest.approx(metrics.win_rate) == 2.0 / 3.0
    assert pytest.approx(metrics.profit_factor) == 142.0 / 54.0


def test_infinite_profit_factor_on_zero_losses():
    """When a strategy has only winning trades, profit factor should be inf rather than erroring."""
    trades = [
        Trade(1, "TEST", "LONG", "2024-01-01", "2024-01-05", 100, 110, 10, 2, 2, 100, 96, 0.096, 4)
    ]
    dates = pd.date_range("2024-01-01", periods=5)
    equity = pd.Series([100_000, 100_050, 100_070, 100_080, 100_096], index=dates)
    daily_rets = equity.pct_change().fillna(0)

    metrics = calculate_performance_metrics(
        equity_curve=equity,
        daily_returns=daily_rets,
        trades=trades
    )
    assert np.isinf(metrics.profit_factor)
    assert metrics.win_rate == 1.0


def test_zero_volatility_edge_case():
    """Flat equity curve should safely produce 0 volatility and Sharpe without ZeroDivisionError."""
    dates = pd.date_range("2024-01-01", periods=20)
    equity = pd.Series(100_000.0, index=dates)
    daily_rets = pd.Series(0.0, index=dates)

    metrics = calculate_performance_metrics(
        equity_curve=equity,
        daily_returns=daily_rets,
        trades=[]
    )
    assert metrics.annualized_volatility == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.total_return == 0.0
