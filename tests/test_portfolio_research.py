"""
Tests for Portfolio Research, Cross-Asset Analytics, Rolling Beta, and Markowitz Optimization.
"""
import pytest
import numpy as np
import pandas as pd
from analytics.portfolio import (
    compute_rolling_beta,
    compute_rolling_correlation,
    compute_risk_contribution,
    optimize_portfolio
)


@pytest.fixture
def multi_asset_returns() -> pd.DataFrame:
    """3 correlated asset returns over 252 days."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    r1 = np.random.normal(0.0008, 0.012, n)
    r2 = 0.6 * r1 + np.random.normal(0.0004, 0.010, n)
    r3 = -0.2 * r1 + np.random.normal(0.0002, 0.008, n)

    return pd.DataFrame({
        "Asset_A": r1,
        "Asset_B": r2,
        "Asset_C": r3
    }, index=dates)


def test_rolling_beta_and_alpha(multi_asset_returns):
    s = multi_asset_returns["Asset_B"]
    b = multi_asset_returns["Asset_A"]
    rolling_df = compute_rolling_beta(s, b, window=63)
    assert "rolling_beta" in rolling_df.columns
    assert "rolling_alpha" in rolling_df.columns
    assert "rolling_r2" in rolling_df.columns
    valid_beta = rolling_df["rolling_beta"].dropna()
    assert len(valid_beta) > 0
    # Mean rolling beta should be close to 0.6
    assert np.mean(valid_beta) == pytest.approx(0.6, abs=0.2)


def test_rolling_correlation(multi_asset_returns):
    r_corr = compute_rolling_correlation(multi_asset_returns["Asset_A"], multi_asset_returns["Asset_B"], window=63)
    valid_corr = r_corr.dropna()
    assert len(valid_corr) > 0
    assert (valid_corr >= -1.0).all() and (valid_corr <= 1.0).all()


def test_markowitz_portfolio_optimization(multi_asset_returns):
    res = optimize_portfolio(multi_asset_returns, risk_free_rate=0.04)

    # Weights sum to 1.0
    sum_max_sharpe = sum(res.weights_max_sharpe.values())
    sum_min_var = sum(res.weights_min_var.values())
    sum_equal = sum(res.weights_equal.values())

    assert sum_max_sharpe == pytest.approx(1.0, abs=1e-5)
    assert sum_min_var == pytest.approx(1.0, abs=1e-5)
    assert sum_equal == pytest.approx(1.0, abs=1e-5)

    # Min variance volatility must be <= equal weight volatility
    assert res.min_var_volatility <= res.equal_weight_volatility + 1e-6
    # Max Sharpe ratio must be >= equal weight Sharpe ratio
    assert res.max_sharpe_ratio >= res.equal_weight_sharpe_ratio - 1e-4

    # Risk contributions sum to 1.0 (100%)
    assert sum(res.risk_contributions_max_sharpe.values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(res.risk_contributions_min_var.values()) == pytest.approx(1.0, abs=1e-4)
