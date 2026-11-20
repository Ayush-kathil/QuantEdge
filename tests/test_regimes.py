"""Unit tests for statistical market regime classification and performance attribution."""
import pytest
import numpy as np
import pandas as pd

from analytics.regimes import classify_market_regimes, attribute_performance_by_regime


@pytest.fixture
def synthetic_regime_data():
    """Generates a 120-day synthetic dataset with clear trend and volatility shifts."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    # Low-vol flat first 40 days, High-vol trending next 80 days
    p1 = 100.0 + np.random.normal(0, 0.2, 40)
    p2 = 100.0 + np.linspace(1, 30, 80) + np.random.normal(0, 2.0, 80)
    prices = np.concatenate([p1, p2])

    df = pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 1.5,
        "Low": prices - 1.5,
        "Close": prices,
        "Volume": np.full(120, 100_000.0)
    }, index=dates)
    return df


def test_regime_classification_labels(synthetic_regime_data):
    """Verifies that regime classification outputs valid categorical strings."""
    regimes = classify_market_regimes(synthetic_regime_data, vol_lookback=20, trend_ma_period=20)

    valid_vol_tags = {"HIGH_VOL", "NORMAL_VOL", "LOW_VOL"}
    valid_trend_tags = {"BULL_TREND", "BEAR_TREND", "RANGE_BOUND"}

    unique_vol = set(regimes.volatility_regime.unique())
    unique_trend = set(regimes.trend_regime.unique())

    assert unique_vol.issubset(valid_vol_tags)
    assert unique_trend.issubset(valid_trend_tags)
    assert len(regimes.combined_regime) == len(synthetic_regime_data)


def test_regime_attribution_computation(synthetic_regime_data):
    """Verifies performance attribution by regime executes without numerical errors."""
    regimes = classify_market_regimes(synthetic_regime_data, vol_lookback=20, trend_ma_period=20)
    strat_rets = synthetic_regime_data["Close"].pct_change().fillna(0.0)
    bm_rets = strat_rets * 0.9

    attr = attribute_performance_by_regime(strat_rets, bm_rets, regimes)

    assert not attr.volatility_attribution.empty
    assert not attr.trend_attribution.empty
    assert "Regime" in attr.volatility_attribution.columns
    assert "Strategy_Sharpe" in attr.volatility_attribution.columns
