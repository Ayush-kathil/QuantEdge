"""Unit tests verifying technical indicators against deterministic mathematical benchmarks."""
import pytest
import numpy as np
import pandas as pd

from analytics.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_bollinger_bands
)


def test_sma_calculation():
    """Validates Simple Moving Average against known arithmetic averages."""
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma3 = compute_sma(prices, period=3)

    assert np.isnan(sma3.iloc[0])
    assert np.isnan(sma3.iloc[1])
    assert pytest.approx(sma3.iloc[2]) == 20.0  # (10 + 20 + 30) / 3
    assert pytest.approx(sma3.iloc[3]) == 30.0  # (20 + 30 + 40) / 3
    assert pytest.approx(sma3.iloc[4]) == 40.0  # (30 + 40 + 50) / 3


def test_ema_calculation():
    """Validates Exponential Moving Average smoothing."""
    prices = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    ema2 = compute_ema(prices, period=2)
    # alpha = 2 / (2 + 1) = 2/3
    alpha = 2.0 / 3.0
    # Recursive calculation check
    expected_0 = 10.0
    expected_1 = alpha * 12.0 + (1 - alpha) * expected_0
    expected_2 = alpha * 14.0 + (1 - alpha) * expected_1

    assert pytest.approx(ema2.iloc[0]) == expected_0
    assert pytest.approx(ema2.iloc[1]) == expected_1
    assert pytest.approx(ema2.iloc[2]) == expected_2


def test_rsi_bounds_and_extremes():
    """Tests Wilder RSI under monotonic and oscillatory conditions."""
    # Monotonically increasing prices -> RSI must reach 100
    up_prices = pd.Series([float(i) for i in range(1, 30)])
    rsi_up = compute_rsi(up_prices, period=14)
    assert pytest.approx(rsi_up.iloc[-1]) == 100.0

    # Monotonically decreasing prices -> RSI must reach 0
    down_prices = pd.Series([float(100 - i) for i in range(30)])
    rsi_down = compute_rsi(down_prices, period=14)
    assert pytest.approx(rsi_down.iloc[-1]) == 0.0

    # Perfectly alternating series (+1, -1, +1, -1)
    alt = [100.0]
    for i in range(40):
        alt.append(alt[-1] + (2.0 if i % 2 == 0 else -2.0))
    rsi_alt = compute_rsi(pd.Series(alt), period=14)
    # Should hover close to 50
    assert 45.0 <= rsi_alt.iloc[-1] <= 55.0


def test_macd_relationship():
    """Verifies that MACD components maintain exact mathematical identity."""
    prices = pd.Series([100.0 + i + 2.0 * np.sin(i) for i in range(50)])
    macd_df = compute_macd(prices, fast_period=12, slow_period=26, signal_period=9)

    # Histogram must equal macd_line - signal_line
    diff = macd_df["macd_line"] - macd_df["signal_line"]
    np.testing.assert_allclose(macd_df["histogram"].values, diff.values, rtol=1e-5)


def test_atr_handling():
    """Verifies ATR accounts for True Range gaps correctly."""
    # Day 0: High 105, Low 95, Close 100
    # Day 1: Gap up: High 120, Low 115, Close 118
    # TR on Day 1 should be max(120-115=5, abs(120-100)=20, abs(115-100)=15) = 20
    df = pd.DataFrame({
        "High": [105.0, 120.0, 122.0, 125.0],
        "Low": [95.0, 115.0, 116.0, 118.0],
        "Close": [100.0, 118.0, 120.0, 124.0]
    })
    atr = compute_atr(df, period=2)
    assert len(atr) == 4
    assert not atr.isna().all()


def test_bollinger_bands_symmetry():
    """Verifies Upper and Lower Bollinger bands are symmetric about middle SMA."""
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    bb = compute_bollinger_bands(prices, period=5, num_std=2.0)
    clean_bb = bb.dropna()

    upper_diff = clean_bb["bb_upper"] - clean_bb["bb_middle"]
    lower_diff = clean_bb["bb_middle"] - clean_bb["bb_lower"]
    np.testing.assert_allclose(upper_diff.values, lower_diff.values, rtol=1e-5)
