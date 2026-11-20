"""Unit tests verifying backtesting engine, execution mechanics, and look-ahead bias prevention."""
import pytest
import numpy as np
import pandas as pd

from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIMeanReversion
from strategies.breakout_strategy import DonchianBreakout
from backtesting.engine import BacktestEngine
from backtesting.execution import ExecutionModel


@pytest.fixture
def synthetic_trending_data() -> pd.DataFrame:
    """Creates a deterministic 60-day trending time series."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    # First 30 days flat/downtrend, next 30 days strong uptrend
    prices = np.concatenate([
        np.linspace(100.0, 90.0, 30),
        np.linspace(91.0, 150.0, 30)
    ])
    return pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": np.full(60, 500_000.0)
    }, index=dates)


def test_look_ahead_bias_prevention(synthetic_trending_data):
    """
    Look-ahead verification:
    Modifying the price of bar t+1 must NOT change the signal at bar t.
    """
    strat = MovingAverageCrossover(short_period=5, long_period=15)
    
    # Run baseline signals
    base_signals = strat.generate_signals(synthetic_trending_data)
    t = 25  # Test bar index
    sig_at_t = base_signals["signal"].iloc[t]

    # Tamper with future price at t+1
    tampered_data = synthetic_trending_data.copy()
    tampered_data.iloc[t + 1, tampered_data.columns.get_loc("Close")] = 999.0  # Massive spike
    tampered_signals = strat.generate_signals(tampered_data)

    # Signal at t must be completely unchanged
    assert tampered_signals["signal"].iloc[t] == sig_at_t


def test_breakout_strategy_no_lookahead(synthetic_trending_data):
    """
    Verifies DonchianBreakout shifts channels by 1 so the current bar cannot breakout
    against a channel that includes its own extreme high.
    """
    strat = DonchianBreakout(lookback_window=10, exit_lookback_window=5)
    signals = strat.generate_signals(synthetic_trending_data)
    
    # Upper channel at index i must equal max(High[i-10 : i])
    i = 20
    expected_upper = synthetic_trending_data["High"].iloc[i-10:i].max()
    assert pytest.approx(signals["breakout_upper"].iloc[i]) == expected_upper


def test_execution_timing_next_open(synthetic_trending_data):
    """
    Verifies that a signal produced at bar t executes on the Open of bar t+1.
    """
    strat = MovingAverageCrossover(short_period=5, long_period=10)
    engine = BacktestEngine(
        initial_capital=100_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        execution_mode="next_open"
    )
    result = engine.run(synthetic_trending_data, strat, symbol="TREND_ASSET")

    # If trade occurred, entry price must match Open price of trade entry date
    if result.trades:
        first_trade = result.trades[0]
        entry_dt = pd.to_datetime(first_trade.entry_date)
        actual_open_price = synthetic_trending_data.loc[entry_dt, "Open"]
        assert pytest.approx(first_trade.entry_price) == actual_open_price


def test_transaction_costs_and_slippage():
    """Verifies that slippage widens fill price and commission reduces net proceeds."""
    model = ExecutionModel(commission_bps=10.0, slippage_bps=5.0)

    # Slippage test: Buy at 100 with 5 bps slippage -> 100 * (1 + 0.0005) = 100.05
    buy_price = model.calculate_execution_price(100.0, "BUY")
    assert pytest.approx(buy_price) == 100.05

    # Sell at 100 with 5 bps slippage -> 100 * (1 - 0.0005) = 99.95
    sell_price = model.calculate_execution_price(100.0, "SELL")
    assert pytest.approx(sell_price) == 99.95

    # Commission test: 10 bps on $10,000 = $10.00
    comm = model.calculate_commission(10_000.0)
    assert pytest.approx(comm) == 10.00


def test_backtest_cash_and_equity_conservation(synthetic_trending_data):
    """Verifies mark-to-market equity equals Cash + Shares * Price across all periods."""
    strat = MovingAverageCrossover(short_period=5, long_period=15)
    engine = BacktestEngine(initial_capital=50_000.0, commission_bps=5.0, slippage_bps=2.0)
    result = engine.run(synthetic_trending_data, strat, symbol="ASSET")

    hist = result.detailed_history
    # For every bar where position is long, Equity should equal Cash + Shares * Close
    long_bars = hist[hist["Position"] == 1]
    if not long_bars.empty:
        calc_equity = long_bars["Cash"] + (long_bars["Shares"] * long_bars["Close"])
        np.testing.assert_allclose(long_bars["Strategy_Equity"].values, calc_equity.values, rtol=1e-5)


def test_rsi_strategy_execution(synthetic_trending_data):
    """Verifies RSI mean reversion executes without runtime errors."""
    strat = RSIMeanReversion(rsi_period=10, oversold=35.0, overbought=65.0)
    engine = BacktestEngine(initial_capital=100_000.0)
    result = engine.run(synthetic_trending_data, strat, symbol="RSI_ASSET")

    assert result.metrics is not None
    assert len(result.equity_curve) == len(synthetic_trending_data)
