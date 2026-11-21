"""
Tests for Futures Accounting, Margin Modeling, and Mark-to-Market Simulation.
"""
import pytest
import numpy as np
import pandas as pd
from analytics.futures_accounting import (
    get_futures_spec,
    get_historical_lot_size,
    simulate_futures_account,
    FUTURES_SPECS_REGISTRY,
    FuturesContractSpec
)


@pytest.fixture
def sample_futures_df() -> pd.DataFrame:
    """10-day price path with known entry, rise, drop, and exit."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    # Day 0: 100, Day 1: 102, Day 2: 105, Day 3: 108, Day 4: 104, Day 5: 100, Day 6: 95, etc.
    opens = [100.0, 101.0, 103.0, 106.0, 107.0, 103.0, 99.0, 94.0, 96.0, 98.0]
    closes = [101.0, 103.0, 106.0, 107.0, 104.0, 99.0, 95.0, 96.0, 98.0, 100.0]
    highs = [max(o, c) + 1.0 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 1.0 for o, c in zip(opens, closes)]
    vols = [1000.0] * 10
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols
    }, index=dates)


def test_futures_specs_integrity():
    spec = get_futures_spec("GC=F")
    assert spec.multiplier == 100.0
    assert spec.tick_size == 0.10
    assert spec.tick_value == pytest.approx(10.0, abs=1e-6)

    nifty_spec = get_futures_spec("NIFTY_FUT")
    # Current active SEBI lot size is 75 (effective Nov 20, 2024)
    assert nifty_spec.multiplier == 75.0
    assert nifty_spec.tick_size == 0.05
    assert nifty_spec.tick_value == pytest.approx(3.75, abs=1e-6)

    # Dynamic historical lot size resolution across regulatory revisions
    assert get_historical_lot_size("NIFTY_FUT", "2024-11-25") == 75.0
    assert get_historical_lot_size("NIFTY_FUT", "2024-05-01") == 25.0
    assert get_historical_lot_size("NIFTY_FUT", "2023-01-01") == 50.0
    assert get_historical_lot_size("NIFTY_FUT", "2018-06-01") == 75.0
    assert get_historical_lot_size("BANKNIFTY_FUT", "2024-11-25") == 30.0
    assert get_historical_lot_size("BANKNIFTY_FUT", "2023-12-01") == 15.0


def test_futures_simulation_long_position(sample_futures_df):
    spec = get_futures_spec("GC=F")
    # Enter long on day 1 (signal generated on day 0)
    signals = pd.Series([1] * 10, index=sample_futures_df.index)
    res = simulate_futures_account(
        sample_futures_df,
        signals,
        spec,
        initial_cash=50_000.0,
        max_contracts=1,
        slippage_ticks=0.0
    )
    assert res.initial_cash == 50_000.0
    assert len(res.daily_equity_curve) == 10
    assert len(res.daily_variation_margin) == 10
    assert "A_etf_return" in res.structural_comparison
    assert "C_actual_futures_pnl" in res.structural_comparison


def test_margin_call_trigger_on_severe_adverse_move():
    # Rapid crash from 100 to 20 on high leverage
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    opens = [100.0, 90.0, 70.0, 40.0, 20.0]
    closes = [95.0, 80.0, 50.0, 25.0, 15.0]
    crash_df = pd.DataFrame({
        "Open": opens, "High": opens, "Low": closes, "Close": closes, "Volume": [100] * 5
    }, index=dates)

    # NIFTY spec with lot size 25: 100 -> 15 is 85 points drop * 25 = 2,125 loss per contract
    # If initial cash is small (e.g. 500), cash will drop below maintenance margin
    spec = FuturesContractSpec.create(
        symbol="TEST", name="Test Spec", exchange="EX", multiplier=25.0, tick_size=0.05,
        currency="INR", initial_margin_rate=0.20, maintenance_margin_rate=0.15,
        transaction_cost_per_contract=0.0, rollover_cycle="Monthly", stitching_caveats="Test"
    )
    signals = pd.Series([1] * 5, index=dates)
    res = simulate_futures_account(crash_df, signals, spec, initial_cash=1000.0, max_contracts=1, slippage_ticks=0)
    assert res.margin_calls_triggered > 0
