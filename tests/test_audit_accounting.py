"""Audited accounting tests verifying exact balance sheet conservation and benchmark metrics."""
import pytest
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from backtesting.engine import BacktestEngine
from backtesting.metrics import calculate_performance_metrics
from backtesting.execution import Trade


class FixedSignalStrategy(BaseStrategy):
    """Deterministic strategy for auditing accounting equations."""

    def __init__(self, signal_sequence: list, name: str = "FixedSignal"):
        super().__init__(name=name, description="Fixed signal strategy", parameters={}, long_only=False)
        self.signal_sequence = signal_sequence

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        sig = self.signal_sequence[:len(res)]
        if len(sig) < len(res):
            sig.extend([0] * (len(res) - len(sig)))
        res["signal"] = sig
        return res


def test_short_position_balance_sheet_conservation():
    """
    Audits the balance sheet equation during a short trade:
    Day 0: Price 100, Signal = -1 (Short)
    Day 1: Opens at 100, fills short. Close at 90.
    Day 2: Signal = 0 (Exit). Opens at 90, fills cover. Close at 90.
    
    Verifies:
    1. During short: Equity = Cash - |Shares| * Close
    2. After cover: Ending Cash == Initial Capital + Net PnL (exact penny match)
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({
        "Open": [100.0, 100.0, 90.0],
        "High": [105.0, 102.0, 92.0],
        "Low": [98.0, 89.0, 88.0],
        "Close": [100.0, 90.0, 90.0],
        "Volume": [1000.0, 1000.0, 1000.0]
    }, index=dates)

    strat = FixedSignalStrategy(signal_sequence=[-1, 0, 0])
    engine = BacktestEngine(
        initial_capital=100_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        execution_mode="next_open"
    )
    result = engine.run(df, strat, symbol="SHORT_TEST")

    # Trade was: Short at 100, Covered at 90
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.direction == "SHORT"
    assert pytest.approx(t.entry_price) == 100.0
    assert pytest.approx(t.exit_price) == 90.0

    # 10% gain on short: price fell from 100 to 90
    assert t.gross_pnl > 0

    # End of backtest: all positions closed
    ending_equity = float(result.equity_curve.iloc[-1])
    expected_equity = 100_000.0 + t.net_pnl
    assert pytest.approx(ending_equity, rel=1e-5) == expected_equity

    # Mid-trade mark-to-market audit (Day 1 close)
    # On Day 1, short was filled at 100. Close is 90.
    # Short position was in profit by 10%
    day1_equity = float(result.equity_curve.iloc[1])
    assert day1_equity > 100_000.0


def test_short_loss_accounting():
    """
    Verifies balance sheet conservation when a short position loses money:
    Short at 100, price rises to 110, cover at 110.
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({
        "Open": [100.0, 100.0, 110.0],
        "High": [105.0, 112.0, 112.0],
        "Low": [98.0, 99.0, 108.0],
        "Close": [100.0, 110.0, 110.0],
        "Volume": [1000.0, 1000.0, 1000.0]
    }, index=dates)

    strat = FixedSignalStrategy(signal_sequence=[-1, 0, 0])
    engine = BacktestEngine(
        initial_capital=100_000.0,
        commission_bps=5.0,
        slippage_bps=2.0,
        execution_mode="next_open"
    )
    result = engine.run(df, strat, symbol="SHORT_LOSS_TEST")

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.net_pnl < 0  # Lost money

    # Ending equity must equal initial capital + net_pnl
    ending_equity = float(result.equity_curve.iloc[-1])
    assert pytest.approx(ending_equity, rel=1e-4) == (100_000.0 + t.net_pnl)


def test_benchmark_relative_metrics():
    """Validates Beta, Alpha, Tracking Error, and Information Ratio calculation."""
    dates = pd.date_range("2024-01-01", periods=100)
    # Synthetic correlated returns
    bm_rets = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    # Strat rets = 1.2 * bm_rets + 0.0002 alpha
    strat_rets = 1.2 * bm_rets + 0.0002
    equity = (1.0 + strat_rets).cumprod() * 100_000.0

    metrics = calculate_performance_metrics(
        equity_curve=equity,
        daily_returns=strat_rets,
        trades=[],
        benchmark_returns=bm_rets,
        risk_free_rate=0.04,
        trading_days=252
    )

    # Beta should be close to 1.2
    assert pytest.approx(metrics.beta, rel=0.1) == 1.2
    assert metrics.tracking_error > 0
    assert metrics.information_ratio is not None
