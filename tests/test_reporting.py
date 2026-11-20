"""Unit tests verifying PDF report compilation and reproducibility metadata."""
import pytest
import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from strategies.moving_average import MovingAverageCrossover
from analytics.statistics import compute_distribution_statistics
from analytics.reporting import generate_pdf_research_report
from analytics.reproducibility import generate_experiment_manifest


def test_pdf_report_generation():
    """Verifies that generate_pdf_research_report generates a valid binary PDF document."""
    dates = pd.date_range("2024-01-01", periods=60)
    prices = np.linspace(100.0, 120.0, 60)
    df = pd.DataFrame({
        "Open": prices - 0.5,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": np.full(60, 100_000.0)
    }, index=dates)

    strat = MovingAverageCrossover(5, 15)
    engine = BacktestEngine(initial_capital=100_000.0)
    res = engine.run(df, strat, symbol="PDF_TEST")

    returns = df["Close"].pct_change().dropna()
    dist_stats = compute_distribution_statistics(returns)

    pdf_bytes = generate_pdf_research_report(
        backtest_result=res,
        dist_stats=dist_stats,
        dataset_fingerprint="test_sha256_hash_1234567890",
        data_source="Test Mock Feed",
        instrument_type="Cash Equity Proxy",
        corporate_action_notes="Split-adjusted series.",
        futures_rollover_notes="N/A",
        experiment_id="TEST-EXP-001"
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Standard PDF header check: all valid PDFs start with %PDF
    assert pdf_bytes.startswith(b"%PDF")


def test_experiment_metadata_manifest():
    """Verifies experiment manifest generation and JSON serialization."""
    manifest = generate_experiment_manifest(
        symbol="SPY",
        dataset_fingerprint="abc123sha",
        data_source="Yahoo Finance",
        date_start="2023-01-01",
        date_end="2024-01-01",
        row_count=250,
        strategy_name="MA Crossover",
        strategy_params={"short": 20, "long": 50},
        initial_capital=100_000.0,
        commission_bps=5.0,
        slippage_bps=2.0,
        execution_mode="next_open",
        risk_free_rate=0.04,
        trading_days=252,
        validation_type="Walk-Forward 4-Fold",
        metrics_dict={"Sharpe": 1.45, "CAGR": 0.18}
    )

    json_str = manifest.to_json()
    assert "QEDGE-EXP-" in json_str
    assert "abc123sha" in json_str
    assert manifest.platform_version is not None
