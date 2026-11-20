"""Unit tests for market data ingestion, validation, and hygiene."""
import io
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.validation import validate_market_data, ValidationResult
from data.ingestion import MarketDataIngestion


@pytest.fixture
def clean_synthetic_data() -> pd.DataFrame:
    """Generates a clean, deterministic 50-day OHLCV fixture."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="B")
    base_price = 100.0
    
    # Deterministic price path
    trend = np.linspace(0, 10, 50)
    cycle = 5.0 * np.sin(np.linspace(0, 3 * np.pi, 50))
    close = base_price + trend + cycle
    
    open_p = close - 0.5
    high_p = close + 1.5
    low_p = close - 1.5
    vol = np.full(50, 1_000_000.0)

    df = pd.DataFrame({
        "Open": open_p,
        "High": high_p,
        "Low": low_p,
        "Close": close,
        "Volume": vol
    }, index=dates)
    return df


def test_validation_clean_data(clean_synthetic_data):
    """Clean data should pass validation with zero errors."""
    res = validate_market_data(clean_synthetic_data, strict_ohlc=True)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.cleaned_df is not None
    assert len(res.cleaned_df) == 50


def test_validation_missing_columns():
    """Missing required columns must be caught and invalidate the data."""
    df = pd.DataFrame({"Price": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2))
    res = validate_market_data(df)
    assert res.is_valid is False
    assert any("Missing mandatory OHLCV columns" in err for err in res.errors)


def test_validation_inverted_high_low(clean_synthetic_data):
    """High < Low must be flagged as a structural violation."""
    corrupted = clean_synthetic_data.copy()
    # Invert High and Low on row 5
    corrupted.iloc[5, corrupted.columns.get_loc("High")] = 80.0
    corrupted.iloc[5, corrupted.columns.get_loc("Low")] = 120.0

    res = validate_market_data(corrupted, strict_ohlc=True)
    assert res.is_valid is False
    assert any("OHLC consistency violations" in err for err in res.errors)


def test_validation_non_positive_prices(clean_synthetic_data):
    """Zero or negative prices must be flagged."""
    corrupted = clean_synthetic_data.copy()
    corrupted.iloc[10, corrupted.columns.get_loc("Close")] = -5.0

    res = validate_market_data(corrupted, strict_ohlc=True)
    assert res.is_valid is False
    assert any("non-positive prices" in err for err in res.errors)


def test_validation_negative_volume(clean_synthetic_data):
    """Negative volume should produce a warning and be auto-cleaned to 0."""
    corrupted = clean_synthetic_data.copy()
    corrupted.iloc[3, corrupted.columns.get_loc("Volume")] = -500.0

    res = validate_market_data(corrupted, auto_clean=True)
    assert res.is_valid is True
    assert any("negative volume" in w for w in res.warnings)
    assert res.cleaned_df.iloc[3]["Volume"] == 0.0


def test_validation_duplicate_timestamps(clean_synthetic_data):
    """Duplicate dates must be detected and deduplicated."""
    corrupted = clean_synthetic_data.copy()
    # Duplicate row 0
    corrupted = pd.concat([corrupted.iloc[:1], corrupted])
    assert len(corrupted) == 51

    res = validate_market_data(corrupted, auto_clean=True)
    assert res.is_valid is True
    assert any("duplicate timestamps" in w for w in res.warnings)
    assert len(res.cleaned_df) == 50


def test_validation_unsorted_index(clean_synthetic_data):
    """Unsorted timestamps must be ordered in ascending sequence."""
    corrupted = clean_synthetic_data.iloc[::-1].copy()  # Reverse order
    assert not corrupted.index.is_monotonic_increasing

    res = validate_market_data(corrupted, auto_clean=True)
    assert res.is_valid is True
    assert res.cleaned_df.index.is_monotonic_increasing


def test_csv_ingestion_parsing():
    """CSV ingestion handles varied casings and date formats."""
    csv_text = """Date,open,HIGH,Low,Close,volume
2024-01-02,150.0,152.0,149.0,151.5,100000
2024-01-03,151.5,153.0,150.0,152.0,120000
2024-01-04,152.0,155.0,151.0,154.5,110000
"""
    ing = MarketDataIngestion(cache_enabled=False)
    payload = ing.load_csv(io.StringIO(csv_text), ticker_name="TEST_CSV")
    assert payload.ticker == "TEST_CSV"
    assert payload.row_count == 3
    assert list(payload.df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert payload.validation.is_valid is True
