"""
Tests for Central Instrument Registry and Comprehensive Indian Market Coverage.
"""
import pytest
from config.instruments import (
    INSTRUMENT_REGISTRY,
    AvailabilityStatus,
    get_all_families,
    get_instruments_by_family,
    get_available_online_instruments,
    get_instrument_by_symbol_or_key
)


def test_registry_has_all_required_categories():
    families = get_all_families()
    required = [
        "NSE Broad Market",
        "NSE Sectoral",
        "NSE Thematic",
        "NSE Strategy",
        "BSE Indices",
        "Indian Indicators",
        "Fixed Income",
        "Hybrid Indices",
        "Global Markets",
        "Global Commodities"
    ]
    for req in required:
        assert req in families, f"Missing required market family: {req}"


def test_key_nse_indices_coverage():
    broad = get_instruments_by_family("NSE Broad Market")
    names = [meta.name for meta in broad]
    assert "NIFTY 50" in names
    assert "NIFTY NEXT 50" in names
    assert "NIFTY 100" in names
    assert "NIFTY 200" in names
    assert "NIFTY 500" in names
    assert "NIFTY MIDCAP 50" in names
    assert "NIFTY SMALLCAP 100" in names


def test_key_sectoral_indices_coverage():
    sectoral = get_instruments_by_family("NSE Sectoral")
    names = [meta.name for meta in sectoral]
    assert "NIFTY BANK" in names
    assert "NIFTY IT" in names
    assert "NIFTY AUTO" in names
    assert "NIFTY FMCG" in names
    assert "NIFTY PHARMA" in names
    assert "NIFTY METAL" in names
    assert "NIFTY REALTY" in names
    assert "NIFTY ENERGY" in names
    assert "NIFTY PSU BANK" in names


def test_bse_indices_coverage():
    bse = get_instruments_by_family("BSE Indices")
    names = [meta.name for meta in bse]
    assert "BSE SENSEX" in names
    assert "BSE 100" in names
    assert "BSE 200" in names
    assert "BSE 500" in names
    assert "BSE BANKEX" in names


def test_indian_indicators_coverage():
    indicators = get_instruments_by_family("Indian Indicators")
    names = [meta.name for meta in indicators]
    assert "INDIA VIX (Volatility Index)" in names
    assert "USD/INR Spot / Currency Rate" in names
    assert "NIFTY 50 ETF (NIFTYBEES)" in names


def test_unsupported_tickers_are_strictly_flagged():
    """Confirms instruments without free yfinance tickers are strictly flagged for CSV upload."""
    nifty_1000 = get_instrument_by_symbol_or_key("NIFTY_1000")
    assert nifty_1000 is not None
    assert nifty_1000.availability == AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV
    assert nifty_1000.symbol is None

    defence = get_instrument_by_symbol_or_key("NIFTY_DEFENCE")
    assert defence is not None
    assert defence.availability == AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV


def test_available_online_instruments_have_valid_symbols():
    available = get_available_online_instruments()
    assert len(available) >= 15
    for item in available:
        assert item.symbol is not None
        assert len(item.symbol) > 0
