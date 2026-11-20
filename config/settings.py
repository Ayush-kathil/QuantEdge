"""Configuration settings and instrument catalog for QuantEdge v3."""
from typing import Dict, Any, List
from config.instruments import (
    INSTRUMENT_REGISTRY,
    InstrumentMetadata,
    AvailabilityStatus,
    get_all_families,
    get_instruments_by_family,
    get_available_online_instruments,
    get_instrument_by_symbol_or_key
)

APP_VERSION = "3.0.0"
PLATFORM_NAME = "QuantEdge v3 – Institutional Quantitative Research & Financial Market Analytics Platform"

# Build unified INSTRUMENT_CATALOG by merging default metadata with INSTRUMENT_REGISTRY
INSTRUMENT_CATALOG: Dict[str, Dict[str, Any]] = {}

for key, meta in INSTRUMENT_REGISTRY.items():
    effective_symbol = meta.symbol if meta.symbol else key
    INSTRUMENT_CATALOG[effective_symbol] = {
        "name": meta.name,
        "asset_class": meta.asset_class,
        "instrument_type": meta.asset_class,
        "exchange": meta.exchange,
        "index_family": meta.index_family,
        "currency": meta.currency,
        "data_provider": meta.data_provider,
        "availability": meta.availability.value,
        "description": meta.notes,
        "corporate_actions": "Dividends and splits adjusted in Close series where applicable." if meta.asset_class in ["ETF", "Index", "Equity"] else "Not applicable to derivative contracts.",
        "futures_rollover_notes": meta.notes if "Futures" in meta.asset_class else "Not applicable.",
        "contract_multiplier": meta.contract_multiplier,
        "tick_size": meta.tick_size,
        "initial_margin_pct": meta.initial_margin_pct,
        "maintenance_margin_pct": meta.maintenance_margin_pct,
        "etf_proxy": meta.etf_proxy
    }

# Ensure single stock equities also remain accessible
INSTRUMENT_CATALOG["AAPL"] = {
    "name": "Apple Inc.",
    "asset_class": "Single Stock Equity",
    "instrument_type": "Cash Equity (Common Stock)",
    "exchange": "NASDAQ",
    "index_family": "Global Equities",
    "currency": "USD",
    "data_provider": "Yahoo Finance API",
    "availability": AvailabilityStatus.AVAILABLE_ONLINE.value,
    "description": "Large-cap technology equity.",
    "corporate_actions": "Historical prices reflect major stock splits (e.g., 4:1 in 2020, 7:1 in 2014).",
    "futures_rollover_notes": "Not applicable (Common stock equity).",
    "contract_multiplier": 1.0,
    "tick_size": 0.01,
    "initial_margin_pct": 0.25,
    "maintenance_margin_pct": 0.25,
    "etf_proxy": None
}

INSTRUMENT_CATALOG["MSFT"] = {
    "name": "Microsoft Corp.",
    "asset_class": "Single Stock Equity",
    "instrument_type": "Cash Equity (Common Stock)",
    "exchange": "NASDAQ",
    "index_family": "Global Equities",
    "currency": "USD",
    "data_provider": "Yahoo Finance API",
    "availability": AvailabilityStatus.AVAILABLE_ONLINE.value,
    "description": "Large-cap enterprise software & cloud equity.",
    "corporate_actions": "Historical prices split-adjusted.",
    "futures_rollover_notes": "Not applicable (Common stock equity).",
    "contract_multiplier": 1.0,
    "tick_size": 0.01,
    "initial_margin_pct": 0.25,
    "maintenance_margin_pct": 0.25,
    "etf_proxy": None
}

# Market Conventions & Constants
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE_USD = 0.04   # 4.0% annualized risk-free rate proxy (SOFR / 3M T-Bill)
DEFAULT_RISK_FREE_RATE_INR = 0.065  # 6.5% annualized risk-free rate proxy (RBI 91-day T-Bill)
DEFAULT_RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE_USD

# Backtest Defaults
DEFAULT_INITIAL_CAPITAL = 100_000.0  # Base capital (USD or INR depending on asset)
DEFAULT_COMMISSION_BPS = 5.0          # 5 bps = 0.05% of trade value
DEFAULT_SLIPPAGE_BPS = 2.0            # 2 bps = 0.02% price slippage

# Cache & Storage
CACHE_DIR = ".quantedge_cache"
CACHE_EXPIRY_HOURS = 24
DEFAULT_CACHE_EXPIRY_HOURS = 24
MIN_OBSERVATIONS_FOR_METRICS = 4

# UI Institutional Palette Constants
UI_THEME_DARK = {
    "bg_primary": "#0B111E",
    "bg_card": "#131D31",
    "bg_card_secondary": "#1A263D",
    "accent_blue": "#3B82F6",
    "accent_cyan": "#06B6D4",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "success_green": "#10B981",
    "danger_red": "#EF4444",
    "border_color": "#1E2E4A"
}

UI_THEME_LIGHT = {
    "bg_primary": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "bg_card_secondary": "#F1F5F9",
    "accent_blue": "#2563EB",
    "accent_cyan": "#0891B2",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "success_green": "#059669",
    "danger_red": "#DC2626",
    "border_color": "#E2E8F0"
}
