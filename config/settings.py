"""Configuration settings for QuantEdge."""
from typing import Dict, Any

# Supported asset classes and standard benchmark instruments
INSTRUMENT_CATALOG: Dict[str, Dict[str, str]] = {
    "^GSPC": {
        "name": "S&P 500 Index",
        "asset_class": "Equity Index",
        "exchange": "CBOE/NYSE",
        "description": "Large-cap US benchmark index"
    },
    "^NDX": {
        "name": "NASDAQ-100",
        "asset_class": "Equity Index",
        "exchange": "NASDAQ",
        "description": "Top 100 non-financial tech and growth equities"
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF",
        "asset_class": "Equity ETF",
        "exchange": "NYSE Arca",
        "description": "High-liquidity physical proxy for S&P 500"
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "asset_class": "Equity ETF",
        "exchange": "NASDAQ",
        "description": "High-liquidity physical proxy for NASDAQ-100"
    },
    "GC=F": {
        "name": "Gold Futures",
        "asset_class": "Commodities",
        "exchange": "COMEX",
        "description": "Front-month gold futures contract"
    },
    "CL=F": {
        "name": "Crude Oil Futures",
        "asset_class": "Commodities",
        "exchange": "NYMEX",
        "description": "Front-month WTI crude oil futures contract"
    },
    "ZN=F": {
        "name": "10-Year US Treasury Note Futures",
        "asset_class": "Fixed Income",
        "exchange": "CBOT",
        "description": "Benchmark US 10-Year Treasury note futures"
    },
    "AAPL": {
        "name": "Apple Inc.",
        "asset_class": "Single Stock Equity",
        "exchange": "NASDAQ",
        "description": "Consumer hardware, services, software"
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "asset_class": "Single Stock Equity",
        "exchange": "NASDAQ",
        "description": "Enterprise cloud, software, AI"
    }
}

# Market Conventions & Constants
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.04  # 4.0% annualized risk-free rate proxy (SOFR / 3M T-Bill)

# Backtest Defaults
DEFAULT_INITIAL_CAPITAL = 100_000.0  # $100,000 USD base capital
DEFAULT_COMMISSION_BPS = 5.0          # 5 bps = 0.05% of trade value
DEFAULT_SLIPPAGE_BPS = 2.0            # 2 bps = 0.02% price slippage

# Cache & Ingestion
CACHE_DIR = ".quantedge_cache"
DEFAULT_CACHE_EXPIRY_HOURS = 24

# Data Validation Thresholds
MAX_CONSECUTIVE_MISSING_DAYS = 5
MIN_OBSERVATIONS_FOR_METRICS = 30
