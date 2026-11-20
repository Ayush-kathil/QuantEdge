"""Configuration settings and instrument catalog for QuantEdge."""
from typing import Dict, Any

APP_VERSION = "2.1.0"
PLATFORM_NAME = "QuantEdge – Quantitative Market Research & Strategy Backtesting Platform"

# Supported asset classes and standard benchmark instruments
INSTRUMENT_CATALOG: Dict[str, Dict[str, Any]] = {
    "^GSPC": {
        "name": "S&P 500 Index",
        "asset_class": "Equity Index",
        "instrument_type": "Cash Index (Non-Tradable Benchmark)",
        "exchange": "CBOE/NYSE",
        "description": "Large-cap US benchmark index. Unmanaged cash index; does not reflect dividend reinvestment or expense drag.",
        "corporate_actions": "Excludes cash dividends. For total return with dividends, refer to SPTR or use ETF proxies (SPY).",
        "futures_rollover_notes": "Not applicable (Cash index)."
    },
    "^NDX": {
        "name": "NASDAQ-100",
        "asset_class": "Equity Index",
        "instrument_type": "Cash Index (Non-Tradable Benchmark)",
        "exchange": "NASDAQ",
        "description": "Top 100 non-financial tech and growth equities.",
        "corporate_actions": "Excludes cash dividends.",
        "futures_rollover_notes": "Not applicable (Cash index)."
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "Equity ETF",
        "instrument_type": "Exchange-Traded Fund (Cash Equity)",
        "exchange": "NYSE Arca",
        "description": "High-liquidity physical proxy for S&P 500.",
        "corporate_actions": "Historical 'Close' is split-adjusted. 'Adj Close' accounts for cash dividend distributions.",
        "futures_rollover_notes": "Not applicable (Cash ETF, no futures rollover distortion)."
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "asset_class": "Equity ETF",
        "instrument_type": "Exchange-Traded Fund (Cash Equity)",
        "exchange": "NASDAQ",
        "description": "High-liquidity physical proxy for NASDAQ-100.",
        "corporate_actions": "Historical 'Close' is split-adjusted. 'Adj Close' accounts for cash dividend distributions.",
        "futures_rollover_notes": "Not applicable (Cash ETF, no futures rollover distortion)."
    },
    "GC=F": {
        "name": "Gold Futures (Front-Month Continuous)",
        "asset_class": "Commodities",
        "instrument_type": "Synthetic Continuous Futures Contract",
        "exchange": "COMEX",
        "description": "Front-month gold futures contract concatenated by data provider.",
        "corporate_actions": "No dividends or stock splits apply to futures.",
        "futures_rollover_notes": "CRITICAL: Uses front-month continuous contract stitching. Prices exhibit artificial step jumps upon monthly contract expiration and rollover. Does NOT reflect physical delivery costs or cumulative roll yield (contango/backwardation drag)."
    },
    "CL=F": {
        "name": "Crude Oil Futures (Front-Month Continuous)",
        "asset_class": "Commodities",
        "instrument_type": "Synthetic Continuous Futures Contract",
        "exchange": "NYMEX",
        "description": "Front-month WTI crude oil futures contract.",
        "corporate_actions": "No corporate actions apply.",
        "futures_rollover_notes": "CRITICAL: Front-month continuous stitching subject to significant roll yield leakage. Historical price series contains unadjusted contract transitions; real futures performance requires pan-contract calendar roll modeling."
    },
    "ZN=F": {
        "name": "10-Year US Treasury Note Futures",
        "asset_class": "Fixed Income",
        "instrument_type": "Synthetic Continuous Futures Contract",
        "exchange": "CBOT",
        "description": "Benchmark US 10-Year Treasury note futures.",
        "corporate_actions": "No dividends. Bond coupon cash flows reflected in futures implied repo and basis pricing.",
        "futures_rollover_notes": "Quarterly contract rollover jumps (H, M, U, Z cycles). Historical returns may diverge from physical bond holdings."
    },
    "AAPL": {
        "name": "Apple Inc.",
        "asset_class": "Single Stock Equity",
        "instrument_type": "Cash Equity (Common Stock)",
        "exchange": "NASDAQ",
        "description": "Large-cap technology equity.",
        "corporate_actions": "Historical prices reflect major stock splits (e.g., 4:1 in 2020, 7:1 in 2014). 'Adj Close' accounts for regular quarterly dividends.",
        "futures_rollover_notes": "Not applicable (Common stock equity)."
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "asset_class": "Single Stock Equity",
        "instrument_type": "Cash Equity (Common Stock)",
        "exchange": "NASDAQ",
        "description": "Large-cap enterprise software & cloud equity.",
        "corporate_actions": "Historical prices split-adjusted. 'Adj Close' accounts for quarterly dividends.",
        "futures_rollover_notes": "Not applicable (Common stock equity)."
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
