"""
Central Instrument Registry and Taxonomy for QuantEdge v3.1.

Provides comprehensive coverage of:
1. NSE Broad Market Indices
2. NSE Sectoral Indices
3. NSE Thematic Indices
4. NSE Strategy (Smart Beta) Indices
5. BSE Broad Market Indices
6. BSE Sectoral Indices
7. BSE Thematic Indices
8. Indian ETFs
9. Indian Derivatives (Futures & Currency Proxies)
10. Indian Fixed Income (G-Sec, SDL, Corporate Bond)
11. Indian Hybrid Indices
12. Indian Market Volatility Indicators (India VIX)
13. Global Markets & Commodities

Maintains strict provenance:
- Official Instrument Name, Exchange, Category, Symbol, Currency,
  Data Provider, Availability Status, ETF Proxy, and Detailed Notes.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class AvailabilityStatus(str, Enum):
    AVAILABLE_ONLINE = "AVAILABLE_ONLINE"
    UNAVAILABLE_YFINANCE_REQUIRES_CSV = "UNAVAILABLE_YFINANCE_REQUIRES_CSV"
    BENCHMARK_REFERENCE = "BENCHMARK_REFERENCE"


@dataclass
class InstrumentMetadata:
    """Standardized metadata representation for every financial instrument."""
    name: str
    symbol: Optional[str]
    exchange: str
    asset_class: str
    index_family: str
    currency: str
    data_provider: str
    availability: AvailabilityStatus
    notes: str
    etf_proxy: Optional[str] = None
    contract_multiplier: float = 1.0
    tick_size: float = 0.05
    initial_margin_pct: float = 0.15
    maintenance_margin_pct: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol or "CSV_REQUIRED",
            "exchange": self.exchange,
            "asset_class": self.asset_class,
            "index_family": self.index_family,
            "currency": self.currency,
            "data_provider": self.data_provider,
            "availability": self.availability.value,
            "notes": self.notes,
            "etf_proxy": self.etf_proxy or "N/A"
        }


# 12 Standardized Indian Market Categories as mandated by exchange taxonomies
INDIAN_MARKET_CATEGORIES: List[str] = [
    "NSE Broad Market",
    "NSE Sectoral",
    "NSE Thematic",
    "NSE Strategy",
    "BSE Broad Market",
    "BSE Sectoral",
    "BSE Thematic",
    "Indian ETFs",
    "Indian Derivatives",
    "Indian Fixed Income",
    "Indian Hybrid Indices",
    "Indian Market Volatility Indicators"
]


# Comprehensive Central Catalog
INSTRUMENT_REGISTRY: Dict[str, InstrumentMetadata] = {
    # -------------------------------------------------------------------------
    # 1. NSE BROAD MARKET INDICES
    # -------------------------------------------------------------------------
    "^NSEI": InstrumentMetadata(
        name="NIFTY 50",
        symbol="^NSEI",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Flagship benchmark index representing 50 large-cap Indian companies across 13 economic sectors.",
        etf_proxy="NIFTYBEES.NS",
        contract_multiplier=75.0,
        tick_size=0.05
    ),
    "NIFTY_NEXT_50": InstrumentMetadata(
        name="NIFTY NEXT 50",
        symbol="^NSMIDCP",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Represents 50 companies from NIFTY 100 after excluding NIFTY 50 constituents. (Proxy: JUNIORBEES.NS)",
        etf_proxy="JUNIORBEES.NS"
    ),
    "^CNX100": InstrumentMetadata(
        name="NIFTY 100",
        symbol="^CNX100",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Diversified top 100 Indian companies based on full market capitalization."
    ),
    "^CNX200": InstrumentMetadata(
        name="NIFTY 200",
        symbol="^CNX200",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Designed to measure the performance of top 200 companies (Large & Midcap universe)."
    ),
    "^CRSLDX": InstrumentMetadata(
        name="NIFTY 500",
        symbol="^CRSLDX",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Represents top 500 companies accounting for ~94% of total free-float market cap on NSE."
    ),
    "NIFTY_1000": InstrumentMetadata(
        name="NIFTY 1000",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Broad index tracking 1,000 liquid securities across market caps. Requires official NSE CSV upload."
    ),
    "NIFTY_TOTAL_MARKET": InstrumentMetadata(
        name="NIFTY TOTAL MARKET",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks top 750 stocks covering large, mid, small, and micro caps. Requires official CSV upload."
    ),
    "^NSEMDCP50": InstrumentMetadata(
        name="NIFTY MIDCAP 50",
        symbol="^NSEMDCP50",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Monitors top 50 mid-sized companies on NSE with active derivative trading."
    ),
    "NIFTY_MIDCAP_100": InstrumentMetadata(
        name="NIFTY MIDCAP 100",
        symbol="MID150BEES.NS",
        exchange="NSE",
        asset_class="ETF Proxy",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Captures movement of the midcap segment. Traded ETF proxy: MID150BEES.NS.",
        etf_proxy="MID150BEES.NS"
    ),
    "NIFTY_MIDCAP_150": InstrumentMetadata(
        name="NIFTY MIDCAP 150",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Official NSE Midcap 150 index. Ticker not available on free API; upload official CSV."
    ),
    "NIFTY_MIDCAP_SELECT": InstrumentMetadata(
        name="NIFTY MIDCAP SELECT",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks 25 liquid midcap stocks for derivatives contracts. Requires CSV upload."
    ),
    "NIFTY_SMALLCAP_50": InstrumentMetadata(
        name="NIFTY SMALLCAP 50",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Top 50 smallcap stocks by free float. Requires CSV upload."
    ),
    "^CNXSC": InstrumentMetadata(
        name="NIFTY SMALLCAP 100",
        symbol="^CNXSC",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Benchmark index for small-cap equities on NSE."
    ),
    "NIFTY_SMALLCAP_250": InstrumentMetadata(
        name="NIFTY SMALLCAP 250",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Reflects the smallcap universe (ranked 251-500). Requires CSV upload."
    ),
    "NIFTY_SMALLCAP_500": InstrumentMetadata(
        name="NIFTY SMALLCAP 500",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Official extended smallcap index. Requires CSV upload."
    ),
    "NIFTY_MICROCAP_250": InstrumentMetadata(
        name="NIFTY MICROCAP 250",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks 250 microcap stocks (ranked 501-750). Requires CSV upload."
    ),
    "NIFTY_LARGE_MIDCAP_250": InstrumentMetadata(
        name="NIFTY LARGE MIDCAP 250",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="50% Largecap (Nifty 100) + 50% Midcap (Nifty Midcap 150) composite. Requires CSV upload."
    ),
    "NIFTY_MIDSMALLCAP_400": InstrumentMetadata(
        name="NIFTY MIDSMALLCAP 400",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Consists of midcap 150 and smallcap 250. Requires CSV upload."
    ),
    "NIFTY_NEXT_100": InstrumentMetadata(
        name="NIFTY NEXT 100",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Stocks ranked 101 to 200 on NSE. Requires CSV upload."
    ),
    "NIFTY500_MULTICAP_50_25_25": InstrumentMetadata(
        name="NIFTY500 MULTICAP 50:25:25",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Fixed weight multicap index (50% Large, 25% Mid, 25% Small). Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 2. NSE SECTORAL INDICES
    # -------------------------------------------------------------------------
    "^NSEBANK": InstrumentMetadata(
        name="NIFTY BANK",
        symbol="^NSEBANK",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Tracks the 12 most liquid Indian banking stocks.",
        etf_proxy="BANKBEES.NS",
        contract_multiplier=30.0,
        tick_size=0.05
    ),
    "NIFTY_FIN_SERVICE": InstrumentMetadata(
        name="NIFTY FINANCIAL SERVICES",
        symbol="^CNXFIN",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Broader financial sector including banks, NBFCs, insurance, and housing finance."
    ),
    "^CNXIT": InstrumentMetadata(
        name="NIFTY IT",
        symbol="^CNXIT",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Tracks major Indian IT software and services exporters (TCS, Infosys, Wipro, etc.).",
        etf_proxy="ITBEES.NS"
    ),
    "^CNXAUTO": InstrumentMetadata(
        name="NIFTY AUTO",
        symbol="^CNXAUTO",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Reflects Indian automotive sector including 2/3 wheelers, passenger vehicles, and auto ancillaries.",
        etf_proxy="AUTOBEES.NS"
    ),
    "^CNXFMCG": InstrumentMetadata(
        name="NIFTY FMCG",
        symbol="^CNXFMCG",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Fast Moving Consumer Goods manufacturers (ITC, HUL, Nestlé, etc.)."
    ),
    "^CNXPHARMA": InstrumentMetadata(
        name="NIFTY PHARMA",
        symbol="^CNXPHARMA",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Indian pharmaceuticals, biotech, and generic drug manufacturers.",
        etf_proxy="PHARMABEES.NS"
    ),
    "^CNXMETAL": InstrumentMetadata(
        name="NIFTY METAL",
        symbol="^CNXMETAL",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Ferrous and non-ferrous metal producers (Tata Steel, JSW, Hindalco, etc.)."
    ),
    "^CNXREALTY": InstrumentMetadata(
        name="NIFTY REALTY",
        symbol="^CNXREALTY",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Real estate development and commercial property developers in India."
    ),
    "^CNXMEDIA": InstrumentMetadata(
        name="NIFTY MEDIA",
        symbol="^CNXMEDIA",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Print, broadcasting, entertainment, and digital media companies."
    ),
    "^CNXENERGY": InstrumentMetadata(
        name="NIFTY ENERGY",
        symbol="^CNXENERGY",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Petroleum, gas, power utilities, and renewable energy companies."
    ),
    "^CNXPSUBANK": InstrumentMetadata(
        name="NIFTY PSU BANK",
        symbol="^CNXPSUBANK",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Public Sector Undertaking (State-owned) commercial banks (SBI, PNB, BoB, etc.).",
        etf_proxy="PSUBNKBEES.NS"
    ),
    "NIFTY_PVT_BANK": InstrumentMetadata(
        name="NIFTY PRIVATE BANK",
        symbol="NIFTY_PVT_BANK.NS",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Privately held Indian banking institutions (HDFC, ICICI, Axis, Kotak)."
    ),
    "NIFTY_HEALTHCARE": InstrumentMetadata(
        name="NIFTY HEALTHCARE",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Hospitals, diagnostic chains, and medical device suppliers. Requires CSV upload."
    ),
    "NIFTY_OIL_GAS": InstrumentMetadata(
        name="NIFTY OIL & GAS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Upstream exploration, downstream refining, and city gas distributors. Requires CSV upload."
    ),
    "NIFTY_CONSUMER_DURABLES": InstrumentMetadata(
        name="NIFTY CONSUMER DURABLES",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Electronics, home appliances, and white goods manufacturers. Requires CSV upload."
    ),
    "NIFTY_CAPITAL_GOODS": InstrumentMetadata(
        name="NIFTY CAPITAL GOODS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Engineering, electrical equipment, and heavy industrial machinery. Requires CSV upload."
    ),
    "NIFTY_CHEMICALS": InstrumentMetadata(
        name="NIFTY CHEMICALS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Specialty chemicals, agrochemicals, and industrial basic chemicals. Requires CSV upload."
    ),
    "NIFTY_CEMENT": InstrumentMetadata(
        name="NIFTY CEMENT",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Major cement manufacturers and building materials producers. Requires CSV upload."
    ),
    "NIFTY_POWER": InstrumentMetadata(
        name="NIFTY POWER",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Thermal, hydro, nuclear, and renewable electricity generation/transmission. Requires CSV upload."
    ),
    "NIFTY_TELECOM": InstrumentMetadata(
        name="NIFTY TELECOMMUNICATIONS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Telecom cellular operators, tower infrastructure, and optic fiber providers. Requires CSV upload."
    ),
    "NIFTY_INSURANCE": InstrumentMetadata(
        name="NIFTY INSURANCE",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Life and general insurance providers in India. Requires CSV upload."
    ),
    "NIFTY_NBFC": InstrumentMetadata(
        name="NIFTY NBFC",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Non-Banking Financial Companies and retail credit lenders. Requires CSV upload."
    ),
    "NIFTY_HOSPITALS": InstrumentMetadata(
        name="NIFTY HOSPITALS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Healthcare service providers, hospital networks, and diagnostics. Requires CSV upload."
    ),
    "NIFTY_RETAIL": InstrumentMetadata(
        name="NIFTY RETAIL",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Organized retail chains, e-commerce, and lifestyle stores. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 3. NSE THEMATIC INDICES
    # -------------------------------------------------------------------------
    "NIFTY_DEFENCE": InstrumentMetadata(
        name="NIFTY INDIA DEFENCE",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks aerospace, defense equipment, and naval shipbuilders. Requires CSV upload."
    ),
    "NIFTY_DIGITAL": InstrumentMetadata(
        name="NIFTY INDIA DIGITAL",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Companies enabling digital transformation, cloud, and fintech. Requires CSV upload."
    ),
    "NIFTY_MANUFACTURING": InstrumentMetadata(
        name="NIFTY INDIA MANUFACTURING",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks 'Make in India' manufacturing sectors across auto, capital goods, metals, and textiles."
    ),
    "^CNXINFRA": InstrumentMetadata(
        name="NIFTY INFRASTRUCTURE",
        symbol="^CNXINFRA",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Companies in telecom, power, ports, roads, railways, and civil construction."
    ),
    "^CNXCOMMODITIES": InstrumentMetadata(
        name="NIFTY COMMODITIES",
        symbol="^CNXCOMMODITIES",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Companies in oil, metals, chemicals, agriculture, and mining."
    ),
    "^CPSE": InstrumentMetadata(
        name="NIFTY CPSE",
        symbol="^CPSE",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Central Public Sector Enterprises index designed for Indian government disinvestment program.",
        etf_proxy="CPSEETF.NS"
    ),
    "^CNXPSE": InstrumentMetadata(
        name="NIFTY PSE",
        symbol="^CNXPSE",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Public Sector Enterprises where Central/State govt holds > 51% stake."
    ),
    "^CNXCONSUM": InstrumentMetadata(
        name="NIFTY INDIA CONSUMPTION",
        symbol="^CNXCONSUM",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Tracks domestic Indian consumer spending: FMCG, autos, consumer durables, and healthcare."
    ),
    "^CNXMNC": InstrumentMetadata(
        name="NIFTY MNC",
        symbol="^CNXMNC",
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Tracks 30 Indian companies where foreign promoter shareholding exceeds 50%."
    ),
    "NIFTY_TOURISM": InstrumentMetadata(
        name="NIFTY INDIA TOURISM",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Hotels, airlines, travel agencies, and tour operators. Requires CSV upload."
    ),
    "NIFTY_RAILWAYS": InstrumentMetadata(
        name="NIFTY INDIA RAILWAYS PSU",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Indian railway PSUs (IRCTC, IRFC, RVNL, RITES, CONCOR). Requires CSV upload."
    ),
    "NIFTY_INTERNET": InstrumentMetadata(
        name="NIFTY INDIA INTERNET",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Internet platforms, classifieds, food delivery, and digital marketplaces. Requires CSV upload."
    ),
    "NIFTY_MOBILITY": InstrumentMetadata(
        name="NIFTY MOBILITY",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Electric vehicles, battery tech, shared mobility, and charging infrastructure. Requires CSV upload."
    ),
    "NIFTY_HOUSING": InstrumentMetadata(
        name="NIFTY HOUSING",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Housing ecosystem: developers, home finance, paints, sanitaryware, and tiles. Requires CSV upload."
    ),
    "NIFTY_LOGISTICS": InstrumentMetadata(
        name="NIFTY TRANSPORTATION & LOGISTICS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Supply chain, cold storage, warehousing, maritime freight, and express cargo. Requires CSV upload."
    ),
    "NIFTY_REITS_INVITS": InstrumentMetadata(
        name="NIFTY REITS & INVITS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Listed Real Estate Investment Trusts and Infrastructure Investment Trusts. Requires CSV upload."
    ),
    "NIFTY_CAPITAL_MARKETS": InstrumentMetadata(
        name="NIFTY CAPITAL MARKETS",
        symbol=None,
        exchange="NSE",
        asset_class="Index",
        index_family="NSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Stock exchanges, depositories, AMCs, rating agencies, and brokerages. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 4. NSE STRATEGY INDICES (Smart Beta)
    # -------------------------------------------------------------------------
    "NIFTY_100_EQUAL_WEIGHT": InstrumentMetadata(
        name="NIFTY 100 EQUAL WEIGHT",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Equal 1% allocation across all NIFTY 100 constituents at rebalance. Requires CSV upload."
    ),
    "NIFTY_ALPHA_50": InstrumentMetadata(
        name="NIFTY ALPHA 50",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Selects 50 top Jensen's Alpha stocks from top 300 listed companies. Requires CSV upload."
    ),
    "NIFTY_LOW_VOLATILITY_50": InstrumentMetadata(
        name="NIFTY LOW VOLATILITY 50",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Selects least volatile 50 stocks from NIFTY 100. Requires CSV upload."
    ),
    "NIFTY_HIGH_BETA_50": InstrumentMetadata(
        name="NIFTY HIGH BETA 50",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Selects top 50 high-beta equities for bull market participation. Requires CSV upload."
    ),
    "NIFTY_QUALITY_30": InstrumentMetadata(
        name="NIFTY QUALITY 30",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Selects 30 high-ROE, low-debt, stable EPS growth companies. Requires CSV upload."
    ),
    "NIFTY_VALUE_20": InstrumentMetadata(
        name="NIFTY VALUE 20",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Selects 20 value-oriented companies from NIFTY 50 based on low P/E, P/B, and high Div Yield."
    ),
    "NIFTY_MOMENTUM_30": InstrumentMetadata(
        name="NIFTY MOMENTUM 30",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Selects 30 top 6-month & 12-month normalized momentum stocks from NIFTY 200. Requires CSV upload."
    ),
    "^CNXDIVOPP": InstrumentMetadata(
        name="NIFTY DIVIDEND OPPORTUNITIES 50",
        symbol="^CNXDIVOPP",
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Tracks top 50 high-dividend-yielding companies on NSE."
    ),
    "NIFTY_500_EQUAL_WEIGHT": InstrumentMetadata(
        name="NIFTY 500 EQUAL WEIGHT",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Equal 0.2% weight per constituent in NIFTY 500. Requires CSV upload."
    ),
    "NIFTY_500_MOMENTUM_50": InstrumentMetadata(
        name="NIFTY 500 MOMENTUM 50",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Top 50 momentum equities across the broad NIFTY 500 universe. Requires CSV upload."
    ),
    "NIFTY_500_QUALITY_50": InstrumentMetadata(
        name="NIFTY 500 QUALITY 50",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Top 50 quality scores based on ROE, D/E, and EPS variance across Nifty 500. Requires CSV upload."
    ),
    "NIFTY_200_QUALITY_30": InstrumentMetadata(
        name="NIFTY 200 QUALITY 30",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Top 30 quality equities from the NIFTY 200. Requires CSV upload."
    ),
    "NIFTY_200_VALUE_30": InstrumentMetadata(
        name="NIFTY 200 VALUE 30",
        symbol=None,
        exchange="NSE",
        asset_class="Strategy Index",
        index_family="NSE Strategy",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Top 30 value score equities from the NIFTY 200. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 5. BSE BROAD MARKET INDICES
    # -------------------------------------------------------------------------
    "^BSESN": InstrumentMetadata(
        name="BSE SENSEX",
        symbol="^BSESN",
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Oldest Indian index comprising 30 financially sound and liquid companies on BSE.",
        etf_proxy="SENSEXBEES.NS"
    ),
    "^BSE100": InstrumentMetadata(
        name="BSE 100",
        symbol="^BSE100",
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Top 100 listed companies on the Bombay Stock Exchange."
    ),
    "^BSE200": InstrumentMetadata(
        name="BSE 200",
        symbol="^BSE200",
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Top 200 companies on BSE."
    ),
    "^BSE500": InstrumentMetadata(
        name="BSE 500",
        symbol="^BSE500",
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Broad index representing ~93% of total market cap on BSE."
    ),
    "BSE_SENSEX_50": InstrumentMetadata(
        name="BSE SENSEX 50",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="50 largest and most liquid stocks on BSE. Requires CSV upload."
    ),
    "BSE_SENSEX_NEXT_50": InstrumentMetadata(
        name="BSE SENSEX NEXT 50",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Next 50 stocks following BSE Sensex 50. Requires CSV upload."
    ),
    "BSE_150_MIDCAP": InstrumentMetadata(
        name="BSE 150 MIDCAP",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Midcap 150 companies on BSE. Requires CSV upload."
    ),
    "BSE_250_SMALLCAP": InstrumentMetadata(
        name="BSE 250 SMALLCAP",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Broad Market",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Smallcap 250 companies on BSE. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 6. BSE SECTORAL INDICES
    # -------------------------------------------------------------------------
    "^BSEBANK": InstrumentMetadata(
        name="BSE BANKEX",
        symbol="^BSEBANK",
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Benchmark index tracking prominent public and private sector banks on BSE."
    ),
    "BSE_AUTO": InstrumentMetadata(
        name="BSE AUTO",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Automobile manufacturers and ancillary companies listed on BSE. Requires CSV upload."
    ),
    "BSE_IT": InstrumentMetadata(
        name="BSE IT",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Information Technology companies on BSE. Requires CSV upload."
    ),
    "BSE_FMCG": InstrumentMetadata(
        name="BSE FMCG",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="FMCG companies on BSE. Requires CSV upload."
    ),
    "BSE_METAL": InstrumentMetadata(
        name="BSE METAL",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Metals and mining companies on BSE. Requires CSV upload."
    ),
    "BSE_HEALTHCARE": InstrumentMetadata(
        name="BSE HEALTHCARE",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Pharmaceuticals and healthcare companies on BSE. Requires CSV upload."
    ),
    "BSE_OIL_GAS": InstrumentMetadata(
        name="BSE OIL & GAS",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Oil exploration, refining, and gas marketing on BSE. Requires CSV upload."
    ),
    "BSE_CAPITAL_GOODS": InstrumentMetadata(
        name="BSE CAPITAL GOODS",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Capital goods and heavy engineering on BSE. Requires CSV upload."
    ),
    "BSE_CONSUMER_DURABLES": InstrumentMetadata(
        name="BSE CONSUMER DURABLES",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Consumer electronics and home appliances on BSE. Requires CSV upload."
    ),
    "BSE_PSU": InstrumentMetadata(
        name="BSE PSU",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Sectoral",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Public Sector Undertakings listed on BSE. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 7. BSE THEMATIC INDICES
    # -------------------------------------------------------------------------
    "BSE_MANUFACTURING": InstrumentMetadata(
        name="BSE INDIA MANUFACTURING",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Manufacturing companies listed on BSE. Requires CSV upload."
    ),
    "BSE_INFRASTRUCTURE": InstrumentMetadata(
        name="BSE INDIA INFRASTRUCTURE",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Infrastructure developers and utilities on BSE. Requires CSV upload."
    ),
    "BSE_HOUSING": InstrumentMetadata(
        name="BSE HOUSING",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Housing developers, construction materials, and home finance. Requires CSV upload."
    ),
    "BSE_INTERNET_ECONOMY": InstrumentMetadata(
        name="BSE INTERNET ECONOMY",
        symbol=None,
        exchange="BSE",
        asset_class="Index",
        index_family="BSE Thematic",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="New-age digital platforms and internet businesses on BSE. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 8. INDIAN ETFS
    # -------------------------------------------------------------------------
    "NIFTYBEES.NS": InstrumentMetadata(
        name="NIFTY 50 ETF (NIFTYBEES)",
        symbol="NIFTYBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Most liquid Indian index ETF physically backed by NIFTY 50 constituents."
    ),
    "BANKBEES.NS": InstrumentMetadata(
        name="BANK NIFTY ETF (BANKBEES)",
        symbol="BANKBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Liquid ETF physically replicating NIFTY BANK index."
    ),
    "GOLDBEES.NS": InstrumentMetadata(
        name="GOLD ETF (GOLDBEES)",
        symbol="GOLDBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Physical domestic gold-backed ETF traded on NSE."
    ),
    "JUNIORBEES.NS": InstrumentMetadata(
        name="NIFTY NEXT 50 ETF (JUNIORBEES)",
        symbol="JUNIORBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund tracking NIFTY Next 50 index."
    ),
    "ITBEES.NS": InstrumentMetadata(
        name="NIFTY IT ETF (ITBEES)",
        symbol="ITBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund tracking NIFTY IT index constituents."
    ),
    "AUTOBEES.NS": InstrumentMetadata(
        name="NIFTY AUTO ETF (AUTOBEES)",
        symbol="AUTOBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund tracking NIFTY Auto index constituents."
    ),
    "PHARMABEES.NS": InstrumentMetadata(
        name="NIFTY PHARMA ETF (PHARMABEES)",
        symbol="PHARMABEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund tracking NIFTY Pharma index constituents."
    ),
    "PSUBNKBEES.NS": InstrumentMetadata(
        name="NIFTY PSU BANK ETF (PSUBNKBEES)",
        symbol="PSUBNKBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund tracking NIFTY PSU Bank index constituents."
    ),
    "CPSEETF.NS": InstrumentMetadata(
        name="CPSE ETF",
        symbol="CPSEETF.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund tracking Central Public Sector Enterprises index."
    ),
    "SILVERBEES.NS": InstrumentMetadata(
        name="SILVER ETF (SILVERBEES)",
        symbol="SILVERBEES.NS",
        exchange="NSE",
        asset_class="ETF",
        index_family="Indian ETFs",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Exchange-traded fund physically backed by domestic silver."
    ),

    # -------------------------------------------------------------------------
    # 9. INDIAN DERIVATIVES (FUTURES & CURRENCY PROXIES)
    # -------------------------------------------------------------------------
    "NIFTY_FUTURES_PROXY": InstrumentMetadata(
        name="NIFTY 50 Futures (Proxy)",
        symbol="^NSEI",
        exchange="NSE",
        asset_class="Futures Proxy",
        index_family="Indian Derivatives",
        currency="INR",
        data_provider="Yahoo Finance API (Spot Proxy)",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Continuous spot proxy for NIFTY 50 index futures contract. Multiplier: 75 (current), Tick size: 0.05.",
        contract_multiplier=75.0,
        tick_size=0.05,
        initial_margin_pct=0.12,
        maintenance_margin_pct=0.08
    ),
    "BANKNIFTY_FUTURES_PROXY": InstrumentMetadata(
        name="BANK NIFTY Futures (Proxy)",
        symbol="^NSEBANK",
        exchange="NSE",
        asset_class="Futures Proxy",
        index_family="Indian Derivatives",
        currency="INR",
        data_provider="Yahoo Finance API (Spot Proxy)",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Continuous spot proxy for BANK NIFTY futures contract. Multiplier: 30 (current), Tick size: 0.05.",
        contract_multiplier=30.0,
        tick_size=0.05,
        initial_margin_pct=0.15,
        maintenance_margin_pct=0.10
    ),
    "USDINR=X": InstrumentMetadata(
        name="USD/INR Spot / Currency Rate",
        symbol="USDINR=X",
        exchange="Interbank / FX",
        asset_class="Currency",
        index_family="Indian Derivatives",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="US Dollar to Indian Rupee spot exchange rate reference for FX risk, hedging, and RBI intervention analytics."
    ),

    # -------------------------------------------------------------------------
    # 10. INDIAN FIXED INCOME (G-SEC, SDL, CORPORATE BOND)
    # -------------------------------------------------------------------------
    "NIFTY_10YR_BENCHMARK_GSEC": InstrumentMetadata(
        name="NIFTY 10 yr Benchmark G-Sec Index",
        symbol=None,
        exchange="NSE / CCIL",
        asset_class="Fixed Income",
        index_family="Indian Fixed Income",
        currency="INR",
        data_provider="CSV Upload / CCIL Data Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks performance of the on-the-run 10-year Indian sovereign Government of India security. Requires CCIL/NSE CSV."
    ),
    "NIFTY_SDL_INDEX": InstrumentMetadata(
        name="NIFTY State Development Loans (SDL) Index",
        symbol=None,
        exchange="NSE",
        asset_class="Fixed Income",
        index_family="Indian Fixed Income",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks debt issued by Indian state governments. Requires CSV upload."
    ),
    "NIFTY_CORP_BOND": InstrumentMetadata(
        name="NIFTY Corporate Bond Index (AAA)",
        symbol=None,
        exchange="NSE",
        asset_class="Fixed Income",
        index_family="Indian Fixed Income",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Measures return of top-rated AAA corporate debt across maturities. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 11. INDIAN HYBRID INDICES
    # -------------------------------------------------------------------------
    "NIFTY_50_HYBRID_DEBT_50_50": InstrumentMetadata(
        name="NIFTY 50 Hybrid Composite Debt 50:50 Index",
        symbol=None,
        exchange="NSE",
        asset_class="Hybrid Index",
        index_family="Indian Hybrid Indices",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks 50% NIFTY 50 + 50% NIFTY Composite Debt Index. Rebalanced quarterly. Requires CSV upload."
    ),
    "NIFTY_50_HYBRID_DEBT_65_35": InstrumentMetadata(
        name="NIFTY 50 Hybrid Composite Debt 65:35 Index",
        symbol=None,
        exchange="NSE",
        asset_class="Hybrid Index",
        index_family="Indian Hybrid Indices",
        currency="INR",
        data_provider="CSV Upload / Exchange License Required",
        availability=AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV,
        notes="Tracks 65% NIFTY 50 + 35% NIFTY Composite Debt Index. Suitable for aggressive hybrid funds. Requires CSV upload."
    ),

    # -------------------------------------------------------------------------
    # 12. INDIAN MARKET VOLATILITY INDICATORS
    # -------------------------------------------------------------------------
    "^INDIAVIX": InstrumentMetadata(
        name="INDIA VIX (Volatility Index)",
        symbol="^INDIAVIX",
        exchange="NSE",
        asset_class="Market Indicator",
        index_family="Indian Market Volatility Indicators",
        currency="INR",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Measures implied annual volatility expectations over the next 30 calendar days from NIFTY option bid/ask quotes."
    ),

    # -------------------------------------------------------------------------
    # 13. GLOBAL EQUITIES & COMMODITY BENCHMARKS
    # -------------------------------------------------------------------------
    "^GSPC": InstrumentMetadata(
        name="S&P 500 (US Broad Market)",
        symbol="^GSPC",
        exchange="Cboe / NYSE",
        asset_class="Index",
        index_family="Global Markets",
        currency="USD",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Benchmark index for large-cap US equities.",
        etf_proxy="SPY",
        contract_multiplier=50.0,
        tick_size=0.25
    ),
    "SPY": InstrumentMetadata(
        name="SPDR S&P 500 ETF Trust (SPY)",
        symbol="SPY",
        exchange="NYSE Arca",
        asset_class="ETF",
        index_family="Global Markets",
        currency="USD",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="World's largest physical equity ETF tracking the S&P 500 index."
    ),
    "^NDX": InstrumentMetadata(
        name="NASDAQ-100 (US Technology)",
        symbol="^NDX",
        exchange="NASDAQ",
        asset_class="Index",
        index_family="Global Markets",
        currency="USD",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Top 100 non-financial tech and growth leaders.",
        etf_proxy="QQQ",
        contract_multiplier=20.0,
        tick_size=0.25
    ),
    "QQQ": InstrumentMetadata(
        name="Invesco QQQ Trust (QQQ)",
        symbol="QQQ",
        exchange="NASDAQ",
        asset_class="ETF",
        index_family="Global Markets",
        currency="USD",
        data_provider="Yahoo Finance API",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Liquid ETF tracking the NASDAQ-100 index."
    ),
    "GC=F": InstrumentMetadata(
        name="Gold Futures (Continuous)",
        symbol="GC=F",
        exchange="COMEX",
        asset_class="Commodity Futures",
        index_family="Global Commodities",
        currency="USD",
        data_provider="Yahoo Finance API (Continuous Stitch)",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Front-month continuous futures proxy for 100 troy oz gold. Does not model roll yields.",
        contract_multiplier=100.0,
        tick_size=0.10,
        initial_margin_pct=0.10,
        maintenance_margin_pct=0.07
    ),
    "CL=F": InstrumentMetadata(
        name="Crude Oil WTI Futures (Continuous)",
        symbol="CL=F",
        exchange="NYMEX",
        asset_class="Commodity Futures",
        index_family="Global Commodities",
        currency="USD",
        data_provider="Yahoo Finance API (Continuous Stitch)",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Front-month continuous futures proxy for 1,000 barrels of Light Sweet Crude Oil. Subject to roll yield distortion.",
        contract_multiplier=1000.0,
        tick_size=0.01,
        initial_margin_pct=0.15,
        maintenance_margin_pct=0.10
    ),
    "ZN=F": InstrumentMetadata(
        name="10-Year US Treasury Note Futures",
        symbol="ZN=F",
        exchange="CBOT",
        asset_class="Fixed Income Futures",
        index_family="Global Fixed Income",
        currency="USD",
        data_provider="Yahoo Finance API (Continuous Stitch)",
        availability=AvailabilityStatus.AVAILABLE_ONLINE,
        notes="Continuous futures contract for US 10-Year Treasury Notes. Par value $100,000.",
        contract_multiplier=1000.0,
        tick_size=0.015625,
        initial_margin_pct=0.05,
        maintenance_margin_pct=0.03
    )
}


FAMILY_ALIASES: Dict[str, List[str]] = {
    "BSE Indices": ["BSE Broad Market", "BSE Sectoral", "BSE Thematic"],
    "Indian Indicators": ["Indian Market Volatility Indicators", "Indian Derivatives", "Indian ETFs"],
    "Fixed Income": ["Indian Fixed Income"],
    "Hybrid Indices": ["Indian Hybrid Indices"],
}


def get_instruments_by_family(family: str) -> List[InstrumentMetadata]:
    """Returns all instruments belonging to a specific index family, supporting category aliases."""
    if family in FAMILY_ALIASES:
        target_families = set(FAMILY_ALIASES[family])
        return [meta for meta in INSTRUMENT_REGISTRY.values() if meta.index_family in target_families or meta.index_family == family]
    return [meta for meta in INSTRUMENT_REGISTRY.values() if meta.index_family == family]


def get_all_families() -> List[str]:
    """Returns list of distinct index families in canonical order."""
    families: List[str] = []
    # Ensure Indian Market Categories appear in primary canonical order
    for cat in INDIAN_MARKET_CATEGORIES:
        if any(m.index_family == cat for m in INSTRUMENT_REGISTRY.values()) and cat not in families:
            families.append(cat)
    for meta in INSTRUMENT_REGISTRY.values():
        if meta.index_family not in families:
            families.append(meta.index_family)
    return families


def get_available_online_instruments() -> List[InstrumentMetadata]:
    """Returns all instruments confirmed to be directly downloadable via Yahoo Finance."""
    return [meta for meta in INSTRUMENT_REGISTRY.values() if meta.availability == AvailabilityStatus.AVAILABLE_ONLINE]


def get_instrument_by_symbol_or_key(key: str) -> Optional[InstrumentMetadata]:
    """Lookup an instrument by registry key or ticker symbol."""
    if key in INSTRUMENT_REGISTRY:
        return INSTRUMENT_REGISTRY[key]
    for meta in INSTRUMENT_REGISTRY.values():
        if meta.symbol == key:
            return meta
    return None


def is_indian_instrument(meta: InstrumentMetadata) -> bool:
    """Returns True if the instrument belongs to the Indian market taxonomy."""
    return meta.index_family in INDIAN_MARKET_CATEGORIES or meta.currency == "INR" or meta.exchange in ["NSE", "BSE", "MCX"]
