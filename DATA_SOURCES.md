# QuantEdge v3.1: Data Sources, Instrument Taxonomy & Provenance

## 1. Overview & Data Architecture

QuantEdge operates a dual-source ingestion architecture designed for research reproducibility and operational resilience:
1. **Live Automated Ingestion:** Powered by Yahoo Finance (`yfinance`), streaming daily end-of-day (EOD) OHLCV bars.
2. **Deterministic Offline Ingestion:** High-speed CSV ingestion parser for proprietary tick/minute/daily files, including pre-packaged deterministic fixtures (`sample_data/sample_sp500.csv`).

---

## 2. Comprehensive Indian Market Taxonomy (12 Standardized Categories)

In compliance with exchange reporting conventions (National Stock Exchange of India and Bombay Stock Exchange), the instrument catalog is structured across 12 distinct families:

| # | Taxonomy Category | Representative Instruments / Tickers | Primary Provider / Availability |
|---|---|---|---|
| 1 | **NSE Broad Market** | `^NSEI` (NIFTY 50), `^NSMIDCP` (NIFTY Next 50), `^CNX100` (NIFTY 100), `^CNX200`, `^CRSLDX` (NIFTY 500), `NIFTY MIDCAP 50`, `NIFTY SMALLCAP 100` | Yahoo Finance / Available Online |
| 2 | **NSE Sectoral** | `^NSEBANK` (Nifty Bank), `^CNXIT` (Nifty IT), `^CNXAUTO` (Nifty Auto), `^CNXFMCG` (Nifty FMCG), `^CNXPHARMA`, `^CNXMETAL`, `^CNXREALTY`, `^CNXENERGY`, `^NSEMDCP50` | Yahoo Finance / Available Online |
| 3 | **NSE Thematic** | `NIFTY CPSE`, `NIFTY INFRA`, `NIFTY COMMODITIES`, `NIFTY INDIA CONSUMPTION`, `NIFTY MNC`, `NIFTY PSE`, `NIFTY SERVICES SECTOR` | Yahoo Finance / Available Online |
| 4 | **NSE Strategy & Factor** | `NIFTY 100 Low Volatility 30`, `NIFTY 200 Momentum 30`, `NIFTY Alpha 50`, `NIFTY Quality 30`, `NIFTY High Beta 50` | CSV Upload Required (Official NSE proprietary index feeds) |
| 5 | **BSE Broad Market** | `^BSESN` (S&P BSE SENSEX), `^BSE100` (BSE 100), `^BSE200`, `^BSE500`, `BSE SENSEX 50`, `BSE MidCap`, `BSE SmallCap` | Yahoo Finance (`^BSESN`, `^BSE100`, etc.) / Available Online |
| 6 | **BSE Sectoral & Thematic** | `^BSEBANK` (BSE BANKEX), `BSE AUTO`, `BSE IT`, `BSE FMCG`, `BSE METAL`, `BSE OIL & GAS`, `BSE POWER`, `BSE REALTY`, `BSE HEALTHCARE` | Yahoo Finance (`^BSEBANK`, etc.) / Available Online |
| 7 | **Indian Fixed Income / Debt** | `CRISIL 10 Year Gilt Index`, `CRISIL Composite Bond Index`, `NIFTY 10 yr Benchmark G-Sec Index`, `NIFTY AAA Bond Index` | CSV Upload Required (Institutional fixed income data) |
| 8 | **Indian Hybrid & Multi-Asset** | `CRISIL Hybrid 50+50 Moderate Index`, `NIFTY 50 & Short Duration Debt Dynamic P/E Index`, `Multi-Asset Allocation Benchmark` | CSV Upload Required (Multi-asset blend feeds) |
| 9 | **Indian Volatility & Indicators**| `^INDIAVIX` (India Volatility Index) | Yahoo Finance / Available Online |
| 10| **Currency & INR Pairs** | `USDINR=X` (USD/INR Spot), `EURINR=X`, `GBPINR=X`, `JPYINR=X` | Yahoo Finance / Available Online |
| 11| **Commodity Derivatives** | `GC=F` (Gold Futures Proxy), `CL=F` (Crude Oil WTI Proxy), `SI=F` (Silver Proxy), `NG=F` (Natural Gas Proxy) | Yahoo Finance / Available Online |
| 12| **Liquid ETF Cash Proxies** | `NIFTYBEES.NS` (Nifty 50 ETF), `BANKBEES.NS` (Bank Nifty ETF), `JUNIORBEES.NS`, `GOLDBEES.NS`, `LIQUIDBEES.NS` | Yahoo Finance / Available Online |

---

## 3. Global Benchmark Coverage

For international multi-asset research and cross-market comparisons:
- **US Equities & Global Indices:** `^GSPC` (S&P 500), `^IXIC` (NASDAQ Composite), `^DJI` (Dow Jones Industrial), `^FTSE` (FTSE 100), `^N225` (Nikkei 225), `^HSI` (Hang Seng), `^DAX` (DAX 40).
- **Global Volatility & Rates:** `^VIX` (CBOE Volatility Index), `^TNX` (US 10-Year Treasury Yield), `^IRX` (US 13-Week Treasury Bill).

---

## 4. Futures Continuous Contracts vs. Spot Proxies: Methodological Disclosure

Institutional quantitative researchers must explicitly distinguish between **continuous stitched futures** and **underlying cash/spot index proxies**:

1. **Spot Index Proxies (`^NSEI`, `^NSEBANK`):**
   - Represents the spot index level, not the tradable futures derivative.
   - Does not incur cost-of-carry premiums, backwardation/contango discounts, or dividend adjustments.
   - For futures modeling, QuantEdge pairs spot series with explicit lot size schedules (`75` for NIFTY, `30` for BANKNIFTY) and exchange margin rates (12% initial, 8% maintenance).

2. **Continuous Stitched Commodity Futures (`GC=F`, `CL=F`, `SI=F`):**
   - Spliced front-month contract series provided by Yahoo Finance.
   - Subject to rollover price step-discontinuities and roll-yield slippage across contract expiries.
   - Backtest results on continuous series should be validated against actual discrete expiry contract files for live production deployment.

---

## 5. Data Provenance & Cryptographic Fingerprinting

To guarantee research integrity and prevent dataset tampering:
- **Download Timestamp:** Recorded at the moment of API retrieval (UTC and IST).
- **Data Freshness Indicator:** Measures elapsed hours between the latest bar's close and current execution time.
- **SHA-256 Dataset Fingerprint:** Computed over canonical normalized timestamps, open, high, low, close, and volume arrays.
- **Audit Verification:** Guarantees that any experiment manifest (`manifest.json`) or PDF research report can be mapped to the exact dataset state.
