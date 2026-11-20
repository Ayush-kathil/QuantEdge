# Sample Financial Data Fixtures

This directory contains verified sample market datasets and documentation for offline exploration and unit testing.

## Files
- `sample_spy.csv`: Historical daily OHLCV dataset of SPDR S&P 500 ETF Trust (`SPY`) from 2023-01-01 to 2024-01-01.
- `synthetic_fixture.csv`: Deterministic, mathematically controlled price path specifically designed for unit testing indicator convergence and execution zero-bias constraints.

## Column Schema
All files adhere to the QuantEdge standard market schema:
- `Date` (YYYY-MM-DD, sorted ascending)
- `Open` (Float, > 0)
- `High` (Float, >= Open, Low, Close)
- `Low` (Float, <= Open, High, Close)
- `Close` (Float, > 0)
- `Volume` (Float, >= 0)
- `Adj Close` (Optional Float)
