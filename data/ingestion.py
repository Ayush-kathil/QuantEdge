"""Market data ingestion module supporting Yahoo Finance API and CSV file imports."""
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import yfinance as yf

from config.settings import CACHE_DIR, DEFAULT_CACHE_EXPIRY_HOURS, INSTRUMENT_CATALOG
from data.validation import validate_market_data, ValidationResult
from data.cache import MarketDataCache

logger = logging.getLogger("QuantEdge.DataIngestion")


@dataclass
class MarketDataPayload:
    """Standardized container for ingested market time-series and ingestion metadata."""
    ticker: str
    df: pd.DataFrame
    source: str
    retrieval_timestamp: datetime
    start_date: str
    end_date: str
    row_count: int
    validation: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataIngestion:
    """Ingests market time-series from real-time / historical Yahoo Finance or user CSVs."""

    def __init__(self, cache_enabled: bool = True, cache_expiry_hours: int = DEFAULT_CACHE_EXPIRY_HOURS):
        self.cache_enabled = cache_enabled
        self.cache = MarketDataCache(CACHE_DIR, cache_expiry_hours) if cache_enabled else None

    def fetch_yfinance(
        self,
        ticker: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = "1d",
        strict_ohlc: bool = False
    ) -> MarketDataPayload:
        """
        Retrieves historical market data for a given ticker from Yahoo Finance.
        
        Parameters:
            ticker: Yahoo Finance ticker (e.g., '^GSPC', 'SPY', 'GC=F')
            start_date: Start date string 'YYYY-MM-DD' or date object
            end_date: End date string 'YYYY-MM-DD' or date object
            interval: Bar frequency ('1d', '1wk', etc.)
            strict_ohlc: Whether OHLC anomalies raise errors or warnings
            
        Returns:
            MarketDataPayload with validated DataFrame and metadata.
            
        Raises:
            ValueError: When ticker is invalid, market data is completely empty, or network fails.
        """
        ticker_sym = ticker.strip().upper()
        s_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, (date, datetime)) else str(start_date)
        e_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, (date, datetime)) else str(end_date)

        # Check cache first
        if self.cache_enabled and self.cache is not None:
            cached_df = self.cache.get(ticker_sym, s_str, e_str, interval)
            if cached_df is not None and not cached_df.empty:
                val_res = validate_market_data(cached_df, strict_ohlc=strict_ohlc)
                if val_res.is_valid:
                    return MarketDataPayload(
                        ticker=ticker_sym,
                        df=val_res.cleaned_df if val_res.cleaned_df is not None else cached_df,
                        source="Local Cache (yfinance)",
                        retrieval_timestamp=datetime.now(),
                        start_date=s_str,
                        end_date=e_str,
                        row_count=len(val_res.cleaned_df or cached_df),
                        validation=val_res,
                        metadata=INSTRUMENT_CATALOG.get(ticker_sym, {})
                    )

        # Download from yfinance
        try:
            raw_data = yf.download(
                tickers=ticker_sym,
                start=s_str,
                end=e_str,
                interval=interval,
                auto_adjust=False,
                progress=False
            )
        except Exception as e:
            raise ValueError(f"Network or service error while retrieving '{ticker_sym}' from Yahoo Finance: {str(e)}")

        if raw_data is None or raw_data.empty:
            raise ValueError(
                f"No historical market data found for ticker '{ticker_sym}' between {s_str} and {e_str}. "
                "Please verify the ticker symbol or try another date window."
            )

        # Handle MultiIndex columns that modern yfinance produces
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Try to grab the first level or match OHLCV
            try:
                # If ticker is level 1
                if ticker_sym in raw_data.columns.get_level_values(1):
                    raw_data = raw_data.xs(ticker_sym, axis=1, level=1)
                elif raw_data.columns.nlevels == 2:
                    raw_data.columns = [col[0] for col in raw_data.columns]
            except Exception:
                raw_data.columns = [col[0] if isinstance(col, tuple) else col for col in raw_data.columns]

        # Validate and clean data
        val_res = validate_market_data(raw_data, strict_ohlc=strict_ohlc)
        if not val_res.is_valid:
            err_msg = "; ".join(val_res.errors)
            raise ValueError(f"Market data validation failed for '{ticker_sym}': {err_msg}")

        clean_df = val_res.cleaned_df if val_res.cleaned_df is not None else raw_data

        # Cache on disk
        if self.cache_enabled and self.cache is not None:
            self.cache.set(ticker_sym, s_str, e_str, clean_df, interval)

        instrument_info = INSTRUMENT_CATALOG.get(ticker_sym, {
            "name": ticker_sym,
            "asset_class": "Custom Asset",
            "exchange": "Various",
            "description": f"Custom user-selected instrument ({ticker_sym})"
        })

        return MarketDataPayload(
            ticker=ticker_sym,
            df=clean_df,
            source="Yahoo Finance API (yfinance)",
            retrieval_timestamp=datetime.now(),
            start_date=s_str,
            end_date=e_str,
            row_count=len(clean_df),
            validation=val_res,
            metadata=instrument_info
        )

    def load_csv(
        self,
        file_or_path: Union[str, io.BytesIO, io.StringIO],
        ticker_name: str = "CUSTOM_CSV",
        strict_ohlc: bool = False
    ) -> MarketDataPayload:
        """
        Parses and validates an uploaded or local CSV file containing OHLCV price history.
        
        Parameters:
            file_or_path: Local filepath or file-like object (e.g. from Streamlit file_uploader)
            ticker_name: Identifier label for the custom asset
            strict_ohlc: Whether OHLC inequalities trigger errors or warnings
            
        Returns:
            MarketDataPayload
        """
        try:
            df = pd.read_csv(file_or_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV input file: {str(e)}")

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        # Identify datetime column
        date_col = None
        for col in df.columns:
            if str(col).lower() in ["date", "datetime", "timestamp", "time"]:
                date_col = col
                break

        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col)
        else:
            # Try parsing first column as date index
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col])
                df = df.set_index(first_col)
            except Exception:
                raise ValueError(
                    f"Could not identify a valid Datetime column in the CSV. "
                    f"Available columns: {list(df.columns)}. Expected 'Date' or 'Datetime'."
                )

        val_res = validate_market_data(df, strict_ohlc=strict_ohlc)
        if not val_res.is_valid:
            raise ValueError(f"CSV validation failed: {'; '.join(val_res.errors)}")

        clean_df = val_res.cleaned_df if val_res.cleaned_df is not None else df
        s_date = str(clean_df.index[0].date())
        e_date = str(clean_df.index[-1].date())

        return MarketDataPayload(
            ticker=ticker_name,
            df=clean_df,
            source="User Uploaded CSV",
            retrieval_timestamp=datetime.now(),
            start_date=s_date,
            end_date=e_date,
            row_count=len(clean_df),
            validation=val_res,
            metadata={
                "name": ticker_name,
                "asset_class": "User Uploaded Series",
                "exchange": "Custom File",
                "description": "Custom CSV dataset supplied by user"
            }
        )
