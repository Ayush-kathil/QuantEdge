"""Market data ingestion module supporting Yahoo Finance API, CSV file imports, and provenance auditing."""
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Optional, Union, Dict, Any, List, Tuple
import pandas as pd
import yfinance as yf

from config.settings import CACHE_DIR, DEFAULT_CACHE_EXPIRY_HOURS, INSTRUMENT_CATALOG
from data.validation import validate_market_data, ValidationResult
from data.cache import MarketDataCache

logger = logging.getLogger("QuantEdge.DataIngestion")


@dataclass
class MarketDataPayload:
    """Standardized container for ingested market time-series and audited provenance metadata."""
    ticker: str
    df: pd.DataFrame
    source: str
    retrieval_timestamp: datetime
    start_date: str
    end_date: str
    row_count: int
    validation: ValidationResult
    dataset_fingerprint: str
    data_freshness_status: str
    calendar_lag_days: int
    instrument_type: str
    corporate_action_notes: str
    futures_rollover_notes: str
    is_realtime: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataIngestion:
    """Ingests market time-series from historical Yahoo Finance or user CSVs with provenance tracking."""

    def __init__(self, cache_enabled: bool = True, cache_expiry_hours: int = DEFAULT_CACHE_EXPIRY_HOURS):
        self.cache_enabled = cache_enabled
        self.cache = MarketDataCache(CACHE_DIR, cache_expiry_hours) if cache_enabled else None

    def _assess_freshness(self, last_date: pd.Timestamp) -> Tuple[str, int]:
        """Calculates lag between most recent bar and current UTC date."""
        today = datetime.now(timezone.utc).date()
        bar_date = last_date.date() if hasattr(last_date, "date") else last_date
        lag = (today - bar_date).days
        if lag <= 1:
            status = "Recent EOD Close (Delayed by Exchange / Provider)"
        elif lag <= 4:
            status = f"Current Weekend / Short Lag ({lag} days)"
        else:
            status = f"Historical EOD Archive (Stale / Historical: {lag} days lag)"
        return status, max(0, lag)

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
            MarketDataPayload with validated DataFrame, fingerprint, and provenance metadata.
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
                    clean_df = val_res.cleaned_df if val_res.cleaned_df is not None else cached_df
                    catalog_entry = INSTRUMENT_CATALOG.get(ticker_sym, {})
                    freshness_status, lag_days = self._assess_freshness(clean_df.index[-1])
                    return MarketDataPayload(
                        ticker=ticker_sym,
                        df=clean_df,
                        source="Local Cache (yfinance EOD)",
                        retrieval_timestamp=datetime.now(timezone.utc),
                        start_date=s_str,
                        end_date=e_str,
                        row_count=len(clean_df),
                        validation=val_res,
                        dataset_fingerprint=val_res.fingerprint_sha256,
                        data_freshness_status=freshness_status,
                        calendar_lag_days=lag_days,
                        instrument_type=catalog_entry.get("instrument_type", "Standard Security"),
                        corporate_action_notes=catalog_entry.get("corporate_actions", "Standard corporate action adjustments apply."),
                        futures_rollover_notes=catalog_entry.get("futures_rollover_notes", "Not applicable."),
                        is_realtime=False,
                        metadata=catalog_entry
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
            try:
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

        catalog_entry = INSTRUMENT_CATALOG.get(ticker_sym, {
            "name": ticker_sym,
            "asset_class": "Custom Asset",
            "instrument_type": "Custom User Symbol",
            "exchange": "Various",
            "description": f"Custom user-selected instrument ({ticker_sym})",
            "corporate_actions": "Verify with external source whether splits or dividends are applied.",
            "futures_rollover_notes": "Continuous contracts may contain unadjusted roll gaps."
        })

        freshness_status, lag_days = self._assess_freshness(clean_df.index[-1])

        return MarketDataPayload(
            ticker=ticker_sym,
            df=clean_df,
            source="Yahoo Finance API (EOD Archive)",
            retrieval_timestamp=datetime.now(timezone.utc),
            start_date=s_str,
            end_date=e_str,
            row_count=len(clean_df),
            validation=val_res,
            dataset_fingerprint=val_res.fingerprint_sha256,
            data_freshness_status=freshness_status,
            calendar_lag_days=lag_days,
            instrument_type=catalog_entry.get("instrument_type", "Standard Security"),
            corporate_action_notes=catalog_entry.get("corporate_actions", "Standard corporate action rules apply."),
            futures_rollover_notes=catalog_entry.get("futures_rollover_notes", "Not applicable."),
            is_realtime=False,
            metadata=catalog_entry
        )

    def load_csv(
        self,
        file_or_path: Union[str, io.BytesIO, io.StringIO],
        ticker_name: str = "CUSTOM_CSV",
        strict_ohlc: bool = False
    ) -> MarketDataPayload:
        """
        Parses and validates an uploaded or local CSV file containing OHLCV price history.
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

        freshness_status, lag_days = self._assess_freshness(clean_df.index[-1])

        return MarketDataPayload(
            ticker=ticker_name,
            df=clean_df,
            source="User Uploaded Offline CSV",
            retrieval_timestamp=datetime.now(timezone.utc),
            start_date=s_date,
            end_date=e_date,
            row_count=len(clean_df),
            validation=val_res,
            dataset_fingerprint=val_res.fingerprint_sha256,
            data_freshness_status=freshness_status,
            calendar_lag_days=lag_days,
            instrument_type="User CSV Time-Series",
            corporate_action_notes="User-supplied series. Verify if dividend reinvestment or split adjustment is embedded in prices.",
            futures_rollover_notes="Check if series reflects cash spot or stitched futures contracts.",
            is_realtime=False,
            metadata={
                "name": ticker_name,
                "asset_class": "User Uploaded Series",
                "instrument_type": "Custom CSV File",
                "exchange": "Custom File",
                "description": "Custom CSV dataset supplied by user"
            }
        )
