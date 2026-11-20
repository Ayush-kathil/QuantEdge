"""Local disk caching layer for market time-series to optimize repeated lookups."""
import os
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
from config.settings import CACHE_DIR, DEFAULT_CACHE_EXPIRY_HOURS


class MarketDataCache:
    """Manages file-based caching of historical market datasets."""

    def __init__(self, cache_dir: str = CACHE_DIR, expiry_hours: int = DEFAULT_CACHE_EXPIRY_HOURS):
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_cache_key(self, ticker: str, start: str, end: str, interval: str) -> str:
        raw_key = f"{ticker.upper()}_{start}_{end}_{interval}"
        hasher = hashlib.sha256()
        hasher.update(raw_key.encode("utf-8"))
        return hasher.hexdigest()[:16]

    def _get_paths(self, ticker: str, start: str, end: str, interval: str) -> Tuple[str, str]:
        key = self._generate_cache_key(ticker, start, end, interval)
        clean_ticker = ticker.replace("^", "").replace("=", "_").replace("/", "_")
        base_name = f"{clean_ticker}_{key}"
        data_path = os.path.join(self.cache_dir, f"{base_name}.parquet")
        meta_path = os.path.join(self.cache_dir, f"{base_name}.json")
        return data_path, meta_path

    def get(self, ticker: str, start: str, end: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Retrieves cached DataFrame if it exists and has not expired."""
        data_path, meta_path = self._get_paths(ticker, start, end, interval)

        if not os.path.exists(data_path) or not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            cached_at = datetime.fromisoformat(metadata["cached_at"])
            if datetime.now() - cached_at > timedelta(hours=self.expiry_hours):
                return None  # Expired

            df = pd.read_parquet(data_path)
            return df
        except Exception:
            # If reading cache encounters corruption, gracefully miss
            return None

    def set(self, ticker: str, start: str, end: str, df: pd.DataFrame, interval: str = "1d") -> None:
        """Persists market DataFrame and metadata to disk."""
        data_path, meta_path = self._get_paths(ticker, start, end, interval)
        try:
            df.to_parquet(data_path)
            metadata = {
                "ticker": ticker,
                "start": start,
                "end": end,
                "interval": interval,
                "cached_at": datetime.now().isoformat(),
                "rows": len(df)
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            # Logging error without crashing ingestion
            print(f"[Warning] Failed to cache data for {ticker}: {e}")

    def clear(self) -> None:
        """Clears all cached files in the cache directory."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
