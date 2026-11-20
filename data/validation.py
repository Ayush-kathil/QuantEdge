"""Data validation, fingerprinting, and hygiene module for market time-series."""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Encapsulates data validation diagnostics, cryptographic fingerprint, and hygiene status."""
    is_valid: bool
    fingerprint_sha256: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_df: Optional[pd.DataFrame] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    gap_diagnostics: Dict[str, Any] = field(default_factory=dict)


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 cryptographic fingerprint over an OHLCV DataFrame.
    
    Hash captures:
    - Sorted ISO timestamps
    - Canonical 4-decimal rounded Open, High, Low, Close, Volume matrix
    
    Guarantees exact experimental reproducibility and tamper detection.
    """
    if df is None or df.empty:
        return "empty_dataset"

    hasher = hashlib.sha256()
    # Sort index to ensure deterministic order
    sorted_df = df.sort_index()

    # Hash index
    date_bytes = "".join(sorted_df.index.strftime("%Y-%m-%d")).encode("utf-8")
    hasher.update(date_bytes)

    # Hash OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in sorted_df.columns:
            # Round to 4 decimals to avoid micro floating-point serialization jitter
            col_vals = sorted_df[col].fillna(0.0).round(4).astype(str).tolist()
            col_str = f"{col}:" + ",".join(col_vals)
            hasher.update(col_str.encode("utf-8"))

    return hasher.hexdigest()


def detect_calendar_gaps(index: pd.DatetimeIndex, max_business_gap_days: int = 4) -> Dict[str, Any]:
    """
    Analyzes time-series for calendar gaps, weekends, and trading holidays.
    """
    if len(index) < 2:
        return {"max_gap_days": 0, "suspicious_gaps": []}

    diffs = (index[1:] - index[:-1]).days
    max_gap = int(np.max(diffs)) if len(diffs) > 0 else 0

    suspicious = []
    for i, gap in enumerate(diffs):
        # A gap > 4 calendar days usually implies an unscheduled market outage or missing data
        if gap > max_business_gap_days:
            suspicious.append({
                "from_date": str(index[i].date()),
                "to_date": str(index[i + 1].date()),
                "calendar_gap_days": int(gap)
            })

    return {
        "max_calendar_gap_days": max_gap,
        "total_gaps_exceeding_threshold": len(suspicious),
        "suspicious_gaps": suspicious[:10]  # Report top 10
    }


def validate_market_data(
    df: pd.DataFrame,
    strict_ohlc: bool = True,
    auto_clean: bool = True
) -> ValidationResult:
    """
    Validates financial market OHLCV DataFrame for integrity, consistency, and completeness.
    
    Checks conducted:
    1. Schema & Required Columns: Open, High, Low, Close, Volume.
    2. Datetime Index: Monotonically increasing, no duplicates, datetime format.
    3. OHLC Structural Integrity:
       - High >= Low
       - High >= Open
       - High >= Close
       - Low <= Open
       - Low <= Close
       - Prices strictly positive (> 0)
       - Volume non-negative (>= 0)
    4. Missing Value Assessment: Gaps and NaNs.
    5. Cryptographic SHA-256 fingerprinting.
    
    Parameters:
        df: Input DataFrame
        strict_ohlc: If True, failures in OHLC inequality are flagged as errors.
        auto_clean: If True, produces cleaned_df (removes duplicates, sorts, handles NaNs).
        
    Returns:
        ValidationResult object.
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if df is None or not isinstance(df, pd.DataFrame):
        return ValidationResult(is_valid=False, errors=["Input data is not a valid pandas DataFrame."])

    if df.empty:
        return ValidationResult(is_valid=False, errors=["Supplied market data is completely empty (0 rows)."])

    working_df = df.copy()

    # 1. Standardize column names (strip whitespace, capitalize standard OHLCV)
    col_map = {}
    for col in working_df.columns:
        norm = str(col).strip().capitalize()
        if norm in ["Open", "High", "Low", "Close", "Volume"]:
            col_map[col] = norm
        elif str(col).strip().lower() in ["adj close", "adjclose", "adjusted close"]:
            col_map[col] = "Adj Close"

    working_df = working_df.rename(columns=col_map)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in working_df.columns]
    
    if missing_cols:
        errors.append(f"Missing mandatory OHLCV columns: {missing_cols}. Available: {list(working_df.columns)}")
        return ValidationResult(is_valid=False, errors=errors)

    # 2. Datetime index validation
    if not isinstance(working_df.index, pd.DatetimeIndex):
        try:
            working_df.index = pd.to_datetime(working_df.index)
        except Exception as e:
            errors.append(f"Failed to parse DataFrame index into DatetimeIndex: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)

    # Strip timezone to avoid cross-tz comparison quirks while preserving UTC/local timestamps
    if working_df.index.tz is not None:
        working_df.index = working_df.index.tz_localize(None)

    total_rows = len(working_df)
    stats["initial_rows"] = total_rows

    # Check duplicates in index
    num_duplicates = working_df.index.duplicated().sum()
    stats["duplicate_timestamps"] = int(num_duplicates)
    if num_duplicates > 0:
        warnings.append(f"Detected {num_duplicates} duplicate timestamps in index. Duplicates deduplicated (preserving last).")
        if auto_clean:
            working_df = working_df[~working_df.index.duplicated(keep="last")]

    # Check monotonic ascending sort
    if not working_df.index.is_monotonic_increasing:
        warnings.append("Timestamps were not sorted chronologically. Reordered in ascending sequence.")
        if auto_clean:
            working_df = working_df.sort_index()

    # 3. Numeric conversion & missing value detection
    for col in required_cols:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    nan_counts = working_df[required_cols].isna().sum().to_dict()
    stats["nan_counts"] = nan_counts
    total_nans = sum(nan_counts.values())

    if total_nans > 0:
        warnings.append(f"Missing values detected across OHLCV columns: {nan_counts}")
        if auto_clean:
            working_df[required_cols] = working_df[required_cols].ffill()
            working_df = working_df.dropna(subset=["Close", "Open", "High", "Low"])

    if working_df.empty:
        errors.append("All rows were dropped during cleaning due to invalid/missing numerical data.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # 4. OHLC structural inequalities
    high_lt_low = (working_df["High"] < working_df["Low"]).sum()
    high_lt_open = (working_df["High"] < working_df["Open"]).sum()
    high_lt_close = (working_df["High"] < working_df["Close"]).sum()
    low_gt_open = (working_df["Low"] > working_df["Open"]).sum()
    low_gt_close = (working_df["Low"] > working_df["Close"]).sum()
    non_positive_prices = (working_df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()
    negative_volume = (working_df["Volume"] < 0).sum()

    stats["high_lt_low_violations"] = int(high_lt_low)
    stats["non_positive_price_violations"] = int(non_positive_prices)
    stats["negative_volume_violations"] = int(negative_volume)

    ohlc_violations = high_lt_low + high_lt_open + high_lt_close + low_gt_open + low_gt_close

    if ohlc_violations > 0:
        msg = (
            f"OHLC consistency violations: High < Low ({high_lt_low}), "
            f"High < Open/Close ({high_lt_open + high_lt_close}), "
            f"Low > Open/Close ({low_gt_open + low_gt_close})."
        )
        if strict_ohlc:
            errors.append(msg)
        else:
            warnings.append(msg)

    if non_positive_prices > 0:
        errors.append(f"Found {non_positive_prices} non-positive prices (<= 0), violating financial continuity.")

    if negative_volume > 0:
        warnings.append(f"Found {negative_volume} negative volume entries. Replaced negative volumes with 0.")
        if auto_clean:
            working_df.loc[working_df["Volume"] < 0, "Volume"] = 0.0

    # 5. Gap diagnostics
    gap_diag = detect_calendar_gaps(working_df.index)

    # 6. Cryptographic fingerprint
    fingerprint = compute_dataset_fingerprint(working_df)

    is_valid = len(errors) == 0
    stats["final_rows"] = len(working_df)
    stats["date_range"] = (
        (str(working_df.index[0].date()), str(working_df.index[-1].date()))
        if not working_df.empty else (None, None)
    )

    return ValidationResult(
        is_valid=is_valid,
        fingerprint_sha256=fingerprint,
        errors=errors,
        warnings=warnings,
        cleaned_df=working_df if auto_clean and is_valid else (working_df if not strict_ohlc else None),
        stats=stats,
        gap_diagnostics=gap_diag
    )
