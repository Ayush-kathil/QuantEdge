"""Technical indicators implementation with mathematically rigorous formulas."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Computes Simple Moving Average (SMA) over a given period.
    Formula: (1 / N) * sum(P_{t-i})
    """
    if period <= 0:
        raise ValueError(f"Period must be positive integer, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int = 20, adjust: bool = False) -> pd.Series:
    """
    Computes Exponential Moving Average (EMA).
    Smoothing factor alpha = 2 / (period + 1)
    """
    if period <= 0:
        raise ValueError(f"Period must be positive integer, got {period}")
    alpha = 2.0 / (period + 1.0)
    return series.ewm(alpha=alpha, adjust=adjust).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Relative Strength Index (RSI) using J. Welles Wilder's exact smoothing method.
    
    Formula:
        U = max(P_t - P_{t-1}, 0)
        D = max(P_{t-1} - P_t, 0)
        Smoothed U_t = alpha * U_t + (1 - alpha) * Smoothed U_{t-1}  where alpha = 1 / period
        RS = Smoothed U / Smoothed D
        RSI = 100 - (100 / (1 + RS))
    """
    if period <= 0:
        raise ValueError(f"RSI period must be positive integer, got {period}")
    if len(series) < period + 1:
        return pd.Series(np.nan, index=series.index)

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing corresponds to ewm with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid zero-division: when avg_loss is 0, RSI is 100; when avg_gain is 0, RSI is 0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Clean edge cases
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    # If both are 0 (flat prices), RSI is 50
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.where(~both_zero, 50.0)

    return rsi


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Computes Moving Average Convergence Divergence (MACD).
    
    Returns DataFrame with columns:
        - 'macd_line': Fast EMA - Slow EMA
        - 'signal_line': EMA of macd_line
        - 'histogram': macd_line - signal_line
    """
    if fast_period >= slow_period:
        raise ValueError(f"Fast period ({fast_period}) must be strictly less than slow period ({slow_period})")

    fast_ema = compute_ema(series, period=fast_period)
    slow_ema = compute_ema(series, period=slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, period=signal_period)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram
    }, index=series.index)


def compute_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Computes Average True Range (ATR) using J. Welles Wilder's formula.
    
    Formula:
        TR = max(High - Low, abs(High - Close_{prev}), abs(Low - Close_{prev}))
        ATR = Wilder's smoothing of TR over period (alpha = 1 / period)
    """
    required = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"ATR requires OHLC columns: {required}")

    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Computes Bollinger Bands.
    
    Returns DataFrame with columns:
        - 'bb_middle': SMA of Close
        - 'bb_upper': SMA + num_std * rolling_std
        - 'bb_lower': SMA - num_std * rolling_std
        - 'bb_bandwidth': (Upper - Lower) / Middle
    """
    sma = compute_sma(series, period=period)
    rolling_std = series.rolling(window=period, min_periods=period).std()

    upper = sma + (num_std * rolling_std)
    lower = sma - (num_std * rolling_std)
    bandwidth = (upper - lower) / sma

    return pd.DataFrame({
        "bb_middle": sma,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth
    }, index=series.index)
