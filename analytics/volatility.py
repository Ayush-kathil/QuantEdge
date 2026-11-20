"""Volatility modeling and historical estimators for quantitative research."""
from typing import Optional
import numpy as np
import pandas as pd


def compute_realized_volatility(
    returns: pd.Series,
    trading_days: int = 252,
    ddof: int = 1
) -> float:
    """
    Computes annualized sample standard deviation of daily returns (Close-to-Close).
    
    Formula:
        sigma_ann = sqrt(trading_days) * std(R)
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        return 0.0
    daily_std = clean_rets.std(ddof=ddof)
    return float(daily_std * np.sqrt(trading_days))


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    trading_days: int = 252,
    ddof: int = 1
) -> pd.Series:
    """
    Computes rolling annualized sample standard deviation.
    """
    clean_rets = returns.dropna()
    rolling_std = clean_rets.rolling(window=window).std(ddof=ddof)
    return (rolling_std * np.sqrt(trading_days)).reindex(returns.index)


def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 21,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculates Parkinson volatility (High-Low volatility estimator).
    5x more efficient than standard close-to-close volatility estimator.
    
    Formula:
        sigma_P = sqrt( (1 / (4 * ln(2) * N)) * sum( (ln(H_i / L_i))^2 ) ) * sqrt(trading_days)
    """
    required = ["High", "Low"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Parkinson volatility requires columns: {required}")

    log_hl = np.log(df["High"] / df["Low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_val = (factor * log_hl.rolling(window=window).mean()) ** 0.5
    return rolling_val * np.sqrt(trading_days)


def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 21,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculates Garman-Klass volatility (OHLC volatility estimator).
    Approx. 8x more efficient than close-to-close volatility.
    
    Formula:
        sigma_GK = sqrt( (1 / N) * sum( 0.5 * (ln(H / L))^2 - (2*ln(2) - 1) * (ln(C / O))^2 ) ) * sqrt(trading_days)
    """
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Garman-Klass volatility requires columns: {required}")

    log_hl = (np.log(df["High"] / df["Low"])) ** 2
    log_co = (np.log(df["Close"] / df["Open"])) ** 2
    term = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co

    # Ensure non-negative under numerical rounding
    term_clean = term.clip(lower=0.0)
    rolling_val = (term_clean.rolling(window=window).mean()) ** 0.5
    return rolling_val * np.sqrt(trading_days)


def compute_ewma_volatility(
    returns: pd.Series,
    decay_factor: float = 0.94,
    trading_days: int = 252
) -> pd.Series:
    """
    Exponentially Weighted Moving Average (EWMA) volatility (RiskMetrics convention).
    
    Formula:
        sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
    """
    clean_rets = returns.fillna(0.0)
    squared_rets = clean_rets ** 2
    # EWMA variance with span = (2 / (1 - lambda)) - 1
    # or alpha = 1 - lambda
    alpha = 1.0 - decay_factor
    ewma_var = squared_rets.ewm(alpha=alpha, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(trading_days)
    return ewma_vol
