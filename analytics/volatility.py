"""Volatility modeling, historical estimators, and jump-robust volatility for quantitative research."""
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
        sigma_ann = sqrt(trading_days) * std(R, ddof=1)
        
    Limitations:
        Assumes discrete close-to-close sampling; ignores intraday high-low paths.
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
    """Computes rolling annualized sample standard deviation."""
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
    5x more statistically efficient than standard close-to-close estimator.
    
    Formula:
        sigma_P = sqrt( (1 / (4 * ln(2) * N)) * sum( (ln(H_i / L_i))^2 ) ) * sqrt(trading_days)
        
    Mathematical Limitations & Jump Bias:
        1. Assumes continuous trading without overnight jumps (i.e. O_t == C_{t-1}).
           In markets with significant opening gap risk (e.g. overnight earnings announcements),
           Parkinson volatility underestimates true total volatility by omitting jump variance.
        2. Assumes geometric Brownian motion with zero drift (mu = 0).
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
    Approx. 8x more statistically efficient than close-to-close estimator.
    
    Formula:
        sigma_GK = sqrt( (1 / N) * sum( 0.5 * (ln(H / L))^2 - (2*ln(2) - 1) * (ln(C / O))^2 ) ) * sqrt(trading_days)
        
    Mathematical Limitations & Jump Bias:
        1. Assumes zero opening jump: does NOT incorporate the overnight price gap (ln(O_t / C_{t-1})).
        2. Assumes zero underlying drift (Brownian motion without drift).
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


def compute_yang_zhang_volatility(
    df: pd.DataFrame,
    window: int = 21,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculates Yang-Zhang (2000) volatility estimator.
    The minimum-variance unbiased estimator that accounts for BOTH non-zero drift
    AND overnight opening price jumps (C_{t-1} -> O_t).
    
    Formula:
        sigma_YZ^2 = sigma_open^2 + k * sigma_close^2 + (1 - k) * sigma_RS^2
        k = 0.34 / (1.34 + (N + 1) / (N - 1))
    """
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Yang-Zhang volatility requires columns: {required}")

    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    log_co = np.log(df["Close"] / df["Open"])

    log_oc = np.log(df["Open"] / df["Close"].shift(1))
    log_cc = np.log(df["Close"] / df["Close"].shift(1))

    # Rogers-Satchell variance (drift-independent intraday component)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_rs = rs.rolling(window=window).mean()

    # Open jump variance
    open_var = log_oc.rolling(window=window).var()

    # Close variance
    close_var = log_co.rolling(window=window).var()

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    yz_var = open_var + k * close_var + (1.0 - k) * sigma_rs
    yz_vol = np.sqrt(yz_var.clip(lower=0.0)) * np.sqrt(trading_days)
    return yz_vol


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
    alpha = 1.0 - decay_factor
    ewma_var = squared_rets.ewm(alpha=alpha, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var) * np.sqrt(trading_days)
    return ewma_vol
