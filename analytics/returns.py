"""Return calculations and time-series return transformations."""
from typing import Optional, Union
import numpy as np
import pandas as pd


def compute_daily_returns(
    series: pd.Series,
    method: str = "arithmetic"
) -> pd.Series:
    """
    Computes daily return series from a price series.
    
    Parameters:
        series: Price series (e.g. Close prices)
        method: 'arithmetic' for simple percentage returns (P_t / P_{t-1} - 1),
                'log' for continuously compounded returns ln(P_t / P_{t-1}).
                
    Returns:
        pd.Series with first observation NaN or dropped.
    """
    if series.empty or len(series) < 2:
        return pd.Series(dtype=float, index=series.index)

    if method == "arithmetic":
        returns = series.pct_change(fill_method=None)
    elif method == "log":
        returns = np.log(series / series.shift(1))
    else:
        raise ValueError(f"Unknown return calculation method '{method}'. Choose 'arithmetic' or 'log'.")

    return returns


def compute_cumulative_returns(
    returns: pd.Series,
    method: str = "arithmetic"
) -> pd.Series:
    """
    Computes cumulative compounding return series from daily returns.
    
    Parameters:
        returns: Daily returns series (arithmetic or log)
        method: 'arithmetic' assumes (1 + R).cumprod() - 1,
                'log' assumes exp(cumsum(r)) - 1.
                
    Returns:
        pd.Series of cumulative total return from base period (starts at 0.0).
    """
    clean_returns = returns.fillna(0.0)
    if method == "arithmetic":
        cum_ret = (1.0 + clean_returns).cumprod() - 1.0
    elif method == "log":
        cum_ret = np.exp(clean_returns.cumsum()) - 1.0
    else:
        raise ValueError(f"Unknown return method '{method}'.")

    return cum_ret


def compute_rolling_returns(
    series: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Calculates rolling multi-day arithmetic return over a given trading window.
    Formula: (P_t / P_{t-window}) - 1
    """
    if len(series) < window:
        return pd.Series(np.nan, index=series.index)
    return series.pct_change(periods=window, fill_method=None)


def compute_excess_returns(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
    compounding: str = "simple"
) -> pd.Series:
    """
    Computes daily excess returns above the annualized risk-free rate.
    
    Daily risk-free conversion:
    - simple: Rf / trading_days
    - compound: (1 + Rf)**(1/trading_days) - 1
    """
    if compounding == "simple":
        rf_daily = risk_free_rate / trading_days
    elif compounding == "compound":
        rf_daily = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    else:
        raise ValueError("compounding must be 'simple' or 'compound'.")

    return returns - rf_daily
