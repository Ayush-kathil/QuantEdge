"""Return calculations and time-series return transformations."""
from typing import Optional, Union
from datetime import date
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


def compute_cagr(
    start_value: float,
    end_value: float,
    num_periods: int,
    periods_per_year: int = 252,
    start_date: Optional[Union[str, pd.Timestamp, date]] = None,
    end_date: Optional[Union[str, pd.Timestamp, date]] = None,
    convention: str = "trading_days"
) -> float:
    """
    Computes Compound Annual Growth Rate (CAGR) with rigorous boundary and elapsed period handling.
    
    Formula:
        Trading days convention: CAGR = (V_T / V_0)**(periods_per_year / num_periods) - 1
        Calendar days convention: CAGR = (V_T / V_0)**(365.25 / calendar_days) - 1
        
    Parameters:
        start_value: Initial capital or starting price (V_0)
        end_value: Ending capital or ending price (V_T)
        num_periods: Number of trading bars observed (N)
        periods_per_year: Convention for trading days per year (default 252)
        start_date: Optional beginning timestamp for calendar elapsed time
        end_date: Optional ending timestamp for calendar elapsed time
        convention: 'trading_days' (standard for trading strategies) or 'calendar'
        
    Returns:
        float: Annualized compound growth rate. Handles edge cases:
               - V_T <= 0: Capital wipeout -> returns -1.0 (-100%)
               - V_0 <= 0: Invalid base capital -> returns 0.0
               - Short durations (< 15 trading days or < 0.05 years): Returns simple total return
                 to prevent compounding illusions on high-frequency noise.
    """
    if start_value <= 0:
        return 0.0
    if end_value <= 0:
        return -1.0

    tot_return = (end_value - start_value) / start_value

    # Compute elapsed years according to chosen convention
    if convention == "calendar" and start_date is not None and end_date is not None:
        try:
            d_start = pd.to_datetime(start_date)
            d_end = pd.to_datetime(end_date)
            calendar_days = (d_end - d_start).days
            n_years = calendar_days / 365.25
        except Exception:
            n_years = num_periods / float(periods_per_year)
    else:
        n_years = num_periods / float(periods_per_year)

    if n_years < 0.05 or num_periods < 15:
        return float(tot_return)

    cagr = (end_value / start_value) ** (1.0 / n_years) - 1.0
    return float(cagr)
