"""Risk analytics, drawdown dynamics, and downside risk metrics."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class DrawdownAnalysis:
    """Detailed drawdown analytics."""
    max_drawdown: float  # As positive fraction (e.g. 0.25 = 25% drawdown)
    max_drawdown_dollars: float
    max_drawdown_peak_date: Optional[str]
    max_drawdown_trough_date: Optional[str]
    max_drawdown_recovery_date: Optional[str]
    max_drawdown_duration_days: int
    drawdown_series: pd.Series
    underwater_series: pd.Series
    is_currently_in_drawdown: bool
    current_drawdown: float


def compute_drawdowns(equity_curve: pd.Series) -> DrawdownAnalysis:
    """
    Computes complete drawdown dynamics from a portfolio value / equity curve.
    
    Parameters:
        equity_curve: Series of portfolio equity over time.
        
    Returns:
        DrawdownAnalysis
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return DrawdownAnalysis(
            max_drawdown=0.0,
            max_drawdown_dollars=0.0,
            max_drawdown_peak_date=None,
            max_drawdown_trough_date=None,
            max_drawdown_recovery_date=None,
            max_drawdown_duration_days=0,
            drawdown_series=pd.Series(0.0, index=equity_curve.index),
            underwater_series=pd.Series(0.0, index=equity_curve.index),
            is_currently_in_drawdown=False,
            current_drawdown=0.0
        )

    running_max = equity_curve.cummax()
    drawdown_pct = (equity_curve - running_max) / running_max
    drawdown_dollars = running_max - equity_curve

    mdd_pct = float(abs(drawdown_pct.min()))
    mdd_dollars = float(drawdown_dollars.max())

    # Find peak, trough, and recovery dates for the maximum drawdown
    trough_idx = drawdown_pct.idxmin()
    peak_candidates = equity_curve.loc[:trough_idx]
    peak_idx = peak_candidates.idxmax()

    # Recovery: first date after trough where equity >= peak value
    recovery_candidates = equity_curve.loc[trough_idx:]
    peak_val = float(equity_curve.loc[peak_idx])
    recovered = recovery_candidates[recovery_candidates >= peak_val]

    recovery_idx = recovered.index[0] if not recovered.empty and recovered.index[0] != trough_idx else None

    # Duration in calendar days
    end_date = recovery_idx if recovery_idx is not None else equity_curve.index[-1]
    duration_days = (end_date - peak_idx).days if hasattr(end_date - peak_idx, "days") else int(len(drawdown_pct))

    current_dd = float(abs(drawdown_pct.iloc[-1]))
    in_dd = current_dd > 1e-4

    return DrawdownAnalysis(
        max_drawdown=mdd_pct,
        max_drawdown_dollars=mdd_dollars,
        max_drawdown_peak_date=str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx),
        max_drawdown_trough_date=str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx),
        max_drawdown_recovery_date=str(recovery_idx.date()) if (recovery_idx is not None and hasattr(recovery_idx, "date")) else (str(recovery_idx) if recovery_idx is not None else "Unrecovered"),
        max_drawdown_duration_days=max(0, duration_days),
        drawdown_series=drawdown_pct,
        underwater_series=drawdown_pct * 100.0,
        is_currently_in_drawdown=in_dd,
        current_drawdown=current_dd
    )


def compute_downside_deviation(
    returns: pd.Series,
    mar_daily: float = 0.0,
    trading_days: int = 252,
    ddof: int = 1
) -> float:
    """
    Computes annualized downside deviation below a Minimum Acceptable Return (MAR).
    
    Formula:
        sqrt( (1 / (N - ddof)) * sum( min(R_t - MAR, 0)^2 ) ) * sqrt(trading_days)
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 2:
        return 0.0

    underperformance = (clean_rets - mar_daily).clip(upper=0.0)
    ss = np.sum(underperformance ** 2)
    denom = max(1, len(clean_rets) - ddof)
    semi_variance = ss / denom
    return float(np.sqrt(semi_variance) * np.sqrt(trading_days))


def compute_parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    trading_days: int = 252
) -> float:
    """
    Computes parametric 1-day Gaussian Value at Risk (VaR).
    
    Formula:
        VaR_alpha = -(mean - z_alpha * std)
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1))
    from scipy.stats import norm
    z = norm.ppf(confidence_level)
    var = -(mu - z * sigma)
    return max(0.0, var)


def compute_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Computes historical 1-day non-parametric Value at Risk.
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    alpha = (1.0 - confidence_level) * 100.0
    q = np.percentile(clean, alpha)
    return float(max(0.0, -q))


def compute_conditional_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Computes 1-day Conditional Value at Risk (Expected Shortfall / CVaR).
    """
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    alpha = (1.0 - confidence_level) * 100.0
    cutoff = np.percentile(clean, alpha)
    tail = clean[clean <= cutoff]
    if len(tail) == 0:
        return float(max(0.0, -cutoff))
    return float(max(0.0, -tail.mean()))
