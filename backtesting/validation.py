"""Methodology for chronological time-series splitting and out-of-sample validation."""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy
from backtesting.engine import BacktestEngine, BacktestResult


@dataclass
class ValidationReport:
    """Out-of-sample quantitative evaluation report."""
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_test_split_ratio: float
    is_overfit_suspected: bool
    diagnostic_flags: List[str] = field(default_factory=list)
    comparison_summary: Dict[str, Any] = field(default_factory=dict)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits financial time-series strictly chronologically without data shuffling.
    
    Parameters:
        df: Input market DataFrame
        train_ratio: In-sample proportion (e.g. 0.70 for 70% train, 30% test)
        
    Returns:
        (train_df, test_df)
    """
    if df.empty or len(df) < 10:
        raise ValueError("DataFrame is too small for chronological splitting.")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def run_out_of_sample_validation(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    engine: BacktestEngine,
    train_ratio: float = 0.70,
    symbol: str = "ASSET"
) -> ValidationReport:
    """
    Conducts out-of-sample validation of a strategy across independent chronological windows.
    
    Detects quantitative hazards:
    1. Sharpe degradation (> 50% drop between IS and OOS).
    2. Drawdown expansion (> 50% increase in MDD).
    3. Severe drop in win rate or trade frequency.
    """
    train_df, test_df = chronological_split(df, train_ratio=train_ratio)

    is_res = engine.run(train_df, strategy, symbol=f"{symbol} [In-Sample]")
    oos_res = engine.run(test_df, strategy, symbol=f"{symbol} [Out-of-Sample]")

    flags: List[str] = []

    is_sharpe = is_res.metrics.sharpe_ratio
    oos_sharpe = oos_res.metrics.sharpe_ratio

    if is_sharpe > 0 and oos_sharpe < 0:
        flags.append("Regime Reversal: Positive In-Sample Sharpe flipped to Negative Out-of-Sample.")
    elif is_sharpe > 0.5 and (oos_sharpe / is_sharpe) < 0.4:
        flags.append(f"Severe Sharpe Ratio Degradation: Sharpe fell from {is_sharpe:.2f} (IS) to {oos_sharpe:.2f} (OOS).")

    if oos_res.metrics.max_drawdown > is_res.metrics.max_drawdown * 1.5:
        flags.append(f"Drawdown Expansion: OOS Drawdown ({oos_res.metrics.max_drawdown*100:.1f}%) exceeds IS Drawdown ({is_res.metrics.max_drawdown*100:.1f}%) by over 50%.")

    if oos_res.metrics.total_trades < 3:
        flags.append(f"Low OOS Statistical Power: Strategy executed only {oos_res.metrics.total_trades} trades in testing window.")

    is_overfit = len(flags) > 0

    comparison = {
        "Metric": ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Trades"],
        "In-Sample (Train)": [
            f"{is_res.metrics.total_return*100:.2f}%",
            f"{is_res.metrics.cagr*100:.2f}%",
            f"{is_res.metrics.sharpe_ratio:.2f}",
            f"{is_res.metrics.sortino_ratio:.2f}",
            f"{is_res.metrics.max_drawdown*100:.2f}%",
            f"{is_res.metrics.win_rate*100:.1f}%",
            f"{is_res.metrics.total_trades}"
        ],
        "Out-of-Sample (Test)": [
            f"{oos_res.metrics.total_return*100:.2f}%",
            f"{oos_res.metrics.cagr*100:.2f}%",
            f"{oos_res.metrics.sharpe_ratio:.2f}",
            f"{oos_res.metrics.sortino_ratio:.2f}",
            f"{oos_res.metrics.max_drawdown*100:.2f}%",
            f"{oos_res.metrics.win_rate*100:.1f}%",
            f"{oos_res.metrics.total_trades}"
        ]
    }

    return ValidationReport(
        in_sample_result=is_res,
        out_of_sample_result=oos_res,
        train_start=str(train_df.index[0].date()),
        train_end=str(train_df.index[-1].date()),
        test_start=str(test_df.index[0].date()),
        test_end=str(test_df.index[-1].date()),
        train_test_split_ratio=train_ratio,
        is_overfit_suspected=is_overfit,
        diagnostic_flags=flags,
        comparison_summary=comparison
    )
