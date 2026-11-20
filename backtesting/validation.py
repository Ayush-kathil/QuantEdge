"""Advanced out-of-sample validation, walk-forward analysis, parameter sensitivity, and friction auditing."""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy
from backtesting.engine import BacktestEngine, BacktestResult


@dataclass
class ValidationReport:
    """Out-of-sample quantitative evaluation report for a single train/test split."""
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


@dataclass
class WalkForwardFoldResult:
    """Represents performance across one chronological walk-forward fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_cagr: float
    out_of_sample_cagr: float
    in_sample_mdd: float
    out_of_sample_mdd: float
    out_of_sample_trades: int
    efficiency_ratio: float  # OOS Sharpe / IS Sharpe


@dataclass
class WalkForwardReport:
    """Multi-window walk-forward validation diagnostics."""
    folds: List[WalkForwardFoldResult]
    mean_is_sharpe: float
    mean_oos_sharpe: float
    walk_forward_efficiency: float  # Aggregate OOS Sharpe / IS Sharpe
    consistency_rate: float        # Fraction of folds with positive OOS Sharpe
    stitched_oos_equity: pd.Series
    summary_df: pd.DataFrame
    robustness_rating: str
    warnings: List[str] = field(default_factory=list)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits financial time-series strictly chronologically without data shuffling.
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
        "Metric": ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Trades", "Beta", "Alpha (Ann.)"],
        "In-Sample (Train)": [
            f"{is_res.metrics.total_return*100:.2f}%",
            f"{is_res.metrics.cagr*100:.2f}%",
            f"{is_res.metrics.sharpe_ratio:.2f}",
            f"{is_res.metrics.sortino_ratio:.2f}",
            f"{is_res.metrics.max_drawdown*100:.2f}%",
            f"{is_res.metrics.win_rate*100:.1f}%",
            f"{is_res.metrics.total_trades}",
            f"{is_res.metrics.beta:.2f}",
            f"{is_res.metrics.alpha_annualized*100:.2f}%"
        ],
        "Out-of-Sample (Test)": [
            f"{oos_res.metrics.total_return*100:.2f}%",
            f"{oos_res.metrics.cagr*100:.2f}%",
            f"{oos_res.metrics.sharpe_ratio:.2f}",
            f"{oos_res.metrics.sortino_ratio:.2f}",
            f"{oos_res.metrics.max_drawdown*100:.2f}%",
            f"{oos_res.metrics.win_rate*100:.1f}%",
            f"{oos_res.metrics.total_trades}",
            f"{oos_res.metrics.beta:.2f}",
            f"{oos_res.metrics.alpha_annualized*100:.2f}%"
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


def run_walk_forward_validation(
    df: pd.DataFrame,
    strategy_factory: Callable[[], BaseStrategy],
    engine: BacktestEngine,
    n_splits: int = 4,
    train_ratio: float = 0.65,
    symbol: str = "ASSET"
) -> WalkForwardReport:
    """
    Executes multi-window chronological walk-forward validation across expanding or rolling splits.
    
    Prevents look-ahead bias and tests structural regime resilience across multiple independent windows.
    """
    n_total = len(df)
    if n_total < 120:
        raise ValueError(f"Insufficient bars for {n_splits}-fold walk-forward validation (n={n_total}, minimum required is 120)")

    fold_size = n_total // n_splits
    folds: List[WalkForwardFoldResult] = []
    oos_equity_pieces: List[pd.Series] = []
    current_capital = engine.initial_capital

    for k in range(n_splits):
        # In-Sample window expands from start or rolls
        train_end_idx = int((k + 1) * fold_size * train_ratio)
        test_start_idx = train_end_idx
        test_end_idx = (k + 1) * fold_size if (k < n_splits - 1) else n_total

        train_slice = df.iloc[:train_end_idx]
        test_slice = df.iloc[test_start_idx:test_end_idx]

        if len(train_slice) < 30 or len(test_slice) < 15:
            continue

        strat_is = strategy_factory()
        strat_oos = strategy_factory()

        fold_engine_is = BacktestEngine(
            initial_capital=current_capital,
            commission_bps=engine.commission_bps,
            slippage_bps=engine.slippage_bps,
            execution_mode=engine.execution_mode,
            risk_free_rate=engine.risk_free_rate,
            trading_days=engine.trading_days
        )
        fold_engine_oos = BacktestEngine(
            initial_capital=current_capital,
            commission_bps=engine.commission_bps,
            slippage_bps=engine.slippage_bps,
            execution_mode=engine.execution_mode,
            risk_free_rate=engine.risk_free_rate,
            trading_days=engine.trading_days
        )

        res_is = fold_engine_is.run(train_slice, strat_is, symbol=f"{symbol}_F{k+1}_IS")
        res_oos = fold_engine_oos.run(test_slice, strat_oos, symbol=f"{symbol}_F{k+1}_OOS")

        is_sh = res_is.metrics.sharpe_ratio
        oos_sh = res_oos.metrics.sharpe_ratio
        eff = oos_sh / is_sh if abs(is_sh) > 1e-4 else (1.0 if oos_sh > 0 else 0.0)

        folds.append(WalkForwardFoldResult(
            fold_idx=k + 1,
            train_start=str(train_slice.index[0].date()),
            train_end=str(train_slice.index[-1].date()),
            test_start=str(test_slice.index[0].date()),
            test_end=str(test_slice.index[-1].date()),
            in_sample_sharpe=is_sh,
            out_of_sample_sharpe=oos_sh,
            in_sample_cagr=res_is.metrics.cagr,
            out_of_sample_cagr=res_oos.metrics.cagr,
            in_sample_mdd=res_is.metrics.max_drawdown,
            out_of_sample_mdd=res_oos.metrics.max_drawdown,
            out_of_sample_trades=res_oos.metrics.total_trades,
            efficiency_ratio=eff
        ))

        oos_equity_pieces.append(res_oos.equity_curve)
        current_capital = float(res_oos.equity_curve.iloc[-1])

    if not folds:
        raise ValueError("Walk forward partitioning failed to generate viable folds.")

    # Aggregate metrics
    mean_is = float(np.mean([f.in_sample_sharpe for f in folds]))
    mean_oos = float(np.mean([f.out_of_sample_sharpe for f in folds]))
    wfe = mean_oos / mean_is if abs(mean_is) > 1e-4 else 0.0
    pos_folds = sum(1 for f in folds if f.out_of_sample_sharpe > 0)
    consistency = pos_folds / float(len(folds))

    # Stitched OOS equity
    stitched_curve = pd.concat(oos_equity_pieces).sort_index()
    # Deduplicate index if any boundaries overlap
    stitched_curve = stitched_curve[~stitched_curve.index.duplicated(keep="last")]

    # Robustness rating
    warnings = []
    if wfe >= 0.7 and consistency >= 0.75:
        rating = "Robust (Institutional Grade WFE > 70%)"
    elif wfe >= 0.4 and consistency >= 0.50:
        rating = "Moderate (Acceptable Persistence with Volatility Degradation)"
    else:
        rating = "Brittle / Overfit Suspected (WFE < 40% or Negative Consistency)"
        warnings.append("Walk-forward efficiency collapsed below 40%; strategy parameters likely overfit to in-sample noise.")

    rows = []
    for f in folds:
        rows.append({
            "Fold": f"Fold #{f.fold_idx}",
            "IS Window": f"{f.train_start} → {f.train_end}",
            "OOS Window": f"{f.test_start} → {f.test_end}",
            "IS Sharpe": f"{f.in_sample_sharpe:.2f}",
            "OOS Sharpe": f"{f.out_of_sample_sharpe:.2f}",
            "IS CAGR": f"{f.in_sample_cagr*100:.1f}%",
            "OOS CAGR": f"{f.out_of_sample_cagr*100:.1f}%",
            "OOS Max DD": f"-{f.out_of_sample_mdd*100:.1f}%",
            "WFE": f"{f.efficiency_ratio:.2f}",
            "OOS Trades": f.out_of_sample_trades
        })
    summary_df = pd.DataFrame(rows)

    return WalkForwardReport(
        folds=folds,
        mean_is_sharpe=mean_is,
        mean_oos_sharpe=mean_oos,
        walk_forward_efficiency=wfe,
        consistency_rate=consistency,
        stitched_oos_equity=stitched_curve,
        summary_df=summary_df,
        robustness_rating=rating,
        warnings=warnings
    )


def run_parameter_sensitivity_scan(
    df: pd.DataFrame,
    strategy_builder: Callable[[Dict[str, Any]], BaseStrategy],
    engine: BacktestEngine,
    param_grid: List[Dict[str, Any]],
    symbol: str = "ASSET"
) -> pd.DataFrame:
    """
    Performs systematic parameter sensitivity scan over a grid of parameters.
    
    Audits whether optimal strategy parameters reside on a wide, stable plateau
    or a narrow, unstable overfit peak.
    """
    records = []
    for params in param_grid:
        try:
            strat = strategy_builder(params)
            res = engine.run(df, strat, symbol=symbol)
            m = res.metrics
            rec = {**params}
            rec["Total_Return"] = m.total_return
            rec["CAGR"] = m.cagr
            rec["Sharpe_Ratio"] = m.sharpe_ratio
            rec["Sortino_Ratio"] = m.sortino_ratio
            rec["Max_Drawdown"] = m.max_drawdown
            rec["Win_Rate"] = m.win_rate
            rec["Total_Trades"] = m.total_trades
            rec["Profit_Factor"] = m.profit_factor if not np.isinf(m.profit_factor) else 999.0
            records.append(rec)
        except Exception:
            continue

    return pd.DataFrame(records)


def run_friction_sensitivity_analysis(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    base_engine: BacktestEngine,
    commissions_bps: List[float] = None,
    slippages_bps: List[float] = None,
    symbol: str = "ASSET"
) -> pd.DataFrame:
    """
    Tests strategy resilience across varied commission and slippage friction regimes.
    
    Identifies the breakeven execution drag threshold where strategy profitability vanishes.
    """
    if commissions_bps is None:
        commissions_bps = [0.0, 2.0, 5.0, 10.0, 15.0, 25.0]
    if slippages_bps is None:
        slippages_bps = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]

    records = []
    for comm in commissions_bps:
        for slip in slippages_bps:
            eng = BacktestEngine(
                initial_capital=base_engine.initial_capital,
                commission_bps=comm,
                slippage_bps=slip,
                execution_mode=base_engine.execution_mode,
                risk_free_rate=base_engine.risk_free_rate,
                trading_days=base_engine.trading_days
            )
            res = eng.run(df, strategy, symbol=symbol)
            m = res.metrics
            records.append({
                "Commission_bps": comm,
                "Slippage_bps": slip,
                "Total_Friction_bps": comm + slip,
                "Total_Return": m.total_return,
                "CAGR": m.cagr,
                "Sharpe_Ratio": m.sharpe_ratio,
                "Max_Drawdown": m.max_drawdown,
                "Trades": m.total_trades,
                "Ending_Capital": m.ending_capital
            })

    return pd.DataFrame(records)
