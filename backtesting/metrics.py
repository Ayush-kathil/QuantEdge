"""Comprehensive quantitative performance and risk metrics computation."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from analytics.volatility import compute_realized_volatility
from analytics.risk import compute_drawdowns, compute_downside_deviation
from backtesting.execution import Trade


@dataclass
class PerformanceMetrics:
    """Standardized performance and risk evaluation metrics container."""
    initial_capital: float
    ending_capital: float
    total_return: float          # e.g. 0.25 = 25%
    cagr: float                  # Compound Annual Growth Rate
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float          # As positive fraction e.g. 0.15 = 15%
    calmar_ratio: float
    win_rate: float              # Fraction of winning trades e.g. 0.55 = 55%
    profit_factor: float         # Gross profits / Gross losses
    average_trade_return: float  # Mean return per trade
    total_trades: int
    winning_trades: int
    losing_trades: int
    market_exposure: float       # Percentage of time invested
    avg_trade_duration_days: float
    max_drawdown_duration_days: int
    risk_free_rate_used: float
    trading_days_used: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics into clean displayable dictionary."""
        return {
            "Initial Capital ($)": f"${self.initial_capital:,.2f}",
            "Ending Capital ($)": f"${self.ending_capital:,.2f}",
            "Total Return": f"{self.total_return * 100.0:.2f}%",
            "CAGR": f"{self.cagr * 100.0:.2f}%" if not np.isnan(self.cagr) else "N/A",
            "Annualized Volatility": f"{self.annualized_volatility * 100.0:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}" if not np.isnan(self.sharpe_ratio) else "N/A",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}" if not np.isnan(self.sortino_ratio) else "N/A",
            "Max Drawdown": f"{self.max_drawdown * 100.0:.2f}%",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}" if not np.isnan(self.calmar_ratio) else "N/A",
            "Win Rate": f"{self.win_rate * 100.0:.1f}%",
            "Profit Factor": f"{self.profit_factor:.2f}" if not np.isinf(self.profit_factor) and not np.isnan(self.profit_factor) else ("Inf (Zero Losses)" if np.isinf(self.profit_factor) else "N/A"),
            "Total Trades": self.total_trades,
            "Win / Loss Count": f"{self.winning_trades} / {self.losing_trades}",
            "Avg Trade Return": f"{self.average_trade_return * 100.0:.2f}%",
            "Market Exposure": f"{self.market_exposure * 100.0:.1f}%",
            "Avg Trade Duration (days)": f"{self.avg_trade_duration_days:.1f}",
            "Max DD Duration (days)": f"{self.max_drawdown_duration_days} days"
        }


def calculate_performance_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trades: List[Trade],
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
    position_series: Optional[pd.Series] = None
) -> PerformanceMetrics:
    """
    Computes rigorous backtest performance metrics.
    
    Parameters:
        equity_curve: Daily portfolio valuation series
        daily_returns: Daily portfolio percentage returns series
        trades: List of closed round-trip Trade objects
        risk_free_rate: Annualized risk-free rate assumption (default 4.0%)
        trading_days: Trading day convention (default 252)
        position_series: Series of positions (+1, -1, 0) for exposure calculation
        
    Returns:
        PerformanceMetrics
    """
    notes: List[str] = []

    if equity_curve.empty or len(equity_curve) < 2:
        return PerformanceMetrics(
            initial_capital=0.0,
            ending_capital=0.0,
            total_return=0.0,
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_trade_return=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            market_exposure=0.0,
            avg_trade_duration_days=0.0,
            max_drawdown_duration_days=0,
            risk_free_rate_used=risk_free_rate,
            trading_days_used=trading_days,
            notes=["Insufficient data observations."]
        )

    init_cap = float(equity_curve.iloc[0])
    end_cap = float(equity_curve.iloc[-1])
    tot_ret = (end_cap - init_cap) / init_cap if init_cap > 0 else 0.0

    # Number of years for CAGR
    n_days = len(equity_curve)
    n_years = n_days / float(trading_days)
    if n_years > 0.05 and end_cap > 0:
        cagr = (end_cap / init_cap) ** (1.0 / n_years) - 1.0
    else:
        cagr = tot_ret

    # Annualized Volatility
    vol_ann = compute_realized_volatility(daily_returns, trading_days=trading_days)

    # Daily risk-free conversion
    rf_daily = risk_free_rate / float(trading_days)
    excess_rets = daily_returns - rf_daily

    # Sharpe Ratio
    if vol_ann > 1e-6:
        # Annualized mean excess return / Annualized volatility
        mean_excess_ann = float(excess_rets.mean() * trading_days)
        sharpe = mean_excess_ann / vol_ann
    else:
        sharpe = 0.0
        notes.append("Volatility near zero; Sharpe ratio constrained to 0.")

    # Downside deviation & Sortino
    downside_vol = compute_downside_deviation(daily_returns, mar_daily=rf_daily, trading_days=trading_days)
    if downside_vol > 1e-6:
        sortino = float(excess_rets.mean() * trading_days) / downside_vol
    else:
        sortino = 0.0 if vol_ann < 1e-6 else np.nan

    # Drawdowns
    dd_res = compute_drawdowns(equity_curve)
    mdd = dd_res.max_drawdown

    # Calmar Ratio
    if mdd > 1e-6 and not np.isnan(cagr):
        calmar = cagr / mdd
    else:
        calmar = np.nan

    # Trade-Level Statistics
    num_trades = len(trades)
    if num_trades > 0:
        pnls = [t.net_pnl for t in trades]
        trade_rets = [t.return_pct for t in trades]
        durations = [t.duration_bars for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / float(num_trades)

        gross_profits = sum(wins)
        gross_losses = abs(sum(losses))

        if gross_losses > 0:
            profit_factor = gross_profits / gross_losses
        elif gross_profits > 0:
            profit_factor = float("inf")  # Perfect win record
        else:
            profit_factor = 0.0

        avg_trade_ret = float(np.mean(trade_rets))
        avg_duration = float(np.mean(durations))
    else:
        win_count = 0
        loss_count = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_ret = 0.0
        avg_duration = 0.0
        notes.append("No completed round-trip trades executed during backtest period.")

    # Market exposure
    if position_series is not None and not position_series.empty:
        active_bars = (position_series != 0).sum()
        exposure = float(active_bars / len(position_series))
    else:
        exposure = 1.0 if num_trades > 0 else 0.0

    return PerformanceMetrics(
        initial_capital=init_cap,
        ending_capital=end_cap,
        total_return=tot_ret,
        cagr=float(cagr),
        annualized_volatility=vol_ann,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino) if not np.isnan(sortino) else 0.0,
        max_drawdown=mdd,
        calmar_ratio=float(calmar) if not np.isnan(calmar) else 0.0,
        win_rate=win_rate,
        profit_factor=profit_factor,
        average_trade_return=avg_trade_ret,
        total_trades=num_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        market_exposure=exposure,
        avg_trade_duration_days=avg_duration,
        max_drawdown_duration_days=dd_res.max_drawdown_duration_days,
        risk_free_rate_used=risk_free_rate,
        trading_days_used=trading_days,
        notes=notes
    )
