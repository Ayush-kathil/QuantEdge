"""Discrete-event backtesting engine with realistic microstructure simulation."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy
from backtesting.execution import Trade, ExecutionModel
from backtesting.metrics import PerformanceMetrics, calculate_performance_metrics


@dataclass
class BacktestResult:
    """Comprehensive backtest outcome container."""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series
    trades: List[Trade]
    metrics: PerformanceMetrics
    benchmark_equity_curve: pd.Series
    benchmark_daily_returns: pd.Series
    benchmark_metrics: PerformanceMetrics
    detailed_history: pd.DataFrame
    commission_bps: float
    slippage_bps: float
    execution_mode: str


class BacktestEngine:
    """
    Simulation engine enforcing realistic execution timing and cost drag.
    
    Execution Paradigm:
        - Signal generated at close of bar t (using prices <= t).
        - In 'next_open' mode (default), trade executes at Open of bar t+1 with slippage and commission.
        - In 'next_close' mode, trade executes at Close of bar t+1.
        - Look-ahead bias is strictly prevented: bar t's signal never executes at bar t's price.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0,
        execution_mode: Literal["next_open", "next_close"] = "next_open",
        risk_free_rate: float = 0.04,
        trading_days: int = 252
    ):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.execution_mode = execution_mode
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.execution_model = ExecutionModel(commission_bps=commission_bps, slippage_bps=slippage_bps)

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "ASSET"
    ) -> BacktestResult:
        """
        Executes backtest for a strategy over historical OHLCV market data.
        
        Parameters:
            df: Historical price DataFrame (must contain Open, High, Low, Close)
            strategy: Subclass of BaseStrategy
            symbol: Ticker label
            
        Returns:
            BacktestResult
        """
        # 1. Generate timestamped signals
        signals_df = strategy.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError(f"Strategy {strategy.name} failed to return 'signal' column.")

        n_bars = len(signals_df)
        dates = signals_df.index

        # Arrays to record portfolio state
        cash_arr = np.zeros(n_bars, dtype=float)
        shares_arr = np.zeros(n_bars, dtype=float)
        portfolio_val_arr = np.zeros(n_bars, dtype=float)
        pos_target_arr = signals_df["signal"].to_numpy(dtype=int)
        actual_pos_arr = np.zeros(n_bars, dtype=int)

        current_cash = float(self.initial_capital)
        current_shares = 0.0
        current_state = 0  # 0: Flat, 1: Long, -1: Short

        closed_trades: List[Trade] = []
        trade_counter = 1

        # Track active trade entry
        open_trade_entry = None  # (entry_date, entry_price, shares, entry_cost, direction, entry_bar_idx)

        # Simulation Loop
        for i in range(n_bars):
            # Price used for end-of-bar portfolio valuation
            c_price = float(signals_df["Close"].iloc[i])

            # Check if an execution order was queued from previous bar's signal
            if i > 0:
                prev_target_signal = pos_target_arr[i - 1]

                # Determine if a position reallocation or flip is required
                if prev_target_signal != current_state:
                    # Determine execution fill price
                    if self.execution_mode == "next_open" and "Open" in signals_df.columns:
                        base_exec_price = float(signals_df["Open"].iloc[i])
                    else:
                        base_exec_price = c_price

                    # 1. Close current position if open
                    if current_state != 0 and open_trade_entry is not None:
                        entry_date, entry_price, entry_shares, entry_cost, direction, entry_idx = open_trade_entry
                        exit_date = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])

                        # Exit trade execution price
                        exit_side = "SELL" if direction == "LONG" else "BUY"
                        fill_exit_price = self.execution_model.calculate_execution_price(base_exec_price, exit_side)
                        notional_exit = abs(entry_shares) * fill_exit_price
                        exit_cost = self.execution_model.calculate_commission(notional_exit)

                        if direction == "LONG":
                            gross_pnl = entry_shares * (fill_exit_price - entry_price)
                            proceeds = (entry_shares * fill_exit_price) - exit_cost
                            current_cash += proceeds
                        else:  # SHORT
                            gross_pnl = abs(entry_shares) * (entry_price - fill_exit_price)
                            # Repay borrowed short value
                            current_cash += gross_pnl - exit_cost

                        net_pnl = gross_pnl - entry_cost - exit_cost
                        invested_capital = abs(entry_shares) * entry_price
                        ret_pct = net_pnl / invested_capital if invested_capital > 0 else 0.0

                        closed_trades.append(Trade(
                            trade_id=trade_counter,
                            symbol=symbol,
                            direction=direction,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            entry_price=entry_price,
                            exit_price=fill_exit_price,
                            shares=abs(entry_shares),
                            entry_cost=entry_cost,
                            exit_cost=exit_cost,
                            gross_pnl=gross_pnl,
                            net_pnl=net_pnl,
                            return_pct=ret_pct,
                            duration_bars=i - entry_idx
                        ))
                        trade_counter += 1
                        current_shares = 0.0
                        current_state = 0
                        open_trade_entry = None

                    # 2. Open new position if target signal != 0
                    if prev_target_signal != 0:
                        direction = "LONG" if prev_target_signal == 1 else "SHORT"
                        entry_side = "BUY" if direction == "LONG" else "SELL"
                        fill_entry_price = self.execution_model.calculate_execution_price(base_exec_price, entry_side)

                        # Sizing: Allocate available capital
                        avail_equity = current_cash
                        if avail_equity > 0 and fill_entry_price > 0:
                            # Reserve 0.2% buffer for fees and fractional shares
                            alloc_capital = avail_equity * 0.998
                            target_shares = alloc_capital / fill_entry_price
                            notional_entry = target_shares * fill_entry_price
                            entry_cost = self.execution_model.calculate_commission(notional_entry)

                            if direction == "LONG":
                                current_cash -= (notional_entry + entry_cost)
                                current_shares = target_shares
                            else:  # SHORT
                                current_cash -= entry_cost  # Short margin
                                current_shares = -target_shares

                            current_state = prev_target_signal
                            entry_date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])
                            open_trade_entry = (entry_date_str, fill_entry_price, current_shares, entry_cost, direction, i)

            # Record end of bar state
            actual_pos_arr[i] = current_state
            shares_arr[i] = current_shares
            cash_arr[i] = current_cash

            # Total mark-to-market portfolio value
            if current_state == 1:
                mv = current_cash + (current_shares * c_price)
            elif current_state == -1 and open_trade_entry is not None:
                # Short mark to market: cash + pnl
                _, entry_p, s_shares, _, _, _ = open_trade_entry
                short_pnl = abs(s_shares) * (entry_p - c_price)
                mv = current_cash + (abs(s_shares) * entry_p) + short_pnl
            else:
                mv = current_cash

            portfolio_val_arr[i] = mv

        # Construct Series
        equity_curve = pd.Series(portfolio_val_arr, index=dates, name="Strategy_Equity")
        daily_returns = equity_curve.pct_change(fill_method=None).fillna(0.0)
        positions = pd.Series(actual_pos_arr, index=dates, name="Position")

        # Benchmark: Buy & Hold the asset with exact same initial capital
        first_open = float(signals_df["Open"].iloc[0]) if "Open" in signals_df.columns else float(signals_df["Close"].iloc[0])
        bm_exec_price = self.execution_model.calculate_execution_price(first_open, "BUY")
        bm_shares = (self.initial_capital * 0.998) / bm_exec_price
        bm_cost = self.execution_model.calculate_commission(bm_shares * bm_exec_price)
        bm_cash = self.initial_capital - (bm_shares * bm_exec_price) - bm_cost

        bm_equity = bm_cash + (bm_shares * signals_df["Close"])
        bm_returns = bm_equity.pct_change(fill_method=None).fillna(0.0)

        # Benchmark Trade
        bm_trade = [Trade(
            trade_id=1,
            symbol=symbol,
            direction="LONG",
            entry_date=str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0]),
            exit_date=str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
            entry_price=bm_exec_price,
            exit_price=float(signals_df["Close"].iloc[-1]),
            shares=bm_shares,
            entry_cost=bm_cost,
            exit_cost=0.0,
            gross_pnl=bm_shares * (float(signals_df["Close"].iloc[-1]) - bm_exec_price),
            net_pnl=(bm_shares * (float(signals_df["Close"].iloc[-1]) - bm_exec_price)) - bm_cost,
            return_pct=(float(signals_df["Close"].iloc[-1]) - bm_exec_price) / bm_exec_price,
            duration_bars=n_bars
        )]

        # Calculate metrics
        strat_metrics = calculate_performance_metrics(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=closed_trades,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
            position_series=positions
        )

        bm_metrics = calculate_performance_metrics(
            equity_curve=bm_equity,
            daily_returns=bm_returns,
            trades=bm_trade,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
            position_series=pd.Series(1, index=dates)
        )

        detailed_history = pd.DataFrame({
            "Close": signals_df["Close"],
            "Signal": signals_df["signal"],
            "Position": positions,
            "Shares": shares_arr,
            "Cash": cash_arr,
            "Strategy_Equity": equity_curve,
            "Benchmark_Equity": bm_equity,
            "Strategy_Daily_Return": daily_returns,
            "Benchmark_Daily_Return": bm_returns
        }, index=dates)

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0]),
            end_date=str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            positions=positions,
            trades=closed_trades,
            metrics=strat_metrics,
            benchmark_equity_curve=bm_equity,
            benchmark_daily_returns=bm_returns,
            benchmark_metrics=bm_metrics,
            detailed_history=detailed_history,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            execution_mode=self.execution_mode
        )
