"""Execution models, slippage, commissions, and trade accounting."""
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime
import pandas as pd


@dataclass
class Trade:
    """Encapsulates a closed round-trip trade."""
    trade_id: int
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    entry_cost: float
    exit_cost: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    duration_bars: int


class ExecutionModel:
    """
    Simulates realistic market microstructure execution with slippage and commission costs.
    """

    def __init__(self, commission_bps: float = 5.0, slippage_bps: float = 2.0):
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def calculate_execution_price(self, nominal_price: float, side: Literal["BUY", "SELL"]) -> float:
        """
        Adjusts nominal price by adverse price slippage.
        Buying executes slightly higher; selling executes slightly lower.
        """
        slip_frac = self.slippage_bps / 10_000.0
        if side == "BUY":
            return nominal_price * (1.0 + slip_frac)
        elif side == "SELL":
            return nominal_price * (1.0 - slip_frac)
        else:
            raise ValueError(f"Unknown side: {side}")

    def calculate_commission(self, notional_value: float) -> float:
        """Computes trading commission based on basis points of traded notional."""
        comm_frac = self.commission_bps / 10_000.0
        return abs(notional_value) * comm_frac
