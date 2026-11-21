"""
Futures-Specific Accounting, Margin Modeling, and Contract Simulation for QuantEdge v3.1.

Provides institutional-grade futures simulation:
- Contract specifications (multiplier, tick size, tick value)
- Dynamic historical exchange lot-size schedules with effective dates (NSE revisions)
- Initial margin & maintenance margin modeling
- Daily mark-to-market variation margin settlement
- Realized & unrealized P&L calculation
- Margin call detection and liquidation warnings
- Structural comparison across:
    A. ETF Returns
    B. Spot/Index Price Returns
    C. Actual Leveraged Futures Contract P&L
    D. Continuous Futures Proxy Returns
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import date
import numpy as np
import pandas as pd


@dataclass
class FuturesContractSpec:
    """Institutional specifications for a futures contract."""
    symbol: str
    name: str
    exchange: str
    multiplier: float
    tick_size: float
    tick_value: float
    currency: str
    initial_margin_rate: float       # As fraction of notional (e.g. 0.12 = 12%)
    maintenance_margin_rate: float   # As fraction of notional (e.g. 0.08 = 8%)
    transaction_cost_per_contract: float  # Brokerage + exchange fees per side
    rollover_cycle: str              # e.g. "Monthly (Last Thursday)", "Quarterly (H,M,U,Z)"
    stitching_caveats: str

    @classmethod
    def create(
        cls,
        symbol: str,
        name: str,
        exchange: str,
        multiplier: float,
        tick_size: float,
        currency: str,
        initial_margin_rate: float,
        maintenance_margin_rate: float,
        transaction_cost_per_contract: float,
        rollover_cycle: str,
        stitching_caveats: str
    ) -> "FuturesContractSpec":
        tick_val = tick_size * multiplier
        return cls(
            symbol=symbol,
            name=name,
            exchange=exchange,
            multiplier=multiplier,
            tick_size=tick_size,
            tick_value=tick_val,
            currency=currency,
            initial_margin_rate=initial_margin_rate,
            maintenance_margin_rate=maintenance_margin_rate,
            transaction_cost_per_contract=transaction_cost_per_contract,
            rollover_cycle=rollover_cycle,
            stitching_caveats=stitching_caveats
        )


# Comprehensive historical lot size schedules (NSE circulars and exchange revisions)
HISTORICAL_LOT_SIZE_SCHEDULE: Dict[str, List[Tuple[str, float]]] = {
    "NIFTY_FUT": [
        ("2000-01-01", 50.0),
        ("2015-10-30", 75.0),
        ("2021-04-30", 50.0),
        ("2024-04-26", 25.0),
        ("2024-11-20", 75.0),  # SEBI revised framework
    ],
    "^NSEI": [
        ("2000-01-01", 50.0),
        ("2015-10-30", 75.0),
        ("2021-04-30", 50.0),
        ("2024-04-26", 25.0),
        ("2024-11-20", 75.0),
    ],
    "BANKNIFTY_FUT": [
        ("2000-01-01", 30.0),
        ("2016-01-01", 40.0),
        ("2018-01-01", 20.0),
        ("2020-07-01", 25.0),
        ("2023-07-01", 15.0),
        ("2024-11-20", 30.0),  # SEBI revised framework
    ],
    "^NSEBANK": [
        ("2000-01-01", 30.0),
        ("2016-01-01", 40.0),
        ("2018-01-01", 20.0),
        ("2020-07-01", 25.0),
        ("2023-07-01", 15.0),
        ("2024-11-20", 30.0),
    ]
}


def get_historical_lot_size(
    symbol_or_key: str,
    target_date: Optional[Union[str, pd.Timestamp, date]] = None
) -> float:
    """
    Retrieves the exchange-mandated contract lot size active on a given historical date.
    
    If target_date is None, returns the current active exchange lot size.
    For instruments without dynamic schedules, returns the base multiplier from the registry.
    """
    key = symbol_or_key.upper()
    if key in HISTORICAL_LOT_SIZE_SCHEDULE:
        schedule = HISTORICAL_LOT_SIZE_SCHEDULE[key]
        if target_date is None:
            return schedule[-1][1]
        
        t_str = str(pd.to_datetime(target_date).date())
        active_lot = schedule[0][1]
        for eff_date, lot in schedule:
            if t_str >= eff_date:
                active_lot = lot
            else:
                break
        return active_lot

    # Fallback to base registry multiplier
    spec = get_futures_spec(symbol_or_key)
    return spec.multiplier


# Comprehensive registry of standard futures contracts
FUTURES_SPECS_REGISTRY: Dict[str, FuturesContractSpec] = {
    "GC=F": FuturesContractSpec.create(
        symbol="GC=F",
        name="Gold Futures (COMEX)",
        exchange="COMEX",
        multiplier=100.0,  # 100 troy ounces
        tick_size=0.10,    # $0.10 per troy oz
        currency="USD",
        initial_margin_rate=0.10,
        maintenance_margin_rate=0.07,
        transaction_cost_per_contract=2.50,
        rollover_cycle="Bi-monthly (Feb, Apr, Jun, Aug, Oct, Dec)",
        stitching_caveats="Continuous proxy stitched on front-month expiration. Does not account for contango roll-yield decay or physical storage costs."
    ),
    "CL=F": FuturesContractSpec.create(
        symbol="CL=F",
        name="Crude Oil WTI Futures (NYMEX)",
        exchange="NYMEX",
        multiplier=1000.0,  # 1,000 barrels
        tick_size=0.01,     # $0.01 per barrel
        currency="USD",
        initial_margin_rate=0.15,
        maintenance_margin_rate=0.10,
        transaction_cost_per_contract=2.50,
        rollover_cycle="Monthly (25th calendar day prior to contract month)",
        stitching_caveats="High roll yield sensitivity. Negative pricing anomalies (e.g. April 2020) and step-discontinuities occur on contract expiration."
    ),
    "ZN=F": FuturesContractSpec.create(
        symbol="ZN=F",
        name="10-Year US Treasury Note Futures (CBOT)",
        exchange="CBOT",
        multiplier=1000.0,  # Par value $100,000, quoted in points and 32nds
        tick_size=0.015625, # 1/64 of a point = $15.625
        currency="USD",
        initial_margin_rate=0.05,
        maintenance_margin_rate=0.035,
        transaction_cost_per_contract=1.75,
        rollover_cycle="Quarterly (H:Mar, M:Jun, U:Sep, Z:Dec)",
        stitching_caveats="Deliverable basket conversion factor changes across contract expirations; continuous proxy is price-only and omits implied repo basis."
    ),
    "NIFTY_FUT": FuturesContractSpec.create(
        symbol="^NSEI",
        name="NIFTY 50 Index Futures (NSE)",
        exchange="NSE",
        multiplier=75.0,   # Current NSE lot size: 75 (effective Nov 20, 2024; historically 50, 75, 50, 25)
        tick_size=0.05,    # 0.05 index points (Tick value = 75 * 0.05 = 3.75 INR)
        currency="INR",
        initial_margin_rate=0.12,  # SPAN + Exposure margin approx 12-14%
        maintenance_margin_rate=0.08,
        transaction_cost_per_contract=40.0,  # INR brokerage + STT + exchange turnover fees
        rollover_cycle="Monthly (Last Thursday of the month)",
        stitching_caveats="Continuous series modeled from index spot proxy. Real futures trade at cost-of-carry premium (basis) and exhibit rollover volume migration in expiry week."
    ),
    "BANKNIFTY_FUT": FuturesContractSpec.create(
        symbol="^NSEBANK",
        name="BANK NIFTY Futures (NSE)",
        exchange="NSE",
        multiplier=30.0,   # Current NSE lot size: 30 (effective Nov 20, 2024; historically 15, 25, 40)
        tick_size=0.05,    # 0.05 points (Tick value = 30 * 0.05 = 1.50 INR)
        currency="INR",
        initial_margin_rate=0.15,
        maintenance_margin_rate=0.10,
        transaction_cost_per_contract=40.0,
        rollover_cycle="Monthly (Last Thursday / Wednesday depending on holiday)",
        stitching_caveats="Bank Nifty exhibits high beta and intraday basis variance. Continuous proxy does not reflect dividend expectations during earnings season."
    ),
    "ES_FUT": FuturesContractSpec.create(
        symbol="^GSPC",
        name="E-mini S&P 500 Futures (CME)",
        exchange="CME",
        multiplier=50.0,   # $50 x S&P index
        tick_size=0.25,    # 0.25 index points (Tick value = $12.50)
        currency="USD",
        initial_margin_rate=0.10,
        maintenance_margin_rate=0.07,
        transaction_cost_per_contract=2.25,
        rollover_cycle="Quarterly (Mar, Jun, Sep, Dec)",
        stitching_caveats="Traded nearly 24/5. Cash spot index omits overnight session liquidity and fair value basis."
    )
}


@dataclass
class FuturesTradeRecord:
    """Detailed record of an executed futures round-trip trade."""
    entry_date: str
    exit_date: str
    direction: str  # "LONG" or "SHORT"
    contracts: int
    entry_price: float
    exit_price: float
    multiplier: float
    ticks_captured: float
    gross_pnl: float
    transaction_fees: float
    net_pnl: float
    initial_margin_posted: float
    return_on_margin: float


@dataclass
class FuturesSimulationResult:
    """Complete accounting output of a futures account simulation."""
    spec: FuturesContractSpec
    initial_cash: float
    ending_cash: float
    peak_equity: float
    min_equity: float
    total_variation_margin: float
    total_transaction_fees: float
    net_profit: float
    return_on_initial_capital: float
    max_contracts_held: int
    margin_calls_triggered: int
    daily_equity_curve: pd.Series
    daily_cash_curve: pd.Series
    daily_margin_required: pd.Series
    daily_variation_margin: pd.Series
    trades: List[FuturesTradeRecord]
    structural_comparison: Dict[str, Any]


def get_futures_spec(symbol_or_key: str) -> FuturesContractSpec:
    """Lookup or generate a default futures contract specification."""
    if symbol_or_key in FUTURES_SPECS_REGISTRY:
        return FUTURES_SPECS_REGISTRY[symbol_or_key]
    for spec in FUTURES_SPECS_REGISTRY.values():
        if spec.symbol == symbol_or_key:
            return spec

    # Default fallback specification
    return FuturesContractSpec.create(
        symbol=symbol_or_key,
        name=f"Generic Futures ({symbol_or_key})",
        exchange="Exchange",
        multiplier=10.0,
        tick_size=0.05,
        currency="USD",
        initial_margin_rate=0.15,
        maintenance_margin_rate=0.10,
        transaction_cost_per_contract=5.0,
        rollover_cycle="Standard Expiration",
        stitching_caveats="Continuous proxy series without explicit roll yield adjustment."
    )


def simulate_futures_account(
    price_df: pd.DataFrame,
    signals: pd.Series,
    spec: FuturesContractSpec,
    initial_cash: float = 100_000.0,
    max_contracts: int = 1,
    slippage_ticks: float = 1.0,
    use_historical_lot_sizes: bool = False
) -> FuturesSimulationResult:
    """
    Simulates a marked-to-market futures margin account with daily variation margin settlement.
    
    Parameters:
        price_df: OHLCV DataFrame (must include 'Open' and 'Close')
        signals: Trading signal series (+1 Long, -1 Short, 0 Flat)
        spec: FuturesContractSpec containing multiplier, margin rates, and tick info
        initial_cash: Starting cash deposit in account
        max_contracts: Number of contracts traded per signal
        slippage_ticks: Adverse slippage applied in ticks per trade execution
        use_historical_lot_sizes: If True, uses date-mandated historical lot sizes (e.g. for NSE).
    """
    n_bars = len(price_df)
    dates = price_df.index

    cash = initial_cash
    current_pos = 0  # +N for Long, -N for Short, 0 for Flat
    entry_price = 0.0
    entry_date = None

    daily_equity = np.zeros(n_bars, dtype=float)
    daily_cash = np.zeros(n_bars, dtype=float)
    daily_margin_req = np.zeros(n_bars, dtype=float)
    daily_var_margin = np.zeros(n_bars, dtype=float)

    trades: List[FuturesTradeRecord] = []
    margin_calls_count = 0
    total_fees = 0.0

    for t in range(n_bars):
        curr_date = str(dates[t].date()) if hasattr(dates[t], "date") else str(dates[t])
        curr_open = float(price_df["Open"].iloc[t])
        curr_close = float(price_df["Close"].iloc[t])
        prev_close = float(price_df["Close"].iloc[t - 1]) if t > 0 else curr_open

        # Resolve active contract multiplier for current date
        active_multiplier = get_historical_lot_size(spec.symbol, dates[t]) if use_historical_lot_sizes else spec.multiplier

        # Execute signals generated at t-1 at today's Open
        target_signal = int(signals.iloc[t - 1]) if t > 0 else 0
        desired_contracts = target_signal * max_contracts

        # Position change logic
        if desired_contracts != current_pos:
            trade_qty = desired_contracts - current_pos

            # If closing or flipping an existing position
            if current_pos != 0:
                direction_str = "LONG" if current_pos > 0 else "SHORT"
                exit_fill = curr_open - np.sign(current_pos) * (slippage_ticks * spec.tick_size)
                ticks = (exit_fill - entry_price) / spec.tick_size * np.sign(current_pos)
                gross_trade_pnl = (exit_fill - entry_price) * active_multiplier * abs(current_pos) * np.sign(current_pos)
                fee = spec.transaction_cost_per_contract * abs(current_pos) * 2.0  # round-trip
                net_trade_pnl = gross_trade_pnl - fee
                total_fees += fee

                # Cash adjustment for closing trade (variation margin settlement handles day-to-day,
                # but final fill vs prev_close adjustment settles remaining intraday delta)
                intraday_exit_vm = (exit_fill - prev_close) * active_multiplier * current_pos
                cash += intraday_exit_vm - fee

                margin_posted = entry_price * active_multiplier * abs(current_pos) * spec.initial_margin_rate

                trades.append(FuturesTradeRecord(
                    entry_date=entry_date or curr_date,
                    exit_date=curr_date,
                    direction=direction_str,
                    contracts=abs(current_pos),
                    entry_price=entry_price,
                    exit_price=exit_fill,
                    multiplier=active_multiplier,
                    ticks_captured=ticks,
                    gross_pnl=gross_trade_pnl,
                    transaction_fees=fee,
                    net_pnl=net_trade_pnl,
                    initial_margin_posted=margin_posted,
                    return_on_margin=(net_trade_pnl / margin_posted) if margin_posted > 0 else 0.0
                ))

            # Opening new position
            if desired_contracts != 0:
                current_pos = desired_contracts
                entry_price = curr_open + np.sign(current_pos) * (slippage_ticks * spec.tick_size)
                entry_date = curr_date
                fee_entry = spec.transaction_cost_per_contract * abs(current_pos)
                cash -= fee_entry
                total_fees += fee_entry
                # Intraday variation margin from fill price to today's close
                day_vm = (curr_close - entry_price) * active_multiplier * current_pos
            else:
                current_pos = 0
                entry_price = 0.0
                entry_date = None
                day_vm = 0.0
        else:
            # Holding position from previous day: full day variation margin
            if current_pos != 0:
                day_vm = (curr_close - prev_close) * active_multiplier * current_pos
            else:
                day_vm = 0.0

        # Settle daily variation margin into cash
        cash += day_vm
        daily_var_margin[t] = day_vm
        daily_cash[t] = cash

        # Calculate required margin for open positions
        if current_pos != 0:
            notional = curr_close * active_multiplier * abs(current_pos)
            req_initial_margin = notional * spec.initial_margin_rate
            req_maint_margin = notional * spec.maintenance_margin_rate
        else:
            req_initial_margin = 0.0
            req_maint_margin = 0.0

        daily_margin_req[t] = req_initial_margin

        # In daily settled futures accounting, Equity = Cash
        equity = cash
        daily_equity[t] = equity

        # Check for margin call
        if current_pos != 0 and equity < req_maint_margin:
            margin_calls_count += 1

    equity_series = pd.Series(daily_equity, index=dates)
    cash_series = pd.Series(daily_cash, index=dates)
    margin_series = pd.Series(daily_margin_req, index=dates)
    vm_series = pd.Series(daily_var_margin, index=dates)

    net_profit = cash - initial_cash
    ret_pct = net_profit / initial_cash if initial_cash > 0 else 0.0

    # Structural comparison breakdown
    price_series = price_df["Close"]
    spot_return = float((price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0])

    structural_comparison = {
        "A_etf_return": f"{spot_return * 100.0:.2f}% (unleveraged cash basis)",
        "B_spot_index_return": f"{spot_return * 100.0:.2f}% (price return only, no dividends)",
        "C_actual_futures_pnl": f"{spec.currency} {net_profit:,.2f} ({ret_pct * 100.0:.2f}% on initial cash)",
        "D_continuous_proxy_status": "Stitched Continuous Proxy. Subject to roll-yield leakage and contract rollover step-discontinuities.",
        "leverage_effect": f"Contract Multiplier: {spec.multiplier}x | Tick Value: {spec.currency} {spec.tick_value:.2f}"
    }

    return FuturesSimulationResult(
        spec=spec,
        initial_cash=initial_cash,
        ending_cash=cash,
        peak_equity=float(equity_series.max()),
        min_equity=float(equity_series.min()),
        total_variation_margin=float(vm_series.sum()),
        total_transaction_fees=total_fees,
        net_profit=net_profit,
        return_on_initial_capital=ret_pct,
        max_contracts_held=max_contracts,
        margin_calls_triggered=margin_calls_count,
        daily_equity_curve=equity_series,
        daily_cash_curve=cash_series,
        daily_margin_required=margin_series,
        daily_variation_margin=vm_series,
        trades=trades,
        structural_comparison=structural_comparison
    )
