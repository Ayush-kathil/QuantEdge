"""Monte Carlo simulation engine for empirical risk-of-ruin and pathway uncertainty modeling."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from backtesting.execution import Trade


@dataclass
class MonteCarloReport:
    """Encapsulates outcomes of Monte Carlo empirical bootstrap simulation."""
    num_simulations: int
    num_trades_sampled: int
    initial_capital: float
    simulated_equity_paths: np.ndarray        # Shape: (num_simulations, num_steps + 1)
    percentile_equity_trajectories: Dict[str, np.ndarray]  # 5th, 25th, 50th, 75th, 95th
    terminal_equity_distribution: np.ndarray  # Shape: (num_simulations,)
    terminal_equity_mean: float
    terminal_equity_median: float
    terminal_equity_std: float
    terminal_percentiles: Dict[str, float]
    max_drawdown_distribution: np.ndarray    # Shape: (num_simulations,)
    mean_max_drawdown: float
    worst_case_max_drawdown: float
    p95_max_drawdown: float                  # 95th percentile worst drawdown
    probability_of_ruin_20pct: float        # Probability of experiencing >= 20% DD
    probability_of_ruin_30pct: float        # Probability of experiencing >= 30% DD
    probability_of_ruin_50pct: float        # Probability of experiencing >= 50% DD
    resampling_assumptions: List[str] = field(default_factory=list)


def run_trade_monte_carlo(
    trades: List[Trade],
    initial_capital: float = 100_000.0,
    num_simulations: int = 1_000,
    horizon_trades: Optional[int] = None,
    seed: int = 42
) -> MonteCarloReport:
    """
    Performs trade-level empirical bootstrapping with replacement.
    
    Parameters:
        trades: List of closed historical Trade objects
        initial_capital: Base starting portfolio capital
        num_simulations: Number of Monte Carlo iterations (default 1,000)
        horizon_trades: Number of trade steps per path (defaults to len(trades))
        seed: Random seed for exact reproducibility
        
    Returns:
        MonteCarloReport with complete pathway distributions and ruin probabilities.
    """
    np.random.seed(seed)
    assumptions = [
        "Resampling draws from the empirical trade return distribution with replacement.",
        "Assumes trade returns are independent and identically distributed (IID).",
        "Does not capture time-dependent market regime shifts or volatility clustering.",
        "Position sizing is held constant at 100% available equity per trade.",
        "Simulations indicate statistical dispersion under historical payoff assumptions, NOT future certainty."
    ]

    if not trades or len(trades) < 3:
        # Fallback when insufficient trades exist
        zeros = np.full(num_simulations, initial_capital)
        dummy_paths = np.full((num_simulations, 2), initial_capital)
        return MonteCarloReport(
            num_simulations=num_simulations,
            num_trades_sampled=0,
            initial_capital=initial_capital,
            simulated_equity_paths=dummy_paths,
            percentile_equity_trajectories={p: dummy_paths[0] for p in ["5th", "25th", "50th", "75th", "95th"]},
            terminal_equity_distribution=zeros,
            terminal_equity_mean=initial_capital,
            terminal_equity_median=initial_capital,
            terminal_equity_std=0.0,
            terminal_percentiles={p: initial_capital for p in ["5%", "25%", "50%", "75%", "95%"]},
            max_drawdown_distribution=np.zeros(num_simulations),
            mean_max_drawdown=0.0,
            worst_case_max_drawdown=0.0,
            p95_max_drawdown=0.0,
            probability_of_ruin_20pct=0.0,
            probability_of_ruin_30pct=0.0,
            probability_of_ruin_50pct=0.0,
            resampling_assumptions=assumptions + ["Insufficient trade history (<3 trades) for empirical resampling."]
        )

    # Extract historical trade returns as percentage fractions
    trade_rets = np.array([t.return_pct for t in trades], dtype=float)
    n_sample = len(trade_rets)
    steps = horizon_trades if (horizon_trades is not None and horizon_trades > 0) else n_sample

    # Matrix of sampled trade returns: shape (num_simulations, steps)
    sampled_indices = np.random.choice(n_sample, size=(num_simulations, steps), replace=True)
    sampled_returns = trade_rets[sampled_indices]

    # Compute equity paths: (1 + r).cumprod()
    equity_paths = np.zeros((num_simulations, steps + 1), dtype=float)
    equity_paths[:, 0] = initial_capital

    for s in range(steps):
        equity_paths[:, s + 1] = equity_paths[:, s] * (1.0 + sampled_returns[:, s])
        # Ensure non-negative equity under extreme loss
        equity_paths[:, s + 1] = np.maximum(equity_paths[:, s + 1], 0.0)

    terminal_equity = equity_paths[:, -1]

    # Calculate Max Drawdown for every single simulated path
    # running_max: cummax along axis 1
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    # Avoid zero division
    safe_max = np.where(running_max > 0, running_max, 1.0)
    drawdowns = (equity_paths - safe_max) / safe_max
    path_mdds = np.abs(np.min(drawdowns, axis=1))

    # Percentiles over time for trajectory fan charts
    percentile_trajectories = {
        "5th": np.percentile(equity_paths, 5.0, axis=0),
        "25th": np.percentile(equity_paths, 25.0, axis=0),
        "50th": np.percentile(equity_paths, 50.0, axis=0),
        "75th": np.percentile(equity_paths, 75.0, axis=0),
        "95th": np.percentile(equity_paths, 95.0, axis=0)
    }

    term_percentiles = {
        "5%": float(np.percentile(terminal_equity, 5.0)),
        "25%": float(np.percentile(terminal_equity, 25.0)),
        "50% (Median)": float(np.percentile(terminal_equity, 50.0)),
        "75%": float(np.percentile(terminal_equity, 75.0)),
        "95%": float(np.percentile(terminal_equity, 95.0))
    }

    # Probability of Ruin (exceeding designated drawdown thresholds)
    p_ruin_20 = float(np.mean(path_mdds >= 0.20))
    p_ruin_30 = float(np.mean(path_mdds >= 0.30))
    p_ruin_50 = float(np.mean(path_mdds >= 0.50))

    return MonteCarloReport(
        num_simulations=num_simulations,
        num_trades_sampled=steps,
        initial_capital=initial_capital,
        simulated_equity_paths=equity_paths,
        percentile_equity_trajectories=percentile_trajectories,
        terminal_equity_distribution=terminal_equity,
        terminal_equity_mean=float(np.mean(terminal_equity)),
        terminal_equity_median=float(np.median(terminal_equity)),
        terminal_equity_std=float(np.std(terminal_equity, ddof=1)),
        terminal_percentiles=term_percentiles,
        max_drawdown_distribution=path_mdds,
        mean_max_drawdown=float(np.mean(path_mdds)),
        worst_case_max_drawdown=float(np.max(path_mdds)),
        p95_max_drawdown=float(np.percentile(path_mdds, 95.0)),
        probability_of_ruin_20pct=p_ruin_20,
        probability_of_ruin_30pct=p_ruin_30,
        probability_of_ruin_50pct=p_ruin_50,
        resampling_assumptions=assumptions
    )


def run_daily_return_block_bootstrap(
    daily_returns: pd.Series,
    initial_capital: float = 100_000.0,
    num_simulations: int = 500,
    block_size: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Performs stationary block-bootstrapping on daily return series.
    Preserves local autocorrelation and volatility clustering patterns.
    """
    np.random.seed(seed)
    clean = daily_returns.dropna().to_numpy()
    n = len(clean)
    if n < block_size * 2:
        return np.full((num_simulations, n + 1), initial_capital)

    num_blocks = int(np.ceil(n / block_size))
    paths = np.zeros((num_simulations, n + 1), dtype=float)
    paths[:, 0] = initial_capital

    for sim in range(num_simulations):
        sampled_series = []
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            sampled_series.extend(clean[start_idx : start_idx + block_size])
        sampled_arr = np.array(sampled_series[:n])
        cum = np.cumprod(1.0 + sampled_arr) * initial_capital
        paths[sim, 1:] = np.maximum(cum, 0.0)

    return paths
