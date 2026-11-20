"""
Portfolio Research, Cross-Asset Analytics, Rolling Beta, and Optimization for QuantEdge v3.

Provides:
- Rolling CAPM Beta and Alpha vs. Benchmark
- Multi-Asset Correlation Matrix & Rolling Pairwise Correlation
- Markowitz Modern Portfolio Theory (MPT) Optimization:
    - Maximum Sharpe Ratio Portfolio
    - Minimum Variance Portfolio
    - Equal-Weight Portfolio Benchmark
    - Efficient Frontier Curve Generation
- Percentage Risk Contribution (%RC) Deconstruction
- Cross-Asset Cumulative Return Normalization
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioOptimizationResult:
    """Output of Markowitz Mean-Variance portfolio optimization."""
    asset_names: List[str]
    weights_max_sharpe: Dict[str, float]
    weights_min_var: Dict[str, float]
    weights_equal: Dict[str, float]
    max_sharpe_return: float
    max_sharpe_volatility: float
    max_sharpe_ratio: float
    min_var_return: float
    min_var_volatility: float
    min_var_sharpe_ratio: float
    equal_weight_return: float
    equal_weight_volatility: float
    equal_weight_sharpe_ratio: float
    efficient_frontier_vols: List[float]
    efficient_frontier_rets: List[float]
    risk_contributions_max_sharpe: Dict[str, float]
    risk_contributions_min_var: Dict[str, float]
    correlation_matrix: pd.DataFrame


def compute_rolling_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Computes rolling CAPM Beta, rolling Alpha, and rolling R-squared.
    
    Formula:
        Beta_t = Cov(R_s, R_b)_window / Var(R_b)_window
        Alpha_t = (Mean(R_s) - Beta_t * Mean(R_b)) * trading_days
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < window:
        empty = pd.Series(np.nan, index=aligned.index)
        return pd.DataFrame({"rolling_beta": empty, "rolling_alpha": empty, "rolling_r2": empty})

    r_s = aligned.iloc[:, 0]
    r_b = aligned.iloc[:, 1]

    rolling_cov = r_s.rolling(window=window).cov(r_b)
    rolling_var_b = r_b.rolling(window=window).var()
    rolling_var_s = r_s.rolling(window=window).var()

    rolling_beta = rolling_cov / rolling_var_b.replace(0, np.nan)
    
    rolling_mean_s = r_s.rolling(window=window).mean()
    rolling_mean_b = r_b.rolling(window=window).mean()
    rolling_alpha = (rolling_mean_s - (rolling_beta * rolling_mean_b)) * trading_days

    # R-squared = Cov(s,b)^2 / (Var(s) * Var(b))
    rolling_r2 = (rolling_cov ** 2) / (rolling_var_s * rolling_var_b).replace(0, np.nan)

    return pd.DataFrame({
        "rolling_beta": rolling_beta,
        "rolling_alpha": rolling_alpha,
        "rolling_r2": rolling_r2.clip(0.0, 1.0)
    }, index=aligned.index)


def compute_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 63
) -> pd.Series:
    """Computes rolling Pearson correlation between two return series."""
    aligned = pd.concat([series1, series2], axis=1, join="inner").dropna()
    if len(aligned) < window:
        return pd.Series(np.nan, index=aligned.index)
    return aligned.iloc[:, 0].rolling(window=window).corr(aligned.iloc[:, 1])


def compute_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Computes Percentage Risk Contribution (%RC) of each asset to total portfolio volatility.
    
    Formula:
        Sigma_p = sqrt(w^T * Sigma * w)
        MRC = (Sigma * w) / Sigma_p
        RC_i = w_i * MRC_i = w_i * (Sigma * w)_i / Sigma_p
        %RC_i = RC_i / Sigma_p = w_i * (Sigma * w)_i / Sigma_p^2
    """
    port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_var < 1e-12:
        return np.ones(len(weights)) / len(weights)
    marginal_contrib = np.dot(cov_matrix, weights)
    pct_risk_contrib = (weights * marginal_contrib) / port_var
    return pct_risk_contrib


def optimize_portfolio(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
    num_frontier_points: int = 40
) -> PortfolioOptimizationResult:
    """
    Executes Markowitz Mean-Variance Portfolio Optimization on a multi-asset DataFrame.
    
    Computes:
        - Maximum Sharpe Ratio Portfolio (Long-only, sum(w) = 1)
        - Global Minimum Variance Portfolio
        - Equal-Weight Baseline Portfolio
        - Efficient Frontier Curve
        - Decomposed Risk Contributions
    """
    clean_rets = returns_df.dropna()
    assets = list(clean_rets.columns)
    n_assets = len(assets)

    if n_assets < 2:
        raise ValueError("Portfolio optimization requires at least 2 distinct assets.")

    # Annualized expected returns and covariance
    mu_daily = clean_rets.mean().to_numpy()
    mu_ann = mu_daily * trading_days
    cov_daily = clean_rets.cov().to_numpy()
    cov_ann = cov_daily * trading_days

    corr_matrix = clean_rets.corr()

    # Base Equal-Weight Portfolio
    w_equal = np.ones(n_assets) / float(n_assets)
    ret_equal = float(np.dot(w_equal, mu_ann))
    vol_equal = float(np.sqrt(np.dot(w_equal.T, np.dot(cov_ann, w_equal))))
    sharpe_equal = (ret_equal - risk_free_rate) / vol_equal if vol_equal > 1e-6 else 0.0

    # Bounds: Long-only [0, 1]
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    # Constraint: sum(w) = 1
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

    # 1. Global Minimum Variance Portfolio
    def portfolio_variance(w: np.ndarray) -> float:
        return float(np.dot(w.T, np.dot(cov_ann, w)))

    res_min_var = minimize(
        portfolio_variance,
        w_equal,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w_min_var = res_min_var.x if res_min_var.success else w_equal
    ret_min_var = float(np.dot(w_min_var, mu_ann))
    vol_min_var = float(np.sqrt(portfolio_variance(w_min_var)))
    sharpe_min_var = (ret_min_var - risk_free_rate) / vol_min_var if vol_min_var > 1e-6 else 0.0

    # 2. Maximum Sharpe Ratio Portfolio
    def negative_sharpe(w: np.ndarray) -> float:
        r = float(np.dot(w, mu_ann))
        v = float(np.sqrt(np.dot(w.T, np.dot(cov_ann, w))))
        if v < 1e-6:
            return 0.0
        return -(r - risk_free_rate) / v

    res_max_sharpe = minimize(
        negative_sharpe,
        w_equal,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w_max_sharpe = res_max_sharpe.x if res_max_sharpe.success else w_equal
    ret_max_sharpe = float(np.dot(w_max_sharpe, mu_ann))
    vol_max_sharpe = float(np.sqrt(portfolio_variance(w_max_sharpe)))
    sharpe_max = (ret_max_sharpe - risk_free_rate) / vol_max_sharpe if vol_max_sharpe > 1e-6 else 0.0

    # 3. Efficient Frontier Curve Points
    target_returns = np.linspace(min(mu_ann.min(), ret_min_var), max(mu_ann.max(), ret_max_sharpe * 1.1), num_frontier_points)
    frontier_vols: List[float] = []
    frontier_rets: List[float] = []

    for target_r in target_returns:
        c = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_r: float(np.dot(w, mu_ann)) - tr}
        )
        res = minimize(portfolio_variance, w_min_var, method="SLSQP", bounds=bounds, constraints=c)
        if res.success:
            frontier_vols.append(float(np.sqrt(res.fun)))
            frontier_rets.append(float(target_r))

    # 4. Risk Contributions
    rc_max_sharpe = compute_risk_contribution(w_max_sharpe, cov_ann)
    rc_min_var = compute_risk_contribution(w_min_var, cov_ann)

    return PortfolioOptimizationResult(
        asset_names=assets,
        weights_max_sharpe={assets[i]: float(w_max_sharpe[i]) for i in range(n_assets)},
        weights_min_var={assets[i]: float(w_min_var[i]) for i in range(n_assets)},
        weights_equal={assets[i]: float(w_equal[i]) for i in range(n_assets)},
        max_sharpe_return=ret_max_sharpe,
        max_sharpe_volatility=vol_max_sharpe,
        max_sharpe_ratio=sharpe_max,
        min_var_return=ret_min_var,
        min_var_volatility=vol_min_var,
        min_var_sharpe_ratio=sharpe_min_var,
        equal_weight_return=ret_equal,
        equal_weight_volatility=vol_equal,
        equal_weight_sharpe_ratio=sharpe_equal,
        efficient_frontier_vols=frontier_vols,
        efficient_frontier_rets=frontier_rets,
        risk_contributions_max_sharpe={assets[i]: float(rc_max_sharpe[i]) for i in range(n_assets)},
        risk_contributions_min_var={assets[i]: float(rc_min_var[i]) for i in range(n_assets)},
        correlation_matrix=corr_matrix
    )
