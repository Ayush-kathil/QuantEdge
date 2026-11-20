"""Statistical analysis engine for financial time-series distributions and properties."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DistributionStats:
    """Encapsulates descriptive and inferential statistics of a return series."""
    count: int
    mean_daily: float
    mean_annualized: float
    median_daily: float
    std_daily: float
    volatility_annualized: float
    variance_daily: float
    skewness: float
    excess_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal_5pct: bool
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    autocorr_lags: Dict[int, float] = field(default_factory=dict)
    interpretations: List[str] = field(default_factory=list)


def compute_distribution_statistics(
    returns: pd.Series,
    trading_days: int = 252,
    max_lags: int = 10
) -> DistributionStats:
    """
    Computes rigorous statistical distribution metrics for a return time-series.
    
    Parameters:
        returns: Daily return series
        trading_days: Trading day convention (default 252)
        max_lags: Number of lags for autocorrelation analysis
        
    Returns:
        DistributionStats dataclass
    """
    clean_rets = returns.dropna()
    n = len(clean_rets)

    if n < 4:
        raise ValueError(f"Insufficient observations to compute statistical moments (n={n}, minimum required is 4)")

    arr = clean_rets.to_numpy(dtype=float)

    mean_d = float(np.mean(arr))
    mean_ann = float(mean_d * trading_days)
    median_d = float(np.median(arr))
    std_d = float(np.std(arr, ddof=1))
    vol_ann = float(std_d * np.sqrt(trading_days))
    var_d = float(std_d ** 2)

    # Skewness and excess kurtosis (Fisher=True means normal dist has kurtosis = 0)
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False))

    # Jarque-Bera test of normality
    jb_stat, jb_p = stats.jarque_bera(arr)
    is_norm = bool(jb_p > 0.05)

    # Historical Value at Risk (VaR) & CVaR (Expected Shortfall)
    # VaR_alpha is the positive loss threshold at alpha confidence
    q_05 = float(np.percentile(arr, 5.0))
    q_01 = float(np.percentile(arr, 1.0))
    var_95 = float(-q_05)  # Expressed as positive loss %
    var_99 = float(-q_01)

    cvar_95_losses = arr[arr <= q_05]
    cvar_95 = float(-np.mean(cvar_95_losses)) if len(cvar_95_losses) > 0 else var_95

    cvar_99_losses = arr[arr <= q_01]
    cvar_99 = float(-np.mean(cvar_99_losses)) if len(cvar_99_losses) > 0 else var_99

    # Percentiles
    percentiles = {
        "1%": float(np.percentile(arr, 1.0)),
        "5%": q_05,
        "25% (Q1)": float(np.percentile(arr, 25.0)),
        "50% (Median)": median_d,
        "75% (Q3)": float(np.percentile(arr, 75.0)),
        "95%": float(np.percentile(arr, 95.0)),
        "99%": float(np.percentile(arr, 99.0)),
    }

    # Autocorrelation
    autocorr = {}
    for lag in range(1, max_lags + 1):
        if n > lag:
            val = float(clean_rets.autocorr(lag=lag))
            autocorr[lag] = 0.0 if np.isnan(val) else val

    # Contextual quantitative interpretations
    interpretations = []
    if kurt > 1.0:
        interpretations.append(
            f"Leptokurtic (Fat-Tailed) Distribution: Excess kurtosis of {kurt:.2f} indicates significantly higher frequency of extreme outliers compared to a Gaussian distribution. Black Swan and tail-risk hedging are essential."
        )
    elif kurt < -0.5:
        interpretations.append(
            f"Platykurtic Distribution: Excess kurtosis of {kurt:.2f} indicates lighter tails and fewer extreme shocks than a standard normal curve."
        )
    else:
        interpretations.append(
            f"Mesokurtic Distribution: Excess kurtosis of {kurt:.2f} aligns closely with Gaussian expectation."
        )

    if skew < -0.2:
        interpretations.append(
            f"Negative Asymmetry (Left-Skewed): Skewness of {skew:.2f} reflects asymmetry toward large negative drawdowns (typical equity crash risk)."
        )
    elif skew > 0.2:
        interpretations.append(
            f"Positive Asymmetry (Right-Skewed): Skewness of {skew:.2f} reflects frequent modest drops compensated by occasional explosive upside spikes."
        )
    else:
        interpretations.append(
            f"Symmetric Distribution: Skewness of {skew:.2f} indicates roughly balanced positive and negative shocks."
        )

    if not is_norm:
        interpretations.append(
            f"Rejection of Normality: Jarque-Bera test (statistic={jb_stat:.1f}, p={jb_p:.4e}) definitively rejects Gaussian normality at the 5% alpha level. Standard parametric CAPM and Gaussian VaR understate risk."
        )
    else:
        interpretations.append(
            "Empirical Gaussianity: Fails to reject the null hypothesis of normality at the 5% significance level."
        )

    lag1 = autocorr.get(1, 0.0)
    if abs(lag1) > 0.08:
        dir_str = "positive (momentum / trend persistence)" if lag1 > 0 else "negative (mean reversion)"
        interpretations.append(
            f"Short-Term Serial Correlation: Lag-1 autocorrelation of {lag1:.3f} indicates statistically noticeable {dir_str}."
        )
    else:
        interpretations.append(
            f"Weak Serial Correlation: Lag-1 autocorrelation of {lag1:.3f} indicates returns closely follow a martingale / random walk structure."
        )

    return DistributionStats(
        count=n,
        mean_daily=mean_d,
        mean_annualized=mean_ann,
        median_daily=median_d,
        std_daily=std_d,
        volatility_annualized=vol_ann,
        variance_daily=var_d,
        skewness=skew,
        excess_kurtosis=kurt,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_p),
        is_normal_5pct=is_norm,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        percentiles=percentiles,
        autocorr_lags=autocorr,
        interpretations=interpretations
    )


def compute_correlation_matrix(
    returns_dict: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Computes pairwise Pearson correlation matrix across multiple return series.
    Aligns dates automatically via inner join.
    """
    df = pd.DataFrame(returns_dict).dropna()
    if df.empty or len(df.columns) < 2:
        return pd.DataFrame()
    return df.corr(method="pearson")
