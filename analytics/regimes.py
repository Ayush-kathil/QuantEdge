"""Transparent statistical market regime classification and strategy performance attribution."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from analytics.volatility import compute_rolling_volatility
from analytics.indicators import compute_sma, compute_atr


@dataclass
class RegimeClassification:
    """Time-series series tagging volatility and trend states."""
    volatility_regime: pd.Series    # 'HIGH_VOL', 'NORMAL_VOL', 'LOW_VOL'
    trend_regime: pd.Series         # 'BULL_TREND', 'BEAR_TREND', 'RANGE_BOUND'
    combined_regime: pd.Series      # e.g. 'BULL_TREND__LOW_VOL'
    regime_proportions: Dict[str, float]


@dataclass
class RegimePerformanceAttribution:
    """Strategy behavior segmented by market environment."""
    volatility_attribution: pd.DataFrame
    trend_attribution: pd.DataFrame
    combined_attribution: pd.DataFrame
    insights: List[str] = field(default_factory=list)


def classify_market_regimes(
    df: pd.DataFrame,
    vol_lookback: int = 60,
    trend_ma_period: int = 50,
    adx_period: int = 14,
    high_vol_percentile: float = 70.0,
    low_vol_percentile: float = 30.0,
    trading_days: int = 252
) -> RegimeClassification:
    """
    Classifies historical market observations into transparent statistical regimes.
    
    1. Volatility Regimes (Historical Percentile of 60-day Rolling Vol):
       - HIGH_VOL: Rolling volatility >= 70th percentile of full historical distribution
       - LOW_VOL:  Rolling volatility <= 30th percentile of full historical distribution
       - NORMAL_VOL: 30th to 70th percentile
       
    2. Trend vs Range Regimes (Directional Distance from 50 SMA / ATR):
       - BULL_TREND: Close > 50 SMA and (Close - 50 SMA) >= 0.5 * ATR(14)
       - BEAR_TREND: Close < 50 SMA and (50 SMA - Close) >= 0.5 * ATR(14)
       - RANGE_BOUND: Within +/- 0.5 ATR of 50 SMA (choppy / consolidating)
    """
    clean_df = df.copy()
    c = clean_df["Close"]
    rets = c.pct_change(fill_method=None)

    # 1. Volatility Regime
    rolling_vol = compute_rolling_volatility(rets, window=vol_lookback, trading_days=trading_days)
    vol_valid = rolling_vol.dropna()

    if len(vol_valid) > 10:
        high_vol_thresh = float(np.percentile(vol_valid, high_vol_percentile))
        low_vol_thresh = float(np.percentile(vol_valid, low_vol_percentile))
    else:
        high_vol_thresh = 0.25
        low_vol_thresh = 0.12

    vol_regime = pd.Series("NORMAL_VOL", index=c.index)
    vol_regime[rolling_vol >= high_vol_thresh] = "HIGH_VOL"
    vol_regime[rolling_vol <= low_vol_thresh] = "LOW_VOL"

    # 2. Trend Regime
    ma50 = compute_sma(c, period=trend_ma_period)
    atr = compute_atr(clean_df, period=adx_period)

    trend_regime = pd.Series("RANGE_BOUND", index=c.index)
    
    # Distance in ATR units
    dist_from_ma = (c - ma50) / atr.replace(0.0, np.nan)
    
    bull_mask = (dist_from_ma >= 0.5) & (~ma50.isna())
    bear_mask = (dist_from_ma <= -0.5) & (~ma50.isna())
    
    trend_regime[bull_mask] = "BULL_TREND"
    trend_regime[bear_mask] = "BEAR_TREND"

    combined = trend_regime + "__" + vol_regime

    # Proportions
    comb_counts = combined.value_counts(normalize=True).to_dict()

    return RegimeClassification(
        volatility_regime=vol_regime,
        trend_regime=trend_regime,
        combined_regime=combined,
        regime_proportions=comb_counts
    )


def attribute_performance_by_regime(
    daily_strategy_returns: pd.Series,
    daily_benchmark_returns: pd.Series,
    regimes: RegimeClassification,
    trading_days: int = 252
) -> RegimePerformanceAttribution:
    """
    Segments strategy and benchmark returns across identified volatility and trend regimes.
    
    Reveals whether a strategy's edge is conditional on bull momentum, panic selloffs, or quiet ranges.
    """
    df_merged = pd.DataFrame({
        "Strat_Ret": daily_strategy_returns,
        "BM_Ret": daily_benchmark_returns,
        "Vol_Regime": regimes.volatility_regime,
        "Trend_Regime": regimes.trend_regime,
        "Combined": regimes.combined_regime
    }).dropna()

    def _summarize_groups(group_col: str) -> pd.DataFrame:
        records = []
        for reg, sub in df_merged.groupby(group_col):
            n_days = len(sub)
            if n_days < 5:
                continue

            r_s = sub["Strat_Ret"].to_numpy()
            r_b = sub["BM_Ret"].to_numpy()

            mean_s_ann = float(np.mean(r_s) * trading_days)
            vol_s_ann = float(np.std(r_s, ddof=1) * np.sqrt(trading_days)) if len(r_s) > 1 else 0.0
            sharpe_s = mean_s_ann / vol_s_ann if vol_s_ann > 1e-4 else 0.0

            mean_b_ann = float(np.mean(r_b) * trading_days)
            vol_b_ann = float(np.std(r_b, ddof=1) * np.sqrt(trading_days)) if len(r_b) > 1 else 0.0
            sharpe_b = mean_b_ann / vol_b_ann if vol_b_ann > 1e-4 else 0.0

            win_rate_s = float(np.mean(r_s > 0))

            records.append({
                "Regime": reg,
                "Days": n_days,
                "Pct_Time": f"{(n_days / len(df_merged))*100:.1f}%",
                "Strategy_Ann_Return": f"{mean_s_ann*100:+.2f}%",
                "Strategy_Vol": f"{vol_s_ann*100:.2f}%",
                "Strategy_Sharpe": f"{sharpe_s:.2f}",
                "Strategy_Win_Rate": f"{win_rate_s*100:.1f}%",
                "Benchmark_Ann_Return": f"{mean_b_ann*100:+.2f}%",
                "Benchmark_Sharpe": f"{sharpe_b:.2f}"
            })
        return pd.DataFrame(records)

    vol_attr = _summarize_groups("Vol_Regime")
    trend_attr = _summarize_groups("Trend_Regime")
    comb_attr = _summarize_groups("Combined")

    # Generate transparent quantitative insights
    insights = []
    # Check high vol behavior
    hv_row = vol_attr[vol_attr["Regime"] == "HIGH_VOL"]
    if not hv_row.empty:
        insights.append(f"High Volatility Regime: Strategy delivers {hv_row['Strategy_Ann_Return'].iloc[0]} annualized return (Sharpe: {hv_row['Strategy_Sharpe'].iloc[0]}) across {hv_row['Days'].iloc[0]} days.")

    range_row = trend_attr[trend_attr["Regime"] == "RANGE_BOUND"]
    if not range_row.empty:
        insights.append(f"Range-Bound Regime: Accounts for {range_row['Pct_Time'].iloc[0]} of market history; Strategy Sharpe is {range_row['Strategy_Sharpe'].iloc[0]}.")

    bull_row = trend_attr[trend_attr["Regime"] == "BULL_TREND"]
    if not bull_row.empty:
        insights.append(f"Bull Trend Regime: Strategy Sharpe is {bull_row['Strategy_Sharpe'].iloc[0]} vs Benchmark Sharpe of {bull_row['Benchmark_Sharpe'].iloc[0]}.")

    return RegimePerformanceAttribution(
        volatility_attribution=vol_attr,
        trend_attribution=trend_attr,
        combined_attribution=comb_attr,
        insights=insights
    )
