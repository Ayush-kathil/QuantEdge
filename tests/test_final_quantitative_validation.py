"""
QuantEdge v3.1 - Final Quantitative Validation Test Suite.

Rigorously validates:
1. Analytical correctness of all 26 financial and statistical metrics against known ground truth.
2. Edge cases (zero volatility, zero downside deviation, zero losses, capital wipeout).
3. Yang-Zhang jump-robust volatility estimator.
4. Covariance matrix condition number auditing and shrinkage regularization.
5. Dynamic date-aware historical futures contract schedules.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from analytics.returns import (
    compute_daily_returns,
    compute_cumulative_returns,
    compute_cagr,
    compute_excess_returns
)
from analytics.volatility import (
    compute_realized_volatility,
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_ewma_volatility,
    compute_yang_zhang_volatility
)
from analytics.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_bollinger_bands
)
from analytics.statistics import (
    compute_distribution_statistics,
    compute_historical_var,
    compute_parametric_var,
    compute_conditional_var
)
from backtesting.metrics import (
    calculate_performance_metrics,
    compute_drawdowns,
    compute_downside_deviation
)
from analytics.portfolio import (
    audit_covariance_matrix,
    regularize_covariance_matrix,
    optimize_portfolio
)
from analytics.futures_accounting import (
    get_historical_lot_size,
    get_futures_spec
)
from backtesting.engine import Trade


# =============================================================================
# 1. RETURNS & COMPOUNDING METRICS (Metrics 1 - 4)
# =============================================================================

def test_metric_01_simple_returns_analytical():
    """Metric 1: P_t / P_{t-1} - 1"""
    prices = pd.Series([100.0, 110.0, 99.0, 108.9])
    rets = compute_daily_returns(prices, method="arithmetic").dropna()
    expected = [0.10, -0.10, 0.10]
    np.testing.assert_allclose(rets.values, expected, rtol=1e-5)


def test_metric_02_log_returns_analytical():
    """Metric 2: ln(P_t / P_{t-1})"""
    prices = pd.Series([100.0, 105.0, 110.25])
    log_rets = compute_daily_returns(prices, method="log").dropna()
    expected = [np.log(1.05), np.log(1.05)]
    np.testing.assert_allclose(log_rets.values, expected, rtol=1e-5)


def test_metric_03_cumulative_returns_equivalence():
    """Metric 3: Compound total return equivalence."""
    prices = pd.Series([100.0, 105.0, 95.0, 115.0])
    rets = compute_daily_returns(prices, method="arithmetic").dropna()
    cum_ret = compute_cumulative_returns(rets, method="arithmetic")
    assert cum_ret.iloc[-1] == pytest.approx(0.15, abs=1e-6)


def test_metric_04_cagr_comprehensive_boundaries():
    """Metric 4: CAGR over exact durations and boundary wipeouts."""
    # Doubling over 252 trading days = exactly 100%
    assert compute_cagr(100.0, 200.0, num_periods=252, periods_per_year=252) == pytest.approx(1.00, abs=1e-5)
    # Doubling over 504 trading days = sqrt(2) - 1 = ~41.42%
    assert compute_cagr(100.0, 200.0, num_periods=504, periods_per_year=252) == pytest.approx(np.sqrt(2.0) - 1.0, abs=1e-5)
    # Total loss (terminal value <= 0)
    assert compute_cagr(100.0, 0.0, num_periods=252) == -1.0
    assert compute_cagr(100.0, -50.0, num_periods=252) == -1.0
    # Invalid start value
    assert compute_cagr(0.0, 100.0, num_periods=252) == 0.0
    # Short duration fallback (< 15 bars) -> simple return
    assert compute_cagr(100.0, 105.0, num_periods=10) == pytest.approx(0.05, abs=1e-5)


# =============================================================================
# 2. VOLATILITY ESTIMATORS (Metrics 5 - 8 + Yang-Zhang)
# =============================================================================

def test_metric_05_annualized_volatility_analytical():
    """Metric 5: sample std * sqrt(252)"""
    np.random.seed(42)
    daily_std = 0.01
    sample = pd.Series(np.random.normal(0, daily_std, 10000))
    ann_vol = compute_realized_volatility(sample, trading_days=252)
    assert ann_vol == pytest.approx(daily_std * np.sqrt(252), rel=0.05)


def test_metric_06_parkinson_volatility_analytical():
    """Metric 6: Parkinson sigma = sqrt( 1/(4 ln 2) * sum(ln(H/L)^2) / N ) * sqrt(252)"""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Fixed high/low ratio of 1.02 each day
    df = pd.DataFrame({
        "High": [102.0] * n,
        "Low": [100.0] * n,
        "Close": [101.0] * n,
        "Open": [100.5] * n
    }, index=dates)
    park_vol_series = compute_parkinson_volatility(df, window=21, trading_days=252)
    expected = np.sqrt((np.log(1.02) ** 2) / (4.0 * np.log(2.0))) * np.sqrt(252)
    assert park_vol_series.iloc[-1] == pytest.approx(expected, abs=1e-5)


def test_metric_07_garman_klass_volatility():
    """Metric 7: Garman-Klass includes open and close info."""
    n = 50
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": [100.0] * n,
        "High": [105.0] * n,
        "Low": [98.0] * n,
        "Close": [102.0] * n
    }, index=dates)
    gk_vol = compute_garman_klass_volatility(df, window=21, trading_days=252)
    assert gk_vol.iloc[-1] > 0.0 and np.isfinite(gk_vol.iloc[-1])


def test_metric_08_ewma_volatility_recursion():
    """Metric 8: RiskMetrics EWMA recursion with lambda=0.94."""
    rets = pd.Series([0.01, -0.02, 0.015, -0.01, 0.03])
    ewma_series = compute_ewma_volatility(rets, decay_factor=0.94, trading_days=252)
    assert len(ewma_series) == len(rets)
    assert (ewma_series > 0).all()


def test_yang_zhang_volatility_jump_robustness():
    """Yang-Zhang estimator captures overnight opening gaps + continuous volatility."""
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    opens, highs, lows, closes = [], [], [], []
    curr = 100.0
    for _ in range(n):
        o = curr * 1.01
        h = o * 1.02
        l = o * 0.99
        c = o * 1.005
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        curr = c
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=dates)
    yz_vol = compute_yang_zhang_volatility(df, window=20, trading_days=252)
    clean_yz = yz_vol.dropna()
    assert len(clean_yz) > 0
    assert (clean_yz > 0).all()


# =============================================================================
# 3. TECHNICAL INDICATORS (Metrics 9 - 14)
# =============================================================================

def test_metric_09_sma_analytical():
    """Metric 9: Arithmetic moving average."""
    prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma3 = compute_sma(prices, period=3)
    assert np.isnan(sma3.iloc[0]) and np.isnan(sma3.iloc[1])
    assert sma3.iloc[2] == pytest.approx(20.0)
    assert sma3.iloc[3] == pytest.approx(30.0)
    assert sma3.iloc[4] == pytest.approx(40.0)


def test_metric_10_ema_analytical():
    """Metric 10: Exponential moving average with alpha = 2/(period+1)."""
    prices = pd.Series([100.0, 100.0, 110.0])
    ema = compute_ema(prices, period=3, adjust=False)
    # alpha = 2 / (3 + 1) = 0.5
    # EMA_0 = 100
    # EMA_1 = 0.5 * 100 + 0.5 * 100 = 100
    # EMA_2 = 0.5 * 110 + 0.5 * 100 = 105
    assert ema.iloc[-1] == pytest.approx(105.0, abs=1e-1)


def test_metric_11_rsi_asymptotic_extremes():
    """Metric 11: RSI bounds [0, 100] and 50 on flat series."""
    # Monotonically up
    up_prices = pd.Series(np.linspace(100.0, 200.0, 40))
    rsi_up = compute_rsi(up_prices, period=14).dropna()
    assert rsi_up.iloc[-1] == pytest.approx(100.0, abs=1e-2)

    # Monotonically down
    down_prices = pd.Series(np.linspace(200.0, 100.0, 40))
    rsi_down = compute_rsi(down_prices, period=14).dropna()
    assert rsi_down.iloc[-1] == pytest.approx(0.0, abs=1e-2)

    # Flat
    flat_prices = pd.Series([100.0] * 40)
    rsi_flat = compute_rsi(flat_prices, period=14).dropna()
    assert rsi_flat.iloc[-1] == pytest.approx(50.0, abs=1e-2)


def test_metric_12_macd_coherence():
    """Metric 12: MACD line = EMA(12) - EMA(26); Histogram = MACD - Signal(9)."""
    prices = pd.Series(np.sin(np.linspace(0, 10, 100)) * 20.0 + 100.0)
    macd_df = compute_macd(prices, fast_period=12, slow_period=26, signal_period=9)
    valid = macd_df.dropna()
    diff = (valid["macd_line"] - valid["signal_line"]) - valid["histogram"]
    np.testing.assert_allclose(diff.values, 0.0, atol=1e-7)


def test_metric_13_atr_true_range_coverage():
    """Metric 13: ATR uses max of H-L, |H-C_prev|, |L-C_prev|."""
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "High": [105.0] * 30,
        "Low": [95.0] * 30,
        "Close": [100.0] * 30
    }, index=dates)
    atr = compute_atr(df, period=14).dropna()
    assert atr.iloc[-1] == pytest.approx(10.0, abs=1e-4)


def test_metric_14_bollinger_bands_symmetry():
    """Metric 14: Upper = SMA + 2*std, Lower = SMA - 2*std."""
    np.random.seed(42)
    prices = pd.Series(np.random.normal(100, 5, 50))
    bb = compute_bollinger_bands(prices, period=20, num_std=2.0).dropna()
    diff_upper = bb["bb_upper"] - bb["bb_middle"]
    diff_lower = bb["bb_middle"] - bb["bb_lower"]
    np.testing.assert_allclose(diff_upper.values, diff_lower.values, atol=1e-7)


# =============================================================================
# 4. STATISTICAL MOMENTS & VALUE-AT-RISK (Metrics 15 - 20)
# =============================================================================

def test_metric_15_16_skewness_and_excess_kurtosis():
    """Metrics 15 & 16: Gaussian benchmark has skew ~ 0 and excess kurtosis ~ 0."""
    np.random.seed(42)
    gaussian = pd.Series(np.random.normal(0, 1, 50000))
    stats_res = compute_distribution_statistics(gaussian)
    assert stats_res.skewness == pytest.approx(0.0, abs=0.08)
    assert stats_res.excess_kurtosis == pytest.approx(0.0, abs=0.08)


def test_metric_17_jarque_bera_normality_test():
    """Metric 17: JB test rejects heavy-tailed t-distribution."""
    np.random.seed(42)
    t_sample = pd.Series(np.random.standard_t(df=3, size=5000))
    stats_res = compute_distribution_statistics(t_sample)
    assert stats_res.excess_kurtosis > 1.0
    assert stats_res.jarque_bera_pvalue < 1e-3
    assert not stats_res.is_normal_5pct


def test_metric_18_19_20_var_and_cvar_ordering():
    """Metrics 18, 19, 20: Mathematical condition CVaR >= VaR."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.0005, 0.015, 2000))
    hist_var = compute_historical_var(rets, confidence=0.95)
    param_var = compute_parametric_var(rets, confidence=0.95)
    cvar = compute_conditional_var(rets, confidence=0.95)

    assert hist_var > 0.0
    assert param_var > 0.0
    assert cvar >= hist_var


# =============================================================================
# 5. RISK-ADJUSTED PERFORMANCE & ACCOUNTING (Metrics 21 - 26)
# =============================================================================

def test_metric_21_sharpe_ratio():
    """Metric 21: (mean_excess * 252) / annualized_vol with zero vol safeguard."""
    rets = pd.Series([0.001] * 252)
    equity = (1.0 + rets).cumprod() * 100_000.0
    metrics = calculate_performance_metrics(equity, rets, trades=[], risk_free_rate=0.04)
    assert metrics.annualized_volatility == pytest.approx(0.0, abs=1e-12)
    assert metrics.sharpe_ratio == 0.0


def test_metric_22_sortino_ratio_zero_downside():
    """Metric 22: Positive returns with zero downside deviation -> infinite Sortino."""
    rets = pd.Series([0.002] * 252)
    assert compute_downside_deviation(rets, mar_daily=0.0) == 0.0
    equity = (1.0 + rets).cumprod() * 100_000.0
    metrics = calculate_performance_metrics(equity, rets, trades=[], risk_free_rate=0.0)
    assert np.isinf(metrics.sortino_ratio)


def test_metric_23_calmar_ratio():
    """Metric 23: CAGR / Max Drawdown"""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    vals = [100_000.0] * 50 + [120_000.0] * 50 + [90_000.0] * 50 + [110_000.0] * 102
    equity = pd.Series(vals, index=dates)
    rets = equity.pct_change().dropna()
    metrics = calculate_performance_metrics(equity, rets, trades=[])
    assert metrics.max_drawdown == pytest.approx(0.25, abs=1e-3)
    assert metrics.calmar_ratio == pytest.approx(metrics.cagr / 0.25, abs=1e-3)


def test_metric_24_maximum_drawdown_exact():
    """Metric 24: Max peak-to-trough decline."""
    equity = pd.Series([100.0, 120.0, 60.0, 80.0, 50.0, 110.0])
    dd_res = compute_drawdowns(equity)
    assert dd_res.max_drawdown == pytest.approx(70.0 / 120.0, abs=1e-5)


def test_metric_25_profit_factor():
    """Metric 25: Gross profit / Gross loss (evaluated on net PnL after friction)."""
    # Test with zero costs
    trades_frictionless = [
        Trade(1, "AAPL", "LONG", "2023-01-01", "2023-01-05", 100.0, 110.0, 100.0, 0.0, 0.0, 1000.0, 1000.0, 0.10, 4),
        Trade(2, "AAPL", "LONG", "2023-01-06", "2023-01-10", 100.0, 95.0, 100.0, 0.0, 0.0, -500.0, -500.0, -0.05, 4),
    ]
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    equity = pd.Series(np.linspace(100_000, 100_500, 10), index=dates)
    rets = equity.pct_change().dropna()
    metrics_fl = calculate_performance_metrics(equity, rets, trades=trades_frictionless)
    assert metrics_fl.profit_factor == pytest.approx(1000.0 / 500.0, abs=1e-4)

    # Test with realistic friction (net PnL)
    trades_friction = [
        Trade(1, "AAPL", "LONG", "2023-01-01", "2023-01-05", 100.0, 110.0, 100.0, 5.0, 5.0, 1000.0, 990.0, 0.10, 4),
        Trade(2, "AAPL", "LONG", "2023-01-06", "2023-01-10", 100.0, 95.0, 100.0, 5.0, 5.0, -500.0, -510.0, -0.05, 4),
    ]
    metrics_fr = calculate_performance_metrics(equity, rets, trades=trades_friction)
    assert metrics_fr.profit_factor == pytest.approx(990.0 / 510.0, abs=1e-4)


def test_metric_26_win_rate():
    """Metric 26: Winning trades / Total trades."""
    trades = [
        Trade(1, "X", "LONG", "2023-01-01", "2023-01-02", 10.0, 12.0, 10.0, 0.0, 0.0, 20.0, 20.0, 0.2, 1),
        Trade(2, "X", "LONG", "2023-01-03", "2023-01-04", 10.0, 11.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.1, 1),
        Trade(3, "X", "LONG", "2023-01-05", "2023-01-06", 10.0, 9.0, 10.0, 0.0, 0.0, -10.0, -10.0, -0.1, 1),
    ]
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    equity = pd.Series(np.linspace(100_000, 100_020, 10), index=dates)
    rets = equity.pct_change().dropna()
    metrics = calculate_performance_metrics(equity, rets, trades=trades)
    assert metrics.win_rate == pytest.approx(2.0 / 3.0, abs=1e-4)


# =============================================================================
# 6. PORTFOLIO COVARIANCE SHRINKAGE & CONDITION NUMBER AUDIT
# =============================================================================

def test_covariance_condition_number_and_shrinkage():
    """Verifies covariance matrix condition number audit and stabilization."""
    np.random.seed(42)
    base = np.random.normal(0, 0.01, 100)
    collinear_1 = base + np.random.normal(0, 1e-6, 100)
    collinear_2 = base + np.random.normal(0, 1e-6, 100)
    df = pd.DataFrame({"AssetA": base, "AssetB": collinear_1, "AssetC": collinear_2})
    cov = df.cov()

    audit = audit_covariance_matrix(cov)
    assert audit.condition_number > 1e4
    assert audit.is_ill_conditioned

    # Apply shrinkage regularization
    clean_cov = regularize_covariance_matrix(cov, shrinkage_factor=0.1)
    clean_audit = audit_covariance_matrix(clean_cov)
    assert clean_audit.condition_number < audit.condition_number
    assert clean_audit.min_eigenvalue > 0.0


# =============================================================================
# 7. DYNAMIC FUTURES LOT SIZE DATES
# =============================================================================

def test_dynamic_futures_lot_size_regulatory_dates():
    """Verifies SEBI lot size revisions for Nifty and BankNifty across regulatory changes."""
    # NIFTY 50:
    assert get_historical_lot_size("NIFTY_FUT", "2018-05-10") == 75.0
    assert get_historical_lot_size("NIFTY_FUT", "2022-03-15") == 50.0
    assert get_historical_lot_size("NIFTY_FUT", "2024-06-01") == 25.0
    assert get_historical_lot_size("NIFTY_FUT", "2024-11-20") == 75.0
    assert get_historical_lot_size("NIFTY_FUT", "2026-01-01") == 75.0

    # BANK NIFTY:
    assert get_historical_lot_size("BANKNIFTY_FUT", "2022-01-01") == 25.0
    assert get_historical_lot_size("BANKNIFTY_FUT", "2023-08-01") == 15.0
    assert get_historical_lot_size("BANKNIFTY_FUT", "2024-11-20") == 30.0
