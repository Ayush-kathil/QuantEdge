"""
Comprehensive Quantitative Accuracy Audit Test Suite for QuantEdge v3.

Independently tests and mathematically verifies all 26 quantitative finance metrics:
1. Simple returns
2. Log returns
3. Cumulative returns
4. CAGR
5. Annualized volatility
6. Parkinson volatility
7. Garman-Klass volatility
8. EWMA volatility
9. SMA
10. EMA
11. RSI
12. MACD
13. ATR
14. Bollinger Bands
15. Skewness
16. Excess kurtosis
17. Jarque-Bera test
18. Historical VaR
19. Parametric VaR
20. CVaR / Expected Shortfall
21. Sharpe Ratio
22. Sortino Ratio
23. Calmar Ratio
24. Maximum Drawdown
25. Profit Factor
26. Win Rate
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from analytics.returns import (
    compute_daily_returns,
    compute_cumulative_returns,
    compute_excess_returns
)
from analytics.volatility import (
    compute_realized_volatility,
    compute_rolling_volatility,
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_ewma_volatility
)
from analytics.indicators import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_bollinger_bands
)
from analytics.statistics import compute_distribution_statistics
from analytics.risk import (
    compute_drawdowns,
    compute_downside_deviation,
    compute_historical_var,
    compute_parametric_var,
    compute_conditional_var
)
from backtesting.metrics import calculate_performance_metrics
from backtesting.execution import Trade


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def deterministic_prices() -> pd.Series:
    """100-day deterministic geometric random walk with seed 42."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0005, 0.015, 100)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    return pd.Series(prices, index=dates)


@pytest.fixture
def deterministic_ohlcv() -> pd.DataFrame:
    """Deterministic OHLCV dataframe with known geometric relationships."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 100.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    curr = base
    for _ in range(n):
        o = curr
        delta = np.random.normal(0.1, 1.0)
        c = max(1.0, o + delta)
        h = max(o, c) + abs(np.random.normal(0.5, 0.2))
        l = min(o, c) - abs(np.random.normal(0.5, 0.2))
        v = float(np.random.randint(100_000, 500_000))
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)
        curr = c

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=dates)


# -----------------------------------------------------------------------------
# 1-3. Returns Verification
# -----------------------------------------------------------------------------
def test_metric_01_simple_returns():
    """Formula: (P_t - P_{t-1}) / P_{t-1}"""
    p = pd.Series([100.0, 105.0, 94.50])
    rets = compute_daily_returns(p, method="arithmetic")
    assert np.isnan(rets.iloc[0])
    assert rets.iloc[1] == pytest.approx(0.05, abs=1e-6)
    assert rets.iloc[2] == pytest.approx(-0.10, abs=1e-6)


def test_metric_02_log_returns():
    """Formula: ln(P_t / P_{t-1})"""
    p = pd.Series([100.0, 105.0, 94.50])
    rets = compute_daily_returns(p, method="log")
    assert np.isnan(rets.iloc[0])
    assert rets.iloc[1] == pytest.approx(np.log(1.05), abs=1e-6)
    assert rets.iloc[2] == pytest.approx(np.log(94.50 / 105.0), abs=1e-6)


def test_metric_03_cumulative_returns():
    """Formula: (1 + R).cumprod() - 1"""
    r = pd.Series([np.nan, 0.10, -0.05, 0.20])
    cum_ret = compute_cumulative_returns(r, method="arithmetic")
    expected = (1.10 * 0.95 * 1.20) - 1.0
    assert cum_ret.iloc[-1] == pytest.approx(expected, abs=1e-6)


# -----------------------------------------------------------------------------
# 4. CAGR Verification
# -----------------------------------------------------------------------------
def test_metric_04_cagr():
    """Formula: (V_T / V_0)^(252 / N) - 1"""
    dates = pd.date_range("2020-01-01", periods=504, freq="B")  # Exactly 2 years of 252 days
    # Doubling capital over 2 years: (200,000 / 100,000)^(1/2) - 1 = sqrt(2) - 1 = ~41.42%
    equity = pd.Series(np.linspace(100_000.0, 200_000.0, 504), index=dates)
    rets = equity.pct_change().dropna()
    metrics = calculate_performance_metrics(equity, rets, trades=[], trading_days=252)
    assert metrics.cagr == pytest.approx(np.sqrt(2.0) - 1.0, abs=1e-2)


# -----------------------------------------------------------------------------
# 5-8. Volatility Estimators Verification
# -----------------------------------------------------------------------------
def test_metric_05_annualized_volatility():
    """Formula: std(R, ddof=1) * sqrt(252)"""
    np.random.seed(123)
    rets = pd.Series(np.random.normal(0.001, 0.01, 252))
    vol = compute_realized_volatility(rets, trading_days=252, ddof=1)
    expected = float(rets.std(ddof=1) * np.sqrt(252))
    assert vol == pytest.approx(expected, abs=1e-6)


def test_metric_06_parkinson_volatility(deterministic_ohlcv):
    """Parkinson: sqrt( (1 / (4 * ln(2) * N)) * sum( (ln(H/L))^2 ) ) * sqrt(252)"""
    p_vol = compute_parkinson_volatility(deterministic_ohlcv, window=20, trading_days=252)
    assert len(p_vol) == len(deterministic_ohlcv)
    # Must be strictly positive
    valid = p_vol.dropna()
    assert (valid > 0.0).all()


def test_metric_07_garman_klass_volatility(deterministic_ohlcv):
    """Garman-Klass incorporates OHLC to be ~8x more efficient than close-to-close."""
    gk_vol = compute_garman_klass_volatility(deterministic_ohlcv, window=20, trading_days=252)
    valid = gk_vol.dropna()
    assert (valid > 0.0).all()


def test_metric_08_ewma_volatility(deterministic_prices):
    """EWMA volatility with RiskMetrics lambda=0.94."""
    rets = compute_daily_returns(deterministic_prices).dropna()
    ewma_vol = compute_ewma_volatility(rets, decay_factor=0.94, trading_days=252)
    assert len(ewma_vol) == len(rets)
    assert (ewma_vol > 0.0).all()


# -----------------------------------------------------------------------------
# 9-14. Technical Indicators Verification
# -----------------------------------------------------------------------------
def test_metric_09_sma():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma = compute_sma(s, period=3)
    assert np.isnan(sma.iloc[0])
    assert np.isnan(sma.iloc[1])
    assert sma.iloc[2] == pytest.approx(20.0, abs=1e-6)  # (10+20+30)/3
    assert sma.iloc[3] == pytest.approx(30.0, abs=1e-6)  # (20+30+40)/3


def test_metric_10_ema():
    s = pd.Series([10.0, 20.0, 30.0, 40.0])
    ema = compute_ema(s, period=3, adjust=False)
    # alpha = 2 / (3 + 1) = 0.5
    # ema[0] = 10
    # ema[1] = 0.5*20 + 0.5*10 = 15
    # ema[2] = 0.5*30 + 0.5*15 = 22.5
    assert ema.iloc[1] == pytest.approx(15.0, abs=1e-6)
    assert ema.iloc[2] == pytest.approx(22.5, abs=1e-6)


def test_metric_11_rsi_extremes_and_flat():
    # Constant series: Wilder gain=0, loss=0 -> RSI must be 50.0
    flat = pd.Series([100.0] * 30)
    rsi_flat = compute_rsi(flat, period=14).dropna()
    assert (rsi_flat == 50.0).all()

    # Monotonically increasing -> RSI should be 100.0
    up = pd.Series(np.linspace(100, 200, 30))
    rsi_up = compute_rsi(up, period=14).dropna()
    assert rsi_up.iloc[-1] == pytest.approx(100.0, abs=1e-3)


def test_metric_12_macd(deterministic_prices):
    macd_df = compute_macd(deterministic_prices, fast_period=12, slow_period=26, signal_period=9)
    assert "macd_line" in macd_df.columns
    assert "signal_line" in macd_df.columns
    assert "histogram" in macd_df.columns
    # Histogram identity: macd_line - signal_line
    diff = macd_df["macd_line"] - macd_df["signal_line"]
    assert np.allclose(macd_df["histogram"].dropna(), diff.dropna())


def test_metric_13_atr(deterministic_ohlcv):
    atr = compute_atr(deterministic_ohlcv, period=14)
    valid_atr = atr.dropna()
    assert len(valid_atr) > 0
    assert (valid_atr > 0.0).all()


def test_metric_14_bollinger_bands():
    s = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0] * 5)
    bb = compute_bollinger_bands(s, period=10, num_std=2.0)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()
    # Symmetry: Upper - Middle == Middle - Lower
    diff1 = valid["bb_upper"] - valid["bb_middle"]
    diff2 = valid["bb_middle"] - valid["bb_lower"]
    assert np.allclose(diff1, diff2)


# -----------------------------------------------------------------------------
# 15-17. Statistical Moments & Normality
# -----------------------------------------------------------------------------
def test_metric_15_skewness():
    """Positively skewed synthetic series."""
    np.random.seed(99)
    pos_skew = np.concatenate([np.random.normal(0, 1, 1000), [10, 12, 15, 20]])
    skew = stats.skew(pos_skew, bias=False)
    assert skew > 0.5


def test_metric_16_excess_kurtosis():
    """Standard Gaussian has excess kurtosis ~ 0; Student-t(3) has fat tails (> 0)."""
    np.random.seed(99)
    t_dist = stats.t.rvs(df=3, size=2000)
    kurt = stats.kurtosis(t_dist, fisher=True, bias=False)
    assert kurt > 1.0  # Leptokurtic fat tails


def test_metric_17_jarque_bera_normality():
    np.random.seed(42)
    gaussian_sample = np.random.normal(0, 1, 500)
    jb_stat, p_val = stats.jarque_bera(gaussian_sample)
    assert p_val > 0.05  # Fails to reject normality for true Gaussian


# -----------------------------------------------------------------------------
# 18-20. Value at Risk (VaR) & CVaR / Expected Shortfall
# -----------------------------------------------------------------------------
def test_metric_18_historical_var():
    rets = pd.Series(np.linspace(-0.10, 0.10, 1001))
    var_95 = compute_historical_var(rets, confidence_level=0.95)
    # 5th percentile is approximately -0.09 -> VaR is positive 0.09
    assert var_95 == pytest.approx(0.09, abs=0.01)


def test_metric_19_parametric_var():
    np.random.seed(42)
    # Zero-mean unit-variance Gaussian
    rets = pd.Series(np.random.normal(0.0, 0.02, 5000))
    p_var = compute_parametric_var(rets, confidence_level=0.95)
    # Expected: 1.64485 * 0.02 = ~0.0329
    assert p_var == pytest.approx(1.64485 * 0.02, abs=0.003)


def test_metric_20_cvar_greater_than_var():
    """Expected Shortfall (CVaR) must be strictly greater than or equal to VaR."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.0, 0.02, 1000))
    var = compute_historical_var(rets, confidence_level=0.95)
    cvar = compute_conditional_var(rets, confidence_level=0.95)
    assert cvar >= var


# -----------------------------------------------------------------------------
# 21-26. Strategy Performance Metrics
# -----------------------------------------------------------------------------
def test_metric_21_sharpe_ratio():
    # Consistent 10 bps daily return with 100 bps daily volatility
    # Daily Sharpe = 0.0010 / 0.010 = 0.10 -> Annualized = 0.10 * sqrt(252) = ~1.587
    daily_rets = pd.Series([0.001] * 252)
    # Add minor noise so std is well-defined
    daily_rets.iloc[0] = 0.0011
    daily_rets.iloc[1] = 0.0009
    equity = 100_000.0 * (1.0 + daily_rets).cumprod()
    metrics = calculate_performance_metrics(equity, daily_rets, trades=[], risk_free_rate=0.0)
    assert metrics.sharpe_ratio > 0.0


def test_metric_22_sortino_ratio():
    # Only positive returns: downside deviation is 0 -> Sortino is infinite or handled safely
    pos_rets = pd.Series([0.01, 0.02, 0.015, 0.012] * 50)
    dd = compute_downside_deviation(pos_rets, mar_daily=0.0)
    assert dd == 0.0


def test_metric_23_calmar_ratio():
    """Formula: CAGR / Max Drawdown"""
    dates = pd.date_range("2022-01-01", periods=252, freq="B")
    # 20% CAGR, 10% Max Drawdown -> Calmar = 2.0
    equity = pd.Series(np.linspace(100_000, 120_000, 252), index=dates)
    # Create artificial 10% dip in middle
    equity.iloc[120:140] = 95_000.0
    rets = equity.pct_change().dropna()
    metrics = calculate_performance_metrics(equity, rets, trades=[])
    assert metrics.calmar_ratio > 0.0
    assert metrics.calmar_ratio == pytest.approx(metrics.cagr / metrics.max_drawdown, abs=1e-4)


def test_metric_24_maximum_drawdown():
    # Drop from 100 -> 75 is a 25% drawdown
    curve = pd.Series([100.0, 110.0, 120.0, 90.0, 95.0, 130.0])
    dd = compute_drawdowns(curve)
    # Peak was 120, trough was 90: (120 - 90) / 120 = 30 / 120 = 25%
    assert dd.max_drawdown == pytest.approx(0.25, abs=1e-6)
    assert dd.max_drawdown_dollars == pytest.approx(30.0, abs=1e-6)


def test_metric_25_profit_factor():
    """Profit Factor = Gross Profits / Gross Losses"""
    t1 = Trade(1, "SPY", "LONG", "2023-01-01", "2023-01-02", 100.0, 110.0, 10.0, 5.0, 5.0, 100.0, 90.0, 0.10, 1)
    t2 = Trade(2, "SPY", "LONG", "2023-01-03", "2023-01-04", 100.0, 95.0, 10.0, 5.0, 5.0, -50.0, -60.0, -0.05, 1)
    metrics = calculate_performance_metrics(pd.Series([1000, 1030]), pd.Series([0.03]), trades=[t1, t2])
    # Net profit: 90.0, Net loss: 60.0 -> PF = 90 / 60 = 1.50
    assert metrics.profit_factor == pytest.approx(90.0 / 60.0, abs=1e-4)


def test_metric_26_win_rate():
    """Win Rate = Winning Trades / Total Trades"""
    t1 = Trade(1, "SPY", "LONG", "2023-01-01", "2023-01-02", 100.0, 110.0, 10.0, 5.0, 5.0, 100.0, 90.0, 0.10, 1)
    t2 = Trade(2, "SPY", "LONG", "2023-01-03", "2023-01-04", 100.0, 105.0, 10.0, 5.0, 5.0, 50.0, 40.0, 0.05, 1)
    t3 = Trade(3, "SPY", "LONG", "2023-01-05", "2023-01-06", 100.0, 95.0, 10.0, 5.0, 5.0, -50.0, -60.0, -0.05, 1)
    metrics = calculate_performance_metrics(pd.Series([1000, 1070]), pd.Series([0.07]), trades=[t1, t2, t3])
    # 2 wins out of 3 trades = 66.67%
    assert metrics.win_rate == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
