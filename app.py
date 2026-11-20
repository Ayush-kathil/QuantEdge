"""
QuantEdge – Quantitative Market Research & Strategy Backtesting Platform
An institutional-grade research platform for quantitative time-series analysis and strategy backtesting.
"""
import io
import sys
import os
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="QuantEdge | Quantitative Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure root directory is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config.settings import (
    INSTRUMENT_CATALOG,
    DEFAULT_TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SLIPPAGE_BPS
)
from data.ingestion import MarketDataIngestion, MarketDataPayload
from data.validation import validate_market_data
from analytics.returns import compute_daily_returns, compute_cumulative_returns, compute_excess_returns
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
from analytics.statistics import compute_distribution_statistics, compute_correlation_matrix
from analytics.risk import compute_drawdowns, compute_downside_deviation
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIMeanReversion
from strategies.breakout_strategy import DonchianBreakout
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.validation import run_out_of_sample_validation
from visualization.charts import (
    plot_market_explorer_candlestick,
    plot_returns_distribution,
    plot_rolling_volatility_comparison,
    plot_technical_indicators_suite,
    plot_equity_curves,
    plot_drawdown_underwater,
    plot_trade_markers_on_price,
    plot_autocorrelation_bar,
    apply_quant_layout
)

# Custom Institutional CSS styling
st.markdown("""
<style>
    /* Sleek Quantitative Terminal Theme */
    .stApp {
        background-color: #0B0E14;
        color: #E6EDF3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top Header Pill */
    .quant-header-badge {
        display: inline-block;
        padding: 4px 12px;
        background: linear-gradient(90deg, #1F6FEB 0%, #1158C7 100%);
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 4px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricValue"] {
        font-family: "JetBrains Mono", "SF Mono", Consolas, monospace;
        font-size: 1.45rem !important;
        font-weight: 700;
        color: #58A6FF !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600;
        color: #8B949E !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F0F6FC;
        border-bottom: 2px solid #21262D;
        padding-bottom: 6px;
        margin-top: 18px;
        margin-bottom: 14px;
    }
    
    /* Table Styling */
    .dataframe {
        font-family: "JetBrains Mono", Consolas, monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Notification pills */
    .quant-tag-bull {
        color: #3FB950;
        background-color: rgba(63, 185, 80, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .quant-tag-bear {
        color: #F85149;
        background-color: rgba(248, 81, 73, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .quant-tag-neutral {
        color: #E3B341;
        background-color: rgba(227, 179, 65, 0.15);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONFIGURATION & DATA INGESTION CONTROLS
# ==============================================================================
st.sidebar.markdown("""
<div style="padding-bottom: 12px;">
    <span class="quant-header-badge">QUANTITATIVE PLATFORM</span>
    <h2 style="margin: 0; padding: 0; color: #58A6FF; font-size: 1.5rem; font-weight: 800;">QuantEdge</h2>
    <p style="margin: 0; color: #8B949E; font-size: 0.8rem;">Market Research & Strategy Backtesting</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("1. Market Data Ingestion")

data_source = st.sidebar.radio(
    "Data Source Mode",
    ["Yahoo Finance API", "CSV File Upload"],
    help="Select live historical API or upload custom financial OHLCV time-series."
)

ingestion_engine = MarketDataIngestion(cache_enabled=True)
market_payload: Optional[MarketDataPayload] = None
ingestion_error: Optional[str] = None

if data_source == "Yahoo Finance API":
    # Preset selection or custom ticker
    ticker_choice = st.sidebar.selectbox(
        "Select Instrument",
        options=list(INSTRUMENT_CATALOG.keys()) + ["CUSTOM"],
        format_func=lambda x: f"{x} - {INSTRUMENT_CATALOG[x]['name']} ({INSTRUMENT_CATALOG[x]['asset_class']})" if x in INSTRUMENT_CATALOG else "Custom Ticker Input"
    )

    if ticker_choice == "CUSTOM":
        active_ticker = st.sidebar.text_input("Enter Symbol (e.g. NVDA, BTC-USD)", value="NVDA").strip().upper()
    else:
        active_ticker = ticker_choice

    # Date Range Selection
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 3)  # 3-year historical window

    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=default_start)
    with col_d2:
        end_date = st.date_input("End Date", value=default_end)

    if start_date >= end_date:
        st.sidebar.error("Error: Start Date must be strictly before End Date.")
    else:
        with st.sidebar:
            with st.spinner(f"Ingesting {active_ticker}..."):
                try:
                    market_payload = ingestion_engine.fetch_yfinance(
                        ticker=active_ticker,
                        start_date=start_date,
                        end_date=end_date,
                        interval="1d",
                        strict_ohlc=False
                    )
                except Exception as e:
                    ingestion_error = str(e)

else:
    # CSV File Upload
    st.sidebar.markdown("**Upload Custom OHLCV CSV**")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Must contain Date and OHLCV columns.")

    use_sample_btn = st.sidebar.button("Load Pre-Packaged SPY Sample", help="Loads verified S&P 500 fixture for offline exploration.")

    sample_csv_path = os.path.join(CURRENT_DIR, "sample_data", "sample_sp500.csv")

    if uploaded_file is not None:
        try:
            market_payload = ingestion_engine.load_csv(uploaded_file, ticker_name="UPLOADED_CSV")
        except Exception as e:
            ingestion_error = str(e)
    elif use_sample_btn or os.path.exists(sample_csv_path):
        try:
            market_payload = ingestion_engine.load_csv(sample_csv_path, ticker_name="SPY_SAMPLE")
        except Exception as e:
            ingestion_error = str(e)

# Global Simulation Parameters in Sidebar
st.sidebar.header("2. Simulation Assumptions")
trading_days_input = st.sidebar.number_input("Trading Days / Year", min_value=200, max_value=365, value=DEFAULT_TRADING_DAYS_PER_YEAR, step=1, help="Annualization factor for volatility and Sharpe (252 for equities).")
rf_rate_input = st.sidebar.number_input("Risk-Free Rate (Annual %)", min_value=0.0, max_value=20.0, value=DEFAULT_RISK_FREE_RATE * 100.0, step=0.25) / 100.0
initial_capital_input = st.sidebar.number_input("Initial Portfolio Capital ($)", min_value=1_000.0, max_value=10_000_000.0, value=DEFAULT_INITIAL_CAPITAL, step=10_000.0)

st.sidebar.header("3. Execution Frictions")
commission_bps_input = st.sidebar.number_input("Commission (bps)", min_value=0.0, max_value=100.0, value=DEFAULT_COMMISSION_BPS, step=0.5, help="Brokerage commission in basis points (1 bp = 0.01%).")
slippage_bps_input = st.sidebar.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=DEFAULT_SLIPPAGE_BPS, step=0.5, help="Adverse price slippage penalty on fills.")
exec_mode_input = st.sidebar.selectbox("Execution Timing", ["next_open", "next_close"], help="Next-period Open fills eliminate look-ahead bias.")

# Data state gatekeeper
if ingestion_error:
    st.error(f"⚠️ Market Data Retrieval Error: {ingestion_error}")
    st.info("Tip: If live internet or Yahoo Finance is unreachable, switch data mode to 'CSV File Upload' and click 'Load Pre-Packaged SPY Sample'.")
    st.stop()

if market_payload is None or market_payload.df.empty:
    st.warning("Please select or upload market data using the sidebar to commence analysis.")
    st.stop()

df_market = market_payload.df
active_symbol = market_payload.ticker

# Compute base analytics
daily_returns = compute_daily_returns(df_market["Close"], method="arithmetic")
cum_returns = compute_cumulative_returns(daily_returns, method="arithmetic")


# ==============================================================================
# MAIN TERMINAL NAVIGATION (8 SPECIALIZED WORKSPACES)
# ==============================================================================
tab_overview, tab_explorer, tab_stats, tab_indicators, tab_strategy, tab_backtest, tab_risk, tab_methodology = st.tabs([
    "📊 Terminal Overview",
    "📈 Market Explorer",
    "🔬 Statistical Analysis",
    "📐 Technical Indicators",
    "🧪 Strategy Lab",
    "⚡ Backtesting Engine",
    "🛡️ Risk Analytics",
    "📚 Methodology & Notes"
])


# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown(f"### {active_symbol} — Quantitative Market Research Overview")
    meta = market_payload.metadata
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.caption("ASSET CLASS")
        st.markdown(f"**{meta.get('asset_class', 'Market Security')}**")
    with col_t2:
        st.caption("PRIMARY VENUE")
        st.markdown(f"**{meta.get('exchange', 'Exchange')}**")
    with col_t3:
        st.caption("OBSERVATION WINDOW")
        st.markdown(f"**{market_payload.start_date} → {market_payload.end_date}**")
    with col_t4:
        st.caption("DATA SOURCE")
        st.markdown(f"**{market_payload.source}**")

    st.markdown("---")
    
    # Top-Level Snapshot Metrics
    c_last = float(df_market["Close"].iloc[-1])
    c_prev = float(df_market["Close"].iloc[-2]) if len(df_market) > 1 else c_last
    d_chg = (c_last - c_prev) / c_prev
    tot_ret = float(cum_returns.iloc[-1])
    ann_vol = compute_realized_volatility(daily_returns, trading_days=int(trading_days_input))
    dd_summary = compute_drawdowns(df_market["Close"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Latest Close", f"${c_last:,.2f}", delta=f"{d_chg * 100:+.2f}%")
    m2.metric("Cumulative Return", f"{tot_ret * 100:+.2f}%")
    m3.metric("Annualized Volatility", f"{ann_vol * 100:.2f}%")
    m4.metric("Historical Max Drawdown", f"-{dd_summary.max_drawdown * 100:.2f}%")
    m5.metric("Total Trading Bars", f"{len(df_market):,}")

    st.markdown('<div class="section-title">Executive Research Summary</div>', unsafe_allow_html=True)
    col_sum1, col_sum2 = st.columns([2, 1])
    with col_sum1:
        st.markdown(f"""
        This dashboard presents an uncompromised quantitative evaluation of **{active_symbol}** ({meta.get('name', active_symbol)}).
        
        **Methodological Highlights:**
        - **No Look-Ahead Leakage:** All signals generated at bar $t$ are executed strictly at bar $t+1$ Open with realistic frictions.
        - **Microstructure Realism:** Transaction fees of **{commission_bps_input:.1f} bps** and adverse slippage of **{slippage_bps_input:.1f} bps** applied to every round-trip.
        - **Auditable Hygiene:** Automated OHLC integrity verification, monotonic chronological sorting, and deduplication applied to all raw feeds.
        """)
    with col_sum2:
        st.info(f"""
        **Data Hygiene Report**
        - Ingested Rows: `{len(df_market)}`
        - Missing Price Cells: `{sum(market_payload.validation.stats.get('nan_counts', {}).values())}`
        - Duplicate Bars Removed: `{market_payload.validation.stats.get('duplicate_timestamps', 0)}`
        - OHLC Inconsistencies: `{market_payload.validation.stats.get('high_lt_low_violations', 0)}`
        """)

    st.plotly_chart(
        plot_market_explorer_candlestick(df_market, active_symbol, show_volume=True),
        use_container_width=True,
        key="overview_candlestick"
    )


# ------------------------------------------------------------------------------
# TAB 2: MARKET EXPLORER
# ------------------------------------------------------------------------------
with tab_explorer:
    st.markdown('<div class="section-title">Market Explorer & Price Time-Series</div>', unsafe_allow_html=True)
    
    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 2])
    with exp_c1:
        show_vol_toggle = st.checkbox("Show Trading Volume Subplot", value=True)
    with exp_c2:
        show_cum_ret_toggle = st.checkbox("Overlay Cumulative Return", value=False)

    if show_cum_ret_toggle:
        fig_cum = apply_quant_layout(
            go.Figure(go.Scatter(x=cum_returns.index, y=cum_returns * 100.0, mode="lines", name="Cumulative Return (%)", line={"color": "#00E5FF", "width": 2})),
            f"{active_symbol} Cumulative Asset Return (%)",
            height=380
        )
        fig_cum.update_yaxes(title_text="Total Return (%)")
        st.plotly_chart(fig_cum, use_container_width=True, key="explorer_cum_returns")

    st.plotly_chart(
        plot_market_explorer_candlestick(df_market, active_symbol, show_volume=show_vol_toggle),
        use_container_width=True,
        key="explorer_candlestick"
    )

    with st.expander("🔍 View Raw OHLCV Time-Series & Data Export"):
        st.dataframe(df_market.tail(100), use_container_width=True)
        csv_bytes = df_market.to_csv().encode("utf-8")
        st.download_button(
            label=f"⬇️ Export {active_symbol} OHLCV Data (CSV)",
            data=csv_bytes,
            file_name=f"{active_symbol}_ohlcv_export.csv",
            mime="text/csv"
        )


# ------------------------------------------------------------------------------
# TAB 3: STATISTICAL ANALYSIS
# ------------------------------------------------------------------------------
with tab_stats:
    st.markdown('<div class="section-title">Statistical Distribution & Moments Engine</div>', unsafe_allow_html=True)
    
    dist_stats = compute_distribution_statistics(daily_returns, trading_days=int(trading_days_input))

    s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
    s_c1.metric("Arithmetic Mean (Ann.)", f"{dist_stats.mean_annualized * 100:+.2f}%")
    s_c2.metric("Median (Daily)", f"{dist_stats.median_daily * 100:+.3f}%")
    s_c3.metric("Sample Variance", f"{dist_stats.variance_daily:.6f}")
    s_c4.metric("Skewness (Fisher)", f"{dist_stats.skewness:+.2f}")
    s_c5.metric("Excess Kurtosis", f"{dist_stats.excess_kurtosis:+.2f}")

    col_plot_dist, col_table_dist = st.columns([3, 2])
    with col_plot_dist:
        st.plotly_chart(
            plot_returns_distribution(daily_returns, active_symbol, dist_stats),
            use_container_width=True,
            key="stats_dist_hist"
        )
    with col_table_dist:
        st.markdown("**Quantile & Tail Risk Measures**")
        q_df = pd.DataFrame({
            "Risk Measure": [
                "1% Quantile (Extreme)", "5% Quantile", "25% (Q1)", "50% (Median)",
                "75% (Q3)", "95% Quantile", "99% Quantile",
                "Historical 1-Day VaR (95%)", "Historical 1-Day CVaR (95%)",
                "Historical 1-Day VaR (99%)", "Historical 1-Day CVaR (99%)"
            ],
            "Value": [
                f"{dist_stats.percentiles['1%']*100:.2f}%",
                f"{dist_stats.percentiles['5%']*100:.2f}%",
                f"{dist_stats.percentiles['25% (Q1)']*100:.2f}%",
                f"{dist_stats.percentiles['50% (Median)']*100:.2f}%",
                f"{dist_stats.percentiles['75% (Q3)']*100:.2f}%",
                f"{dist_stats.percentiles['95%']*100:.2f}%",
                f"{dist_stats.percentiles['99%']*100:.2f}%",
                f"{dist_stats.var_95*100:.2f}%",
                f"{dist_stats.cvar_95*100:.2f}%",
                f"{dist_stats.var_99*100:.2f}%",
                f"{dist_stats.cvar_99*100:.2f}%",
            ]
        })
        st.dataframe(q_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Quantitative Diagnostic Interpretations</div>', unsafe_allow_html=True)
    for interp in dist_stats.interpretations:
        st.markdown(f"- {interp}")

    st.markdown('<div class="section-title">Serial Correlation & Autocorrelation (ACF)</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_autocorrelation_bar(dist_stats.autocorr_lags), use_container_width=True, key="stats_acf")


# ------------------------------------------------------------------------------
# TAB 4: TECHNICAL INDICATORS
# ------------------------------------------------------------------------------
with tab_indicators:
    st.markdown('<div class="section-title">Technical Indicators & Oscillator Suite</div>', unsafe_allow_html=True)

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        rsi_p = st.slider("RSI Lookback Period", min_value=5, max_value=50, value=14)
    with t_col2:
        fast_p = st.slider("MACD Fast Period", min_value=5, max_value=30, value=12)
    with t_col3:
        slow_p = st.slider("MACD Slow Period", min_value=15, max_value=60, value=26)
    with t_col4:
        signal_p = st.slider("MACD Signal Period", min_value=3, max_value=20, value=9)

    st.plotly_chart(
        plot_technical_indicators_suite(
            df_market, active_symbol,
            rsi_period=rsi_p,
            macd_fast=fast_p,
            macd_slow=slow_p,
            macd_signal=signal_p,
            atr_period=14
        ),
        use_container_width=True,
        key="tech_indicators_suite"
    )


# ------------------------------------------------------------------------------
# TAB 5: STRATEGY LAB
# ------------------------------------------------------------------------------
with tab_strategy:
    st.markdown('<div class="section-title">Strategy Parameter Lab & Signal Inspector</div>', unsafe_allow_html=True)

    strat_select = st.selectbox(
        "Select Strategy to Inspect",
        ["Moving Average Crossover", "RSI Mean Reversion", "Donchian Breakout"]
    )

    if strat_select == "Moving Average Crossover":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            short_ma = st.number_input("Short MA Period", min_value=3, max_value=100, value=20)
        with c2:
            long_ma = st.number_input("Long MA Period", min_value=10, max_value=250, value=50)
        with c3:
            ma_kind = st.selectbox("MA Type", ["SMA", "EMA"])
        with c4:
            lo_ma = st.checkbox("Long Only Mode", value=True)

        active_strategy = MovingAverageCrossover(short_period=int(short_ma), long_period=int(long_ma), ma_type=ma_kind, long_only=lo_ma)

    elif strat_select == "RSI Mean Reversion":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            r_period = st.number_input("RSI Period", min_value=5, max_value=50, value=14)
        with c2:
            os_thresh = st.number_input("Oversold Level", min_value=10.0, max_value=45.0, value=30.0)
        with c3:
            ob_thresh = st.number_input("Overbought Level", min_value=55.0, max_value=90.0, value=70.0)
        with c4:
            exit_neutral = st.checkbox("Exit at Neutral (50)", value=True)

        active_strategy = RSIMeanReversion(rsi_period=int(r_period), oversold=float(os_thresh), overbought=float(ob_thresh), exit_at_neutral=exit_neutral)

    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            breakout_w = st.number_input("Entry Lookback Window (days)", min_value=5, max_value=100, value=20)
        with c2:
            exit_w = st.number_input("Exit Lookback Window (days)", min_value=3, max_value=50, value=10)
        with c3:
            mid_exit = st.checkbox("Exit on Midpoint Breach", value=False)

        active_strategy = DonchianBreakout(lookback_window=int(breakout_w), exit_lookback_window=int(exit_w), use_midpoint_exit=mid_exit)

    # Generate signals
    signals_output = active_strategy.generate_signals(df_market)
    
    st.markdown(f"**Strategy Architecture:** {active_strategy.description}")
    
    # Signal timeline chart
    fig_sig = apply_quant_layout(
        go.Figure(go.Scatter(x=signals_output.index, y=signals_output["signal"], mode="lines", name="Position Signal", line={"color": "#58A6FF", "width": 1.5})),
        f"{active_strategy.name} Signal Sequence (+1 = Long, 0 = Flat, -1 = Short)",
        height=320
    )
    fig_sig.update_yaxes(range=[-1.2, 1.2], tickvals=[-1, 0, 1], ticktext=["Short (-1)", "Flat (0)", "Long (+1)"])
    st.plotly_chart(fig_sig, use_container_width=True, key="strategy_signal_timeline")

    with st.expander("View Signal Data Matrix"):
        st.dataframe(signals_output.tail(50), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 6: BACKTESTING ENGINE & MULTI-STRATEGY COMPARISON
# ------------------------------------------------------------------------------
with tab_backtest:
    st.markdown('<div class="section-title">Discrete-Event Backtesting & Benchmark Engine</div>', unsafe_allow_html=True)

    engine = BacktestEngine(
        initial_capital=float(initial_capital_input),
        commission_bps=float(commission_bps_input),
        slippage_bps=float(slippage_bps_input),
        execution_mode=exec_mode_input,
        risk_free_rate=float(rf_rate_input),
        trading_days=int(trading_days_input)
    )

    # Instantiate all 3 strategies for multi-comparison
    strat_ma = MovingAverageCrossover(short_period=20, long_period=50, ma_type="SMA")
    strat_rsi = RSIMeanReversion(rsi_period=14, oversold=30.0, overbought=70.0)
    strat_breakout = DonchianBreakout(lookback_window=20, exit_lookback_window=10)

    # Run backtests
    res_ma = engine.run(df_market, strat_ma, symbol=active_symbol)
    res_rsi = engine.run(df_market, strat_rsi, symbol=active_symbol)
    res_breakout = engine.run(df_market, strat_breakout, symbol=active_symbol)

    # Multi-Strategy Comparison Map
    all_results = {
        "MA Crossover (20/50)": res_ma,
        "RSI Mean Reversion (14)": res_rsi,
        "Donchian Breakout (20/10)": res_breakout
    }

    # Primary Focus Strategy Selector
    chosen_strat_name = st.selectbox("Primary Strategy for Trade Audit", list(all_results.keys()))
    active_backtest_res = all_results[chosen_strat_name]

    # Performance Scorecards
    st.markdown("#### Performance Metric Matrix")
    m_strat = active_backtest_res.metrics
    m_bm = active_backtest_res.benchmark_metrics

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Strategy Total Return", f"{m_strat.total_return * 100:+.2f}%", delta=f"{(m_strat.total_return - m_bm.total_return)*100:+.2f}% vs B&H")
    p2.metric("CAGR", f"{m_strat.cagr * 100:.2f}%", delta=f"{(m_strat.cagr - m_bm.cagr)*100:+.2f}%")
    p3.metric("Sharpe Ratio", f"{m_strat.sharpe_ratio:.2f}", delta=f"{m_strat.sharpe_ratio - m_bm.sharpe_ratio:+.2f}")
    p4.metric("Sortino Ratio", f"{m_strat.sortino_ratio:.2f}")
    p5.metric("Max Drawdown", f"-{m_strat.max_drawdown * 100:.2f}%", delta=f"-{(m_strat.max_drawdown - m_bm.max_drawdown)*100:.2f}%", delta_color="inverse")
    p6.metric("Win Rate", f"{m_strat.win_rate * 100:.1f}%", f"{m_strat.winning_trades}W / {m_strat.losing_trades}L")

    # Multi-Equity curves
    st.plotly_chart(
        plot_equity_curves(all_results, active_backtest_res.benchmark_equity_curve, title=f"{active_symbol} Multi-Strategy Backtest vs Buy & Hold Benchmark"),
        use_container_width=True,
        key="backtest_equity_curves"
    )

    # Trade Markers on Price
    st.plotly_chart(
        plot_trade_markers_on_price(df_market, active_backtest_res.trades, active_symbol),
        use_container_width=True,
        key="backtest_trade_markers"
    )

    # Strategy Comparison Table
    st.markdown("#### Institutional Strategy Benchmark Comparison")
    comp_rows = []
    headers = ["Strategy", "Ending Equity", "Total Return", "CAGR", "Sharpe", "Sortino", "Max DD", "Win Rate", "Trades", "Profit Factor", "Market Exp."]
    
    # Add Buy & Hold
    comp_rows.append({
        "Strategy": "Buy & Hold Benchmark",
        "Ending Equity": f"${m_bm.ending_capital:,.2f}",
        "Total Return": f"{m_bm.total_return*100:.2f}%",
        "CAGR": f"{m_bm.cagr*100:.2f}%",
        "Sharpe": f"{m_bm.sharpe_ratio:.2f}",
        "Sortino": f"{m_bm.sortino_ratio:.2f}",
        "Max DD": f"{m_bm.max_drawdown*100:.2f}%",
        "Win Rate": "N/A",
        "Trades": 1,
        "Profit Factor": "N/A",
        "Market Exp.": "100.0%"
    })
    for s_name, s_res in all_results.items():
        sm = s_res.metrics
        comp_rows.append({
            "Strategy": s_name,
            "Ending Equity": f"${sm.ending_capital:,.2f}",
            "Total Return": f"{sm.total_return*100:.2f}%",
            "CAGR": f"{sm.cagr*100:.2f}%",
            "Sharpe": f"{sm.sharpe_ratio:.2f}",
            "Sortino": f"{sm.sortino_ratio:.2f}",
            "Max DD": f"{sm.max_drawdown*100:.2f}%",
            "Win Rate": f"{sm.win_rate*100:.1f}%",
            "Trades": sm.total_trades,
            "Profit Factor": f"{sm.profit_factor:.2f}" if not np.isinf(sm.profit_factor) else "Inf",
            "Market Exp.": f"{sm.market_exposure*100:.1f}%"
        })
    df_comp = pd.DataFrame(comp_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Trade History Table & Download
    st.markdown("#### Trade Audit Log")
    if active_backtest_res.trades:
        trades_records = [{
            "Trade #": t.trade_id,
            "Direction": t.direction,
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Entry Price ($)": round(t.entry_price, 2),
            "Exit Price ($)": round(t.exit_price, 2),
            "Shares": round(t.shares, 4),
            "Gross PnL ($)": round(t.gross_pnl, 2),
            "Net PnL ($)": round(t.net_pnl, 2),
            "Return (%)": round(t.return_pct * 100.0, 2),
            "Holding Bars": t.duration_bars
        } for t in active_backtest_res.trades]
        df_trades = pd.DataFrame(trades_records)
        st.dataframe(df_trades, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label=f"⬇️ Download {chosen_strat_name} Trade Log (CSV)",
                data=df_trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{active_symbol}_{chosen_strat_name.replace(' ', '_')}_trades.csv",
                mime="text/csv"
            )
        with col_d2:
            st.download_button(
                label="⬇️ Download Strategy Metrics Comparison (CSV)",
                data=df_comp.to_csv(index=False).encode("utf-8"),
                file_name=f"{active_symbol}_strategy_metrics.csv",
                mime="text/csv"
            )
    else:
        st.info("No closed trades generated by this strategy over the selected date range.")

    # In-Sample / Out-of-Sample Validation Section
    st.markdown('<div class="section-title">Chronological Out-of-Sample (OOS) Validation</div>', unsafe_allow_html=True)
    st.caption("Splits data chronologically (70% In-Sample training, 30% Out-of-Sample testing) to test parameter robustness and detect overfitting.")
    
    if len(df_market) >= 60:
        val_report = run_out_of_sample_validation(df_market, active_strategy, engine, train_ratio=0.70, symbol=active_symbol)
        
        c_v1, c_v2 = st.columns([1, 1])
        with c_v1:
            st.markdown(f"**In-Sample Window:** `{val_report.train_start}` → `{val_report.train_end}`")
            st.markdown(f"**Out-of-Sample Window:** `{val_report.test_start}` → `{val_report.test_end}`")
        with c_v2:
            if val_report.is_overfit_suspected:
                st.error("⚠️ Overfitting / Degradation Detected in Out-of-Sample Period:")
                for flag in val_report.diagnostic_flags:
                    st.write(f"- {flag}")
            else:
                st.success("✅ Robust Parameter Behavior: Out-of-Sample performance maintained consistency without severe Sharpe collapse.")

        st.dataframe(pd.DataFrame(val_report.comparison_summary), use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient data points (< 60 bars) for meaningful out-of-sample chronological partitioning.")


# ------------------------------------------------------------------------------
# TAB 7: RISK ANALYTICS
# ------------------------------------------------------------------------------
with tab_risk:
    st.markdown('<div class="section-title">Risk Analytics & Drawdown Decomposition</div>', unsafe_allow_html=True)

    # Underwater Drawdown Curve
    dd_strat = compute_drawdowns(active_backtest_res.equity_curve)
    dd_bm = compute_drawdowns(active_backtest_res.benchmark_equity_curve)

    st.plotly_chart(
        plot_drawdown_underwater(dd_strat.drawdown_series, dd_bm.drawdown_series, title=f"{chosen_strat_name} vs Benchmark Underwater Drawdown Profile"),
        use_container_width=True,
        key="risk_underwater_drawdown"
    )

    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    r_c1.metric("Maximum Drawdown", f"-{dd_strat.max_drawdown * 100:.2f}%")
    r_c2.metric("Peak Loss ($)", f"${dd_strat.max_drawdown_dollars:,.2f}")
    r_c3.metric("Max Drawdown Duration", f"{dd_strat.max_drawdown_duration_days} days")
    r_c4.metric("Current Drawdown", f"-{dd_strat.current_drawdown * 100:.2f}%")

    st.markdown(f"""
    **Drawdown Anatomy:**
    - **Peak Date:** `{dd_strat.max_drawdown_peak_date}`
    - **Trough Date:** `{dd_strat.max_drawdown_trough_date}`
    - **Recovery Date:** `{dd_strat.max_drawdown_recovery_date}`
    """)

    # Rolling Volatility Multi-Estimators
    st.markdown('<div class="section-title">Multi-Estimator Volatility Dynamics</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_rolling_volatility_comparison(daily_returns, df_market, active_symbol, window=21),
        use_container_width=True,
        key="risk_rolling_volatility"
    )


# ------------------------------------------------------------------------------
# TAB 8: ABOUT & METHODOLOGY (INTERNSHIP & FUTURES FIRST PREP)
# ------------------------------------------------------------------------------
with tab_methodology:
    st.markdown('<div class="section-title">Academic & Institutional Methodology Guide</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### 1. Mathematical Formulas & Conventions
    
    #### A. Return Definitions
    - **Simple Arithmetic Daily Return:**
      $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
    - **Continuously Compounded Log Return:**
      $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
    - **Cumulative Return:**
      $$\prod_{t=1}^T (1 + R_t) - 1$$
    - **Compound Annual Growth Rate (CAGR):**
      $$\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{252}{N}} - 1$$
      where $N$ is the number of trading days in the backtest.
      
    #### B. Risk & Volatility Estimators
    - **Annualized Realized Volatility (Close-to-Close):**
      $$\sigma_{\text{ann}} = \sqrt{252} \times \sqrt{\frac{1}{N-1} \sum_{t=1}^N (R_t - \bar{R})^2}$$
    - **Parkinson Volatility (High-Low Estimator):**
      $$\sigma_{\text{Parkinson}} = \sqrt{\frac{252}{4 \ln(2) \cdot N} \sum_{i=1}^N \ln\left(\frac{H_i}{L_i}\right)^2}$$
      *5x more statistically efficient than close-to-close volatility as it captures intraday dispersion.*
    - **Garman-Klass Volatility (OHLC Estimator):**
      $$\sigma_{\text{GK}} = \sqrt{\frac{252}{N} \sum_{i=1}^N \left[ 0.5 \left(\ln \frac{H_i}{L_i}\right)^2 - (2\ln 2 - 1) \left(\ln \frac{C_i}{O_i}\right)^2 \right]}$$
      *Approx. 8x more efficient than standard realized volatility by incorporating overnight jumps and intraday extremes.*
      
    #### C. Risk-Adjusted Performance Ratios
    - **Sharpe Ratio:**
      $$\text{Sharpe} = \frac{\bar{R}_{\text{excess}} \times 252}{\sigma_{\text{ann}}}$$
    - **Sortino Ratio (Downside Semi-Deviation):**
      $$\text{Sortino} = \frac{\bar{R}_{\text{excess}} \times 252}{\sqrt{252} \times \sqrt{\frac{1}{N-1} \sum_{t=1}^N \min(R_t - R_{f,t}, 0)^2}}$$
    - **Maximum Drawdown (MDD):**
      $$\text{DD}_t = \frac{V_t - \max_{s \le t} V_s}{\max_{s \le t} V_s}, \quad \text{MDD} = |\min_t \text{DD}_t|$$
    - **Calmar Ratio:**
      $$\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}}$$
      
    ---
    
    ### 2. Quantitative Research Pitfalls & Guardrails
    
    1. **Look-Ahead Bias Elimination:**
       - All indicators and strategies in QuantEdge are strictly causal. 
       - Signal evaluated at bar $t$ Close is executed at bar $t+1$ Open.
       - Breakout channels are lagged with `.shift(1)` so the current bar cannot breakout against a threshold containing its own extreme.
       
    2. **Data Snooping & Multiple Testing:**
       - Tuning strategy parameters repeatedly on the same historical sample leads to spurious discoveries (Harvey, Liu, Zhu 2016).
       - We provide **Chronological Train/Test Partitioning (70% / 30%)** to audit parameter degradation without shuffling time order.
       
    3. **Survivorship Bias:**
       - When selecting historical instruments, testing only companies that survived to the present date inflates returns. 
       - Always backtest on continuous futures contracts or broad composite indices (`^GSPC`, `SPY`).
       
    4. **Transaction Costs & Slippage Realism:**
       - High-turnover strategies that appear wildly profitable with zero costs frequently collapse under realistic execution drag.
       - QuantEdge embeds configurable commission basis points and adverse execution slippage.
    """)
