"""
QuantEdge – Quantitative Market Research & Strategy Backtesting Platform
An institutional-grade research platform for quantitative time-series analysis, walk-forward validation, and risk modeling.
"""
import io
import sys
import os
import json
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, date, timedelta, timezone
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
    APP_VERSION,
    PLATFORM_NAME,
    INSTRUMENT_CATALOG,
    DEFAULT_TRADING_DAYS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SLIPPAGE_BPS
)
from data.ingestion import MarketDataIngestion, MarketDataPayload
from data.validation import validate_market_data, compute_dataset_fingerprint
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
from analytics.monte_carlo import run_trade_monte_carlo, run_daily_return_block_bootstrap, MonteCarloReport
from analytics.regimes import classify_market_regimes, attribute_performance_by_regime
from analytics.reproducibility import generate_experiment_manifest
from analytics.reporting import generate_pdf_research_report
from strategies.moving_average import MovingAverageCrossover
from strategies.rsi_strategy import RSIMeanReversion
from strategies.breakout_strategy import DonchianBreakout
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.validation import (
    run_out_of_sample_validation,
    run_walk_forward_validation,
    run_parameter_sensitivity_scan,
    run_friction_sensitivity_analysis
)
from visualization.charts import (
    plot_market_explorer_candlestick,
    plot_returns_distribution,
    plot_rolling_volatility_comparison,
    plot_technical_indicators_suite,
    plot_equity_curves,
    plot_drawdown_underwater,
    plot_trade_markers_on_price,
    plot_autocorrelation_bar,
    plot_monte_carlo_fan,
    plot_monte_carlo_drawdown_distribution,
    plot_market_regimes,
    plot_parameter_sensitivity_heatmap,
    plot_friction_sensitivity,
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
    
    /* Monospace Code and Hashes */
    .quant-hash {
        font-family: "JetBrains Mono", Consolas, monospace;
        font-size: 0.78rem;
        background-color: #161B22;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #30363D;
        color: #79C0FF;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONFIGURATION & DATA INGESTION CONTROLS
# ==============================================================================
st.sidebar.markdown(f"""
<div style="padding-bottom: 12px;">
    <span class="quant-header-badge">RESEARCH PLATFORM v{APP_VERSION}</span>
    <h2 style="margin: 0; padding: 0; color: #58A6FF; font-size: 1.5rem; font-weight: 800;">QuantEdge</h2>
    <p style="margin: 0; color: #8B949E; font-size: 0.8rem;">Quantitative Finance & Backtesting Engine</p>
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
# MAIN TERMINAL NAVIGATION (9 SPECIALIZED WORKSPACES)
# ==============================================================================
(
    tab_overview,
    tab_explorer,
    tab_stats,
    tab_indicators,
    tab_strategy,
    tab_backtest,
    tab_regimes,
    tab_risk,
    tab_methodology
) = st.tabs([
    "📊 Terminal Overview",
    "📈 Market Explorer",
    "🔬 Statistical Analysis",
    "📐 Technical Indicators",
    "🧪 Strategy Lab",
    "⚡ Advanced Backtesting",
    "🌐 Market Regimes",
    "🛡️ Risk & Monte Carlo",
    "📚 Methodology & Dossier"
])


# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW & PROVENANCE
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown(f"### {active_symbol} — Quantitative Market Research Overview")
    meta = market_payload.metadata
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.caption("ASSET CLASS & TYPE")
        st.markdown(f"**{market_payload.instrument_type}**")
    with col_t2:
        st.caption("PRIMARY VENUE")
        st.markdown(f"**{meta.get('exchange', 'Exchange')}**")
    with col_t3:
        st.caption("OBSERVATION WINDOW")
        st.markdown(f"**{market_payload.start_date} → {market_payload.end_date}**")
    with col_t4:
        st.caption("DATA FRESHNESS STATUS")
        st.markdown(f"**{market_payload.data_freshness_status}**")

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

    st.markdown('<div class="section-title">Audited Data Provenance & Integrity</div>', unsafe_allow_html=True)
    col_prov1, col_prov2 = st.columns([3, 2])
    with col_prov1:
        st.markdown(f"""
        **Dataset Cryptographic Fingerprint (SHA-256):**  
        <span class="quant-hash">{market_payload.dataset_fingerprint}</span>
        
        **Financial Structure Disclosures:**
        - **Data Mode:** Historical End-of-Day (EOD) archive (*Never mislabeled as zero-latency real-time tick stream*).
        - **Corporate Actions:** {market_payload.corporate_action_notes}
        - **Continuous Futures Limitations:** {market_payload.futures_rollover_notes}
        """, unsafe_allow_html=True)
    with col_prov2:
        gaps = market_payload.validation.gap_diagnostics
        st.info(f"""
        **Data Hygiene Diagnostics**
        - Ingested Bars: `{len(df_market)}`
        - Missing Price Cells: `{sum(market_payload.validation.stats.get('nan_counts', {}).values())}`
        - Duplicate Bars Removed: `{market_payload.validation.stats.get('duplicate_timestamps', 0)}`
        - Max Calendar Gap: `{gaps.get('max_calendar_gap_days', 0)} days`
        - Suspicious Outages (>4d): `{gaps.get('total_gaps_exceeding_threshold', 0)}`
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
# TAB 6: ADVANCED BACKTESTING & WALK-FORWARD VALIDATION
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

    # Benchmark-Relative Metrics row
    b_c1, b_c2, b_c3, b_c4, b_c5 = st.columns(5)
    b_c1.metric("Beta vs Benchmark", f"{m_strat.beta:.2f}")
    b_c2.metric("Jensen's Alpha (Ann.)", f"{m_strat.alpha_annualized * 100:+.2f}%")
    b_c3.metric("Information Ratio", f"{m_strat.information_ratio:.2f}")
    b_c4.metric("Tracking Error", f"{m_strat.tracking_error * 100:.2f}%")
    b_c5.metric("Up / Down Capture", f"{m_strat.up_capture*100:.0f}% / {m_strat.down_capture*100:.0f}%")

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
    comp_rows.append({
        "Strategy": "Buy & Hold Benchmark",
        "Ending Equity": f"${m_bm.ending_capital:,.2f}",
        "Total Return": f"{m_bm.total_return*100:.2f}%",
        "CAGR": f"{m_bm.cagr*100:.2f}%",
        "Sharpe": f"{m_bm.sharpe_ratio:.2f}",
        "Sortino": f"{m_bm.sortino_ratio:.2f}",
        "Max DD": f"{m_bm.max_drawdown*100:.2f}%",
        "Beta": "1.00",
        "Alpha (Ann.)": "0.00%",
        "Info Ratio": "N/A",
        "Win Rate": "N/A",
        "Trades": 1,
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
            "Beta": f"{sm.beta:.2f}",
            "Alpha (Ann.)": f"{sm.alpha_annualized*100:+.2f}%",
            "Info Ratio": f"{sm.information_ratio:.2f}",
            "Win Rate": f"{sm.win_rate*100:.1f}%",
            "Trades": sm.total_trades,
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

    # Walk-Forward Validation & Robustness Diagnostics
    st.markdown('<div class="section-title">Walk-Forward Multi-Fold Chronological Validation</div>', unsafe_allow_html=True)
    st.caption("Partitions dataset into sequential chronological folds. Evaluates out-of-sample persistence and computes Walk-Forward Efficiency (WFE).")
    
    wf_splits = st.slider("Number of Walk-Forward Folds", min_value=2, max_value=6, value=3)
    wf_report: Optional[Any] = None

    if len(df_market) >= 120:
        try:
            wf_report = run_walk_forward_validation(
                df=df_market,
                strategy_factory=lambda: MovingAverageCrossover(20, 50) if "MA" in chosen_strat_name else (RSIMeanReversion(14, 30, 70) if "RSI" in chosen_strat_name else DonchianBreakout(20, 10)),
                engine=engine,
                n_splits=int(wf_splits),
                train_ratio=0.65,
                symbol=active_symbol
            )
            
            w_c1, w_c2, w_c3 = st.columns(3)
            w_c1.metric("Walk-Forward Efficiency (WFE)", f"{wf_report.walk_forward_efficiency*100:.1f}%")
            w_c2.metric("OOS Positive Consistency", f"{wf_report.consistency_rate*100:.0f}%")
            w_c3.metric("Robustness Assessment", wf_report.robustness_rating)

            st.dataframe(wf_report.summary_df, use_container_width=True, hide_index=True)

            if wf_report.warnings:
                for w in wf_report.warnings:
                    st.warning(f"⚠️ {w}")
        except Exception as e:
            st.warning(f"Could not compute walk-forward folds: {e}")
    else:
        st.warning("Insufficient historical bars (<120) for multi-fold walk-forward validation.")

    # Parameter & Friction Sensitivity Section
    st.markdown('<div class="section-title">Parameter & Microstructure Sensitivity Auditing</div>', unsafe_allow_html=True)
    col_sens1, col_sens2 = st.columns(2)
    
    with col_sens1:
        st.markdown("**Parameter Grid Scan (Moving Average)**")
        if st.button("Run 16-Cell MA Sensitivity Scan"):
            grid = []
            for s in [10, 15, 20, 25]:
                for l in [40, 50, 60, 75]:
                    grid.append({"short_period": s, "long_period": l, "ma_type": "SMA"})
            with st.spinner("Scanning parameter surface..."):
                scan_res = run_parameter_sensitivity_scan(
                    df=df_market,
                    strategy_builder=lambda p: MovingAverageCrossover(**p),
                    engine=engine,
                    param_grid=grid,
                    symbol=active_symbol
                )
                st.plotly_chart(
                    plot_parameter_sensitivity_heatmap(scan_res, "short_period", "long_period", "Sharpe_Ratio", title="MA Parameter Sensitivity (Sharpe Ratio)"),
                    use_container_width=True,
                    key="param_sens_heatmap"
                )

    with col_sens2:
        st.markdown("**Transaction Drag & Slippage Sensitivity**")
        if st.button("Audit Friction Breakeven Curve"):
            with st.spinner("Simulating execution frictions..."):
                friction_res = run_friction_sensitivity_analysis(
                    df=df_market,
                    strategy=active_strategy,
                    base_engine=engine,
                    commissions_bps=[0.0, 2.5, 5.0, 10.0, 15.0, 25.0],
                    slippages_bps=[0.0, 2.0, 5.0, 10.0],
                    symbol=active_symbol
                )
                st.plotly_chart(
                    plot_friction_sensitivity(friction_res, title=f"{chosen_strat_name} Sharpe Resilience vs Friction Drag"),
                    use_container_width=True,
                    key="friction_sens_chart"
                )


# ------------------------------------------------------------------------------
# TAB 7: MARKET REGIMES
# ------------------------------------------------------------------------------
with tab_regimes:
    st.markdown('<div class="section-title">Statistical Market Regime Deconstruction</div>', unsafe_allow_html=True)
    st.caption("Segments market history into Volatility Regimes (High / Normal / Low) and Trend Regimes (Bull / Bear / Range-Bound) using transparent statistical rules.")

    market_regimes = classify_market_regimes(df_market, vol_lookback=60, trend_ma_period=50)

    st.plotly_chart(
        plot_market_regimes(df_market, market_regimes, active_symbol),
        use_container_width=True,
        key="market_regimes_chart"
    )

    # Attribution Breakdown
    st.markdown("#### Strategy Performance Attribution by Market Regime")
    regime_attr = attribute_performance_by_regime(
        daily_strategy_returns=active_backtest_res.daily_returns,
        daily_benchmark_returns=active_backtest_res.benchmark_daily_returns,
        regimes=market_regimes,
        trading_days=int(trading_days_input)
    )

    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        st.markdown("**Volatility Regimes Breakdown**")
        st.dataframe(regime_attr.volatility_attribution, use_container_width=True, hide_index=True)
    with reg_col2:
        st.markdown("**Trend Regimes Breakdown**")
        st.dataframe(regime_attr.trend_attribution, use_container_width=True, hide_index=True)

    st.markdown("**Regime Behavior Insights:**")
    for ins in regime_attr.insights:
        st.markdown(f"- {ins}")


# ------------------------------------------------------------------------------
# TAB 8: RISK & MONTE CARLO SIMULATION
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

    # Multi-Estimator Volatility Dynamics
    st.markdown('<div class="section-title">Multi-Estimator Volatility Dynamics</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_rolling_volatility_comparison(daily_returns, df_market, active_symbol, window=21),
        use_container_width=True,
        key="risk_rolling_volatility"
    )

    # Monte Carlo Risk Analysis Section
    st.markdown('<div class="section-title">Monte Carlo Resampling & Risk of Ruin Analysis</div>', unsafe_allow_html=True)
    st.caption("Simulates 1,000 independent equity trajectories via trade-level empirical bootstrapping with replacement.")

    mc_report = run_trade_monte_carlo(
        trades=active_backtest_res.trades,
        initial_capital=float(initial_capital_input),
        num_simulations=1000,
        seed=42
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Median Simulated Ending Equity", f"${mc_report.terminal_equity_median:,.2f}")
    mc2.metric("Worst 5% Outcome (P05)", f"${mc_report.terminal_percentiles['5%']:,.2f}")
    mc3.metric("P(Drawdown >= 20%)", f"{mc_report.probability_of_ruin_20pct * 100:.1f}%")
    mc4.metric("P(Drawdown >= 30%)", f"{mc_report.probability_of_ruin_30pct * 100:.1f}%")

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        st.plotly_chart(
            plot_monte_carlo_fan(mc_report, title=f"{chosen_strat_name} Monte Carlo Trajectory Fan (5th to 95th Percentile)"),
            use_container_width=True,
            key="monte_carlo_fan"
        )
    with col_mc2:
        st.plotly_chart(
            plot_monte_carlo_drawdown_distribution(mc_report),
            use_container_width=True,
            key="monte_carlo_dd_dist"
        )

    with st.expander("Methodological Assumptions of Monte Carlo Simulation"):
        for a in mc_report.resampling_assumptions:
            st.markdown(f"- {a}")


# ------------------------------------------------------------------------------
# TAB 9: METHODOLOGY, DOSSIER & REPRODUCIBILITY
# ------------------------------------------------------------------------------
with tab_methodology:
    st.markdown('<div class="section-title">Institutional Research Dossier & Experiment Manifest</div>', unsafe_allow_html=True)

    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.markdown("#### 📄 Export Institutional PDF Dossier")
        st.caption("Generates a publication-grade PDF report containing provenance, distribution moments, strategy scorecards, walk-forward diagnostics, and risk disclosures.")
        
        pdf_bytes = generate_pdf_research_report(
            backtest_result=active_backtest_res,
            dist_stats=dist_stats,
            dataset_fingerprint=market_payload.dataset_fingerprint,
            data_source=market_payload.source,
            instrument_type=market_payload.instrument_type,
            corporate_action_notes=market_payload.corporate_action_notes,
            futures_rollover_notes=market_payload.futures_rollover_notes,
            monte_carlo_report=mc_report,
            walk_forward_summary=wf_report.summary_df if wf_report is not None else None,
            experiment_id=f"EXP-{active_symbol}-{datetime.now().strftime('%Y%m%d')}"
        )
        
        st.download_button(
            label="⬇️ Download Institutional PDF Research Report",
            data=pdf_bytes,
            file_name=f"QuantEdge_{active_symbol}_{chosen_strat_name.replace(' ', '_')}_Report.pdf",
            mime="application/pdf"
        )

    with col_rep2:
        st.markdown("#### 📋 Cryptographic Reproducibility Manifest (JSON)")
        st.caption("Export exact audit trail with dataset SHA-256 hash, parameters, friction settings, and software version.")
        
        manifest = generate_experiment_manifest(
            symbol=active_symbol,
            dataset_fingerprint=market_payload.dataset_fingerprint,
            data_source=market_payload.source,
            date_start=market_payload.start_date,
            date_end=market_payload.end_date,
            row_count=len(df_market),
            strategy_name=chosen_strat_name,
            strategy_params=active_backtest_res.metrics.to_dict(),
            initial_capital=float(initial_capital_input),
            commission_bps=float(commission_bps_input),
            slippage_bps=float(slippage_bps_input),
            execution_mode=exec_mode_input,
            risk_free_rate=float(rf_rate_input),
            trading_days=int(trading_days_input),
            validation_type=f"Walk-Forward {wf_splits}-Fold" if wf_report else "Out-of-Sample",
            metrics_dict=active_backtest_res.metrics.to_dict()
        )
        
        st.download_button(
            label="⬇️ Export Reproducibility Manifest (JSON)",
            data=manifest.to_json(),
            file_name=f"{manifest.experiment_id}_manifest.json",
            mime="application/json"
        )

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
      
    #### C. Benchmark-Relative Performance Ratios
    - **Jensen's Alpha (Annualized):**
      $$\alpha = \bar{R}_{\text{strategy}} \times 252 - \left[R_f + \beta (\bar{R}_{\text{benchmark}} \times 252 - R_f)\right]$$
    - **Beta:**
      $$\beta = \frac{\text{Cov}(R_{\text{strategy}}, R_{\text{benchmark}})}{\text{Var}(R_{\text{benchmark}})}$$
    - **Information Ratio:**
      $$\text{IR} = \frac{(\bar{R}_{\text{strategy}} - \bar{R}_{\text{benchmark}}) \times 252}{\sigma(R_{\text{strategy}} - R_{\text{benchmark}}) \times \sqrt{252}}$$
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
       - We provide **Chronological Train/Test Partitioning** and **Multi-Fold Walk-Forward Validation** to audit parameter degradation without shuffling time order.
       
    3. **Survivorship Bias & Futures Rollover:**
       - Stitched continuous futures (`GC=F`, `CL=F`) contain unadjusted roll yield gaps upon contract transition.
       - Cash ETFs (`SPY`, `QQQ`) incorporate management expense ratios and cash dividends.
       
    4. **Transaction Costs & Slippage Realism:**
       - High-turnover strategies that appear wildly profitable with zero costs frequently collapse under realistic execution drag.
       - QuantEdge embeds configurable commission basis points and adverse execution slippage.
    """)
