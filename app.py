"""
QuantEdge v3 – Institutional Quantitative Research & Financial Market Analytics Platform.

A high-performance research terminal providing:
- Multi-Asset Ingestion & Provenance (NSE, BSE, US Equities, Global Commodities, FX)
- Futures-Specific Accounting, Margin Modeling, and Variation Margin Settlement
- Comprehensive Indian Market Taxonomy (Broad, Sectoral, Thematic, Strategy, BSE, Fixed Income, Hybrid)
- 26 Audited Statistical & Financial Risk Metrics
- Markowitz Portfolio Optimization, Rolling Beta, and Multi-Asset Correlation
- Walk-Forward Validation, Monte Carlo Resampling, and Market Regime Analysis
- Institutional Vector PDF Research Dossier (ReportLab) & JSON Reproducibility Manifest
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
    page_title="QuantEdge v3 | Institutional Quantitative Terminal",
    page_icon="🏛️",
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
    DEFAULT_SLIPPAGE_BPS,
    UI_THEME_DARK,
    UI_THEME_LIGHT
)
from config.instruments import (
    INSTRUMENT_REGISTRY,
    InstrumentMetadata,
    AvailabilityStatus,
    get_all_families,
    get_instruments_by_family,
    get_available_online_instruments,
    get_instrument_by_symbol_or_key
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
from analytics.risk import (
    compute_drawdowns,
    compute_downside_deviation,
    compute_historical_var,
    compute_parametric_var,
    compute_conditional_var
)
from analytics.monte_carlo import run_trade_monte_carlo, run_daily_return_block_bootstrap, MonteCarloReport
from analytics.regimes import classify_market_regimes, attribute_performance_by_regime
from analytics.futures_accounting import (
    get_futures_spec,
    simulate_futures_account,
    FuturesSimulationResult,
    FUTURES_SPECS_REGISTRY
)
from analytics.portfolio import (
    optimize_portfolio,
    compute_rolling_beta,
    compute_rolling_correlation,
    compute_risk_contribution,
    PortfolioOptimizationResult
)
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
    plot_futures_margin_and_equity,
    plot_efficient_frontier,
    plot_portfolio_allocation,
    plot_correlation_heatmap,
    plot_rolling_beta,
    apply_quant_layout
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Institutional Dark"

# Theme Toggle in Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="padding-bottom: 8px;">
        <span style="background: #1E3A8A; color: #93C5FD; font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.08em; text-transform: uppercase;">
            INSTITUTIONAL v{APP_VERSION}
        </span>
        <h2 style="margin: 4px 0 0 0; color: #38BDF8; font-size: 1.4rem; font-weight: 800; font-family: 'Inter', sans-serif;">QuantEdge</h2>
        <p style="margin: 0; color: #94A3B8; font-size: 0.75rem;">Market Research & Strategy Backtesting Terminal</p>
    </div>
    """, unsafe_allow_html=True)

    theme_select = st.selectbox(
        "Terminal Display Theme",
        ["Institutional Dark", "Clean Light"],
        index=0 if st.session_state["theme_mode"] == "Institutional Dark" else 1
    )
    st.session_state["theme_mode"] = theme_select

is_dark = st.session_state["theme_mode"] == "Institutional Dark"
theme_key = "dark" if is_dark else "light"
theme_cfg = UI_THEME_DARK if is_dark else UI_THEME_LIGHT

# Apply Dynamic CSS Styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    .stApp {{
        background-color: {theme_cfg["bg_primary"]};
        color: {theme_cfg["text_primary"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
    }}
    
    div[data-testid="stMetric"] {{
        background-color: {theme_cfg["bg_card"]};
        border: 1px solid {theme_cfg["border_color"]};
        border-radius: 6px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 1.35rem !important;
        font-weight: 700;
        color: {theme_cfg["accent_blue"]} !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.75rem !important;
        font-weight: 600;
        color: {theme_cfg["text_secondary"]} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {theme_cfg["text_primary"]};
        border-bottom: 2px solid {theme_cfg["border_color"]};
        padding-bottom: 5px;
        margin-top: 14px;
        margin-bottom: 12px;
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .badge-online {{ background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981; }}
    .badge-csv {{ background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid #F59E0B; }}
    
    .quant-hash {{
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.75rem;
        background-color: {theme_cfg["bg_card"]};
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid {theme_cfg["border_color"]};
        color: {theme_cfg["accent_cyan"]};
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: DATA INGESTION & SEARCHABLE REGISTRY SELECTOR
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="section-title">1. Market Ingestion & Universe</div>', unsafe_allow_html=True)

    data_source_mode = st.radio(
        "Source Provider",
        ["Instrument Registry (Live / Online)", "CSV File Upload", "Pre-Packaged Offline Sample"],
        help="Select live historical API, custom OHLCV CSV, or deterministic offline fixtures."
    )

    ingestion_engine = MarketDataIngestion(cache_enabled=True)
    market_payload: Optional[MarketDataPayload] = None
    ingestion_error: Optional[str] = None
    selected_meta: Optional[InstrumentMetadata] = None

    if data_source_mode == "Instrument Registry (Live / Online)":
        all_fams = ["ALL"] + get_all_families()
        chosen_family = st.selectbox("Filter Category / Index Family", all_fams, index=0)

        if chosen_family == "ALL":
            available_items = list(INSTRUMENT_REGISTRY.values())
        else:
            available_items = get_instruments_by_family(chosen_family)

        def format_instrument_entry(item: InstrumentMetadata) -> str:
            status_indicator = "✓" if item.availability == AvailabilityStatus.AVAILABLE_ONLINE else "⚠ (CSV)"
            sym_str = f"[{item.symbol}]" if item.symbol else "[CSV Required]"
            return f"{status_indicator} {item.name} {sym_str} — {item.exchange}"

        selected_meta = st.selectbox(
            "Select Searchable Instrument",
            available_items,
            format_func=format_instrument_entry,
            help="Choose from official NSE, BSE, US, or Commodity index and derivative families."
        )

        active_ticker = selected_meta.symbol if selected_meta and selected_meta.symbol else None

        # Check if selected instrument requires CSV
        if selected_meta and selected_meta.availability == AvailabilityStatus.UNAVAILABLE_YFINANCE_REQUIRES_CSV:
            st.warning(f"⚠️ {selected_meta.name} does not have a free public Yahoo Finance ticker.")
            st.caption(f"Reason: {selected_meta.notes}")
            st.markdown("**Please upload the official exchange CSV below:**")
            up_csv = st.file_uploader(f"Upload CSV for {selected_meta.name}", type=["csv"], key="csv_req_up")
            if up_csv is not None:
                try:
                    market_payload = ingestion_engine.load_csv(up_csv, ticker_name=selected_meta.name)
                except Exception as e:
                    ingestion_error = str(e)
            else:
                st.info("Or fallback to the pre-packaged SPY sample to evaluate features offline.")
                if st.button("Load Offline Sample SPY"):
                    sample_path = os.path.join(CURRENT_DIR, "sample_data", "sample_sp500.csv")
                    market_payload = ingestion_engine.load_csv(sample_path, ticker_name="SPY_SAMPLE")
        else:
            # Online fetch
            default_end = date.today()
            default_start = default_end - timedelta(days=365 * 3)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input("Start Date", value=default_start)
            with col_d2:
                end_date = st.date_input("End Date", value=default_end)

            if start_date >= end_date:
                st.error("Start Date must be before End Date.")
            else:
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

    elif data_source_mode == "CSV File Upload":
        uploaded_csv = st.file_uploader("Upload Clean OHLCV CSV", type=["csv"], help="Must contain Date, Open, High, Low, Close, Volume.")
        custom_name = st.text_input("Dataset Label", value="CUSTOM_SERIES")
        if uploaded_csv is not None:
            try:
                market_payload = ingestion_engine.load_csv(uploaded_csv, ticker_name=custom_name)
            except Exception as e:
                ingestion_error = str(e)
        else:
            st.info("Upload your custom backtest time-series file.")

    else:
        sample_path = os.path.join(CURRENT_DIR, "sample_data", "sample_sp500.csv")
        try:
            market_payload = ingestion_engine.load_csv(sample_path, ticker_name="SPY_SAMPLE")
        except Exception as e:
            ingestion_error = str(e)

    # Simulation Parameters
    st.markdown('<div class="section-title">2. Simulation Parameters</div>', unsafe_allow_html=True)
    is_inr = selected_meta.currency == "INR" if selected_meta else False
    default_rf = 6.5 if is_inr else 4.0
    default_tdays = 250 if is_inr else 252

    trading_days_input = st.number_input("Trading Days / Year", min_value=200, max_value=365, value=default_tdays, step=1)
    rf_rate_input = st.number_input("Risk-Free Rate (Annual %)", min_value=0.0, max_value=20.0, value=float(default_rf), step=0.25) / 100.0
    initial_capital_input = st.number_input("Initial Capital", min_value=1_000.0, max_value=100_000_000.0, value=DEFAULT_INITIAL_CAPITAL, step=10_000.0)

    st.markdown('<div class="section-title">3. Execution Frictions</div>', unsafe_allow_html=True)
    commission_bps_input = st.number_input("Commission (bps)", min_value=0.0, max_value=100.0, value=DEFAULT_COMMISSION_BPS, step=0.5)
    slippage_bps_input = st.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=DEFAULT_SLIPPAGE_BPS, step=0.5)
    exec_mode_input = st.selectbox("Execution Timing", ["next_open", "next_close"], help="next_open eliminates signal look-ahead bias.")

    # 13 Dedicated Vertical Navigation Items
    st.markdown('<div class="section-title">4. Navigation Workspace</div>', unsafe_allow_html=True)
    NAV_WORKSPACES = [
        "1. Overview",
        "2. Market Explorer",
        "3. Indian Markets",
        "4. Global Markets",
        "5. Statistical Research",
        "6. Technical Analysis",
        "7. Strategy Lab",
        "8. Backtesting",
        "9. Risk Analytics",
        "10. Futures Analytics",
        "11. Portfolio Research",
        "12. Research Reports",
        "13. Methodology"
    ]
    active_workspace = st.radio("Select Research View", NAV_WORKSPACES, index=0)


# ==============================================================================
# DATA INTEGRITY GATEKEEPER
# ==============================================================================
if ingestion_error:
    st.error(f"⚠️ Market Data Ingestion Error: {ingestion_error}")
    st.info("Tip: If live internet or Yahoo Finance is unreachable, switch data mode to 'Pre-Packaged Offline Sample' in the sidebar.")
    st.stop()

if market_payload is None or market_payload.df.empty:
    st.warning("Please select or upload market data using the sidebar to commence analysis.")
    st.stop()

df_market = market_payload.df
active_symbol = market_payload.ticker
currency_label = selected_meta.currency if selected_meta else "USD"

# Compute Base Return Series
daily_returns = compute_daily_returns(df_market["Close"], method="arithmetic")
cum_returns = compute_cumulative_returns(daily_returns, method="arithmetic")


# ==============================================================================
# INSTITUTIONAL TOP HEADER STATUS BAR
# ==============================================================================
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([3, 2, 2, 2, 3])
with col_h1:
    inst_name = selected_meta.name if selected_meta else active_symbol
    st.markdown(f"### {active_symbol} <span style='font-size:0.9rem; color:{theme_cfg['text_secondary']};'>({inst_name})</span>", unsafe_allow_html=True)
with col_h2:
    st.caption("PRIMARY VENUE / ASSET")
    st.markdown(f"**{market_payload.metadata.get('exchange', 'Exchange')} | {market_payload.instrument_type}**")
with col_h3:
    st.caption("CURRENCY")
    st.markdown(f"**{currency_label}**")
with col_h4:
    st.caption("DATA FRESHNESS")
    st.markdown(f"<span class='status-badge badge-online'>{market_payload.data_freshness_status}</span>", unsafe_allow_html=True)
with col_h5:
    st.caption("SHA-256 DATASET FINGERPRINT")
    short_hash = f"{market_payload.dataset_fingerprint[:12]}...{market_payload.dataset_fingerprint[-8:]}"
    st.markdown(f"<span class='quant-hash' title='{market_payload.dataset_fingerprint}'>{short_hash}</span>", unsafe_allow_html=True)

st.markdown("---")


# ==============================================================================
# WORKSPACE 1: OVERVIEW & PROVENANCE
# ==============================================================================
if active_workspace == "1. Overview":
    st.markdown(f"#### Executive Research Summary — {active_symbol}")

    c_last = float(df_market["Close"].iloc[-1])
    c_prev = float(df_market["Close"].iloc[-2]) if len(df_market) > 1 else c_last
    d_chg = (c_last - c_prev) / c_prev
    tot_ret = float(cum_returns.iloc[-1])
    ann_vol = compute_realized_volatility(daily_returns, trading_days=int(trading_days_input))
    dd_summary = compute_drawdowns(df_market["Close"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Latest Close", f"{currency_label} {c_last:,.2f}", delta=f"{d_chg * 100:+.2f}%")
    m2.metric("Cumulative Return", f"{tot_ret * 100:+.2f}%")
    m3.metric("Annualized Volatility", f"{ann_vol * 100:.2f}%")
    m4.metric("Max Drawdown", f"-{dd_summary.max_drawdown * 100:.2f}%")
    m5.metric("Observation Count", f"{len(df_market):,} bars")

    st.markdown('<div class="section-title">Cryptographic Provenance & Data Hygiene Audit</div>', unsafe_allow_html=True)
    c_prov1, c_prov2 = st.columns([3, 2])
    with c_prov1:
        st.markdown(f"""
        **Full Dataset SHA-256 Fingerprint:**  
        <span class="quant-hash">{market_payload.dataset_fingerprint}</span>
        
        **Provenance & Methodology Disclosures:**
        - **Data Mode:** Historical End-of-Day (EOD) verified observations.
        - **Corporate Actions:** {market_payload.corporate_action_notes}
        - **Continuous Futures Limitations:** {market_payload.futures_rollover_notes}
        """, unsafe_allow_html=True)
    with c_prov2:
        gaps = market_payload.validation.gap_diagnostics
        st.info(f"""
        **Data Validation Audit**
        - Total Valid Bars: `{len(df_market)}`
        - Missing Price Cells: `{sum(market_payload.validation.stats.get('nan_counts', {}).values())}`
        - Duplicate Bars Scrubbed: `{market_payload.validation.stats.get('duplicate_timestamps', 0)}`
        - Calendar Outages (>4d): `{gaps.get('total_gaps_exceeding_threshold', 0)}`
        - Max Calendar Gap: `{gaps.get('max_calendar_gap_days', 0)} days`
        """)

    st.plotly_chart(
        plot_market_explorer_candlestick(df_market, active_symbol, show_volume=True),
        use_container_width=True,
        key="overview_candlestick_chart"
    )


# ==============================================================================
# WORKSPACE 2: MARKET EXPLORER
# ==============================================================================
elif active_workspace == "2. Market Explorer":
    st.markdown('<div class="section-title">Market Explorer & Price Action Dynamics</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([3, 1])
    with col_e2:
        show_vol = st.checkbox("Display Trading Volume", value=True)
        show_ma = st.checkbox("Overlay 20/50/200 SMAs", value=True)

    st.plotly_chart(
        plot_market_explorer_candlestick(df_market, active_symbol, show_volume=show_vol),
        use_container_width=True,
        key="explorer_main_chart"
    )

    st.markdown('<div class="section-title">Cleaned Historical Market Feed</div>', unsafe_allow_html=True)
    st.dataframe(df_market.tail(20), use_container_width=True)
    st.download_button(
        "Download Sanitized OHLCV CSV",
        data=df_market.to_csv().encode("utf-8"),
        file_name=f"{active_symbol}_clean_ohlcv.csv",
        mime="text/csv"
    )


# ==============================================================================
# WORKSPACE 3: INDIAN MARKETS (COMPREHENSIVE COVERAGE)
# ==============================================================================
elif active_workspace == "3. Indian Markets":
    st.markdown('<div class="section-title">NSE & BSE Indian Market Universe Directory</div>', unsafe_allow_html=True)
    st.markdown("""
    Explore the complete official taxonomy of Indian capital markets categorized across **NSE Broad Market**, **NSE Sectoral**, 
    **NSE Thematic**, **NSE Strategy (Smart Beta)**, **BSE Benchmark Indices**, **Indian Market Indicators**, **Fixed Income (G-Sec)**, and **Hybrid Indices**.
    """)

    fam_filter = st.selectbox(
        "Filter Indian Family Segment",
        ["ALL INDIAN MARKETS", "NSE Broad Market", "NSE Sectoral", "NSE Thematic", "NSE Strategy", "BSE Indices", "Indian Indicators", "Fixed Income", "Hybrid Indices"]
    )

    if fam_filter == "ALL INDIAN MARKETS":
        indian_items = [meta for meta in INSTRUMENT_REGISTRY.values() if "NSE" in meta.index_family or "BSE" in meta.index_family or meta.currency == "INR" or meta.index_family in ["Fixed Income", "Hybrid Indices"]]
    else:
        indian_items = get_instruments_by_family(fam_filter)

    records = []
    for item in indian_items:
        records.append({
            "Instrument Name": item.name,
            "Exchange": item.exchange,
            "Category": item.index_family,
            "Symbol / Ticker": item.symbol or "CSV_REQUIRED",
            "Availability": "✓ Live Download" if item.availability == AvailabilityStatus.AVAILABLE_ONLINE else "⚠ CSV Upload Required",
            "ETF Proxy": item.etf_proxy or "None",
            "Notes & Provenance": item.notes
        })

    ind_df = pd.DataFrame(records)
    st.dataframe(ind_df, use_container_width=True, height=450)

    st.info("""
    **Data Provenance Note on Indian Indices:**  
    Indices marked with `✓ Live Download` are retrieved directly via verified Yahoo Finance tickers (e.g. `^NSEI`, `^BSESN`, `^NSEBANK`, `^CNXIT`, `^CNXAUTO`, `^INDIAVIX`, `NIFTYBEES.NS`).  
    Specialized strategy and thematic indices without free public tickers are marked with `⚠ CSV Upload Required`. Under no circumstances does QuantEdge substitute false proxies silently.
    """)


# ==============================================================================
# WORKSPACE 4: GLOBAL MARKETS
# ==============================================================================
elif active_workspace == "4. Global Markets":
    st.markdown('<div class="section-title">Global Equities, Commodities & Benchmark Comparisons</div>', unsafe_allow_html=True)

    global_symbols = ["^GSPC", "^NDX", "SPY", "QQQ", "GC=F", "CL=F", "ZN=F"]
    st.markdown(f"**Active Selected Asset:** `{active_symbol}`")

    st.info("""
    **Global Macro Benchmarks:**
    - **S&P 500 (`^GSPC` / `SPY`):** US Large-Cap Equity Benchmark.
    - **NASDAQ-100 (`^NDX` / `QQQ`):** Tech & Innovation Equities.
    - **Gold Continuous (`GC=F`):** Inflation & Currency Hedge.
    - **Crude Oil WTI (`CL=F`):** Global Industrial Energy Demand.
    - **10-Year US Treasury (`ZN=F`):** Sovereign Risk-Free Benchmark.
    """)

    st.plotly_chart(
        plot_market_explorer_candlestick(df_market, active_symbol, show_volume=True),
        use_container_width=True,
        key="global_candlestick_view"
    )


# ==============================================================================
# WORKSPACE 5: STATISTICAL RESEARCH
# ==============================================================================
elif active_workspace == "5. Statistical Research":
    st.markdown('<div class="section-title">Higher-Order Statistical Moments & Distribution Properties</div>', unsafe_allow_html=True)

    dist_stats = compute_distribution_statistics(daily_returns, trading_days=int(trading_days_input))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean Annualized Return", f"{dist_stats.mean_annualized * 100:+.2f}%")
    s2.metric("Annualized Volatility", f"{dist_stats.volatility_annualized * 100:.2f}%")
    s3.metric("Sample Skewness", f"{dist_stats.skewness:+.3f}")
    s4.metric("Excess Kurtosis", f"{dist_stats.excess_kurtosis:+.3f}")

    jb_status = "Gaussian (Normal)" if dist_stats.is_normal_5pct else "Non-Gaussian (Fat-Tailed)"
    st.markdown(rf"""
    **Normality Evaluation:** Jarque-Bera Test Statistic = `{dist_stats.jarque_bera_stat:.2f}` (p-value = `{dist_stats.jarque_bera_pvalue:.4e}`)  
    **Null Hypothesis ($H_0$):** Returns are normally distributed $\implies$ **{jb_status}** at $\alpha = 0.05$.
    """)

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.plotly_chart(plot_returns_distribution(daily_returns, dist_stats, active_symbol), use_container_width=True, key="dist_hist_chart")
    with c_g2:
        st.plotly_chart(plot_autocorrelation_bar(dist_stats.autocorr_lags, active_symbol), use_container_width=True, key="autocorr_chart")

    st.markdown('<div class="section-title">Automated Quantitative Diagnostic Interpretations</div>', unsafe_allow_html=True)
    for interp in dist_stats.interpretations:
        st.info(f"💡 {interp}")


# ==============================================================================
# WORKSPACE 6: TECHNICAL ANALYSIS
# ==============================================================================
elif active_workspace == "6. Technical Analysis":
    st.markdown('<div class="section-title">Multi-Indicator Technical Analysis Suite</div>', unsafe_allow_html=True)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        rsi_p = st.slider("RSI Lookback Period", min_value=5, max_value=30, value=14)
    with col_t2:
        fast_ma = st.slider("MACD Fast EMA", min_value=5, max_value=20, value=12)
        slow_ma = st.slider("MACD Slow EMA", min_value=21, max_value=50, value=26)
    with col_t3:
        atr_p = st.slider("ATR Lookback Period", min_value=5, max_value=30, value=14)
        bb_std = st.slider("Bollinger Bands Std Dev", min_value=1.0, max_value=3.0, value=2.0, step=0.5)

    rsi_series = compute_rsi(df_market["Close"], period=rsi_p)
    macd_df = compute_macd(df_market["Close"], fast_period=fast_ma, slow_period=slow_ma, signal_period=9)
    atr_series = compute_atr(df_market, period=atr_p)
    bb_df = compute_bollinger_bands(df_market["Close"], period=20, num_std=bb_std)

    st.plotly_chart(
        plot_technical_indicators_suite(df_market, rsi_series, macd_df, bb_df, atr_series, active_symbol),
        use_container_width=True,
        key="tech_indicators_suite_chart"
    )


# ==============================================================================
# WORKSPACE 7: STRATEGY LAB
# ==============================================================================
elif active_workspace == "7. Strategy Lab":
    st.markdown('<div class="section-title">Strategy Formulation & Signal Dynamics</div>', unsafe_allow_html=True)

    strat_choice = st.selectbox(
        "Select Quantitative Strategy Archetype",
        ["Moving Average Crossover", "RSI Mean Reversion", "Donchian Breakout"]
    )

    if strat_choice == "Moving Average Crossover":
        c1, c2 = st.columns(2)
        f_win = c1.number_input("Fast MA Window", min_value=5, max_value=50, value=20)
        s_win = c2.number_input("Slow MA Window", min_value=20, max_value=200, value=50)
        strat = MovingAverageCrossover(fast_window=f_win, slow_window=s_win)
    elif strat_choice == "RSI Mean Reversion":
        c1, c2, c3 = st.columns(3)
        rp = c1.number_input("RSI Period", min_value=5, max_value=30, value=14)
        ob = c2.number_input("Overbought Level", min_value=60, max_value=90, value=70)
        os_level = c3.number_input("Oversold Level", min_value=10, max_value=40, value=30)
        strat = RSIMeanReversion(rsi_period=rp, overbought=ob, oversold=os_level)
    else:
        c1, c2 = st.columns(2)
        entry_b = c1.number_input("Channel Lookback (Bars)", min_value=10, max_value=60, value=20)
        exit_b = c2.number_input("Channel Exit (Bars)", min_value=5, max_value=30, value=10)
        strat = DonchianBreakout(lookback_bars=entry_b, exit_bars=exit_b)

    sig_series = strat.generate_signals(df_market)
    pos_series = strat.generate_positions(sig_series)

    st.plotly_chart(
        plot_trade_markers_on_price(df_market, pos_series, active_symbol),
        use_container_width=True,
        key="strategy_trade_markers_chart"
    )


# ==============================================================================
# WORKSPACE 8: BACKTESTING (DISCRETE-EVENT ENGINE)
# ==============================================================================
elif active_workspace == "8. Backtesting":
    st.markdown('<div class="section-title">Discrete-Event Backtesting & Execution Engine</div>', unsafe_allow_html=True)

    strat_choice = st.selectbox(
        "Strategy for Backtest Evaluation",
        ["Moving Average Crossover (20/50)", "RSI Mean Reversion (14/30/70)", "Donchian Breakout (20/10)"],
        key="bt_strat_choice"
    )

    if "Moving Average" in strat_choice:
        strategy_obj = MovingAverageCrossover(fast_window=20, slow_window=50)
    elif "RSI" in strat_choice:
        strategy_obj = RSIMeanReversion(rsi_period=14, overbought=70, oversold=30)
    else:
        strategy_obj = DonchianBreakout(lookback_bars=20, exit_bars=10)

    # Initialize Engine
    engine = BacktestEngine(
        initial_capital=float(initial_capital_input),
        commission_bps=float(commission_bps_input),
        slippage_bps=float(slippage_bps_input),
        execution_timing=exec_mode_input,
        risk_free_rate=float(rf_rate_input),
        trading_days=int(trading_days_input)
    )

    with st.spinner("Running causality-enforced discrete-event simulation..."):
        bt_res = engine.run(df_market, strategy_obj, symbol=active_symbol)

    metrics = bt_res.metrics
    b_metrics = bt_res.benchmark_metrics

    # Scorecard
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Strategy Ending Capital", f"{currency_label} {metrics.ending_capital:,.2f}", delta=f"{metrics.total_return * 100:+.2f}%")
    b2.metric("Annualized Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    b3.metric("Annualized Sortino", f"{metrics.sortino_ratio:.2f}")
    b4.metric("Max Drawdown", f"-{metrics.max_drawdown * 100:.2f}%")
    b5.metric("Profit Factor", f"{metrics.profit_factor:.2f}" if not np.isinf(metrics.profit_factor) else "Inf")

    # CAPM Alpha/Beta Scorecard
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAPM Beta vs Benchmark", f"{metrics.beta:.2f}")
    c2.metric("Jensen's Alpha (Ann.)", f"{metrics.alpha_annualized * 100:+.2f}%")
    c3.metric("Information Ratio", f"{metrics.information_ratio:.2f}")
    c4.metric("Tracking Error", f"{metrics.tracking_error * 100:.2f}%")

    st.plotly_chart(
        plot_equity_curves(bt_res.equity_curve, bt_res.benchmark_equity, active_symbol, strategy_obj.name),
        use_container_width=True,
        key="bt_equity_curves_chart"
    )

    st.plotly_chart(
        plot_drawdown_underwater(bt_res.drawdown_analysis.drawdown_series, active_symbol),
        use_container_width=True,
        key="bt_drawdown_chart"
    )

    st.markdown('<div class="section-title">Closed Round-Trip Trade Audit Log</div>', unsafe_allow_html=True)
    if bt_res.trades:
        trade_rows = []
        for t in bt_res.trades:
            trade_rows.append({
                "ID": t.trade_id, "Direction": t.direction, "Entry Date": t.entry_date, "Exit Date": t.exit_date,
                "Entry Price": f"{t.entry_price:.2f}", "Exit Price": f"{t.exit_price:.2f}",
                "Shares": f"{t.shares:.2f}", "Gross PnL": f"{t.gross_pnl:+,.2f}",
                "Fees": f"{t.entry_cost + t.exit_cost:,.2f}", "Net PnL": f"{t.net_pnl:+,.2f}",
                "Return %": f"{t.return_pct * 100:+.2f}%", "Bars": t.duration_bars
            })
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, height=250)
    else:
        st.info("No completed round-trip trades executed during this backtest period.")


# ==============================================================================
# WORKSPACE 9: RISK ANALYTICS
# ==============================================================================
elif active_workspace == "9. Risk Analytics":
    st.markdown('<div class="section-title">Value at Risk (VaR), CVaR & Downside Analytics</div>', unsafe_allow_html=True)

    h_var95 = compute_historical_var(daily_returns, 0.95)
    h_var99 = compute_historical_var(daily_returns, 0.99)
    p_var95 = compute_parametric_var(daily_returns, 0.95)
    c_var95 = compute_conditional_var(daily_returns, 0.95)
    down_vol = compute_downside_deviation(daily_returns, mar_daily=0.0, trading_days=int(trading_days_input))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("1-Day Historical VaR (95%)", f"{h_var95 * 100:.2f}%")
    r2.metric("1-Day Parametric VaR (95%)", f"{p_var95 * 100:.2f}%")
    r3.metric("1-Day CVaR (Expected Shortfall)", f"{c_var95 * 100:.2f}%")
    r4.metric("Downside Volatility", f"{down_vol * 100:.2f}%")

    # Rolling Volatility Estimator Comparison
    p_vol = compute_parkinson_volatility(df_market, window=21, trading_days=int(trading_days_input))
    gk_vol = compute_garman_klass_volatility(df_market, window=21, trading_days=int(trading_days_input))
    ew_vol = compute_ewma_volatility(daily_returns, decay_factor=0.94, trading_days=int(trading_days_input))

    st.plotly_chart(
        plot_rolling_volatility_comparison(daily_returns, p_vol, gk_vol, ew_vol, active_symbol),
        use_container_width=True,
        key="risk_rolling_vol_chart"
    )


# ==============================================================================
# WORKSPACE 10: FUTURES ANALYTICS (PHASE 2)
# ==============================================================================
elif active_workspace == "10. Futures Analytics":
    st.markdown('<div class="section-title">Futures Contract Simulation, Margin & Mark-to-Market Accounting</div>', unsafe_allow_html=True)
    st.markdown("""
    This module simulates an authentic marked-to-market futures margin account with:
    - **Contract Multiplier** and **Tick Value** scaling.
    - **Daily Variation Margin** cash debits and credits.
    - **Initial Margin & Maintenance Margin** tracking with automated **Margin Call** detection.
    - Direct structural comparison against cash ETF and spot index returns.
    """)

    fut_spec = get_futures_spec(active_symbol)

    col_fs1, col_fs2, col_fs3, col_fs4 = st.columns(4)
    col_fs1.metric("Contract Multiplier", f"{fut_spec.multiplier:,.0f}x")
    col_fs2.metric("Tick Size / Value", f"{fut_spec.tick_size} / {fut_spec.currency} {fut_spec.tick_value:.2f}")
    col_fs3.metric("Initial Margin Rate", f"{fut_spec.initial_margin_rate * 100:.1f}%")
    col_fs4.metric("Maintenance Margin Rate", f"{fut_spec.maintenance_margin_rate * 100:.1f}%")

    st.caption(f"**Rollover Cycle:** {fut_spec.rollover_cycle} | **Exchange:** {fut_spec.exchange}")
    st.info(f"**Futures Stitching Caveat:** {fut_spec.stitching_caveats}")

    num_contracts = st.slider("Contracts Traded Per Signal", min_value=1, max_value=10, value=1)
    slip_ticks = st.slider("Adverse Slippage (Ticks)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

    # Strategy signals
    strat_obj = MovingAverageCrossover(fast_window=20, slow_window=50)
    sig_s = strat_obj.generate_signals(df_market)

    with st.spinner("Simulating marked-to-market futures margin dynamics..."):
        fut_res = simulate_futures_account(
            df_market,
            sig_s,
            fut_spec,
            initial_cash=float(initial_capital_input),
            max_contracts=num_contracts,
            slippage_ticks=slip_ticks
        )

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Futures Ending Cash", f"{fut_spec.currency} {fut_res.ending_cash:,.2f}", delta=f"{fut_res.return_on_initial_capital * 100:+.2f}%")
    f2.metric("Net Futures P&L", f"{fut_spec.currency} {fut_res.net_profit:,.2f}")
    f3.metric("Total Variation Margin", f"{fut_spec.currency} {fut_res.total_variation_margin:,.2f}")
    f4.metric("Margin Calls Triggered", f"{fut_res.margin_calls_triggered}")

    st.plotly_chart(
        plot_futures_margin_and_equity(fut_res, theme=theme_key),
        use_container_width=True,
        key="futures_equity_margin_chart"
    )

    st.markdown('<div class="section-title">Structural Comparison: ETF vs Spot vs Futures Contract P&L</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame([
        {"Structure": "A. Cash ETF Return", "Characteristics": "100% unleveraged cash equity", "Calculated Outcome": fut_res.structural_comparison["A_etf_return"]},
        {"Structure": "B. Spot Index Return", "Characteristics": "Unmanaged price index without dividends", "Calculated Outcome": fut_res.structural_comparison["B_spot_index_return"]},
        {"Structure": "C. Actual Leveraged Futures Contract P&L", "Characteristics": f"Marked-to-market with {fut_spec.multiplier}x multiplier and daily variation margin", "Calculated Outcome": fut_res.structural_comparison["C_actual_futures_pnl"]},
        {"Structure": "D. Continuous Futures Proxy", "Characteristics": "Stitched front-month prices subject to roll yield drift", "Calculated Outcome": fut_res.structural_comparison["D_continuous_proxy_status"]}
    ])
    st.table(comp_df)


# ==============================================================================
# WORKSPACE 11: PORTFOLIO RESEARCH (PHASE 5)
# ==============================================================================
elif active_workspace == "11. Portfolio Research":
    st.markdown('<div class="section-title">Portfolio Research, Rolling Beta & Markowitz Optimization</div>', unsafe_allow_html=True)
    st.markdown("""
    Construct and analyze multi-asset investment portfolios across Indian and Global assets:
    - **Markowitz Modern Portfolio Theory:** Maximum Sharpe Ratio, Minimum Variance, and Efficient Frontier.
    - **Rolling CAPM Beta & R-Squared** against the market benchmark.
    - **Correlation Matrix Heatmap** and **Percentage Risk Contributions (%RC)**.
    """)

    available_universe = ["^NSEI", "^NSEBANK", "^CNXIT", "SPY", "QQQ", "GC=F", "CL=F", "NIFTYBEES.NS", "GOLDBEES.NS"]
    chosen_assets = st.multiselect(
        "Select Multi-Asset Portfolio Universe (Min 2)",
        available_universe,
        default=["^NSEI", "^NSEBANK", "^CNXIT"] if "^NSE" in active_symbol else ["SPY", "QQQ", "GC=F"]
    )

    if len(chosen_assets) >= 2:
        with st.spinner("Retrieving multi-asset data and computing portfolio optimization..."):
            aligned_returns_list = []
            for sym in chosen_assets:
                try:
                    payload = ingestion_engine.fetch_yfinance(sym, start_date=date.today() - timedelta(days=365*2), end_date=date.today())
                    r = compute_daily_returns(payload.df["Close"]).rename(sym)
                    aligned_returns_list.append(r)
                except Exception:
                    pass

            if len(aligned_returns_list) >= 2:
                port_returns_df = pd.concat(aligned_returns_list, axis=1, join="inner").dropna()
                opt_res = optimize_portfolio(port_returns_df, risk_free_rate=float(rf_rate_input), trading_days=int(trading_days_input))

                op1, op2, op3 = st.columns(3)
                op1.metric("Max Sharpe Expected Return", f"{opt_res.max_sharpe_return * 100:.2f}%", delta=f"{opt_res.max_sharpe_ratio:.2f} Sharpe")
                op2.metric("Min Variance Volatility", f"{opt_res.min_var_volatility * 100:.2f}%", delta=f"{opt_res.min_var_return * 100:.2f}% Return")
                op3.metric("Equal Weight Sharpe", f"{opt_res.equal_weight_sharpe_ratio:.2f}", delta=f"{opt_res.equal_weight_volatility * 100:.2f}% Vol")

                c_opt1, c_opt2 = st.columns(2)
                with c_opt1:
                    st.plotly_chart(plot_efficient_frontier(opt_res, theme=theme_key), use_container_width=True, key="eff_frontier_chart")
                with c_opt2:
                    st.plotly_chart(plot_portfolio_allocation(opt_res.weights_max_sharpe, "Optimal Max Sharpe Weights", theme=theme_key), use_container_width=True, key="alloc_donut_chart")

                st.plotly_chart(plot_correlation_heatmap(opt_res.correlation_matrix, theme=theme_key), use_container_width=True, key="corr_matrix_heatmap")

                # Rolling Beta vs first asset as benchmark
                bench_sym = chosen_assets[0]
                asset_sym = chosen_assets[1]
                rolling_b_df = compute_rolling_beta(port_returns_df[asset_sym], port_returns_df[bench_sym], window=63)
                st.plotly_chart(plot_rolling_beta(rolling_b_df, asset_sym, bench_sym, theme=theme_key), use_container_width=True, key="rolling_beta_chart")
            else:
                st.warning("Insufficient multi-asset observations to perform portfolio optimization.")
    else:
        st.info("Please select at least 2 distinct assets to run portfolio optimization and correlation analysis.")


# ==============================================================================
# WORKSPACE 12: RESEARCH REPORTS & EXPORTS
# ==============================================================================
elif active_workspace == "12. Research Reports":
    st.markdown('<div class="section-title">Institutional Research Dossier & Reproducibility Export</div>', unsafe_allow_html=True)
    st.markdown("""
    Generate professional quantitative research dossiers compliant with institutional reporting standards.
    Includes:
    - **Vector-Rendered PDF Research Dossier** with executive scorecards, walk-forward tables, and model risk disclosures.
    - **Cryptographic JSON Experiment Manifest** with dataset SHA-256 fingerprint and exact hyperparameter state.
    """)

    # Run quick baseline backtest to compile report
    engine = BacktestEngine(initial_capital=float(initial_capital_input), trading_days=int(trading_days_input))
    strat = MovingAverageCrossover(20, 50)
    res_report = engine.run(df_market, strat, symbol=active_symbol)
    dist_stats = compute_distribution_statistics(daily_returns, trading_days=int(trading_days_input))

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.markdown("#### 1. Formal PDF Research Dossier")
        pdf_bytes = generate_pdf_research_report(
            backtest_result=res_report,
            dist_stats=dist_stats,
            dataset_fingerprint=market_payload.dataset_fingerprint,
            data_source="Yahoo Finance / Cleaned Market Feed",
            instrument_type=market_payload.instrument_type,
            corporate_action_notes=market_payload.corporate_action_notes,
            futures_rollover_notes=market_payload.futures_rollover_notes
        )
        st.download_button(
            label="📄 Download Institutional PDF Research Dossier",
            data=pdf_bytes,
            file_name=f"QuantEdge_{active_symbol}_Research_Dossier.pdf",
            mime="application/pdf"
        )
    with col_rep2:
        st.markdown("#### 2. Cryptographic Experiment Manifest (`manifest.json`)")
        manifest_dict = generate_experiment_manifest(
            experiment_id=f"EXP-{active_symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            dataset_fingerprint=market_payload.dataset_fingerprint,
            ticker=active_symbol,
            strategy_name=strat.name,
            strategy_params={"fast_window": 20, "slow_window": 50},
            frictions={"commission_bps": commission_bps_input, "slippage_bps": slippage_bps_input, "timing": exec_mode_input},
            in_sample_metrics={"Sharpe": res_report.metrics.sharpe_ratio, "CAGR": res_report.metrics.cagr, "MaxDD": res_report.metrics.max_drawdown}
        )
        st.download_button(
            label="💾 Download JSON Reproducibility Manifest",
            data=json.dumps(manifest_dict, indent=2),
            file_name=f"QuantEdge_{active_symbol}_manifest.json",
            mime="application/json"
        )


# ==============================================================================
# WORKSPACE 13: METHODOLOGY & DISCLOSURES
# ==============================================================================
elif active_workspace == "13. Methodology":
    st.markdown('<div class="section-title">Quantitative Methodology, Mathematical Formulations & Disclosures</div>', unsafe_allow_html=True)
    st.markdown(r"""
    ### 1. Mathematical Formulations for Audited Metrics
    - **Arithmetic Return:** $R_t = \\frac{P_t - P_{t-1}}{P_{t-1}}$
    - **Log Return:** $r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)$
    - **Annualized Realized Volatility:** $\\sigma_{\\text{ann}} = \\sqrt{252} \\times \\operatorname{std}(R)$
    - **Parkinson Volatility (High-Low):** $\\sigma_P = \\sqrt{\\frac{1}{4 \\ln(2) N} \\sum_{i=1}^N \\left(\\ln\\frac{H_i}{L_i}\\right)^2} \\times \\sqrt{252}$
    - **Garman-Klass Volatility (OHLC):** $\\sigma_{GK} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N \\left[ 0.5\\left(\\ln\\frac{H_i}{L_i}\\right)^2 - (2\\ln 2 - 1)\\left(\\ln\\frac{C_i}{O_i}\\right)^2 \\right]} \\times \\sqrt{252}$
    - **Wilder's RSI:** $\\text{RSI} = 100 - \\frac{100}{1 + \\text{RS}}, \\quad \\alpha = \\frac{1}{N}$
    - **Annualized Sharpe Ratio:** $\\text{Sharpe} = \\frac{(\\bar{R} - R_f) \\times 252}{\\sigma_{\\text{ann}}}$
    - **Sortino Ratio (Downside Only):** $\\text{Sortino} = \\frac{(\\bar{R} - R_f) \\times 252}{\\sqrt{252} \\times \\sqrt{\\frac{1}{N-1} \\sum \\min(R_t - R_f, 0)^2}}$
    - **Maximum Drawdown (MDD):** $\\text{MDD} = \\max_t \\left( \\frac{\\max_{s \\le t} V_s - V_t}{\\max_{s \\le t} V_s} \\right)$

    ### 2. Futures Margin Accounting Mechanics
    - **Initial Margin:** Posted upon position entry ($P_t \\times \\text{Lot Size} \\times N \\times \\text{Margin Rate}$).
    - **Daily Variation Margin:** Credited/debited to cash daily:
      $$\\Delta \\text{VM}_t = (P_t - P_{t-1}) \\times \\text{Lot Size} \\times N \\times \\text{Direction}$$
    - **Margin Call Condition:** Triggered whenever $\\text{Equity}_t < \\text{Maintenance Margin}_t$.

    ### 3. Model Risk & Research Disclosures
    1. **Execution Reality:** Historical backtesting assumes execution at next bar Open with static friction penalties. Live execution encounters spread variation, adverse market impact, and latency.
    2. **Resampling Caveat:** Monte Carlo simulations resample observed empirical trade returns under the i.i.d. assumption; they do not predict novel future macroeconomic black-swan regimes.
    3. **Continuous Futures Rollover:** Front-month stitched continuous futures contain artificial step-discontinuities and omit negative or positive roll yields (contango/backwardation drag).
    """)
