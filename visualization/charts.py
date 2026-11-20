"""Interactive Plotly charting engine tailored for quantitative market analysis."""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting.execution import Trade
from analytics.statistics import DistributionStats


# Professional quantitative styling templates
QUANT_THEME_DARK = {
    "bg_color": "#0B111E",
    "paper_color": "#0B111E",
    "grid_color": "#1E2E4A",
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
    "font_color": "#F8FAFC",
    "bull_color": "#10B981",
    "bear_color": "#EF4444",
    "accent_cyan": "#06B6D4",
    "accent_blue": "#3B82F6",
    "accent_amber": "#F59E0B",
    "accent_purple": "#8B5CF6",
    "benchmark_color": "#94A3B8"
}

QUANT_THEME_LIGHT = {
    "bg_color": "#F8FAFC",
    "paper_color": "#FFFFFF",
    "grid_color": "#E2E8F0",
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
    "font_color": "#0F172A",
    "bull_color": "#059669",
    "bear_color": "#DC2626",
    "accent_cyan": "#0891B2",
    "accent_blue": "#2563EB",
    "accent_amber": "#D97706",
    "accent_purple": "#7C3AED",
    "benchmark_color": "#64748B"
}

QUANT_THEME = QUANT_THEME_DARK


def apply_quant_layout(fig: go.Figure, title: str, height: int = 500, theme: str = "dark") -> go.Figure:
    """Applies clean institutional financial terminal theme to Plotly figures."""
    th = QUANT_THEME_LIGHT if theme == "light" else QUANT_THEME_DARK
    template = "plotly_white" if theme == "light" else "plotly_dark"
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 15, "color": th["font_color"], "family": th["font_family"]},
            "x": 0.02,
            "y": 0.96
        },
        template=template,
        paper_bgcolor=th["paper_color"],
        plot_bgcolor=th["bg_color"],
        font={"family": th["font_family"], "color": th["font_color"], "size": 12},
        height=height,
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11}
        }
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=th["grid_color"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=th["grid_color"], zeroline=False)
    return fig


def plot_market_explorer_candlestick(
    df: pd.DataFrame,
    ticker: str,
    show_volume: bool = True
) -> go.Figure:
    """Creates an interactive OHLC candlestick and volume chart."""
    if show_volume and "Volume" in df.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # Candlestick
    candlestick = go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color=QUANT_THEME["bull_color"],
        decreasing_line_color=QUANT_THEME["bear_color"],
        increasing_fillcolor=QUANT_THEME["bull_color"],
        decreasing_fillcolor=QUANT_THEME["bear_color"],
    )
    fig.add_trace(candlestick, row=1, col=1)

    # 20 & 50 MA overlays if enough bars exist
    if len(df) >= 20:
        ma20 = df["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma20, mode="lines", name="SMA(20)", line={"color": QUANT_THEME["accent_cyan"], "width": 1.2}), row=1, col=1)
    if len(df) >= 50:
        ma50 = df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma50, mode="lines", name="SMA(50)", line={"color": QUANT_THEME["accent_amber"], "width": 1.2}), row=1, col=1)

    # Volume bars
    if show_volume and "Volume" in df.columns:
        vol_colors = np.where(df["Close"] >= df["Open"], QUANT_THEME["bull_color"], QUANT_THEME["bear_color"])
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    apply_quant_layout(fig, f"{ticker} Market Explorer — Historical Price & Volume", height=580)
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    ticker: str,
    stats: Optional[DistributionStats] = None
) -> go.Figure:
    """Plots empirical return distribution with normal curve fit, VaR thresholds, and moments."""
    clean = returns.dropna()
    arr = clean.to_numpy()

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=arr * 100.0,
        histnorm="probability density",
        name="Empirical Density",
        marker_color="rgba(0, 229, 255, 0.45)",
        marker_line={"color": QUANT_THEME["accent_cyan"], "width": 1},
        nbinsx=60
    ))

    # Gaussian normal distribution fit curve
    mu = np.mean(arr) * 100.0
    sigma = np.std(arr, ddof=1) * 100.0
    x_range = np.linspace(np.min(arr) * 100.0, np.max(arr) * 100.0, 200)
    from scipy.stats import norm
    norm_pdf = norm.pdf(x_range, mu, sigma)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=norm_pdf,
        mode="lines",
        name=f"Gaussian Fit (N({mu:.2f}%, {sigma:.2f}%))",
        line={"color": QUANT_THEME["accent_amber"], "width": 2, "dash": "dash"}
    ))

    # VaR 95% and VaR 99% vertical threshold markers
    if stats is not None:
        var95_pct = -stats.var_95 * 100.0
        var99_pct = -stats.var_99 * 100.0

        fig.add_vline(x=var95_pct, line_dash="dot", line_color="#FF9100",
                      annotation_text=f"95% VaR: {var95_pct:.2f}%", annotation_position="top left")
        fig.add_vline(x=var99_pct, line_dash="dot", line_color=QUANT_THEME["bear_color"],
                      annotation_text=f"99% VaR: {var99_pct:.2f}%", annotation_position="top left")

    apply_quant_layout(fig, f"{ticker} Return Distribution & Risk Fat-Tail Analysis", height=450)
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Probability Density")
    return fig


def plot_rolling_volatility_comparison(
    returns: pd.Series,
    df: pd.DataFrame,
    ticker: str,
    window: int = 21
) -> go.Figure:
    """Compares Realized, Parkinson, and Garman-Klass rolling annualized volatility estimators."""
    from analytics.volatility import (
        compute_rolling_volatility,
        compute_parkinson_volatility,
        compute_garman_klass_volatility,
        compute_ewma_volatility
    )

    r_vol = compute_rolling_volatility(returns, window=window) * 100.0
    p_vol = compute_parkinson_volatility(df, window=window) * 100.0
    gk_vol = compute_garman_klass_volatility(df, window=window) * 100.0
    ewma_vol = compute_ewma_volatility(returns) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_vol.index, y=r_vol, mode="lines", name=f"Realized (Close-to-Close {window}d)", line={"color": QUANT_THEME["accent_cyan"], "width": 1.5}))
    fig.add_trace(go.Scatter(x=p_vol.index, y=p_vol, mode="lines", name=f"Parkinson (High-Low {window}d)", line={"color": QUANT_THEME["accent_amber"], "width": 1.5}))
    fig.add_trace(go.Scatter(x=gk_vol.index, y=gk_vol, mode="lines", name=f"Garman-Klass (OHLC {window}d)", line={"color": QUANT_THEME["bull_color"], "width": 1.5}))
    fig.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, mode="lines", name="EWMA (RiskMetrics λ=0.94)", line={"color": QUANT_THEME["accent_purple"], "width": 1.2, "dash": "dot"}))

    apply_quant_layout(fig, f"{ticker} Rolling Volatility Dynamics ({window}-Day Annualized)", height=450)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Annualized Volatility (%)")
    return fig


def plot_technical_indicators_suite(
    df: pd.DataFrame,
    ticker: str,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_period: int = 14
) -> go.Figure:
    """Plots RSI, MACD, and ATR in coordinated subplots."""
    from analytics.indicators import compute_rsi, compute_macd, compute_atr

    rsi = compute_rsi(df["Close"], period=rsi_period)
    macd_df = compute_macd(df["Close"], fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
    atr = compute_atr(df, period=atr_period)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.35, 0.40, 0.25],
        subplot_titles=[
            f"RSI ({rsi_period}) with 30/70 Overbought/Oversold Bands",
            f"MACD ({macd_fast}, {macd_slow}, {macd_signal})",
            f"ATR ({atr_period}) Average True Range"
        ]
    )

    # 1. RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI", line={"color": QUANT_THEME["accent_purple"], "width": 1.5}), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=QUANT_THEME["bear_color"], row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=QUANT_THEME["bull_color"], row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#757575", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=1)

    # 2. MACD
    hist_colors = np.where(macd_df["histogram"] >= 0, QUANT_THEME["bull_color"], QUANT_THEME["bear_color"])
    fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["histogram"], name="MACD Hist", marker_color=hist_colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["macd_line"], mode="lines", name="MACD Line", line={"color": QUANT_THEME["accent_cyan"], "width": 1.5}), row=2, col=1)
    fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["signal_line"], mode="lines", name="Signal Line", line={"color": QUANT_THEME["accent_amber"], "width": 1.5}), row=2, col=1)

    # 3. ATR
    fig.add_trace(go.Scatter(x=atr.index, y=atr, mode="lines", name="ATR", line={"color": "#FF9100", "width": 1.5}), row=3, col=1)

    apply_quant_layout(fig, f"{ticker} Technical Indicator Suite", height=650)
    return fig


def plot_equity_curves(
    results_map: Dict[str, Any],
    benchmark_equity: pd.Series,
    title: str = "Backtest Equity Curve vs Benchmark"
) -> go.Figure:
    """Plots comparative cumulative portfolio equity curves for multiple strategies and benchmark."""
    fig = go.Figure()

    # Benchmark trace
    fig.add_trace(go.Scatter(
        x=benchmark_equity.index,
        y=benchmark_equity,
        mode="lines",
        name="Buy & Hold Benchmark",
        line={"color": QUANT_THEME["benchmark_color"], "width": 1.8, "dash": "dash"}
    ))

    colors = [QUANT_THEME["accent_cyan"], QUANT_THEME["bull_color"], QUANT_THEME["accent_amber"], QUANT_THEME["accent_purple"]]

    color_idx = 0
    for strat_name, res in results_map.items():
        eq = res.equity_curve if hasattr(res, "equity_curve") else res
        line_color = colors[color_idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=eq.index,
            y=eq,
            mode="lines",
            name=strat_name,
            line={"color": line_color, "width": 2.0}
        ))
        color_idx += 1

    apply_quant_layout(fig, title, height=480)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Portfolio Value ($)", tickformat="$,.0f")
    return fig


def plot_drawdown_underwater(
    drawdown_series: pd.Series,
    benchmark_drawdown: Optional[pd.Series] = None,
    title: str = "Underwater Drawdown Profile"
) -> go.Figure:
    """Creates filled underwater drawdown chart showing depth and duration of declines."""
    fig = go.Figure()

    # Benchmark DD
    if benchmark_drawdown is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_drawdown.index,
            y=benchmark_drawdown * 100.0,
            mode="lines",
            name="Benchmark Drawdown",
            line={"color": QUANT_THEME["benchmark_color"], "width": 1.2, "dash": "dot"}
        ))

    # Strategy DD
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series * 100.0,
        mode="lines",
        name="Strategy Drawdown",
        line={"color": QUANT_THEME["bear_color"], "width": 1.5},
        fill="tozeroy",
        fillcolor="rgba(255, 61, 0, 0.25)"
    ))

    apply_quant_layout(fig, title, height=400)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Drawdown (%)", tickformat=".1f")
    return fig


def plot_trade_markers_on_price(
    df: pd.DataFrame,
    trades: List[Trade],
    ticker: str
) -> go.Figure:
    """Visualizes trade entry and exit executions overlaid on price line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name=f"{ticker} Close",
        line={"color": "#90CAF9", "width": 1.5}
    ))

    if trades:
        # Long entries
        long_entries = [t for t in trades if t.direction == "LONG"]
        if long_entries:
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(t.entry_date) for t in long_entries],
                y=[t.entry_price for t in long_entries],
                mode="markers",
                name="Long Entry",
                marker={"symbol": "triangle-up", "size": 11, "color": QUANT_THEME["bull_color"], "line": {"color": "#FFFFFF", "width": 1}},
                hovertext=[f"Trade #{t.trade_id} LONG Entry: ${t.entry_price:.2f}" for t in long_entries]
            ))

        # Short entries
        short_entries = [t for t in trades if t.direction == "SHORT"]
        if short_entries:
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(t.entry_date) for t in short_entries],
                y=[t.entry_price for t in short_entries],
                mode="markers",
                name="Short Entry",
                marker={"symbol": "triangle-down", "size": 11, "color": QUANT_THEME["bear_color"], "line": {"color": "#FFFFFF", "width": 1}},
                hovertext=[f"Trade #{t.trade_id} SHORT Entry: ${t.entry_price:.2f}" for t in short_entries]
            ))

        # Exits (Winning vs Losing exits)
        win_exits = [t for t in trades if t.net_pnl >= 0]
        loss_exits = [t for t in trades if t.net_pnl < 0]

        if win_exits:
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(t.exit_date) for t in win_exits],
                y=[t.exit_price for t in win_exits],
                mode="markers",
                name="Profitable Exit",
                marker={"symbol": "circle", "size": 8, "color": QUANT_THEME["bull_color"], "line": {"color": "#FFFFFF", "width": 1}},
                hovertext=[f"Trade #{t.trade_id} Exit (+${t.net_pnl:.2f}, {t.return_pct*100:.1f}%)" for t in win_exits]
            ))

        if loss_exits:
            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(t.exit_date) for t in loss_exits],
                y=[t.exit_price for t in loss_exits],
                mode="markers",
                name="Loss Exit",
                marker={"symbol": "x", "size": 8, "color": QUANT_THEME["bear_color"], "line": {"color": "#FFFFFF", "width": 1.5}},
                hovertext=[f"Trade #{t.trade_id} Exit (-${abs(t.net_pnl):.2f}, {t.return_pct*100:.1f}%)" for t in loss_exits]
            ))

    apply_quant_layout(fig, f"{ticker} Executed Trades & Order Fill Locations", height=500)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)")
    return fig


def plot_autocorrelation_bar(autocorr_dict: Dict[int, float]) -> go.Figure:
    """Plots bar chart of return serial correlation across lags."""
    lags = list(autocorr_dict.keys())
    vals = list(autocorr_dict.values())

    fig = go.Figure()
    bar_colors = np.where(np.array(vals) >= 0, QUANT_THEME["accent_cyan"], QUANT_THEME["accent_amber"])

    fig.add_trace(go.Bar(
        x=[f"Lag {lag}" for lag in lags],
        y=vals,
        name="ACF",
        marker_color=bar_colors
    ))

    # 95% Bartlett confidence bands (+/- 1.96 / sqrt(N)) approx +/- 0.08
    fig.add_hline(y=0.08, line_dash="dash", line_color="#E0E0E0", annotation_text="+95% CI")
    fig.add_hline(y=-0.08, line_dash="dash", line_color="#E0E0E0", annotation_text="-95% CI")
    fig.add_hline(y=0.0, line_color="#757575")

    apply_quant_layout(fig, "Return Autocorrelation Function (ACF)", height=350)
    fig.update_yaxes(title_text="Autocorrelation Coefficient", range=[-0.25, 0.25])
    return fig



def plot_monte_carlo_fan(mc_report: Any, title: str = "Monte Carlo Simulated Equity Trajectories") -> go.Figure:
    """Plots Monte Carlo percentile fan chart (5th to 95th percentile)."""
    fig = go.Figure()
    trajs = mc_report.percentile_equity_trajectories
    steps = len(trajs["50th"])
    x_axis = np.arange(steps)

    # Shaded 5th - 95th band
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=trajs["95th"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=trajs["5th"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(0, 229, 255, 0.12)",
        line={"width": 0},
        name="90% Confidence Interval (5th-95th %ile)"
    ))

    # Shaded 25th - 75th band (Interquartile Range)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=trajs["75th"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=trajs["25th"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(0, 229, 255, 0.25)",
        line={"width": 0},
        name="Interquartile Range (25th-75th %ile)"
    ))

    # Median (50th percentile)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=trajs["50th"],
        mode="lines",
        name="Median Trajectory (50th %ile)",
        line={"color": QUANT_THEME["accent_cyan"], "width": 2.5}
    ))

    # Baseline initial capital
    fig.add_hline(y=mc_report.initial_capital, line_dash="dash", line_color="#888888", annotation_text="Initial Capital")

    apply_quant_layout(fig, title, height=450)
    fig.update_xaxes(title_text="Trade Sequence Step")
    fig.update_yaxes(title_text="Portfolio Equity ($)", tickformat="$,.0f")
    return fig


def plot_monte_carlo_drawdown_distribution(mc_report: Any) -> go.Figure:
    """Plots histogram of Monte Carlo maximum drawdowns with ruin thresholds."""
    fig = go.Figure()
    mdds_pct = mc_report.max_drawdown_distribution * 100.0

    fig.add_trace(go.Histogram(
        x=mdds_pct,
        name="Simulated Max DD",
        marker_color="rgba(255, 61, 0, 0.55)",
        marker_line={"color": QUANT_THEME["bear_color"], "width": 1},
        nbinsx=40
    ))

    fig.add_vline(x=20.0, line_dash="dash", line_color="#FFD600",
                  annotation_text=f"20% DD Risk: {mc_report.probability_of_ruin_20pct*100:.1f}%")
    fig.add_vline(x=30.0, line_dash="dash", line_color="#FF9100",
                  annotation_text=f"30% DD Risk: {mc_report.probability_of_ruin_30pct*100:.1f}%")
    fig.add_vline(x=mc_report.p95_max_drawdown * 100.0, line_dash="dot", line_color=QUANT_THEME["bear_color"],
                  annotation_text=f"95th %ile Worst DD: {mc_report.p95_max_drawdown*100.0:.1f}%")

    apply_quant_layout(fig, f"Monte Carlo Maximum Drawdown Distribution ({mc_report.num_simulations} Runs)", height=400)
    fig.update_xaxes(title_text="Simulated Maximum Drawdown (%)")
    fig.update_yaxes(title_text="Frequency")
    return fig


def plot_market_regimes(df: pd.DataFrame, regimes: Any, ticker: str) -> go.Figure:
    """Visualizes market price history tagged with statistical volatility and trend regimes."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{ticker} Price & Trend Regimes", "Volatility Regimes (High / Normal / Low)"]
    )

    # Row 1: Price colored or annotated
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{ticker} Close", line={"color": "#FFFFFF", "width": 1.5}), row=1, col=1)

    # 50 SMA overlay
    sma50 = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma50, mode="lines", name="50-day Regime SMA", line={"color": QUANT_THEME["accent_amber"], "width": 1.2, "dash": "dash"}), row=1, col=1)

    # Row 2: Volatility Regime bands
    vol_map = {"LOW_VOL": 0, "NORMAL_VOL": 1, "HIGH_VOL": 2}
    vol_num = [vol_map.get(s, 1) for s in regimes.volatility_regime]

    fig.add_trace(go.Scatter(
        x=df.index,
        y=vol_num,
        mode="lines",
        name="Volatility State",
        line={"color": QUANT_THEME["accent_cyan"], "width": 1.5}
    ), row=2, col=1)

    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["Low Vol", "Normal Vol", "High Vol"], row=2, col=1)

    apply_quant_layout(fig, f"{ticker} Statistical Market Regime Deconstruction", height=500)
    return fig


def plot_parameter_sensitivity_heatmap(
    scan_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str = "Sharpe_Ratio",
    title: str = "Parameter Sensitivity Heatmap"
) -> go.Figure:
    """Creates a 2D parameter sensitivity heatmap."""
    pivot = scan_df.pivot(index=y_col, columns=x_col, values=z_col)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale="Viridis",
        colorbar={"title": z_col.replace("_", " ")}
    ))

    apply_quant_layout(fig, title, height=450)
    fig.update_xaxes(title_text=x_col.replace("_", " "))
    fig.update_yaxes(title_text=y_col.replace("_", " "))
    return fig


def plot_friction_sensitivity(friction_df: pd.DataFrame, title: str = "Transaction Cost & Slippage Resilience") -> go.Figure:
    """Plots strategy Sharpe degradation across total friction drag (bps)."""
    fig = go.Figure()

    # Group by total friction and take average Sharpe
    grouped = friction_df.groupby("Total_Friction_bps")["Sharpe_Ratio"].mean().reset_index()

    fig.add_trace(go.Scatter(
        x=grouped["Total_Friction_bps"],
        y=grouped["Sharpe_Ratio"],
        mode="lines+markers",
        name="Sharpe vs Friction",
        line={"color": QUANT_THEME["accent_cyan"], "width": 2.5},
        marker={"size": 8, "color": QUANT_THEME["accent_amber"]}
    ))

    fig.add_hline(y=0.0, line_dash="dash", line_color=QUANT_THEME["bear_color"], annotation_text="Breakeven Zero-Alpha Threshold")

    apply_quant_layout(fig, title, height=380)
    fig.update_xaxes(title_text="Total Execution Drag (Commission + Slippage in bps)")
    fig.update_yaxes(title_text="Annualized Sharpe Ratio")
    return fig


def plot_futures_margin_and_equity(
    sim_result: Any,
    title: str = "Futures Marked-to-Market Equity & Margin Dynamics",
    theme: str = "dark"
) -> go.Figure:
    """Plots futures account equity, daily cash, variation margin, and initial margin requirements."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=["Marked-to-Market Equity vs Required Margin", "Daily Variation Margin (Settlement P&L)"]
    )

    dates = sim_result.daily_equity_curve.index

    # Equity Curve
    fig.add_trace(go.Scatter(
        x=dates,
        y=sim_result.daily_equity_curve,
        mode="lines",
        name=f"Account Equity ({sim_result.spec.currency})",
        line={"color": QUANT_THEME["accent_cyan"], "width": 2.0}
    ), row=1, col=1)

    # Initial Margin Requirement line
    fig.add_trace(go.Scatter(
        x=dates,
        y=sim_result.daily_margin_required,
        mode="lines",
        name="Initial Margin Required",
        line={"color": QUANT_THEME["accent_amber"], "width": 1.2, "dash": "dot"}
    ), row=1, col=1)

    # Daily Variation Margin bars
    vm = sim_result.daily_variation_margin
    colors = np.where(vm >= 0, QUANT_THEME["bull_color"], QUANT_THEME["bear_color"])
    fig.add_trace(go.Bar(
        x=dates,
        y=vm,
        name="Variation Margin",
        marker_color=colors
    ), row=2, col=1)

    apply_quant_layout(fig, title, height=520, theme=theme)
    fig.update_yaxes(title_text=f"Capital ({sim_result.spec.currency})", row=1, col=1)
    fig.update_yaxes(title_text="Daily VM", row=2, col=1)
    return fig


def plot_efficient_frontier(
    opt_result: Any,
    title: str = "Markowitz Modern Portfolio Theory: Efficient Frontier",
    theme: str = "dark"
) -> go.Figure:
    """Plots the Markowitz efficient frontier with highlighted optimal portfolios."""
    fig = go.Figure()

    # Frontier line
    if opt_result.efficient_frontier_vols:
        fig.add_trace(go.Scatter(
            x=[v * 100.0 for v in opt_result.efficient_frontier_vols],
            y=[r * 100.0 for r in opt_result.efficient_frontier_rets],
            mode="lines",
            name="Efficient Frontier",
            line={"color": QUANT_THEME["accent_cyan"], "width": 2.5}
        ))

    # Maximum Sharpe Ratio Portfolio
    fig.add_trace(go.Scatter(
        x=[opt_result.max_sharpe_volatility * 100.0],
        y=[opt_result.max_sharpe_return * 100.0],
        mode="markers",
        name=f"Max Sharpe ({opt_result.max_sharpe_ratio:.2f})",
        marker={"size": 14, "color": QUANT_THEME["bull_color"], "symbol": "star"}
    ))

    # Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[opt_result.min_var_volatility * 100.0],
        y=[opt_result.min_var_return * 100.0],
        mode="markers",
        name=f"Min Variance ({opt_result.min_var_volatility*100.0:.1f}% Vol)",
        marker={"size": 12, "color": QUANT_THEME["accent_amber"], "symbol": "diamond"}
    ))

    # Equal Weight Portfolio
    fig.add_trace(go.Scatter(
        x=[opt_result.equal_weight_volatility * 100.0],
        y=[opt_result.equal_weight_return * 100.0],
        mode="markers",
        name=f"Equal Weight ({opt_result.equal_weight_sharpe_ratio:.2f} Sharpe)",
        marker={"size": 11, "color": QUANT_THEME["accent_purple"], "symbol": "circle"}
    ))

    apply_quant_layout(fig, title, height=450, theme=theme)
    fig.update_xaxes(title_text="Annualized Volatility (%)")
    fig.update_yaxes(title_text="Annualized Expected Return (%)")
    return fig


def plot_portfolio_allocation(
    weights_dict: Dict[str, float],
    title: str = "Optimal Portfolio Asset Weights",
    theme: str = "dark"
) -> go.Figure:
    """Creates a clean donut chart of portfolio allocation weights."""
    filtered = {k: v for k, v in weights_dict.items() if v > 0.005}
    labels = list(filtered.keys())
    values = [v * 100.0 for v in filtered.values()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
        insidetextorientation="radial"
    )])

    apply_quant_layout(fig, title, height=400, theme=theme)
    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Multi-Asset Return Correlation Matrix",
    theme: str = "dark"
) -> go.Figure:
    """Generates an annotated multi-asset correlation matrix heatmap."""
    labels = list(corr_df.columns)
    z_vals = corr_df.values

    # Format annotations
    annotations = []
    for i, row in enumerate(z_vals):
        for j, val in enumerate(row):
            annotations.append({
                "x": labels[j],
                "y": labels[i],
                "text": f"{val:.2f}",
                "showarrow": False,
                "font": {"color": "#FFFFFF" if abs(val) > 0.4 else "#94A3B8", "size": 11}
            })

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        colorbar={"title": "Correlation"}
    ))

    fig.update_layout(annotations=annotations)
    apply_quant_layout(fig, title, height=450, theme=theme)
    return fig


def plot_rolling_beta(
    rolling_df: pd.DataFrame,
    ticker: str,
    benchmark_ticker: str,
    theme: str = "dark"
) -> go.Figure:
    """Plots rolling CAPM Beta and rolling R-squared."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"Rolling CAPM Beta ({ticker} vs {benchmark_ticker})", "Rolling R-Squared (Variance Explained)"]
    )

    dates = rolling_df.index

    # Beta
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_df["rolling_beta"],
        mode="lines",
        name="Rolling Beta",
        line={"color": QUANT_THEME["accent_cyan"], "width": 2.0}
    ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color=QUANT_THEME["benchmark_color"],
                  annotation_text="Market Beta (1.0)", row=1, col=1)

    # R2
    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_df["rolling_r2"] * 100.0,
        mode="lines",
        name="Rolling R² (%)",
        line={"color": QUANT_THEME["accent_amber"], "width": 1.5}
    ), row=2, col=1)

    apply_quant_layout(fig, f"Rolling Systematic Risk Profile: {ticker} vs {benchmark_ticker}", height=480, theme=theme)
    fig.update_yaxes(title_text="Beta", row=1, col=1)
    fig.update_yaxes(title_text="R² (%)", row=2, col=1)
    return fig
