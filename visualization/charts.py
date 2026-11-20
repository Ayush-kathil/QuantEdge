"""Interactive Plotly charting engine tailored for quantitative market analysis."""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting.execution import Trade
from analytics.statistics import DistributionStats


# Professional quantitative styling template
QUANT_THEME = {
    "bg_color": "#0E1117",
    "paper_color": "#0E1117",
    "grid_color": "#262730",
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
    "font_color": "#FAFAFA",
    "bull_color": "#00C853",
    "bear_color": "#FF3D00",
    "accent_cyan": "#00E5FF",
    "accent_amber": "#FFD600",
    "accent_purple": "#7C4DFF",
    "benchmark_color": "#9E9E9E"
}


def apply_quant_layout(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    """Applies clean financial terminal theme to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": QUANT_THEME["font_color"], "family": QUANT_THEME["font_family"]},
            "x": 0.02,
            "y": 0.96
        },
        template="plotly_dark",
        paper_bgcolor=QUANT_THEME["paper_color"],
        plot_bgcolor=QUANT_THEME["bg_color"],
        font={"family": QUANT_THEME["font_family"], "color": QUANT_THEME["font_color"], "size": 12},
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
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=QUANT_THEME["grid_color"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=QUANT_THEME["grid_color"], zeroline=False)
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
