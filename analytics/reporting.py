"""Research report generation module compiling institutional PDF dossiers and structured summaries."""
import io
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)

from config.settings import APP_VERSION, PLATFORM_NAME
from analytics.statistics import DistributionStats
from analytics.monte_carlo import MonteCarloReport


def generate_pdf_research_report(
    backtest_result: Any,
    dist_stats: DistributionStats,
    dataset_fingerprint: str,
    data_source: str,
    instrument_type: str,
    corporate_action_notes: str,
    futures_rollover_notes: str,
    monte_carlo_report: Optional[MonteCarloReport] = None,
    walk_forward_summary: Optional[pd.DataFrame] = None,
    futures_summary: Optional[Dict[str, Any]] = None,
    experiment_id: str = "QEDGE-REPORT"
) -> bytes:
    """
    Compiles an institutional-grade PDF research dossier.
    
    Returns:
        bytes: Binary PDF data suitable for direct Streamlit download or disk storage.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom palette
    primary_color = colors.HexColor("#0D1B2A")
    accent_blue = colors.HexColor("#1B4965")
    accent_gold = colors.HexColor("#C59B27")
    text_dark = colors.HexColor("#1A1A1A")
    table_bg_light = colors.HexColor("#F8F9FA")
    table_border = colors.HexColor("#DEE2E6")

    # Typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555"),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=accent_blue,
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#333333")
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph(f"<b>QuantEdge Institutional Research Dossier</b>", title_style))
    story.append(Paragraph(
        f"Platform: {PLATFORM_NAME} v{APP_VERSION} | Experiment ID: {experiment_id} | Asset: <b>{backtest_result.symbol}</b>",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_gold, spaceAfter=8))

    # SECTION 1: PROVENANCE & DATA INTEGRITY
    story.append(Paragraph("<b>1. Asset Profile & Data Provenance</b>", h1_style))
    prov_data = [
        [Paragraph("<b>Symbol:</b>", body_style), Paragraph(backtest_result.symbol, body_style),
         Paragraph("<b>Instrument Class:</b>", body_style), Paragraph(instrument_type, body_style)],
        [Paragraph("<b>Data Source:</b>", body_style), Paragraph(data_source, body_style),
         Paragraph("<b>Date Window:</b>", body_style), Paragraph(f"{backtest_result.start_date} to {backtest_result.end_date}", body_style)],
        [Paragraph("<b>Dataset SHA-256:</b>", body_style), Paragraph(f"<font size=7>{dataset_fingerprint}</font>", body_style),
         Paragraph("<b>Total Bars:</b>", body_style), Paragraph(str(len(backtest_result.equity_curve)), body_style)]
    ]
    prov_table = Table(prov_data, colWidths=[100, 170, 110, 160])
    prov_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), table_bg_light),
        ("GRID", (0, 0), (-1, -1), 0.5, table_border),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(prov_table)
    story.append(Spacer(1, 4))
    
    # Corporate actions & futures rollover disclosure
    story.append(Paragraph(f"<b>Corporate Actions Note:</b> {corporate_action_notes}", callout_style))
    if "Continuous" in instrument_type or "=" in backtest_result.symbol:
        story.append(Paragraph(f"<b>Futures Contract Disclosure:</b> {futures_rollover_notes}", callout_style))
    story.append(Spacer(1, 8))

    # SECTION 2: STATISTICAL DISTRIBUTION & MOMENTS
    story.append(Paragraph("<b>2. Statistical Distribution & Risk Moments</b>", h1_style))
    dist_data = [
        [Paragraph("<b>Mean (Ann.):</b>", body_style), Paragraph(f"{dist_stats.mean_annualized*100:+.2f}%", body_style),
         Paragraph("<b>Daily Std Dev:</b>", body_style), Paragraph(f"{dist_stats.std_daily*100:.2f}%", body_style),
         Paragraph("<b>Skewness:</b>", body_style), Paragraph(f"{dist_stats.skewness:+.2f}", body_style)],
        [Paragraph("<b>Excess Kurtosis:</b>", body_style), Paragraph(f"{dist_stats.excess_kurtosis:+.2f}", body_style),
         Paragraph("<b>Jarque-Bera p-val:</b>", body_style), Paragraph(f"{dist_stats.jarque_bera_pvalue:.4e}", body_style),
         Paragraph("<b>Gaussian Normal?</b>", body_style), Paragraph("Yes (p>0.05)" if dist_stats.is_normal_5pct else "No (Rejected)", body_style)],
        [Paragraph("<b>Historical 95% VaR:</b>", body_style), Paragraph(f"{dist_stats.var_95*100:.2f}%", body_style),
         Paragraph("<b>Historical 95% CVaR:</b>", body_style), Paragraph(f"{dist_stats.cvar_95*100:.2f}%", body_style),
         Paragraph("<b>Historical 99% VaR:</b>", body_style), Paragraph(f"{dist_stats.var_99*100:.2f}%", body_style)]
    ]
    dist_table = Table(dist_data, colWidths=[90, 85, 95, 85, 95, 90])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), table_bg_light),
        ("GRID", (0, 0), (-1, -1), 0.5, table_border),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 8))

    # SECTION 3: STRATEGY & BACKTEST PERFORMANCE SCORECARD
    story.append(Paragraph("<b>3. Strategy Performance & Benchmark Attribution</b>", h1_style))
    sm = backtest_result.metrics
    bm = backtest_result.benchmark_metrics

    perf_headers = ["Metric", "Strategy Result", "Benchmark (Buy & Hold)", "Delta / Spread"]
    perf_rows = [
        [Paragraph(f"<b>{perf_headers[0]}</b>", body_style),
         Paragraph(f"<b>{perf_headers[1]}</b>", body_style),
         Paragraph(f"<b>{perf_headers[2]}</b>", body_style),
         Paragraph(f"<b>{perf_headers[3]}</b>", body_style)],
        ["Initial Capital", f"${sm.initial_capital:,.2f}", f"${bm.initial_capital:,.2f}", "$0.00"],
        ["Ending Capital", f"${sm.ending_capital:,.2f}", f"${bm.ending_capital:,.2f}", f"${sm.ending_capital - bm.ending_capital:+,.2f}"],
        ["Total Return", f"{sm.total_return*100:+.2f}%", f"{bm.total_return*100:+.2f}%", f"{(sm.total_return - bm.total_return)*100:+.2f}%"],
        ["CAGR", f"{sm.cagr*100:.2f}%", f"{bm.cagr*100:.2f}%", f"{(sm.cagr - bm.cagr)*100:+.2f}%"],
        ["Annualized Volatility", f"{sm.annualized_volatility*100:.2f}%", f"{bm.annualized_volatility*100:.2f}%", f"{(sm.annualized_volatility - bm.annualized_volatility)*100:+.2f}%"],
        ["Sharpe Ratio (Rf=4%)", f"{sm.sharpe_ratio:.2f}", f"{bm.sharpe_ratio:.2f}", f"{sm.sharpe_ratio - bm.sharpe_ratio:+.2f}"],
        ["Sortino Ratio", f"{sm.sortino_ratio:.2f}", f"{bm.sortino_ratio:.2f}", f"{sm.sortino_ratio - bm.sortino_ratio:+.2f}"],
        ["Maximum Drawdown", f"-{sm.max_drawdown*100:.2f}%", f"-{bm.max_drawdown*100:.2f}%", f"-{(sm.max_drawdown - bm.max_drawdown)*100:.2f}%"],
        ["Calmar Ratio", f"{sm.calmar_ratio:.2f}", f"{bm.calmar_ratio:.2f}", f"{sm.calmar_ratio - bm.calmar_ratio:+.2f}"],
        ["Beta vs Benchmark", f"{sm.beta:.2f}", "1.00", f"{sm.beta - 1.00:+.2f}"],
        ["Alpha (Ann.)", f"{sm.alpha_annualized*100:+.2f}%", "0.00%", f"{sm.alpha_annualized*100:+.2f}%"],
        ["Information Ratio", f"{sm.information_ratio:.2f}", "N/A", f"{sm.information_ratio:.2f}"],
        ["Tracking Error", f"{sm.tracking_error*100:.2f}%", "0.00%", f"{sm.tracking_error*100:.2f}%"],
        ["Win Rate / Trades", f"{sm.win_rate*100:.1f}% ({sm.total_trades} trades)", "N/A (1 trade)", f"{sm.winning_trades}W / {sm.losing_trades}L"],
        ["Profit Factor", f"{sm.profit_factor:.2f}" if not np.isinf(sm.profit_factor) else "Inf", "N/A", "N/A"],
        ["Execution Frictions", f"Comm: {backtest_result.commission_bps} bps | Slip: {backtest_result.slippage_bps} bps", "Same Execution Drag", "Strict next_open timing"]
    ]
    perf_table = Table(perf_rows, colWidths=[150, 130, 130, 130])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent_blue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, table_border),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, table_bg_light]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 8))

    # SECTION 4: WALK-FORWARD VALIDATION (IF PRESENT)
    if walk_forward_summary is not None and not walk_forward_summary.empty:
        story.append(Paragraph("<b>4. Walk-Forward Chronological Validation</b>", h1_style))
        wf_rows = [[Paragraph(f"<b>{c}</b>", body_style) for c in walk_forward_summary.columns]]
        for _, r in walk_forward_summary.iterrows():
            wf_rows.append([Paragraph(str(v), body_style) for v in r])
        wf_table = Table(wf_rows, colWidths=[60, 110, 110, 50, 55, 50, 50, 50])
        wf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334E68")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, table_border),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, table_bg_light]),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(wf_table)
        story.append(Spacer(1, 8))

    # SECTION 5: MONTE CARLO RISK ANALYSIS (IF PRESENT)
    if monte_carlo_report is not None:
        story.append(Paragraph("<b>5. Monte Carlo Resampling & Risk of Ruin</b>", h1_style))
        mc_data = [
            [Paragraph("<b>Simulations:</b>", body_style), Paragraph(f"{monte_carlo_report.num_simulations:,}", body_style),
             Paragraph("<b>Median Ending Equity:</b>", body_style), Paragraph(f"${monte_carlo_report.terminal_equity_median:,.2f}", body_style)],
            [Paragraph("<b>5th Percentile (Worst 5%):</b>", body_style), Paragraph(f"${monte_carlo_report.terminal_percentiles.get('5%', 0):,.2f}", body_style),
             Paragraph("<b>95th Percentile (Best 5%):</b>", body_style), Paragraph(f"${monte_carlo_report.terminal_percentiles.get('95%', 0):,.2f}", body_style)],
            [Paragraph("<b>P(Drawdown >= 20%):</b>", body_style), Paragraph(f"{monte_carlo_report.probability_of_ruin_20pct*100:.1f}%", body_style),
             Paragraph("<b>P(Drawdown >= 30%):</b>", body_style), Paragraph(f"{monte_carlo_report.probability_of_ruin_30pct*100:.1f}%", body_style)],
            [Paragraph("<b>Mean Max Drawdown:</b>", body_style), Paragraph(f"{monte_carlo_report.mean_max_drawdown*100:.1f}%", body_style),
             Paragraph("<b>Worst-Case Simulated DD:</b>", body_style), Paragraph(f"{monte_carlo_report.worst_case_max_drawdown*100:.1f}%", body_style)]
        ]
        mc_table = Table(mc_data, colWidths=[130, 140, 130, 140])
        mc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), table_bg_light),
            ("GRID", (0, 0), (-1, -1), 0.5, table_border),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(mc_table)
        story.append(Spacer(1, 8))

    # SECTION 5.5: FUTURES SPECIFICATIONS & MARGIN ACCOUNTING (IF PRESENT)
    if futures_summary is not None:
        story.append(Paragraph("<b>Futures Contract Specifications & Margin Accounting</b>", h1_style))
        fut_data = [
            [Paragraph("<b>Contract Spec:</b>", body_style), Paragraph(str(futures_summary.get("name", "Futures")), body_style),
             Paragraph("<b>Lot Multiplier:</b>", body_style), Paragraph(str(futures_summary.get("multiplier", "1.0")), body_style)],
            [Paragraph("<b>Tick Size / Value:</b>", body_style), Paragraph(f"{futures_summary.get('tick_size', 0.05)} / {futures_summary.get('tick_value', 1.0)}", body_style),
             Paragraph("<b>Margin Requirement:</b>", body_style), Paragraph(f"Init: {futures_summary.get('initial_margin_rate', 0.1)*100:.1f}% | Maint: {futures_summary.get('maintenance_margin_rate', 0.08)*100:.1f}%", body_style)],
            [Paragraph("<b>Leveraged Futures P&L:</b>", body_style), Paragraph(str(futures_summary.get("net_profit", "N/A")), body_style),
             Paragraph("<b>Margin Calls:</b>", body_style), Paragraph(str(futures_summary.get("margin_calls", 0)), body_style)],
            [Paragraph("<b>Structural Caveat:</b>", body_style), Paragraph(str(futures_summary.get("stitching_caveats", "Continuous proxy")), body_style),
             Paragraph("<b>Rollover Schedule:</b>", body_style), Paragraph(str(futures_summary.get("rollover_cycle", "Standard")), body_style)]
        ]
        fut_table = Table(fut_data, colWidths=[130, 140, 130, 140])
        fut_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), table_bg_light),
            ("GRID", (0, 0), (-1, -1), 0.5, table_border),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(fut_table)
        story.append(Spacer(1, 8))

    # SECTION 6: MODEL RISK & LIMITATIONS
    story.append(KeepTogether([
        Paragraph("<b>6. Model Risk Disclosures & Research Limitations</b>", h1_style),
        Paragraph(
            "1. <b>Execution Realism:</b> Backtests simulate discrete fills at next-period Open with fixed basis point slippage. Live order-book liquidity, spread variations, and partial fills are not fully replicated.<br/>"
            "2. <b>Resampling Assumptions:</b> Monte Carlo simulations assume historical trade returns are independent and identically distributed (IID). Real-world markets exhibit regime clustering and structural shifts.<br/>"
            "3. <b>Continuous Futures Limitations:</b> Front-month stitched futures contain unadjusted roll gaps and do not capture carrying costs or dynamic roll yield.<br/>"
            "4. <b>Past Performance:</b> Historical backtest outcomes do not guarantee future statistical persistence or profitability.",
            body_style
        )
    ]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
