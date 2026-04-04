# -*- coding: utf-8 -*-
"""Visualisation et rapports pour l'analyse immobilière."""
from .market import (
    fig_heatmap_monthly,
    fig_price_comparison,
    fig_purchasing_power,
    fig_summary_table,
    fig_trend_iqr,
    fig_volatility,
    fig_volume,
    fig_yoy,
    setup_style,
)
from .reports import (
    ReportConfig,
    generate_full_report,
    generate_market_report,
    generate_signal_report,
)
from .signals import (
    fig_composite_indicator,
    fig_signal_dashboard,
    fig_signal_heatmap,
    fig_signal_overlay,
)

__all__ = [
    # market
    "setup_style",
    "fig_trend_iqr",
    "fig_volume",
    "fig_yoy",
    "fig_volatility",
    "fig_price_comparison",
    "fig_heatmap_monthly",
    "fig_summary_table",
    "fig_purchasing_power",
    # signals
    "fig_signal_overlay",
    "fig_signal_dashboard",
    "fig_composite_indicator",
    "fig_signal_heatmap",
    # reports
    "ReportConfig",
    "generate_market_report",
    "generate_signal_report",
    "generate_full_report",
]
