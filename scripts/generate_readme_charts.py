#!/usr/bin/env python3
"""Generate showcase charts for the README.

Outputs SVG figures into docs/assets/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# -- project imports ----------------------------------------------------------
from immo.viz.market import (
    setup_style,
    fig_trend_iqr,
    fig_volume,
    fig_yoy,
    fig_volatility,
    _auto_interval,
    _euro_formatter,
    _format_date_axis,
)
from immo.viz.signals import fig_signal_overlay
from immo.analysis.signals import composite_signal
from immo.analysis.forecasting import forecast_ensemble
from immo.analysis.rates import plot_salary_vs_capital

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "outputs"
CHARTS_DIR = ROOT / "charts"
OUT_DIR = ROOT / "docs" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
agg = pd.read_csv(DATA_DIR / "metrics_communes_monthly.csv")
agg["date_mutation"] = pd.to_datetime(agg["date_mutation"])
communes = sorted(agg["commune"].unique())
print(f"  Communes: {communes}")
print(f"  Rows: {len(agg)}")

# Pick one commune for single-commune charts
FOCUS = "Brest"

# ===========================================================================
# 1. fig_trend.svg  -- Price trend with IQR band
# ===========================================================================
print("\n[1/7] fig_trend.svg ...")
setup_style()
fig = fig_trend_iqr(agg)
fig.savefig(OUT_DIR / "fig_trend.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("  -> saved")

# ===========================================================================
# 2. fig_signals.svg  -- Signal overlay for one commune
# ===========================================================================
print("\n[2/7] fig_signals.svg ...")
signals = composite_signal(agg)
fig = fig_signal_overlay(agg, signals, commune=FOCUS)
fig.savefig(OUT_DIR / "fig_signals.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("  -> saved")

# ===========================================================================
# 3. fig_forecast.svg  -- Forecast chart for one commune
# ===========================================================================
print("\n[3/7] fig_forecast.svg ...")
setup_style()
fc = forecast_ensemble(agg, commune=FOCUS, horizon_months=12)

# Historical data for the focus commune
hist = agg[agg["commune"] == FOCUS].sort_values("date_mutation").copy()

fig, ax = plt.subplots(figsize=(11, 5))

# Historical price
ax.plot(
    hist["date_mutation"],
    hist["prix_m2_median"],
    color="#1f77b4",
    linewidth=2,
    label="Historique (mediane)",
)

# Forecast line
ax.plot(
    fc["ds"],
    fc["yhat"],
    color="#ff7f0e",
    linewidth=2,
    linestyle="--",
    label="Prevision",
)

# Confidence band
ax.fill_between(
    fc["ds"],
    fc["yhat_lower"],
    fc["yhat_upper"],
    alpha=0.15,
    color="#ff7f0e",
    label="Intervalle de confiance",
)

# Vertical line at forecast start
last_hist = hist["date_mutation"].max()
ax.axvline(last_hist, linestyle=":", linewidth=1, color="#7f7f7f", alpha=0.7)

ax.set_title(f"Prevision des prix au m2 -- {FOCUS}")
ax.set_xlabel("Mois")
ax.set_ylabel("EUR / m2")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
_format_date_axis(ax, interval=_auto_interval(agg))
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_forecast.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("  -> saved")

# ===========================================================================
# 4. fig_rates.svg  -- Salary vs capital for 5 rates
# ===========================================================================
print("\n[4/7] fig_rates.svg ...")
setup_style()
fig = plot_salary_vs_capital(
    rates=[0.02, 0.025, 0.03, 0.035, 0.04],
    duration=25,
    capital_range=(100_000, 400_000),
)
fig.set_size_inches(11, 5)
fig.savefig(OUT_DIR / "fig_rates.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("  -> saved")

# ===========================================================================
# 5. fig_renovation.svg  -- Copy best renovation chart from charts/
# ===========================================================================
print("\n[5/7] fig_renovation.svg ...")
# Pick the waterfall totals SVG as the showcase chart
src_chart = CHARTS_DIR / "waterfall_totals.svg"
if src_chart.exists():
    shutil.copy2(src_chart, OUT_DIR / "fig_renovation.svg")
    print("  -> copied waterfall_totals.svg")
else:
    # Fallback: any SVG in charts/
    svgs = sorted(CHARTS_DIR.glob("*.svg"))
    if svgs:
        shutil.copy2(svgs[0], OUT_DIR / "fig_renovation.svg")
        print(f"  -> copied {svgs[0].name} (fallback)")
    else:
        print("  !! no SVG found in charts/")

# ===========================================================================
# 6. fig_volume.svg  -- Transaction volume stacked bars
# ===========================================================================
print("\n[6/7] fig_volume.svg ...")
setup_style()
fig = fig_volume(agg)
fig.savefig(OUT_DIR / "fig_volume.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("  -> saved")

# ===========================================================================
# 7. fig_dashboard.svg  -- Combined 2x2 dashboard
# ===========================================================================
print("\n[7/7] fig_dashboard.svg ...")
setup_style()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

palette = sns.color_palette("deep", n_colors=len(communes))

# --- Top-left: Trend (smoothed price) ---
ax = axes[0, 0]
for idx, (commune, sub) in enumerate(agg.sort_values("date_mutation").groupby("commune")):
    sub = sub.sort_values("date_mutation")
    color = palette[idx]
    ax.fill_between(sub["date_mutation"], sub["prix_m2_p25"], sub["prix_m2_p75"],
                    alpha=0.12, color=color)
    ax.plot(sub["date_mutation"], sub["prix_m2_smooth"], color=color, linewidth=2,
            label=commune)
ax.set_title("Prix au m2 (mediane lissee + IQR)")
ax.set_ylabel("EUR / m2")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
ax.legend(loc="best", fontsize=8)
_format_date_axis(ax, interval=6)

# --- Top-right: YoY change ---
ax = axes[0, 1]
for idx, (commune, sub) in enumerate(agg.sort_values("date_mutation").groupby("commune")):
    sub = sub.sort_values("date_mutation")
    ax.plot(sub["date_mutation"], sub["pct_chg_12m"], color=palette[idx], linewidth=1.8,
            label=commune)
ax.axhline(0, linestyle="--", linewidth=1, color="#7f7f7f", alpha=0.8)
ax.set_title("Evolution annuelle glissante (YoY %)")
ax.set_ylabel("Variation (%)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f} %"))
ax.legend(loc="best", fontsize=8)
_format_date_axis(ax, interval=6)

# --- Bottom-left: Volatility ---
ax = axes[1, 0]
for idx, (commune, sub) in enumerate(agg.sort_values("date_mutation").groupby("commune")):
    sub = sub.sort_values("date_mutation")
    color = palette[idx]
    ax.plot(sub["date_mutation"], sub["vol_6m"], color=color, linewidth=1.8, label=commune)
    ax.fill_between(sub["date_mutation"], 0, sub["vol_6m"], alpha=0.08, color=color)
ax.set_title("Volatilite (ecart-type glissant 6 mois)")
ax.set_ylabel("EUR / m2")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
ax.legend(loc="best", fontsize=8)
_format_date_axis(ax, interval=6)

# --- Bottom-right: Volume stacked bars ---
ax = axes[1, 1]
pivot = (
    agg.pivot_table(index="date_mutation", columns="commune",
                    values="n_transactions", aggfunc="sum")
    .fillna(0).sort_index()
)
bar_width = 20
bottoms = np.zeros(len(pivot))
for i, col in enumerate(pivot.columns):
    ax.bar(pivot.index, pivot[col], bottom=bottoms, width=bar_width,
           label=col, color=palette[i], edgecolor="white", linewidth=0.3)
    bottoms += pivot[col].values
ax.set_title("Volume de transactions (empile)")
ax.set_ylabel("Nb transactions")
ax.legend(loc="upper left", fontsize=8)
_format_date_axis(ax, interval=6)

fig.suptitle("Tableau de bord du marche immobilier", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_dashboard.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("  -> saved")

# ===========================================================================
# Summary
# ===========================================================================
print("\n--- Done! Generated files in docs/assets/ ---")
for f in sorted(OUT_DIR.glob("fig_*.svg")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
