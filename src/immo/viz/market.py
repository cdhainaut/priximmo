# -*- coding: utf-8 -*-
"""Visualisations du marché immobilier — prix, volumes, tendances."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Palette & couleurs
# ---------------------------------------------------------------------------
_PALETTE = sns.color_palette("deep")
_BUY_COLOR = "#2ca02c"
_SELL_COLOR = "#d62728"
_NEUTRAL_COLOR = "#7f7f7f"

_MONTHS_FR = [
    "Jan",
    "Fév",
    "Mar",
    "Avr",
    "Mai",
    "Jun",
    "Jul",
    "Aoû",
    "Sep",
    "Oct",
    "Nov",
    "Déc",
]


# ---------------------------------------------------------------------------
# Style global
# ---------------------------------------------------------------------------

def setup_style() -> None:
    """Configure le style seaborn/matplotlib pour tous les graphiques."""
    sns.set_context("paper", font_scale=1.1)
    sns.set_style(
        "darkgrid",
        {
            "axes.facecolor": "#f5f5f5",
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        },
    )
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.family": "sans-serif",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.85,
            "figure.autolayout": True,
        }
    )


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _format_date_axis(ax: plt.Axes, interval: int = 3) -> None:
    """Formate l'axe X en dates mensuelles."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _euro_formatter(x: float, _pos: int | None = None) -> str:
    if abs(x) >= 1_000:
        return f"{x:,.0f} €".replace(",", "\u202f")
    return f"{x:.0f} €"


def _pct_formatter(x: float, _pos: int | None = None) -> str:
    return f"{x:+.1f} %"


def _auto_interval(agg: pd.DataFrame) -> int:
    """Calcule un intervalle de ticks raisonnable selon l'étendue temporelle."""
    span_months = (
        (agg["date_mutation"].max() - agg["date_mutation"].min()).days / 30.0
    )
    if span_months > 60:
        return 6
    if span_months > 24:
        return 3
    return 1


# ---------------------------------------------------------------------------
# fig_trend_iqr
# ---------------------------------------------------------------------------

def fig_trend_iqr(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Prix au m² — médiane lissée avec bande inter-quartile (IQR).

    Affiche une courbe lissée par commune/groupe et la zone P25-P75.
    """
    setup_style()
    groups = agg.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        color = palette[idx % len(palette)]
        ax.fill_between(
            sub["date_mutation"],
            sub["prix_m2_p25"],
            sub["prix_m2_p75"],
            alpha=0.15,
            color=color,
        )
        ax.plot(
            sub["date_mutation"],
            sub["prix_m2_smooth"],
            color=color,
            linewidth=2,
            label=f"{label} (lissé)",
        )
        # Médiane brute en pointillés fins
        ax.plot(
            sub["date_mutation"],
            sub["prix_m2_median"],
            color=color,
            linewidth=0.7,
            linestyle=":",
            alpha=0.5,
        )

    ax.set_title("Prix au m² — médiane mensuelle + lissage (bande IQR)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("€ / m²")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
    _format_date_axis(ax, interval=_auto_interval(agg))
    ax.legend(loc="best", ncol=min(n_groups, 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_volume
# ---------------------------------------------------------------------------

def fig_volume(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Volume de transactions empilé par mois."""
    setup_style()
    pivot = (
        agg.pivot_table(
            index="date_mutation",
            columns=label_col,
            values="n_transactions",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = max(15, 25 - pivot.shape[1] * 2)
    bottoms = np.zeros(len(pivot))
    palette = sns.color_palette("deep", n_colors=max(len(pivot.columns), 1))

    for i, col in enumerate(pivot.columns):
        ax.bar(
            pivot.index,
            pivot[col],
            bottom=bottoms,
            width=bar_width,
            label=col,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.3,
        )
        bottoms += pivot[col].values

    ax.set_title("Volume de transactions par mois (empilé)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de transactions")
    _format_date_axis(ax, interval=_auto_interval(agg))
    ax.legend(loc="upper left", ncol=min(len(pivot.columns), 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_yoy
# ---------------------------------------------------------------------------

def fig_yoy(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Évolution annuelle glissante (Year-over-Year %)."""
    setup_style()
    groups = agg.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        ax.plot(
            sub["date_mutation"],
            sub["pct_chg_12m"],
            linewidth=1.8,
            label=label,
            color=palette[idx % len(palette)],
        )

    ax.axhline(0, linestyle="--", linewidth=1, color=_NEUTRAL_COLOR, alpha=0.8)
    # Zone positive / négative
    xlim = ax.get_xlim()
    ax.axhspan(0, ax.get_ylim()[1] * 2, alpha=0.03, color=_BUY_COLOR)
    ax.axhspan(ax.get_ylim()[0] * 2, 0, alpha=0.03, color=_SELL_COLOR)
    ax.set_xlim(xlim)

    ax.set_title("Évolution annuelle glissante (YoY %)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Variation annuelle (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    _format_date_axis(ax, interval=_auto_interval(agg))
    ax.legend(loc="best", ncol=min(n_groups, 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_volatility
# ---------------------------------------------------------------------------

def fig_volatility(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Volatilité glissante 6 mois (écart-type)."""
    setup_style()
    groups = agg.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        color = palette[idx % len(palette)]
        ax.plot(
            sub["date_mutation"],
            sub["vol_6m"],
            linewidth=1.8,
            label=label,
            color=color,
        )
        ax.fill_between(
            sub["date_mutation"],
            0,
            sub["vol_6m"],
            alpha=0.08,
            color=color,
        )

    ax.set_title("Volatilité (écart-type glissant 6 mois)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("€ / m²")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
    _format_date_axis(ax, interval=_auto_interval(agg))
    ax.legend(loc="best", ncol=min(n_groups, 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_price_comparison
# ---------------------------------------------------------------------------

def fig_price_comparison(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Comparaison multi-communes normalisée (base 100 au début)."""
    setup_style()
    groups = agg.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation").copy()
        series = sub["prix_m2_smooth"].dropna()
        if series.empty:
            continue
        first_val = series.iloc[0]
        if first_val == 0:
            continue
        normalized = (series / first_val) * 100
        ax.plot(
            sub.loc[normalized.index, "date_mutation"],
            normalized,
            linewidth=2,
            label=label,
            color=palette[idx % len(palette)],
        )

    ax.axhline(100, linestyle="--", linewidth=1, color=_NEUTRAL_COLOR, alpha=0.6)
    ax.set_title("Comparaison des prix (base 100 au départ)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Indice (base 100)")
    _format_date_axis(ax, interval=_auto_interval(agg))
    ax.legend(loc="best", ncol=min(n_groups, 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_heatmap_monthly
# ---------------------------------------------------------------------------

def fig_heatmap_monthly(
    agg: pd.DataFrame,
    commune: str,
) -> Figure:
    """Heatmap mensuelle des prix (mois x années) pour une commune."""
    setup_style()
    sub = agg[agg["commune"] == commune].copy() if "commune" in agg.columns else agg.copy()
    sub = sub.sort_values("date_mutation")
    sub["year"] = sub["date_mutation"].dt.year
    sub["month"] = sub["date_mutation"].dt.month

    pivot = sub.pivot_table(
        index="month", columns="year", values="prix_m2_median", aggfunc="median"
    )
    # Renommer les mois en français
    pivot.index = [_MONTHS_FR[m - 1] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "€ / m²"},
        ax=ax,
    )
    ax.set_title(f"Prix au m² mensuel — {commune}")
    ax.set_xlabel("Année")
    ax.set_ylabel("Mois")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_summary_table
# ---------------------------------------------------------------------------

def fig_summary_table(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Tableau récapitulatif du dernier mois disponible, rendu en figure matplotlib."""
    setup_style()

    # Dernier mois par label
    last_idx = (
        agg.groupby(label_col)["date_mutation"].transform("max") == agg["date_mutation"]
    )
    last = agg[last_idx].copy().sort_values(label_col)

    display_cols = [
        label_col,
        "date_mutation",
        "prix_m2_median",
        "prix_m2_p25",
        "prix_m2_p75",
        "n_transactions",
        "pct_chg_3m",
        "pct_chg_12m",
        "vol_6m",
        "anomaly_score",
    ]
    available = [c for c in display_cols if c in last.columns]
    last = last[available].copy()

    headers_fr = {
        "commune": "Commune",
        "groupe": "Groupe",
        "date_mutation": "Dernier mois",
        "prix_m2_median": "Médiane €/m²",
        "prix_m2_p25": "P25 €/m²",
        "prix_m2_p75": "P75 €/m²",
        "n_transactions": "Nb transactions",
        "pct_chg_3m": "Δ 3 mois",
        "pct_chg_12m": "Δ 12 mois",
        "vol_6m": "Volatilité 6m",
        "anomaly_score": "Score anomalie",
    }

    # Formatage
    last["date_mutation"] = last["date_mutation"].dt.strftime("%Y-%m")

    def _safe_fmt(val: object, fmt_fn) -> str:
        try:
            return fmt_fn(float(val))
        except (ValueError, TypeError):
            return "—"

    fmt_map = {
        "prix_m2_median": lambda x: f"{x:,.0f}".replace(",", "\u202f"),
        "prix_m2_p25": lambda x: f"{x:,.0f}".replace(",", "\u202f"),
        "prix_m2_p75": lambda x: f"{x:,.0f}".replace(",", "\u202f"),
        "n_transactions": lambda x: f"{x:.0f}",
        "pct_chg_3m": lambda x: f"{x:+.1f} %",
        "pct_chg_12m": lambda x: f"{x:+.1f} %",
        "vol_6m": lambda x: f"{x:,.0f}".replace(",", "\u202f"),
        "anomaly_score": lambda x: f"{x:+.2f}",
    }
    for col, fn in fmt_map.items():
        if col in last.columns:
            last[col] = last[col].apply(lambda v, _fn=fn: _safe_fmt(v, _fn))

    col_labels = [headers_fr.get(c, c) for c in available]

    fig = plt.figure(figsize=(12, max(3, 1.5 + 0.45 * len(last))))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(
        "Tableau récapitulatif — dernier mois disponible",
        loc="left",
        pad=20,
        fontsize=14,
        weight="bold",
    )

    tbl = ax.table(
        cellText=last.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # En-têtes en gras, fond gris
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d9d9d9")
        cell.set_edgecolor("white")

    # Lignes alternées
    for i in range(1, len(last) + 1):
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            cell.set_edgecolor("white")
            if i % 2 == 0:
                cell.set_facecolor("#f2f2f2")
            else:
                cell.set_facecolor("white")

    tbl.auto_set_column_width(list(range(len(col_labels))))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_purchasing_power
# ---------------------------------------------------------------------------

def fig_purchasing_power(
    prix_m2: pd.Series,
    rates: pd.Series,
    salary: float = 2500.0,
) -> Figure:
    """Surface achetable au fil du temps étant donné l'évolution des taux.

    Hypothèses simplifiées :
    - Mensualité max = 35 % du salaire net
    - Durée d'emprunt = 20 ans (240 mois)
    - Taux fourni en % annuel, converti en mensuel

    Parameters
    ----------
    prix_m2 : pd.Series
        Série indexée par date, contenant le prix médian au m².
    rates : pd.Series
        Série indexée par date, contenant le taux d'intérêt annuel (en %).
    salary : float
        Salaire mensuel net en euros.
    """
    setup_style()

    mensualite_max = salary * 0.35
    duree_mois = 240  # 20 ans

    # Aligner les deux séries
    common_idx = prix_m2.dropna().index.intersection(rates.dropna().index)
    prix_m2 = prix_m2.loc[common_idx].sort_index()
    rates = rates.loc[common_idx].sort_index()

    taux_mensuel = (rates / 100) / 12
    # Capacité d'emprunt = mensualité * [(1 - (1+r)^-n) / r]
    # Quand r ~ 0 on utilise la limite : capacité = mensualité * n
    capacite = pd.Series(index=common_idx, dtype=float)
    for dt_idx in common_idx:
        r = taux_mensuel.loc[dt_idx]
        if abs(r) < 1e-9:
            capacite.loc[dt_idx] = mensualite_max * duree_mois
        else:
            capacite.loc[dt_idx] = mensualite_max * (1 - (1 + r) ** (-duree_mois)) / r

    m2_achetables = capacite / prix_m2

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    color_m2 = _PALETTE[0]
    color_rate = _PALETTE[3]

    ax1.fill_between(common_idx, 0, m2_achetables, alpha=0.15, color=color_m2)
    ax1.plot(
        common_idx,
        m2_achetables,
        color=color_m2,
        linewidth=2,
        label=f"m² achetables (salaire {salary:,.0f} €)".replace(",", "\u202f"),
    )
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Surface achetable (m²)", color=color_m2)
    ax1.tick_params(axis="y", labelcolor=color_m2)

    ax2 = ax1.twinx()
    ax2.plot(
        common_idx,
        rates,
        color=color_rate,
        linewidth=1.5,
        linestyle="--",
        label="Taux d'intérêt (%)",
    )
    ax2.set_ylabel("Taux d'intérêt annuel (%)", color=color_rate)
    ax2.tick_params(axis="y", labelcolor=color_rate)

    ax1.set_title(
        f"Pouvoir d'achat immobilier — salaire {salary:,.0f} €/mois, emprunt 20 ans".replace(
            ",", "\u202f"
        )
    )
    _format_date_axis(ax1, interval=_auto_interval(
        pd.DataFrame({"date_mutation": common_idx})
    ) if len(common_idx) > 2 else 3)

    # Légendes combinées
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    return fig
