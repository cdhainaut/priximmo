"""Visualisations des signaux d'achat/vente du marché immobilier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from immo.analysis.signals import SignalType

from .market import _auto_interval, _euro_formatter, _format_date_axis, setup_style

# ---------------------------------------------------------------------------
# Signal protocol — duck-typing compatible avec n'importe quel objet Signal
# ---------------------------------------------------------------------------


class SignalLike(Protocol):
    date: Any  # datetime-like
    commune: str
    signal_type: SignalType  # BUY / SELL / HOLD / STRONG_BUY / STRONG_SELL
    confidence: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_SIGNAL_COLORS = {
    SignalType.STRONG_BUY: "#006400",
    SignalType.BUY: "#2ca02c",
    SignalType.HOLD: "#ff7f0e",
    SignalType.SELL: "#d62728",
    SignalType.STRONG_SELL: "#8b0000",
}

_SIGNAL_LABELS_FR = {
    SignalType.STRONG_BUY: "Achat fort",
    SignalType.BUY: "Achat",
    SignalType.HOLD: "Neutre",
    SignalType.SELL: "Vente",
    SignalType.STRONG_SELL: "Vente forte",
}

_SIGNAL_MARKERS = {
    SignalType.STRONG_BUY: "^",
    SignalType.BUY: "^",
    SignalType.HOLD: "D",
    SignalType.SELL: "v",
    SignalType.STRONG_SELL: "v",
}

_SIGNAL_SIZES = {
    SignalType.STRONG_BUY: 160,
    SignalType.BUY: 100,
    SignalType.HOLD: 60,
    SignalType.SELL: 100,
    SignalType.STRONG_SELL: 160,
}

_SIGNAL_NUMERIC = {
    SignalType.STRONG_BUY: 2,
    SignalType.BUY: 1,
    SignalType.HOLD: 0,
    SignalType.SELL: -1,
    SignalType.STRONG_SELL: -2,
}


# ---------------------------------------------------------------------------
# fig_signal_overlay
# ---------------------------------------------------------------------------


def fig_signal_overlay(
    agg: pd.DataFrame,
    signals: Sequence[SignalLike],
    commune: str,
) -> Figure:
    """Prix au m² avec marqueurs de signaux superposés.

    Flèches vertes vers le haut pour achat, rouges vers le bas pour vente.
    La taille du marqueur reflète la confiance.
    """
    setup_style()

    sub = agg[agg["commune"] == commune].copy() if "commune" in agg.columns else agg.copy()
    sub = sub.sort_values("date_mutation")

    commune_signals = [s for s in signals if s.commune == commune]

    fig, ax = plt.subplots(figsize=(13, 6))

    # Bande IQR
    if "prix_m2_p25" in sub.columns and "prix_m2_p75" in sub.columns:
        ax.fill_between(
            sub["date_mutation"],
            sub["prix_m2_p25"],
            sub["prix_m2_p75"],
            alpha=0.12,
            color="#1f77b4",
        )

    # Courbe lissée
    price_col = "prix_m2_smooth" if "prix_m2_smooth" in sub.columns else "prix_m2_median"
    ax.plot(
        sub["date_mutation"],
        sub[price_col],
        linewidth=2,
        color="#1f77b4",
        label="Prix lissé (€/m²)",
    )
    # Médiane brute
    if "prix_m2_median" in sub.columns and price_col != "prix_m2_median":
        ax.plot(
            sub["date_mutation"],
            sub["prix_m2_median"],
            linewidth=0.7,
            linestyle=":",
            color="#1f77b4",
            alpha=0.45,
        )

    # Marqueurs de signaux
    legend_handles: dict[str, Any] = {}
    for sig in commune_signals:
        sig_date = pd.Timestamp(sig.date)
        sig_type = sig.signal_type
        color = _SIGNAL_COLORS.get(sig_type, "#7f7f7f")
        marker = _SIGNAL_MARKERS.get(sig_type, "o")
        base_size = _SIGNAL_SIZES.get(sig_type, 80)
        size = base_size * max(0.5, min(sig.confidence, 1.0))

        # Trouver le prix le plus proche de la date du signal
        diffs = (sub["date_mutation"] - sig_date).abs()
        if diffs.empty:
            continue
        closest_idx = diffs.idxmin()
        y_val = sub.loc[closest_idx, price_col]
        if pd.isna(y_val):
            continue

        ax.scatter(
            sig_date,
            y_val,
            marker=marker,
            s=size,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

        # Légende unique par type
        label_fr = _SIGNAL_LABELS_FR.get(sig_type, sig_type)
        if sig_type not in legend_handles:
            legend_handles[sig_type] = ax.scatter(
                [],
                [],
                marker=marker,
                s=base_size,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                label=label_fr,
            )

    ax.set_title(f"Signaux de marché — {commune}")
    ax.set_xlabel("Mois")
    ax.set_ylabel("€ / m²")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
    _format_date_axis(
        ax,
        interval=_auto_interval(sub.rename(columns={}) if "date_mutation" in sub.columns else sub),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_signal_dashboard
# ---------------------------------------------------------------------------


def fig_signal_dashboard(
    signals: Sequence[SignalLike],
) -> Figure:
    """Tableau de bord des derniers signaux par commune, avec code couleur."""
    setup_style()

    if not signals:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Aucun signal disponible", ha="center", va="center", fontsize=14)
        return fig

    # Regrouper par commune, garder le dernier signal
    by_commune: dict[str, SignalLike] = {}
    for sig in sorted(signals, key=lambda s: pd.Timestamp(s.date)):
        by_commune[sig.commune] = sig

    communes = sorted(by_commune.keys())
    n_rows = len(communes)

    col_labels = ["Commune", "Date", "Signal", "Confiance", "Raisons"]
    cell_text = []
    cell_colors = []

    for commune in communes:
        sig = by_commune[commune]
        sig_type = sig.signal_type
        color = _SIGNAL_COLORS.get(sig_type, "#7f7f7f")
        label_fr = _SIGNAL_LABELS_FR.get(sig_type, sig_type)
        date_str = pd.Timestamp(sig.date).strftime("%Y-%m-%d")
        conf_str = f"{sig.confidence:.0%}"
        reasons_str = "; ".join(sig.reasons[:3]) if sig.reasons else "—"

        cell_text.append([commune, date_str, label_fr, conf_str, reasons_str])

        # Couleur de la ligne entière — très léger
        import matplotlib.colors as mcolors

        rgba = mcolors.to_rgba(color, alpha=0.12)
        cell_colors.append([rgba] * len(col_labels))

    fig_height = max(3, 1.5 + 0.5 * n_rows)
    fig = plt.figure(figsize=(14, fig_height))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(
        "Tableau de bord des signaux — dernières recommandations",
        loc="left",
        pad=18,
        fontsize=14,
        weight="bold",
    )

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # En-têtes
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d9d9d9")
        cell.set_edgecolor("white")

    for i in range(1, n_rows + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_edgecolor("white")

    tbl.auto_set_column_width(list(range(len(col_labels))))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_composite_indicator
# ---------------------------------------------------------------------------


def fig_composite_indicator(
    components: dict[str, pd.Series],
    commune: str,
) -> Figure:
    """Graphique multi-panneaux des composantes du signal.

    Parameters
    ----------
    components : dict[str, pd.Series]
        Dictionnaire nom -> série temporelle de chaque composante.
        Clés typiques : "z_score", "momentum", "mean_reversion",
        "volume", "rate_adjusted".
    commune : str
        Nom de la commune (pour le titre).
    """
    setup_style()

    n_panels = len(components)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Aucune composante disponible", ha="center", va="center")
        return fig

    component_labels_fr = {
        "z_score": "Z-score prix",
        "momentum": "Momentum",
        "mean_reversion": "Retour à la moyenne",
        "volume": "Indicateur de volume",
        "rate_adjusted": "Ajusté aux taux",
    }

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.8 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    palette = sns.color_palette("deep", n_colors=n_panels)

    for idx, (name, series) in enumerate(components.items()):
        ax = axes[idx]
        series = series.dropna().sort_index()
        if series.empty:
            ax.text(0.5, 0.5, "Pas de données", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(component_labels_fr.get(name, name))
            continue

        color = palette[idx % len(palette)]
        ax.plot(series.index, series.values, color=color, linewidth=1.5)
        ax.fill_between(series.index, 0, series.values, alpha=0.1, color=color)
        ax.axhline(0, linestyle="--", linewidth=0.8, color="#7f7f7f", alpha=0.6)

        label_fr = component_labels_fr.get(name, name)
        ax.set_ylabel(label_fr, fontsize=10)

        # Zones coloriées
        ax.axhspan(0, ax.get_ylim()[1] * 1.5, alpha=0.03, color="#2ca02c")
        ax.axhspan(ax.get_ylim()[0] * 1.5, 0, alpha=0.03, color="#d62728")

    axes[-1].set_xlabel("Mois")
    if hasattr(axes[-1].xaxis, "set_major_locator"):
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(
        f"Indicateurs composites — {commune}",
        fontsize=14,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_signal_heatmap
# ---------------------------------------------------------------------------


def fig_signal_heatmap(
    signals_df: pd.DataFrame,
) -> Figure:
    """Heatmap : communes x temps, coloré par force du signal.

    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame avec au minimum les colonnes :
        ``date``, ``commune``, ``signal_type``.
    """
    setup_style()

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["signal_num"] = df["signal_type"].map(
        lambda st: _SIGNAL_NUMERIC.get(SignalType(st) if not isinstance(st, SignalType) else st, 0)
    )
    df["period"] = df["date"].dt.to_period("M").astype(str)

    pivot = df.pivot_table(
        index="commune",
        columns="period",
        values="signal_num",
        aggfunc="last",
    ).sort_index()

    # Trier les colonnes chronologiquement
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), max(4, len(pivot) * 0.6)))

    cmap = sns.diverging_palette(10, 130, s=85, l=50, as_cmap=True)
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=0,
        vmin=-2,
        vmax=2,
        annot=False,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Signal (−2 Vente forte → +2 Achat fort)", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Carte des signaux — communes × période")
    ax.set_xlabel("Période")
    ax.set_ylabel("Commune")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig
