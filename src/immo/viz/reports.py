# -*- coding: utf-8 -*-
"""Génération de rapports PDF multi-pages pour l'analyse immobilière."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .market import (
    fig_price_comparison,
    fig_summary_table,
    fig_trend_iqr,
    fig_volatility,
    fig_volume,
    fig_yoy,
    setup_style,
)
from .signals import (
    SignalLike,
    fig_signal_dashboard,
    fig_signal_heatmap,
    fig_signal_overlay,
)


# ---------------------------------------------------------------------------
# Configuration du rapport
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Paramètres de mise en page et de métadonnées du rapport."""

    title: str = "Rapport Immobilier"
    subtitle: str = ""
    author: str = ""
    date: str = field(default_factory=lambda: dt.date.today().strftime("%d/%m/%Y"))
    logo_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Pages internes
# ---------------------------------------------------------------------------

def _fig_cover(config: ReportConfig) -> Figure:
    """Page de couverture avec titre, sous-titre, auteur et date."""
    setup_style()
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 paysage
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Bande décorative en haut
    ax.axhspan(0.88, 1.0, color="#1f3864", alpha=0.95)
    ax.axhspan(0.86, 0.88, color="#2ca02c", alpha=0.7)

    # Logo optionnel
    if config.logo_path:
        try:
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage
            import matplotlib.image as mpimg

            logo = mpimg.imread(config.logo_path)
            imagebox = OffsetImage(logo, zoom=0.3)
            ab = AnnotationBbox(imagebox, (0.92, 0.94), frameon=False)
            ax.add_artist(ab)
        except Exception:
            pass  # logo manquant — on continue sans

    # Textes
    ax.text(
        0.06, 0.72, config.title,
        fontsize=32, weight="bold", color="#1f3864",
        verticalalignment="top",
    )
    ax.text(
        0.06, 0.58, config.subtitle,
        fontsize=18, color="#444444",
        verticalalignment="top",
    )

    # Ligne de séparation fine
    ax.plot([0.06, 0.94], [0.50, 0.50], color="#cccccc", linewidth=1)

    if config.author:
        ax.text(
            0.06, 0.42, f"Auteur : {config.author}",
            fontsize=12, color="#666666",
        )
    ax.text(
        0.06, 0.36, f"Date : {config.date}",
        fontsize=12, color="#666666",
    )

    # Pied de page
    ax.text(
        0.06, 0.06,
        "Données DVF (Demandes de Valeurs Foncières) — data.gouv.fr",
        fontsize=9, color="#999999", style="italic",
    )

    return fig


def _fig_section_title(title: str, description: str = "") -> Figure:
    """Page de titre de section (optionnelle, pour le rapport complet)."""
    setup_style()
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.axhspan(0.45, 0.55, color="#1f3864", alpha=0.08)
    ax.text(
        0.5, 0.55, title,
        fontsize=26, weight="bold", color="#1f3864",
        ha="center", va="center",
    )
    if description:
        ax.text(
            0.5, 0.42, description,
            fontsize=14, color="#666666",
            ha="center", va="center",
        )
    return fig


# ---------------------------------------------------------------------------
# Helpers PDF
# ---------------------------------------------------------------------------

def _save_and_close(pdf: PdfPages, fig: Figure) -> None:
    """Sauvegarde une figure dans le PDF et la ferme."""
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# generate_market_report
# ---------------------------------------------------------------------------

def generate_market_report(
    agg: Any,  # pd.DataFrame
    label_col: str,
    config: ReportConfig,
    output_path: Path,
) -> None:
    """Génère un rapport PDF multi-pages d'analyse de marché.

    Pages :
        1. Couverture
        2. Tendance des prix (IQR + lissage)
        3. Volume de transactions
        4. Évolution annuelle (YoY)
        5. Volatilité
        6. Comparaison normalisée
        7. Tableau récapitulatif
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    communes_txt = ", ".join(sorted(agg[label_col].unique()))
    if not config.subtitle:
        config.subtitle = f"Communes : {communes_txt}"

    with PdfPages(str(output_path)) as pdf:
        # 1. Couverture
        _save_and_close(pdf, _fig_cover(config))

        # 2. Tendance des prix
        _save_and_close(pdf, fig_trend_iqr(agg, label_col))

        # 3. Volumes
        _save_and_close(pdf, fig_volume(agg, label_col))

        # 4. YoY
        _save_and_close(pdf, fig_yoy(agg, label_col))

        # 5. Volatilité
        _save_and_close(pdf, fig_volatility(agg, label_col))

        # 6. Comparaison normalisée
        _save_and_close(pdf, fig_price_comparison(agg, label_col))

        # 7. Tableau récapitulatif
        _save_and_close(pdf, fig_summary_table(agg, label_col))


# ---------------------------------------------------------------------------
# generate_signal_report
# ---------------------------------------------------------------------------

def generate_signal_report(
    agg: Any,  # pd.DataFrame
    signals: Sequence[SignalLike],
    config: ReportConfig,
    output_path: Path,
) -> None:
    """Génère un rapport PDF centré sur les signaux d'achat/vente.

    Pages :
        1. Couverture
        2. Tableau de bord des signaux
        3. Une page par commune avec overlay des signaux
        4. Heatmap des signaux (si DataFrame fourni)
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.subtitle:
        config.subtitle = "Analyse des signaux de marché"

    with PdfPages(str(output_path)) as pdf:
        # 1. Couverture
        _save_and_close(pdf, _fig_cover(config))

        # 2. Dashboard signaux
        _save_and_close(pdf, fig_signal_dashboard(signals))

        # 3. Overlay par commune
        communes = sorted({s.commune for s in signals})
        for commune in communes:
            sub = agg[agg["commune"] == commune] if "commune" in agg.columns else agg
            if sub.empty:
                continue
            _save_and_close(pdf, fig_signal_overlay(agg, signals, commune))

        # 4. Heatmap
        if signals:
            signals_df = pd.DataFrame(
                [
                    {
                        "date": s.date,
                        "commune": s.commune,
                        "signal_type": s.signal_type,
                        "confidence": s.confidence,
                    }
                    for s in signals
                ]
            )
            _save_and_close(pdf, fig_signal_heatmap(signals_df))


# ---------------------------------------------------------------------------
# generate_full_report
# ---------------------------------------------------------------------------

def generate_full_report(
    agg: Any,  # pd.DataFrame
    signals: Sequence[SignalLike],
    label_col: str,
    config: ReportConfig,
    output_path: Path,
) -> None:
    """Génère un rapport complet combinant analyse de marché et signaux.

    Sections :
        I.  Couverture
        II. Analyse de marché (tendance, volume, YoY, volatilité, comparaison, récap)
        III. Analyse des signaux (dashboard, overlays, heatmap)
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    communes_txt = ", ".join(sorted(agg[label_col].unique()))
    if not config.subtitle:
        config.subtitle = f"Communes : {communes_txt}"

    with PdfPages(str(output_path)) as pdf:
        # ── Couverture ──────────────────────────────────────────
        _save_and_close(pdf, _fig_cover(config))

        # ── Section Marché ──────────────────────────────────────
        _save_and_close(
            pdf,
            _fig_section_title(
                "I. Analyse de marché",
                "Tendances de prix, volumes, évolution et volatilité",
            ),
        )
        _save_and_close(pdf, fig_trend_iqr(agg, label_col))
        _save_and_close(pdf, fig_volume(agg, label_col))
        _save_and_close(pdf, fig_yoy(agg, label_col))
        _save_and_close(pdf, fig_volatility(agg, label_col))
        _save_and_close(pdf, fig_price_comparison(agg, label_col))
        _save_and_close(pdf, fig_summary_table(agg, label_col))

        # ── Section Signaux ─────────────────────────────────────
        if signals:
            _save_and_close(
                pdf,
                _fig_section_title(
                    "II. Signaux de marché",
                    "Recommandations d'achat et de vente basées sur les indicateurs",
                ),
            )
            _save_and_close(pdf, fig_signal_dashboard(signals))

            communes = sorted({s.commune for s in signals})
            for commune in communes:
                sub = agg[agg[label_col] == commune] if label_col in agg.columns else agg
                if sub.empty:
                    continue
                _save_and_close(pdf, fig_signal_overlay(agg, signals, commune))

            signals_df = pd.DataFrame(
                [
                    {
                        "date": s.date,
                        "commune": s.commune,
                        "signal_type": s.signal_type,
                        "confidence": s.confidence,
                    }
                    for s in signals
                ]
            )
            _save_and_close(pdf, fig_signal_heatmap(signals_df))
