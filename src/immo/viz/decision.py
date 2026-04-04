"""Visualisations decisionnelles — graphiques d'aide a l'achat immobilier.

Ces graphiques repondent aux questions "faut-il acheter ?", "ou ?", "quand ?"
et ne sont pas de simples representations de donnees.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from immo.analysis.decision import drawdown_from_peak

from .market import _auto_interval, _euro_formatter, _format_date_axis, _pct_formatter, setup_style

# ---------------------------------------------------------------------------
# Palette & constantes
# ---------------------------------------------------------------------------
_PALETTE = sns.color_palette("deep")
_BUY_COLOR = "#2ca02c"
_SELL_COLOR = "#d62728"
_NEUTRAL_COLOR = "#7f7f7f"

_BOOM_COLOR = "#d62728"
_BULL_COLOR = "#b6d7a8"
_CAPITULATION_COLOR = "#f6b26b"
_BOTTOM_COLOR = "#006400"


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _compute_volume_zscore(agg: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Ajoute une colonne ``vol_zscore`` normalisant n_transactions par commune."""
    df = agg.copy()
    grp = df.groupby(label_col)["n_transactions"]
    df["vol_zscore"] = grp.transform(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
    return df


# ---------------------------------------------------------------------------
# 1. fig_market_phase_diagram
# ---------------------------------------------------------------------------


def fig_market_phase_diagram(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Diagramme de phase du marche : momentum x volume z-score.

    Bulles dimensionnees par la volatilite, couleur par commune.
    Quadrants : Boom, Bull, Capitulation, Bottom.
    """
    setup_style()
    df = _compute_volume_zscore(agg, label_col)
    groups = df.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 9))

    # ------ Quadrant backgrounds ------
    xlim_data = df["pct_chg_3m"].dropna()
    ylim_data = df["vol_zscore"].dropna()
    x_margin = max(abs(xlim_data.min()), abs(xlim_data.max()), 5) * 1.3
    y_margin = max(abs(ylim_data.min() - 0.5), abs(ylim_data.max() - 0.5), 2) * 1.3

    x_lo, x_hi = -x_margin, x_margin
    y_lo, y_hi = 0.5 - y_margin, 0.5 + y_margin

    # Top-right: Boom (red)
    ax.axhspan(0.5, y_hi, xmin=0, xmax=1, alpha=0.0)  # placeholder for limits
    ax.fill_between([0, x_hi], 0.5, y_hi, alpha=0.07, color=_BOOM_COLOR, zorder=0)
    # Bottom-right: Bull (light green)
    ax.fill_between([0, x_hi], y_lo, 0.5, alpha=0.07, color=_BULL_COLOR, zorder=0)
    # Top-left: Capitulation (orange)
    ax.fill_between([x_lo, 0], 0.5, y_hi, alpha=0.07, color=_CAPITULATION_COLOR, zorder=0)
    # Bottom-left: Bottom (dark green)
    ax.fill_between([x_lo, 0], y_lo, 0.5, alpha=0.07, color=_BOTTOM_COLOR, zorder=0)

    # Quadrant dividers
    ax.axvline(0, color=_NEUTRAL_COLOR, linewidth=1, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(0.5, color=_NEUTRAL_COLOR, linewidth=1, linestyle="--", alpha=0.6, zorder=1)

    # Quadrant labels
    label_kw = dict(fontsize=11, fontweight="bold", alpha=0.35, ha="center", va="center", zorder=1)
    ax.text(x_hi * 0.55, y_hi * 0.75 + 0.5 * 0.25, "BOOM", color=_BOOM_COLOR, **label_kw)
    ax.text(x_hi * 0.55, y_lo * 0.55 + 0.5 * 0.45, "BULL", color="#38761d", **label_kw)
    ax.text(x_lo * 0.55, y_hi * 0.75 + 0.5 * 0.25, "CAPITULATION", color="#b45f06", **label_kw)
    ax.text(x_lo * 0.55, y_lo * 0.55 + 0.5 * 0.45, "BOTTOM", color=_BOTTOM_COLOR, **label_kw)

    # ------ Plot each commune ------
    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation").dropna(subset=["pct_chg_3m", "vol_zscore"])
        if sub.empty:
            continue
        color = palette[idx % len(palette)]
        vol_sizes = sub["vol_6m"].fillna(0)
        # Normalize bubble size: min 30, max 400
        if vol_sizes.max() > vol_sizes.min():
            norm_sizes = (
                30 + (vol_sizes - vol_sizes.min()) / (vol_sizes.max() - vol_sizes.min()) * 370
            )
        else:
            norm_sizes = pd.Series(120, index=sub.index)

        # Trail: last 6 months (smaller, fading)
        n_trail = min(6, len(sub))
        trail = sub.iloc[-n_trail:]

        for j in range(len(trail)):
            alpha = 0.15 + 0.85 * (j / max(len(trail) - 1, 1))
            size_factor = 0.3 + 0.7 * (j / max(len(trail) - 1, 1))
            is_latest = j == len(trail) - 1
            ax.scatter(
                trail.iloc[j]["pct_chg_3m"],
                trail.iloc[j]["vol_zscore"],
                s=float(norm_sizes.iloc[-n_trail + j]) * (1.0 if is_latest else size_factor),
                color=color,
                alpha=alpha,
                edgecolors="white" if is_latest else "none",
                linewidths=1.2 if is_latest else 0,
                zorder=4 if is_latest else 3,
                label=label if is_latest else None,
            )

        # Arrow trajectory: connect last 3 points
        if len(trail) >= 2:
            n_arrow = min(3, len(trail))
            arrow_pts = trail.iloc[-n_arrow:]
            for k in range(len(arrow_pts) - 1):
                ax.annotate(
                    "",
                    xy=(arrow_pts.iloc[k + 1]["pct_chg_3m"], arrow_pts.iloc[k + 1]["vol_zscore"]),
                    xytext=(arrow_pts.iloc[k]["pct_chg_3m"], arrow_pts.iloc[k]["vol_zscore"]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.5,
                        alpha=0.6 + 0.4 * (k / max(n_arrow - 2, 1)),
                    ),
                    zorder=2,
                )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Momentum prix 3 mois (%)")
    ax.set_ylabel("Volume z-score")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_title("Phase de marche \u2014 Ou en est chaque commune ?")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. fig_affordability_evolution
# ---------------------------------------------------------------------------


def fig_affordability_evolution(
    affordability_df: pd.DataFrame,
    commune: str,
) -> Figure:
    """Pouvoir d'achat en m2 pour un budget mensuel de 3000 EUR.

    Parameters
    ----------
    affordability_df : pd.DataFrame
        Colonnes attendues: ``date``, ``commune``, ``m2_achetables``,
        ``taux_interet`` (en %, e.g. 3.5).
    commune : str
        Commune a afficher.
    """
    setup_style()
    sub = affordability_df[affordability_df["commune"] == commune].copy()
    sub = sub.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    m2 = sub["m2_achetables"]
    dates = sub["date"]
    mean_m2 = float(m2.mean())
    peak_m2 = float(m2.max())
    current_m2 = float(m2.iloc[-1]) if len(m2) > 0 else 0.0
    pct_vs_peak = ((current_m2 - peak_m2) / peak_m2 * 100) if peak_m2 > 0 else 0.0

    # Area fill: green above mean, red below
    ax1.fill_between(
        dates,
        m2,
        mean_m2,
        where=(m2 >= mean_m2),
        interpolate=True,
        alpha=0.25,
        color=_BUY_COLOR,
        label="Au-dessus de la moyenne",
    )
    ax1.fill_between(
        dates,
        m2,
        mean_m2,
        where=(m2 < mean_m2),
        interpolate=True,
        alpha=0.25,
        color=_SELL_COLOR,
        label="En-dessous de la moyenne",
    )
    ax1.plot(dates, m2, color="#1f77b4", linewidth=2)

    # Horizontal lines
    ax1.axhline(
        mean_m2,
        color=_NEUTRAL_COLOR,
        linewidth=1,
        linestyle=":",
        alpha=0.7,
        label=f"Moyenne ({mean_m2:.0f} m\u00b2)",
    )
    ax1.axhline(
        peak_m2,
        color=_BUY_COLOR,
        linewidth=1,
        linestyle="--",
        alpha=0.5,
        label=f"Pic ({peak_m2:.0f} m\u00b2)",
    )

    ax1.set_ylabel("Surface achetable (m\u00b2)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    # Annotation
    if len(dates) > 0:
        ax1.annotate(
            f"{current_m2:.0f} m\u00b2 ({pct_vs_peak:+.0f}% vs pic)",
            xy=(dates.iloc[-1], current_m2),
            xytext=(15, 20),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1f77b4", alpha=0.85),
        )

    # Secondary axis: interest rate
    if "taux_interet" in sub.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            dates,
            sub["taux_interet"],
            color=_PALETTE[3],
            linewidth=1.5,
            linestyle="--",
            label="Taux d'int\u00e9r\u00eat (%)",
        )
        ax2.set_ylabel("Taux d'int\u00e9r\u00eat annuel (%)", color=_PALETTE[3])
        ax2.tick_params(axis="y", labelcolor=_PALETTE[3])
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax1.legend(loc="upper right", fontsize=8)

    ax1.set_xlabel("Mois")
    ax1.set_title("Pouvoir d\u2019achat \u2014 Combien de m\u00b2 pour 3\u202f000 \u20ac/mois ?")
    _format_date_axis(ax1, interval=3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. fig_drawdown
# ---------------------------------------------------------------------------


def fig_drawdown(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Distance au pic par commune (drawdown en %).

    Zones colorees: 0/-5 % = normal, -5/-15 % = correction, <-15 % = opportunite.
    """
    setup_style()
    groups = agg.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    all_min = 0.0
    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        price_col = "prix_m2_smooth" if "prix_m2_smooth" in sub.columns else "prix_m2_median"
        dd = drawdown_from_peak(sub[price_col].dropna())
        color = palette[idx % len(palette)]

        dates = sub.loc[dd.index, "date_mutation"]
        ax.plot(dates, dd, linewidth=1.8, label=label, color=color)
        ax.fill_between(dates, 0, dd, alpha=0.08, color=color)
        all_min = min(all_min, dd.min())

    # Color zones
    y_bottom = min(all_min * 1.1, -20)
    ax.axhspan(0, -5, alpha=0.06, color=_BUY_COLOR, zorder=0)
    ax.axhspan(-5, -15, alpha=0.06, color="#ff7f0e", zorder=0)
    ax.axhspan(-15, y_bottom, alpha=0.06, color=_SELL_COLOR, zorder=0)

    # Zone labels on the right edge
    right_x = 0.98
    zone_kw = dict(fontsize=8, ha="right", va="center", transform=ax.transAxes, alpha=0.5)
    ax.text(
        right_x, ax.transData.inverted().transform(ax.transAxes.transform((0, 0)))[1], "", **zone_kw
    )  # dummy

    ax.annotate(
        "Normal",
        xy=(1.01, -2.5),
        xycoords=("axes fraction", "data"),
        fontsize=8,
        color=_BUY_COLOR,
        alpha=0.7,
        va="center",
    )
    ax.annotate(
        "Correction",
        xy=(1.01, -10),
        xycoords=("axes fraction", "data"),
        fontsize=8,
        color="#ff7f0e",
        alpha=0.7,
        va="center",
    )
    ax.annotate(
        "Opportunit\u00e9",
        xy=(1.01, -17.5),
        xycoords=("axes fraction", "data"),
        fontsize=8,
        color=_SELL_COLOR,
        alpha=0.7,
        va="center",
    )

    ax.axhline(0, color=_NEUTRAL_COLOR, linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axhline(-5, color=_NEUTRAL_COLOR, linewidth=0.5, linestyle=":", alpha=0.3)
    ax.axhline(-15, color=_NEUTRAL_COLOR, linewidth=0.5, linestyle=":", alpha=0.3)

    ax.set_ylim(y_bottom, 2)
    ax.set_title("Distance au pic \u2014 Opportunit\u00e9s d\u2019achat")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f} %"))
    _format_date_axis(ax, interval=_auto_interval(agg))
    ax.legend(loc="lower left", ncol=min(n_groups, 3))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. fig_price_volume_divergence
# ---------------------------------------------------------------------------


def fig_price_volume_divergence(
    agg: pd.DataFrame,
    commune: str,
) -> Figure:
    """Divergence prix/volume pour une commune : momentum normalise + score de divergence."""
    setup_style()
    sub = agg[agg["commune"] == commune].copy() if "commune" in agg.columns else agg.copy()
    sub = sub.sort_values("date_mutation").dropna(subset=["pct_chg_3m"])

    dates = sub["date_mutation"]

    # Price momentum (smoothed)
    price_mom = sub["pct_chg_3m"].rolling(3, min_periods=1, center=True).mean()
    # Volume momentum: pct change of n_transactions, smoothed
    vol_raw = sub["n_transactions"].pct_change(3) * 100
    vol_mom = vol_raw.rolling(3, min_periods=1, center=True).mean()

    # Normalize both to [-1, 1] range for comparison
    def _normalize(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if s_clean.empty or s_clean.std() == 0:
            return s * 0
        return (s - s.mean()) / s.std()

    price_norm = _normalize(price_mom)
    vol_norm = _normalize(vol_mom)

    # Divergence score: volume momentum - price momentum
    divergence = vol_norm - price_norm

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    # --- Top panel: normalized momentum ---
    ax_top.plot(dates, price_norm, linewidth=1.8, color=_PALETTE[0], label="Momentum prix")
    ax_top.plot(dates, vol_norm, linewidth=1.8, color=_PALETTE[1], label="Momentum volume")
    ax_top.axhline(0, linewidth=0.8, linestyle="--", color=_NEUTRAL_COLOR, alpha=0.5)
    ax_top.set_ylabel("Momentum normalis\u00e9")
    ax_top.legend(loc="upper left", fontsize=9)
    ax_top.set_title(f"Divergence prix/volume \u2014 {commune}")

    # --- Bottom panel: divergence bars ---
    colors_div = [_BUY_COLOR if v >= 0 else _SELL_COLOR for v in divergence]
    ax_bot.bar(dates, divergence, width=20, color=colors_div, alpha=0.7, edgecolor="none")
    ax_bot.axhline(0, linewidth=0.8, linestyle="-", color=_NEUTRAL_COLOR, alpha=0.5)
    ax_bot.set_ylabel("Score de divergence")
    ax_bot.set_xlabel("Mois")

    # Annotations
    pos_patch = mpatches.Patch(color=_BUY_COLOR, alpha=0.7, label="Accumulation (achat)")
    neg_patch = mpatches.Patch(color=_SELL_COLOR, alpha=0.7, label="Distribution (vente)")
    ax_bot.legend(handles=[pos_patch, neg_patch], loc="upper left", fontsize=8)

    _format_date_axis(ax_bot, interval=_auto_interval(sub))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. fig_commune_scorecard
# ---------------------------------------------------------------------------


def fig_commune_scorecard(
    scorecard_df: pd.DataFrame,
) -> Figure:
    """Classement des communes par score d'attractivite (barres horizontales).

    Parameters
    ----------
    scorecard_df : pd.DataFrame
        Colonnes attendues: ``commune``, ``score`` (0-100), ``phase`` (str),
        ``signal`` (str, e.g. "Achat", "Vente", "Neutre").
    """
    setup_style()
    df = scorecard_df.sort_values("score", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(df) + 1.5)))

    # Color gradient: red (0) -> yellow (50) -> green (100)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_cmap", [_SELL_COLOR, "#ff7f0e", _BUY_COLOR]
    )
    norm = mcolors.Normalize(vmin=0, vmax=100)
    colors = [cmap(norm(s)) for s in df["score"]]

    bars = ax.barh(df["commune"], df["score"], color=colors, edgecolor="white", linewidth=0.5)

    # Annotations on bars
    for bar_obj, (_, row) in zip(bars, df.iterrows()):
        score = row["score"]
        phase = row.get("phase", "")
        signal = row.get("signal", "")
        annotation = f" {score:.0f}"
        if phase:
            annotation += f" | {phase}"
        if signal:
            annotation += f" | {signal}"

        text_x = score + 1
        text_color = "black"
        if score > 85:
            text_x = score - 2
            text_color = "white"

        ax.text(
            text_x,
            bar_obj.get_y() + bar_obj.get_height() / 2,
            annotation,
            va="center",
            ha="left" if score <= 85 else "right",
            fontsize=8,
            color=text_color,
            fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Score d'attractivit\u00e9 (0-100)")
    ax.set_title("Classement des communes \u2014 Score d\u2019attractivit\u00e9")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. fig_rate_adjusted_comparison
# ---------------------------------------------------------------------------


def fig_rate_adjusted_comparison(
    agg: pd.DataFrame,
    label_col: str = "commune",
    reference_rate: float = 0.035,
) -> Figure:
    """Prix nominal vs prix ajuste des taux par commune.

    Le prix ajuste est recalcule comme si le taux etait fixe a ``reference_rate``.
    Quand ajuste > nominal : les taux actuels rendent le bien "moins cher" en cout reel.
    Quand ajuste < nominal : les taux masquent le cout reel.

    L'ajustement utilise le ratio des mensualites a taux courant vs taux de reference
    pour re-ponderer le prix nominal.
    """
    setup_style()

    # We need a rate column; if not present, we cannot adjust
    rate_col = None
    for candidate in ("taux_interet", "rate", "taux"):
        if candidate in agg.columns:
            rate_col = candidate
            break

    groups = agg.groupby(label_col)
    n_groups = groups.ngroups
    palette = sns.color_palette("deep", n_colors=max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    n_months = 240  # 20 years

    def _annuity_factor(annual_rate: float) -> float:
        r = annual_rate / 12
        if abs(r) < 1e-9:
            return float(n_months)
        return (1 - (1 + r) ** (-n_months)) / r

    ref_factor = _annuity_factor(reference_rate)

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        color = palette[idx % len(palette)]
        price_col = "prix_m2_smooth" if "prix_m2_smooth" in sub.columns else "prix_m2_median"
        dates = sub["date_mutation"]

        # Nominal
        ax.plot(
            dates,
            sub[price_col],
            linewidth=1.8,
            color=color,
            label=f"{label} (nominal)",
            linestyle="-",
        )

        # Rate-adjusted
        if rate_col is not None:
            rates_annual = sub[rate_col] / 100 if sub[rate_col].median() > 1 else sub[rate_col]
            adjusted = sub[price_col].copy()
            for i, (dt_idx, row) in enumerate(sub.iterrows()):
                current_factor = _annuity_factor(float(rates_annual.loc[dt_idx]))
                if current_factor > 0:
                    adjusted.iloc[i] = row[price_col] * (ref_factor / current_factor)

            ax.plot(
                dates,
                adjusted,
                linewidth=1.5,
                color=color,
                label=f"{label} (ajust\u00e9 @{reference_rate * 100:.1f}%)",
                linestyle="--",
            )

            # Shading
            ax.fill_between(
                dates,
                sub[price_col],
                adjusted,
                where=(adjusted >= sub[price_col]),
                interpolate=True,
                alpha=0.1,
                color=_BUY_COLOR,
            )
            ax.fill_between(
                dates,
                sub[price_col],
                adjusted,
                where=(adjusted < sub[price_col]),
                interpolate=True,
                alpha=0.1,
                color=_SELL_COLOR,
            )

    # Legend for shading
    green_patch = mpatches.Patch(
        color=_BUY_COLOR, alpha=0.2, label="Taux favorables (moins cher en r\u00e9el)"
    )
    red_patch = mpatches.Patch(
        color=_SELL_COLOR, alpha=0.2, label="Taux d\u00e9favorables (plus cher en r\u00e9el)"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [green_patch, red_patch], loc="best", fontsize=8)

    ax.set_title("Prix ajust\u00e9 des taux \u2014 Co\u00fbt r\u00e9el vs nominal")
    ax.set_xlabel("Mois")
    ax.set_ylabel("\u20ac / m\u00b2")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_euro_formatter))
    _format_date_axis(ax, interval=_auto_interval(agg))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. fig_decision_dashboard
# ---------------------------------------------------------------------------


def fig_decision_dashboard(
    agg: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    affordability_df: pd.DataFrame,
    label_col: str = "commune",
) -> Figure:
    """Tableau de bord decisionnaire complet — grille 2x3.

    Combine les 6 graphiques precedents dans une figure unique.
    """
    setup_style()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Determine top-scored commune for single-commune charts
    top_commune = (
        scorecard_df.sort_values("score", ascending=False).iloc[0]["commune"]
        if len(scorecard_df) > 0
        else (agg[label_col].iloc[0] if label_col in agg.columns and len(agg) > 0 else "N/A")
    )

    today_str = datetime.now().strftime("%d/%m/%Y")

    # ------ [0,0] Market phase diagram ------
    ax00 = fig.add_subplot(gs[0, 0])
    _draw_phase_diagram_on_ax(ax00, agg, label_col)

    # ------ [0,1] Commune scorecard ------
    ax01 = fig.add_subplot(gs[0, 1])
    _draw_scorecard_on_ax(ax01, scorecard_df)

    # ------ [0,2] Drawdown ------
    ax02 = fig.add_subplot(gs[0, 2])
    _draw_drawdown_on_ax(ax02, agg, label_col)

    # ------ [1,0] Affordability ------
    ax10 = fig.add_subplot(gs[1, 0])
    _draw_affordability_on_ax(ax10, affordability_df, top_commune)

    # ------ [1,1] Rate-adjusted comparison ------
    ax11 = fig.add_subplot(gs[1, 1])
    _draw_rate_adjusted_on_ax(ax11, agg, label_col)

    # ------ [1,2] Price-volume divergence ------
    ax12 = fig.add_subplot(gs[1, 2])
    _draw_divergence_on_ax(ax12, agg, top_commune)

    fig.suptitle(
        f"Tableau de bord d\u00e9cisionnaire \u2014 {today_str}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# Dashboard sub-drawing helpers (draw onto existing axes)
# ---------------------------------------------------------------------------


def _draw_phase_diagram_on_ax(ax: Any, agg: pd.DataFrame, label_col: str) -> None:
    """Simplified market phase diagram drawn onto a given axes."""
    df = _compute_volume_zscore(agg, label_col)
    groups = df.groupby(label_col)
    palette = sns.color_palette("deep", n_colors=max(groups.ngroups, 1))

    # Quadrant dividers
    ax.axvline(0, color=_NEUTRAL_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.5, color=_NEUTRAL_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation").dropna(subset=["pct_chg_3m", "vol_zscore"])
        if sub.empty:
            continue
        color = palette[idx % len(palette)]
        # Latest point only
        latest = sub.iloc[-1]
        ax.scatter(
            latest["pct_chg_3m"],
            latest["vol_zscore"],
            s=100,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=label,
            zorder=4,
        )
        # Small trail
        trail = sub.iloc[-min(4, len(sub)) : -1]
        if not trail.empty:
            ax.scatter(
                trail["pct_chg_3m"], trail["vol_zscore"], s=30, color=color, alpha=0.3, zorder=3
            )

    ax.set_xlabel("Momentum 3m (%)", fontsize=8)
    ax.set_ylabel("Volume z-score", fontsize=8)
    ax.set_title("Phase de march\u00e9", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left")


def _draw_scorecard_on_ax(ax: Any, scorecard_df: pd.DataFrame) -> None:
    """Simplified scorecard bars on given axes."""
    df = scorecard_df.sort_values("score", ascending=True).copy()
    cmap = mcolors.LinearSegmentedColormap.from_list("sc", [_SELL_COLOR, "#ff7f0e", _BUY_COLOR])
    norm = mcolors.Normalize(vmin=0, vmax=100)
    colors = [cmap(norm(s)) for s in df["score"]]

    bars = ax.barh(df["commune"], df["score"], color=colors, edgecolor="white", linewidth=0.4)
    for bar_obj, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            row["score"] + 1,
            bar_obj.get_y() + bar_obj.get_height() / 2,
            f"{row['score']:.0f}",
            va="center",
            fontsize=7,
            fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_title("Scores d\u2019attractivit\u00e9", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)


def _draw_drawdown_on_ax(ax: Any, agg: pd.DataFrame, label_col: str) -> None:
    """Simplified drawdown chart on given axes."""
    groups = agg.groupby(label_col)
    palette = sns.color_palette("deep", n_colors=max(groups.ngroups, 1))

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        price_col = "prix_m2_smooth" if "prix_m2_smooth" in sub.columns else "prix_m2_median"
        dd = drawdown_from_peak(sub[price_col].dropna())
        color = palette[idx % len(palette)]
        dates = sub.loc[dd.index, "date_mutation"]
        ax.plot(dates, dd, linewidth=1.2, label=label, color=color)
        ax.fill_between(dates, 0, dd, alpha=0.06, color=color)

    ax.axhline(0, color=_NEUTRAL_COLOR, linewidth=0.5, alpha=0.4)
    ax.axhline(-5, color=_NEUTRAL_COLOR, linewidth=0.4, linestyle=":", alpha=0.3)
    ax.axhline(-15, color=_NEUTRAL_COLOR, linewidth=0.4, linestyle=":", alpha=0.3)
    ax.set_title("Distance au pic", fontsize=10, fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left")


def _draw_affordability_on_ax(ax: Any, affordability_df: pd.DataFrame, commune: str) -> None:
    """Simplified affordability chart on given axes."""
    sub = affordability_df[affordability_df["commune"] == commune].copy()
    if sub.empty:
        ax.text(
            0.5,
            0.5,
            f"Pas de donn\u00e9es\n{commune}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title(f"Pouvoir d\u2019achat \u2014 {commune}", fontsize=10, fontweight="bold")
        return

    sub = sub.sort_values("date")
    m2 = sub["m2_achetables"]
    dates = sub["date"]
    mean_m2 = m2.mean()

    ax.fill_between(
        dates, m2, mean_m2, where=(m2 >= mean_m2), interpolate=True, alpha=0.2, color=_BUY_COLOR
    )
    ax.fill_between(
        dates, m2, mean_m2, where=(m2 < mean_m2), interpolate=True, alpha=0.2, color=_SELL_COLOR
    )
    ax.plot(dates, m2, color="#1f77b4", linewidth=1.5)
    ax.axhline(mean_m2, color=_NEUTRAL_COLOR, linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_title(f"Pouvoir d\u2019achat \u2014 {commune}", fontsize=10, fontweight="bold")
    ax.set_ylabel("m\u00b2 achetables", fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_rate_adjusted_on_ax(
    ax: Any,
    agg: pd.DataFrame,
    label_col: str,
    reference_rate: float = 0.035,
) -> None:
    """Simplified rate-adjusted comparison on given axes."""
    groups = agg.groupby(label_col)
    palette = sns.color_palette("deep", n_colors=max(groups.ngroups, 1))
    n_months = 240

    def _annuity_factor(annual_rate: float) -> float:
        r = annual_rate / 12
        if abs(r) < 1e-9:
            return float(n_months)
        return (1 - (1 + r) ** (-n_months)) / r

    ref_factor = _annuity_factor(reference_rate)

    rate_col = None
    for candidate in ("taux_interet", "rate", "taux"):
        if candidate in agg.columns:
            rate_col = candidate
            break

    for idx, (label, sub) in enumerate(groups):
        sub = sub.sort_values("date_mutation")
        color = palette[idx % len(palette)]
        price_col = "prix_m2_smooth" if "prix_m2_smooth" in sub.columns else "prix_m2_median"
        dates = sub["date_mutation"]
        ax.plot(dates, sub[price_col], linewidth=1.2, color=color, label=label)

        if rate_col is not None:
            rates_annual = sub[rate_col] / 100 if sub[rate_col].median() > 1 else sub[rate_col]
            adjusted = sub[price_col].copy()
            for i, (dt_idx, row) in enumerate(sub.iterrows()):
                curr_f = _annuity_factor(float(rates_annual.loc[dt_idx]))
                if curr_f > 0:
                    adjusted.iloc[i] = row[price_col] * (ref_factor / curr_f)
            ax.plot(dates, adjusted, linewidth=1, color=color, linestyle="--", alpha=0.7)

    ax.set_title("Prix ajust\u00e9 des taux", fontsize=10, fontweight="bold")
    ax.set_ylabel("\u20ac / m\u00b2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best")


def _draw_divergence_on_ax(ax: Any, agg: pd.DataFrame, commune: str) -> None:
    """Simplified price-volume divergence bars on given axes."""
    sub = agg[agg["commune"] == commune].copy() if "commune" in agg.columns else agg.copy()
    sub = sub.sort_values("date_mutation").dropna(subset=["pct_chg_3m"])

    if sub.empty:
        ax.text(
            0.5,
            0.5,
            f"Pas de donn\u00e9es\n{commune}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title(f"Divergence \u2014 {commune}", fontsize=10, fontweight="bold")
        return

    price_mom = sub["pct_chg_3m"].rolling(3, min_periods=1, center=True).mean()
    vol_raw = sub["n_transactions"].pct_change(3) * 100
    vol_mom = vol_raw.rolling(3, min_periods=1, center=True).mean()

    def _normalize(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if s_clean.empty or s_clean.std() == 0:
            return s * 0
        return (s - s.mean()) / s.std()

    divergence = _normalize(vol_mom) - _normalize(price_mom)
    dates = sub["date_mutation"]
    colors_div = [_BUY_COLOR if v >= 0 else _SELL_COLOR for v in divergence]
    ax.bar(dates, divergence, width=20, color=colors_div, alpha=0.7, edgecolor="none")
    ax.axhline(0, linewidth=0.6, color=_NEUTRAL_COLOR, alpha=0.5)

    ax.set_title(f"Divergence prix/vol \u2014 {commune}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=8)
    ax.tick_params(labelsize=7)
