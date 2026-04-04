"""
Fonctions de visualisation pour l'estimateur de rénovation — Maisons en pierre.

Tous les graphiques utilisent des labels en français.
Chaque fonction plot_* sauvegarde en PNG (150 dpi) et SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

plt.style.use("seaborn-v0_8")


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Crée le répertoire parent si nécessaire."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _fmt_euros(x, pos=None) -> str:
    try:
        return f"{int(x):,} €".replace(",", " ")
    except Exception:
        return f"{x} €"


_euro_formatter = FuncFormatter(_fmt_euros)


def _annotate_bars(ax) -> None:
    for c in ax.containers:
        ax.bar_label(
            c,
            labels=[f"{int(v):,} €".replace(",", " ") for v in c.datavalues],
            label_type="edge",
            padding=3,
            fontsize=9,
        )


def _finalize(ax, ylabel_euros: bool = True) -> None:
    if ylabel_euros:
        ax.yaxis.set_major_formatter(_euro_formatter)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)


def _save_fig(fig, out_path: Path, *, bbox_inches: str | None = None) -> None:
    """Sauvegarde PNG + SVG."""
    _ensure_dir(out_path)
    fig.tight_layout()
    kw = {"bbox_inches": bbox_inches} if bbox_inches else {}
    fig.savefig(out_path, dpi=150, **kw)
    fig.savefig(out_path.with_suffix(".svg"), **kw)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fonctions d'agrégation
# ---------------------------------------------------------------------------


def infer_room(label: str) -> str:
    """Déduit la famille / pièce à partir du libellé du poste."""
    low = label.lower()
    if "salle de bain" in low or "sdb" in low:
        return "Salle de bain"
    if "wc" in low:
        return "WC"
    if "cuisine" in low:
        return "Cuisine"
    if "chambre" in low:
        return "Chambres/nuits"
    if "sols" in low and "séches" in low:
        return "Pièces sèches"
    if "sols" in low and "humides" in low:
        return "Pièces humides"
    if "vmc" in low:
        return "Ventilation"
    if (
        "électricité" in low
        or "electricité" in low
        or "tableau" in low
        or "gtl" in low
        or "consuel" in low
    ):
        return "Électricité"
    if "plomberie" in low or "chauffe-eau" in low:
        return "Plomberie"
    if "pac" in low or "poêle" in low or "poele" in low:
        return "Chauffage"
    if "peinture" in low:
        return "Peinture"
    if "isolation plafonds" in low or "combles" in low:
        return "Combles/Plafonds"
    if "isolation murs" in low:
        return "Murs (si activé)"
    if "cloisons" in low or "placo" in low:
        return "Cloisons/Placo"
    if "porte intérieure" in low:
        return "Menuiseries int."
    if "chape" in low:
        return "Chape"
    return "Divers"


def aggregate_materials_vs_labor(df: pd.DataFrame) -> dict[str, float]:
    """Matériaux vs main d'oeuvre (totaux)."""
    return {
        "Matériaux": float(df["Matériaux €"].sum()),
        "Main d'œuvre": float(df["Main d'œuvre €"].sum()),
    }


def aggregate_by_room(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les coûts par famille / pièce."""
    tmp = df.copy()
    tmp["Pièce"] = tmp["Poste"].apply(infer_room)
    return (
        tmp.groupby("Pièce")[["Matériaux €", "Main d'œuvre €", "Total €"]]
        .sum()
        .sort_values("Total €", ascending=False)
    )


def aggregate_by_post(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Top K postes par coût total."""
    grouped = (
        df.groupby("Poste")[["Matériaux €", "Main d'œuvre €", "Total €"]]
        .sum()
        .sort_values("Total €", ascending=False)
    )
    return grouped.head(top_k)


def aggregate_hours_by_room(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les heures MO estimées par famille / pièce."""
    tmp = df.copy()
    tmp["Pièce"] = tmp["Poste"].apply(infer_room)
    return (
        tmp.groupby("Pièce")[["Heures MO estimées"]]
        .sum()
        .sort_values("Heures MO estimées", ascending=False)
    )


def aggregate_hours_by_post(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Top K postes par heures MO estimées."""
    grouped = (
        df.groupby("Poste")[["Heures MO estimées"]]
        .sum()
        .sort_values("Heures MO estimées", ascending=False)
    )
    return grouped.head(top_k)


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------


def plot_materials_vs_labor(df: pd.DataFrame, out_path: Path) -> None:
    """Barres : matériaux vs main d'oeuvre."""
    agg = aggregate_materials_vs_labor(df)
    labels, values = list(agg.keys()), list(agg.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("Répartition globale: Matériaux vs Main d'œuvre")
    ax.set_ylabel("€")
    _finalize(ax, ylabel_euros=True)
    _annotate_bars(ax)
    _save_fig(fig, out_path)


def plot_costs_by_room(df: pd.DataFrame, out_path: Path) -> None:
    """Barres verticales : coûts par famille."""
    grouped = aggregate_by_room(df)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(grouped.index, grouped["Total €"])
    ax.set_title("Coûts par 'pièce' / famille")
    ax.set_ylabel("€")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _finalize(ax, ylabel_euros=True)
    _annotate_bars(ax)
    _save_fig(fig, out_path)


def plot_top_posts(df: pd.DataFrame, out_path: Path, top_k: int = 10) -> None:
    """Barres : top K postes de coût."""
    grouped = aggregate_by_post(df, top_k=top_k)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(grouped.index, grouped["Total €"])
    ax.set_title(f"Top {top_k} postes de coût")
    ax.set_ylabel("€")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _finalize(ax, ylabel_euros=True)
    _annotate_bars(ax)
    _save_fig(fig, out_path)


def plot_pie_by_room(df: pd.DataFrame, out_path: Path) -> None:
    """Camembert : répartition des coûts en pourcentage."""
    grouped = aggregate_by_room(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(grouped["Total €"], labels=grouped.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Camembert — répartition des coûts (%)")
    ax.axis("equal")
    _save_fig(fig, out_path)


def plot_pie_by_room_absolute(df: pd.DataFrame, out_path: Path) -> None:
    """Camembert : coûts en valeurs absolues (légende externe)."""
    grouped = aggregate_by_room(df)
    values = grouped["Total €"].values
    labels = [
        f"{name} — {int(val):,} €".replace(",", " ")
        for name, val in zip(grouped.index, values)
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts = ax.pie(values, startangle=90)
    ax.legend(
        wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
    )
    ax.set_title("Camembert — coûts (valeurs absolues)")
    ax.axis("equal")
    _save_fig(fig, out_path, bbox_inches="tight")


def plot_costs_barh_by_room(df: pd.DataFrame, out_path: Path) -> None:
    """Barres horizontales : coûts par famille."""
    grouped = aggregate_by_room(df)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(grouped.index[::-1], grouped["Total €"][::-1])
    ax.set_title("Coûts par famille (barres horizontales)")
    ax.set_xlabel("€")
    _finalize(ax, ylabel_euros=False)
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{int(width):,} €".replace(",", " "),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    _save_fig(fig, out_path)


def plot_waterfall_totals(
    df: pd.DataFrame, totals: dict[str, float], out_path: Path
) -> None:
    """Waterfall simplifié : construction du total."""
    mat = float(df["Matériaux €"].sum())
    mo = float(df["Main d'œuvre €"].sum())
    cont_key = [k for k in totals.keys() if k.startswith("Aléas")][0]
    cont = float(totals[cont_key])
    grand = float(totals["Total avec aléas"])
    steps = ["Matériaux", "Main d'œuvre", "Aléas", "Total"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(steps[0], mat)
    ax.bar(steps[1], mo, bottom=mat)
    ax.bar(steps[2], cont, bottom=mat + mo)
    ax.bar(steps[3], grand)
    ax.set_title("Waterfall simplifié — construction du total")
    ax.set_ylabel("€")
    _finalize(ax, ylabel_euros=True)
    _annotate_bars(ax)
    _save_fig(fig, out_path)


def plot_hours_by_room(df: pd.DataFrame, out_path: Path) -> None:
    """Barres horizontales : heures MO par famille."""
    grouped = aggregate_hours_by_room(df)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(grouped.index[::-1], grouped["Heures MO estimées"][::-1])
    ax.set_title("Heures de main d'œuvre par famille de postes")
    ax.set_xlabel("Heures")
    ax.grid(True, axis="x", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.0f} h",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    _save_fig(fig, out_path)


def plot_hours_top_posts(
    df: pd.DataFrame, out_path: Path, top_k: int = 10
) -> None:
    """Barres : top K postes par heures MO."""
    grouped = aggregate_hours_by_post(df, top_k=top_k)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(grouped.index, grouped["Heures MO estimées"])
    ax.set_title(f"Top {top_k} postes — Heures de main d'œuvre")
    ax.set_ylabel("Heures")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for c in ax.containers:
        ax.bar_label(
            c,
            labels=[f"{v:.0f} h" for v in c.datavalues],
            label_type="edge",
            padding=3,
            fontsize=9,
        )
    _save_fig(fig, out_path)


def plot_hours_pie_by_room(df: pd.DataFrame, out_path: Path) -> None:
    """Camembert : répartition des heures MO par famille."""
    grouped = aggregate_hours_by_room(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts = ax.pie(grouped["Heures MO estimées"], startangle=90)
    labels = [
        f"{name} — {val:.0f} h"
        for name, val in zip(grouped.index, grouped["Heures MO estimées"])
    ]
    ax.legend(
        wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
    )
    ax.set_title("Répartition des heures de main d'œuvre (par famille)")
    ax.axis("equal")
    _save_fig(fig, out_path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Fonction de commodité
# ---------------------------------------------------------------------------


def generate_all_charts(
    df: pd.DataFrame,
    totals: dict[str, float],
    output_dir: Path,
) -> None:
    """Génère tous les graphiques dans le répertoire donné."""
    output_dir = Path(output_dir)
    plot_materials_vs_labor(df, output_dir / "materials_vs_labor.png")
    plot_costs_by_room(df, output_dir / "costs_by_room.png")
    plot_top_posts(df, output_dir / "top_posts.png")
    plot_pie_by_room(df, output_dir / "pie_by_room.png")
    plot_pie_by_room_absolute(df, output_dir / "pie_by_room_absolute.png")
    plot_costs_barh_by_room(df, output_dir / "costs_by_room_barh.png")
    plot_waterfall_totals(df, totals, output_dir / "waterfall_totals.png")
    plot_hours_by_room(df, output_dir / "hours_by_room.png")
    plot_hours_top_posts(df, output_dir / "hours_top_posts.png")
    plot_hours_pie_by_room(df, output_dir / "hours_pie_by_room.png")
