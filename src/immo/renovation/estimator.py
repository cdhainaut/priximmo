"""
Logique de calcul de l'estimateur de rénovation — Maisons en pierre (FR).

Fonctions principales :
- compute_estimate(cfg) : calcul complet du chiffrage
- print_breakdown(df)  : affichage détaillé
- format_totals(totals) : mise en forme des totaux
"""

from __future__ import annotations

from math import ceil
from typing import Any

import pandas as pd

from .models import ProjectConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_row(
    rows: list[dict[str, Any]],
    label: str,
    qty: float,
    unit_str: str,
    mat_u: float,
    mo_u: float,
    note: str = "",
) -> None:
    """Ajoute une ligne de chiffrage au tableau."""
    materials = qty * mat_u
    labor = qty * mo_u
    rows.append(
        {
            "Poste": label,
            "Base": f"{qty:.2f} {unit_str}",
            "Unités": unit_str,
            "Matériaux €/U": mat_u,
            "Main d'œuvre €/U": mo_u,
            "Matériaux €": round(materials, 2),
            "Main d'œuvre €": round(labor, 2),
            "Total €": round(materials + labor, 2),
            "Notes": note,
        }
    )


def infer_trade(label: str) -> str:
    """Déduit le corps de métier à partir du libellé du poste."""
    low = label.lower()
    if any(
        k in low
        for k in ["placo", "cloisons", "isolation plafonds", "combles", "peinture"]
    ):
        return "plaquiste"
    if any(k in low for k in ["sols", "carrelage", "chape"]):
        return "carreleur"
    if any(
        k in low
        for k in [
            "électricité",
            "electricité",
            "tableau",
            "gtl",
            "consuel",
            "vmc",
            "circuit électrique",
        ]
    ):
        return "electricien"
    if any(
        k in low
        for k in [
            "plomberie",
            "chauffe-eau",
            "wc",
            "salle de bain",
            "cuisine (réseaux)",
        ]
    ):
        return "plombier"
    if any(
        k in low
        for k in [
            "pac",
            "groupe extérieur",
            "liaisons frigorifiques",
            "pompe(s) de relevage",
        ]
    ):
        return "frigoriste"
    if any(k in low for k in ["porte intérieure"]):
        return "menuisier"
    if any(k in low for k in ["cuisine (meubles"]):
        return "cuisiniste"
    return "autre"


def _labor_unit_from_hourly(
    rate: float, productivity_per_hour: float, team_size: int = 1
) -> float:
    """Calcule le coût MO unitaire (€/unité) à partir du taux horaire et de la productivité."""
    if productivity_per_hour <= 0 or team_size <= 0:
        return 0.0
    hours_per_unit = 1.0 / (productivity_per_hour * team_size)
    return rate * hours_per_unit


# ---------------------------------------------------------------------------
# Calcul principal
# ---------------------------------------------------------------------------


def compute_estimate(
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calcule le chiffrage complet de la rénovation.

    Retourne un DataFrame détaillé par poste et un dict des totaux.
    """
    dims = cfg.dims
    unit = cfg.surfaces_costs
    H = cfg.hourly

    floor = dims.floor_area
    walls = dims.wall_area
    ceiling = dims.ceiling_area if cfg.traiter_plafonds else 0.0

    # -- Surfaces dérivées --------------------------------------------------

    # Cloisons
    if cfg.cloisons_surface_m2 is None:
        cloisons_surface = 0.35 * floor * (dims.hauteur_m / 2.5)
    else:
        cloisons_surface = cfg.cloisons_surface_m2

    # Sols
    sdb_total = cfg.sdb_surface_m2 + cfg.sdb2_surface_m2
    sols_secs = (
        cfg.sols_secs_surface_m2
        if cfg.sols_secs_surface_m2 is not None
        else max(floor - sdb_total - cfg.cuisine_surface_m2, 0.0)
    )
    sols_humides = max(sdb_total, 0.0)

    # -- Modèle horaire : surcharges MO ------------------------------------

    placo_mo_u = unit.placo_mo
    isol_plaf_mo_u = unit.isol_plaf_mo
    sols_secs_mo_u = unit.sols_secs_mo
    sols_humides_mo_u = unit.sols_humides_mo
    chape_mo_u = unit.chape_mo_per_m2
    point_labor_u = cfg.electric.point_labor
    roughin_labor_u = cfg.electric.roughin_per_m2_labor

    # On travaille sur des copies pour ne pas muter le modele appelant
    plumbing = cfg.plumbing.model_copy()
    heating = cfg.heating.model_copy()

    if H.use_hourly_model:
        placo_mo_u = _labor_unit_from_hourly(
            H.rate_plaquiste, H.cloisons_m2_per_hour, H.cloisons_team_size
        )
        isol_plaf_mo_u = _labor_unit_from_hourly(
            H.rate_plaquiste, H.rampants_m2_per_hour, H.rampants_team_size
        )
        sols_secs_mo_u = _labor_unit_from_hourly(
            H.rate_carreleur, H.sols_stratif_m2_per_hour, 1
        )
        sols_humides_mo_u = _labor_unit_from_hourly(
            H.rate_carreleur, H.sols_carrelage_m2_per_hour, 1
        )
        chape_mo_u = _labor_unit_from_hourly(H.rate_carreleur, H.chape_m2_per_hour, 1)
        point_labor_u = _labor_unit_from_hourly(
            H.rate_electricien, H.elec_points_per_hour, 1
        )
        roughin_labor_u = _labor_unit_from_hourly(
            H.rate_electricien, H.elec_roughin_m2_per_hour, 1
        )

        # Surcharges plomberie
        plumbing.reseaux_per_m2_labor = _labor_unit_from_hourly(
            H.rate_plombier, H.plb_reseaux_m2_per_hour, 1
        )
        plumbing.sdb_labor = H.rate_plombier * H.hours_per_sdb
        plumbing.wc_labor = H.rate_plombier * H.hours_per_wc
        plumbing.cuisine_plomberie_labor = H.rate_plombier * H.hours_cuisine_plomberie

        # Surcharges PAC
        heating.outdoor_unit_labor = H.rate_frigoriste * H.hours_pac_outdoor
        heating.split_labor = H.rate_frigoriste * H.hours_pac_split
        heating.ref_line_labor_per_m = H.rate_frigoriste / H.ref_line_m_per_hour
        heating.cond_pump_labor = H.rate_frigoriste * H.hours_cond_pump
        heating.pac_elec_circuit_labor = H.rate_electricien * H.hours_pac_elec_circuit

    rows: list[dict[str, Any]] = []

    # -- Isolation murs (optionnelle, off par defaut) -----------------------
    if cfg.include_wall_insulation:
        if cfg.wall_insulation_mode == "perspirant":
            _add_row(
                rows,
                "Isolation murs (perspirant)",
                walls,
                "m²",
                unit.isol_murs_mat_perspirant,
                unit.isol_murs_mo_perspirant,
                "Fibre bois + chaux/terre + membrane hygro",
            )
        else:
            _add_row(
                rows,
                "Isolation murs (standard)",
                walls,
                "m²",
                unit.isol_murs_mat_standard,
                unit.isol_murs_mo_standard,
                "Non recommandé pour murs pierre épais",
            )

    # -- Combles / plafonds -------------------------------------------------
    if ceiling > 0:
        _add_row(
            rows,
            "Isolation plafonds/combles",
            ceiling,
            "m²",
            unit.isol_plaf_mat,
            isol_plaf_mo_u,
            "R>=7 visé",
        )

    # -- Cloisons / Placo ---------------------------------------------------
    _add_row(
        rows,
        "Placo/cloisons (doublage & séparatives)",
        cloisons_surface,
        "m²",
        unit.placo_mat,
        placo_mo_u,
        "Hors isolation murs ext.",
    )

    # -- Peintures ----------------------------------------------------------
    _add_row(
        rows,
        "Peinture murs/plafonds",
        floor * 2.5,
        "m² équiv.",
        unit.peinture_mat,
        unit.peinture_mo,
        "~= 2.5 x surface",
    )

    # -- Sols ---------------------------------------------------------------
    _add_row(
        rows,
        "Sols pièces sèches (stratifié)",
        sols_secs,
        "m²",
        unit.sols_secs_mat,
        sols_secs_mo_u,
        "Séjour/chambres/circulations",
    )
    _add_row(
        rows,
        "Sols pièces humides (carrelage)",
        sols_humides,
        "m²",
        unit.sols_humides_mat,
        sols_humides_mo_u,
        "SDB/SDD/WC/cellier humide",
    )

    # -- Chape --------------------------------------------------------------
    chape_surface = floor
    _add_row(
        rows,
        "Chape de finition/ravoirage",
        chape_surface,
        "m²",
        unit.chape_mat_per_m2,
        chape_mo_u,
        "Sous carrelage/stratifié selon besoins",
    )

    # -- ELECTRICITE (rough-in complet) -------------------------------------
    e = cfg.electric
    _add_row(
        rows,
        "Electricité - rough-in (gaine/câbles/cheminements)",
        floor,
        "m²",
        e.roughin_per_m2_materials,
        roughin_labor_u,
        "Maison sans réseau: tout est à créer",
    )
    _add_row(
        rows,
        "Tableau électrique",
        1,
        "forfait",
        e.tableau_materials,
        e.tableau_labor,
        "Tableau + protections",
    )
    _add_row(
        rows,
        "GTL + coffret com",
        1,
        "forfait",
        e.gtl_materials,
        e.gtl_labor,
        "Goulotte technique logement",
    )
    _add_row(
        rows,
        "Consuel / attestations",
        1,
        "forfait",
        e.consuel_materials,
        e.consuel_labor,
        "Contrôles/mesures",
    )

    points = ceil(e.points_per_m2 * floor)
    _add_row(
        rows,
        "Électricité - points (prises/lumières/commandes)",
        points,
        "point",
        e.point_materials,
        point_labor_u,
        f"~{e.points_per_m2}/m² -> {points} pts",
    )

    # VMC
    if e.vmc_type == "simple_flux":
        _add_row(
            rows,
            "VMC simple flux",
            1,
            "forfait",
            e.vmc_sf_materials,
            e.vmc_sf_labor,
            "Pièces humides",
        )
    else:
        _add_row(
            rows,
            "VMC double flux",
            1,
            "forfait",
            e.vmc_df_materials,
            e.vmc_df_labor,
            "Insufflation/extraction",
        )

    # -- PLOMBERIE ----------------------------------------------------------
    pconf = plumbing
    _add_row(
        rows,
        "Plomberie - réseaux (PER + évacuations)",
        floor,
        "m²",
        pconf.reseaux_per_m2_materials,
        pconf.reseaux_per_m2_labor,
        "Maison sans réseau: distribution complète",
    )
    for i in range(pconf.nb_sdb):
        _add_row(
            rows,
            f"Salle de bain #{i + 1} (équipements+réseaux)",
            1,
            "forfait",
            pconf.sdb_materials,
            pconf.sdb_labor,
            "Douche/vasque+mitigeurs",
        )
    for i in range(pconf.nb_wc):
        _add_row(
            rows,
            f"WC #{i + 1}",
            1,
            "forfait",
            pconf.wc_materials,
            pconf.wc_labor,
            "Cuvette, réservoir, raccordements",
        )
    _add_row(
        rows,
        "Plomberie - cuisine (réseaux)",
        1,
        "forfait",
        pconf.cuisine_plomberie_materials,
        pconf.cuisine_plomberie_labor,
        "Évier, LL, LV, évacuations",
    )
    if pconf.include_chauffe_eau:
        if pconf.chauffe_eau_type == "thermodynamique":
            _add_row(
                rows,
                "Chauffe-eau thermodynamique",
                1,
                "forfait",
                pconf.ce_thermo_materials,
                pconf.ce_thermo_labor,
                "250-270L typ.",
            )
        else:
            _add_row(
                rows,
                "Chauffe-eau électrique",
                1,
                "forfait",
                pconf.ce_elec_materials,
                pconf.ce_elec_labor,
                "200-300L",
            )

    # -- CHAUFFAGE (PAC + options) ------------------------------------------
    heat = heating
    split_count = (
        max(1, ceil(floor / 45.0)) if heat.split_count is None else heat.split_count
    )
    if split_count > 0 and heat.include_outdoor_unit:
        _add_row(
            rows,
            "PAC air/air — groupe extérieur + accessoires",
            1,
            "forfait",
            heat.outdoor_unit_materials,
            heat.outdoor_unit_labor,
            "Support/plots, mise en service",
        )
        _add_row(
            rows,
            "PAC — circuit électrique dédié (disj. + câblage)",
            1,
            "forfait",
            heat.pac_elec_circuit_materials,
            heat.pac_elec_circuit_labor,
            "Alim dédiée depuis tableau",
        )
        _add_row(
            rows,
            "PAC — liaisons frigorifiques (cuivre + isolant)",
            heat.ref_lines_total_m,
            "m",
            heat.ref_line_mat_per_m,
            heat.ref_line_labor_per_m,
            "Longueur cumulée de toutes les liaisons",
        )
        if heat.cond_pump_count > 0:
            _add_row(
                rows,
                "PAC — pompe(s) de relevage condensats",
                heat.cond_pump_count,
                "unité",
                heat.cond_pump_materials,
                heat.cond_pump_labor,
                "Pour évacuation condensats des splits",
            )
    for i in range(split_count):
        _add_row(
            rows,
            f"PAC air/air split #{i + 1}",
            1,
            "forfait",
            heat.split_materials,
            heat.split_labor,
            "Unité intérieure (murale) et raccordements",
        )
    for i in range(heat.poele_count):
        _add_row(
            rows,
            f"Poêle à granulés #{i + 1}",
            1,
            "forfait",
            heat.poele_materials,
            heat.poele_labor,
            "Hors conduit si neuf",
        )

    # -- MENUISERIES INT. + CUISINE -----------------------------------------
    join = cfg.joinery
    for i in range(join.nb_portes):
        _add_row(
            rows,
            f"Porte intérieure #{i + 1}",
            1,
            "unité",
            join.porte_materials,
            join.porte_labor,
            "Bloc-porte + pose",
        )
    kuch = cfg.kitchen
    _add_row(
        rows,
        "Cuisine (meubles/plan/pose)",
        1,
        "forfait",
        kuch.materials,
        kuch.labor,
        "Kit entrée de gamme",
    )

    # -- Totaux -------------------------------------------------------------
    df = pd.DataFrame(rows)

    # Heures MO estimées par ligne via les taux horaires
    rate_map = {
        "plaquiste": cfg.hourly.rate_plaquiste,
        "carreleur": cfg.hourly.rate_carreleur,
        "electricien": cfg.hourly.rate_electricien,
        "plombier": cfg.hourly.rate_plombier,
        "frigoriste": cfg.hourly.rate_frigoriste,
        "menuisier": cfg.hourly.rate_plaquiste,  # fallback
        "cuisiniste": cfg.hourly.rate_plaquiste,  # fallback
        "autre": max(
            cfg.hourly.rate_plaquiste,
            cfg.hourly.rate_carreleur,
            cfg.hourly.rate_electricien,
            cfg.hourly.rate_plombier,
        ),
    }
    trades = df["Poste"].apply(infer_trade)
    rates = trades.map(rate_map).astype(float)
    hours = df["Main d'œuvre €"] / rates
    df["Heures MO estimées"] = hours.replace([float("inf"), float("-inf")], float("nan")).round(2)

    totals: dict[str, float] = {
        "Total matériaux": round(df["Matériaux €"].sum(), 2),
        "Total main d'œuvre": round(df["Main d'œuvre €"].sum(), 2),
        "Sous-total": round(df["Total €"].sum(), 2),
    }
    contingency = round(cfg.contingency_pct * totals["Sous-total"], 2)
    grand_total = round(totals["Sous-total"] + contingency, 2)
    totals[f"Aléas ({int(cfg.contingency_pct * 100)}%)"] = contingency
    totals["Total avec aléas"] = grand_total

    return df, totals


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------


def print_breakdown(df: pd.DataFrame) -> None:
    """Affiche le tableau de chiffrage complet."""
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 140
    ):
        print(df.to_string(index=False))


def format_totals(totals: dict[str, float]) -> str:
    """Met en forme les totaux pour affichage console."""
    lines = ["\n--- Totaux ---"]
    for k, v in totals.items():
        lines.append(f"{k}: {v:,.2f} €".replace(",", " "))
    return "\n".join(lines)
