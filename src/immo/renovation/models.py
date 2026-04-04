"""
Modèles Pydantic v2 pour l'estimateur de rénovation — Maisons en pierre (FR).

Tous les montants sont en euros TTC.
Les surfaces sont en m², les longueurs en m.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Dimensions du bâtiment
# ---------------------------------------------------------------------------


class Dimensions(BaseModel):
    """Dimensions du bâtiment (longueur x largeur x hauteur)."""

    longueur_m: float = Field(..., gt=0, description="Longueur du bâtiment en mètres")
    largeur_m: float = Field(..., gt=0, description="Largeur du bâtiment en mètres")
    hauteur_m: float = Field(..., gt=0, description="Hauteur sous plafond en mètres")
    surface_habitable_m2: float | None = Field(
        default=None, gt=0, description="Surface habitable réelle (prioritaire si renseignée)"
    )

    @property
    def floor_area(self) -> float:
        """Surface au sol (m²)."""
        return (
            self.surface_habitable_m2
            if self.surface_habitable_m2
            else self.longueur_m * self.largeur_m
        )

    @property
    def perimeter(self) -> float:
        """Périmètre du bâtiment (m)."""
        return 2 * (self.longueur_m + self.largeur_m)

    @property
    def wall_area(self) -> float:
        """Surface totale des murs (m²)."""
        return self.perimeter * self.hauteur_m

    @property
    def ceiling_area(self) -> float:
        """Surface plafond (m²), identique à la surface au sol."""
        return self.floor_area


# ---------------------------------------------------------------------------
# Electricité
# ---------------------------------------------------------------------------


class ElectricConfig(BaseModel):
    """Configuration du lot électricité (rough-in complet)."""

    # Points & tableau
    points_per_m2: float = Field(default=0.6, gt=0)
    point_materials: float = Field(default=70.0, ge=0)
    point_labor: float = Field(default=120.0, ge=0)
    tableau_materials: float = Field(default=800.0, ge=0)
    tableau_labor: float = Field(default=400.0, ge=0)
    # Rough-in par m²
    roughin_per_m2_materials: float = Field(default=12.0, ge=0)
    roughin_per_m2_labor: float = Field(default=8.0, ge=0)
    # GTL + coffret com
    gtl_materials: float = Field(default=180.0, ge=0)
    gtl_labor: float = Field(default=120.0, ge=0)
    # Consuel
    consuel_materials: float = Field(default=0.0, ge=0)
    consuel_labor: float = Field(default=200.0, ge=0)
    # Ventilation
    vmc_type: Literal["simple_flux", "double_flux"] = "simple_flux"
    vmc_sf_materials: float = Field(default=250.0, ge=0)
    vmc_sf_labor: float = Field(default=350.0, ge=0)
    vmc_df_materials: float = Field(default=1400.0, ge=0)
    vmc_df_labor: float = Field(default=1600.0, ge=0)


# ---------------------------------------------------------------------------
# Plomberie
# ---------------------------------------------------------------------------


class PlumbingConfig(BaseModel):
    """Configuration du lot plomberie (réseaux + équipements sanitaires)."""

    nb_sdb: int = Field(default=1, ge=0)
    nb_wc: int = Field(default=1, ge=0)
    sdb_materials: float = Field(default=2500.0, ge=0)
    sdb_labor: float = Field(default=2500.0, ge=0)
    wc_materials: float = Field(default=400.0, ge=0)
    wc_labor: float = Field(default=300.0, ge=0)
    cuisine_plomberie_materials: float = Field(default=500.0, ge=0)
    cuisine_plomberie_labor: float = Field(default=600.0, ge=0)
    # Réseaux généraux (PER + évacuations) par m² habitable
    reseaux_per_m2_materials: float = Field(default=8.0, ge=0)
    reseaux_per_m2_labor: float = Field(default=7.0, ge=0)
    # Chauffe-eau
    include_chauffe_eau: bool = True
    chauffe_eau_type: Literal["thermodynamique", "elec_simple"] = "thermodynamique"
    ce_thermo_materials: float = Field(default=1500.0, ge=0)
    ce_thermo_labor: float = Field(default=500.0, ge=0)
    ce_elec_materials: float = Field(default=500.0, ge=0)
    ce_elec_labor: float = Field(default=200.0, ge=0)


# ---------------------------------------------------------------------------
# Chauffage
# ---------------------------------------------------------------------------


class HeatingConfig(BaseModel):
    """Configuration du lot chauffage (PAC air/air + poêle optionnel)."""

    split_count: int | None = Field(default=None, ge=0, description="Auto si None: 1 split ~ 45 m²")
    split_materials: float = Field(default=2000.0, ge=0)
    split_labor: float = Field(default=800.0, ge=0)
    # Groupe extérieur
    include_outdoor_unit: bool = True
    outdoor_unit_materials: float = Field(default=5000.0, ge=0)
    outdoor_unit_labor: float = Field(default=1200.0, ge=0)
    # Liaisons frigorifiques
    ref_lines_total_m: float = Field(default=15.0, ge=0)
    ref_line_mat_per_m: float = Field(default=25.0, ge=0)
    ref_line_labor_per_m: float = Field(default=20.0, ge=0)
    # Pompes relevage
    cond_pump_count: int = Field(default=0, ge=0)
    cond_pump_materials: float = Field(default=120.0, ge=0)
    cond_pump_labor: float = Field(default=80.0, ge=0)
    # Circuit élec dédié
    pac_elec_circuit_materials: float = Field(default=120.0, ge=0)
    pac_elec_circuit_labor: float = Field(default=80.0, ge=0)
    # Poêle
    poele_count: int = Field(default=0, ge=0)
    poele_materials: float = Field(default=2500.0, ge=0)
    poele_labor: float = Field(default=800.0, ge=0)


# ---------------------------------------------------------------------------
# Menuiseries intérieures
# ---------------------------------------------------------------------------


class InteriorJoineryConfig(BaseModel):
    """Configuration des menuiseries intérieures (portes)."""

    nb_portes: int = Field(default=7, ge=0)
    porte_materials: float = Field(default=150.0, ge=0)
    porte_labor: float = Field(default=100.0, ge=0)


# ---------------------------------------------------------------------------
# Cuisine
# ---------------------------------------------------------------------------


class KitchenConfig(BaseModel):
    """Configuration du lot cuisine (meubles + pose)."""

    materials: float = Field(default=3000.0, ge=0)
    labor: float = Field(default=800.0, ge=0)


# ---------------------------------------------------------------------------
# Coûts unitaires surfaces
# ---------------------------------------------------------------------------


class SurfacesUnitCosts(BaseModel):
    """Coûts unitaires (€/m²) pour les travaux de surface."""

    # Isolation murs (désactivée par défaut pour pierre)
    isol_murs_mat_standard: float = Field(default=18.0, ge=0)
    isol_murs_mo_standard: float = Field(default=22.0, ge=0)
    isol_murs_mat_perspirant: float = Field(default=35.0, ge=0)
    isol_murs_mo_perspirant: float = Field(default=28.0, ge=0)
    # Toujours pertinent
    isol_plaf_mat: float = Field(default=45.0, ge=0, description="Prix réel constaté")
    isol_plaf_mo: float = Field(
        default=65.0, ge=0, description="MO par défaut (écrasé si modèle horaire actif)"
    )
    placo_mat: float = Field(default=25.0, ge=0, description="Consommables inclus")
    placo_mo: float = Field(default=30.0, ge=0)
    peinture_mat: float = Field(default=2.5, ge=0)
    peinture_mo: float = Field(default=7.5, ge=0)
    sols_secs_mat: float = Field(default=20.0, ge=0)
    sols_secs_mo: float = Field(default=45.0, ge=0)
    sols_humides_mat: float = Field(default=30.0, ge=0)
    sols_humides_mo: float = Field(default=70.0, ge=0)
    # Chape (ravoirage)
    chape_mat_per_m2: float = Field(default=15.0, ge=0)
    chape_mo_per_m2: float = Field(default=25.0, ge=0)


# ---------------------------------------------------------------------------
# Modèle horaire
# ---------------------------------------------------------------------------


class HourlyModelConfig(BaseModel):
    """Taux horaires (€/h TTC) et productivités pour calculer la MO."""

    use_hourly_model: bool = True
    # Taux
    rate_plaquiste: float = Field(default=55.0, gt=0)
    rate_carreleur: float = Field(default=60.0, gt=0)
    rate_electricien: float = Field(default=65.0, gt=0)
    rate_plombier: float = Field(default=65.0, gt=0)
    rate_frigoriste: float = Field(default=75.0, gt=0)
    # Productivités
    rampants_m2_per_hour: float = Field(
        default=5.0, gt=0, description="Productivité équipe / heure"
    )
    rampants_team_size: int = Field(default=2, gt=0)
    cloisons_m2_per_hour: float = Field(
        default=2.0, gt=0, description="Par personne, joints inclus en moyenne"
    )
    cloisons_team_size: int = Field(default=2, gt=0)
    sols_stratif_m2_per_hour: float = Field(default=3.0, gt=0)
    sols_carrelage_m2_per_hour: float = Field(default=2.0, gt=0)
    chape_m2_per_hour: float = Field(default=5.0, gt=0)
    # Electricité
    elec_points_per_hour: float = Field(default=3.0, gt=0)
    elec_roughin_m2_per_hour: float = Field(default=6.0, gt=0)
    # Plomberie
    plb_reseaux_m2_per_hour: float = Field(default=5.0, gt=0)
    hours_per_sdb: float = Field(default=28.0, gt=0)
    hours_per_wc: float = Field(default=5.0, gt=0)
    hours_cuisine_plomberie: float = Field(default=8.0, gt=0)
    # PAC
    hours_pac_outdoor: float = Field(default=12.0, gt=0)
    hours_pac_split: float = Field(default=5.0, gt=0)
    ref_line_m_per_hour: float = Field(default=6.0, gt=0)
    hours_cond_pump: float = Field(default=1.5, gt=0)
    hours_pac_elec_circuit: float = Field(default=2.0, gt=0)


# ---------------------------------------------------------------------------
# Configuration projet (racine)
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    """Configuration complète d'un projet de rénovation maison en pierre."""

    model_config = {"arbitrary_types_allowed": False}

    dims: Dimensions
    surfaces_costs: SurfacesUnitCosts = Field(default_factory=SurfacesUnitCosts)
    electric: ElectricConfig = Field(default_factory=ElectricConfig)
    plumbing: PlumbingConfig = Field(default_factory=PlumbingConfig)
    heating: HeatingConfig = Field(default_factory=HeatingConfig)
    joinery: InteriorJoineryConfig = Field(default_factory=InteriorJoineryConfig)
    kitchen: KitchenConfig = Field(default_factory=KitchenConfig)
    hourly: HourlyModelConfig = Field(default_factory=HourlyModelConfig)
    # Programme
    sdb_surface_m2: float = Field(default=6.0, ge=0)
    sdb2_surface_m2: float = Field(default=0.0, ge=0)
    cuisine_surface_m2: float = Field(default=10.0, ge=0)
    sols_secs_surface_m2: float | None = Field(default=None, ge=0)
    traiter_plafonds: bool = True
    cloisons_surface_m2: float | None = Field(default=None, ge=0)
    # Spécifique pierre
    include_wall_insulation: bool = False
    wall_insulation_mode: Literal["perspirant", "standard"] = "perspirant"
    # Aléas
    contingency_pct: float = Field(default=0.10, ge=0, le=1.0)

    @model_validator(mode="after")
    def _validate_surfaces(self) -> ProjectConfig:
        """Vérifie la cohérence des surfaces du programme."""
        floor = self.dims.floor_area
        sdb_total = self.sdb_surface_m2 + self.sdb2_surface_m2
        if sdb_total + self.cuisine_surface_m2 > floor:
            raise ValueError(
                f"La somme SDB ({sdb_total} m²) + cuisine ({self.cuisine_surface_m2} m²) "
                f"dépasse la surface habitable ({floor} m²)"
            )
        return self
