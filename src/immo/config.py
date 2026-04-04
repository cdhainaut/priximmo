"""Configuration system for the immo real estate analysis toolkit.

Loads and validates YAML configuration using Pydantic v2 models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CommuneSource(BaseModel):
    """A single commune data source, identified by name, department, and INSEE code."""

    name: str = ""
    department_code: int = Field(..., alias="depart", description="Code du departement (ex: 29)")
    insee_code: int = Field(..., alias="ninsee", description="Code INSEE de la commune")

    model_config = {"populate_by_name": True}


class FiltersConfig(BaseModel):
    """Filtres appliques aux transactions DVF."""

    property_types: list[str] = Field(
        default=["Appartement"],
        alias="type_local",
        description="Types de locaux a inclure",
    )
    max_price: float = Field(
        500_000,
        alias="valeur_fonciere_max",
        description="Prix maximum (euros)",
    )
    surface_min: float = Field(60, description="Surface minimale (m2)")
    surface_max: float = Field(300, description="Surface maximale (m2)")

    model_config = {"populate_by_name": True}


class SmoothingConfig(BaseModel):
    """Parametres de lissage des series temporelles."""

    kind: Literal["rolling_median", "rolling_mean", "ewm", "butterworth"] = "rolling_median"
    window_months: int = Field(4, description="Fenetre en mois pour rolling_*")
    center: bool = True
    ewm_span: int = Field(4, description="Span pour le lissage exponentiel")


class GroupingConfig(BaseModel):
    """Configuration du regroupement geographique."""

    group_by: Literal["commune", "groupe", "departement", "region"] = "commune"
    groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Cohortes personnalisees (si group_by='groupe')",
    )
    include_overall: bool = Field(
        False,
        description="Ajouter une serie 'Global (selection)'",
    )
    min_n_per_month: int | None = Field(
        None,
        description="Nb minimum de transactions par mois (masque les mois maigres)",
    )


class OutputConfig(BaseModel):
    """Chemins de sortie pour les resultats."""

    metrics_csv: Path = Path("outputs/metrics_communes_monthly.csv")
    report_pdf: Path = Path("reports/rapport_dvf.pdf")
    charts_dir: Path = Path("charts")


class InterestRateConfig(BaseModel):
    """Parametres relatifs aux taux d'interet et a la capacite d'emprunt."""

    source: Literal["banque_de_france", "manual"] = "manual"
    manual_rates: list[float] = Field(
        default_factory=list,
        description="Taux manuels a tester (ex: [0.03, 0.035, 0.04])",
    )
    loan_duration_years: int = Field(25, description="Duree du pret en annees")
    insurance_rate: float = Field(0.003, description="Taux d'assurance annuel")
    debt_ratio: float = Field(0.34, description="Taux d'endettement maximum")


class ForecastConfig(BaseModel):
    """Parametres de prevision des prix."""

    enabled: bool = True
    horizon_months: int = Field(12, description="Horizon de prevision (mois)")
    model: Literal["prophet", "linear", "ensemble"] = "ensemble"


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Configuration racine de l'application immo."""

    communes: dict[str, CommuneSource] = Field(
        default_factory=dict,
        description="Communes a analyser, cle = nom d'affichage",
    )
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig, alias="outputs")
    interest_rates: InterestRateConfig = Field(default_factory=InterestRateConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    dvf_url_root: str = Field(
        "https://files.data.gouv.fr/geo-dvf/latest/csv/",
        alias="url_root",
        description="Racine des URLs DVF",
    )

    model_config = {"populate_by_name": True}

    # ------------------------------------------------------------------
    # Back-fill commune names from dict keys
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _fill_commune_names(self) -> AppConfig:
        for key, commune in self.communes.items():
            if not commune.name:
                commune.name = key
        return self

    # ------------------------------------------------------------------
    # Propagate top-level legacy fields into sub-models when loading
    # from the flat YAML format used in config.yml
    # ------------------------------------------------------------------
    @classmethod
    def from_raw(cls, data: dict) -> AppConfig:
        """Build an AppConfig from a raw YAML dict, handling legacy flat keys."""
        # Move flat grouping keys into a nested dict if needed
        grouping = data.get("grouping", {})
        for legacy_key in ("group_by", "groups", "include_overall", "min_n_per_month"):
            if legacy_key in data and legacy_key not in grouping:
                grouping[legacy_key] = data.pop(legacy_key)
        if grouping:
            data["grouping"] = grouping

        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_SEARCH_PATHS = [
    Path("config/default.yml"),
    Path("config.yml"),
]


def load_config(path: Path | None = None) -> AppConfig:
    """Load application configuration from a YAML file.

    Resolution order:
    1. Explicit *path* argument.
    2. ``config/default.yml`` in the current directory.
    3. ``config.yml`` in the current directory.
    4. Pure defaults (no YAML file needed).

    Returns
    -------
    AppConfig
        Validated configuration object.
    """
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Configuration introuvable : {resolved}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        return AppConfig.from_raw(raw)

    for candidate in _DEFAULT_SEARCH_PATHS:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            return AppConfig.from_raw(raw)

    # No file found -- return pure defaults
    return AppConfig()
