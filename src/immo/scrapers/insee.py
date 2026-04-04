"""INSEE (Institut National de la Statistique et des Etudes Economiques) scrapers.

Provides access to population, unemployment, and construction-permit data
from the INSEE API (api.insee.fr).  Functions accept an optional API token;
when unavailable, structured fallback/mock data is returned so that
downstream code can be developed and tested independently.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# INSEE API v3 base URL
_INSEE_API_BASE = "https://api.insee.fr/donnees-locales/V0.1"
_INSEE_METADATA_BASE = "https://api.insee.fr/metadonnees/V1"

# Useful dataset identifiers on api.insee.fr
_DATASET_POPULATION = "GEO2024REE2024"  # Recensement -- population
_DATASET_UNEMPLOYMENT = "TAUX-DE-CHOMAGE-LOCAUX"
_DATASET_PERMITS = "SIT-CONJ"  # Situation conjoncturelle (includes permits)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _build_headers(token: str | None) -> dict[str, str]:
    """Build HTTP headers for the INSEE API.

    The API requires a Bearer token obtained via api.insee.fr application
    credentials (consumer key / secret -> OAuth2 token).
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _insee_get(
    url: str,
    token: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict:
    """Perform a GET request against the INSEE API and return parsed JSON."""
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url, headers=_build_headers(token), params=params)
        resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def fetch_population_by_commune(
    insee_codes: list[int],
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch the latest population figures for the given communes.

    Parameters
    ----------
    insee_codes:
        List of 5-digit INSEE commune codes (e.g. ``[29019, 56121]``).
    token:
        Optional INSEE API bearer token.

    Returns
    -------
    pd.DataFrame
        Columns: ``insee_code``, ``commune_name``, ``population``, ``year``.
    """
    logger.info("Fetching population data for {} communes", len(insee_codes))

    rows: list[dict] = []
    for code in insee_codes:
        code_str = str(code).zfill(5)
        try:
            url = f"{_INSEE_API_BASE}/donnees/geo-commune/{code_str}.all"
            data = _insee_get(url, token=token)

            # Navigate the nested JSON response structure
            cellules = data.get("Cellule", [])
            pop_value = None
            for cell in cellules:
                mesure = cell.get("Mesure", {})
                if "population" in mesure.get("@code", "").lower():
                    pop_value = float(cell.get("Valeur", 0))
                    break

            if pop_value is not None:
                rows.append(
                    {
                        "insee_code": code,
                        "commune_name": data.get("Zone", {}).get("Libelle", ""),
                        "population": int(pop_value),
                        "year": data.get("Annee", ""),
                    }
                )
                continue

        except Exception as exc:
            logger.debug("INSEE API call failed for {}: {}", code_str, exc)

        # Fallback: append placeholder
        rows.append(
            {
                "insee_code": code,
                "commune_name": "",
                "population": None,
                "year": None,
            }
        )
        logger.debug("Using placeholder for commune {}", code_str)

    df = pd.DataFrame(rows)

    # If no API data came through, fill with mock data for development
    if df["population"].isna().all():
        logger.warning(
            "No population data from INSEE API; returning mock data. "
            "Provide a valid token for real data."
        )
        df = _mock_population(insee_codes)

    return df


def _mock_population(insee_codes: list[int]) -> pd.DataFrame:
    """Return representative mock population data for development."""
    # A few well-known communes for convenience
    known: dict[int, tuple[str, int]] = {
        29019: ("Brest", 139_386),
        56121: ("Lorient", 57_149),
        76351: ("Le Havre", 170_352),
        75056: ("Paris", 2_133_111),
        69123: ("Lyon", 522_228),
        13055: ("Marseille", 873_076),
        29042: ("Crozon", 7_510),
        29238: ("Roscanvel", 894),
    }
    rows = []
    for code in insee_codes:
        name, pop = known.get(code, ("Unknown", 0))
        rows.append(
            {
                "insee_code": code,
                "commune_name": name,
                "population": pop,
                "year": 2021,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unemployment
# ---------------------------------------------------------------------------


def fetch_unemployment_by_department(
    departments: list[int],
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch quarterly unemployment rates by department.

    Parameters
    ----------
    departments:
        List of department numbers (e.g. ``[29, 56]``).
    token:
        Optional INSEE API bearer token.

    Returns
    -------
    pd.DataFrame
        Columns: ``department``, ``department_name``, ``quarter``, ``unemployment_rate_pct``.
    """
    logger.info("Fetching unemployment data for departments {}", departments)

    rows: list[dict] = []
    for dept in departments:
        dept_str = str(dept).zfill(2)
        try:
            url = f"{_INSEE_API_BASE}/donnees/geo-departement/{dept_str}.all"
            params = {"croisement": "TXCHO-TRIM"}
            data = _insee_get(url, token=token, params=params)

            for cell in data.get("Cellule", []):
                rows.append(
                    {
                        "department": dept,
                        "department_name": data.get("Zone", {}).get("Libelle", ""),
                        "quarter": cell.get("Periode", ""),
                        "unemployment_rate_pct": float(cell.get("Valeur", 0)),
                    }
                )
            if rows:
                continue

        except Exception as exc:
            logger.debug("INSEE unemployment API failed for dept {}: {}", dept_str, exc)

    if not rows:
        logger.warning(
            "No unemployment data from INSEE API; returning mock data. "
            "Provide a valid token for real data."
        )
        return _mock_unemployment(departments)

    return pd.DataFrame(rows)


def _mock_unemployment(departments: list[int]) -> pd.DataFrame:
    """Representative mock unemployment data."""
    dept_names: dict[int, str] = {
        29: "Finistere",
        56: "Morbihan",
        76: "Seine-Maritime",
        75: "Paris",
        69: "Rhone",
        13: "Bouches-du-Rhone",
    }
    quarters = ["2023-T1", "2023-T2", "2023-T3", "2023-T4", "2024-T1", "2024-T2"]
    rows = []
    import random

    rng = random.Random(42)
    for dept in departments:
        base = rng.uniform(6.0, 9.5)
        for q in quarters:
            rows.append(
                {
                    "department": dept,
                    "department_name": dept_names.get(dept, f"Dept {dept}"),
                    "quarter": q,
                    "unemployment_rate_pct": round(base + rng.gauss(0, 0.3), 1),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Construction permits
# ---------------------------------------------------------------------------


def fetch_construction_permits(
    departments: list[int],
    token: str | None = None,
) -> pd.DataFrame:
    """Fetch building-permit (autorisations de construire) data by department.

    Building permits are a leading indicator: a spike in permits today
    suggests increased housing supply 18-24 months later.

    Parameters
    ----------
    departments:
        Department numbers.
    token:
        Optional INSEE API bearer token.

    Returns
    -------
    pd.DataFrame
        Columns: ``department``, ``date`` (Period[M]), ``permits_count``,
        ``type`` (individual / collective).
    """
    logger.info("Fetching construction-permit data for departments {}", departments)

    rows: list[dict] = []
    for dept in departments:
        dept_str = str(dept).zfill(2)
        try:
            # SITADEL2 open data on data.gouv.fr (nationwide CSV, filtered locally)
            # In practice the INSEE API or SITADEL API may be more targeted
            data = _insee_get(
                f"{_INSEE_API_BASE}/donnees/geo-departement/{dept_str}.all",
                token=token,
                params={"croisement": "CONSTR-PERMIT"},
            )
            for cell in data.get("Cellule", []):
                rows.append(
                    {
                        "department": dept,
                        "date": pd.Period(cell.get("Periode", "")[:7], freq="M"),
                        "permits_count": int(float(cell.get("Valeur", 0))),
                        "type": cell.get("Modalite", {}).get("Libelle", "total"),
                    }
                )
        except Exception as exc:
            logger.debug("Construction permits API failed for dept {}: {}", dept_str, exc)

    if not rows:
        logger.warning(
            "No permit data from INSEE API; returning mock data. "
            "Provide a valid token for real data."
        )
        return _mock_permits(departments)

    return pd.DataFrame(rows)


def _mock_permits(departments: list[int]) -> pd.DataFrame:
    """Representative mock building-permit data."""
    import random

    rng = random.Random(42)
    rows = []
    for dept in departments:
        base = rng.randint(80, 400)
        for year in range(2020, 2026):
            for month in range(1, 13):
                for ptype in ("individuel", "collectif"):
                    count = max(0, int(base + rng.gauss(0, base * 0.15)))
                    rows.append(
                        {
                            "department": dept,
                            "date": pd.Period(f"{year}-{month:02d}", freq="M"),
                            "permits_count": count,
                            "type": ptype,
                        }
                    )
    return pd.DataFrame(rows)
