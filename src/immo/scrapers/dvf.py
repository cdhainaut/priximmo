"""DVF (Demandes de Valeurs Foncieres) scraper.

Fetches real estate transaction data from the French open-data portal
https://files.data.gouv.fr/geo-dvf/latest/csv/ and provides filtering,
caching, and concurrent download capabilities.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

DEFAULT_URL_ROOT = "https://files.data.gouv.fr/geo-dvf/latest/csv/"

_USECOLS = [
    "date_mutation",
    "valeur_fonciere",
    "surface_reelle_bati",
    "type_local",
    "nom_commune",
    "code_commune",
]

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def list_available_years(url_root: str = DEFAULT_URL_ROOT) -> list[str]:
    """Scrape year directory URLs from the DVF file listing page.

    Returns a sorted list of full URLs such as
    ``https://files.data.gouv.fr/geo-dvf/latest/csv/2024/``.
    """
    logger.info("Listing available DVF years from {}", url_root)
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url_root)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    years: list[str] = []
    for anchor in soup.find_all("a"):
        href = anchor.get("href", "")
        if href and ".." not in href and href.strip("/").isdigit():
            full = url_root.rstrip("/") + "/" + href.strip("/") + "/"
            years.append(full)

    years.sort()
    logger.info("Found {} year(s): {}", len(years), [y.rstrip("/").split("/")[-1] for y in years])
    return years


def build_commune_url(year_url: str, department: int, insee_code: int) -> str:
    """Build the CSV download URL for a single commune/year.

    Example result:
        ``https://…/csv/2023/communes/29/29019.csv``
    """
    return "/".join([
        year_url.rstrip("/"),
        "communes",
        str(department),
        f"{insee_code}.csv",
    ])


# ---------------------------------------------------------------------------
# Single commune fetch
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _download_csv(url: str) -> str:
    """Download raw CSV text with retry logic."""
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return resp.text


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Apply optional column-based filters to a DVF DataFrame.

    Recognised filter keys
    ----------------------
    - ``type_local``: str or list[str] -- keep only these property types
    - ``max_price``: float -- upper bound on ``valeur_fonciere``
    - ``surface_min``: float -- minimum ``surface_reelle_bati``
    - ``surface_max``: float -- maximum ``surface_reelle_bati``
    """
    if not filters:
        return df

    mask = pd.Series(True, index=df.index)

    if "type_local" in filters:
        types = filters["type_local"]
        if isinstance(types, str):
            types = [types]
        mask &= df["type_local"].isin(types)

    if "max_price" in filters:
        mask &= df["valeur_fonciere"].between(1, filters["max_price"], inclusive="both")

    if "surface_min" in filters:
        mask &= df["surface_reelle_bati"] >= filters["surface_min"]

    if "surface_max" in filters:
        mask &= df["surface_reelle_bati"] <= filters["surface_max"]

    return df.loc[mask]


def fetch_commune_year(
    year_url: str,
    department: int,
    insee_code: int,
    filters: dict | None = None,
) -> pd.DataFrame:
    """Fetch and parse the DVF CSV for one commune and one year.

    Parameters
    ----------
    year_url:
        Full URL to the year directory, e.g. ``https://…/csv/2023/``.
    department:
        French department number (e.g. 29).
    insee_code:
        INSEE commune code (e.g. 29019 for Brest).
    filters:
        Optional dict of filters (see :func:`_apply_filters`).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an added ``prix_m2`` column.
    """
    url = build_commune_url(year_url, department, insee_code)
    year_label = year_url.rstrip("/").split("/")[-1]
    logger.debug("Fetching {} (year {})", url, year_label)

    raw = _download_csv(url)
    df = pd.read_csv(
        io.StringIO(raw),
        usecols=lambda c: c in _USECOLS,
        dtype={"type_local": "category", "nom_commune": "category"},
    )

    # Parse dates
    df["date_mutation"] = pd.to_datetime(
        df["date_mutation"], errors="coerce", utc=True,
    ).dt.tz_localize(None)

    # Drop rows missing critical values
    df = df.dropna(subset=["date_mutation", "valeur_fonciere", "surface_reelle_bati"])

    # Apply user filters
    df = _apply_filters(df, filters)

    # Compute price per square metre
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)
    df = df.dropna(subset=["prix_m2"])

    logger.debug(
        "  -> {} rows after filtering (commune {}, year {})",
        len(df), insee_code, year_label,
    )
    return df


# ---------------------------------------------------------------------------
# Multi-commune / multi-year fetch
# ---------------------------------------------------------------------------

def fetch_all_communes(
    communes: dict[str, dict],
    url_root: str = DEFAULT_URL_ROOT,
    filters: dict | None = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Fetch DVF data for multiple communes across all available years.

    Parameters
    ----------
    communes:
        Mapping of commune names to metadata dicts.  Each dict must contain
        ``"depart"`` (department number) and ``"ninsee"`` (INSEE code).
        Example::

            {"Brest": {"depart": 29, "ninsee": 29019}}

    url_root:
        Root URL of the DVF CSV directory listing.
    filters:
        Passed to :func:`fetch_commune_year`.
    max_workers:
        Maximum number of concurrent download threads.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with a ``commune`` label column.
    """
    years = list_available_years(url_root)
    if not years:
        logger.warning("No years found at {}", url_root)
        return pd.DataFrame()

    tasks: list[tuple[str, str, int, int]] = []
    for year_url in years:
        for name, meta in communes.items():
            tasks.append((name, year_url, meta["depart"], meta["ninsee"]))

    logger.info(
        "Scheduling {} download tasks ({} communes x {} years) with {} workers",
        len(tasks), len(communes), len(years), max_workers,
    )

    frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_key = {}
        for name, year_url, dept, code in tasks:
            fut = pool.submit(fetch_commune_year, year_url, dept, code, filters)
            future_to_key[fut] = (name, year_url)

        for fut in as_completed(future_to_key):
            name, year_url = future_to_key[fut]
            year_label = year_url.rstrip("/").split("/")[-1]
            try:
                df = fut.result()
                df["commune"] = name
                frames.append(df)
                logger.debug("OK  {} / {}: {} rows", name, year_label, len(df))
            except Exception:
                logger.warning("SKIP {} / {} (download failed)", name, year_label)

    if not frames:
        logger.error("No data was successfully downloaded")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Downloaded {} total rows for {} communes",
        len(combined), len(communes),
    )
    return combined


# ---------------------------------------------------------------------------
# Parquet caching
# ---------------------------------------------------------------------------

def cache_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a Parquet file for local caching.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Cached {} rows to {}", len(df), path)


def load_from_parquet(path: Path) -> pd.DataFrame:
    """Load a cached DataFrame from Parquet.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded {} rows from cache {}", len(df), path)
    return df
