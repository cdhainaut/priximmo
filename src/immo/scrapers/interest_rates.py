"""French mortgage interest-rate scrapers.

Provides functions to fetch historical mortgage rates from the Banque de France
and ECB, with fallback to manually-supplied data.
"""

from __future__ import annotations

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

# Banque de France Webstat SDMx API -- series key for housing-loan rates
# BSI1.M.FR.N.A.A20T.A.1.U2.2250.EUR.N  (new business, housing loans, France)
_BDF_SERIES_KEY = "BSI1.M.FR.N.A.A20T.A.1.U2.2250.EUR.N"
_BDF_API_BASE = "https://webstat.banque-de-france.fr/api/explore/v2.1/catalog/datasets"
_BDF_DATASET = "taux-dinterets-des-nouvelles-operations"

# ECB Data Portal -- MFI interest rates, housing loans, new business, France
_ECB_API_URL = (
    "https://data-api.ecb.europa.eu/service/data/MIR/"
    "M.FR.B.A2C.AM.R.A.2250.EUR.N?format=csvdata"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _get_json(url: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
    return resp.json()


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _get_csv_text(url: str) -> str:
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Banque de France
# ---------------------------------------------------------------------------

def fetch_banque_de_france_rates() -> pd.DataFrame:
    """Fetch historical French mortgage rates from Banque de France open data.

    Attempts the Webstat SDMX-REST API.  If unavailable, falls back to a
    curated set of manual rates covering 2015-2025 for reference.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (Period[M]), ``rate_pct`` (float), ``duration_years`` (int).
    """
    logger.info("Fetching Banque de France mortgage rates...")

    try:
        url = f"{_BDF_API_BASE}/{_BDF_DATASET}/records"
        params = {
            "limit": 100,
            "order_by": "-date",
        }
        data = _get_json(url, params=params)
        records = data.get("results", [])
        if not records:
            raise ValueError("Empty result set from BdF API")

        rows = []
        for rec in records:
            period = rec.get("date") or rec.get("period")
            value = rec.get("taux") or rec.get("value") or rec.get("obs_value")
            if period is not None and value is not None:
                rows.append({
                    "date": pd.Period(str(period)[:7], freq="M"),
                    "rate_pct": float(value),
                    "duration_years": 20,  # default duration for BdF housing-loan series
                })

        if rows:
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            logger.info("Retrieved {} rate observations from BdF API", len(df))
            return df
        raise ValueError("No parseable records")

    except Exception as exc:
        logger.warning(
            "BdF API unavailable ({}), using fallback manual rates", exc,
        )
        return _fallback_manual_rates()


def _fallback_manual_rates() -> pd.DataFrame:
    """Curated average French housing-loan rates (OAT-indexed, ~20y duration).

    Sources: Banque de France bulletins, Observatoire Credit Logement / CSA.
    """
    data = [
        ("2015-01", 2.20), ("2015-06", 1.99), ("2015-12", 2.14),
        ("2016-06", 1.56), ("2016-12", 1.34),
        ("2017-06", 1.56), ("2017-12", 1.51),
        ("2018-06", 1.49), ("2018-12", 1.44),
        ("2019-06", 1.29), ("2019-12", 1.12),
        ("2020-06", 1.24), ("2020-12", 1.17),
        ("2021-06", 1.06), ("2021-12", 1.06),
        ("2022-03", 1.12), ("2022-06", 1.49), ("2022-09", 1.88), ("2022-12", 2.34),
        ("2023-03", 2.84), ("2023-06", 3.28), ("2023-09", 3.77), ("2023-12", 4.20),
        ("2024-03", 3.90), ("2024-06", 3.65), ("2024-09", 3.54), ("2024-12", 3.35),
        ("2025-03", 3.20),
    ]
    df = pd.DataFrame(data, columns=["date", "rate_pct"])
    df["date"] = df["date"].apply(lambda d: pd.Period(d, freq="M"))
    df["duration_years"] = 20
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ECB key rates
# ---------------------------------------------------------------------------

def get_ecb_rates() -> pd.DataFrame:
    """Fetch ECB MFI interest-rate statistics for French housing loans.

    Falls back to a curated ECB main refinancing rate series if the
    detailed MIR dataset is unavailable.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (Period[M]), ``rate_pct`` (float), ``source`` (str).
    """
    logger.info("Fetching ECB interest rates...")

    try:
        import io
        csv_text = _get_csv_text(_ECB_API_URL)
        df = pd.read_csv(io.StringIO(csv_text))

        # The ECB CSV typically has TIME_PERIOD and OBS_VALUE columns
        period_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else df.columns[0]
        value_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else df.columns[-1]

        result = pd.DataFrame({
            "date": df[period_col].apply(lambda d: pd.Period(str(d)[:7], freq="M")),
            "rate_pct": pd.to_numeric(df[value_col], errors="coerce"),
            "source": "ECB_MIR",
        }).dropna(subset=["rate_pct"])

        logger.info("Retrieved {} ECB rate observations", len(result))
        return result.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.warning("ECB MIR API unavailable ({}), using fallback", exc)
        return _fallback_ecb_refi()


def _fallback_ecb_refi() -> pd.DataFrame:
    """ECB main refinancing operations rate (key policy rate) -- curated."""
    data = [
        ("2015-01", 0.05), ("2016-03", 0.00), ("2022-07", 0.50),
        ("2022-09", 1.25), ("2022-11", 2.00), ("2023-02", 2.50),
        ("2023-03", 3.00), ("2023-05", 3.25), ("2023-06", 3.50),
        ("2023-07", 3.75), ("2023-09", 4.00), ("2023-10", 4.50),
        ("2024-06", 4.25), ("2024-09", 3.65), ("2024-10", 3.40),
        ("2025-01", 2.90),
    ]
    df = pd.DataFrame(data, columns=["date", "rate_pct"])
    df["date"] = df["date"].apply(lambda d: pd.Period(d, freq="M"))
    df["source"] = "ECB_refi_fallback"
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined / manual rate builder
# ---------------------------------------------------------------------------

def build_rate_history(
    manual_rates: list[dict] | None = None,
) -> pd.DataFrame:
    """Build a combined mortgage-rate history from available sources.

    Parameters
    ----------
    manual_rates:
        Optional list of dicts with keys ``"date"`` (``"YYYY-MM"``) and
        ``"rate"`` (float).  These override any API-sourced values for the
        same month.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (Period[M]), ``rate_pct``, ``source``.
    """
    frames: list[pd.DataFrame] = []

    # Try BdF
    try:
        bdf = fetch_banque_de_france_rates()
        bdf["source"] = "banque_de_france"
        frames.append(bdf[["date", "rate_pct", "source"]])
    except Exception as exc:
        logger.warning("Could not fetch BdF rates: {}", exc)

    # Manual overrides
    if manual_rates:
        manual_df = pd.DataFrame(manual_rates)
        manual_df = manual_df.rename(columns={"rate": "rate_pct"})
        manual_df["date"] = manual_df["date"].apply(lambda d: pd.Period(d, freq="M"))
        manual_df["source"] = "manual"
        frames.append(manual_df[["date", "rate_pct", "source"]])

    if not frames:
        logger.error("No rate data available at all")
        return pd.DataFrame(columns=["date", "rate_pct", "source"])

    combined = pd.concat(frames, ignore_index=True)

    # Manual entries take precedence: keep last occurrence per date
    combined = (
        combined
        .sort_values(["date", "source"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    logger.info("Built rate history with {} entries", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Current rate helper
# ---------------------------------------------------------------------------

def current_market_rate(duration_years: int = 25) -> float:
    """Return the latest available mortgage rate estimate.

    Parameters
    ----------
    duration_years:
        Loan duration in years.  A rough adjustment is applied to the
        base ~20-year rate fetched from Banque de France data:
        shorter durations get a discount, longer ones a premium.

    Returns
    -------
    float
        Estimated annual interest rate in percent (e.g. 3.45).
    """
    history = build_rate_history()
    if history.empty:
        logger.warning("No rate history available, returning NaN")
        return float("nan")

    latest = history.iloc[-1]
    base_rate: float = float(latest["rate_pct"])

    # Duration adjustment heuristic (approx. French market spread):
    # Reference is ~20y.  Each year of difference adds/subtracts ~5 bps.
    reference_duration = 20
    adjustment = (duration_years - reference_duration) * 0.05
    adjusted = round(base_rate + adjustment, 2)

    logger.info(
        "Current rate estimate: {:.2f}% for {}y (base {:.2f}% at ~{}y + {:.2f}pp adjustment)",
        adjusted, duration_years, base_rate, reference_duration, adjustment,
    )
    return adjusted
