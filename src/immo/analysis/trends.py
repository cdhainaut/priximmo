"""Trend analysis at multiple geographic scales.

Provides monthly aggregation, derived metrics (smoothing, percent changes,
volatility, anomaly scores), seasonal decomposition, and regional comparison
utilities for French real estate DVF data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.filters import smooth_series


# ---------------------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------------------

def monthly_aggregate(
    df: pd.DataFrame,
    label_col: str = "commune",
) -> pd.DataFrame:
    """Aggregate raw transaction data to monthly statistics per label.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data with columns ``date_mutation``, ``prix_m2``, and
        the column named by *label_col*.
    label_col : str
        Column used to group data (e.g. ``"commune"``, ``"groupe"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``date_mutation``, *label_col*, ``prix_m2_median``,
        ``prix_m2_p25``, ``prix_m2_p75``, ``n_transactions``.
    """
    tmp = df.copy()
    tmp["date_mutation"] = pd.to_datetime(tmp["date_mutation"])
    tmp = tmp.set_index("date_mutation").sort_index()

    grouped = tmp.groupby(label_col).resample("MS")

    agg = grouped.agg(
        prix_m2_median=pd.NamedAgg(column="prix_m2", aggfunc="median"),
        prix_m2_p25=pd.NamedAgg(column="prix_m2", aggfunc=lambda s: s.quantile(0.25)),
        prix_m2_p75=pd.NamedAgg(column="prix_m2", aggfunc=lambda s: s.quantile(0.75)),
        n_transactions=pd.NamedAgg(column="prix_m2", aggfunc="count"),
    )
    agg = agg.reset_index()

    return agg.sort_values([label_col, "date_mutation"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def add_derived_metrics(
    agg: pd.DataFrame,
    label_col: str = "commune",
    smoothing_kind: str = "rolling_median",
    window: int = 4,
) -> pd.DataFrame:
    """Enrich monthly aggregate with derived analytics columns.

    Added columns:
    - ``prix_m2_smooth`` -- smoothed median price via :func:`~immo.utils.filters.smooth`.
    - ``pct_chg_3m`` -- 3-month percent change of median price.
    - ``pct_chg_12m`` -- 12-month (year-over-year) percent change.
    - ``vol_6m`` -- 6-month rolling standard deviation.
    - ``anomaly_score`` -- deviation of raw median from smooth, normalised by IQR.

    Parameters
    ----------
    agg : pd.DataFrame
        Output of :func:`monthly_aggregate`.
    label_col : str
        Label column name.
    smoothing_kind : str
        Smoothing method forwarded to :func:`~immo.utils.filters.smooth`.
    window : int
        Smoothing window (months).

    Returns
    -------
    pd.DataFrame
        A copy of *agg* with additional columns.
    """
    out = agg.sort_values([label_col, "date_mutation"]).copy()

    def _per_group(sub: pd.DataFrame) -> pd.DataFrame:
        s = sub["prix_m2_median"]
        sub = sub.copy()
        sub["prix_m2_smooth"] = smooth_series(s, kind=smoothing_kind, window=window)
        sub["pct_chg_3m"] = s.pct_change(3) * 100
        sub["pct_chg_12m"] = s.pct_change(12) * 100
        sub["vol_6m"] = s.rolling(6, min_periods=3).std()

        iqr = sub["prix_m2_p75"] - sub["prix_m2_p25"]
        sub["anomaly_score"] = (s - sub["prix_m2_smooth"]) / iqr.replace(0, np.nan)
        return sub

    parts = []
    for _label, sub in out.groupby(label_col, sort=False):
        parts.append(_per_group(sub))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Overall / global series
# ---------------------------------------------------------------------------

def add_overall_series(
    agg: pd.DataFrame,
    label_col: str = "commune",
    overall_name: str = "Global",
) -> pd.DataFrame:
    """Pool all communes into a single ``"Global"`` series appended to *agg*.

    The global series uses the **median** of per-commune medians for price
    columns, and the **sum** of transaction counts.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data.
    label_col : str
        Label column.
    overall_name : str
        Name for the pooled series.

    Returns
    -------
    pd.DataFrame
        Concatenation of original *agg* and the global series.
    """
    pooled = (
        agg.groupby("date_mutation")
        .agg(
            prix_m2_median=("prix_m2_median", "median"),
            prix_m2_p25=("prix_m2_p25", "median"),
            prix_m2_p75=("prix_m2_p75", "median"),
            n_transactions=("n_transactions", "sum"),
        )
        .reset_index()
    )
    pooled[label_col] = overall_name
    return pd.concat([agg, pooled], ignore_index=True)


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------

def decompose_trend(
    series: pd.Series,
    period: int = 12,
) -> dict[str, pd.Series]:
    """Decompose a time series into trend, seasonal, and residual components.

    Uses a simple moving-average decomposition (additive model) that does not
    require ``statsmodels``.

    1. **Trend** -- centred moving average of length *period*.
    2. **Seasonal** -- average deviation from trend for each calendar position,
       tiled to match the series length.
    3. **Residual** -- ``series - trend - seasonal``.

    Parameters
    ----------
    series : pd.Series
        Numeric time series, ideally with a regular frequency (e.g. monthly).
    period : int
        Expected seasonal period (default 12 for monthly data).

    Returns
    -------
    dict
        ``{"trend": Series, "seasonal": Series, "residual": Series}``
    """
    s = series.copy().astype(float)

    # 1. Trend: centred moving average
    trend = s.rolling(period, center=True, min_periods=period // 2).mean()

    # 2. De-trended
    detrended = s - trend

    # 3. Seasonal pattern: average over each position in the cycle
    positions = np.arange(len(s)) % period
    seasonal_means = pd.Series(detrended.values).groupby(positions).transform("mean")
    seasonal = pd.Series(seasonal_means.values, index=s.index, name="seasonal")

    # 4. Residual
    residual = s - trend - seasonal

    return {
        "trend": trend.rename("trend"),
        "seasonal": seasonal,
        "residual": residual.rename("residual"),
    }


# ---------------------------------------------------------------------------
# Regional comparison
# ---------------------------------------------------------------------------

def compare_regions(
    agg: pd.DataFrame,
    regions: dict[str, list[str]],
    label_col: str = "commune",
) -> pd.DataFrame:
    """Aggregate and compare user-defined regions.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data with *label_col* identifying communes.
    regions : dict[str, list[str]]
        Mapping of region name to a list of commune names present in *agg*.
    label_col : str
        Column that holds commune names in *agg*.

    Returns
    -------
    pd.DataFrame
        Same schema as *agg* but with *label_col* set to region names.
    """
    parts: list[pd.DataFrame] = []
    for region_name, communes in regions.items():
        sub = agg.loc[agg[label_col].isin(communes)]
        if sub.empty:
            continue
        region_agg = (
            sub.groupby("date_mutation")
            .agg(
                prix_m2_median=("prix_m2_median", "median"),
                prix_m2_p25=("prix_m2_p25", "median"),
                prix_m2_p75=("prix_m2_p75", "median"),
                n_transactions=("n_transactions", "sum"),
            )
            .reset_index()
        )
        region_agg[label_col] = region_name
        parts.append(region_agg)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Year-over-year acceleration
# ---------------------------------------------------------------------------

def yoy_acceleration(series: pd.Series) -> pd.Series:
    """Compute the second derivative of year-over-year change.

    A positive value means the YoY growth rate is *accelerating* (prices
    rising faster or falling slower); a negative value means *decelerating*.

    Parameters
    ----------
    series : pd.Series
        Price series (e.g. monthly median prix/m2).

    Returns
    -------
    pd.Series
        YoY acceleration (percentage-point change per month of the YoY rate).
    """
    yoy = series.pct_change(12) * 100  # year-over-year percent change
    return yoy.diff().rename("yoy_acceleration")
