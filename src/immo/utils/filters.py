"""Statistical filtering and smoothing utilities for time-series data.

Provides outlier removal (MAD-based), high-pass Butterworth filtering,
and various smoothing methods (rolling, EWM) with a unified dispatch
function.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------


def remove_outliers_mad(
    df: pd.DataFrame,
    col: str,
    by: str,
    threshold: float = 4.0,
) -> pd.DataFrame:
    """Remove outliers using the Median Absolute Deviation (MAD) method.

    For each group defined by *by*, compute the modified Z-score of *col*
    and drop rows whose absolute Z-score exceeds *threshold*.

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Numeric column to check for outliers.
    by:
        Column name used to group the data (e.g. ``"commune"``).
    threshold:
        Modified Z-score cutoff.  The default of 4.0 is fairly permissive;
        use 3.5 for a tighter filter.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with outlier rows removed.
    """

    def _modified_z(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = np.median(np.abs(x - med))
        if mad == 0:
            mad = 1e-9  # avoid division by zero for constant series
        return 0.6745 * (x - med) / mad

    z_scores = df.groupby(by)[col].transform(_modified_z).abs()
    n_before = len(df)
    result = df.loc[z_scores <= threshold].copy()
    n_removed = n_before - len(result)
    if n_removed > 0:
        logger.debug(
            "MAD outlier removal: dropped {} / {} rows (threshold={:.1f})",
            n_removed,
            n_before,
            threshold,
        )
    return result


# ---------------------------------------------------------------------------
# Butterworth filter
# ---------------------------------------------------------------------------


def butterworth_smooth(
    series: pd.Series,
    cutoff_freq: float,
    fs: float,
    order: int = 5,
) -> pd.Series:
    """Apply a high-pass Butterworth filter and subtract it from the signal.

    This effectively produces a *low-pass* smoothed version by removing
    high-frequency noise from the input series.

    Parameters
    ----------
    series:
        Input time series (must not contain NaN -- interpolate first).
    cutoff_freq:
        Cutoff frequency relative to the sampling rate.
    fs:
        Sampling frequency (e.g. 1.0 for daily, 12.0 for monthly with
        yearly units).
    order:
        Filter order.  Higher values give a sharper roll-off but may
        introduce ringing.

    Returns
    -------
    pd.Series
        Smoothed series (original minus high-frequency component).
    """
    if series.isna().any():
        logger.warning(
            "Butterworth filter received series with {} NaN values; "
            "forward-filling before filtering",
            series.isna().sum(),
        )
        series = series.ffill().bfill()

    if len(series) < 2 * order + 1:
        logger.warning(
            "Series too short ({}) for Butterworth order {}; returning as-is",
            len(series),
            order,
        )
        return series

    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    # Clamp to valid range (0, 1) exclusive
    normal_cutoff = max(1e-6, min(normal_cutoff, 1.0 - 1e-6))

    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    high_freq = filtfilt(b, a, series.values)
    smoothed = series.values - high_freq

    return pd.Series(smoothed, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# Rolling smoothing
# ---------------------------------------------------------------------------


def rolling_smooth(
    series: pd.Series,
    window: int,
    method: str = "median",
    center: bool = True,
) -> pd.Series:
    """Apply a rolling-window smoother.

    Parameters
    ----------
    series:
        Input series.
    window:
        Window size (number of observations).
    method:
        ``"median"`` or ``"mean"``.
    center:
        Whether the window is centred on each observation.

    Returns
    -------
    pd.Series
    """
    min_periods = max(1, window // 2)
    roller = series.rolling(window, center=center, min_periods=min_periods)
    if method == "mean":
        return roller.mean()
    if method == "median":
        return roller.median()
    raise ValueError(f"Unknown rolling method: {method!r}. Use 'mean' or 'median'.")


# ---------------------------------------------------------------------------
# EWM smoothing
# ---------------------------------------------------------------------------


def ewm_smooth(series: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted moving average.

    Parameters
    ----------
    series:
        Input series.
    span:
        Decay span in number of observations.

    Returns
    -------
    pd.Series
    """
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

SmoothKind = Literal[
    "rolling_mean",
    "rolling_median",
    "ewm",
    "butterworth",
]


def smooth_series(series: pd.Series, kind: str, **kwargs) -> pd.Series:
    """Dispatch smoothing to the appropriate method.

    Parameters
    ----------
    series:
        Input time series.
    kind:
        One of ``"rolling_mean"``, ``"rolling_median"``, ``"ewm"``,
        or ``"butterworth"``.
    **kwargs:
        Forwarded to the underlying function.

        - ``rolling_mean`` / ``rolling_median``: ``window`` (int),
          ``center`` (bool, default True).
        - ``ewm``: ``span`` (int).
        - ``butterworth``: ``cutoff_freq`` (float), ``fs`` (float),
          ``order`` (int, default 5).

    Returns
    -------
    pd.Series
    """
    if kind == "rolling_mean":
        return rolling_smooth(
            series,
            window=kwargs.get("window", 3),
            method="mean",
            center=kwargs.get("center", True),
        )

    if kind == "rolling_median":
        return rolling_smooth(
            series,
            window=kwargs.get("window", 3),
            method="median",
            center=kwargs.get("center", True),
        )

    if kind == "ewm":
        return ewm_smooth(series, span=kwargs.get("span", 3))

    if kind == "butterworth":
        return butterworth_smooth(
            series,
            cutoff_freq=kwargs["cutoff_freq"],
            fs=kwargs["fs"],
            order=kwargs.get("order", 5),
        )

    raise ValueError(
        f"Unknown smoothing kind: {kind!r}. "
        f"Choose from: rolling_mean, rolling_median, ewm, butterworth."
    )
