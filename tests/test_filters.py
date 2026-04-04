"""Tests for immo.utils.filters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from immo.utils.filters import remove_outliers_mad, rolling_smooth, smooth_series


def test_remove_outliers_mad():
    rng = np.random.default_rng(0)
    values = rng.normal(100, 5, size=50).tolist()
    # Inject obvious outliers
    values[10] = 500.0
    values[30] = -200.0
    df = pd.DataFrame({"commune": "A", "prix_m2": values})
    result = remove_outliers_mad(df, col="prix_m2", by="commune", threshold=3.5)
    assert len(result) < len(df), "Outliers should have been removed"
    assert 500.0 not in result["prix_m2"].values
    assert -200.0 not in result["prix_m2"].values


def test_rolling_smooth_median():
    s = pd.Series(range(20), dtype=float)
    smoothed = rolling_smooth(s, window=5, method="median")
    assert len(smoothed) == len(s)
    assert smoothed.notna().sum() > 0


def test_smooth_series_dispatch():
    s = pd.Series(np.random.default_rng(1).normal(100, 5, 30))
    result = smooth_series(s, kind="rolling_median", window=3)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
