"""Tests for immo.analysis.signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from immo.analysis.signals import (
    Signal,
    composite_signal,
    signal_summary,
    z_score_signal,
)
from immo.analysis.trends import monthly_aggregate


def test_z_score_signal():
    rng = np.random.default_rng(7)
    s = pd.Series(rng.normal(2500, 100, 36))
    result = z_score_signal(s, window=12)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)


def test_composite_signal_returns_signals(sample_transactions):
    agg = monthly_aggregate(sample_transactions)
    signals = composite_signal(agg)
    assert isinstance(signals, list)
    assert all(isinstance(s, Signal) for s in signals)
    assert len(signals) > 0


def test_signal_summary(sample_transactions):
    agg = monthly_aggregate(sample_transactions)
    signals = composite_signal(agg)
    df = signal_summary(signals)
    assert isinstance(df, pd.DataFrame)
    assert "commune" in df.columns
    assert "signal" in df.columns
    assert "confidence" in df.columns
    assert len(df) == len(signals)
