"""Tests for immo.analysis.trends."""

from __future__ import annotations

from immo.analysis.trends import add_derived_metrics, monthly_aggregate


def test_monthly_aggregate(sample_transactions):
    agg = monthly_aggregate(sample_transactions)
    assert "prix_m2_median" in agg.columns
    assert "prix_m2_p25" in agg.columns
    assert "prix_m2_p75" in agg.columns
    assert "n_transactions" in agg.columns
    assert len(agg) > 0
    # Should have entries for both communes
    assert set(agg["commune"].unique()) == {"Brest", "Lorient"}
    assert (agg["prix_m2_median"] > 0).all()


def test_add_derived_metrics(sample_transactions):
    agg = monthly_aggregate(sample_transactions)
    enriched = add_derived_metrics(agg)
    expected_cols = {"prix_m2_smooth", "pct_chg_3m", "pct_chg_12m", "vol_6m", "anomaly_score"}
    assert expected_cols.issubset(set(enriched.columns))
    assert len(enriched) == len(agg)
