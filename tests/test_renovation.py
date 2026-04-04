"""Tests for immo.renovation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from immo.renovation.models import Dimensions, ProjectConfig
from immo.renovation.estimator import compute_estimate


def test_compute_estimate_default():
    cfg = ProjectConfig(dims=Dimensions(longueur_m=10, largeur_m=8, hauteur_m=2.5))
    df, totals = compute_estimate(cfg)
    assert not df.empty
    assert "Total avec aléas" in totals
    assert totals["Total avec aléas"] > 0


def test_pydantic_validation():
    with pytest.raises(ValidationError):
        Dimensions(longueur_m=-5, largeur_m=8, hauteur_m=2.5)
