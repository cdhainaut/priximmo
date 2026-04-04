"""Shared fixtures for the immo test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_config_path() -> Path:
    return Path("config/default.yml")


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """Synthetic transaction DataFrame with ~100 rows across 2 communes."""
    rng = np.random.default_rng(42)
    communes = ["Brest", "Lorient"]
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")

    rows = []
    for commune in communes:
        for dt in dates:
            # 2-3 transactions per commune per month
            n = rng.integers(2, 4)
            for _ in range(n):
                prix_m2 = rng.normal(2500 if commune == "Brest" else 2200, 200)
                surface = rng.uniform(60, 80)
                rows.append(
                    {
                        "date_mutation": dt,
                        "commune": commune,
                        "prix_m2": round(prix_m2, 2),
                        "valeur_fonciere": round(prix_m2 * surface, 2),
                        "surface_reelle_bati": round(surface, 2),
                        "type_local": "Appartement",
                    }
                )

    return pd.DataFrame(rows)
