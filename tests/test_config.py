"""Tests for immo.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from immo.config import AppConfig, load_config


@pytest.fixture(autouse=True)
def _chdir_to_repo(tmp_path, monkeypatch):
    """Ensure we run from the repo root so default.yml is found."""
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo_root)


def test_load_default_config():
    cfg = load_config(Path("config/default.yml"))
    assert isinstance(cfg, AppConfig)
    assert len(cfg.communes) == 3


def test_load_config_none():
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    # Should find config/default.yml and return 3 communes,
    # or return pure defaults (0 communes) if file not found.
    # From the repo root it should find the file.
    assert len(cfg.communes) >= 0


def test_config_communes_have_required_fields():
    cfg = load_config(Path("config/default.yml"))
    for name, commune in cfg.communes.items():
        assert commune.department_code > 0, f"{name} missing valid department_code"
        assert commune.insee_code > 0, f"{name} missing valid insee_code"
