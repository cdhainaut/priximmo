"""Tests for DVF scraper helpers."""

from __future__ import annotations

from immo.scrapers.dvf import _parse_year_urls


def test_parse_year_urls_accepts_absolute_listing_paths():
    html = """
    <a href="../">..</a>
    <a href="/geo-dvf/latest/csv/2023/">2023/</a>
    <a href="/geo-dvf/latest/csv/2024/">2024/</a>
    <a href="/geo-dvf/latest/csv/2025/">2025/</a>
    """

    assert _parse_year_urls(html, "https://files.data.gouv.fr/geo-dvf/latest/csv/") == [
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2025/",
    ]


def test_parse_year_urls_accepts_relative_listing_paths():
    html = """
    <a href="../">..</a>
    <a href="2023/">2023/</a>
    <a href="2024/">2024/</a>
    """

    assert _parse_year_urls(html, "https://files.data.gouv.fr/geo-dvf/latest/csv/") == [
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/",
    ]
