# Plan de nettoyage — immo

## 1. Bugs critiques (crashs garantis)

| # | Bug | Fichiers |
|---|-----|----------|
| C1 | CLI passe `{"department": ...}` mais scraper attend `meta["depart"]` / `meta["ninsee"]` | `cli.py` + `scrapers/dvf.py` |
| C2 | `trends.py` cree `pct_change_3m`, `volatility_6m` mais `viz/market.py` attend `pct_chg_3m`, `vol_6m` | `analysis/trends.py` + `viz/market.py` |
| C3 | Dicts couleurs dans `viz/signals.py` utilisent cles string `"STRONG_BUY"` mais recoivent `SignalType.STRONG_BUY` (enum) | `viz/signals.py` |
| C4 | `composite_signal` attend `rate_history["date_mutation"]` + `["rate"]` mais scraper produit `["date"]` + `["rate_pct"]` | `analysis/signals.py` |

## 2. Bugs importants

| # | Bug | Impact |
|---|-----|--------|
| I1 | `_labor_unit_from_hourly` divise par `team_size` -> cout plaquiste /2 | Devis faux |
| I2 | CLI `forecast` ignore `cfg.forecast.model`, appelle toujours `ensemble` | Config inutile |
| I3 | Protocol `SignalLike` declare `str` mais `Signal` a `SignalType` enum | Type mismatch |
| I4 | `reports.py` met enum brut dans DataFrame -> heatmap = zeros | Charts casses |
| I5 | Pas de garde si `cfg.communes` vide -> parquet vide, analyse silencieuse | UX confuse |
| I6 | `monthly_aggregate` lambdas fragiles pour noms de colonnes | Risque pandas |

## 3. Nice-to-have

- N1: Imports inutilises (interest_rates.py, insee.py, market.py)
- N2: geo.py importe logger inutilement
- N3: `SmoothingConfig.kind` n'inclut pas `"butterworth"`
- N4: `forecast_linear` utilise `dates.append()` deprecated pandas 2.0
- N5-N6: Variables inutilisees dans forecasting.py
- N7: backtest MAPE NaN si actual=0, utiliser nanmean
- N9: `rate_adjusted_signal` attend decimales mais scraper retourne pourcentages
- N10: `purchasing_power_index` saute `debt_ratio` param

## 4. Fichiers a supprimer

- `main.py`, `main2.py`, `capital_emprunt.py`, `renov_estimator_stone.py`
- `requirements.txt`, `config.yml` (racine)
- `renov_estimate_breakdown*.csv`, `*.xlsx`, `*.ods`, `*.zip`
- `charts/`, `outputs/`, `reports/` (regenerables)
- Tous les `__pycache__/`

## 5. Fichiers a creer

- `src/immo/__main__.py` — support `python -m immo`
- `src/immo/py.typed` — marqueur PEP 561
- `LICENSE` — MIT
- `tests/test_config.py` — smoke test minimum
- `tests/conftest.py` — fixtures pytest
- `docs/assets/` — images README

## 6. pyproject.toml

- Ajouter `pyarrow>=14.0` aux deps
- Ajouter `license`, `authors`, `readme`, `urls`
- Version dynamique depuis `__init__.py`
- mypy overrides pour prophet/loguru/tenacity
- ruff per-file-ignores pour tests

## 7. .gitignore

Ajouter : `*.csv` (racine), `*.pdf`, `.coverage`, `htmlcov/`, `*.parquet` (racine)

## 8. README

Structure : Hero + badges -> One-liner -> Features tableau -> Diagramme Mermaid
-> Quickstart 3 commandes -> CLI reference -> Config -> Exemples visuels 2x2
-> Structure projet -> License

## 9. CI/CD (GitHub Actions)

### Pipeline 1 : ci.yml (push/PR)
- lint (ruff check + format)
- typecheck (mypy strict)
- test (pytest + coverage)
- build (wheel/sdist)

### Pipeline 2 : data-refresh.yml (cron mensuel 1er du mois)
- Fetch DVF -> Analyze -> Report -> Forecast
- GitHub Release (pas de commit binaires)
- Commit uniquement `assets/prixm2.svg`
- Cache parquet entre runs

## 10. Ordre d'execution

1. Fix bugs critiques (C1-C4)
2. Fix bugs importants (I1-I6)
3. Cleanup repo (suppression, .gitignore, pyproject.toml, fichiers manquants)
4. README + assets
5. CI/CD workflows
6. Tests
7. Nice-to-have (N1-N10)
