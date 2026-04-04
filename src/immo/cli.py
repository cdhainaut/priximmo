"""CLI entry-point for the immo real estate analysis toolkit.

Usage:  immo --help
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger

from immo.config import AppConfig, load_config

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="immo",
    help="Outil d'analyse du marche immobilier francais (DVF, taux, previsions).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ---------------------------------------------------------------------------
# Common option types
# ---------------------------------------------------------------------------

ConfigOpt = Annotated[
    Optional[Path],
    typer.Option("--config", "-c", help="Chemin vers le fichier de configuration YAML."),
]


def _load(config_path: Path | None) -> AppConfig:
    """Load config and configure loguru in one shot."""
    cfg = load_config(config_path)
    logger.info("Configuration chargee ({} communes).", len(cfg.communes))
    return cfg


# ---------------------------------------------------------------------------
# immo fetch
# ---------------------------------------------------------------------------

@app.command()
def fetch(
    config: ConfigOpt = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Repertoire de sortie (parquet)."),
    ] = None,
) -> None:
    """Telecharger les donnees DVF pour les communes configurees."""
    cfg = _load(config)

    if not cfg.communes:
        logger.error("Aucune commune configuree. Verifiez votre fichier de configuration.")
        raise typer.Exit(code=1)

    out_path = Path(output) if output else Path("data/dvf.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from immo.scrapers.dvf import cache_to_parquet, fetch_all_communes

    communes = {
        name: {"depart": c.department_code, "ninsee": c.insee_code}
        for name, c in cfg.communes.items()
    }
    filters = {
        "type_local": cfg.filters.property_types,
        "max_price": cfg.filters.max_price,
        "surface_min": cfg.filters.surface_min,
        "surface_max": cfg.filters.surface_max,
    }

    df = fetch_all_communes(communes, cfg.dvf_url_root, filters=filters)
    cache_to_parquet(df, out_path)
    logger.success("DVF sauvegarde -> {} ({} lignes)", out_path, len(df))


# ---------------------------------------------------------------------------
# immo analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    config: ConfigOpt = None,
) -> None:
    """Lancer l'analyse de tendances et generer les signaux achat/vente."""
    cfg = _load(config)

    if not cfg.communes:
        logger.error("Aucune commune configuree. Verifiez votre fichier de configuration.")
        raise typer.Exit(code=1)

    from immo.analysis.trends import add_derived_metrics, add_overall_series, monthly_aggregate
    from immo.analysis.signals import composite_signal, signal_summary
    from immo.scrapers.dvf import fetch_all_communes, load_from_parquet

    # Load data (cached or fresh)
    cache_path = Path("data/dvf.parquet")
    if cache_path.exists():
        logger.info("Chargement depuis le cache {}", cache_path)
        df = load_from_parquet(cache_path)
    else:
        logger.info("Pas de cache, telechargement en cours...")
        communes = {
            name: {"depart": c.department_code, "ninsee": c.insee_code}
            for name, c in cfg.communes.items()
        }
        filters = {
            "type_local": cfg.filters.property_types,
            "max_price": cfg.filters.max_price,
            "surface_min": cfg.filters.surface_min,
            "surface_max": cfg.filters.surface_max,
        }
        df = fetch_all_communes(communes, cfg.dvf_url_root, filters=filters)

    label_col = cfg.grouping.group_by
    agg = monthly_aggregate(df, label_col=label_col)
    agg = add_derived_metrics(
        agg, label_col=label_col,
        smoothing_kind=cfg.smoothing.kind,
        window=cfg.smoothing.window_months,
    )
    if cfg.grouping.include_overall:
        agg = add_overall_series(agg, label_col=label_col)
        agg = add_derived_metrics(
            agg, label_col=label_col,
            smoothing_kind=cfg.smoothing.kind,
            window=cfg.smoothing.window_months,
        )

    # Export metrics CSV
    cfg.output.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(cfg.output.metrics_csv, index=False)
    logger.success("Metriques CSV -> {}", cfg.output.metrics_csv)

    # Generate signals
    signals = composite_signal(agg)
    summary = signal_summary(signals)
    logger.info("Signaux generes pour {} communes", len(summary))
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# immo report
# ---------------------------------------------------------------------------

@app.command()
def report(
    config: ConfigOpt = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Chemin du rapport PDF genere."),
    ] = None,
) -> None:
    """Generer un rapport PDF avec tous les graphiques."""
    import datetime as dt

    from immo.viz.reports import ReportConfig, generate_full_report
    from immo.analysis.signals import composite_signal

    cfg = _load(config)

    report_path = Path(output) if output else cfg.output.report_pdf
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Load aggregated data
    import pandas as pd
    metrics_path = cfg.output.metrics_csv
    if not metrics_path.exists():
        logger.error("Fichier de metriques introuvable: {}. Lancez 'immo analyze' d'abord.", metrics_path)
        raise typer.Exit(code=1)

    agg = pd.read_csv(metrics_path, parse_dates=["date_mutation"])
    label_col = cfg.grouping.group_by
    signals = composite_signal(agg)

    labels_txt = ", ".join(sorted(agg[label_col].unique()))
    rcfg = ReportConfig(
        title="Marche immobilier - Rapport DVF",
        subtitle=f"{label_col.capitalize()} : {labels_txt}",
        author="immo CLI",
        date=dt.date.today().strftime("%d/%m/%Y"),
    )

    generate_full_report(agg, signals, label_col, rcfg, report_path)
    logger.success("Rapport genere -> {}", report_path)


# ---------------------------------------------------------------------------
# immo rates
# ---------------------------------------------------------------------------

@app.command()
def rates(
    config: ConfigOpt = None,
    rate: Annotated[
        Optional[list[float]],
        typer.Option("--rate", "-r", help="Taux d'interet a tester (repetable)."),
    ] = None,
) -> None:
    """Analyser l'impact des taux d'interet sur la capacite d'emprunt."""
    cfg = _load(config)

    from immo.analysis.rates import borrowing_capacity, rate_sensitivity

    test_rates = rate if rate else cfg.interest_rates.manual_rates
    if not test_rates:
        test_rates = [0.02, 0.025, 0.03, 0.035, 0.04]
        logger.info("Aucun taux specifie, utilisation des defauts : {}", test_rates)

    duration = cfg.interest_rates.loan_duration_years
    insurance = cfg.interest_rates.insurance_rate
    debt_ratio = cfg.interest_rates.debt_ratio

    # Sensitivity table
    sensitivity = rate_sensitivity(250_000, test_rates, duration)
    print(sensitivity.to_string(index=False))

    # Example capacity for 3000 EUR/month salary
    example_salary = 3_000.0
    logger.info("\nCapacite d'emprunt pour {} EUR/mois net :", example_salary)
    for r in test_rates:
        cap = borrowing_capacity(example_salary, r, duration, debt_ratio, insurance)
        logger.info("  Taux {:.2%} -> {:.0f} EUR", r, cap)


# ---------------------------------------------------------------------------
# immo forecast
# ---------------------------------------------------------------------------

@app.command()
def forecast(
    config: ConfigOpt = None,
    horizon: Annotated[
        Optional[int],
        typer.Option("--horizon", "-H", help="Horizon de prevision en mois."),
    ] = None,
) -> None:
    """Lancer la prevision des prix immobiliers."""
    cfg = _load(config)

    h = horizon if horizon is not None else cfg.forecast.horizon_months
    model_name = cfg.forecast.model

    if not cfg.forecast.enabled:
        logger.warning("Previsions desactivees dans la configuration.")
        raise typer.Exit(code=0)

    import pandas as pd
    from immo.analysis.forecasting import forecast_ensemble, forecast_prophet, forecast_linear, prepare_prophet_data, backtest

    metrics_path = cfg.output.metrics_csv
    if not metrics_path.exists():
        logger.error("Fichier de metriques introuvable: {}. Lancez 'immo analyze' d'abord.", metrics_path)
        raise typer.Exit(code=1)

    agg = pd.read_csv(metrics_path, parse_dates=["date_mutation"])
    label_col = cfg.grouping.group_by

    for commune in agg[label_col].unique():
        logger.info("Prevision pour {} (modele={}, horizon={} mois)", commune, model_name, h)
        if model_name == "prophet":
            prophet_df = prepare_prophet_data(agg, commune)
            fc = forecast_prophet(prophet_df, horizon_months=h)
        elif model_name == "linear":
            sub = agg[agg[label_col] == commune].sort_values("date_mutation")
            series = sub.set_index("date_mutation")["prix_m2_median"].dropna()
            fc = forecast_linear(series, horizon_months=h)
        else:
            fc = forecast_ensemble(agg, commune, horizon_months=h)
        bt = backtest(agg, commune, test_months=min(h, 12))
        logger.info("  {} : MAE={:.0f}, MAPE={:.1f}%", commune, bt["mae"], bt["mape"])
        print(fc.tail(h).to_string(index=False))


# ---------------------------------------------------------------------------
# immo renovation
# ---------------------------------------------------------------------------

@app.command()
def renovation(
    config: ConfigOpt = None,
    surface: Annotated[
        float,
        typer.Option("--surface", "-s", help="Surface du bien a renover (m2)."),
    ] = 80.0,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Chemin du fichier de sortie (CSV)."),
    ] = None,
) -> None:
    """Estimer les couts de renovation pour un bien."""
    from immo.renovation import (
        Dimensions, ProjectConfig, compute_estimate,
        print_breakdown, format_totals, generate_all_charts,
    )

    out_path = Path(output) if output else Path("outputs/renovation_estimate.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Estimation renovation : surface={:.1f} m2", surface)

    project = ProjectConfig(
        dims=Dimensions(
            longueur_m=10.0,
            largeur_m=surface / 10.0,
            hauteur_m=2.5,
            surface_habitable_m2=surface,
        ),
    )
    df, totals = compute_estimate(project)
    print_breakdown(df)
    print(format_totals(totals))

    df.to_csv(out_path, index=False)
    logger.success("CSV -> {}", out_path)

    charts_dir = Path("charts")
    generate_all_charts(df, totals, charts_dir)
    logger.success("Charts -> {}", charts_dir)


# ---------------------------------------------------------------------------
# immo dashboard
# ---------------------------------------------------------------------------

@app.command()
def dashboard(
    config: ConfigOpt = None,
) -> None:
    """(Placeholder) Lancer le tableau de bord interactif."""
    _load(config)

    logger.info("Lancement du dashboard interactif...")
    logger.warning(
        "Le dashboard n'est pas encore implemente. "
        "Prevoyez une integration Streamlit ou Panel."
    )
    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# Main guard (allows `python -m immo.cli`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
