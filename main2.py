# -*- coding: utf-8 -*-
import os, sys, argparse, datetime as dt, json
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import seaborn as sns

# ===================== Defaults =====================
DEFAULT_CFG = {
    "communes": {
        # "Brest": {"depart": 29, "ninsee": 29019},
        # "Lorient": {"depart": 56, "ninsee": 56121},
    },
    "url_root": "https://files.data.gouv.fr/geo-dvf/latest/csv/",
    "filters": {
        "type_local": ["Appartement"],
        "valeur_fonciere_max": 400_000,
        "surface_min": 60,
        "surface_max": 80,
    },
    "group_by": "commune",  # "commune" | "groupe"
    "groups": {},  # {"Bretagne Ouest": ["Brest","Lorient"]}
    "include_overall": True,
    "min_n_per_month": None,
    "smoothing": {
        "kind": "rolling_median",  # "rolling_mean" | "rolling_median" | "ewm"
        "window_months": 3,
        "center": True,
        "ewm_span": 3,
    },
    "outputs": {
        "metrics_csv": "outputs/metrics_communes_monthly.csv",
        "report_pdf": "reports/rapport_dvf.pdf",
    },
}

# ===================== Style =====================
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".97"})


# ===================== Config helpers =====================
def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # deep-merge simple (sans libs externes)
        cfg = deep_merge(cfg, user_cfg)
    return cfg


def deep_merge(base, other):
    if not isinstance(other, dict):
        return other
    out = dict(base)
    for k, v in other.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_cli_overrides(cfg, args):
    if args.group_by:
        cfg["group_by"] = args.group_by
    if args.include_overall is not None:
        cfg["include_overall"] = args.include_overall
    if args.min_n is not None:
        cfg["min_n_per_month"] = args.min_n
    if args.smoothing:
        # ex: --smoothing '{"kind":"ewm","ewm_span":6}'
        cfg["smoothing"] = deep_merge(cfg["smoothing"], json.loads(args.smoothing))
    if args.groups:
        # ex: --groups '{"Bretagne Ouest":["Brest","Lorient"]}'
        cfg["groups"] = json.loads(args.groups)
    return cfg


# ===================== Data loading =====================
def list_year_urls(root: str) -> list[str]:
    html = requests.get(root, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    return [
        root + a.get("href") for a in soup.find_all("a") if ".." not in a.get("href")
    ]


def build_commune_url(year_url: str, depart: int, ninsee: int) -> str:
    return "/".join([year_url, "communes", str(depart), f"{ninsee}.csv"])


def read_commune_year(year_url: str, name: str, meta: dict, F: dict) -> pd.DataFrame:
    url = build_commune_url(year_url, meta["depart"], meta["ninsee"])
    usecols = [
        "date_mutation",
        "valeur_fonciere",
        "surface_reelle_bati",
        "type_local",
        "nom_commune",
    ]
    df = pd.read_csv(
        url,
        usecols=usecols,
        dtype={"type_local": "category", "nom_commune": "category"},
    )
    df["date_mutation"] = pd.to_datetime(
        df["date_mutation"], errors="coerce", utc=True
    ).dt.tz_localize(None)

    m = df["type_local"].isin(F["type_local"])
    m &= df["valeur_fonciere"].between(1, F["valeur_fonciere_max"], inclusive="both")
    m &= df["surface_reelle_bati"].between(F["surface_min"], F["surface_max"])
    df = df[m].dropna(
        subset=["date_mutation", "valeur_fonciere", "surface_reelle_bati"]
    )
    df["commune"] = name
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)
    return df.dropna(subset=["prix_m2"])


def load_all(cfg) -> pd.DataFrame:
    dfs = []
    for yurl in list_year_urls(cfg["url_root"]):
        for name, meta in cfg["communes"].items():
            try:
                dfs.append(read_commune_year(yurl, name, meta, cfg["filters"]))
            except Exception as e:
                print(f"Skip {name} {yurl} ({e})")
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ===================== Transform =====================
def add_group_labels(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if cfg["groups"]:
        lookup = {comm: grp for grp, lst in cfg["groups"].items() for comm in lst}
        df["groupe"] = df["commune"].map(lookup).fillna("Autres")
    else:
        df["groupe"] = df["commune"]
    return df


def remove_outliers_mad(
    df: pd.DataFrame, col="prix_m2", by="commune", thr=4.0
) -> pd.DataFrame:
    def _mad_z(x):
        med = x.median()
        mad = np.median(np.abs(x - med)) or 1e-9
        return 0.6745 * (x - med) / mad

    z = df.groupby(by)[col].transform(_mad_z).abs()
    return df.loc[z <= thr].copy()


def monthly_agg_generic(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.set_index("date_mutation").sort_index()
    g = df.groupby(label_col).resample("MS")
    agg = g.agg(
        prix_m2_median=("prix_m2", "median"),
        prix_m2_p25=("prix_m2", lambda s: s.quantile(0.25)),
        prix_m2_p75=("prix_m2", lambda s: s.quantile(0.75)),
        n_transactions=("prix_m2", "count"),
    ).reset_index()
    return agg


def smooth_series(s: pd.Series, cfg) -> pd.Series:
    kind = cfg["smoothing"]["kind"]
    if kind == "rolling_mean":
        return s.rolling(
            cfg["smoothing"]["window_months"],
            center=cfg["smoothing"]["center"],
            min_periods=max(1, cfg["smoothing"]["window_months"] // 2),
        ).mean()
    if kind == "rolling_median":
        return s.rolling(
            cfg["smoothing"]["window_months"],
            center=cfg["smoothing"]["center"],
            min_periods=max(1, cfg["smoothing"]["window_months"] // 2),
        ).median()
    if kind == "ewm":
        return s.ewm(span=cfg["smoothing"]["ewm_span"], adjust=False).mean()
    return s


def add_metrics(agg: pd.DataFrame, label_col: str, cfg) -> pd.DataFrame:
    agg = agg.sort_values([label_col, "date_mutation"]).copy()
    agg["prix_m2_smooth"] = agg.groupby(label_col)["prix_m2_median"].transform(
        lambda s: smooth_series(s, cfg)
    )
    agg["pct_chg_3m"] = agg.groupby(label_col)["prix_m2_median"].transform(
        lambda s: s.pct_change(3) * 100
    )
    agg["pct_chg_12m"] = agg.groupby(label_col)["prix_m2_median"].transform(
        lambda s: s.pct_change(12) * 100
    )
    agg["vol_6m"] = agg.groupby(label_col)["prix_m2_median"].transform(
        lambda s: s.rolling(6).std()
    )
    iqr = agg["prix_m2_p75"] - agg["prix_m2_p25"]
    agg["anomaly_score"] = (agg["prix_m2_median"] - agg["prix_m2_smooth"]) / (
        iqr.replace(0, np.nan)
    )
    if cfg["min_n_per_month"]:
        agg.loc[
            agg["n_transactions"] < cfg["min_n_per_month"],
            ["prix_m2_median", "prix_m2_smooth", "pct_chg_3m", "pct_chg_12m", "vol_6m"],
        ] = np.nan
    return agg


def add_overall_series(
    agg: pd.DataFrame, label_col: str, overall_name="Global (sélection)"
) -> pd.DataFrame:
    # pool sur toutes les lignes par date
    pooled = (
        agg.groupby("date_mutation")
        .agg(
            {
                "prix_m2_median": "median",
                "prix_m2_p25": "median",
                "prix_m2_p75": "median",
                "n_transactions": "sum",
            }
        )
        .reset_index()
    )
    pooled[label_col] = overall_name
    return pd.concat([agg, pooled], ignore_index=True)


# ===================== Plots =====================
def fig_cover(title="Marché immobilier – Rapport", subtitle="", footer=""):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 paysage
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.75, title, fontsize=28, weight="bold")
    ax.text(0.05, 0.63, subtitle, fontsize=16)
    ax.text(0.05, 0.15, footer, fontsize=10, alpha=0.7)
    return fig


def fig_trend_iqr(agg: pd.DataFrame, label_col: str):
    fig = plt.figure(figsize=(11, 5))
    ax = plt.gca()
    for label, sub in agg.groupby(label_col):
        ax.plot(sub["date_mutation"], sub["prix_m2_smooth"], label=f"{label} (lissé)")
        ax.fill_between(
            sub["date_mutation"],
            sub["prix_m2_p25"],
            sub["prix_m2_p75"],
            alpha=0.15,
            label=f"{label} IQR",
        )
    ax.set_title(
        f"Prix €/m² – médiane mensuelle (bande IQR) + lissage – par {label_col}"
    )
    ax.set_xlabel("Mois")
    ax.set_ylabel("€ / m²")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig


def fig_volume(agg: pd.DataFrame, label_col: str):
    pivot = agg.pivot_table(
        index="date_mutation", columns=label_col, values="n_transactions", aggfunc="sum"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 20  # jours
    bottoms = np.zeros(len(pivot))
    for label in pivot.columns:
        ax.bar(pivot.index, pivot[label], bottom=bottoms, width=width, label=label)
        bottoms += pivot[label].values
    ax.set_title(f"Volume de transactions par mois (empilé) – par {label_col}")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de transactions")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig


def fig_yoy(agg: pd.DataFrame, label_col: str):
    fig = plt.figure(figsize=(11, 4.5))
    ax = plt.gca()
    for label, sub in agg.groupby(label_col):
        ax.plot(sub["date_mutation"], sub["pct_chg_12m"], label=label)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(f"% évolution annuelle (YoY) – par {label_col}")
    ax.set_xlabel("Mois")
    ax.set_ylabel("% YoY")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig


def fig_volatility(agg: pd.DataFrame, label_col: str):
    fig = plt.figure(figsize=(11, 4.5))
    ax = plt.gca()
    for label, sub in agg.groupby(label_col):
        ax.plot(sub["date_mutation"], sub["vol_6m"], label=label)
    ax.set_title(f"Volatilité (écart-type roulante 6 mois) – par {label_col}")
    ax.set_xlabel("Mois")
    ax.set_ylabel("€/m²")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig


def fig_summary_table(agg: pd.DataFrame, label_col: str):
    last_idx = (
        agg.groupby(label_col)["date_mutation"].transform("max") == agg["date_mutation"]
    )
    last = agg[last_idx].copy().sort_values(label_col)
    cols = [
        label_col,
        "date_mutation",
        "prix_m2_median",
        "prix_m2_p25",
        "prix_m2_p75",
        "n_transactions",
        "pct_chg_3m",
        "pct_chg_12m",
        "vol_6m",
        "anomaly_score",
    ]
    last = last[cols].copy()
    last["date_mutation"] = last["date_mutation"].dt.strftime("%Y-%m")

    fmt = {
        "prix_m2_median": lambda x: f"{x:,.0f}".replace(",", " "),
        "prix_m2_p25": lambda x: f"{x:,.0f}".replace(",", " "),
        "prix_m2_p75": lambda x: f"{x:,.0f}".replace(",", " "),
        "n_transactions": lambda x: f"{x:.0f}",
        "pct_chg_3m": lambda x: f"{x:+.1f}%",
        "pct_chg_12m": lambda x: f"{x:+.1f}%",
        "vol_6m": lambda x: f"{x:,.0f}".replace(",", " "),
        "anomaly_score": lambda x: f"{x:+.2f}",
    }
    for c, f in fmt.items():
        last[c] = last[c].astype(float).map(f)

    fig = plt.figure(figsize=(11.69, 8.27))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title(
        f"Tableau récap – dernier mois dispo – par {label_col}",
        loc="left",
        pad=20,
        fontsize=14,
        weight="bold",
    )
    tbl = ax.table(
        cellText=last.values, colLabels=last.columns, loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    return fig


# ===================== Report =====================
def build_report(agg: pd.DataFrame, cfg, label_col: str):
    out_pdf = cfg["outputs"]["report_pdf"]
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    today = dt.date.today().strftime("%d/%m/%Y")
    labels_txt = ", ".join(sorted(agg[label_col].unique()))
    subtitle = f"{label_col.capitalize()} : {labels_txt}"
    sm = cfg["smoothing"]
    footer = f"Généré le {today} • Lissage: {sm['kind']} ({sm.get('window_months', sm.get('ewm_span'))} mois)"

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig_cover("Marché immobilier – Rapport DVF", subtitle, footer))
        plt.close()
        pdf.savefig(fig_trend_iqr(agg, label_col))
        plt.close()
        pdf.savefig(fig_volume(agg, label_col))
        plt.close()
        pdf.savefig(fig_yoy(agg, label_col))
        plt.close()
        pdf.savefig(fig_volatility(agg, label_col))
        plt.close()
        pdf.savefig(fig_summary_table(agg, label_col))
        plt.close()

    print(f"✅ Rapport exporté -> {out_pdf}")


# ===================== CLI =====================
def parse_args():
    p = argparse.ArgumentParser(description="DVF report (config YAML + CLI overrides)")
    p.add_argument("-c", "--config", help="Chemin du fichier YAML")
    p.add_argument(
        "--group-by",
        choices=["commune", "groupe"],
        help="Regrouper par commune ou par groupe",
    )
    p.add_argument(
        "--include-overall",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        help="Ajouter la série globale (true/false)",
    )
    p.add_argument("--min-n", type=int, help="Masquer mois avec < N transactions")
    p.add_argument(
        "--smoothing", help='JSON pour override (ex: \'{"kind":"ewm","ewm_span":6}\')'
    )
    p.add_argument(
        "--groups",
        help='JSON pour définir des cohortes (ex: \'{"Bretagne Ouest":["Brest","Lorient"]}\')',
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Load
    print("Chargement des données…")
    df = load_all(cfg)
    if df.empty:
        print("Aucune donnée chargée.")
        sys.exit(1)

    # Groups
    df = add_group_labels(df, cfg)
    by = cfg["group_by"]
    outlier_group = (
        "commune" if by == "commune" else "groupe"
    )  # robuste: supprimer outliers à l'échelle du label choisi
    print(f"Nettoyage outliers (MAD) par {outlier_group}…")
    df = remove_outliers_mad(df, col="prix_m2", by=outlier_group, thr=4.0)

    # Aggregate
    print(f"Agrégation mensuelle par {by}…")
    agg = monthly_agg_generic(df, by)
    agg = add_metrics(agg, by, cfg)

    # Overall
    if cfg["include_overall"]:
        agg = add_overall_series(agg, by)
        # recalculer les métriques lissées pour la série globale
        agg = add_metrics(agg, by, cfg)

    # Export CSV
    out_csv = cfg["outputs"]["metrics_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print(f"Export CSV -> {out_csv}")

    # Report
    print("Génération du PDF…")
    build_report(agg, cfg, by)


if __name__ == "__main__":
    main()
