"""Price forecasting for French real estate markets.

Supports Facebook Prophet (when installed), linear regression as a baseline,
and an ensemble that averages both with confidence bands.  Includes a
walk-forward backtesting function.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Prophet is an optional heavy dependency -- guard the import.
_PROPHET_AVAILABLE = False
try:
    from prophet import Prophet  # type: ignore[import-untyped]

    _PROPHET_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_prophet_data(
    agg: pd.DataFrame,
    commune: str,
    label_col: str = "commune",
    price_col: str = "prix_m2_median",
) -> pd.DataFrame:
    """Format aggregated data for Prophet (``ds``, ``y`` columns).

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data.
    commune : str
        Commune name to filter on.
    label_col : str
        Column holding commune labels.
    price_col : str
        Column with the target variable (median price).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame ready for ``Prophet().fit()``.
    """
    sub = agg.loc[agg[label_col] == commune, ["date_mutation", price_col]].copy()
    sub = sub.dropna(subset=[price_col]).sort_values("date_mutation")
    sub = sub.rename(columns={"date_mutation": "ds", price_col: "y"})
    sub["ds"] = pd.to_datetime(sub["ds"])
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prophet forecast
# ---------------------------------------------------------------------------

def forecast_prophet(
    df: pd.DataFrame,
    horizon_months: int = 12,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
) -> pd.DataFrame:
    """Run a Facebook Prophet forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Prophet-ready data with ``ds`` and ``y`` columns (from
        :func:`prepare_prophet_data`).
    horizon_months : int
        Number of months to forecast.
    yearly_seasonality : bool
        Whether to model yearly seasonality.
    changepoint_prior_scale : float
        Flexibility of the trend.  Lower = smoother.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.

    Raises
    ------
    ImportError
        If ``prophet`` is not installed.
    """
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "Facebook Prophet is not installed. "
            "Install it with: pip install prophet"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=horizon_months, freq="MS")
        forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()


# ---------------------------------------------------------------------------
# Linear forecast
# ---------------------------------------------------------------------------

def forecast_linear(
    series: pd.Series,
    horizon_months: int = 12,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Simple linear regression extrapolation as a baseline forecast.

    Parameters
    ----------
    series : pd.Series
        Numeric series indexed by datetime (monthly).
    horizon_months : int
        Forecast horizon.
    confidence : float
        Confidence level for prediction interval.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
        Covers the full period (historical fit + forecast).
    """
    s = series.dropna()
    if len(s) < 3:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    # Numeric time axis (months since start)
    dates = pd.to_datetime(s.index)
    t = np.arange(len(s), dtype=float)

    slope, intercept, _r_value, _p_value, std_err = stats.linregress(t, s.values)

    # Residual standard error
    fitted = intercept + slope * t
    residuals = s.values - fitted
    n = len(s)
    se = np.sqrt(np.sum(residuals ** 2) / max(1, n - 2))

    # Forecast period
    t_future = np.arange(n, n + horizon_months, dtype=float)
    t_all = np.concatenate([t, t_future])

    yhat_all = intercept + slope * t_all

    # Prediction interval (widens with distance from mean of t)
    t_mean = t.mean()
    t_crit = stats.t.ppf((1 + confidence) / 2, df=max(1, n - 2))
    margin = t_crit * se * np.sqrt(
        1 + 1.0 / n + (t_all - t_mean) ** 2 / max(1e-9, np.sum((t - t_mean) ** 2))
    )

    # Build date index for full period
    last_date = dates[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon_months,
        freq="MS",
    )
    all_dates = dates.union(future_dates)

    return pd.DataFrame(
        {
            "ds": all_dates,
            "yhat": yhat_all,
            "yhat_lower": yhat_all - margin,
            "yhat_upper": yhat_all + margin,
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ensemble forecast
# ---------------------------------------------------------------------------

def forecast_ensemble(
    agg: pd.DataFrame,
    commune: str,
    horizon_months: int = 12,
    label_col: str = "commune",
    price_col: str = "prix_m2_median",
) -> pd.DataFrame:
    """Average of Prophet (if available) and linear forecasts with confidence bands.

    When Prophet is not installed, falls back to linear-only.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data.
    commune : str
        Target commune.
    horizon_months : int
        Forecast horizon (months).
    label_col : str
        Label column.
    price_col : str
        Price column.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``, ``model``.
    """
    # Prepare series for linear forecast
    sub = agg.loc[agg[label_col] == commune].copy()
    sub = sub.dropna(subset=[price_col]).sort_values("date_mutation")
    series = sub.set_index("date_mutation")[price_col]

    fc_linear = forecast_linear(series, horizon_months)
    fc_linear["model"] = "linear"

    if _PROPHET_AVAILABLE:
        prophet_df = prepare_prophet_data(agg, commune, label_col, price_col)
        if len(prophet_df) >= 12:
            try:
                fc_prophet = forecast_prophet(prophet_df, horizon_months)
                fc_prophet["model"] = "prophet"

                # Merge on ds and average
                merged = fc_linear.merge(
                    fc_prophet, on="ds", suffixes=("_lin", "_pro"), how="outer"
                )
                ensemble = pd.DataFrame({"ds": merged["ds"]})
                for col in ("yhat", "yhat_lower", "yhat_upper"):
                    lin = merged.get(f"{col}_lin", merged.get(col))
                    pro = merged.get(f"{col}_pro", merged.get(col))
                    ensemble[col] = (
                        pd.concat([lin, pro], axis=1).mean(axis=1)
                    )
                ensemble["model"] = "ensemble"
                return ensemble.sort_values("ds").reset_index(drop=True)
            except Exception:
                pass  # Fall back to linear only

    return fc_linear.sort_values("ds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest(
    agg: pd.DataFrame,
    commune: str,
    test_months: int = 12,
    label_col: str = "commune",
    price_col: str = "prix_m2_median",
) -> dict[str, Any]:
    """Walk-forward backtest: train on all-but-last-N, predict N, evaluate.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data.
    commune : str
        Target commune.
    test_months : int
        Number of months to hold out for testing.
    label_col : str
        Label column.
    price_col : str
        Price column.

    Returns
    -------
    dict
        Keys: ``mae``, ``mape``, ``rmse``, ``predictions`` (DataFrame with
        ``ds``, ``actual``, ``predicted``).
    """
    sub = agg.loc[agg[label_col] == commune].copy()
    sub = sub.dropna(subset=[price_col]).sort_values("date_mutation")

    if len(sub) < test_months + 6:
        return {
            "mae": np.nan,
            "mape": np.nan,
            "rmse": np.nan,
            "predictions": pd.DataFrame(),
        }

    train = sub.iloc[:-test_months]
    test = sub.iloc[-test_months:]

    actual = test[price_col].values
    test_dates = test["date_mutation"].values

    # Linear forecast on training set
    series_train = train.set_index("date_mutation")[price_col]
    fc = forecast_linear(series_train, horizon_months=test_months)

    # The forecast portion is the last `test_months` rows
    predicted = fc["yhat"].values[-test_months:]

    # If prophet is available, also compute ensemble
    if _PROPHET_AVAILABLE:
        prophet_df = prepare_prophet_data(train, commune, label_col, price_col)
        if len(prophet_df) >= 12:
            try:
                fc_p = forecast_prophet(prophet_df, horizon_months=test_months)
                pred_prophet = fc_p["yhat"].values[-test_months:]
                predicted = (predicted + pred_prophet) / 2.0
            except Exception:
                pass

    # Trim to matching length
    n = min(len(actual), len(predicted))
    actual = actual[:n]
    predicted = predicted[:n]
    dates_out = test_dates[:n]

    mae = float(np.mean(np.abs(actual - predicted)))
    mape = float(np.nanmean(np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual)))) * 100
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    predictions_df = pd.DataFrame(
        {
            "ds": dates_out,
            "actual": actual,
            "predicted": predicted,
        }
    )

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "predictions": predictions_df,
    }
