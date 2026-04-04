"""Actionable decision metrics for French real estate markets.

Combines price data, interest-rate dynamics, volume analysis, and signal
detection into high-level decision tools: affordability timelines, market
phase classification, price-volume divergence, rate-adjusted pricing, and
per-commune scorecards with a composite buy-attractiveness score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rates import borrowing_capacity, monthly_payment
from .signals import composite_signal, SignalType


# ---------------------------------------------------------------------------
# Default rate proxy
# ---------------------------------------------------------------------------

_RATE_ANCHORS = {
    "2020-01-01": 0.015,
    "2024-01-01": 0.038,
    "2025-01-01": 0.032,
}


def _default_rate_series(dates: pd.DatetimeIndex) -> pd.Series:
    """Build a linearly-interpolated rate series from anchor points.

    Used when the caller does not supply actual mortgage-rate history.
    """
    anchor_idx = pd.DatetimeIndex(list(_RATE_ANCHORS.keys()))
    anchor_vals = list(_RATE_ANCHORS.values())
    anchor_series = pd.Series(anchor_vals, index=anchor_idx, dtype=float)

    # Combine anchors with requested dates, interpolate, then select
    combined = anchor_series.reindex(anchor_idx.union(dates)).sort_index()
    combined = combined.interpolate(method="time")
    # Forward/backward fill for dates outside the anchor range
    combined = combined.ffill().bfill()
    return combined.reindex(dates)


# ---------------------------------------------------------------------------
# 1. Affordability timeline
# ---------------------------------------------------------------------------

def affordability_timeline(
    agg: pd.DataFrame,
    commune: str,
    salary: float = 3000.0,
    rates_over_time: pd.Series | None = None,
    duration: int = 25,
    insurance: float = 0.003,
    debt_ratio: float = 0.34,
    label_col: str = "commune",
) -> pd.DataFrame:
    """For each month, compute how many m2 the given salary can buy.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data with at least *label_col*, ``date_mutation``,
        and ``prix_m2_median`` columns.
    commune : str
        Commune to analyse.
    salary : float
        Net monthly salary (euros).
    rates_over_time : pd.Series | None
        Annual interest rate indexed by date.  If *None*, a linear proxy
        from 1.5 % (2020) to 3.8 % (2024) to 3.2 % (2025) is used.
    duration : int
        Loan duration in years.
    insurance : float
        Annual insurance rate.
    debt_ratio : float
        Maximum fraction of salary allocated to debt service.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``prix_m2``, ``rate``, ``monthly_payment_per_m2``,
        ``purchasable_m2``, ``pct_change_vs_peak``.
    """
    sub = (
        agg.loc[agg[label_col] == commune]
        .sort_values("date_mutation")
        .copy()
    )
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "date", "prix_m2", "rate", "monthly_payment_per_m2",
                "purchasable_m2", "pct_change_vs_peak",
            ]
        )

    dates = pd.DatetimeIndex(sub["date_mutation"])
    prix = sub["prix_m2_median"].values.astype(float)

    if rates_over_time is not None:
        rates = rates_over_time.reindex(dates, method="ffill").bfill().values
    else:
        rates = _default_rate_series(dates).values

    capacities = np.array([
        borrowing_capacity(salary, float(r), duration, debt_ratio, insurance)
        for r in rates
    ])
    purchasable = np.where(prix > 0, capacities / prix, np.nan)

    # Monthly payment per m2 (what 1 m2 costs per month over the loan)
    mp_per_m2 = np.array([
        monthly_payment(float(p), float(r), duration, insurance)
        for p, r in zip(prix, rates)
    ])

    peak_m2 = np.maximum.accumulate(np.nan_to_num(purchasable, nan=0.0))
    pct_vs_peak = np.where(
        peak_m2 > 0,
        (purchasable - peak_m2) / peak_m2 * 100,
        np.nan,
    )

    return pd.DataFrame({
        "date": dates,
        "prix_m2": prix,
        "rate": rates,
        "monthly_payment_per_m2": np.round(mp_per_m2, 2),
        "purchasable_m2": np.round(purchasable, 1),
        "pct_change_vs_peak": np.round(pct_vs_peak, 2),
    })


# ---------------------------------------------------------------------------
# 2. Drawdown from peak
# ---------------------------------------------------------------------------

def drawdown_from_peak(series: pd.Series) -> pd.Series:
    """Compute drawdown: (current - running_max) / running_max * 100.

    Parameters
    ----------
    series : pd.Series
        Numeric time series (e.g. prix/m2).

    Returns
    -------
    pd.Series
        Negative percentages.  -15 means "15 % below the recent peak".
    """
    running_max = series.expanding().max()
    dd = (series - running_max) / running_max.replace(0, np.nan) * 100
    return dd.rename("drawdown_pct")


# ---------------------------------------------------------------------------
# 3. Market phase classification
# ---------------------------------------------------------------------------

def market_phase(prix_momentum_3m: float, volume_zscore: float) -> str:
    """Classify the market into one of four regimes.

    Quadrants
    ---------
    - momentum > 0  AND volume_z > 0.5  --> ``"Boom"``
      Distribution phase, risky to buy.
    - momentum > 0  AND volume_z <= 0.5 --> ``"Bull"``
      Trend continuation, hold.
    - momentum <= 0 AND volume_z > 0.5  --> ``"Capitulation"``
      Panic selling, evaluate for buy.
    - momentum <= 0 AND volume_z <= 0.5 --> ``"Bottom"``
      Accumulation phase, best buy zone.

    Parameters
    ----------
    prix_momentum_3m : float
        3-month price momentum (percent change or similar).
    volume_zscore : float
        Volume z-score relative to recent history.

    Returns
    -------
    str
        One of ``"Boom"``, ``"Bull"``, ``"Capitulation"``, ``"Bottom"``.
    """
    if prix_momentum_3m > 0 and volume_zscore > 0.5:
        return "Boom"
    if prix_momentum_3m > 0 and volume_zscore <= 0.5:
        return "Bull"
    if prix_momentum_3m <= 0 and volume_zscore > 0.5:
        return "Capitulation"
    return "Bottom"


# ---------------------------------------------------------------------------
# 4. Market phase history
# ---------------------------------------------------------------------------

def market_phase_history(
    agg: pd.DataFrame,
    label_col: str = "commune",
) -> pd.DataFrame:
    """Compute the market phase for each commune and date.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data with *label_col*, ``date_mutation``,
        ``prix_m2_median``, and ``n_transactions``.
    label_col : str
        Column identifying the commune or group.

    Returns
    -------
    pd.DataFrame
        Original rows augmented with ``phase``, ``momentum_3m``, and
        ``volume_zscore`` columns.
    """
    frames: list[pd.DataFrame] = []

    for _commune, sub in agg.sort_values("date_mutation").groupby(label_col):
        sub = sub.copy().sort_values("date_mutation")

        price = sub["prix_m2_median"]
        volume = sub["n_transactions"]

        # 3-month price momentum (percent change)
        momentum_3m = price.pct_change(periods=3) * 100

        # Volume z-score (6-month rolling)
        vol_mean = volume.rolling(6, min_periods=3).mean()
        vol_std = volume.rolling(6, min_periods=3).std()
        vol_z = (volume - vol_mean) / vol_std.replace(0, np.nan)

        sub["momentum_3m"] = momentum_3m.values
        sub["volume_zscore"] = vol_z.values
        sub["phase"] = [
            market_phase(m, v)
            for m, v in zip(
                momentum_3m.fillna(0).values,
                vol_z.fillna(0).values,
            )
        ]
        frames.append(sub)

    if not frames:
        return agg.assign(momentum_3m=np.nan, volume_zscore=np.nan, phase="")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Price-volume divergence
# ---------------------------------------------------------------------------

def price_volume_divergence(
    price_series: pd.Series,
    volume_series: pd.Series,
    window: int = 6,
) -> pd.Series:
    """Detect price-volume divergence as a reversal signal.

    When price momentum and volume momentum move in opposite directions,
    it signals underlying weakness or hidden strength.

    Parameters
    ----------
    price_series : pd.Series
        Monthly price series (e.g. prix_m2_median).
    volume_series : pd.Series
        Monthly transaction count.
    window : int
        Rolling window for momentum calculation.

    Returns
    -------
    pd.Series
        Divergence score.  Positive = price up but volume down (weak rally,
        potential sell).  Negative = price down but volume up (accumulation,
        potential buy).
    """
    price_mom = price_series.pct_change(periods=window).fillna(0)
    vol_mom = volume_series.pct_change(periods=window).fillna(0)

    # Normalise each to [-1, 1] range using tanh for stability
    price_norm = np.tanh(price_mom * 5)
    vol_norm = np.tanh(vol_mom * 5)

    divergence = price_norm - vol_norm
    return divergence.rename("pv_divergence")


# ---------------------------------------------------------------------------
# 6. Rate-adjusted price
# ---------------------------------------------------------------------------

def rate_adjusted_price(
    prix_m2_series: pd.Series,
    rate_at_time: pd.Series,
    reference_rate: float = 0.035,
    duration: int = 25,
    insurance: float = 0.003,
) -> pd.Series:
    """Convert prices to their equivalent at a fixed reference rate.

    Same monthly-payment logic: what price at *reference_rate* produces the
    same monthly cost as *prix_m2* at *rate_at_time*?  This strips out
    interest-rate effects and shows whether prices are truly up.

    Parameters
    ----------
    prix_m2_series : pd.Series
        Monthly median price per m2.
    rate_at_time : pd.Series
        Annual interest rate at each time step (aligned index).
    reference_rate : float
        Reference rate for normalisation (decimal).
    duration : int
        Loan duration in years.
    insurance : float
        Annual insurance rate.

    Returns
    -------
    pd.Series
        Price per m2 expressed in reference-rate-equivalent euros.
    """
    n = duration * 12
    ref_i = reference_rate / 12
    ref_ins = insurance / 12

    if ref_i <= 0:
        ref_annuity = 1.0 / n
    else:
        ref_annuity = ref_i / (1 - (1 + ref_i) ** (-n))
    ref_factor = ref_annuity + ref_ins

    def _annuity_factor(rate: float) -> float:
        i = rate / 12
        ins = insurance / 12
        if i <= 0:
            return 1.0 / n + ins
        return i / (1 - (1 + i) ** (-n)) + ins

    actual_factors = rate_at_time.apply(_annuity_factor)

    # monthly_cost = prix * actual_factor
    # equivalent_prix = monthly_cost / ref_factor
    adjusted = prix_m2_series * actual_factors / ref_factor
    return adjusted.rename("rate_adjusted_price")


# ---------------------------------------------------------------------------
# 7. Commune scorecard
# ---------------------------------------------------------------------------

_PHASE_SCORE = {
    "Bottom": 90,
    "Capitulation": 70,
    "Bull": 30,
    "Boom": 10,
}


def commune_scorecard(
    agg: pd.DataFrame,
    label_col: str = "commune",
    salary: float = 3000.0,
    duration: int = 25,
    insurance: float = 0.003,
    debt_ratio: float = 0.34,
) -> pd.DataFrame:
    """One-row-per-commune summary with a composite buy-attractiveness score.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data.
    label_col : str
        Column identifying the commune.
    salary : float
        Net monthly salary for affordability calculations.
    duration : int
        Loan duration in years.
    insurance : float
        Annual insurance rate.
    debt_ratio : float
        Maximum debt ratio.

    Returns
    -------
    pd.DataFrame
        Columns: ``commune``, ``current_price``, ``yoy_pct``,
        ``drawdown_from_peak``, ``phase``, ``signal``, ``affordability_m2``,
        ``affordability_vs_peak_pct``, ``volatility``, ``score``.
        Score is 0-100 composite where higher = more attractive to buy.
    """
    # Pre-compute signals once
    signals = composite_signal(agg, label_col=label_col)
    signal_map = {s.commune: s for s in signals}

    # Pre-compute phase history
    phase_df = market_phase_history(agg, label_col=label_col)

    rows: list[dict] = []

    for commune_name, sub in agg.sort_values("date_mutation").groupby(label_col):
        sub = sub.sort_values("date_mutation").copy()
        price = sub["prix_m2_median"]

        if price.empty or price.dropna().empty:
            continue

        current_price = float(price.iloc[-1])

        # Year-over-year percent change
        if len(price) >= 13:
            yoy_pct = float(
                (price.iloc[-1] - price.iloc[-13]) / price.iloc[-13] * 100
            )
        else:
            yoy_pct = np.nan

        # Drawdown from peak
        dd = drawdown_from_peak(price)
        current_dd = float(dd.iloc[-1])

        # Latest phase
        phase_sub = phase_df.loc[phase_df[label_col] == commune_name]
        if not phase_sub.empty:
            latest_phase = phase_sub.sort_values("date_mutation").iloc[-1]["phase"]
        else:
            latest_phase = "Unknown"

        # Signal
        sig = signal_map.get(str(commune_name))
        signal_str = sig.signal_type.value if sig else "N/A"
        confidence = sig.confidence if sig else 0.0

        # Affordability
        aff = affordability_timeline(
            agg, str(commune_name),
            salary=salary, duration=duration,
            insurance=insurance, debt_ratio=debt_ratio,
            label_col=label_col,
        )
        if not aff.empty:
            current_m2 = float(aff["purchasable_m2"].iloc[-1])
            aff_vs_peak = float(aff["pct_change_vs_peak"].iloc[-1])
        else:
            current_m2 = np.nan
            aff_vs_peak = np.nan

        # Volatility (rolling 12-month std of pct changes)
        pct_changes = price.pct_change()
        if len(pct_changes.dropna()) >= 6:
            volatility = float(pct_changes.rolling(12, min_periods=6).std().iloc[-1] * 100)
        else:
            volatility = np.nan

        # --- Composite score (0-100, higher = more attractive to buy) ---
        # Components (each normalised to ~0-100):
        # 1. Phase score
        phase_sc = _PHASE_SCORE.get(latest_phase, 50)

        # 2. Drawdown score: deeper drawdown = more attractive
        #    Map [-30, 0] -> [100, 0]
        dd_sc = float(np.clip(-current_dd * (100 / 30), 0, 100))

        # 3. Affordability trend: negative pct_change_vs_peak = worse
        #    Map [-40, 0] -> [0, 100]
        aff_sc = float(np.clip((40 + (aff_vs_peak if not np.isnan(aff_vs_peak) else 0)) * (100 / 40), 0, 100))

        # 4. Signal score: STRONG_BUY=100, BUY=80, HOLD=50, SELL=20, STRONG_SELL=0
        signal_scores = {
            SignalType.STRONG_BUY: 100,
            SignalType.BUY: 80,
            SignalType.HOLD: 50,
            SignalType.SELL: 20,
            SignalType.STRONG_SELL: 0,
        }
        sig_sc = signal_scores.get(
            sig.signal_type if sig else SignalType.HOLD, 50
        )

        # 5. Volatility score: lower vol = more attractive
        #    Map [0, 10] -> [100, 0]
        vol_val = volatility if not np.isnan(volatility) else 5.0
        vol_sc = float(np.clip((10 - vol_val) * 10, 0, 100))

        # Weighted composite
        score = (
            0.25 * phase_sc
            + 0.20 * dd_sc
            + 0.20 * aff_sc
            + 0.20 * sig_sc
            + 0.15 * vol_sc
        )

        rows.append({
            "commune": commune_name,
            "current_price": round(current_price, 0),
            "yoy_pct": round(yoy_pct, 1) if not np.isnan(yoy_pct) else np.nan,
            "drawdown_from_peak": round(current_dd, 1),
            "phase": latest_phase,
            "signal": signal_str,
            "confidence": round(confidence, 2),
            "affordability_m2": round(current_m2, 1) if not np.isnan(current_m2) else np.nan,
            "affordability_vs_peak_pct": round(aff_vs_peak, 1) if not np.isnan(aff_vs_peak) else np.nan,
            "volatility": round(volatility, 2) if not np.isnan(volatility) else np.nan,
            "score": round(score, 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 8. Decision summary
# ---------------------------------------------------------------------------

def decision_summary(
    agg: pd.DataFrame,
    label_col: str = "commune",
    salary: float = 3000.0,
) -> str:
    """Return a formatted text summary for each commune.

    Example line::

        Brest -- PHASE: Bottom | Prix: 1920 EUR/m2 (-8% du pic) |
        Avec 3000 EUR/mois: 86 m2 (-18% vs pic) | Signal: BUY (conf 0.27) |
        Score: 72/100

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data.
    label_col : str
        Column identifying the commune.
    salary : float
        Net monthly salary.

    Returns
    -------
    str
        Multi-line formatted summary.
    """
    sc = commune_scorecard(agg, label_col=label_col, salary=salary)

    if sc.empty:
        return "Aucune donnee disponible."

    lines: list[str] = []
    for _, row in sc.iterrows():
        dd_str = f"{row['drawdown_from_peak']:+.0f}%" if pd.notna(row["drawdown_from_peak"]) else "N/A"
        aff_str = (
            f"{row['affordability_m2']:.0f} m2"
            if pd.notna(row["affordability_m2"])
            else "N/A"
        )
        aff_peak_str = (
            f"{row['affordability_vs_peak_pct']:+.0f}% vs pic"
            if pd.notna(row["affordability_vs_peak_pct"])
            else ""
        )
        conf_str = f"{row['confidence']:.2f}" if pd.notna(row.get("confidence")) else "N/A"
        score_str = f"{row['score']:.0f}/100" if pd.notna(row["score"]) else "N/A"

        line = (
            f"{row['commune']} -- "
            f"PHASE: {row['phase']} | "
            f"Prix: {row['current_price']:.0f} EUR/m2 ({dd_str} du pic) | "
            f"Avec {salary:.0f} EUR/mois: {aff_str}"
        )
        if aff_peak_str:
            line += f" ({aff_peak_str})"
        line += f" | Signal: {row['signal']} (conf {conf_str})"
        line += f" | Score: {score_str}"

        lines.append(line)

    return "\n".join(lines)
