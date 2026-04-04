"""Buy / sell signal detection for French real estate markets.

Combines multiple signal generators (z-score, momentum, mean-reversion,
volume anomalies, interest-rate adjustment) into a composite recommendation
per commune and date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SignalType(Enum):
    """Market signal classification."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Signal:
    """A single buy/sell signal for a commune at a point in time."""

    date: pd.Timestamp
    commune: str
    signal_type: SignalType
    confidence: float  # 0.0 -- 1.0
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Individual signal generators
# ---------------------------------------------------------------------------

def z_score_signal(series: pd.Series, window: int = 24) -> pd.Series:
    """Z-score of current price vs its rolling mean.

    A *negative* z-score indicates the price is below its recent average
    (cheaper than usual, potential **buy** signal).  A *positive* z-score
    suggests above-average pricing (potential **sell** signal).

    Parameters
    ----------
    series : pd.Series
        Monthly median price per m2.
    window : int
        Rolling window length (months).

    Returns
    -------
    pd.Series
        Z-score series (same index). Negative = buy bias, positive = sell bias.
    """
    roll_mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    roll_std = series.rolling(window, min_periods=max(1, window // 2)).std()
    return ((series - roll_mean) / roll_std.replace(0, np.nan)).rename("z_score")


def momentum_signal(
    series: pd.Series,
    short_window: int = 3,
    long_window: int = 12,
) -> pd.Series:
    """MACD-style momentum indicator for real estate prices.

    Returns the difference between a short and a long exponential moving
    average, normalised by the long EMA.  Positive values indicate upward
    momentum (sell bias); negative values indicate downward momentum (buy
    bias).

    Parameters
    ----------
    series : pd.Series
        Monthly median price.
    short_window : int
        Short EMA span (months).
    long_window : int
        Long EMA span (months).

    Returns
    -------
    pd.Series
        Normalised momentum (percent).  Negative = buy bias.
    """
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    momentum = (ema_short - ema_long) / ema_long.replace(0, np.nan) * 100
    return momentum.rename("momentum")


def mean_reversion_signal(series: pd.Series, window: int = 36) -> pd.Series:
    """Distance from the long-term rolling mean as a reversion indicator.

    When the price is far *below* the long-term mean, mean-reversion theory
    suggests it will rise (buy signal).  When far *above*, it suggests a
    future drop (sell signal).

    Returns a normalised deviation: ``(price - long_mean) / long_std``.
    Negative = buy bias.

    Parameters
    ----------
    series : pd.Series
        Monthly median price.
    window : int
        Long-term window (months).

    Returns
    -------
    pd.Series
        Normalised deviation.  Negative = buy bias.
    """
    roll_mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    roll_std = series.rolling(window, min_periods=max(1, window // 2)).std()
    deviation = (series - roll_mean) / roll_std.replace(0, np.nan)
    return deviation.rename("mean_reversion")


def volume_signal(transactions: pd.Series, window: int = 6) -> pd.Series:
    """Detect unusual transaction volume as a market-shift indicator.

    A spike in volume can precede a price move.  Returns the z-score of
    current volume vs its rolling mean -- high positive values indicate
    unusual activity.

    Parameters
    ----------
    transactions : pd.Series
        Monthly transaction count.
    window : int
        Rolling window (months).

    Returns
    -------
    pd.Series
        Volume z-score.  Large positive values = unusual activity.
    """
    roll_mean = transactions.rolling(window, min_periods=max(1, window // 2)).mean()
    roll_std = transactions.rolling(window, min_periods=max(1, window // 2)).std()
    return ((transactions - roll_mean) / roll_std.replace(0, np.nan)).rename("volume_z")


def rate_adjusted_signal(
    prix_m2: pd.Series,
    rates: pd.Series,
    loan_duration: int = 25,
    insurance_rate: float = 0.003,
) -> pd.Series:
    """Adjust price signal for interest-rate changes.

    Converts prix/m2 into an equivalent monthly-payment-per-m2, so that a
    price increase offset by a rate decrease results in a neutral signal.

    Returns the z-score of the monthly-payment-per-m2 series.

    Parameters
    ----------
    prix_m2 : pd.Series
        Monthly median price per m2.
    rates : pd.Series
        Annual interest rate series (aligned index with *prix_m2*).
    loan_duration : int
        Loan duration in years.
    insurance_rate : float
        Annual insurance rate.

    Returns
    -------
    pd.Series
        Z-score of payment-per-m2.  Negative = affordable = buy bias.
    """
    # Guard: if rates appear to be percentages (values > 1), convert to decimals
    if (rates.dropna() > 1).any():
        rates = rates / 100

    n_months = loan_duration * 12
    monthly_rate = rates / 12
    monthly_insurance = insurance_rate / 12

    # Annuity factor: monthly payment per euro borrowed
    annuity = monthly_rate / (1 - (1 + monthly_rate) ** (-n_months))
    annuity = annuity + monthly_insurance  # insurance on capital

    payment_per_m2 = prix_m2 * annuity

    roll_mean = payment_per_m2.rolling(24, min_periods=6).mean()
    roll_std = payment_per_m2.rolling(24, min_periods=6).std()
    z = (payment_per_m2 - roll_mean) / roll_std.replace(0, np.nan)
    return z.rename("rate_adjusted")


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "z_score": 0.25,
    "momentum": 0.25,
    "mean_reversion": 0.20,
    "volume": 0.15,
    "rate_adjusted": 0.15,
}


def _classify(score: float) -> tuple[SignalType, float]:
    """Map a composite score to a SignalType and confidence.

    The *score* is a weighted average of individual signal z-scores where
    negative means buy-biased and positive means sell-biased.
    """
    abs_score = abs(score)
    confidence = float(np.clip(abs_score / 3.0, 0.0, 1.0))

    if score <= -1.5:
        return SignalType.STRONG_BUY, confidence
    if score <= -0.5:
        return SignalType.BUY, confidence
    if score >= 1.5:
        return SignalType.STRONG_SELL, confidence
    if score >= 0.5:
        return SignalType.SELL, confidence
    return SignalType.HOLD, confidence


def composite_signal(
    agg: pd.DataFrame,
    rate_history: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    label_col: str = "commune",
) -> list[Signal]:
    """Combine all signal generators into per-commune buy/sell recommendations.

    For each commune, the function computes the individual signals, takes a
    weighted average, and classifies the latest observation.

    Parameters
    ----------
    agg : pd.DataFrame
        Monthly aggregated data (output of :func:`trends.add_derived_metrics`
        or at minimum containing ``date_mutation``, *label_col*,
        ``prix_m2_median``, ``n_transactions``).
    rate_history : pd.DataFrame | None
        Optional DataFrame with columns ``date_mutation`` and ``rate``
        (annual interest rate).  When provided, the rate-adjusted signal is
        included; otherwise its weight is redistributed.
    weights : dict | None
        Override default signal weights.  Keys: ``z_score``, ``momentum``,
        ``mean_reversion``, ``volume``, ``rate_adjusted``.
    label_col : str
        Column name for the commune / group label.

    Returns
    -------
    list[Signal]
        One :class:`Signal` per commune, representing the latest composite
        recommendation.
    """
    w = dict(_DEFAULT_WEIGHTS if weights is None else weights)

    # Normalise rate_history column names to expected convention
    if rate_history is not None and not rate_history.empty:
        if "date" in rate_history.columns and "date_mutation" not in rate_history.columns:
            rate_history = rate_history.rename(columns={"date": "date_mutation"})
        if "rate_pct" in rate_history.columns and "rate" not in rate_history.columns:
            rate_history = rate_history.copy()
            rate_history["rate"] = rate_history["rate_pct"] / 100

    # If no rate data, redistribute rate_adjusted weight proportionally
    has_rates = rate_history is not None and not rate_history.empty
    if not has_rates and "rate_adjusted" in w:
        rate_w = w.pop("rate_adjusted")
        total = sum(w.values()) or 1.0
        w = {k: v + rate_w * (v / total) for k, v in w.items()}

    # Normalise weights to sum to 1
    w_total = sum(w.values()) or 1.0
    w = {k: v / w_total for k, v in w.items()}

    signals: list[Signal] = []

    for commune, sub in agg.sort_values("date_mutation").groupby(label_col):
        sub = sub.set_index("date_mutation").sort_index()
        price = sub["prix_m2_median"]
        volume = sub["n_transactions"]

        components: dict[str, pd.Series] = {}
        reasons: list[str] = []

        # -- z_score
        if "z_score" in w:
            components["z_score"] = z_score_signal(price)

        # -- momentum
        if "momentum" in w:
            components["momentum"] = momentum_signal(price)

        # -- mean_reversion
        if "mean_reversion" in w:
            components["mean_reversion"] = mean_reversion_signal(price)

        # -- volume
        if "volume" in w:
            components["volume"] = volume_signal(volume)

        # -- rate_adjusted
        if has_rates and "rate_adjusted" in w:
            rate_df = rate_history.set_index("date_mutation")["rate"].sort_index()
            # Align rates to the commune index
            rate_aligned = rate_df.reindex(sub.index, method="ffill")
            if rate_aligned.notna().sum() > 6:
                components["rate_adjusted"] = rate_adjusted_signal(price, rate_aligned)

        # Weighted composite for each time step
        composite = pd.Series(0.0, index=sub.index)
        for key, comp_series in components.items():
            comp_aligned = comp_series.reindex(sub.index)
            weight = w.get(key, 0.0)
            composite = composite + comp_aligned.fillna(0.0) * weight

        # Latest observation
        if composite.empty or composite.isna().all():
            continue

        latest_date = composite.dropna().index[-1]
        latest_score = composite.loc[latest_date]

        # Build reasons
        for key, comp_series in components.items():
            val = comp_series.reindex(sub.index).get(latest_date, np.nan)
            if pd.notna(val):
                direction = "buy bias" if val < -0.3 else ("sell bias" if val > 0.3 else "neutral")
                reasons.append(f"{key}: {val:+.2f} ({direction})")

        signal_type, confidence = _classify(latest_score)
        signals.append(
            Signal(
                date=latest_date,
                commune=str(commune),
                signal_type=signal_type,
                confidence=confidence,
                reasons=reasons,
            )
        )

    return signals


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def signal_summary(signals: Sequence[Signal]) -> pd.DataFrame:
    """Create a summary DataFrame of the latest signals per commune.

    Parameters
    ----------
    signals : list[Signal]
        Output of :func:`composite_signal`.

    Returns
    -------
    pd.DataFrame
        Columns: ``commune``, ``date``, ``signal``, ``confidence``, ``reasons``.
    """
    rows = [
        {
            "commune": s.commune,
            "date": s.date,
            "signal": s.signal_type.value,
            "confidence": round(s.confidence, 2),
            "reasons": "; ".join(s.reasons),
        }
        for s in signals
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("commune").reset_index(drop=True)
    return df
