"""Interest rate dynamics and purchasing power analysis.

Provides loan calculations, borrowing capacity estimation, purchasing power
indexing, rate sensitivity tables, and a salary-vs-capital visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Core loan maths
# ---------------------------------------------------------------------------

def monthly_payment(
    capital: float,
    annual_rate: float,
    duration_years: int = 25,
    insurance_rate: float = 0.003,
) -> float:
    """Compute the total monthly loan payment including insurance.

    Parameters
    ----------
    capital : float
        Borrowed amount (euros).
    annual_rate : float
        Annual interest rate as a decimal (e.g. 0.035 for 3.5 %).
    duration_years : int
        Loan duration in years.
    insurance_rate : float
        Annual insurance rate on the remaining capital (approximated on
        the initial capital for simplicity, as French banks typically quote
        it this way).

    Returns
    -------
    float
        Total monthly payment (credit amortisation + insurance), in euros.
    """
    n = duration_years * 12
    i = annual_rate / 12

    if i <= 0:
        # Edge case: zero or negative rate -- simple division
        credit_payment = capital / n
    else:
        credit_payment = capital * i / (1 - (1 + i) ** (-n))

    insurance_payment = capital * insurance_rate / 12
    return credit_payment + insurance_payment


def borrowing_capacity(
    monthly_salary: float,
    annual_rate: float,
    duration_years: int = 25,
    debt_ratio: float = 0.34,
    insurance_rate: float = 0.003,
) -> float:
    """Maximum capital one can borrow given salary and rate constraints.

    The French ``taux d'endettement`` rule limits total debt payments to
    *debt_ratio* of net monthly salary.

    Parameters
    ----------
    monthly_salary : float
        Net monthly salary (euros, after tax).
    annual_rate : float
        Annual interest rate (decimal).
    duration_years : int
        Loan duration in years.
    debt_ratio : float
        Maximum fraction of salary allocated to loan repayment.
    insurance_rate : float
        Annual insurance rate.

    Returns
    -------
    float
        Maximum borrowable capital (euros).
    """
    max_monthly = monthly_salary * debt_ratio
    n = duration_years * 12
    i = annual_rate / 12
    ins_monthly = insurance_rate / 12

    if i <= 0:
        # credit_payment = capital / n => capital = max_monthly_credit * n
        # max_monthly_credit = max_monthly - capital * ins_monthly
        # capital / n + capital * ins_monthly = max_monthly
        # capital * (1/n + ins_monthly) = max_monthly
        return max_monthly / (1.0 / n + ins_monthly)

    # credit_payment = capital * i / (1 - (1+i)^-n)
    # insurance_payment = capital * ins_monthly
    # total = capital * (i / (1 - (1+i)^-n) + ins_monthly) = max_monthly
    annuity_factor = i / (1 - (1 + i) ** (-n))
    return max_monthly / (annuity_factor + ins_monthly)


# ---------------------------------------------------------------------------
# Purchasing power index
# ---------------------------------------------------------------------------

def purchasing_power_index(
    prix_m2_series: pd.Series,
    rate_series: pd.Series,
    reference_salary: float = 2500.0,
    duration: int = 25,
    insurance_rate: float = 0.003,
) -> pd.Series:
    """How many m2 can a fixed-salary buyer afford over time?

    Combines price evolution and interest-rate dynamics into a single
    purchasing-power metric.

    Parameters
    ----------
    prix_m2_series : pd.Series
        Monthly median price per m2.
    rate_series : pd.Series
        Annual interest rate series (aligned with *prix_m2_series*).
    reference_salary : float
        Reference net monthly salary (euros).
    duration : int
        Loan duration (years).
    insurance_rate : float
        Annual insurance rate.

    Returns
    -------
    pd.Series
        Purchasable surface (m2) at each time step.
    """
    capacity = rate_series.apply(
        lambda r: borrowing_capacity(reference_salary, r, duration, debt_ratio=0.34, insurance_rate=insurance_rate)
    )
    m2_affordable = capacity / prix_m2_series.replace(0, np.nan)
    return m2_affordable.rename("purchasable_m2")


# ---------------------------------------------------------------------------
# Rate sensitivity table
# ---------------------------------------------------------------------------

def rate_sensitivity(
    capital: float,
    rates: list[float],
    duration: int = 25,
    insurance_rate: float = 0.003,
    debt_ratio: float = 0.34,
) -> pd.DataFrame:
    """Table mapping rate to monthly payment and required salary.

    Parameters
    ----------
    capital : float
        Loan capital (euros).
    rates : list[float]
        Interest rates to evaluate.
    duration : int
        Loan duration (years).
    insurance_rate : float
        Annual insurance rate.
    debt_ratio : float
        Maximum debt ratio for salary computation.

    Returns
    -------
    pd.DataFrame
        Columns: ``rate_pct``, ``monthly_payment``, ``required_salary``.
    """
    rows = []
    for r in sorted(rates):
        mp = monthly_payment(capital, r, duration, insurance_rate)
        salary = mp / debt_ratio
        rows.append(
            {
                "rate_pct": r * 100,
                "monthly_payment": round(mp, 2),
                "required_salary": round(salary, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Equivalent price at different rate
# ---------------------------------------------------------------------------

def equivalent_price_at_rate(
    current_price: float,
    current_rate: float,
    target_rate: float,
    duration: int = 25,
    insurance_rate: float = 0.003,
) -> float:
    """What price at *target_rate* gives the same monthly payment as
    *current_price* at *current_rate*?

    Useful for answering: "If rates drop from 4 % to 3 %, what price is
    effectively the same cost for a buyer?"

    Parameters
    ----------
    current_price : float
        Current property price (euros).
    current_rate : float
        Current annual interest rate.
    target_rate : float
        Target annual interest rate.
    duration : int
        Loan duration (years).
    insurance_rate : float
        Annual insurance rate.

    Returns
    -------
    float
        Equivalent price at *target_rate*.
    """
    current_mp = monthly_payment(current_price, current_rate, duration, insurance_rate)

    # Solve: monthly_payment(X, target_rate) = current_mp
    n = duration * 12
    i = target_rate / 12
    ins_monthly = insurance_rate / 12

    if i <= 0:
        annuity = 1.0 / n
    else:
        annuity = i / (1 - (1 + i) ** (-n))

    # current_mp = X * (annuity + ins_monthly)
    return current_mp / (annuity + ins_monthly)


# ---------------------------------------------------------------------------
# Salary vs capital chart
# ---------------------------------------------------------------------------

def plot_salary_vs_capital(
    rates: list[float],
    duration: int = 25,
    capital_range: tuple[float, float] = (100_000, 400_000),
    n_points: int = 100,
    insurance_rate: float = 0.003,
    debt_ratio: float = 0.34,
) -> matplotlib.figure.Figure:
    """Chart: required net salary as a function of borrowed capital for
    several interest rates.

    Rewrite of the standalone ``capital_emprunt.py`` script, returning a
    matplotlib :class:`~matplotlib.figure.Figure` instead of calling
    ``plt.show()``.

    Parameters
    ----------
    rates : list[float]
        Interest rates to plot (decimal, e.g. ``[0.02, 0.03, 0.04]``).
    duration : int
        Loan duration (years).
    capital_range : tuple[float, float]
        ``(min_capital, max_capital)`` in euros.
    n_points : int
        Number of points along the capital axis.
    insurance_rate : float
        Annual insurance rate.
    debt_ratio : float
        Maximum debt ratio.

    Returns
    -------
    matplotlib.figure.Figure
    """
    capitals = np.linspace(capital_range[0], capital_range[1], n_points)

    fig, ax = plt.subplots(figsize=(9, 5))

    for rate in sorted(rates):
        n_months = duration * 12
        i_monthly = rate / 12
        ins_monthly = insurance_rate / 12

        if i_monthly <= 0:
            credit = capitals / n_months
        else:
            credit = capitals * i_monthly / (1 - (1 + i_monthly) ** (-n_months))

        insurance = capitals * ins_monthly
        total_monthly = credit + insurance
        salaries = total_monthly / debt_ratio

        ax.plot(salaries, capitals, linewidth=2, label=f"Taux {rate * 100:.1f} %")

    ax.set_title(
        "Salaire minimal en fonction du capital et du taux d'interet",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Salaire mensuel net (euros)", fontsize=12)
    ax.set_ylabel("Capital empruntable (euros)", fontsize=12)

    euro_fmt = FuncFormatter(lambda x, _: f"{x:,.0f} EUR".replace(",", " "))
    ax.xaxis.set_major_formatter(euro_fmt)
    ax.yaxis.set_major_formatter(euro_fmt)

    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(title="Taux d'interet")
    fig.tight_layout()

    return fig
