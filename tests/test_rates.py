"""Tests for immo.analysis.rates."""

from __future__ import annotations

from immo.analysis.rates import borrowing_capacity, monthly_payment


def test_monthly_payment():
    # 200k at 3% over 25 years, no insurance
    payment = monthly_payment(200_000, 0.03, duration_years=25, insurance_rate=0.0)
    # Expected ~948 EUR/month
    assert 940 < payment < 960, f"Expected ~948, got {payment:.2f}"


def test_borrowing_capacity():
    # Round-trip: find capacity, then verify the payment fits the salary constraint
    salary = 3000.0
    rate = 0.035
    duration = 25
    debt_ratio = 0.34
    insurance = 0.003

    capacity = borrowing_capacity(
        salary, rate, duration, debt_ratio=debt_ratio, insurance_rate=insurance
    )
    assert capacity > 0

    # The monthly payment for that capacity should equal salary * debt_ratio
    payment = monthly_payment(capacity, rate, duration, insurance_rate=insurance)
    max_allowed = salary * debt_ratio
    assert abs(payment - max_allowed) < 1.0, (
        f"Payment {payment:.2f} should match max allowed {max_allowed:.2f}"
    )
