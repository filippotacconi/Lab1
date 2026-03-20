"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from date_functions import (
    year_frac_act_x,
    date_series,
    year_frac_30e_360,
    schedule_year_fraction,
)
from ex0_utilities import (
    get_discount_factor_by_zero_rates_linear_interp,
)

from scipy.stats import norm

from typing import Union, List, Tuple


class SwapType(Enum):
    """
    Types of swaptions.
    """

    RECEIVER = "receiver"
    PAYER = "payer"


def swaption_price_calculator(
    S0: float,
    strike: float,
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    underlying_expiry: Union[dt.date, pd.Timestamp],
    sigma_black: float,
    freq: int,
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Return the swaption price defined by the input parameters.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Value date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Underlying forward starting swap expiry.
        sigma_black (float): Swaption implied volatility.
        freq (int): Number of times a year the fixed leg pays the coupon.
        discount_factors (pd.Series): Discount factors.
        swaption_type (SwapType): Swaption type, default to receiver.

    Returns:
        Union[float, Tuple[float, float]]: Swaption price (and possibly delta).
    """

    # Time to maturity in years (ACT/365)
    ttm = year_frac_act_x(ref_date, expiry, 365)

    # Black's d1 and d2
    d1 = (np.log(S0 / strike) + 0.5 * sigma_black**2 * ttm) / (sigma_black * np.sqrt(ttm))
    d2 = d1 - sigma_black * np.sqrt(ttm)

    # Fixed leg payment dates of the underlying forward-starting swap
    fixed_leg_payment_dates = date_series(expiry, underlying_expiry, freq)

    # Forward BPV of the underlying swap (discounted from expiry onwards)
    bpv = basis_point_value(fixed_leg_payment_dates[1:], discount_factors,
                            settlement_date=expiry)

    # Discount factor at swaption expiry B(t0, t_alpha)
    # Needed to bring the forward price back to t0
    ref = discount_factors.index[0]
    df_expiry = get_discount_factor_by_zero_rates_linear_interp(
        ref, expiry, discount_factors.index, discount_factors.values
    )

    # Black's formula: Price = B(t0, t_alpha) * BPV_fwd * { ... }
    if swaption_type == SwapType.PAYER:
        price = df_expiry * bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))
        delta = df_expiry * bpv * norm.cdf(d1)

    elif swaption_type == SwapType.RECEIVER:
        price = df_expiry * bpv * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        delta = df_expiry * bpv * (norm.cdf(d1)- 1.0)
    else:
        raise ValueError("Invalid swaption type.")

    if compute_delta:
        return price, delta
    else:
        return price


def irs_proxy_duration(
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: pd.Series,
) -> float:
    """
    Given the specifics of an interest rate swap (IRS), return its rate sensitivity calculated as
    the duration of a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Reference date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.

    Returns:
        (float): Swap duration.
    """


    ref = discount_factors.index[0]
    dates = discount_factors.index
    dfs = discount_factors.values

    # Year fractions for coupon periods (30E/360 convention)
    year_fracs = schedule_year_fraction([ref_date] + list(fixed_leg_payment_dates))

    numerator = 0.0
    denominator = 0.0

    for i, dt in enumerate(fixed_leg_payment_dates):
        # Discount factor at payment date
        df = get_discount_factor_by_zero_rates_linear_interp(ref, dt, dates, dfs)

        # Time from ref_date to payment date (ACT/365, real time for duration)
        t_i = year_frac_act_x(ref_date, dt, 365)

        # Bond cash flow: coupon at every date, plus principal at maturity
        cf = swap_rate * year_fracs[i]
        if i == len(fixed_leg_payment_dates) - 1:
            cf += 1.0

        numerator += cf * t_i * df
        denominator += cf * df

    mac_duration = numerator / denominator

    # Negative sign: receiver IRS loses value when rates increase
    return -mac_duration


def basis_point_value(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    settlement_date: dt.datetime | None = None,
) -> float:
    """
    Given a swap fixed leg payment dates and the discount factors, return the basis point value.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        settlement_date (dt.datetime | None): Settlement date, default to None, i.e. to today.
            Needed in case of forward starting swaps.

    Returns:
        float: Basis point value.
    """
    # Extract reference date and discount curve data
    ref_date = discount_factors.index[0]
    dates = discount_factors.index
    dfs = discount_factors.values

    # Build the full schedule: start date + payment dates
    start = settlement_date if settlement_date is not None else ref_date
    full_schedule = [start] + list(fixed_leg_schedule)

    # Compute year fractions between consecutive dates (30E/360 convention)
    year_fracs = schedule_year_fraction(full_schedule)

    # For forward-starting swaps, divide by B(t0, t_settlement) to get forward discounts
    # For spot swaps, this is simply 1.0
    df_settlement = (
        get_discount_factor_by_zero_rates_linear_interp(ref_date, settlement_date, dates, dfs)
        if settlement_date is not None
        else 1.0
    )

    # BPV = sum of year_frac(t_{i-1}, t_i) * B_fwd(t0; t_settlement, t_i)
    # where B_fwd = B(t0, t_i) / B(t0, t_settlement)
    bpv = sum(
        yf * get_discount_factor_by_zero_rates_linear_interp(ref_date, dt, dates, dfs) / df_settlement
        for yf, dt in zip(year_fracs, fixed_leg_schedule)
    )

    return bpv


def swap_par_rate(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: dt.datetime | None = None,
) -> float:
    """
    Given a fixed leg payment schedule and the discount factors, return the swap par rate. If a
    forward start date is provided, a forward swap rate is returned.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        fwd_start_date (dt.datetime | None): Forward start date, default to None.

    Returns:
        float: Swap par rate.
    """

    ref_date = discount_factors.index[0]
    dates = discount_factors.index
    dfs = discount_factors.values

    # Forward BPV if forward-starting, spot BPV otherwise
    bpv = basis_point_value(fixed_leg_schedule, discount_factors,
                            settlement_date=fwd_start_date)

    # B(t0, t_n): discount factor at forward start date
    if fwd_start_date is not None:
        df_start = get_discount_factor_by_zero_rates_linear_interp(
            ref_date, fwd_start_date, dates, dfs
        )
    else:
        df_start = 1.0

    # B(t0, t_N): discount factor at last payment date
    df_end = get_discount_factor_by_zero_rates_linear_interp(
        ref_date, fixed_leg_schedule[-1], dates, dfs
    )

    # Float leg in forward terms: 1 - B(t0, t_N) / B(t0, t_n)
    # Spot case: df_start = 1.0, so float_leg = 1 - B(t0, t_N)
    float_leg = 1.0 - df_end / df_start

    return float_leg / bpv


def swap_mtm(
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Given a swap rate, a fixed leg payment schedule and the discount factors, return the swap
    mark-to-market.

    Parameters:
        swap_rate (float): Swap rate.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        swap_type (SwapType): Swap type, either 'payer' or 'receiver', default to 'payer'.

    Returns:
        float: Swap mark-to-market.
    """

    # Spot BPV (no settlement_date needed for spot swaps)
    bpv = basis_point_value(fixed_leg_schedule, discount_factors)

    # Discount factor at last payment date
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0], fixed_leg_schedule[-1],
        discount_factors.index, discount_factors.values,
    )

    # Float leg value: 1 - B(t0, tN) (telescopic sum in single-curve)
    float_leg = 1.0 - P_term

    # Fixed leg value: S * BPV
    fixed_leg = swap_rate * bpv

    # Payer: receives float, pays fixed -> MtM = float - fixed
    # Receiver: receives fixed, pays float -> MtM = fixed - float
    if swap_type == SwapType.PAYER:
        return float_leg - fixed_leg
    elif swap_type == SwapType.RECEIVER:
        return fixed_leg - float_leg
    else:
        raise ValueError("Unknown swap type.")
