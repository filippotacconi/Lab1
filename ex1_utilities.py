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

    
    ttm = year_frac_act_x(ref_date, expiry, 365)

    d1 = (np.log(S0 / strike) + 0.5 * sigma_black**2 * ttm) / (sigma_black * np.sqrt(ttm))
    d2 = d1 - sigma_black * np.sqrt(ttm)

    
    fixed_leg_payment_dates = date_series(expiry, underlying_expiry, freq)

    
    bpv = basis_point_value(fixed_leg_payment_dates[1:], discount_factors,
                            settlement_date=expiry)

    # B(t0, t_alpha): discount factor at swaption maturity 
    # forward price back to t0
    ref = discount_factors.index[0]
    df_expiry = get_discount_factor_by_zero_rates_linear_interp(
        ref, expiry, discount_factors.index, discount_factors.values
    )


    if swaption_type == SwapType.PAYER:
        # using the appropriate Black's forumala
        price = df_expiry * bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))  # df_expiry added
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
   
    ref   = discount_factors.index[0]
    dates = discount_factors.index
    dfs   = discount_factors.values

    # 30E/360 year fractions for each coupon period
    year_fracs = schedule_year_fraction([ref_date] + list(fixed_leg_payment_dates))

    numerator, denominator = 0.0, 0.0
    for i, dt in enumerate(fixed_leg_payment_dates):
        df  = get_discount_factor_by_zero_rates_linear_interp(ref, dt, dates, dfs)
        t_i = year_frac_act_x(ref_date, dt, 365)  # ACT/365: real time weight for duration

        # Coupon + principal at maturity (replicates fixed bond cash flow)
        cf = swap_rate * year_fracs[i] + (1.0 if i == len(fixed_leg_payment_dates) - 1 else 0.0)

        numerator   += cf * t_i * df
        denominator += cf * df

    # Macaulay duration; negative because a receiver IRS loses value when rates rise
    return -(numerator / denominator)


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
    ref_date = discount_factors.index[0]
    dates    = discount_factors.index
    dfs      = discount_factors.values

    # Full schedule and 30E/360 year fractions
    start         = settlement_date if settlement_date is not None else ref_date
    year_fracs    = schedule_year_fraction([start] + list(fixed_leg_schedule))

    # Forward adjustment: 1.0 for spot swaps, B(t0, t_settlement) for forward-starting
    df_settlement = (
        get_discount_factor_by_zero_rates_linear_interp(ref_date, settlement_date, dates, dfs)
        if settlement_date is not None else 1.0
    )

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
    dates    = discount_factors.index
    dfs      = discount_factors.values

    bpv = basis_point_value(fixed_leg_schedule, discount_factors, settlement_date=fwd_start_date)

    # B(t0, t_start): 1.0 for spot swaps, interpolated for forward-starting
    df_start = (
        get_discount_factor_by_zero_rates_linear_interp(ref_date, fwd_start_date, dates, dfs)
        if fwd_start_date is not None else 1.0
    )

    # B(t0, t_N): terminal discount factor
    df_end = get_discount_factor_by_zero_rates_linear_interp(ref_date, fixed_leg_schedule[-1], dates, dfs)

    # Par condition: R = (B_start - B_end) / BPV  (float leg = 1 - B_end/B_start in forward terms)
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

    # Single curve framework, returns price and basis point value 
    bpv = basis_point_value(fixed_leg_schedule, discount_factors)

    # Discount factor at last date
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )

    float_leg = 1.0 - P_term

    fixed_leg = swap_rate * bpv

    # Payer: receives float, pays fixed 
    # Receiver: receives fixed, pays float 
    # error corrected (switched payer and receiver)
    if swap_type == SwapType.PAYER:
        muliplier = 1
    elif swap_type == SwapType.RECEIVER:
        muliplier = -1
    else:
        raise ValueError("Unknown swap type.")
    
    return muliplier * (float_leg - fixed_leg)
    

