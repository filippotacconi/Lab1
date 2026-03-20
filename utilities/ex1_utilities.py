"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from utilities.date_functions import (
    year_frac_act_x,
    date_series,
    year_frac_30e_360,
    schedule_year_fraction,
)
from utilities.ex0_utilities import (
    get_discount_factor_by_zero_rates_linear_interp,
)

from scipy.stats import norm

from typing import Union, List, Tuple


class SwapType(Enum):
    """
    Types of swaptions / IRS.
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
    Price a European swaption using the Black (1976) model.

    The Black model for a payer swaption gives:
        Price = BPV * [S0 * N(d1) - K * N(d2)]
    and for a receiver swaption:
        Price = BPV * [K * N(-d2) - S0 * N(-d1)]

    where:
        d1 = [ln(S0/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        BPV = sum of discount_factor(t_i) * delta_t_i  (basis point value / annuity)

    The delta of the swaption w.r.t. the forward swap rate S0 is:
        Payer delta   = BPV * [N(d1) - 1]   (negative: long payer loses when rates fall)
        Receiver delta = BPV * N(d1)         (positive)

    Parameters:
        S0 (float): Forward swap rate (ATM strike for ATM swaption).
        strike (float): Swaption strike price (= S0 for ATM).
        ref_date: Value date (today).
        expiry: Swaption expiry date.
        underlying_expiry: Maturity of the underlying forward-starting swap.
        sigma_black (float): Black implied volatility.
        freq (int): Number of fixed-leg coupon payments per year.
        discount_factors (pd.Series): Bootstrapped discount curve indexed by date.
        swaption_type (SwapType): PAYER or RECEIVER.
        compute_delta (bool): If True, also return the swaption delta.

    Returns:
        float or (float, float): Swaption price per unit notional (and delta if requested).
    """

    # Time to maturity (ACT/365)
    ttm = year_frac_act_x(ref_date, expiry, 365)

    # Black d1 and d2
    d1 = (np.log(S0 / strike) + 0.5 * sigma_black ** 2 * ttm) / (sigma_black * np.sqrt(ttm))
    d2 = d1 - sigma_black * np.sqrt(ttm)

    # Fixed leg payment dates of the underlying forward-starting swap
    # (includes expiry as start, then annual dates up to underlying_expiry)
    fixed_leg_payment_dates = date_series(expiry, underlying_expiry, freq)

    # BPV (annuity): sum over coupon periods of year_fraction * discount_factor
    bpv = basis_point_value(fixed_leg_payment_dates[1:], discount_factors,
                             settlement_date=fixed_leg_payment_dates[0])

    # Black pricing formulas
    if swaption_type == SwapType.PAYER:
        # Payer swaption: right to enter a payer swap (pay fixed, receive float)
        # Price = BPV * [S0 * N(d1) - K * N(d2)]
        price = bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))
        # Delta w.r.t. S0: dPrice/dS0 = BPV * N(d1)
        # But since a long payer benefits when rates RISE, the sign convention
        # used in the notebook for hedging is: delta = BPV * (N(d1) - 1)
        # This represents the sensitivity to parallel rate shifts (negative for a payer,
        # since the BPV annuity itself has negative duration).
        delta = bpv * (norm.cdf(d1) - 1)
    elif swaption_type == SwapType.RECEIVER:
        # Receiver swaption: right to enter a receiver swap (receive fixed, pay float)
        # Price = BPV * [K * N(-d2) - S0 * N(-d1)]
        price = bpv * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        # Delta w.r.t. S0: dPrice/dS0 = -BPV * N(-d1) = BPV * (N(d1) - 1)
        delta = bpv * norm.cdf(d1)
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
    Compute the rate sensitivity of an IRS approximated as the (modified) duration
    of the equivalent fixed-rate bond.

    The IRS MtM (receiver) = fixed_leg - float_leg = swap_rate * BPV - (1 - P(T))
    Its sensitivity to a parallel rate shift dr is approximated by the bond duration:
        dMtM/dr ≈ -D_mod * MtM_bond

    However, for the proxy DV01 used in the notebook:
        IRS_DV01 ≈ -duration * 1e-4
    where duration is the modified duration of the fixed coupon bond:

        D_mod = [sum_i t_i * c_i * P(t_i) + T * P(T)] / Bond_Price
        Bond_Price = swap_rate * BPV + P(T)    (par bond since swap_rate = par rate)

    For an ATM IRS (swap_rate = par rate), Bond_Price = 1 and:
        D_mod = sum_i t_i * c_i * P(t_i) + T * P(T)

    But the notebook formula uses:
        ptf_proxy_dv01 = (N_swaption * swaption_delta + N_irs * irs_duration) * 1e-4
    so irs_duration must be the *negative* of dMtM/dr per unit, i.e.
        irs_duration = -D_mod   (negative, since receiver IRS loses when rates rise)

    Parameters:
        ref_date (dt.date): Reference/value date (today).
        swap_rate (float): Par swap rate (= ATM rate since the IRS is ATM).
        fixed_leg_payment_dates (List[dt.date]): Annual fixed leg payment dates.
        discount_factors (pd.Series): Bootstrapped discount curve.

    Returns:
        float: Proxy duration (negative for a receiver IRS, reflecting rate sensitivity).
    """

    ref = discount_factors.index[0]  # reference date for the discount curve

    # Year fractions from today to each payment date (ACT/365 for discounting)
    t_i = [year_frac_act_x(ref_date, d, 365) for d in fixed_leg_payment_dates]

    # Discount factors at each payment date
    df_i = [
        get_discount_factor_by_zero_rates_linear_interp(
            ref, d, discount_factors.index, discount_factors.values
        )
        for d in fixed_leg_payment_dates
    ]

    # Year fractions for coupon calculation (30E/360 between consecutive dates)
    dates_with_today = [ref_date] + list(fixed_leg_payment_dates)
    coupon_yf = schedule_year_fraction(dates_with_today)  # 30E/360 fractions

    # Bond price = sum of coupon cash flows + notional repayment
    # For an ATM IRS, bond_price ≈ 1 (par)
    bond_price = sum(swap_rate * yf * df for yf, df in zip(coupon_yf, df_i)) + df_i[-1]

    # Numerator: sum of t_i * cash_flow_i * df_i (weighted average maturity)
    numerator = sum(
        t * (swap_rate * yf * df)
        for t, yf, df in zip(t_i, coupon_yf, df_i)
    ) + t_i[-1] * df_i[-1]  # principal at final date

    # Modified duration of the equivalent bond
    # D_mod = (1/bond_price) * numerator (no /(1+y) since we use continuous rates)
    modified_duration = numerator / bond_price

    # For a receiver IRS: dMtM/dr ≈ -D_mod * bond_price
    # The notebook uses: ptf_proxy_dv01 = (N_sw * delta_sw + N_irs * irs_duration) * 1e-4
    # where the IRS is a PAYER (loses value when rates fall), so we return -D_mod
    # (the duration proxy for the rate sensitivity dMtM/d(parallel_shift))
    return -modified_duration


def basis_point_value(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    settlement_date: dt.datetime | None = None,
) -> float:
    """
    Compute the Basis Point Value (BPV / annuity) of a swap fixed leg.

    BPV = sum_i [ delta_t_i * P(t_i) ]

    where delta_t_i is the year fraction of the i-th coupon period (30E/360)
    and P(t_i) is the discount factor at payment date t_i.

    For a spot-starting swap, consecutive year fractions run from today.
    For a forward-starting swap, a settlement_date (= swap start date) must be
    provided and year fractions run from that start date.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates (excluding start).
        discount_factors (pd.Series): Bootstrapped discount curve indexed by date.
        settlement_date (dt.datetime | None): Start date of the swap.
            If None, the first date of discount_factors is used (today for a spot swap).

    Returns:
        float: Basis point value of the fixed leg (annuity).
    """

    ref = discount_factors.index[0]  # curve reference date

    # Determine the schedule start for year-fraction computation
    if settlement_date is not None:
        start = settlement_date
    else:
        start = ref  # spot-starting: fractions run from today

    # Build the full schedule including the start date for 30E/360 fractions
    full_schedule = [start] + list(fixed_leg_schedule)

    # Year fractions between consecutive dates using 30E/360
    year_fractions = schedule_year_fraction(full_schedule)

    bpv = 0.0
    for pay_date, yf in zip(fixed_leg_schedule, year_fractions):
        # Interpolated discount factor at the payment date
        df = get_discount_factor_by_zero_rates_linear_interp(
            ref, pay_date, discount_factors.index, discount_factors.values
        )
        bpv += yf * df

    return bpv


def swap_par_rate(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: dt.datetime | None = None,
) -> float:
    """
    Compute the par swap rate (or forward swap rate for a forward-starting swap).

    The par rate K is the fixed rate that makes the swap MtM equal to zero:
        K = (P(t0) - P(tN)) / BPV

    where:
        P(t0) = discount factor at the start of the swap
                (= 1 for spot-starting, or P(fwd_start_date) for forward-starting)
        P(tN) = discount factor at the final maturity
        BPV   = sum_i [ delta_t_i * P(t_i) ]  (annuity factor)

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates (excluding start).
        discount_factors (pd.Series): Bootstrapped discount curve.
        fwd_start_date (dt.datetime | None): Forward start date. If None, spot-starting swap.

    Returns:
        float: Par swap rate.
    """

    ref = discount_factors.index[0]  # curve reference date

    # Discount factor at the start of the swap
    if fwd_start_date is not None:
        # Forward-starting: P(t0) = discount factor at the forward start date
        discount_factor_t0 = get_discount_factor_by_zero_rates_linear_interp(
            ref, fwd_start_date, discount_factors.index, discount_factors.values
        )
    else:
        # Spot-starting: P(t0) = 1 (today = reference date)
        discount_factor_t0 = 1.0

    # BPV of the fixed leg
    bpv = basis_point_value(fixed_leg_schedule, discount_factors,
                             settlement_date=fwd_start_date)

    # Discount factor at final maturity
    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(
        ref,
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )

    # Float leg PV = P(t0) - P(tN)  [single-curve framework]
    float_leg = discount_factor_t0 - discount_factor_tN

    return float_leg / bpv


def swap_mtm(
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.RECEIVER,
) -> float:
    """
    Compute the mark-to-market of an interest rate swap (per unit notional).

    In the single-curve framework:
        Float leg PV = 1 - P(tN)
        Fixed leg PV = swap_rate * BPV

    MtM (receiver) = Fixed - Float = swap_rate * BPV - (1 - P(tN))
    MtM (payer)    = Float - Fixed = (1 - P(tN)) - swap_rate * BPV

    Note: a spot-starting ATM swap has MtM = 0 by construction.

    Parameters:
        swap_rate (float): Fixed rate of the IRS (= par rate for an ATM swap → MtM = 0).
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Bootstrapped discount curve.
        swap_type (SwapType): RECEIVER (receive fixed, pay float) or PAYER (pay fixed, receive float).

    Returns:
        float: Swap MtM per unit notional.
    """

    ref = discount_factors.index[0]  # curve reference date

    # BPV of the fixed leg
    bpv = basis_point_value(fixed_leg_schedule, discount_factors)

    # Discount factor at final maturity
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        ref,
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )

    # PV of floating leg (single-curve: 1 - P(tN))
    float_leg = 1.0 - P_term
    # PV of fixed leg
    fixed_leg = swap_rate * bpv

    if swap_type == SwapType.RECEIVER:
        # Receiver: we receive fixed, pay float → MtM = Fixed - Float
        multiplier = 1
    elif swap_type == SwapType.PAYER:
        # Payer: we pay fixed, receive float → MtM = Float - Fixed
        multiplier = -1
    else:
        raise ValueError("Unknown swap type.")

    return multiplier * (float_leg - fixed_leg)
