"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 0: Discount Factors Bootstrap
"""

import numpy as np
import pandas as pd
import datetime as dt
from utilities.date_functions import (
    business_date_offset,
    year_frac_act_x,
    year_frac_30e_360
)
from typing import Iterable, Union, List, Union, Tuple

def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> List[float]:
    """
    Compute the zero rates from the discount factors.

    Parameters:
        dates (Union[List[float], pd.DatetimeIndex]): List of year fractions or dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        List[float]: List of zero rates.
    """

    effDates, effDf = dates, discount_factors
    if isinstance(effDates, pd.DatetimeIndex):   # if the input are dates, it must be converted to year fractions, if it is already year fractions, nothing to do. 
                                                 # complete the function 
        ref_date = effDates[0] # fix the first date as the initial date in which we consider the year fractions
        effDates = effDates[1:] # we do not consider the first date (we would get 0 as year fraction)
        effDf = discount_factors[1:] # we do not consider the first discount (since we are not using it)

        effDates = [year_frac_act_x(ref_date, t, 365) for t in effDates] # Use the function in date_functions, with act365 convention

    zero_rates=[]
    for t, df in zip(effDates, effDf):
        if t > 0:
            # Formula for zero rates: Z = -ln(DF) / T
            z = -np.log(df) / t
            zero_rates.append(z) # adds this term to the vector
        else:
            # Fallback to avoid division by zero if t=0 slips through
            zero_rates.append(0.0)

    return zero_rates


def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    """
    Given a list of discount factors, return the discount factor at a given date by linear
    interpolation.

    Parameters:
        reference_date (Union[dt.datetime, pd.Timestamp]): Reference date.
        interp_date (Union[dt.datetime, pd.Timestamp]): Date at which the discount factor is
            interpolated.
        dates (Union[List[dt.datetime], pd.DatetimeIndex]): List of dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        float: Discount factor at the interpolated date.
    """

    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")

    # if we already have that date in the list we just return the discount
    if interp_date in dates:
        # Convert dates to a list just to be safe when finding the index
        idx = list(dates).index(interp_date)
        return discount_factors[idx]
    # otherwise

    # compute relevant year fractions for available set of dates
    t_arr = [year_frac_act_x(reference_date, d, 365) for d in dates] # dates for which I have the discount
    t_interp = year_frac_act_x(reference_date, interp_date, 365) # date in which I want to find the discount factor
    # We did it only because the previous function does not return the year fractions
    
    # convert discounts into zero rates
    z_arr = from_discount_factors_to_zero_rates(t_arr, discount_factors) # use the previous function implemented

    # we remove the first date (you don't consider it)
    xp = t_arr[1:] if len(t_arr) > 1 else t_arr
    fp = z_arr[1:] if len(z_arr) > 1 else z_arr

    z_interp = np.interp(t_interp, xp, fp) # already given function to interpolate

    # convert zero rate into discount (reverting the formula above)
    discount = np.exp(-z_interp * t_interp)

    return discount


def bootstrap(
    reference_date: dt.datetime, # today
    depo: pd.DataFrame, # pandas dataframe with rates (already absolute values)
    futures: pd.DataFrame, # dataframe with BID/ASK rates + Settle and Expiry columns
    swaps: pd.DataFrame, # dataframe with rates (already absolute values)
    shock: float = 0.0,
) -> pd.Series:
    """
    Bootstrap the discount factors from the given bid/ask market data.
    - Deposit rates are used up to and including the first depo whose maturity >= settle
      of the first future (exact settle date read from futures['Settle']).
    - The first 7 futures are used, with exact settle/expiry dates read from
      futures['Settle'] and futures['Expiry'].
    - Swaps whose maturity is strictly after the expiry of the 7th future are used.

    Parameters:
        reference_date (dt.datetime): Reference date.
        depo (pd.DataFrame): Deposit rates.
        futures (pd.DataFrame): Futures rates with columns BID, ASK, Settle, Expiry.
        swaps (pd.DataFrame): Swaps rates.
        shock (Union[float, pd.Series]): Parallel shift to apply to the market rates, default to
            zero.

    Returns:
        Tuple[pd.Series, pd.Series]: Discount factors and zero rates.
    """

    termDates, discounts = [reference_date], [1.0]

    #### DEPOS

    # Exact settle of the first future, read directly from the futures DataFrame
    first_future_settle = futures.iloc[0]['Settle']

    # Include depos up to and including the first one whose maturity >= first future settle
    mask = depo.index >= first_future_settle
    if mask.any():
        first_crossing_date = depo.index[mask][0]   # first depo that reaches or exceeds the settle
        cutoff_loc = depo.index.get_loc(first_crossing_date)
        depoDates = depo.index[:cutoff_loc + 1].to_list()
    else:
        # all depos mature before the first future settle: take them all
        depoDates = depo.index.to_list()

    depoRates = depo.loc[depoDates].mean(axis=1).values
    depoRates = depoRates + (shock if isinstance(shock, float) else shock[depoDates].values)

    for d, r in zip(depoDates, depoRates):
        yf = year_frac_act_x(reference_date, d, 360)
        df = 1.0 / (1.0 + r * yf)
        termDates.append(d)
        discounts.append(df)

    #### FUTURES

    # Take the first 7 futures; Settle and Expiry are already columns of the DataFrame
    futures_7 = futures.iloc[:7]

    fwd_rates = futures_7[['BID', 'ASK']].mean(axis=1).values
    fwd_rates = fwd_rates + (shock if isinstance(shock, float) else shock[futures_7.index].values)

    for i, trade_date in enumerate(futures_7.index):
        # exact settle and expiry from the futures DataFrame columns
        t_start = futures_7.loc[trade_date, 'Settle']
        t_end   = futures_7.loc[trade_date, 'Expiry']

        yf_fwd = year_frac_act_x(t_start, t_end, 360)
        fwd_discount = 1.0 / (1.0 + fwd_rates[i] * yf_fwd)  # forward discount B(t0, t_start, t_end)

        # interpolate B(t0, t_start)
        spot_discount_start = get_discount_factor_by_zero_rates_linear_interp(
            reference_date, t_start, termDates, discounts
        )

        # B(t0, t_end) = B(t0, t_start) * B(t_start, t_end)
        spot_discount_end = spot_discount_start * fwd_discount

        termDates.append(t_end)
        discounts.append(spot_discount_end)

    #### SWAPS

    # Expiry of the 7th future: read directly from the futures DataFrame
    seventh_future_expiry = futures_7.iloc[6]['Expiry']

    # Bootstrap only swaps whose maturity is strictly after the expiry of the 7th future
    swaps_to_bootstrap = swaps[swaps.index > seventh_future_expiry]

    spot_date = business_date_offset(reference_date, day_offset=2)

    swapRates = swaps_to_bootstrap.mean(axis=1).values + (
        shock if isinstance(shock, float) else shock[swaps_to_bootstrap.index].values
    )

    for idx, swapDate in enumerate(swaps_to_bootstrap.index):
        rate = swapRates[idx]

        coupon_dates = []
        for year in range(1, 51):
            d_pay = business_date_offset(spot_date, year_offset=year)
            coupon_dates.append(d_pay)
            if d_pay >= swapDate:
                coupon_dates[-1] = swapDate
                break

        # BPV (annuity) — sum over all coupon periods except the last
        sum_annuity = 0.0
        for n in range(len(coupon_dates) - 1):
            t_prev = spot_date if n == 0 else coupon_dates[n - 1]
            t_curr = coupon_dates[n]

            yf_coupon = year_frac_30e_360(t_prev, t_curr)
            df_n = get_discount_factor_by_zero_rates_linear_interp(
                reference_date, t_curr, termDates, discounts
            )
            sum_annuity += yf_coupon * df_n

        # Final period
        t_last_prev = coupon_dates[-2] if len(coupon_dates) > 1 else spot_date
        yf_final = year_frac_30e_360(t_last_prev, swapDate)

        df = (1.0 - rate * sum_annuity) / (1.0 + rate * yf_final)

        termDates.append(swapDate)
        discounts.append(df)

    discount_factors = pd.Series(index=termDates, data=discounts)
    zero = from_discount_factors_to_zero_rates(discount_factors.index, discount_factors.values)
    zero_rates = pd.Series(index=termDates[1:], data=zero)

    return discount_factors, zero_rates
