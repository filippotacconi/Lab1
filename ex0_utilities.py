"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 0: Discount Factors Bootstrap
"""

import numpy as np
import pandas as pd
import datetime as dt
from date_functions import (
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

    
    effDates, effDf = dates, np.array(list(discount_factors), dtype=float)

    # We made a control on the object of the list:
    # The input could be a timestamp (from pandas) or a datetime (from datetime library). If so, we need to convert it 
    # to year fraction with respect to the reference date we want by which we compute the zero rate

    if len(effDates) > 0 and isinstance(effDates[0], (dt.datetime, pd.Timestamp)):  
        reference_date = effDates[0]
        
        # We cut out t0 (reference date) since B(t0,t0)=1 and zero rate at t0 is undefined

        effDates = effDates[1:] 
        effDf = effDf[1:]       
        
        # We create an array of the year fractions with a list comprehension
        effDates = np.array([year_frac_act_x(reference_date, d, 365) for d in effDates], dtype=float)
    else:
        # In this case, the input dates are already expressed in year fractions
        # We cut out t0 (reference date) since B(t0,t0)=1 and zero rate at t0 is undefined
        
        effDates = np.array(list(dates[1:]), dtype=float)  
        effDf    = effDf[1:]                                 


    # Continuous compounding: B(t0,t) = exp(-z*t) → z = -log(B)/t
    zero_rates = list(-np.log(effDf) / effDates)
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

    # We check that for each input date and there exist an input discount factor
    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")
    
    # We compute relevant yearfractions for available set of dates
    
    # We use ACT/365 to compute yearfractions of the input data and create an array of them,

    y_fr_dates = np.array([year_frac_act_x(reference_date, d, 365) for d in dates]) 
    y_fr_interp = year_frac_act_x(reference_date,  interp_date,365) # year fraction related to the date
    
    zero_rates  = from_discount_factors_to_zero_rates(y_fr_dates, discount_factors) 
   
    # from_discount_factors_to_zero_rates excludes the zero rate at the reference date, so we need to cut the first y_fr_dates, which
    # refers to the reference date
    z_rate_int = np.interp(y_fr_interp, y_fr_dates[1:], zero_rates, right=zero_rates[-1])
    # convert zero rate into discount

    discount = np.exp(-z_rate_int*y_fr_interp)
     
    return discount 

def bootstrap(
    reference_date: dt.datetime,
    depo: pd.DataFrame,
    futures: pd.DataFrame,
    swaps: pd.DataFrame,
    shock: float = 0.0,
) -> pd.Series:
    """
    Bootstrap discount factors from bid/ask market data.
    - Deposits cover up to (and including) the first future settlement date.
    - Futures cover from the first future settlement up to the 2y swap maturity.
    - Swaps cover from the 2y maturity onward.

    Parameters:
        reference_date (dt.datetime): Reference date.
        depo (pd.DataFrame): Deposit rates.
        futures (pd.DataFrame): Futures rates.
        swaps (pd.DataFrame): Swap rates.
        shock (Union[float, pd.Series]): Rate shift in decimal (e.g. 1e-4 = 1bp). Defaults to 0.
    Returns:
        pd.Series: Discount factors and zero rates.
    """
    termDates, discounts = [reference_date], [1.0]

    #### DEPOS
    # Deposits are used up to the first future settlement (quotation date + 2 business days)
    first_future_settle = business_date_offset(futures.index[0], 0, 0, day_offset=2)
    depoDates = depo[depo.index <= first_future_settle].index.to_list()

    # Convert to decimal then apply shock
    depoRates = depo.loc[depoDates].mean(axis=1).values / 100.0
    depoRates = depoRates + (shock if isinstance(shock, float) else shock[depoDates].values)

    # B(t0, ti) = 1 / (1 + yf * L) for each deposit
    y_fr_depo = [year_frac_act_x(reference_date, d, 360) for d in depoDates]
    new_depo_disc = [1 / (1 + yf * r) for yf, r in zip(y_fr_depo, depoRates)]
    termDates.extend(depoDates)
    discounts.extend(new_depo_disc)

    #### FUTURES
    # Futures are used from first settlement up to the 2y swap maturity
    swap_2y_date = swaps.index[1]
    future_dates = futures.index
    future_settle = pd.DatetimeIndex([business_date_offset(d, day_offset=2) for d in futures.index])
    future_expiry = pd.DatetimeIndex([business_date_offset(m, month_offset=3) for m in future_settle])

    # Drop futures whose expiry exceeds the 2y swap
    mask = future_expiry <= swap_2y_date
    future_settle = future_settle[mask]
    future_expiry = future_expiry[mask]
    future_dates  = future_dates[mask]

    # Price → forward rate (already in decimal, shock applied directly)
    Prices   = futures.loc[future_dates, ['BID', 'ASK']].mean(axis=1).values
    fwdRates = (100 - Prices) / 100
    fwdRates = fwdRates + (shock if isinstance(shock, float) else shock[future_dates].values)

    y_fr_future = [year_frac_act_x(s, e, 360) for s, e in zip(future_settle, future_expiry)]

    # Chain rule: B(t0, ti) = B(t0, ti-1) * B(t0; ti-1, ti)
    for t_start, t_end, fwd_rate, y_fr in zip(future_settle, future_expiry, fwdRates, y_fr_future):
        fwd_discount = 1 / (1 + fwd_rate * y_fr)
        if t_start in termDates:
            discounts_start = discounts[termDates.index(t_start)]
        else:
            discounts_start = get_discount_factor_by_zero_rates_linear_interp(
                reference_date, t_start, termDates, discounts
            )
        termDates.append(t_end)
        discounts.append(discounts_start * fwd_discount)

    #### SWAPS
    # Start from the 2y swap; earlier maturities are already covered by futures.
    # If the 2y maturity itself was reached by the futures chain, that iteration is skipped below.
    swaps_to_bootstrap = swaps[swaps.index >= swap_2y_date]
    spot_date = reference_date

    # Convert to decimal then apply shock
    swapRates = swaps_to_bootstrap.mean(axis=1).values / 100.0
    swapRates = swapRates + (shock if isinstance(shock, float) else shock[swaps_to_bootstrap.index].values)

    for idx, swapDate in enumerate(swaps_to_bootstrap.index):
        rate = swapRates[idx]

        # Build annual coupon dates up to swap maturity
        coupon_dates = []
        for year in range(1, 51):
            d_pay = business_date_offset(spot_date, year_offset=year)
            coupon_dates.append(d_pay)
            if d_pay >= swapDate:
                break

        # BPV = Σ yf(ti-1, ti) * B(t0, ti)  over all but the final coupon
        BPV = 0.0
        for n in range(len(coupon_dates) - 1):
            t_prev = spot_date if n == 0 else coupon_dates[n - 1]
            t_curr = coupon_dates[n]
            yf_coupon = year_frac_30e_360(t_prev, t_curr)
            if t_curr in termDates:
                df_n = discounts[termDates.index(t_curr)]
            else:
                df_n = get_discount_factor_by_zero_rates_linear_interp(
                    reference_date, t_curr, termDates, discounts
                )
            BPV += yf_coupon * df_n

        # Final period: tN-1 → swapDate
        yf_final = year_frac_30e_360(coupon_dates[-2], swapDate)

        # Skip if this maturity was already bootstrapped via futures
        if swapDate > termDates[-1]:
            # Par condition: 1 = R * BPV(tN-1) + B(t0,tN) * (1 + R * yf_N)
            # → B(t0, tN) = (1 - R * BPV) / (1 + R * yf_N)
            df = (1.0 - rate * BPV) / (1.0 + rate * yf_final)
            termDates.append(swapDate)
            discounts.append(df)

    discount_factors = pd.Series(index=termDates, data=discounts)
    zero = from_discount_factors_to_zero_rates(discount_factors.index, discount_factors.values)
    zero_rates = pd.Series(index=termDates[1:], data=zero)
    return discount_factors, zero_rates