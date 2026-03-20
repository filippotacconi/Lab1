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
    Bootstrap the discount factors from the given bid/ask market data. Deposit rates are used until
    the first future settlement date (included), futures rates are used until the 2y-swap settlement.

    Parameters:
        reference_date (dt.datetime): Reference date.
        depo (pd.DataFrame): Deposit rates.
        futures (pd.DataFrame): Futures rates.
        swaps (pd.DataFrame): Swaps rates.
        shock (Union[float, pd.Series]): Shift to apply to the market rates in decimal (e.g. 1e-4 = 1bp).
            Default to zero.

    Returns:
        pd.Series: Discount factors.
    """

    # initialize the list of terminal dates and discounts
    termDates, discounts = [reference_date], [1.0]

    #### DEPOS
    
    # select the correct depos and their rates
    first_future_settle = futures.index[0] # quotation date of the first future
    first_future_settle =  business_date_offset(first_future_settle,0, 0, day_offset = 2) #settle = quotation + 2 business day
    
    # We make a list of the depo dates we will use for bootstrapping
    depoDates=depo[depo.index <= first_future_settle].index.to_list()
    # We make an array of the depo rates needed 
    depoRates = depo.loc[depoDates].mean(axis=1).values 

    # We convert the deporates in decimal values FIRST
    depoRates = depoRates / 100.0

    # THEN apply the shock (already in decimal: 1e-4 = 1bp)
    depoRates = depoRates + (shock if isinstance(shock, float) else shock[depoDates].values)

    # convert rate L(t0,ti) to discount B(t0,ti) and append the results to the current list of dates and discounts
    
    termDates.extend(depoDates) # We store the terminal dates
    
    y_fr_depo = [year_frac_act_x(reference_date, d, 360) for d in depoDates ] #We compute the year fractions and make an array by list comprehension

    new_depo_disc = [1/(1+y_fr*d_rate) for y_fr,d_rate in zip(y_fr_depo, depoRates)]
    
    discounts.extend(new_depo_disc)
    
    
    #### FUTURES

    # select the correct futures and their rates
    swap_2y_date = swaps.index[1]  # maturity date of the 2y swap
    future_dates = futures.index
    
    # We convert in settlement date the future dates:
    future_settle = pd.DatetimeIndex([business_date_offset(d,day_offset= 2) for d in futures.index])

    # We store the expiry of the futures as a pandas index:
    future_expiry = pd.DatetimeIndex([business_date_offset(m,month_offset= 3) for m in future_settle])
    
    # We keep only futures whose settle date is before or equal to the 2y swap maturity
    
    mask = future_expiry <= swap_2y_date # We impose the limit on the coverage of futures
    future_settle = future_settle[mask]
    future_expiry = future_expiry[mask]
    future_dates = future_dates[mask]
    
    # We access to the prices of the futures to get the associated forward Rates:

    Prices = futures.loc[future_dates,['BID','ASK']].mean(axis=1).values 
    
    # Futures price = 100 - fwd rate → fwd rate = (100 - Price) / 100
    fwdRates = (100 - Prices) / 100
    
    # Apply the shock directly (already in decimal, no /100)
    fwdRates = fwdRates + (shock if isinstance(shock, float) else shock[future_dates].values)

    y_fr_future = [year_frac_act_x(fut_set, fut_ex, 360) for fut_set, fut_ex in zip(future_settle,future_expiry)]
    
    # We make a cycle to compute, from the fwd rate (ti-1,ti) the discount factors at ti:
    for t_start,t_end,fwd_rate,y_fr in zip(future_settle,future_expiry,fwdRates,y_fr_future): 
         
        # convert the forward rates L(t0;ti-1, ti) to the forward discount B(t0;ti-1,ti)
        fwd_discount = 1/(1+fwd_rate*y_fr)

        # We search for the discount B(t0;ti-1) in the already bootstrapped discounts. 
        # If necessary, we interpolate:

        if t_start in termDates:
            discounts_start = discounts[termDates.index(t_start)]
        else:
            discounts_start = get_discount_factor_by_zero_rates_linear_interp(reference_date, t_start, termDates, discounts)
        
        # We compute B(t0;ti) = B(t0;ti-1,ti)*B(t0;ti-1)
        discount_end= discounts_start*fwd_discount

        #We store the terminal dates and the associated discount factors
        termDates.append(t_end)
        discounts.append(discount_end)

    
   
    #### SWAPS

    # According to lecture notes: "first swap I should consider is the second"
    # "Futures rates are used until the 2y-swap settlement"
    # We filter swaps starting from the 2y-swap, as the expiry date of the last future could not cover the 2y-swap expiry
    #### SWAPS

    swap_2y_date = swaps.index[1]
    swaps_to_bootstrap = swaps[swaps.index >= swap_2y_date] # We take into account from the 2y swap

    swapDate = swaps_to_bootstrap.index[0] # We initialize the list
    spot_date = reference_date #The reference date is already the settlement date

    # Convert to decimal FIRST
    swapRates = swaps_to_bootstrap.mean(axis=1).values / 100.0

    # THEN apply the shock (already in decimal: 1e-4 = 1bp)
    swapRates = swapRates + (shock if isinstance(shock, float) else shock[swaps_to_bootstrap.index].values)

    # We make a cycle on the swaps:
    # enumerate returns both the index and the value of each element in the iterable,
    # we need idx to access the corresponding swap rate in swapRates

    for idx, swapDate in enumerate(swaps_to_bootstrap.index):
        rate = swapRates[idx]

        coupon_dates = [] # We initialize the list coupon dates

        # We compute the coupon dates and store in a list:

        for year in range(1, 51):
            d_pay = business_date_offset(spot_date, year_offset=year)
            coupon_dates.append(d_pay)
            if d_pay >= swapDate: # If we are above the swap maturity, we stop storing coupon dates
                break

            BPV = 0.0 # We initialize the Basis Point Value (t0)
        for n in range(len(coupon_dates) - 1):
            t_prev = spot_date if n == 0 else coupon_dates[n-1] # ti-1
            t_curr = coupon_dates[n] #ti

            yf_coupon = year_frac_30e_360(t_prev, t_curr) # year fraction between ti-1 and ti

            # We search for discount factors in ti
            # If not already computed, we interpolate
            if t_curr in termDates:
                df_n = discounts[termDates.index(t_curr)]
            else:
                df_n = get_discount_factor_by_zero_rates_linear_interp(
                    reference_date, t_curr, termDates, discounts
                )

            BPV += yf_coupon * df_n #BPV = sum(year fraction(ti-1,ti)*B(t0,ti))
            #This will be equal to BPV(t0,tN-1)

            t_last_prev = coupon_dates[-2] # We take the coupon date tN-1
            yf_final = year_frac_30e_360(t_last_prev, swapDate) #We compute the year fraction (tN-1,tN)

            # We make a control: it can happen that the 2y-swap maturity has been covered 
        # by the futures bootstrapping, so we check: if it has happened so, we skip the iteration:
        
        if swapDate > termDates [-1]:
        
            # Bootstrap formula: B(t0,TN) = (1 - R * BPV) / (1 + R * yf_N)
            # derived from par swap condition: 1 = R * sum(yf_i * B_i) + B_N

            df = (1.0 - rate * BPV) / (1.0 + rate * yf_final)
            
            #We store terminal dates and discount factors

            termDates.append(swapDate)
            discounts.append(df)
        
        
    # Output values are both discount factors and zero rates:

    discount_factors = pd.Series(index=termDates, data=discounts)
    
    zero = from_discount_factors_to_zero_rates(discount_factors.index, discount_factors.values)
    zero_rates = pd.Series(index=termDates[1:], data=zero)
    
    return discount_factors, zero_rates