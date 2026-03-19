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

    effDates, effDf = dates, discount_factors
    if isinstance(effDates, pd.DatetimeIndex):   # if the input are dates, it must be converted to year fractions, if it is already year fractions, nothing to do. 
                                                 # complete the function 
        
        effDates = effDates[1:]
        effDf = discount_factors[1:]


    zero_rates = list(np.repeat(0.1,len(effDates)))   # correct formula to convert a list of discounts into zero rates
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
    
    # compute relevant yearfractions for available set of dates
    
    # convert discounts into zero rates
    
    # apply the interpolation on the target day
    
    # convert zero rate into discount
    
    discount = 0.9 
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
        shock (Union[float, pd.Series]): Parallel shift to apply to the market rates, default to
            zero.

    Returns:
        pd.Series: Discount factors.
    """

    # initialize the list of dates and discounts
    termDates, discounts = [reference_date], [1.0]

    #### DEPOS
    
    # select the correct depos and their rates
    depoDates = depo.index[0:8].to_list()    # write the correct condition to filter the dates and depo data needed
    depoRates = depo.loc[depoDates].mean(axis=1).values 

    # needed for the bumped bootstrap: if shock is a float, shift all the mkt data by that number, otherwise for each pillar its value
    depoRates = depoRates +  ( shock if isinstance(shock, float) else shock[depoDates].values )

    # convert rate L(t0,ti) to discount B(t0,ti) and append the results to the current list of dates and discounts
    termDates += depoDates
    discounts += list(np.repeat(0.9, len(depoDates) ) ) 

    
    #### FUTURES

    # select the correct futures and their rates
    futures_of_interest = futures.iloc[0:7, :]


    # convert the forward rates L(t0;ti-1, ti) to the forward discount B(t0;ti-1,ti)
    
    # compute the spot discount B(t0, ti) using the compound rule, interpolate if needed
        
    for t, rowFut in futures.iterrows(): 
        termDates += [t]
        discounts += [0.9]  
   
    #### SWAPS

    # initialize the BPV_1
    swapDate = swaps.index[0] 
    swapYearFrac, swapDisc, df = list(), list(), 0.0

    swapYearFrac.append(year_frac_act_x(reference_date, swapDate, 360))
    
    df = 0.9
    swap_old = swapDate
    swapDisc=[df]

    swapRates = swaps.mean(axis=1).values + (
        shock if isinstance(shock, float) else shock[swaps.index].values
    )
    
    for idx, swapDate in enumerate(swaps.index):
            rate, yf = swapRates[idx], year_frac_act_x(swap_old, swapDate, 360)
            swapYearFrac.append(yf)
            df = 0.9
            termDates.append(swapDate)
            discounts.append(df)

    discount_factors = pd.Series(index=termDates, data=discounts)
    zero = from_discount_factors_to_zero_rates(discount_factors.index, discount_factors.values)
    zero_rates = pd.Series(index=termDates[1:], data=zero)
    
    return discount_factors, zero_rates
