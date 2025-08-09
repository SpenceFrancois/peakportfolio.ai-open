# b_backtest.py
import numpy as np
import pandas as pd
from typing import Union

def portfolio_returns(returns: pd.DataFrame, weights: Union[np.ndarray, list]) -> pd.Series:
    """
    Dot‐product of asset returns and weight vector → portfolio return series.
    """
    w = np.array(weights)
    return returns.dot(w)

def cumulative_growth(r: pd.Series) -> pd.Series:
    """
    1) (1 + r).cumprod() − 1  
    2) force first value to zero  
    """
    cum = (1 + r).cumprod() - 1
    cum.iloc[0] = 0.0
    return cum

def extend_end_date(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Compute calendar offset from inferred freq or last two dates.
    """
    freq = pd.infer_freq(dates)
    offset = pd.tseries.frequencies.to_offset(freq) if freq else dates[-1] - dates[-2]
    return dates.append(dates[-1:] + offset)
