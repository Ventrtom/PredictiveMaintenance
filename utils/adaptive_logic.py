import pandas as pd
import numpy as np
from typing import Tuple, Dict

def infer_base_frequency(series: pd.Series) -> str:
    """
    Odhadne vhodnou cílovou frekvenci na základě časových rozdílů.
    """
    deltas = series.dropna().sort_values().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return '1H'  # default

    median_delta = np.median(deltas)
    if median_delta < 60:
        return '1T'
    elif median_delta < 300:
        return '5T'
    elif median_delta < 1800:
        return '15T'
    elif median_delta < 7200:
        return '1H'
    else:
        return '3H'

def infer_best_aggregation(series: pd.Series) -> str:
    """
    Vybere vhodnou agregační metodu na základě variability dat.
    """
    if series.nunique() == 1:
        return 'last'
    std_dev = series.std()
    if std_dev < 0.1 * abs(series.mean()):
        return 'mean'
    elif std_dev < 0.3 * abs(series.mean()):
        return 'median'
    elif series.skew() > 2:
        return 'max'
    else:
        return 'mean'

def choose_resample_strategy(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Vybere vhodnou resample frekvenci a agregační metodu pro každý KPI.

    Returns:
        {kpi_id: {'freq': '15T', 'agg': 'mean'}}
    """
    strategies = {}
    grouped = df.groupby("kpi_id")
    for kpi_id, group in grouped:
        freq = infer_base_frequency(group["timestamp"])
        agg = infer_best_aggregation(group["value"])
        strategies[kpi_id] = {"freq": freq, "agg": agg}
    return strategies

def choose_imputation_method(series: pd.Series) -> str:
    """
    Rozhodne, jakou metodu imputace použít.
    """
    missing_ratio = series.isnull().sum() / len(series)
    if missing_ratio < 0.02:
        return 'none'
    elif missing_ratio < 0.1:
        return 'interpolate'
    elif series.ffill().nunique() <= 3:
        return 'mean'
    else:
        return 'ffill'

def choose_scaling_method(series: pd.Series) -> str:
    """
    Rozhodne, jakým způsobem normalizovat data.
    """
    if series.max() - series.min() < 1e-3:
        return 'none'
    elif series.skew() > 2:
        return 'robust'
    elif abs(series.mean()) > 10 * series.std():
        return 'minmax'
    else:
        return 'zscore'
