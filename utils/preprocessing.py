import pandas as pd
import numpy as np
import re
from utils.adaptive_logic import (
    choose_resample_strategy,
    choose_imputation_method,
    choose_scaling_method
)

def _clean_numeric_column(series: pd.Series) -> pd.Series:
    def parse(x):
        if isinstance(x, str):
            x = re.sub(r"\s*\(.*?\)", "", x)
        try:
            return float(x)
        except:
            return np.nan
    return series.apply(parse)

def _normalize(s: pd.Series, method: str) -> pd.Series:
    m, M = s.min(), s.max()
    if method == 'zscore':
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 0 else s - mu
    if method == 'minmax':
        span = M - m
        return (s - m) / span if span > 0 else s - m
    if method == 'robust':
        med = s.median()
        iqr = s.quantile(.75) - s.quantile(.25)
        return (s - med) / iqr if iqr > 0 else s - med
    return s

def winsorize_series(
    s: pd.Series,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95
) -> pd.Series:
    """Ořeže extrémy na dané kvantily."""
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower, upper)

def preprocess_sensor_data(df: pd.DataFrame, impute: bool = True) -> dict:
    """
    Adaptivní předzpracování senzoru:
      1) Pro každý KPI zvolí frekvenci a agregační metodu.
      2) Pro každý KPI zvolí vhodnou metodu imputace.
      3) Pro každý KPI detekuje outliery (winsorizace) a zvolí optimální škálování.
    Vrací slovník {object_id: DataFrame}.
    """
    # --- 1) Rename & type casting ---
    df2 = df.rename(columns={
        'data_timestamp': 'timestamp',
        'id_fc_object': 'object_id',
        'id_fc_kpi_definition': 'kpi_id',
        'value': 'value'
    }).copy()
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df2['value'] = _clean_numeric_column(df2['value'])

    processed_data = {}

    # --- 2) Zpracování pro každý objekt ---
    for obj_id, df_obj in df2.groupby('object_id'):
        # pivot na long form: každé KPI zvlášť
        strategies = choose_resample_strategy(df_obj)
        series_list = []

        # 2a) Adaptive resampling & aggregation per KPI
        for kpi, strat in strategies.items():
            s = (df_obj[df_obj['kpi_id'] == kpi]
                 .set_index('timestamp')['value']
                 .sort_index()
            )
            freq = strat['freq'].upper()  # např. '1T', '15T', '1H'
            agg = strat['agg']            # 'mean', 'median', 'max', 'last', ...
            s_res = s.resample(freq).agg(agg)
            s_res.name = kpi
            series_list.append(s_res)

        if not series_list:
            continue

        # spojení všech KPI podle časové osy
        df_res = pd.concat(series_list, axis=1).sort_index()

        # --- 3) Impute (dynamicky voleno per KPI) ---
        if impute:
            for col in df_res.columns:
                method = choose_imputation_method(df_res[col])
                if method == 'interpolate':
                    df_res[col] = df_res[col].interpolate(
                        method='time', limit_direction='both'
                    )
                elif method == 'mean':
                    df_res[col] = df_res[col].fillna(df_res[col].mean())
                elif method == 'ffill':
                    df_res[col] = df_res[col].ffill().bfill()
                # method == 'none' → necháme NaN

        # --- 4) Outlier handling & scaling ---
        df_scaled = pd.DataFrame(index=df_res.index)
        for col in df_res.columns:
            s = df_res[col]

            # 4a) Winsorizace
            s_win = winsorize_series(s, 0.05, 0.95)

            # 4b) Zvol normalizaci
            method = choose_scaling_method(s_win)
            if method == 'none':
                df_scaled[col] = s_win
            else:
                df_scaled[col] = _normalize(s_win, method)

        processed_data[obj_id] = df_scaled

    return processed_data
