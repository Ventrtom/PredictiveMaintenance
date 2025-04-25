import pandas as pd
import numpy as np
import re
from utils.adaptive_logic import choose_scaling_method

def preprocess_sensor_data(df, freq='1min', impute=True, windows=[3,6,12]):
    """
    Adaptivní předzpracování senzoru pro více objektů.

    Args:
        df (pd.DataFrame): DataFrame se sloupci ['timestamp','object_id','kpi_id','value']
        freq (str): frekvence resamplingu (např. '1min')
        impute (bool): zda imputovat chybějící hodnoty
        windows (List[int]): velikosti oken pro rolling statistiky

    Returns:
        dict: { object_id: DataFrame } s předzpracovanými daty (featuremi) pro každý objekt
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
        if df_obj['kpi_id'].nunique() == 0:
            print(f"⚠️ Objekt {obj_id} nemá žádné KPI data.")
            continue
        # Pivot do wide formy
        df_wide = df_obj.set_index(['timestamp', 'kpi_id'])['value'] \
                        .unstack('kpi_id') \
                        .sort_index()

        # --- 3) Resample ---
        df_res = df_wide.resample(freq).asfreq()

        # --- 4) Impute ---
        if impute:
            df_res = df_res.interpolate(method='time', limit_direction='both')
            df_res = df_res.ffill().bfill()

        # --- 5) Drop konstantní KPI ---
        variances = df_res.var()
        zero_var = variances[variances == 0].index.tolist()
        if zero_var:
            print(f"⚠️ Dropping constant KPIs for object {obj_id}: {zero_var}")
            df_res = df_res.drop(columns=zero_var)

        # --- 6) Scale ---
        df_scaled = pd.DataFrame(index=df_res.index)
        for col in df_res.columns:
            method = choose_scaling_method(df_res[col])
            if method == 'none':
                df_scaled[col] = df_res[col]
            else:
                df_scaled[col] = _normalize(df_res[col], method)

        # --- 7) Feature engineering (optimalizováno) ---
        feature_blocks = [df_scaled]  # základní škálovaná data

        # 7a) Rolling mean/std
        for w in windows:
            rolling_means = df_scaled.rolling(w, min_periods=1).mean().add_suffix(f"_mean_{w}")
            rolling_stds = df_scaled.rolling(w, min_periods=1).std().add_suffix(f"_std_{w}")
            feature_blocks.extend([rolling_means, rolling_stds])

        # 7b) Diference a procentuální změna
        diffs1 = df_scaled.diff(1).fillna(0).add_suffix("_diff1")
        diffs2 = df_scaled.diff(2).fillna(0).add_suffix("_diff2")
        pct_changes = df_scaled.pct_change().fillna(0).add_suffix("_pct_change")
        feature_blocks.extend([diffs1, diffs2, pct_changes])

        # 7c) Čas od posledního měření (proxy pomocí první KPI)
        last_obs = df_wide.notna().cumsum(axis=0)
        time_since = (last_obs != last_obs.shift(1)).cumsum(axis=0)
        if time_since.shape[1] > 0:
            time_since_last = time_since.iloc[:, 0].rename("time_since_last")
        else:
            time_since_last = pd.Series(0, index=df_scaled.index, name="time_since_last")
        feature_blocks.append(time_since_last)

        # Finální spojení featur
        df_feat = pd.concat(feature_blocks, axis=1)

        processed_data[obj_id] = df_feat

    return processed_data


def _clean_numeric_column(series):
    def parse(x):
        if isinstance(x, str):
            x = re.sub(r"\s*\(.*?\)", "", x)
        try:
            return float(x)
        except:
            return np.nan
    return series.apply(parse)


def _normalize(s, method):
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
