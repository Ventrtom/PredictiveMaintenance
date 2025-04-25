# preprocessing.py

import pandas as pd
import numpy as np
import re
from utils.adaptive_logic import (
    choose_imputation_method,
    choose_resample_strategy,
    choose_scaling_method
)

def clean_numeric_column(series):
    """
    Vyčistí a převede sloupec na numerické hodnoty.
    Odstraní např. komentáře v závorkách ("81 (expected)") a převede na float.
    """
    def parse(val):
        if isinstance(val, str):
            val = re.sub(r"\s*\(.*?\)", "", val)
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan
    return series.apply(parse)

def preprocess_sensor_data(df_sensor, impute=True):
    """
    Plně adaptivní předzpracování: strategie resamplingu, imputace a škálování na základě dat.

    Args:
        df_sensor (DataFrame): surová data se sloupci ['timestamp', 'object_id', 'kpi_id', 'value']
        impute (bool): zda doplňovat chybějící hodnoty

    Returns:
        Dict[object_id, DataFrame]: předzpracovaná data pro každý objekt zvlášť
    """
    df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
    df_sensor['value'] = clean_numeric_column(df_sensor['value'])

    required_cols = {'timestamp', 'object_id', 'kpi_id', 'value'}
    missing_cols = required_cols - set(df_sensor.columns)
    if missing_cols:
        raise ValueError(f"❌ Chybí sloupce: {missing_cols}")

    df_pivot = pivot_sensor_data(df_sensor)

    processed_data = {}
    for machine_id, df_machine in df_pivot.groupby(level=0):
        df_machine = df_machine.copy()
        print(f"🛠️ Zpracovávám objekt {machine_id}")

        # Získání strategie pro každý KPI (freq, agg)
        resample_strategies = choose_resample_strategy(
            df_sensor[df_sensor['object_id'] == machine_id]
        )

        # 1) Odstraníme úroveň object_id z indexu:
        df_machine = df_machine.reset_index(level='object_id', drop=True)

        # 2) Ujistíme se, že index nese jméno 'timestamp'
        df_machine.index.name = 'timestamp'

        # 3) Pokud potřebujete sloupec object_id pro další zpracování:
        df_machine['object_id'] = machine_id


        global_start = df_machine.index.min()
        global_end = df_machine.index.max()
        start = global_start.floor('min')
        end   = global_end.ceil('min')
        df_resampled = pd.DataFrame(index=pd.date_range(start, end, freq='min'))

        for kpi_id in df_machine.columns:
            if kpi_id == 'object_id':
                continue

            strategy = resample_strategies.get(kpi_id, {'freq': '1h', 'agg': 'mean'})
            print(f"📏 KPI {kpi_id}: freq={strategy['freq']}, agg={strategy['agg']}")

            series = df_machine[kpi_id].resample(strategy['freq']).agg(strategy['agg'])

            if impute:
                method = choose_imputation_method(series)
                series = handle_missing_values(series, method)

            method = choose_scaling_method(series)
            series = normalize_sensor_data(series, method)

            df_resampled[kpi_id] = series

        df_final = generate_features(df_resampled)
        processed_data[machine_id] = df_final

    return processed_data

def handle_missing_values(series, method='ffill'):
    """Zpracuje chybějící hodnoty flexibilně podle zvolené metody."""
    if method == 'none':
        return series
    elif method == 'ffill':
        return series.ffill()
    elif method == 'bfill':
        return series.bfill()
    elif method == 'mean':
        return series.fillna(series.mean())
    elif method == 'interpolate':
        return series.interpolate()
    else:
        return series.fillna(method='ffill')

def normalize_sensor_data(series, method='zscore'):
    """Standardizuje data (Z-score, MinMax, Robust...) podle zvolené metody."""
    if method == 'none':
        return series

    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        # Pokud je rozptyl nulový nebo NaN, vrátíme konstantní řadu (např. nulovou)
        if std == 0 or np.isnan(std):
            return series - mean  # nebo: return series.fillna(0)
        return (series - mean) / std

    elif method == 'minmax':
        min_ = series.min()
        max_ = series.max()
        span = max_ - min_
        if span == 0 or np.isnan(span):
            return series - min_
        return (series - min_) / span

    elif method == 'robust':
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        if iqr == 0 or np.isnan(iqr):
            return series - median
        return (series - median) / iqr

    else:
        # fallback na žádnou změnu
        return series

def pivot_sensor_data(df_sensor):
    """
    Převede data do wide-formy (čas × KPI × stroj) pro ML.

    Předpokládá sloupce: ['timestamp', 'object_id', 'kpi_id', 'value']
    """
    df = df_sensor.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index(['object_id', 'timestamp', 'kpi_id'], inplace=True)
    df = df.unstack('kpi_id')['value']
    return df

def generate_features(df, window_sizes=[3, 6, 12]):
    """
    Vypočítá odvozené časové charakteristiky (rolling mean, std...).

    Args:
        df (DataFrame): DataFrame s resamplovanými daty.
        window_sizes (List[int]): Velikosti oken pro rolling metriky.

    Returns:
        DataFrame: Data s přidanými featurami.
    """
    df_feat = df.copy()
    for window in window_sizes:
        for col in df.columns:
            df_feat[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
            df_feat[f'{col}_std_{window}'] = df[col].rolling(window).std()
    return df_feat
