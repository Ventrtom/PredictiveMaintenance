import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Společná pomocná funkce pro filtraci KPI ---
def _filter_kpi_columns(df, kpi_ids=None, feature_types=None):
    feature_types = feature_types or ['raw', 'mean', 'std', 'diff1', 'diff2', 'pct_change', 'time_since', 'rolling']
    available_cols = df.select_dtypes(include=[np.number]).columns
    selected = []

    for col in available_cols:
        col_str = str(col)
        if 'raw' in feature_types and col_str.isdigit():
            selected.append(col)
        elif 'rolling' in feature_types and any(x in col_str for x in ['mean_', 'std_']):
            selected.append(col)
        elif any(ftype in col_str for ftype in feature_types if ftype not in ['raw', 'rolling']):
            selected.append(col)

    if kpi_ids:
        kpi_ids_str = [str(k) for k in kpi_ids]
        selected_filtered = []
        for col in selected:
            col_str = str(col)
            for k in kpi_ids_str:
                if ('raw' in feature_types and col_str == k) or (k in col_str and not col_str.isdigit()):
                    selected_filtered.append(col)
                    break
        selected = selected_filtered

    return selected

# 1. Trend vsech KPI v jednom grafu
def plot_kpi_raw_trends(df, object_id=None, kpi_ids=None, feature_types=None, agg_freq='1H',
                        highlight_missing=False, start_time=None, end_time=None):
    if isinstance(df, dict) and object_id is not None:
        df = df.get(object_id, pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("\u26a0\ufe0f Nevalidní nebo prázdná data pro vykreslení.")
        return

    df = df.resample(agg_freq).mean().dropna(how='all')

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    selected = _filter_kpi_columns(df, kpi_ids, feature_types)

    if not selected:
        print("\u26a0\ufe0f Žádné KPI neodpovídá výběru.")
        return

    print(f"\u2705 Vykresluji následující KPI: {selected}")

    plt.figure(figsize=(14, 6))
    for col in selected:
        series = df[col]
        plt.plot(series.index, series, label=col)
        if highlight_missing:
            nan_mask = series.isna()
            plt.plot(series.index[nan_mask], [np.nan]*nan_mask.sum(), 'rx', alpha=0.2)

    plt.title(f"KPI Trends (object {object_id})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 2. Rolling trend - samostatne grafy
def plot_kpi_rolling(df, object_id=None, kpi_ids=None, window=6, method='mean',
                     start_time=None, end_time=None):
    if isinstance(df, dict) and object_id is not None:
        df = df.get(object_id, pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("\u26a0\ufe0f Nevalidní nebo prázdná data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    available = df.select_dtypes(include=[np.number]).columns
    selected = kpi_ids if kpi_ids is not None else available
    selected = [col for col in selected if col in available]

    if not selected:
        print("\u26a0\ufe0f Žádné KPI k dispozici pro rolling výpočet.")
        return

    print(f"\u2705 Rolling výpočet pro KPI: {selected}")

    for col in selected:
        series = df[col]
        roll_series = getattr(series.rolling(window=window, min_periods=1), method)()

        plt.figure(figsize=(12, 5))
        plt.plot(df.index, series, label=f'{col} (raw)', alpha=0.5)
        plt.plot(df.index, roll_series, label=f'{col} ({method}, w={window})', linewidth=2)
        plt.title(f"Rolling {method} of KPI {col} (object {object_id})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

# 3. Korelacni matice
def plot_correlation_heatmap(df, object_id=None, kpi_ids=None, feature_types=None,
                             start_time=None, end_time=None):
    if isinstance(df, dict) and object_id is not None:
        df = df.get(object_id, pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("\u26a0\ufe0f Nevalidní nebo prázdná data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    selected = _filter_kpi_columns(df, kpi_ids, feature_types)

    if len(selected) < 2:
        print("\u26a0\ufe0f Nedostatečný počet sloupců pro korelaci.")
        return

    corr = df[selected].corr()

    print(f"\u2705 Výběr KPI pro korelační matici: {selected}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(f"Correlation heatmap for object {object_id}")
    plt.tight_layout()
    plt.show()

# 4. Histogramy
def plot_feature_distributions(df, object_id=None, kpi_ids=None, raw=False, log=False,
                               start_time=None, end_time=None):
    if isinstance(df, dict) and object_id is not None:
        df = df.get(object_id, pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("\u26a0\ufe0f Nevalidní nebo prázdná data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    df_plot = df.copy()
    if raw:
        df_plot = df_plot[[col for col in df_plot.columns if str(col).isdigit()]]

    if kpi_ids:
        df_plot = df_plot[[col for col in df_plot.columns if col in kpi_ids]]

    selected = df_plot.select_dtypes(include=[np.number]).columns
    if selected.empty:
        print("\u26a0\ufe0f Žádné číselné sloupce.")
        return

    print(f"\u2705 Histogramy budou vykresleny pro KPI: {selected}")

    for col in selected:
        series = df_plot[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            print(f"\u26a0\ufe0f Sloupec {col} obsahuje pouze NaN/inf – přeskočeno.")
            continue

        plt.figure(figsize=(8, 4))
        series.hist(bins=30, log=log)
        plt.title(f"Histogram of {col} {'(raw)' if raw else ''}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# 5. Missing data pattern
def plot_missing_data_pattern(df, object_id=None, n_rows=500,
                              start_time=None, end_time=None):
    if isinstance(df, dict) and object_id is not None:
        df = df.get(object_id, pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("\u26a0\ufe0f Nevalidní nebo prázdná data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    df_cut = df.iloc[:n_rows].isna()
    if df_cut.empty:
        print("\u26a0\ufe0f Žádná data pro vykreslení chyb.")
        return

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_cut.T, cbar=False, cmap='viridis', xticklabels=False)
    plt.title("Missing data pattern (first rows)")
    plt.xlabel("Time steps")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
