# 🧠 Project Snapshot

Tento soubor obsahuje strukturu projektu a obsah jednotlivých souborů pro použití s AI asistenty.

## 📂 Struktura projektu

```
PredictiveMaintenance/
├── README.md
├── config
│   └── db_config.py
├── data
│   ├── data_loader.py
│   └── database_connection.py
├── main.py
├── models
│   ├── base_model.py
│   └── random_forest.py
├── notebooks
│   ├── data_connection_test.ipynb
│   └── data_filtering_test.ipynb
├── requirements.txt
└── utils
    ├── adaptive_logic.py
    ├── preprocessing.py
    ├── preprocessing2.py
    └── visualization.py
```

## 📄 Obsahy souborů


---

### `main.py`

```python

```


---

### `README.md`

```python
Projekt je určen pro vytvoření prediktivního modelu údržby na základě dat sbíraných přímo z výrobních strojů. Nasbíraná data v databázi jsou získána z reálného provozu.


Jak používat tento project:
Aktivace venv:
 .venv\Scripts\activate

Instalace knihoven z requirements.txt:
 pip install -r requirements.txt


Stav projektu a návrh dalšího postupu
Současný stav
Projekt je napojen na Oracle databázi pomocí SQLAlchemy. K dispozici jsou funkce pro načtení:

senzorových dat z tabulky FC_OBJECT_DATA

KPI definic z tabulky FC_KPI_DEFINITION

informací o zařízeních z tabulky EQUIPMENT

záznamů o údržbě z tabulky MAINTENANCE_ORDER

Data ze strojů jsou již předzpracovávána pomocí adaptivních strategií (preprocessing.py - OBSOLETE, lze smazat, preprocessing2.py). Vizualizační nástroje (visualization.py) umožňují sledovat trendy, rolling metriky, korelace a chybějící hodnoty.

Identifikované mezery a potřeby
1. Propojení senzorových dat se záznamy údržby
Maintenance order reprezentuje zásah prováděný nad konkrétním zařízením (EQUIPMENT). Každé zařízení je svázáno s FC_OBJECT, a senzorová data jsou vázána právě na id_fc_object. Pro jakoukoliv analýzu a predikci je nutné propojit senzorová data a záznamy údržby skrz tyto vazby.

Akce:

Ověřit a implementovat propojení MAINTENANCE_ORDER → EQUIPMENT → FC_OBJECT → FC_OBJECT_DATA.
Zatím nám v DB chybí data o vazbě mezi EQUIPMENT a FC_OBJECT kde je pouze jediný object ale equipmentů je hned několik. Equipmenty mají ve slouci EQUIPMENT_NUMBER uvedené kódové označení kdy teoreticky přes prefix můžeme vytvořit vazbu equipmentu a konkrétního KPI které zdá se používá stejný prefix? Ale to už je specifický case pro tento konkrétní dataset a my potřebujeme vytvořit univerzální funkci. Proto i kdybychom měli pozměnit tato aktuální data v databázi tak, že nebudou všechna KPI pod jediným objectem ale založíme FC_OBJECT pro každý EQUIPMENT a správně namapujeme FC_KPI_DEFINITION a FC_OBJECT a EQUIPMENT. Tím bychom měli získat správnou podobu datasetu.

Vytvořit funkci fetch_maintenance_with_object_ids().

2. Rozlišení typů údržby
Pro predikční účely nás zajímají především breakdown/corrective typy poruch. Preventivní údržby je třeba ze značení poruch odfiltrovat.

Akce:

Zajistit rozlišení typů v rámci dat z MAINTENANCE_ORDER (podle sloupce type ID_LISTING_VALUE=1 kde 1 reprezentuje typ breakdown, 3 corrective a 101 preventive).

Přidat filtr poruch podle typu.

3. Časová prodleva mezi poruchou a vytvořením/zahájením údržby
V reálném provozu často existuje prodleva mezi skutečným vznikem poruchy a vytvořením/zahájením MAINTENANCE_ORDER. Pro predikci je nejvhodnější orientovat se podle času vzniku nebo vytvoření záznamu, ne podle jeho zahájení.

Akce:

Pracovat s created_at jako s aproximací času poruchy.

Vytvořit časové okno (např. 60–180 minut před created_at) a označit data v tomto okně jako již potenciálně odpovídající poruše tedy že během této doby už porucha probíhá a analyzovat převážně ještě dřívější data.

Doporučený další postup
Fáze 1 – Sloučení dat
Implementovat funkci pro propojení záznamů údržby se senzory na základě equipment_id a id_fc_object.

Vytvořit funkci create_labeled_dataset(sensor_df, maintenance_df), která vytvoří trénovací dataset se sloupci featur a binárním labelem označujícím přítomnost poruchy.

Fáze 2 – Modelování
Použít preprocess_sensor_data2 pro generování vstupních featur.

Vytvořit datové matice X a y pro trénink klasifikačního modelu.

Implementovat základní model (např. RandomForest) v models/random_forest.py.

Fáze 3 – Evaluace a interpretace
Vyhodnotit model pomocí metrik jako přesnost, recall a F1-score.

Vizualizovat význam jednotlivých featur (feature importance).

Vytvořit ROC křivku a matici záměn.

Doporučení k implementaci
Vytvořit nový skript labeled_dataset.py v utils/, který bude:

Spojovat data senzorů a údržby.

Generovat binární label pro přítomnost poruchy.

Přidat notebook notebooks/model_training.ipynb pro experimenty:

Načtení dat.

Preprocessing.

Trénování modelu.

Vizualizace výsledků.
```


---

### `requirements.txt`

```python
# Práce s databází Oracle
oracledb>=1.3  # oficiální knihovna pro Oracle (náhrada za cx_Oracle)
SQLAlchemy>=2.0

# Datové modelování a validace
pydantic>=2.5

# Základní práce s daty
pandas>=2.2
numpy>=1.26

# Logování a ladění
loguru>=0.7

# .env konfigurace
python-dotenv>=1.0

# vizualizace
matplotlib
seaborn

# interaktivní testování
jupyterlab

# ML knihovny
scikit-learn
```


---

### `config\db_config.py`

```python

```


---

### `data\database_connection.py`

```python
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Načtení proměnných z .env souboru
load_dotenv()

ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE")

# Sestavení SQLAlchemy URI
DB_URI = f"oracle+oracledb://{ORACLE_USER}:{ORACLE_PASSWORD}@{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"

def get_engine():
    """Vrací SQLAlchemy engine pro připojení k databázi."""
    try:
        engine = create_engine(DB_URI, echo=False, future=True)
        print("✅ SQLAlchemy engine vytvořen.")
        return engine
    except Exception as e:
        print("❌ Chyba při vytváření SQLAlchemy engine:", e)
        return None
    
def test_db_connection():
    engine = get_engine()
    if engine:
        try:
            df = pd.read_sql("SELECT * FROM dual", engine)
            print("📦 Testovací výstup:", df)
        except Exception as e:
            print("❌ Test spojení selhal:", e)

```


---

### `data\data_loader.py`

```python
import pandas as pd
from data.database_connection import get_engine

def fetch_equipment_data():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM EQUIPMENT", engine)

def fetch_kpi_definitions():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM FC_KPI_DEFINITION", engine)

def fetch_sensor_data():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM FC_OBJECT_DATA", engine)

def fetch_maintenance_orders():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM MAINTENANCE_ORDER", engine)
    
# Rychlé testovací náhledy
def preview_equipment_data():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM EQUIPMENT FETCH FIRST 8 ROWS ONLY", engine)

def preview_kpi_definitions():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM FC_KPI_DEFINITION FETCH FIRST 8 ROWS ONLY", engine)

def preview_sensor_data():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM FC_OBJECT_DATA FETCH FIRST 8 ROWS ONLY", engine)

def preview_maintenance_orders():
    engine = get_engine()
    if engine:
        return pd.read_sql("SELECT * FROM MAINTENANCE_ORDER FETCH FIRST 8 ROWS ONLY", engine)
    
def fetch_sensor_data_filtered(object_id: int = None, kpi_ids: list = None, start_time: str = None, end_time: str = None):
    import pandas as pd
    from sqlalchemy import text

    engine = get_engine()
    if not engine:
        return None

    # Základní dotaz
    query = "SELECT * FROM FC_OBJECT_DATA WHERE 1=1"

    # Filtrování dle objektu
    if object_id is not None:
        query += f" AND id_fc_object = {object_id}"

        # Pokud nejsou kpi_ids, dynamicky je získáme z databáze
        if not kpi_ids:
            kpi_query = f"SELECT DISTINCT id_fc_kpi_definition FROM FC_OBJECT_DATA WHERE id_fc_object = {object_id}"
            kpi_ids_df = pd.read_sql(text(kpi_query), engine)
            kpi_ids = kpi_ids_df['id_fc_kpi_definition'].tolist()

    # Filtrování dle KPI
    if kpi_ids:
        formatted_ids = ", ".join(str(kpi) for kpi in kpi_ids)
        query += f" AND id_fc_kpi_definition IN ({formatted_ids})"

    # Filtrování dle časového rozsahu
    if start_time and end_time:
        query += (
            f" AND data_timestamp BETWEEN "
            f"TO_TIMESTAMP('{start_time}', 'YYYY-MM-DD HH24:MI:SS') AND "
            f"TO_TIMESTAMP('{end_time}', 'YYYY-MM-DD HH24:MI:SS')"
        )

    return pd.read_sql(text(query), engine)




def get_unique_equipment_ids():
    """Vrací unikátní ID zařízení z tabulky EQUIPMENT."""
    engine = get_engine()
    if engine:
        query = "SELECT DISTINCT ID AS equipment_id FROM EQUIPMENT ORDER BY ID"
        df = pd.read_sql(query, engine)
        return df['equipment_id'].dropna().tolist()
    return []


def get_unique_kpi_ids():
    """Vrací unikátní ID KPI z tabulky FC_OBJECT_DATA."""
    engine = get_engine()
    if engine:
        query = "SELECT DISTINCT ID_FC_KPI_DEFINITION AS kpi_id FROM FC_OBJECT_DATA ORDER BY ID_FC_KPI_DEFINITION"
        df = pd.read_sql(query, engine)
        return df['kpi_id'].dropna().tolist()
    return []

def get_unique_object_ids():
    """Vrací unikátní id_fc_object z tabulky FC_OBJECT_DATA, pokud existuje."""
    engine = get_engine()
    if engine:
        query = "SELECT DISTINCT id_fc_object FROM FC_OBJECT_DATA ORDER BY id_fc_object"
        df = pd.read_sql(query, engine)
        return pd.read_sql(query, engine)['id_fc_object'].dropna().tolist()
    return []
```


---

### `models\base_model.py`

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Equipment(BaseModel):
    id: int
    equipment_number: str
    description: Optional[str]
    serial_number: Optional[str]

class KPIDefinition(BaseModel):
    id: int
    name: str
    description: Optional[str]
    unit: Optional[str]

class SensorReading(BaseModel):
    id: int
    equipment_id: int
    kpi_id: int
    value: float
    timestamp: datetime

class MaintenanceOrder(BaseModel):
    id: int
    equipment_id: int
    type: str
    created_at: datetime
    description: Optional[str]
```


---

### `models\random_forest.py`

```python

```


---

### `notebooks\data_connection_test.ipynb`

```python
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4cbe4e0b",
   "metadata": {},
   "source": [
    "# 🔍 Test načítání dat z Oracle DB\n",
    "Tento notebook ověřuje, že funguje připojení k databázi a že lze načíst záznamy ze všech klíčových tabulek."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67531868",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import pandas as pd\n",
    "sys.path.append(os.path.abspath(\"..\"))\n",
    "from data.data_loader import (\n",
    "    preview_equipment_data,\n",
    "    preview_kpi_definitions,\n",
    "    fetch_kpi_definitions,\n",
    "    preview_sensor_data,\n",
    "    preview_maintenance_orders\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58c5631e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📄 EQUIPMENT – informace o zařízeních\n",
    "equipment_df = preview_equipment_data()\n",
    "equipment_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dde49f47",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📄 FC_KPI_DEFINITION – seznam měřených veličin\n",
    "#kpi_df = preview_kpi_definitions()\n",
    "kpi_df = fetch_kpi_definitions()\n",
    "kpi_df.head(50)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08ed11c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📄 FC_OBJECT_DATA – historická senzorová data\n",
    "sensor_df = preview_sensor_data()\n",
    "sensor_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a9d9056",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📄 MAINTENANCE_ORDER – záznamy o údržbě\n",
    "maintenance_df = preview_maintenance_orders()\n",
    "maintenance_df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

```


---

### `notebooks\data_filtering_test.ipynb`

```python
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7cb6f67e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 🧠 Predictive Maintenance – Test načítání dat s filtrem\n",
    "# 📦 Importy:\n",
    "\n",
    "import sys\n",
    "import os\n",
    "import pandas as pd\n",
    "sys.path.append(os.path.abspath(\"..\"))\n",
    "from data.database_connection import get_engine\n",
    "from data.data_loader import (\n",
    "    get_unique_equipment_ids,\n",
    "    get_unique_kpi_ids,\n",
    "    get_unique_object_ids,\n",
    "    fetch_sensor_data_filtered\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b012ef5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 1. Získání dostupných hodnot ---\n",
    "\n",
    "print(\"🔎 Načítám dostupné hodnoty z databáze...\")\n",
    "\n",
    "equipment_ids = get_unique_equipment_ids()\n",
    "kpi_ids = get_unique_kpi_ids()\n",
    "object_ids = get_unique_object_ids()\n",
    "\n",
    "print(f\"✅ {len(equipment_ids)} unikátních equipment_id\")\n",
    "print(f\"✅ {len(kpi_ids)} unikátních kpi_id\")\n",
    "print(f\"✅ {len(object_ids)} unikátních object_id\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef94b4d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 2. Ruční výběr hodnot pro testování ---\n",
    "\n",
    "# ZDE MĚŇ PARAMETRY PRO TEST\n",
    "selected_object_id = 1\n",
    "selected_kpi_id = None # nebo konkrétní KPI stylem [1,2,3] \n",
    "start_time = \"2025-01-01 00:00:00\"\n",
    "end_time = \"2025-03-01 00:00:00\"\n",
    "\n",
    "print(f\"\\n🎯 Vybrané parametry:\\n- FC OBJECT ID: {selected_object_id}\\n- KPI ID: {selected_kpi_id}\\n- Období: {start_time} – {end_time}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8470d388",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 3. Filtrování dat ---\n",
    "filtered_df = fetch_sensor_data_filtered(\n",
    "    object_id=selected_object_id,\n",
    "    kpi_ids=selected_kpi_id,\n",
    "    start_time=start_time,\n",
    "    end_time=end_time\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27df6407",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4. Náhled výsledku ---\n",
    "print(f\"\\n📊 Načteno {len(filtered_df)} řádků.\")\n",
    "filtered_df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34c07619",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Předzpracování vyžaduje sloupce: timestamp, object_id, kpi_id, value\n",
    "\n",
    "df = filtered_df.rename(columns={\n",
    "    'data_timestamp': 'timestamp',\n",
    "    'id_fc_object': 'object_id',\n",
    "    'id_fc_kpi_definition': 'kpi_id',\n",
    "    'value': 'value'\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c78c4d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Spuštění předzpracování\n",
    "from utils.preprocessing import preprocess_sensor_data\n",
    "from utils.preprocessing2 import preprocess_sensor_data2\n",
    "\n",
    "processed = preprocess_sensor_data2(df, impute=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d597748f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vezmeme první stroj (object_id) a zobrazíme jeho dataframe\n",
    "if processed:\n",
    "    first_object_id = list(processed.keys())[0]\n",
    "    print(f\"🔍 Výstup pro object_id: {first_object_id}\")\n",
    "    display(processed[first_object_id].head(5))\n",
    "else:\n",
    "    print(\"⚠️ Nebyla nalezena žádná data po zpracování.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17887340",
   "metadata": {},
   "outputs": [],
   "source": [
    "from utils.visualization import (\n",
    "    plot_kpi_raw_trends,\n",
    "    plot_kpi_rolling,\n",
    "    plot_correlation_heatmap,\n",
    "    plot_feature_distributions,\n",
    "    plot_missing_data_pattern\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22670ce2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trendy pro KPI\n",
    "plot_kpi_raw_trends(\n",
    "    df=processed,            # Vstupní data – buď slovník {object_id: DataFrame}, nebo přímo DataFrame\n",
    "    object_id=1,             # ID objektu – použije se pouze pokud je df slovník\n",
    "\n",
    "    kpi_ids=None,            # Seznam KPI, které chceš vykreslit (např. [7, 8, 23])\n",
    "                             # Pokud None, zobrazí se všechny číselné sloupce (raw i engineered)\n",
    "\n",
    "    feature_types=[          # Volitelné zúžení podle typu feature:\n",
    "        'raw'              \n",
    "        #   'raw' Originální KPI hodnoty (např. '7', '8', '23')             \n",
    "        #   'mean' Rolling průměry (např. 'mean_6')               \n",
    "        #   'std' Rolling směrodatné odchylky (např. 'std_3')       \n",
    "        #   'diff1' První diference\n",
    "        #   'diff2' Druhá diference        \n",
    "        #   'pct_change' Procentuální změna        \n",
    "        #   'time_since' Speciální sloupec 'time_since_last'\n",
    "        #   'rolling' Zástupný typ – zahrne mean + std dohromady\n",
    "    ],\n",
    "\n",
    "    agg_freq='1h',           # Agregační frekvence (resampling), např. '1min', '15t', '1h'\n",
    "                             # Použije se na časovou osu – např. 1h = průměrování po hodinách\n",
    "\n",
    "    highlight_missing=True,   # Zvýraznit chybějící hodnoty červenými značkami (např. kde je NaN)\n",
    "    start_time=None,   # Počáteční časový filtr buďto nechat \"start_time\" nebo konkrétní čas (např. \"2025-01-01 00:00:00\")\n",
    "    end_time=None       # Koncový časový filtr buďto nechat \"end_time\" nebo konkrétní čas (např. \"2025-03-01 00:00:00\")\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7e40cfc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Rolling mean s oknem 6 vzorků\n",
    "plot_kpi_rolling(\n",
    "    df=processed,                # Dict nebo DataFrame\n",
    "    object_id=1,                 # ID objektu (stroje)\n",
    "    kpi_ids=[1],                # Seznam KPI (např. [23, 24]) nebo None = všechny KPI\n",
    "    window=6,                    # Rolling okno (počet časových kroků), např. 6\n",
    "    method='mean',                # Typ výpočtu: 'mean', 'std', 'max', 'min', ...\n",
    "    start_time=start_time,       # Počáteční časový filtr\n",
    "    end_time=end_time            # Koncový časový filtr\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac41a839",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_correlation_heatmap(\n",
    "    df=processed,            # Vstupní data – buď DataFrame nebo dict ve formátu {object_id: DataFrame}\n",
    "    object_id=1,             # ID objektu – použije se jako klíč pro výběr ze slovníku (pokud je df typu dict)\n",
    "\n",
    "    kpi_ids=None,            # Volitelný výběr konkrétních KPI podle ID – např. [7, 8, 23]\n",
    "                             # Pokud None, použijí se všechny KPI (dle výběru feature types níže)\n",
    "\n",
    "    feature_types=[          # Výběr typů sloupců (feature), které se mají zahrnout do korelační matice:\n",
    "        'raw'           \n",
    "        #   'raw' Pouze původní KPI sloupce, např. '7', '8', '23'\n",
    "        #   'mean' Rolling průměr, např. '7_mean_6'\n",
    "        #   'std' Rolling směrodatná odchylka, např. '7_std_6'\n",
    "        #   'diff1' První diference, např. '7_diff1'\n",
    "        #   'diff2' Druhá diference, např. '7_diff2'\n",
    "        #   'pct_change' Procentuální změna, např. '7_pct_change'\n",
    "        #   'time_since' Speciální feature např. 'time_since_last'\n",
    "        #   'rolling' Zahrne všechny rolling features (mean_, std_), alias pro pohodlí\n",
    "    ],\n",
    "    start_time=start_time,       # Počáteční časový filtr\n",
    "    end_time=end_time            # Koncový časový filtr\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5e60be2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Histogramy dat\n",
    "plot_feature_distributions(\n",
    "    df=processed,              # Dict nebo DataFrame\n",
    "    object_id=1,               # ID objektu\n",
    "    kpi_ids=[1],              # Konkrétní KPI (např. [8, 23, 25]); None = všechny\n",
    "    raw=False,                 # True = použít neskalovaná data (pouze originální KPI)\n",
    "    log=False,                  # True = logaritmická osa pro histogram\n",
    "    start_time=start_time,       # Počáteční časový filtr\n",
    "    end_time=end_time            # Koncový časový filtr\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ab9fd29",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vizualizace chybějících dat\n",
    "plot_missing_data_pattern(\n",
    "    df=processed,              # Dict nebo DataFrame\n",
    "    object_id=1,               # ID objektu\n",
    "    n_rows=500,                 # Počet časových řádků k zobrazení (např. 500 = prvních 500 timestamps)\n",
    "    start_time=start_time,       # Počáteční časový filtr\n",
    "    end_time=end_time            # Koncový časový filtr\n",
    ")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

```


---

### `utils\adaptive_logic.py`

```python
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

```


---

### `utils\preprocessing.py`

```python
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

```


---

### `utils\preprocessing2.py`

```python
import pandas as pd
import numpy as np
import re
from utils.adaptive_logic import choose_scaling_method

def preprocess_sensor_data2(df, freq='1min', impute=True, windows=[3,6,12]):
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

```


---

### `utils\visualization.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Trend vsech KPI v jednom grafu

def plot_kpi_raw_trends(df, object_id=None, kpi_ids=None, feature_types=None, agg_freq='1H', highlight_missing=False,
                        start_time=None, end_time=None):
    if isinstance(df, dict) and object_id is not None:
        df = df.get(object_id, pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("\u26a0\ufe0f Nevalidni nebo prazdna data pro vykresleni.")
        return

    df = df.resample(agg_freq).mean().dropna(how='all')

    if start_time or end_time:
        df = df.loc[start_time:end_time]

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
        selected = [col for col in selected if any(k in str(col) for k in kpi_ids_str)]

    if not selected:
        print("\u26a0\ufe0f Zadny KPI odpovidajici vyberu.")
        return

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
        print("\u26a0\ufe0f Nevalidni nebo prazdna data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    available = df.select_dtypes(include=[np.number]).columns
    selected = kpi_ids if kpi_ids is not None else available
    selected = [col for col in selected if col in available]

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
        print("\u26a0\ufe0f Nevalidni nebo prazdna data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        print("\u26a0\ufe0f Zadna ciselna data k dispozici.")
        return

    feature_types = feature_types or ['raw', 'mean', 'std', 'diff1', 'diff2', 'pct_change', 'time_since']
    selected_cols = []

    for col in df_numeric.columns:
        col_str = str(col)
        if 'raw' in feature_types and col_str.isdigit():
            selected_cols.append(col)
        elif 'rolling' in feature_types and any(x in col_str for x in ['mean_', 'std_']):
            selected_cols.append(col)
        elif any(ftype in col_str for ftype in feature_types if ftype not in ['raw', 'rolling']):
            selected_cols.append(col)

    if kpi_ids:
        kpi_ids_str = [str(k) for k in kpi_ids]
        selected_cols = [col for col in selected_cols if any(k in str(col) for k in kpi_ids_str)]

    if len(selected_cols) < 2:
        print("\u26a0\ufe0f Nedostatecny pocet sloupcu pro korelaci.")
        return

    corr = df_numeric[selected_cols].corr()
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
        print("\u26a0\ufe0f Nevalidni nebo prazdna data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    df_plot = df.copy()
    if raw:
        df_plot = df_plot[[col for col in df_plot.columns if str(col).isdigit()]]

    if kpi_ids:
        df_plot = df_plot[[col for col in kpi_ids if col in df_plot.columns]]

    selected = df_plot.select_dtypes(include=[np.number]).columns
    if selected.empty:
        print("\u26a0\ufe0f Zadne ciselne sloupce.")
        return

    for col in selected:
        series = df_plot[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            print(f"\u26a0\ufe0f Sloupec {col} obsahuje pouze NaN/inf – preskoceno.")
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
        print("\u26a0\ufe0f Nevalidni nebo prazdna data.")
        return

    if start_time or end_time:
        df = df.loc[start_time:end_time]

    df_cut = df.iloc[:n_rows].isna()
    if df_cut.empty:
        print("\u26a0\ufe0f Zadne data pro vykresleni chyb.")
        return

    plt.figure(figsize=(12, 6))
    sns.heatmap(df_cut.T, cbar=False, cmap='viridis', xticklabels=False)
    plt.title("Missing data pattern (first rows)")
    plt.xlabel("Time steps")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

```
