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