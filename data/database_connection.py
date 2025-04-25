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
