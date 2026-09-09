"""Conexión compartida a PostgreSQL/pgvector."""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

EMBEDDING_DIM = 384


def connect(register: bool = True):
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "parachute_faqs"),
        user=os.getenv("POSTGRES_USER", "parachute"),
        password=os.getenv("POSTGRES_PASSWORD", "parachute"),
    )
    if register:
        register_vector(conn)
    return conn


def get_connection():
    """Conexión con el tipo `vector` ya registrado (requiere que la extensión exista)."""
    return connect(register=True)
