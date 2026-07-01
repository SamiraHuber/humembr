import psycopg
from pgvector.psycopg import register_vector

from humembr.util.config import load_config


def get_db_connection(autocommit: bool = True) -> psycopg.Connection:
    cfg = load_config()
    conn_string = cfg.db_conn_str
    conn = psycopg.connect(conn_string, autocommit=autocommit)
    register_vector(conn)
    return conn
