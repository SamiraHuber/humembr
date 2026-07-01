from dataclasses import dataclass

from psycopg.rows import dict_row

from humembr.db.repositories.util import get_db_connection


@dataclass
class RobotStatus:
    id: int
    status: str


def up_status():
    insert_status("UP")


def down_status():
    insert_status("DOWN")


def insert_status(status: str):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("INSERT INTO robot_status (status) VALUES (%s)", (status,))
    conn.close()


def get_latest_status() -> RobotStatus:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT id, status FROM robot_status ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            return RobotStatus(id=0, status="DOWN")

        return RobotStatus(**row)
