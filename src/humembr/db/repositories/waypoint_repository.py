from dataclasses import dataclass

import numpy as np
from psycopg.rows import dict_row

from humembr.db.repositories.util import get_db_connection


@dataclass
class Waypoint:
    id: str
    name: str
    coords: np.ndarray


def clear_waypoints():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM waypoints")
        cursor.execute("DELETE FROM rooms")
    conn.close()


def instert_room(name: str) -> int:
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO rooms (name)
            VALUES (%s)
            RETURNING id
            """,
            (name,),
        )
        room_id = cursor.fetchone()[0]  # type: ignore
    conn.close()
    return room_id


def insert_waypoint(id: str, name: str, coords: np.ndarray, room_id: int):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO waypoints (id, name, coords, room_id)
            VALUES (%s, %s, %s, %s)
            """,
            (id, name, coords.tolist(), room_id),
        )
    conn.close()


def get_waypoint_by_name(waypoint_name: str) -> Waypoint | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                name,
                coords
            FROM waypoints
            WHERE name = %s
            """,
            (waypoint_name,),
        )
        row = cursor.fetchone()
    assert row
    conn.close()
    return Waypoint(**row)
