from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal

from psycopg.rows import dict_row

from humembr.db.repositories.util import get_db_connection
from humembr.util.types import Direction


@dataclass
class SpotCmd:
    id: int
    status: Literal["PROCESSING", "SUCCESS", "REJECTED"]
    direction: Direction
    rotation: float | None
    updated_timestamp: datetime
    creation_timestamp: datetime
    waypoint_name: str
    prompt: str


def get_cmd_by_id(id: int) -> SpotCmd:
    conn = get_db_connection()

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT waypoint_name, id, status, prompt, direction, rotation, updated_timestamp, creation_timestamp FROM cmd_queue WHERE id = %s",
            (id,),
        )
        row = cursor.fetchone()
        assert row is not None
        return SpotCmd(**row)


def get_next_cmd() -> SpotCmd | None:
    conn = get_db_connection()

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT waypoint_name, id, prompt, status, direction, rotation, updated_timestamp, creation_timestamp FROM cmd_queue WHERE status = 'PENDING' ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return SpotCmd(**row)


def get_next_cmd_for_ws() -> Dict[str, Any] | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                cq.id as id,
                cq.creation_timestamp as creation_timestamp,
                cq.waypoint_name as waypoint_name,
                cq.status as status,
                cq.prompt as prompt,
                w.name as waypoint_name,
                r.name as room_name
            FROM cmd_queue cq
            LEFT JOIN waypoints w ON cq.waypoint_name = w.name
            LEFT JOIN rooms r ON w.room_id = r.id
            WHERE cq.status != 'PENDING'
            ORDER BY cq.id DESC LIMIT 1
            """
        )
        cmd_row = cursor.fetchone()
    conn.close()
    return cmd_row


def get_cmd_status(id: int) -> Literal["PROCESSING", "SUCCESS", "REJECTED"]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT status FROM cmd_queue WHERE id = %s", (id,))
        row = cursor.fetchone()
        assert row
    conn.close()
    return row["status"]


def add_cmd(waypoint_name: str, prompt: str, rotation: float):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        sql_query = "INSERT INTO cmd_queue (waypoint_name, prompt, rotation) VALUES (%s, %s, %s) RETURNING id;"
        cursor.execute(sql_query, (waypoint_name, prompt, rotation))
        row = cursor.fetchone()
        assert row
        id = row["id"]
    conn.close()
    return id


def process_cmd(id: int):
    update_cmd_status(id, "PROCESSING")


def reject_cmd(id: int):
    update_cmd_status(id, "REJECTED")


def success_cmd(id: int):
    update_cmd_status(id, "SUCCESS")


def update_cmd_status(id: int, status: Literal["PROCESSING", "SUCCESS", "REJECTED"]):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "UPDATE cmd_queue SET status = %s, updated_timestamp = NOW() WHERE id = %s",
            (status, id),
        )


def clear_cmds():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("TRUNCATE TABLE cmd_queue")
