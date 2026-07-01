import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List

from psycopg.rows import dict_row

from humembr.db.repositories.util import get_db_connection
from humembr.util.types import Camera


@dataclass
class ImageQueueItem:
    id: int
    creation_timestamp: datetime
    image_path: str
    caption: str
    waypoint_id: str
    waypoint_name: str
    room_name: str


@dataclass
class CaptionInfo:
    id: int
    caption: str


@dataclass
class CaptionWaypointItem:
    waypoint_name: str
    room_name: str
    captions: List[CaptionInfo]


@dataclass
class SimilarCaption:
    id: int
    caption: str
    creation_timestamp: datetime
    waypoint_id: int
    waypoint_name: str
    camera_id: Camera
    room_name: str
    rotation: float
    distance: float
    final_score: float | None
    time_penalty: float | None


@dataclass
class Observation:
    id: int
    image_path: str
    caption: str | None
    creation_timestamp: datetime
    camera_id: Camera
    waypoint_id: str | None
    rotation: float | None
    waypoint_name: str | None = None


def get_latest_image() -> ImageQueueItem | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT 
                iq.id, 
                iq.creation_timestamp, 
                iq.image_path, 
                iq.caption, 
                iq.waypoint as waypoint_id,
                w.name as waypoint_name,
                r.name as room_name
            FROM image_queue iq
            JOIN waypoints w on iq.waypoint = w.id
            JOIN rooms r on w.room_id = r.id
            WHERE iq.image_path LIKE '%front_fisheye_image.jpg'
            ORDER BY iq.id DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row:
            return ImageQueueItem(**row)
    return None


def get_waypoints_with_captions() -> List[CaptionWaypointItem]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            WITH RankedCaptions AS (
                SELECT
                    iq.id,
                    iq.caption,
                    w.name AS waypoint_name,
                    r.name AS room_name,
                    ROW_NUMBER() OVER(PARTITION BY w.id, iq.caption ORDER BY iq.id DESC) as rn
                FROM image_queue iq
                JOIN waypoints w ON iq.waypoint = w.id
                JOIN rooms r ON w.room_id = r.id
                WHERE iq.caption IS NOT NULL
            )
            SELECT
                waypoint_name,
                room_name,
                json_agg(json_build_object('id', id, 'caption', caption)) as captions
            FROM RankedCaptions
            WHERE rn = 1
            GROUP BY waypoint_name, room_name
            ORDER BY CAST(SPLIT_PART(waypoint_name, '_', 2) AS INTEGER);
            """
        )
        items = cursor.fetchall()
    conn.close()

    waypoint_items = []
    if items:
        for item in items:
            captions = [CaptionInfo(**c) for c in item["captions"]]
            waypoint_items.append(
                CaptionWaypointItem(
                    waypoint_name=item["waypoint_name"],
                    room_name=item["room_name"],
                    captions=captions,
                )
            )
    return waypoint_items


def get_images_by_ids(
    image_ids: list[int], limit: int | None = 20
) -> list[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        query = """
            SELECT
                iq.id,
                iq.creation_timestamp,
                iq.image_path,
                iq.caption,
                iq.rotation,
                iq.waypoint as waypoint_id,
                w.name as waypoint_name,
                iq.camera_id
            FROM image_queue iq
            JOIN waypoints w on iq.waypoint = w.id
            WHERE iq.id = ANY(%s)
            ORDER BY iq.creation_timestamp DESC
            """
        if limit is not None:
            query += " LIMIT %s"
            params = (image_ids, limit)
        else:
            params = (image_ids,)

        cursor.execute(query, params)
        result = cursor.fetchall()
    conn.close()
    return [Observation(**res) for res in result]


def get_observation_by_id(image_id: int) -> Observation | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                creation_timestamp,
                image_path,
                caption,
                rotation,
                waypoint as waypoint_id,
                camera_id
            FROM image_queue
            WHERE id = %s
            """,
            (image_id,),
        )
        row = cursor.fetchone()
        if row:
            return Observation(**row)
    return None


def find_similar_caption(
    embedding,
    person=None,
    until: datetime | None = None,
    limit=20,
    near_max_hours=120,
    time_decay_factor=0.95,
    weight_cosine=1.5,
    weight_time=1.0,
) -> List[SimilarCaption]:
    """
    Method to find similar caption based on sentence embedding and observation timestamp.
    A final score is computed with cosine distance and an exponential time decay penalty.
    A high score is bad, low score is good. Cosine distance and time penalty are in the range of
    Args:
    - embedding: the sentence embedding with 384 dim.
    - limit: number of rows to query
    - near_max_hours: hours passed to archieve a 0.95 time penalty
    """
    conn = get_db_connection()

    decay = -math.log(1 - time_decay_factor) / near_max_hours

    with conn.cursor(row_factory=dict_row) as cursor:
        where_clauses = [
            "iq.caption IS NOT NULL",
        ]
        params = [embedding, decay]

        if person:
            where_clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM person_observations po
                    JOIN person_identity pi ON po.person_identity_id = pi.id
                    WHERE po.image_id = iq.id
                    AND LOWER(pi.name) = LOWER(%s)
                )
                """
            )
            params.append(person)
        if until:
            where_clauses.append("iq.creation_timestamp <= %s")
            params.append(until)

        params.extend([weight_cosine, weight_time, limit])

        sql_query = f"""
            WITH ranked_images AS (
                SELECT
                    iq.id as id,
                    iq.caption as caption,
                    iq.creation_timestamp as creation_timestamp,
                    iq.waypoint AS waypoint_id,
                    iq.camera_id as camera_id,
                    iq.rotation,
                    w.name AS waypoint_name,
                    r.name AS room_name,
                    (iq.caption_vector <=> %s) AS distance,
                    1-EXP(-%s * EXTRACT(EPOCH FROM (NOW() - creation_timestamp))/3600) as time_penalty
                FROM
                    image_queue iq
                JOIN waypoints w ON iq.waypoint = w.id
                JOIN rooms r ON w.room_id = r.id
                WHERE {" AND ".join(where_clauses)}
            )
            SELECT *, (%s*distance + %s*time_penalty) as final_score
            FROM ranked_images
            ORDER BY final_score
            LIMIT %s;
                """
        cursor.execute(sql_query, tuple(params))  # type: ignore
        results = cursor.fetchall()
    conn.close()
    return [SimilarCaption(**res) for res in results]


def add_image(
    creation_time_dt: datetime,
    image_path: str,
    waypoint: str,
    camera_id: Camera,
    rotation: float,
):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        sql_query = """
        INSERT INTO image_queue
            (creation_timestamp, image_path, waypoint, camera_id, rotation)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        cursor.execute(
            sql_query,
            (creation_time_dt, image_path, waypoint, camera_id, rotation),
        )
        row = cursor.fetchone()
        assert row
        id = row["id"]

    conn.close()
    return id


def get_all_images() -> List[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT id, image_path, caption, creation_timestamp, rotation, camera_id, waypoint as waypoint_id FROM image_queue ORDER BY creation_timestamp DESC;"
        )
        items = cursor.fetchall()
    conn.close()
    return [Observation(**item) for item in items]


def get_all_images_chronological() -> List[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                iq.id,
                iq.image_path,
                iq.caption,
                iq.creation_timestamp,
                iq.rotation,
                iq.camera_id,
                iq.waypoint as waypoint_id,
                w.name as waypoint_name
            FROM image_queue iq
            LEFT JOIN waypoints w ON iq.waypoint = w.id
            ORDER BY iq.creation_timestamp DESC;
            """
        )
        items = cursor.fetchall()
    conn.close()
    return [Observation(**item) for item in items]


def search_images_by_text(
    embedding: list[float],
    threshold: float = 0.6,
    filter_image_ids: list[int] | None = None,
) -> List[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        query = """
            SELECT
                iq.id,
                iq.image_path,
                iq.caption,
                iq.creation_timestamp,
                iq.rotation,
                iq.camera_id,
                iq.waypoint as waypoint_id,
                w.name as waypoint_name
            FROM image_queue iq
            LEFT JOIN waypoints w ON iq.waypoint = w.id
            WHERE iq.caption_vector <=> %s < %s
        """
        params = [embedding, threshold]

        if filter_image_ids is not None:
            query += " AND iq.id = ANY(%s)"
            params.append(filter_image_ids)

        query += " ORDER BY iq.creation_timestamp DESC"

        cursor.execute(query, tuple(params))
        items = cursor.fetchall()
    conn.close()
    return [Observation(**item) for item in items]


def get_latest_observation_by_waypoint_and_camera(
    waypoint: str, camera: str
) -> Observation | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                       SELECT id, image_path, caption, creation_timestamp, camera_id, waypoint as waypoint_id, rotation
                       FROM image_queue 
                       WHERE waypoint = %s AND camera_id = %s 
                       ORDER BY creation_timestamp DESC LIMIT 1;
                       """,
            (waypoint, camera),
        )
        row = cursor.fetchone()
        if row:
            return Observation(**row)


def get_uncaptioned_observations(
    n=1, exclude_ids: List[int] | None = None
) -> List[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        query = "SELECT id, image_path, caption, camera_id, rotation, creation_timestamp, waypoint as waypoint_id FROM image_queue WHERE caption IS NULL"
        params = []

        if exclude_ids:
            query += " AND NOT (id = ANY(%s))"
            params.append(exclude_ids)

        query += " ORDER BY id DESC LIMIT %s;"
        params.append(n)

        cursor.execute(query, tuple(params))
        items = cursor.fetchall()
    conn.close()
    return [Observation(**item) for item in items]


def get_uncaptioned_count() -> int:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT COUNT(*) as count FROM image_queue WHERE caption IS NULL;"
        )
        items = cursor.fetchone()
    conn.close()
    assert items
    return items["count"]


def update_image_path(id: int, new_path: str):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "UPDATE image_queue SET image_path = %s WHERE id = %s",
            (new_path, id),
        )
    conn.close()


def update_image_caption(
    id: int, caption: str, caption_emb, caption_model: str, caption_prompt: str
):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "UPDATE image_queue SET caption = %s, caption_vector = %s, caption_model = %s, caption_prompt = %s WHERE id = %s",
            (caption, caption_emb, caption_model, caption_prompt, id),
        )
    conn.close()


def update_camera_id(id: int, camera_id: str):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "UPDATE image_queue SET camera_id = %s WHERE id = %s",
            (camera_id, id),
        )
    conn.close()


def delete_image(id: int, image_path: str):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "DELETE FROM image_queue WHERE id = %s AND image_path = %s",
            (id, image_path),
        )
    conn.close()
    try:
        os.remove(image_path)
    except FileNotFoundError:
        pass


def get_obersvation_since(timestamp: datetime) -> List[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            select
                iq.id,
                iq.image_path,
                iq.caption,
                iq.creation_timestamp,
                iq.camera_id,
                iq.rotation,
                iq.waypoint as waypoint_id,
                w.name as waypoint_name
            from
                image_queue iq
            join
                waypoints w on iq.waypoint = w.id
            where
                iq.creation_timestamp >= %s
            and iq.caption is not null
            order by
                iq.creation_timestamp desc
            limit 50;
            """,
            (timestamp,),
        )
        results = cursor.fetchall()
    conn.close()
    return [Observation(**res) for res in results]


def get_trajectory_by_time(minutes: int) -> List[Observation]:
    conn = get_db_connection()
    time_threshold = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                iq.id,
                iq.image_path,
                iq.caption,
                iq.creation_timestamp,
                iq.camera_id,
                iq.waypoint as waypoint_id,
                w.name as waypoint_name
            FROM
                image_queue iq
            JOIN
                waypoints w ON iq.waypoint = w.id
            WHERE
                iq.creation_timestamp >= %s
            ORDER BY
                iq.creation_timestamp DESC
            LIMIT 50;
            """,
            (time_threshold,),
        )
        results = cursor.fetchall()
    conn.close()
    return [Observation(**res) for res in results]


def get_observation_by_waypoint_name(
    waypoint_name: str,
    start: datetime | None = None,
    until: datetime | None = None,
    n: int | None = 25,
) -> List[Observation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        query = """
            SELECT
                iq.id,
                iq.image_path,
                iq.caption,
                iq.creation_timestamp,
                iq.camera_id,
                iq.rotation,
                iq.waypoint as waypoint_id,
                w.name as waypoint_name
            FROM image_queue iq
            JOIN waypoints w ON iq.waypoint = w.id
            WHERE w.name = %s
        """
        params: list[Any] = [waypoint_name]

        if start:
            query += " AND iq.creation_timestamp >= %s"
            params.append(start)
        if until:
            query += " AND iq.creation_timestamp <= %s"
            params.append(until)

        query += " ORDER BY iq.creation_timestamp DESC"
        if n is not None:
            query += " LIMIT %s"
            params.append(n)

        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
    conn.close()
    return [Observation(**res) for res in results]


def find_by_image_path(filename: str) -> Observation | None:
    like_filename = f"%{filename}"
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                image_path,
                caption,
                creation_timestamp,
                camera_id,
                rotation,
                waypoint as waypoint_id
            FROM image_queue
            WHERE image_path like %s
            """,
            (like_filename,),
        )
        row = cursor.fetchone()
        if row:
            return Observation(**row)
    return None


def get_caption_models_and_prompts() -> tuple[set[str], set[str]]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT DISTINCT caption_model, caption_prompt
            FROM image_queue
            WHERE caption_model IS NOT NULL AND caption_prompt IS NOT NULL;
            """
        )
        items = cursor.fetchall()
    conn.close()
    models = set()
    prompts = set()
    for item in items:
        models.add(item["caption_model"])
        prompts.add(item["caption_prompt"])
    return models, prompts
