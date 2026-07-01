from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from humembr.db.repositories.util import get_db_connection


@dataclass
class PersonObservation:
    id: int
    image_id: int
    reid_vector: np.ndarray
    face_vector: np.ndarray | None
    distance: float
    crop_path: str
    keypoints: list[tuple[float, float, float]]
    neg_keypoints: list[list[tuple[float, float, float]]]
    person_identity_id: int | None = None
    waypoint_name: str | None = None
    identity_name: str | None = None
    face_det_score: float | None = None


@dataclass
class PersonIdentity:
    id: int
    name: str
    person_obs: list[PersonObservation] = field(default_factory=list)


def get_observation_ids_by_name(name: str, start: datetime, end: datetime) -> list[int]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                po.id
            FROM person_observations po
            JOIN person_identity pi ON po.person_identity_id = pi.id
            JOIN image_queue iq on po.image_id = iq.id
            WHERE LOWER(pi.name) = LOWER(%s)
            AND date_trunc('second', iq.creation_timestamp) BETWEEN %s AND %s
            ORDER BY po.id DESC;
        """,
            (name, start, end),
        )

        results = cursor.fetchall()
    conn.close()
    return [res["id"] for res in results]


def add_person_identity(name: str, person_ids: list[int]):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT id FROM person_identity WHERE name = %s", (name,))
        result = cursor.fetchone()

        if result:
            identity_id = result["id"]
            cursor.execute(
                """
                UPDATE person_identity
                SET updated_timestamp = NOW()
                WHERE id = %s
                """,
                (identity_id,),
            )
            # Unlink all current observations for this identity
            cursor.execute(
                "UPDATE person_observations SET person_identity_id = NULL WHERE person_identity_id = %s",
                (identity_id,),
            )
        else:
            cursor.execute(
                """
                INSERT INTO person_identity (name)
                VALUES (%s)
                RETURNING id
                """,
                (name,),
            )
            row = cursor.fetchone()
            assert row
            identity_id = row["id"]

        # Link new observations
        if person_ids:
            cursor.execute(
                """
                UPDATE person_observations
                SET person_identity_id = %s
                WHERE id = ANY(%s)
                """,
                (identity_id, person_ids),
            )

    conn.close()


def add_observation_to_identity(identity_name: str, person_obs_id: int):
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT id FROM person_identity WHERE name = %s", (identity_name,)
        )
        result = cursor.fetchone()

        if result:
            identity_id = result["id"]
            cursor.execute(
                """
                UPDATE person_identity
                SET updated_timestamp = NOW()
                WHERE id = %s
                """,
                (identity_id,),
            )
        else:
            cursor.execute(
                """
                INSERT INTO person_identity (name)
                VALUES (%s)
                RETURNING id
                """,
                (identity_name,),
            )
            row = cursor.fetchone()
            assert row
            identity_id = row["id"]

        cursor.execute(
            "UPDATE person_observations SET person_identity_id = %s WHERE id = %s",
            (identity_id, person_obs_id),
        )
    conn.close()


def get_all_person_observations(
    unmapped_only: bool = False,
    identity_id: int | None = None,
) -> list[PersonObservation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        query = """
            SELECT
                p.id,
                p.image_id,
                p.reid_vector,
                p.face_vector,
                p.face_det_score,
                p.crop_path,
                p.keypoints,
                p.neg_keypoints,
                p.person_identity_id,
                pi.name as identity_name
            FROM person_observations p
            LEFT JOIN person_identity pi ON p.person_identity_id = pi.id
        """
        params: tuple[Any, ...] = ()
        if identity_id is not None:
            query += " WHERE p.person_identity_id = %s"
            params = (identity_id,)
        elif unmapped_only:
            query += " WHERE p.person_identity_id IS NULL"
        query += " ORDER BY p.id DESC;"
        cursor.execute(query, params)
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(distance=0, **res) for res in results]


def get_unique_image_ids_with_persons() -> list[int]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT DISTINCT image_id FROM person_observations")
        results = cursor.fetchall()
    conn.close()
    return [res["image_id"] for res in results]


def get_image_ids_with_unmapped_persons() -> list[int]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT DISTINCT image_id FROM person_observations WHERE person_identity_id IS NULL"
        )
        results = cursor.fetchall()
    conn.close()
    return [res["image_id"] for res in results]


def get_unmapped_face_observations() -> list[PersonObservation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                image_id,
                reid_vector,
                face_vector,
                crop_path,
                keypoints,
                neg_keypoints,
                person_identity_id
            FROM person_observations p
            WHERE p.face_vector IS NOT NULL
            AND p.person_identity_id IS NULL
            """,
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(distance=0, **res) for res in results]


def get_all_identity_names() -> list[str]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT name FROM person_identity")
        results = cursor.fetchall()
    conn.close()
    return [res["name"].lower() for res in results]


def get_identity_options() -> list[dict[str, Any]]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT id, name
            FROM person_identity
            ORDER BY lower(name) ASC;
            """
        )
        results = cursor.fetchall()
    conn.close()
    return [dict(res) for res in results]


def update_person_observation_identity(person_id: int, identity_id: int | None) -> None:
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE person_observations SET person_identity_id = %s WHERE id = %s",
            (identity_id, person_id),
        )
    conn.close()


def get_person_observation_by_id(person_id: int) -> PersonObservation | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                SELECT
                    id,
                    image_id,
                    reid_vector,
                    face_vector,
                    crop_path,
                    keypoints,
                    neg_keypoints,
                    person_identity_id
                FROM person_observations
                WHERE id = %s;
            """,
            (person_id,),
        )
        result = cursor.fetchone()
    conn.close()
    if result:
        return PersonObservation(distance=0, **result)
    return None


def get_image_ids_by_observation_ids(person_ids: list[int]) -> list[int]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                image_id
            FROM person_observations
            WHERE id = ANY(%s)
            """,
            (person_ids,),
        )
        results = cursor.fetchall()
    conn.close()
    return [res["image_id"] for res in results]


def get_similar_observation(embedding: np.ndarray, top_k=20):
    conn = get_db_connection()

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                SELECT
                    id,
                    image_id,
                    reid_vector,
                    face_vector,
                    keypoints,
                    neg_keypoints,
                    crop_path,
                    person_identity_id,
                    (reid_vector <=> %s) AS distance
                FROM person_observations
                ORDER BY distance
                LIMIT %s;
            """,
            (embedding, top_k),
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(**res) for res in results]


def add_person_observation(
    image_id: int,
    embedding: np.ndarray,
    face_embedding: np.ndarray | None,
    face_det_score: float | None,
    crop_path: str,
    keypoints: list[Any],
    neg_keypoints: list[Any],
) -> int:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "INSERT INTO person_observations (image_id, reid_vector, face_vector, face_det_score, crop_path, keypoints, neg_keypoints) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                image_id,
                embedding,
                face_embedding,
                face_det_score,
                crop_path,
                Jsonb(keypoints),
                Jsonb(neg_keypoints),
            ),
        )
        row = cursor.fetchone()
        assert row is not None
        new_id = row["id"]
    conn.close()
    return new_id


def get_identities_containing_observations(
    person_obs_ids: list[int],
) -> list[PersonIdentity]:
    if not person_obs_ids:
        return []
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT DISTINCT
                pi.id,
                pi.name
            FROM person_identity pi
            JOIN person_observations po ON pi.id = po.person_identity_id
            WHERE po.id = ANY(%s);
            """,
            (person_obs_ids,),
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonIdentity(**res) for res in results]


def get_person_identity(person_name: str) -> PersonIdentity | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                name
            FROM person_identity
            WHERE lower(name) = lower(%s);
        """,
            (person_name,),
        )
        result = cursor.fetchone()
    conn.close()
    if not result:
        return None
    pi = PersonIdentity(**result)
    pi.person_obs = get_person_observations_by_identity_id(pi.id)
    return pi


def get_person_identity_by_id(identity_id: int) -> PersonIdentity | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                name
            FROM person_identity
            WHERE id = %s;
        """,
            (identity_id,),
        )
        result = cursor.fetchone()
    conn.close()

    if not result:
        return None

    pi = PersonIdentity(**result)
    pi.person_obs = get_person_observations_by_identity_id(pi.id)
    return pi


def get_all_person_identities() -> list[PersonIdentity]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                name
            FROM person_identity;
        """
        )
        results = cursor.fetchall()
    conn.close()
    if not results:
        return []

    mappings = [PersonIdentity(**res) for res in results]
    for mapping in mappings:
        mapping.person_obs = get_person_observations_by_identity_id(mapping.id)

    return mappings


def delete_person_identity(mapping_id: int):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM person_identity WHERE id = %s", (mapping_id,))
    conn.close()


def update_person_identity_name(mapping_id: int, new_name: str):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "UPDATE person_identity SET name = %s, updated_timestamp = NOW() WHERE id = %s",
            (new_name, mapping_id),
        )
    conn.close()


def get_person_observations_by_ids(person_ids: list[int]) -> list[PersonObservation]:
    if not person_ids:
        return []
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                SELECT
                    id,
                    image_id,
                    reid_vector,
                    face_vector,
                    crop_path,
                    keypoints,
                    neg_keypoints,
                    person_identity_id
                FROM person_observations
                WHERE id = ANY(%s);
            """,
            (person_ids,),
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(distance=0, **res) for res in results]


def get_person_observations_by_identity_id(identity_id: int) -> list[PersonObservation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                SELECT
                    id,
                    image_id,
                    reid_vector,
                    face_vector,
                    crop_path,
                    keypoints,
                    neg_keypoints,
                    person_identity_id
                FROM person_observations
                WHERE person_identity_id = %s
                ORDER BY face_det_score;
            """,
            (identity_id,),
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(distance=0, **res) for res in results]


def get_person_observations_by_image_ids(
    image_ids: list[int],
) -> list[PersonObservation]:
    if not image_ids:
        return []
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                SELECT
                    p.id,
                    p.image_id,
                    p.reid_vector,
                    p.face_vector,
                    p.crop_path,
                    p.keypoints,
                    p.neg_keypoints,
                    p.person_identity_id,
                    w.name as waypoint_name,
                    pi.name as identity_name
                FROM person_observations p
                JOIN image_queue iq on p.image_id = iq.id
                JOIN waypoints w on iq.waypoint = w.id
                LEFT JOIN person_identity pi on p.person_identity_id = pi.id
                WHERE p.image_id = ANY(%s);
            """,
            (image_ids,),
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(distance=0, **res) for res in results]


def get_face_observation_neighbors(
    embedding: np.ndarray, threshold: float = 0.62
) -> list[PersonObservation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
                SELECT
                    id,
                    image_id,
                    reid_vector,
                    face_vector,
                    crop_path,
                    keypoints,
                    neg_keypoints,
                    person_identity_id,
                    (face_vector <=> %s) AS distance
                FROM person_observations
                WHERE face_vector IS NOT NULL
                AND (face_vector <=> %s) < %s
                ORDER BY distance;
            """,
            (embedding, embedding, threshold),
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(**res) for res in results]


def get_observations_without_face_and_identity() -> list[PersonObservation]:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                image_id,
                reid_vector,
                face_vector,
                crop_path,
                keypoints,
                neg_keypoints,
                person_identity_id,
                face_det_score
            FROM person_observations
            WHERE face_vector IS NULL
            AND person_identity_id IS NULL;
            """
        )
        results = cursor.fetchall()
    conn.close()
    return [PersonObservation(distance=0, **res) for res in results]


def get_nearest_observation_with_face(
    reid_vector: np.ndarray,
) -> PersonObservation | None:
    conn = get_db_connection()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                id,
                image_id,
                reid_vector,
                face_vector,
                crop_path,
                keypoints,
                neg_keypoints,
                person_identity_id,
                face_det_score,
                (reid_vector <=> %s) as distance
            FROM person_observations
            WHERE face_vector IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
            """,
            (reid_vector,),
        )
        result = cursor.fetchone()
    conn.close()
    if result:
        return PersonObservation(**result)
    return None


def find_by_date(date: datetime) -> list[str]:
    conn = get_db_connection()
    # query all person names
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT DISTINCT
                pi.name
            FROM person_identity pi
            JOIN person_observations po ON pi.id = po.person_identity_id
            JOIN image_queue iq on po.image_id = iq.id
            WHERE date_trunc('day', iq.creation_timestamp) = date_trunc('day', %s)
            """,
            (date,),
        )
        results = cursor.fetchall()
    conn.close()
    return [res["name"] for res in results]


def has_reid_identity_matching() -> bool:
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM person_observations
                WHERE person_identity_id IS NOT NULL and face_vector IS NULL
            );
            """
        )
        result = cursor.fetchone()
    conn.close()
    return result is not None and result[0]
