import glob
import os

from humembr.db.repositories.util import get_db_connection
from humembr.util.config import load_config


def cleanup_orphan_files(conn):
    """
    Finds and deletes image and json files that do not have a corresponding entry in the database.
    """
    print("\n--- Starting orphan file cleanup ---")
    config = load_config()
    img_dir = config.img_dir

    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM image_queue;")
    db_image_paths = {row[0] for row in cursor.fetchall()}
    cursor.close()

    db_json_paths = {p.replace(".jpg", ".json") for p in db_image_paths}
    expected_files_in_db = db_image_paths.union(db_json_paths)

    all_files_on_disk = set(glob.glob(f"{img_dir}/**/*.jpg", recursive=True))
    all_files_on_disk.update(set(glob.glob(f"{img_dir}/**/*.json", recursive=True)))

    files_to_delete = all_files_on_disk - expected_files_in_db

    if not files_to_delete:
        print("No orphan files found.")
        return

    print(f"Found {len(files_to_delete)} orphan files to delete.")
    deleted_count = 0
    for file_path in files_to_delete:
        if "crop_persons" in file_path or "faces" in file_path:
            continue
        try:
            os.remove(file_path)
            print(f"[SUCCESS] Deleted orphan file: {file_path}")
            deleted_count += 1
        except FileNotFoundError:
            print(
                f"[WARNING] Orphan file not found, could have been deleted by another process: {file_path}"
            )
        except PermissionError:
            print(f"[ERROR] Permission denied for file. Skipping: {file_path}")
        except Exception as e:
            print(f"[ERROR] Could not delete file. Skipping: {e}")

    print("\n--- Orphan file cleanup summary ---")
    print(f"Successfully deleted {deleted_count} orphan files.")


def cleanup_duplicate_db_entries(conn):
    """
    Finds and deletes duplicate rows from the image_queue table and their associated files.
    Duplicates are identified based on identical 'caption' and 'waypoint'.
    """
    print("\n--- Starting duplicate DB entry cleanup ---")
    find_duplicates_sql = """
        WITH NumberedRows AS (
            SELECT
                id,
                image_path,
                ROW_NUMBER() OVER(
                    PARTITION BY caption, waypoint
                    ORDER BY id ASC
                ) as rn
            FROM
                image_queue
        )
        SELECT
            id,
            image_path
        FROM
            NumberedRows
        WHERE
            rn > 1;
        """
    delete_sql = "DELETE FROM image_queue WHERE id = ANY(%s);"

    duplicates_found = []
    ids_to_delete_from_db = []

    cursor = conn.cursor()
    cursor.execute(find_duplicates_sql)
    duplicates_found = cursor.fetchall()

    if not duplicates_found:
        print("No duplicates found based on caption and waypoint.")
        cursor.close()
        return

    print(f"Found {len(duplicates_found)} duplicate rows to process.")
    for row_id, image_path in duplicates_found:
        try:
            json_path = image_path.replace(".jpg", ".json")
            os.remove(image_path)
            print(f"[SUCCESS] Deleted file: {image_path}")
            ids_to_delete_from_db.append(row_id)
            os.remove(json_path)
            print(f"[SUCCESS] Deleted file: {json_path}")

        except FileNotFoundError:
            print(
                f"[WARNING] File not found, but will delete DB row {row_id}: {image_path}"
            )
            ids_to_delete_from_db.append(row_id)

        except PermissionError:
            print(
                f"[ERROR] Permission denied for file. Skipping row {row_id}: {image_path}"
            )

        except Exception as e:
            print(f"[ERROR] Could not delete file for row {row_id}. Skipping: {e}")

    if not ids_to_delete_from_db:
        print("No database rows could be safely deleted.")
        cursor.close()
        return

    print(f"Attempting to delete {len(ids_to_delete_from_db)} rows from database...")
    cursor.execute(delete_sql, (ids_to_delete_from_db,))
    deleted_row_count = cursor.rowcount

    print("\n--- Duplicate DB entry cleanup summary ---")
    print(f"Successfully deleted {deleted_row_count} database rows.")
    # No commit needed if autocommit is on by default.

    if cursor:
        cursor.close()


def cleanup_missing_image_files(conn):
    """
    Deletes image rows from the database if the corresponding image file does not exist.
    """
    print("\n--- Starting cleanup of rows with missing image files ---")
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path FROM image_queue;")
    image_rows = cursor.fetchall()

    ids_to_delete = []
    for row_id, image_path in image_rows:
        if not os.path.exists(image_path):
            print(
                f"[INFO] Image file not found, scheduling DB row for deletion: {image_path}"
            )
            ids_to_delete.append(row_id)

    if not ids_to_delete:
        print("No DB rows with missing image files found.")
        cursor.close()
        return

    print(f"Found {len(ids_to_delete)} DB rows with missing files to delete.")
    delete_sql = "DELETE FROM image_queue WHERE id = ANY(%s);"
    cursor.execute(delete_sql, (ids_to_delete,))
    deleted_row_count = cursor.rowcount

    print("\n--- Rows with missing files cleanup summary ---")
    print(f"Successfully deleted {deleted_row_count} database rows.")

    if cursor:
        cursor.close()


def main():
    """
    Main function to run the database cleanup tasks.
    """
    conn = None
    try:
        # autocommit=False to handle transaction manually if needed, but default is True
        conn = get_db_connection()

        cleanup_duplicate_db_entries(conn)
        cleanup_orphan_files(conn)
        cleanup_missing_image_files(conn)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
